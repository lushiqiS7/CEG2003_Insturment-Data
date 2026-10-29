"""Build one pre-cutoff feature row per observed, identified captain."""

import numpy as np
import pandas as pd

ALARM_NAMES = {
    "Harsh Braking": "braking",
    "Harsh Acceleration": "acceleration",
    "Forward Collision Warning": "forward_collision",
    "Headway Monitoring Alert": "headway",
    "Pedestrian Collision Alert": "pedestrian",
    "Fatigue Alert": "fatigue",
    "Yawn": "yawn",
    "Sharp Cornering": "cornering",
    "Speeding": "speeding",
    "Swaying": "swaying",
}
FEATURE_COLUMNS = [f"log_{name}" for name in ALARM_NAMES.values()] + [
    "log_alarm_days",
    "night_share",
    "mean_alarm_speed",
    "log_mean_duration",
]

FATIGUE_TYPES = ["Fatigue Alert", "Yawn"]
COLLISION_TYPES = [
    "Forward Collision Warning",
    "Headway Monitoring Alert",
    "Pedestrian Collision Alert",
]
RISK_GROUPS = {
    "Fatigue alerts": [
        "log_fatigue",
        "log_yawn",
        "fatigue_share",
        "log_fatigue_days",
        "fatigue_recency_days",
    ],
    "Collision warnings": [
        "log_forward_collision",
        "log_headway",
        "log_pedestrian",
        "collision_share",
        "log_collision_days",
        "collision_recency_days",
    ],
    "Harsh driving": [
        "log_braking",
        "log_acceleration",
        "log_cornering",
        "log_speeding",
        "log_swaying",
        "braking_share",
    ],
    "Recent changes": ["fatigue_change_7d", "collision_change_7d", "braking_change_7d"],
    "Compounding alerts": [
        "fatigue_braking_interaction",
        "fatigue_collision_interaction",
        "fatigue_before_collision_share",
    ],
    "Observation context": [
        "log_alarm_days",
        "night_share",
        "mean_alarm_speed",
        "log_mean_duration",
        "night_fatigue_share",
        "missing_route_share",
    ],
}
RISK_FEATURE_COLUMNS = [column for group in RISK_GROUPS.values() for column in group]


def build_risk_features(events, settings):
    """Observed alert patterns only; no inferred hours, trips, severity or shifts."""
    lower = pd.Timestamp(settings["feature_start"])
    upper = pd.Timestamp(settings["feature_end"]) + pd.Timedelta(days=1)
    if (upper - lower).days < 14:
        raise ValueError("Two seven-day comparison windows require at least 14 observed days")
    selected = events.loc[
        events["captain_id"].notna() & events["start"].ge(lower) & events["start"].lt(upper)
    ].copy()
    selected["day"] = selected["start"].dt.normalize()
    selected["night"] = selected["start"].dt.hour.ge(settings["night_start_hour"]) | selected[
        "start"
    ].dt.hour.lt(settings["night_end_hour"])
    features = build_features(selected, settings).set_index("captain_id")
    for name, types in (
        ("fatigue", FATIGUE_TYPES),
        ("collision", COLLISION_TYPES),
        ("braking", ["Harsh Braking"]),
    ):
        category = selected.loc[selected["alarm_type"].isin(types)]
        grouped = category.groupby("captain_id", observed=True)
        count = grouped.size().reindex(features.index, fill_value=0)
        features[f"{name}_events"] = count
        features[f"{name}_share"] = count / features["total_alarms"]
        if name != "braking":
            features[f"log_{name}_days"] = np.log1p(
                grouped["day"].nunique().reindex(features.index, fill_value=0)
            )
            features[f"{name}_recency_days"] = (
                upper - grouped["start"].max().reindex(features.index)
            ).dt.total_seconds() / 86400
        recent = (
            category.loc[category["start"].ge(upper - pd.Timedelta(days=7))]
            .groupby("captain_id", observed=True)
            .size()
            .reindex(features.index, fill_value=0)
        )
        previous = (
            category.loc[
                category["start"].ge(upper - pd.Timedelta(days=14))
                & category["start"].lt(upper - pd.Timedelta(days=7))
            ]
            .groupby("captain_id", observed=True)
            .size()
            .reindex(features.index, fill_value=0)
        )
        features[f"{name}_recent_7d"] = recent
        features[f"{name}_previous_7d"] = previous
        features[f"{name}_change_7d"] = np.log1p(recent) - np.log1p(previous)
    fatigue = selected.loc[selected["alarm_type"].isin(FATIGUE_TYPES)]
    collision = selected.loc[selected["alarm_type"].isin(COLLISION_TYPES)]
    night_fatigue = (
        fatigue.loc[fatigue["night"]]
        .groupby("captain_id", observed=True)
        .size()
        .reindex(features.index, fill_value=0)
    )
    features["night_fatigue_share"] = night_fatigue / features["fatigue_events"].replace(0, np.nan)
    features["missing_route_share"] = (
        selected["route"]
        .isna()
        .groupby(selected["captain_id"], observed=True)
        .mean()
        .reindex(features.index)
    )
    features["fatigue_braking_interaction"] = features["fatigue_share"] * features["braking_share"]
    features["fatigue_collision_interaction"] = (
        features["fatigue_share"] * features["collision_share"]
    )
    features["fatigue_before_collision_share"] = 0.0 / features["collision_events"].replace(
        0, np.nan
    )
    if not fatigue.empty and not collision.empty:
        preceding = pd.merge_asof(
            collision[["captain_id", "start"]].sort_values("start"),
            fatigue[["captain_id", "start"]]
            .rename(columns={"start": "fatigue_time"})
            .sort_values("fatigue_time"),
            left_on="start",
            right_on="fatigue_time",
            by="captain_id",
            direction="backward",
            tolerance=pd.Timedelta(minutes=15),
        )
        matched = (
            preceding["fatigue_time"]
            .notna()
            .groupby(preceding["captain_id"], observed=True)
            .sum()
            .reindex(features.index, fill_value=0)
        )
        features["fatigue_before_collision_share"] = matched / features["collision_events"].replace(
            0, np.nan
        )
    recent = features["fatigue_recent_7d"] + features["collision_recent_7d"]
    previous = features["fatigue_previous_7d"] + features["collision_previous_7d"]
    features["trend"] = np.select(
        [recent > previous, recent < previous], ["Rising", "Falling"], default="Flat"
    )
    assert len(RISK_FEATURE_COLUMNS) == len(set(RISK_FEATURE_COLUMNS))
    assert features["last_alarm"].lt(upper).all()
    return features.reset_index()


def build_features(events, settings):
    start = pd.Timestamp(settings["feature_start"])
    end = pd.Timestamp(settings["feature_end"]) + pd.Timedelta(days=1)
    subset = events.loc[
        events["captain_id"].notna() & events["start"].ge(start) & events["start"].lt(end)
    ].copy()
    if subset.empty:
        raise ValueError("No identified alarm events in the feature window")
    counts = pd.crosstab(subset["captain_id"], subset["alarm_type"])
    counts = counts.reindex(columns=list(ALARM_NAMES), fill_value=0)
    counts.columns = list(ALARM_NAMES.values())
    grouped = subset.groupby("captain_id", observed=True)
    summary = grouped.agg(
        total_alarms=("alarm_type", "size"),
        alarm_days=("day", "nunique"),
        night_share=("night", "mean"),
        mean_alarm_speed=("speed", "mean"),
        mean_duration=("duration", "mean"),
        first_alarm=("start", "min"),
        last_alarm=("start", "max"),
    )
    features = summary.join(counts)
    for name in ALARM_NAMES.values():
        features[f"log_{name}"] = np.log1p(features[name])
    features["log_alarm_days"] = np.log1p(features["alarm_days"])
    features["log_mean_duration"] = np.log1p(features["mean_duration"])
    assert features.index.is_unique
    assert features["last_alarm"].lt(end).all()
    assert features["total_alarms"].sum() == len(subset)
    return features.reset_index()


def attach_target(features, incidents, settings):
    lower, upper = pd.Timestamp(settings["outcome_start"]), pd.Timestamp(settings["outcome_end"])
    if lower <= pd.Timestamp(settings["feature_end"]):
        raise ValueError("Outcome window must follow the feature window")
    selected = incidents.loc[incidents["day"].between(lower, upper)]
    positive = set(selected.loc[selected["included"], "captain_id"].dropna())
    excluded = set(selected.loc[~selected["included"], "captain_id"].dropna())
    result = features.copy()
    # An excluded-only report is ambiguous, rather than evidence of a negative label.
    result["label_known"] = ~result["captain_id"].isin(excluded - positive)
    result["target"] = result["captain_id"].isin(positive).astype(int)
    return result
