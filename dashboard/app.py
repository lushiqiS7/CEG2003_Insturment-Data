"""A focused, local captain review queue with calibrated-model explanations."""

import base64
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.risk_training import predict_captains  # noqa: E402 - Streamlit script path bootstrap

st.set_page_config(page_title="Captain Safety Lab", layout="wide")
assets = Path(__file__).parent / "assets"
font = base64.b64encode((assets / "Geist.ttf").read_bytes()).decode()
css = (Path(__file__).parent / "style.css").read_text(encoding="utf-8").replace("FONT_DATA", font)
st.html(f"<style>{css}</style>")
st.html(
    '<div class="brand"><span>Captain Safety Lab</span><span class="muted">CEG2003 · Local workspace</span></div>'
)


@st.cache_data(show_spinner=False)
def load_data(version):
    preparation = (ROOT / "outputs/preparation.json").read_bytes()
    result = json.loads((ROOT / "outputs/metrics.json").read_text(encoding="utf-8"))
    if result.get("schema_version") != 2:
        raise ValueError(
            "The revised risk model is still being prepared. Run ML: train and evaluate."
        )
    if result["preparation_sha256"] != hashlib.sha256(preparation).hexdigest():
        raise ValueError("Data preparation changed after training. Run ML: train and evaluate.")
    if json.loads((ROOT / "config.json").read_text(encoding="utf-8")) != result["settings"]:
        raise ValueError("Configuration changed after training. Run ML: full pipeline to apply it.")
    for name, key in [
        ("models/incident_model.joblib", "model_hash"),
        ("outputs/driver_scores.parquet", "scores_hash"),
        ("data/processed/driver_daily.parquet", "daily_hash"),
    ]:
        with (ROOT / name).open("rb") as stream:
            actual_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        if result.get(key) != actual_hash:
            raise ValueError("Model outputs do not match their evaluation. Run ML: full pipeline.")
    return (
        result,
        pd.read_parquet(ROOT / "outputs/driver_scores.parquet"),
        pd.read_parquet(ROOT / "data/processed/driver_daily.parquet"),
    )


@st.cache_resource(show_spinner=False)
def load_model(version):
    return joblib.load(ROOT / "models/incident_model.joblib")


def trend_chart(activity, names, color):
    selected = activity.loc[activity["alarm_type"].isin(names)]
    values = (
        selected.groupby("day", observed=True)["events"]
        .sum()
        .reindex(pd.date_range(settings["feature_start"], settings["feature_end"]), fill_value=0)
    )
    figure = go.Figure(
        go.Scatter(x=values.index, y=values.values, mode="lines", line=dict(color=color, width=2))
    )
    figure.update_layout(
        height=195,
        margin=dict(l=8, r=8, t=8, b=8),
        font=dict(family="Geist, Arial", size=12, color="#62666b"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    figure.update_xaxes(showgrid=False, tickformat="%d %b", nticks=5)
    figure.update_yaxes(
        gridcolor="#eef0f2",
        zeroline=False,
        rangemode="tozero",
        nticks=4,
        title_text="Recorded alerts",
    )
    st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})


required = [
    ROOT / "outputs/metrics.json",
    ROOT / "outputs/preparation.json",
    ROOT / "models/incident_model.joblib",
    ROOT / "outputs/driver_scores.parquet",
    ROOT / "data/processed/driver_daily.parquet",
    ROOT / "config.json",
]
st.title("Captain risk review")
if not all(path.exists() for path in required):
    st.info("Risk outputs are not available. Run ML: full pipeline in VS Code.")
    st.stop()
version = tuple(path.stat().st_mtime_ns for path in required)
try:
    result, scores, daily = load_data(version)
except (OSError, ValueError, KeyError) as error:
    st.info(str(error))
    st.stop()
settings = result["settings"]
st.caption(
    f"Alarms: {settings['feature_start']} to {settings['feature_end']}. Later recorded incidents: {settings['outcome_start']} to {settings['outcome_end']}."
)
summary = st.columns(2)
summary[0].metric("Captains scored", f"{len(scores):,}")
summary[1].metric("Provisional review flags", f"{int(scores['provisional_at_risk'].sum()):,}")
st.caption("Alert severity is not recorded in the supplied data.")
validation = result["validation"]
gate = result["evidence_gate"]["passed"]
readiness = (
    "Development evidence checks passed; independent validation is still required."
    if gate
    else "Predictive evidence checks have not passed. These are provisional review suggestions."
)
st.html(
    f'<div class="notice">{readiness} Nested development CV: ROC-AUC {validation["roc_auc"]:.3f}; precision among the top {settings["review_fraction"]:.0%}: {validation["precision_at_k"]:.1%}. No new independent test has been evaluated.</div>'
)

controls = st.columns([1, 1])
with controls[0]:
    scope = st.selectbox("Queue", ["Flagged captains", "All captains"])
with controls[1]:
    search = st.text_input("Find captain", placeholder="Pseudonymous captain ID")
ranked = scores.sort_values(["risk_score", "captain_id"], ascending=[False, True])
if scope == "Flagged captains":
    ranked = ranked.loc[ranked["provisional_at_risk"]]
if search.strip():
    ranked = ranked.loc[ranked["captain_id"].str.contains(search.strip(), case=False, regex=False)]
st.subheader("Prioritized captain queue")
st.caption(
    "Priority 0–100 is relative rank against development cross-fitted scores, not accident probability. Factors are signed grouped SHAP effects; pp means percentage points."
)
if ranked.empty:
    st.info("No captains match this queue selection.")
else:
    display = ranked[["captain_id", "risk_score", "flag", "top_factors", "trend"]].reset_index(
        drop=True
    )
    table = st.dataframe(
        display,
        hide_index=True,
        width="stretch",
        height=300,
        on_select="rerun",
        selection_mode="single-row",
        key="risk_queue",
        column_config={
            "captain_id": st.column_config.TextColumn("Captain", width=140, pinned=True),
            "risk_score": st.column_config.ProgressColumn(
                "Priority /100", min_value=0, max_value=100, format="%.1f", width=120
            ),
            "flag": st.column_config.TextColumn("Review flag", width=140),
            "top_factors": st.column_config.TextColumn("Three contributing factors", width=650),
            "trend": st.column_config.TextColumn("Recent alerts", width=110),
        },
    )
    st.caption(
        f"{len(ranked):,} captains in this queue. Rising/falling compares recorded fatigue and collision counts in the latest seven days with the preceding seven days. Counts are not driving-exposure rates."
    )
    selected_rows = table.selection.rows
    options = ranked["captain_id"].tolist()
    row_captain = (
        display.iloc[selected_rows[0]]["captain_id"]
        if selected_rows and selected_rows[0] < len(display)
        else None
    )
    if row_captain != st.session_state.get("last_row_captain"):
        st.session_state["last_row_captain"] = row_captain
        if row_captain is not None:
            st.session_state["detail_captain"] = row_captain
    if st.session_state.get("detail_captain") not in options:
        st.session_state["detail_captain"] = options[0]
    with st.expander("Selected captain · why review?", expanded=bool(selected_rows)):
        captain = st.selectbox("Captain detail", options, key="detail_captain")
        row = ranked.loc[ranked["captain_id"].eq(captain)].iloc[0]
        st.write(
            f"**{captain}** · priority **{row['risk_score']:.1f}/100** · {row['score_source']}"
        )
        st.caption(
            f"Sigmoid-calibrated incident estimate: {row['probability']:.2%}; model threshold: {row['probability_threshold']:.2%}. Calibration is estimated from sparse development data and is not independently established."
        )
        values = st.columns(3)
        values[0].metric("Fatigue + yawn alerts", f"{int(row['fatigue_events']):,}")
        values[1].metric("Collision warnings", f"{int(row['collision_events']):,}")
        values[2].metric("Days with alarms", str(int(row["alarm_days"])))
        contributions = pd.Series(
            {group: row[f"shap_{group}"] for group in result["groups"]}
        ).sort_values(key=abs, ascending=False)
        for group, contribution in contributions.head(3).items():
            direction = (
                "raises"
                if contribution > 0
                else "lowers"
                if contribution < 0
                else "does not change"
            )
            st.write(
                f"**{group}** {direction} the model estimate by **{abs(contribution) * 100:.2f} pp** relative to the training reference."
            )
        st.caption(
            f"All six grouped contributions plus the reference estimate ({row['shap_base']:.2%}) reconstruct this captain’s estimate. These are model associations, not causes or a finding of fault."
        )
        if st.button("Verify selected score"):
            probability, _ = predict_captains(load_model(version), row.to_frame().T)
            if np.isclose(probability[0], row["probability"]):
                st.success("Current model inference matches the displayed score.")
            else:
                st.error("Saved and current scores disagree. Refresh training outputs.")
        activity = daily.loc[daily["captain_id"].eq(captain)].copy()
        activity["day"] = pd.to_datetime(activity["day"])
        left, right = st.columns(2)
        with left:
            st.subheader("Fatigue and yawn alerts")
            trend_chart(activity, ["Fatigue Alert", "Yawn"], "#1d5fbf")
        with right:
            st.subheader("Collision warnings")
            trend_chart(
                activity,
                [
                    "Forward Collision Warning",
                    "Headway Monitoring Alert",
                    "Pedestrian Collision Alert",
                ],
                "#62666b",
            )

with st.expander("Validation and score definitions"):
    st.write(
        f"{result['development_drivers']:,} development captains, including {result['development_positives']} provisional positive labels. {result['legacy_test_drivers_excluded']} previously inspected test captains remain excluded from model selection and were not reevaluated."
    )
    st.write(
        f"Final candidate: **{result['selected_model']}**. Three outer folds assess the selection process; model choice, preprocessing and sigmoid calibration occur within the corresponding training data. Development captains are shown with their outer-fold model, which never fitted them."
    )
    st.dataframe(
        pd.DataFrame(
            [
                {"Metric": name, "Value": validation[name]}
                for name in [
                    "roc_auc",
                    "average_precision",
                    "precision_at_k",
                    "recall_at_k",
                    "precision",
                    "recall",
                    "f1",
                    "brier",
                ]
            ]
        ),
        hide_index=True,
        width="stretch",
    )
    st.write("Evidence checks are engineering criteria, not course grading requirements:")
    for name, passed in result["evidence_gate"]["checks"].items():
        st.write(f"{'Passed' if passed else 'Not passed'}: {name}")
    st.caption(result["intervals"]["method"])
    st.write(
        f"Exploratory ROC-AUC interval: {result['intervals']['roc_auc'][0]:.3f}–{result['intervals']['roc_auc'][1]:.3f}."
    )
    st.write(result["explanation_method"])
    st.write(settings["label_assumption"])
    st.write(
        "No driving-hour, mileage, trip, shift, alert-severity or demographic fields were supplied. No 30/90-day trend is inferred from the 15-day history. A day without recorded alarms does not establish a day worked or safe driving."
    )
