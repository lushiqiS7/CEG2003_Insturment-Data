"""Test temporal signal definitions, exact group explanations and validation isolation."""

import json

import numpy as np
import pandas as pd
import pytest

from src.common import ROOT, config
from src.explain import grouped_shap
from src.features import RISK_FEATURE_COLUMNS, RISK_GROUPS, build_risk_features
from src.risk_training import validate_registry
from src.validation import validate_development


def test_risk_temporal_features_and_recency():
    times = [
        "2024-03-02 10:00",
        "2024-03-02 10:10",
        "2024-03-15 23:00",
        "2024-03-15 23:05",
        "2024-03-16 00:00",
    ]
    starts = pd.to_datetime(times)
    events = pd.DataFrame(
        {
            "captain_id": ["A"] * 5,
            "start": starts,
            "day": starts.normalize(),
            "alarm_type": [
                "Fatigue Alert",
                "Forward Collision Warning",
                "Yawn",
                "Forward Collision Warning",
                "Yawn",
            ],
            "speed": [10.0] * 5,
            "duration": [2.0] * 5,
            "route": [None] * 5,
            "night": [False] * 5,
        }
    )
    result = build_risk_features(events, config()).iloc[0]
    assert result["fatigue_events"] == 2
    assert result["collision_events"] == 2
    assert result["night_fatigue_share"] == 0.5
    assert result["fatigue_before_collision_share"] == 1.0
    assert result["fatigue_recency_days"] == pytest.approx(1 / 24)
    assert result["fatigue_recent_7d"] == 1 and result["fatigue_previous_7d"] == 1
    assert result["missing_route_share"] == 1
    assert len(RISK_FEATURE_COLUMNS) == len(set(RISK_FEATURE_COLUMNS))
    assert set(RISK_FEATURE_COLUMNS) == {c for group in RISK_GROUPS.values() for c in group}


def test_no_collision_ratio_is_missing_not_zero():
    starts = pd.to_datetime(["2024-03-01 12:00", "2024-03-15 12:00"])
    events = pd.DataFrame(
        {
            "captain_id": ["A", "A"],
            "start": starts,
            "day": starts.normalize(),
            "alarm_type": ["Harsh Braking"] * 2,
            "speed": [10.0, 10.0],
            "duration": [2.0, 2.0],
            "route": [None, None],
            "night": [False, False],
        }
    )
    result = build_risk_features(events, config()).iloc[0]
    assert pd.isna(result["fatigue_before_collision_share"])
    assert pd.isna(result["fatigue_recency_days"])
    events["alarm_type"] = "Forward Collision Warning"
    result = build_risk_features(events, config()).iloc[0]
    assert result["fatigue_before_collision_share"] == 0


def test_development_requires_both_classes_and_enough_positives():
    with pytest.raises(ValueError, match="both"):
        validate_development(pd.DataFrame({"target": [0] * 30}))
    with pytest.raises(ValueError, match="12"):
        validate_development(pd.DataFrame({"target": [0] * 30 + [1] * 11}))
    validate_development(pd.DataFrame({"target": [0] * 30 + [1] * 12}))


def test_partition_registry_rejects_invalid_membership():
    with pytest.raises(ValueError, match="duplicate"):
        validate_registry(pd.DataFrame({"captain_id": ["A", "A"], "partition": ["train", "test"]}))
    with pytest.raises(ValueError, match="invalid"):
        validate_registry(pd.DataFrame({"captain_id": ["A"], "partition": ["typo"]}))
    with pytest.raises(ValueError, match="missing"):
        validate_registry(pd.DataFrame({"captain_id": ["A"]}))


class LinearProbability:
    def predict_proba(self, frame):
        probability = 0.1 + 0.1 * frame["a"] + 0.2 * frame["b"]
        return np.column_stack([1 - probability, probability])


def test_exact_group_shapley_matches_known_solution():
    frame = pd.DataFrame({"a": [1.0, 2.0], "b": [2.0, 1.0]})
    background = pd.DataFrame({"a": [0.0, 1.0], "b": [0.0, 1.0]})
    values, base = grouped_shap(LinearProbability(), frame, background, {"A": ["a"], "B": ["b"]})
    np.testing.assert_allclose(values["A"], [0.05, 0.15])
    np.testing.assert_allclose(values["B"], [0.3, 0.1])
    np.testing.assert_allclose(
        base + values.sum(axis=1), LinearProbability().predict_proba(frame)[:, 1]
    )
    with pytest.raises(ValueError, match="partition"):
        grouped_shap(LinearProbability(), frame, background, {"Wrong": ["a"]})


@pytest.mark.skipif(
    not (ROOT / "outputs/nested_predictions.parquet").exists(), reason="Revised training not built"
)
def test_nested_driver_isolation_and_real_explanation_additivity():
    result = json.loads((ROOT / "outputs/metrics.json").read_text())
    registry = pd.read_parquet(ROOT / "data/processed/partition_registry.parquet")
    excluded_test = set(registry.loc[registry["partition"].eq("test"), "captain_id"])
    seen = set()
    for fold in result["outer_folds"]:
        train, validation = set(fold["train_ids"]), set(fold["validation_ids"])
        assert not train & validation
        assert not train & excluded_test and not validation & excluded_test
        assert not seen & validation
        seen |= validation
    assert len(seen) == result["development_drivers"]
    scores = pd.read_parquet(ROOT / "outputs/driver_scores.parquet")
    effects = scores[[f"shap_{name}" for name in RISK_GROUPS]]
    np.testing.assert_allclose(
        scores["shap_base"] + effects.sum(axis=1), scores["probability"], atol=1e-9
    )
    assert scores["risk_score"].between(0, 100).all()
    assert scores["probability"].between(0, 1).all()
    assert scores["top_factors"].str.count(";").eq(2).all()
