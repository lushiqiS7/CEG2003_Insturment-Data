"""Boundary, leakage, reconciliation and saved-inference checks."""

import json
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import pytest

from src.common import ROOT, config
from src.excel_stream import normalize_key, parse_date
from src.features import attach_target, build_features
from src.validation import split_cohort, threshold_for_capacity


def test_source_key_and_date_formats():
    assert normalize_key(123.0) == "123"
    assert normalize_key("00123") == "00123"
    assert normalize_key(" 307A ") == "307A"
    assert normalize_key(" ") is None
    assert parse_date("20-May-2024 (Mon)") == datetime(2024, 5, 20)
    assert parse_date("2024-03-15 23:59:59") == datetime(2024, 3, 15, 23, 59, 59)
    assert parse_date("bad") is None


def test_feature_cutoff_and_unassigned_drivers():
    settings = config()
    events = pd.DataFrame(
        {
            "captain_id": ["A", "A", "A", None],
            "alarm_type": ["Harsh Braking"] * 4,
            "start": pd.to_datetime(
                ["2024-03-01 12:00", "2024-03-15 23:59", "2024-03-16 00:00", "2024-03-15 12:00"]
            ),
            "day": pd.to_datetime(["2024-03-01", "2024-03-15", "2024-03-16", "2024-03-15"]),
            "night": [False, True, True, False],
            "speed": [10.0] * 4,
            "duration": [2.0] * 4,
        }
    )
    features = build_features(events, settings)
    assert len(features) == 1
    assert features.iloc[0]["total_alarms"] == 2
    assert features.iloc[0]["alarm_days"] == 2
    assert features.iloc[0]["night_share"] == 0.5
    assert features.iloc[0]["last_alarm"] < pd.Timestamp("2024-03-16")


def test_outcome_horizon_and_ambiguous_exclusions():
    features = pd.DataFrame({"captain_id": ["A", "B", "C", "D", "E"]})
    incidents = pd.DataFrame(
        {
            "captain_id": ["A", "B", "C", "D", "E", "E"],
            "day": pd.to_datetime(
                ["2024-03-15", "2024-03-16", "2024-04-01", "2024-03-20", "2024-03-20", "2024-03-21"]
            ),
            "included": [True, True, True, False, False, True],
        }
    )
    joined = attach_target(features, incidents, config()).set_index("captain_id")
    assert joined["target"].to_dict() == {"A": 0, "B": 1, "C": 0, "D": 0, "E": 1}
    assert not joined.loc["D", "label_known"]
    assert joined.loc["E", "label_known"]
    invalid = dict(config(), outcome_start="2024-03-15")
    with pytest.raises(ValueError, match="must follow"):
        attach_target(features, incidents, invalid)


def test_driver_partition_isolation_and_reproducibility():
    data = pd.DataFrame(
        {"captain_id": [f"synthetic-{i}" for i in range(100)], "target": [1] * 20 + [0] * 80}
    )
    a, b = split_cohort(data, 42), split_cohort(data, 42)
    assert {name: len(rows) for name, rows in a.items()} == {
        "train": 60,
        "validation": 20,
        "test": 20,
    }
    for name in a:
        assert np.array_equal(a[name], b[name])
        assert data.loc[a[name], "target"].nunique() == 2
    assert not set(a["train"]) & set(a["test"])
    duplicate = data.copy()
    duplicate.loc[1, "captain_id"] = duplicate.loc[0, "captain_id"]
    with pytest.raises(ValueError, match="one feature row"):
        split_cohort(duplicate, 42)


def test_threshold_uses_validation_scores_and_handles_ties():
    assert threshold_for_capacity(np.arange(10), 0.1) == 9
    assert threshold_for_capacity(np.zeros(10), 0.1) == 0
    with pytest.raises(ValueError):
        threshold_for_capacity([], 0.1)


@pytest.mark.skipif(
    not (ROOT / "outputs/metrics.json").exists(), reason="Training outputs not built"
)
def test_actual_saved_model_and_data_reconcile():
    prep = json.loads((ROOT / "outputs/preparation.json").read_text())
    result = json.loads((ROOT / "outputs/metrics.json").read_text())
    daily = pd.read_parquet(ROOT / "data/processed/daily.parquet")
    driver_daily = pd.read_parquet(ROOT / "data/processed/driver_daily.parquet")
    assert daily["events"].sum() == driver_daily["events"].sum() == prep["counts"]["clean_events"]
    assert prep["counts"]["raw_events"] == 2266768
    artifact = joblib.load(ROOT / "models/incident_model.joblib")
    scores = pd.read_parquet(ROOT / "outputs/driver_scores.parquet")
    from src.risk_training import predict_captains

    probability, thresholds = predict_captains(artifact, scores)
    np.testing.assert_allclose(probability, scores["probability"])
    np.testing.assert_allclose(thresholds, scores["probability_threshold"])
    assert scores["captain_id"].is_unique
    assert result["development_positives"] == 16
    assert result["legacy_test_drivers_excluded"] == 302
    splits = pd.read_parquet(ROOT / "outputs/splits.parquet")
    assert splits["captain_id"].is_unique
    assert len(splits) == prep["model_cohort_drivers"]
