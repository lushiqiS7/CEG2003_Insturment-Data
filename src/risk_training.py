"""Nested model selection, cross-fitted calibration and grouped explanations."""

from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .common import ROOT, config, logger, save_json
from .excel_stream import sha256
from .explain import factor_text, grouped_shap
from .features import RISK_FEATURE_COLUMNS, RISK_GROUPS, attach_target, build_risk_features
from .validation import (
    bootstrap_intervals,
    evaluate,
    evidence_gate,
    split_cohort,
    threshold_for_capacity,
    validate_development,
)

LOG = logger("train")
CANDIDATES = ["Logistic C0.1", "Logistic C1", "Random forest", "Histogram boosting"]


def make_model(name, seed):
    if name.startswith("Logistic"):
        estimator = make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler(),
            LogisticRegression(
                C=0.1 if name.endswith("0.1") else 1.0,
                class_weight="balanced",
                max_iter=2000,
                random_state=seed,
            ),
        )
    elif name == "Random forest":
        estimator = make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            RandomForestClassifier(
                n_estimators=160,
                max_depth=4,
                min_samples_leaf=20,
                class_weight="balanced_subsample",
                random_state=seed,
                n_jobs=2,
            ),
        )
    elif name == "Histogram boosting":
        estimator = make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            HistGradientBoostingClassifier(
                max_iter=100,
                learning_rate=0.05,
                max_leaf_nodes=7,
                min_samples_leaf=30,
                l2_regularization=10.0,
                class_weight="balanced",
                random_state=seed,
            ),
        )
    else:
        raise ValueError(name)
    return CalibratedClassifierCV(
        estimator,
        method="sigmoid",
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=seed + 1),
        ensemble=True,
        n_jobs=1,
    )


def select_candidate(X, y, seed, fraction):
    results, predictions = {}, {}
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    for name in CANDIDATES:
        probability = cross_val_predict(
            make_model(name, seed), X, y, cv=cv, method="predict_proba", n_jobs=1
        )[:, 1]
        threshold = threshold_for_capacity(probability, fraction)
        results[name] = evaluate(y, probability, threshold, fraction)
        predictions[name] = probability
        LOG.info(
            "Inner CV %s: precision@k %.4f, AP %.4f, ROC-AUC %.3f",
            name,
            results[name]["precision_at_k"],
            results[name]["average_precision"],
            results[name]["roc_auc"],
        )
    chosen = max(
        CANDIDATES,
        key=lambda name: (
            results[name]["precision_at_k"],
            results[name]["average_precision"],
            results[name]["roc_auc"],
        ),
    )
    threshold = threshold_for_capacity(predictions[chosen], fraction)
    return chosen, threshold, results


def partition_registry(features, seed):
    path = ROOT / "data/processed/partition_registry.parquet"
    if path.exists():
        registry = pd.read_parquet(path)
        validate_registry(registry)
        return registry
    previous = ROOT / "outputs/splits.parquet"
    if previous.exists():
        registry = pd.read_parquet(previous)[["captain_id", "partition"]]
    else:
        cohort = features.loc[features["label_known"]].reset_index(drop=True)
        partitions = split_cohort(cohort, seed)
        registry = pd.concat(
            [
                cohort.loc[index, ["captain_id"]].assign(partition=name)
                for name, index in partitions.items()
            ],
            ignore_index=True,
        )
    validate_registry(registry)
    registry.to_parquet(path, index=False)
    return registry


def validate_registry(registry):
    if not {"captain_id", "partition"}.issubset(registry.columns):
        raise ValueError("Partition registry is missing captain_id or partition")
    if registry["captain_id"].isna().any() or not registry["captain_id"].is_unique:
        raise ValueError("Partition registry contains missing or duplicate captains")
    if not registry["partition"].isin(["train", "validation", "test"]).all():
        raise ValueError("Partition registry contains an invalid partition")


@threadpool_limits.wrap(limits=2)
def predict_captains(artifact, features):
    probability = np.empty(len(features))
    thresholds = np.empty(len(features))
    folds = features["captain_id"].map(artifact["model_fold"]).fillna(-1).astype(int)
    for fold in sorted(folds.unique()):
        indices = np.flatnonzero(folds.to_numpy() == fold)
        model = artifact["model"] if fold == -1 else artifact["fold_models"][fold]
        probability[indices] = model.predict_proba(features.iloc[indices][artifact["features"]])[
            :, 1
        ]
        thresholds[indices] = (
            artifact["threshold"] if fold == -1 else artifact["fold_thresholds"][fold]
        )
    return probability, thresholds


@threadpool_limits.wrap(limits=2)
def train():
    settings = config()
    fraction, seed = settings["review_fraction"], settings["random_seed"]
    LOG.info(
        "Building risk features from existing cleaned events; legacy test captains stay outside selection"
    )
    events = pd.read_parquet(ROOT / "data/processed/events.parquet")
    if (
        pd.Timestamp(settings["feature_start"]) < events["start"].min().normalize()
        or pd.Timestamp(settings["feature_end"]) > events["start"].max().normalize()
    ):
        raise ValueError("Requested feature period exceeds supplied alarm coverage")
    features = build_risk_features(events, settings)
    incidents = pd.read_parquet(ROOT / "data/processed/incidents.parquet")
    incidents["included"] = ~incidents["flag"].isin(settings["excluded_incident_flags"])
    features = attach_target(features, incidents, settings)
    registry = partition_registry(features, seed)
    features = features.merge(registry, on="captain_id", how="left", validate="one_to_one")
    development = (
        features.loc[features["label_known"] & features["partition"].isin(["train", "validation"])]
        .copy()
        .reset_index(drop=True)
    )
    legacy_test_ids = set(registry.loc[registry["partition"].eq("test"), "captain_id"])
    assert not set(development["captain_id"]) & legacy_test_ids
    validate_development(development)
    X, y = development[RISK_FEATURE_COLUMNS], development["target"]
    probability, thresholds, baseline = np.empty(len(X)), np.empty(len(X)), np.empty(len(X))
    fold_models, fold_backgrounds, fold_thresholds, fold_details, model_fold = [], [], [], [], {}
    outer = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    for fold, (train_index, validation_index) in enumerate(outer.split(X, y)):
        LOG.info(
            "Outer fold %s/3: model selection uses %s captains and %s positives",
            fold + 1,
            len(train_index),
            int(y.iloc[train_index].sum()),
        )
        name, threshold, candidates = select_candidate(
            X.iloc[train_index], y.iloc[train_index], seed + fold * 10, fraction
        )
        model = make_model(name, seed + fold * 10).fit(X.iloc[train_index], y.iloc[train_index])
        probability[validation_index] = model.predict_proba(X.iloc[validation_index])[:, 1]
        thresholds[validation_index] = threshold
        baseline[validation_index] = y.iloc[train_index].mean()
        fold_models.append(model)
        fold_backgrounds.append(X.iloc[train_index].sample(n=8, random_state=seed))
        fold_thresholds.append(threshold)
        model_fold.update(
            {captain: fold for captain in development.iloc[validation_index]["captain_id"]}
        )
        fold_result = evaluate(
            y.iloc[validation_index], probability[validation_index], threshold, fraction
        )
        fold_details.append(
            {
                "fold": fold,
                "selected": name,
                "train_ids": development.iloc[train_index]["captain_id"].tolist(),
                "validation_ids": development.iloc[validation_index]["captain_id"].tolist(),
                "metrics": fold_result,
                "inner_candidates": candidates,
            }
        )
        LOG.info(
            "Outer fold %s selected %s; unseen-fold AP %.4f, ROC-AUC %.3f, recall@k %.3f",
            fold + 1,
            name,
            fold_result["average_precision"],
            fold_result["roc_auc"],
            fold_result["recall_at_k"],
        )
    nested = evaluate(y, probability, thresholds, fraction)
    baseline_result = evaluate(y, baseline, 0.5, fraction)
    intervals = bootstrap_intervals(y, probability, seed)
    gate = evidence_gate(nested, baseline_result, intervals)
    LOG.info(
        "Nested OOF: AP %.4f, ROC-AUC %.3f, precision@k %.3f, recall@k %.3f; evidence checks passed=%s",
        nested["average_precision"],
        nested["roc_auc"],
        nested["precision_at_k"],
        nested["recall_at_k"],
        gate["passed"],
    )
    LOG.info("Selecting the final model using development-only inner CV")
    name, threshold, candidates = select_candidate(X, y, seed, fraction)
    model = make_model(name, seed).fit(X, y)
    artifact = {
        "model": model,
        "features": RISK_FEATURE_COLUMNS,
        "name": name,
        "groups": RISK_GROUPS,
        "threshold": threshold,
        "settings": settings,
        "model_fold": model_fold,
        "fold_models": fold_models,
        "fold_thresholds": fold_thresholds,
        "fold_backgrounds": fold_backgrounds,
        "background": X.sample(n=8, random_state=seed),
        "preparation_sha256": sha256(ROOT / "outputs/preparation.json"),
        "evidence_gate": gate,
        "evaluation_scope": "Nested CV on development captains; legacy test excluded",
    }
    features["probability"], features["probability_threshold"] = predict_captains(
        artifact, features
    )
    # Scores are relative priority, not a probability multiplied by an arbitrary factor.
    features["risk_score"] = (
        np.searchsorted(np.sort(probability), features["probability"], side="right")
        / len(probability)
        * 100
    )
    features["provisional_at_risk"] = features["probability"].ge(features["probability_threshold"])
    features["flag"] = np.where(
        features["provisional_at_risk"], "Review suggested", "Below review threshold"
    )
    features["score_source"] = np.where(
        features["captain_id"].isin(model_fold),
        "Cross-fitted development",
        "Excluded from development",
    )
    features["shap_base"] = np.nan
    explanations = pd.DataFrame(index=features.index, columns=list(RISK_GROUPS), dtype=float)
    for fold in [-1, 0, 1, 2]:
        indices = features.index[
            features["captain_id"].map(model_fold).fillna(-1).astype(int).eq(fold)
        ]
        if len(indices) == 0:
            continue
        LOG.info(
            "Computing exact six-group SHAP for %s captains (model fold %s)", len(indices), fold
        )
        explaining_model = model if fold == -1 else fold_models[fold]
        background = artifact["background"] if fold == -1 else fold_backgrounds[fold]
        values, base = grouped_shap(
            explaining_model, features.loc[indices, RISK_FEATURE_COLUMNS], background, RISK_GROUPS
        )
        explanations.loc[indices] = values
        features.loc[indices, "shap_base"] = base
    features["top_factors"] = explanations.apply(factor_text, axis=1)
    for group in RISK_GROUPS:
        features[f"shap_{group}"] = explanations[group]
    np.testing.assert_allclose(
        features["shap_base"] + explanations.sum(axis=1), features["probability"], atol=1e-9
    )
    del events
    features.to_parquet(ROOT / "data/processed/risk_features.parquet", index=False)
    features.sort_values(["risk_score", "captain_id"], ascending=[False, True]).to_parquet(
        ROOT / "outputs/driver_scores.parquet", index=False
    )
    development[["captain_id", "target"]].assign(
        probability=probability,
        threshold=thresholds,
        fold=development["captain_id"].map(model_fold),
    ).to_parquet(ROOT / "outputs/nested_predictions.parquet", index=False)
    joblib.dump(artifact, ROOT / "models/incident_model.joblib")
    observed_rate, mean_probability = calibration_curve(
        y, probability, n_bins=5, strategy="quantile"
    )
    calibration = [
        {"observed_rate": float(observed), "mean_probability": float(predicted)}
        for observed, predicted in zip(observed_rate, mean_probability)
    ]
    fairness = []
    for label, mask in {
        "At least 25% night alarms": development["night_share"].ge(0.25),
        "Under 25% night alarms": development["night_share"].lt(0.25),
    }.items():
        indices = np.flatnonzero(mask.to_numpy())
        fairness.append(
            {
                "group": label,
                "drivers": len(indices),
                "positives": int(y.iloc[indices].sum()),
                "flag_rate": float((probability[indices] >= thresholds[indices]).mean())
                if len(indices)
                else None,
                "limitation": "Coverage audit only; no demographics, exposure, or reliable causal fairness conclusion",
            }
        )
    report = {
        "schema_version": 2,
        "created_at": datetime.now().astimezone().isoformat(),
        "selected_model": name,
        "settings": settings,
        "preparation_sha256": artifact["preparation_sha256"],
        "feature_hash": sha256(ROOT / "data/processed/risk_features.parquet"),
        "model_hash": sha256(ROOT / "models/incident_model.joblib"),
        "scores_hash": sha256(ROOT / "outputs/driver_scores.parquet"),
        "daily_hash": sha256(ROOT / "data/processed/driver_daily.parquet"),
        "features": RISK_FEATURE_COLUMNS,
        "groups": RISK_GROUPS,
        "cohort_drivers": len(features),
        "development_drivers": len(development),
        "development_positives": int(y.sum()),
        "legacy_test_drivers_excluded": len(legacy_test_ids),
        "validation": nested,
        "baseline": baseline_result,
        "intervals": intervals,
        "evidence_gate": gate,
        "outer_folds": fold_details,
        "final_selection": candidates,
        "calibration": calibration,
        "coverage_audit": fairness,
        "selection_rule": f"Inner-CV precision@{fraction:.0%}, then average precision, then ROC-AUC; four predefined candidates",
        "calibration_method": "Sigmoid calibration with three training-only folds at every fitted estimator",
        "explanation_method": "Exact six-group interventional Shapley values over eight training-only reference captains; 64 coalitions",
        "legacy_test_status": "Previously inspected before this revision; excluded from further selection and not reevaluated",
        "limitations": [
            f"Only {int(y.sum())} development positives; calibration and nested-CV uncertainty remain substantial",
            "No independent future cohort available; nested CV is development evidence, not external validation",
            "Incident flags and no-incident negatives remain provisional; follow-up unverified",
            "No driving hours, mileage, trips, shifts, alert severity or protected demographics supplied",
            "30-day/90-day trends and true exposure rates cannot be estimated from 15 days",
            "SHAP explains model predictions relative to a small empirical background, not causation",
        ],
    }
    save_json(ROOT / "outputs/metrics.json", report)
    save_json(
        ROOT / "outputs/runs" / datetime.now().strftime("%Y%m%d_%H%M%S") / "metrics.json", report
    )
    save_json(
        ROOT / "models/model_metadata.json",
        {
            "name": name,
            "features": RISK_FEATURE_COLUMNS,
            "groups": RISK_GROUPS,
            "settings": settings,
            "calibration": report["calibration_method"],
            "validation_scope": artifact["evaluation_scope"],
        },
    )
    LOG.info(
        "Saved calibrated models, cross-fitted scores, grouped SHAP, nested validation and evidence status"
    )
