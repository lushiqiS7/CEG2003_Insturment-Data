"""Driver-level evaluation and predeclared evidence checks."""

import math

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


def split_cohort(cohort, seed):
    if not cohort["captain_id"].is_unique:
        raise ValueError("Expected exactly one feature row per captain")
    if cohort["target"].nunique() != 2 or cohort["target"].value_counts().min() < 10:
        raise ValueError("Fewer than 10 drivers in a class; a three-way split is too sparse")
    development, test = train_test_split(
        cohort.index, test_size=0.2, stratify=cohort["target"], random_state=seed
    )
    train, validation = train_test_split(
        development, test_size=0.25, stratify=cohort.loc[development, "target"], random_state=seed
    )
    return {"train": train, "validation": validation, "test": test}


def validate_development(cohort):
    if cohort.empty or cohort["target"].nunique() != 2:
        raise ValueError("Development data must contain both incident and no-incident classes")
    if cohort["target"].value_counts().min() < 12:
        raise ValueError("At least 12 captains in each class are required for nested calibration")


def threshold_for_capacity(scores, fraction):
    if not 0 < fraction <= 1 or len(scores) == 0:
        raise ValueError("Review fraction must be in (0, 1] and scores nonempty")
    return float(np.quantile(scores, 1 - fraction, method="higher"))


def evaluate(y, scores, threshold, fraction):
    y, scores = np.asarray(y), np.asarray(scores)
    if len(np.unique(y)) != 2:
        raise ValueError("Evaluation requires both outcome classes")
    predicted = scores >= threshold
    k = max(1, math.ceil(len(y) * fraction))
    top = np.argsort(-scores, kind="stable")[:k]
    prevalence = float(y.mean())
    return {
        "drivers": len(y),
        "positives": int(y.sum()),
        "prevalence": prevalence,
        "average_precision": float(average_precision_score(y, scores)),
        "roc_auc": float(roc_auc_score(y, scores)),
        "brier": float(brier_score_loss(y, scores)),
        "log_loss": float(log_loss(y, scores)),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "confusion_matrix": confusion_matrix(y, predicted, labels=[0, 1]).tolist(),
        "review_k": k,
        "precision_at_k": float(y[top].mean()),
        "recall_at_k": float(y[top].sum() / y.sum()),
        "precision_lift_at_k": float(y[top].mean() / prevalence),
        "flagged_drivers": int(predicted.sum()),
        "ties_at_k": int((scores == scores[top[-1]]).sum()),
    }


def bootstrap_intervals(y, scores, seed, repetitions=500):
    y, scores = np.asarray(y), np.asarray(scores)
    rng, auc, ap = np.random.default_rng(seed), [], []
    for _ in range(repetitions):
        index = rng.integers(0, len(y), len(y))
        if len(np.unique(y[index])) == 2:
            auc.append(roc_auc_score(y[index], scores[index]))
            ap.append(average_precision_score(y[index], scores[index]))
    return {
        "roc_auc": np.quantile(auc, [0.025, 0.975]).tolist(),
        "average_precision": np.quantile(ap, [0.025, 0.975]).tolist(),
        "valid_resamples": len(auc),
        "method": "Exploratory driver bootstrap of fixed nested-CV predictions; folds are dependent",
    }


def evidence_gate(result, baseline, intervals):
    checks = {
        "Discrimination lower interval exceeds chance": intervals["roc_auc"][0] > 0.5,
        "Top-review precision is at least twice prevalence": result["precision_lift_at_k"] >= 2,
        "Top-review queue captures at least 25% of positives": result["recall_at_k"] >= 0.25,
        "Brier score improves on training-prior baseline": result["brier"] < baseline["brier"],
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "definition": "Predeclared engineering checks, not a course grading rubric or proof of deployment readiness",
    }
