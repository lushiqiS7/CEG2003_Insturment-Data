"""Exact Shapley values for a small set of observed alert groups.

The game value is the mean prediction on hybrids of this captain and a fixed
training-only background. Exactness refers to the six-group empirical game,
not individual-feature SHAP, causal effects, or a population expectation.
"""

from math import comb

import numpy as np
import pandas as pd


def grouped_shap(model, X, background, groups, batch_size=48):
    columns = list(X.columns)
    flattened = [column for group in groups.values() for column in group]
    if len(flattened) != len(set(flattened)) or set(flattened) != set(columns):
        raise ValueError("Explanation groups must partition all model features exactly once")
    if background.empty:
        raise ValueError("A training background is required")
    values, reference = X.to_numpy(dtype=float), background[columns].to_numpy(dtype=float)
    indices = [[columns.index(column) for column in group] for group in groups.values()]
    size, coalition_count = len(groups), 2 ** len(groups)
    masks = np.array(
        [[bool(mask & (1 << group)) for group in range(size)] for mask in range(coalition_count)]
    )
    results = np.zeros((len(X), size))
    base = float(model.predict_proba(background[columns])[:, 1].mean())
    for start in range(0, len(X), batch_size):
        current = values[start : start + batch_size]
        hybrids = np.broadcast_to(
            reference, (len(current), coalition_count, *reference.shape)
        ).copy()
        for group, feature_indices in enumerate(indices):
            for feature_index in feature_indices:
                hybrids[..., feature_index] = np.where(
                    masks[None, :, None, group],
                    current[:, None, None, feature_index],
                    hybrids[..., feature_index],
                )
        probabilities = model.predict_proba(
            pd.DataFrame(hybrids.reshape(-1, len(columns)), columns=columns)
        )[:, 1]
        game = probabilities.reshape(len(current), coalition_count, len(reference)).mean(axis=2)
        for group in range(size):
            for mask in range(coalition_count):
                if not mask & (1 << group):
                    weight = 1 / (size * comb(size - 1, mask.bit_count()))
                    results[start : start + len(current), group] += weight * (
                        game[:, mask | (1 << group)] - game[:, mask]
                    )
    prediction = model.predict_proba(X)[:, 1]
    np.testing.assert_allclose(base + results.sum(axis=1), prediction, atol=1e-9)
    return pd.DataFrame(results, index=X.index, columns=list(groups)), base


def factor_text(row, count=3):
    ordered = row.abs().sort_values(ascending=False).index[:count]
    return "; ".join(f"{name} {row[name] * 100:+.2f} pp" for name in ordered)
