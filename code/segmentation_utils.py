"""Deterministic training-only operational-segmentation utilities."""

from __future__ import annotations

import numpy as np


def ward_labels(x: np.ndarray, k: int) -> np.ndarray:
    """Ward minimum-variance clustering without a scikit-learn dependency."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if not 1 < k <= n:
        raise ValueError(f"k must be in [2, {n}], received {k}")
    means = x.copy()
    counts = np.ones(n, dtype=float)
    active = np.ones(n, dtype=bool)
    members = [np.array([idx], dtype=int) for idx in range(n)]
    cost = np.full((n, n), np.inf, dtype=float)
    for left in range(n):
        diff = means[left + 1:] - means[left]
        values = .5 * np.einsum("ij,ij->i", diff, diff)
        cost[left, left + 1:] = values
        cost[left + 1:, left] = values
    for _ in range(n - k):
        left, right = np.unravel_index(np.argmin(cost), cost.shape)
        if left == right or not active[left] or not active[right]:
            raise RuntimeError("Ward active-cluster invariant failed")
        if left > right:
            left, right = right, left
        total = counts[left] + counts[right]
        means[left] = (counts[left] * means[left] + counts[right] * means[right]) / total
        counts[left] = total
        members[left] = np.concatenate([members[left], members[right]])
        members[right] = np.empty(0, dtype=int)
        active[right] = False
        cost[right, :] = np.inf
        cost[:, right] = np.inf
        indices = np.flatnonzero(active)
        indices = indices[indices != left]
        if len(indices):
            diff = means[indices] - means[left]
            values = counts[left] * counts[indices] / (counts[left] + counts[indices])
            values *= np.einsum("ij,ij->i", diff, diff)
            cost[left, indices] = values
            cost[indices, left] = values
        cost[left, left] = np.inf
    clusters = [idx for idx in np.flatnonzero(active)]
    clusters.sort(key=lambda idx: int(np.min(members[idx])))
    labels = np.empty(n, dtype=np.int32)
    for label, idx in enumerate(clusters):
        labels[members[idx]] = label
    return labels


def cluster_labels(x: np.ndarray, k: int, method: str, seed: int, kmeans):
    if method == "kmeans":
        labels, _ = kmeans(x, k, seed)
        return labels.astype(np.int32, copy=False)
    if method == "ward":
        return ward_labels(x, k)
    raise ValueError(f"unknown cluster method: {method}")
