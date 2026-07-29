"""Model-agnostic cross-fitted stability-screened calibration-granularity routing.

The router only needs base prediction intervals, a labelled rolling calibration
block, user identifiers, operational-segment identifiers, and user scales.  It
does not depend on the forecasting backbone or on labels from the deployment
window.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


GLOBAL = "rolling_global_norm"
GROUP = "rolling_group_norm"
USER = "rolling_user_norm"
POLICIES = (GLOBAL, GROUP, USER)


@dataclass(frozen=True)
class RouterDecision:
    selected_policy: str
    losses: dict[str, float]
    mean_gains_over_global: dict[str, float]
    lower_confidence_gains: dict[str, float]
    fold_count: int


def conformal_quantile(scores: np.ndarray, target: float) -> float:
    """Finite-sample split-conformal upper order statistic."""
    values = np.asarray(scores, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("cannot calibrate from an empty score array")
    rank = min(values.size, int(np.ceil((values.size + 1) * target)))
    return float(np.partition(values, rank - 1)[rank - 1])


def fit_corrections(
    y: np.ndarray,
    qlo: np.ndarray,
    qhi: np.ndarray,
    users: np.ndarray,
    groups_by_user: np.ndarray,
    scales: np.ndarray,
    target: float,
) -> dict[str, np.ndarray]:
    """Fit global, operational-segment, and per-user CQR corrections."""
    users = np.asarray(users, dtype=int)
    groups_by_user = np.asarray(groups_by_user, dtype=int)
    scales = np.asarray(scales, dtype=float)
    scores = np.maximum(qlo - y, y - qhi) / np.maximum(scales[users], 1e-12)
    n_users = len(groups_by_user)
    n_groups = int(groups_by_user.max()) + 1
    global_q = conformal_quantile(scores, target)
    observation_groups = groups_by_user[users]
    group_q = np.array(
        [conformal_quantile(scores[observation_groups == group], target) for group in range(n_groups)]
    )
    user_q = np.array(
        [conformal_quantile(scores[users == user], target) for user in range(n_users)]
    )
    return {
        GLOBAL: np.full(n_users, global_q, dtype=float),
        GROUP: group_q[groups_by_user],
        USER: user_q,
    }


def evaluate_policy(
    y: np.ndarray,
    qlo: np.ndarray,
    qhi: np.ndarray,
    users: np.ndarray,
    groups_by_user: np.ndarray,
    scales: np.ndarray,
    correction: np.ndarray,
    target: float,
) -> dict[str, float]:
    """Evaluate coverage at two granularities and a normalized interval score."""
    users = np.asarray(users, dtype=int)
    groups_by_user = np.asarray(groups_by_user, dtype=int)
    scales = np.asarray(scales, dtype=float)
    delta = correction[users] * scales[users]
    lo = np.asarray(qlo, dtype=float) - delta
    hi = np.asarray(qhi, dtype=float) + delta
    crossed = hi < lo
    midpoint = 0.5 * (lo[crossed] + hi[crossed])
    lo[crossed], hi[crossed] = midpoint, midpoint
    covered = (y >= lo) & (y <= hi)

    n_users = len(groups_by_user)
    user_n = np.bincount(users, minlength=n_users)
    if np.any(user_n == 0):
        raise ValueError("evaluation fold misses at least one frozen user")
    user_covered = np.bincount(users, weights=covered.astype(float), minlength=n_users)
    user_coverage = user_covered / user_n

    groups = groups_by_user[users]
    n_groups = int(groups_by_user.max()) + 1
    group_n = np.bincount(groups, minlength=n_groups)
    group_covered = np.bincount(groups, weights=covered.astype(float), minlength=n_groups)
    group_coverage = group_covered / group_n

    alpha = 1.0 - target
    interval_score = (
        hi - lo
        + (2.0 / alpha) * np.maximum(lo - y, 0.0)
        + (2.0 / alpha) * np.maximum(y - hi, 0.0)
    )
    normalized_interval_score = np.mean(interval_score / np.maximum(scales[users], 1e-12))
    return {
        "n": int(len(y)),
        "picp": float(np.mean(covered)),
        "mpiw": float(np.mean(hi - lo)),
        "macro_user_abs_coverage_gap": float(np.mean(np.abs(user_coverage - target))),
        "max_abs_cluster_coverage_gap": float(np.max(np.abs(group_coverage - target))),
        "normalized_interval_score": float(normalized_interval_score),
    }


def scalar_loss(metrics: dict[str, float], user_weight: float, efficiency_weight: float) -> float:
    """Scale-free deployment loss used by the router and its evaluation."""
    if not 0.0 <= user_weight <= 1.0:
        raise ValueError("user_weight must lie in [0, 1]")
    if efficiency_weight < 0.0:
        raise ValueError("efficiency_weight must be non-negative")
    return float(
        user_weight * metrics["macro_user_abs_coverage_gap"]
        + (1.0 - user_weight) * metrics["max_abs_cluster_coverage_gap"]
        + efficiency_weight * metrics["normalized_interval_score"]
    )


def select_policy(
    fold_metrics: Iterable[dict[str, dict[str, float]]],
    user_weight: float,
    efficiency_weight: float = 0.0,
    standard_error_multiplier: float = 1.0,
) -> RouterDecision:
    """Select a granularity only when its estimated gain over global is stable.

    For each non-global policy, the router computes its fold-wise loss gain over
    global and subtracts ``standard_error_multiplier`` standard errors.  A more
    local policy is used only when this lower-confidence gain is positive.
    Otherwise, the stable global calibrator is retained.
    """
    folds = list(fold_metrics)
    if not folds:
        raise ValueError("at least one chronological validation fold is required")
    losses = {
        policy: np.array(
            [scalar_loss(fold[policy], user_weight, efficiency_weight) for fold in folds],
            dtype=float,
        )
        for policy in POLICIES
    }
    mean_losses = {policy: float(values.mean()) for policy, values in losses.items()}
    mean_gains = {GLOBAL: 0.0}
    lower_gains = {GLOBAL: 0.0}
    for policy in (GROUP, USER):
        gains = losses[GLOBAL] - losses[policy]
        mean_gain = float(gains.mean())
        standard_error = float(gains.std(ddof=1) / np.sqrt(len(gains))) if len(gains) > 1 else np.inf
        mean_gains[policy] = mean_gain
        lower_gains[policy] = mean_gain - standard_error_multiplier * standard_error

    eligible = [policy for policy in (GROUP, USER) if lower_gains[policy] > 0.0]
    selected = max(eligible, key=lambda policy: (lower_gains[policy], policy == GROUP)) if eligible else GLOBAL
    return RouterDecision(
        selected_policy=selected,
        losses=mean_losses,
        mean_gains_over_global=mean_gains,
        lower_confidence_gains=lower_gains,
        fold_count=len(folds),
    )
