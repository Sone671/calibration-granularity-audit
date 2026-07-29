"""Strictly-forward exponentially recency-weighted CQR baselines.

This extension leaves the frozen benchmark untouched.  It reuses the same
persistence-quantile backbone, users, operational segments, windows and
training-only scales, while adding recency-weighted global, segment and user
calibration quantiles.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd

# The persistence-only extension reuses data loaders from the frozen benchmark.
# Those loaders share a module with optional LightGBM training code, although no
# LightGBM object is touched here.  A minimal import stub keeps this extension
# independent of the compiled LightGBM runtime and avoids changing the frozen
# loader modules.
if "lightgbm" not in sys.modules:
    sys.modules["lightgbm"] = types.ModuleType("lightgbm")

import run_naive_robustness as frozen


HERE = Path(__file__).resolve().parent
HALF_LIFE_DAYS = 14.0
TEST_ATOM_WEIGHT = 1.0
METHODS = (
    "rolling_global_norm",
    "rolling_group_norm",
    "rolling_user_norm",
    "erw_global_norm",
    "erw_group_norm",
    "erw_user_norm",
)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def weighted_conformal_quantile(
    scores: np.ndarray,
    weights: np.ndarray,
    target: float,
    test_atom_weight: float = TEST_ATOM_WEIGHT,
) -> float:
    scores = np.asarray(scores, dtype=float)
    weights = np.asarray(weights, dtype=float)
    valid = np.isfinite(scores) & np.isfinite(weights) & (weights > 0)
    scores, weights = scores[valid], weights[valid]
    if scores.size == 0:
        raise RuntimeError("empty weighted conformal score set")
    threshold = target * (float(weights.sum()) + float(test_atom_weight))
    if threshold > float(weights.sum()):
        return math.inf
    order = np.argsort(scores, kind="mergesort")
    cumulative = np.cumsum(weights[order])
    index = int(np.searchsorted(cumulative, threshold, side="left"))
    return float(scores[order[min(index, scores.size - 1)]])


def interval_arrays_with_times(values, dt, start, stop, lag, low, high):
    start = max(start, lag)
    y2 = values[start:stop]
    lagged = values[start - lag:stop - lag]
    qlo2, qhi2 = lagged + low, lagged + high
    valid = np.isfinite(y2) & np.isfinite(qlo2) & np.isfinite(qhi2)
    time_rows, users = np.nonzero(valid)
    observed_times = dt[start:stop].to_numpy()[time_rows]
    return (
        y2[valid].astype(float),
        qlo2[valid].astype(float),
        qhi2[valid].astype(float),
        users.astype(int),
        observed_times,
    )


def calibration_corrections(y, qlo, qhi, users, observed_times, cutoff, labels, scales, target, half_life_days=HALF_LIFE_DAYS):
    scores = np.maximum(qlo - y, y - qhi) / scales[users]
    age_days = (np.datetime64(cutoff.to_datetime64()) - observed_times) / np.timedelta64(1, "D")
    weights = np.power(2.0, -np.asarray(age_days, dtype=float) / half_life_days)
    if np.any(age_days <= 0):
        raise RuntimeError("weighted calibration includes a non-historical observation")

    n_groups = int(np.max(labels)) + 1
    classic_global = frozen.conformal_quantile(scores, target)
    classic_group = np.array([
        frozen.conformal_quantile(scores[labels[users] == group], target)
        for group in range(n_groups)
    ])
    classic_user = np.array([
        frozen.conformal_quantile(scores[users == user], target)
        for user in range(len(labels))
    ])
    weighted_global = weighted_conformal_quantile(scores, weights, target)
    weighted_group = np.array([
        weighted_conformal_quantile(
            scores[labels[users] == group], weights[labels[users] == group], target
        )
        for group in range(n_groups)
    ])
    weighted_user = np.array([
        weighted_conformal_quantile(scores[users == user], weights[users == user], target)
        for user in range(len(labels))
    ])
    return {
        "rolling_global_norm": np.full(len(labels), classic_global),
        "rolling_group_norm": classic_group[labels],
        "rolling_user_norm": classic_user,
        "erw_global_norm": np.full(len(labels), weighted_global),
        "erw_group_norm": weighted_group[labels],
        "erw_user_norm": weighted_user,
    }, {
        "mean_calibration_age_days": float(np.average(age_days, weights=weights)),
        "effective_weight_sum": float(weights.sum()),
        "erw_global_correction": float(weighted_global),
    }


def run_configuration(dataset, values, dt, names, labels, scales, train0, train1, windows, cadence, coverage, horizon, half_life_days=HALF_LIFE_DAYS):
    lag = int(round(horizon * 60 / cadence))
    low, high = frozen.residual_quantiles(values, train0, train1, lag, coverage)
    configuration = f"persistence_qi__c{int(coverage * 100)}__h{horizon:g}"
    window_rows, user_rows = [], []
    for window, ts, te in windows:
        frozen.log(f"ERW {dataset} coverage={coverage:.2f} horizon={horizon:g}h window={window}")
        cal0 = int(dt.searchsorted(ts - pd.Timedelta(days=frozen.CALIBRATION_DAYS)))
        test0, test1 = int(dt.searchsorted(ts)), int(dt.searchsorted(te))
        y, qlo, qhi, users, observed_times = interval_arrays_with_times(
            values, dt, cal0, test0, lag, low, high
        )
        corrections, weight_diag = calibration_corrections(
            y, qlo, qhi, users, observed_times, ts, labels, scales, coverage, half_life_days
        )
        y, qlo, qhi, users = frozen.interval_arrays(values, test0, test1, lag, low, high)
        for method in METHODS:
            metrics, user_cov, user_n = frozen.evaluate(
                y, qlo, qhi, users, labels, scales, corrections[method], coverage
            )
            window_rows.append({
                "dataset": frozen.DATASET_NAMES[dataset],
                "configuration": configuration,
                "window": window,
                "coverage": coverage,
                "horizon_hours": horizon,
                "method": method,
                "half_life_days": half_life_days if method.startswith("erw_") else "",
                **metrics,
                **weight_diag,
            })
            for user in range(len(labels)):
                user_rows.append({
                    "dataset": frozen.DATASET_NAMES[dataset],
                    "configuration": configuration,
                    "window": window,
                    "coverage_target": coverage,
                    "horizon_hours": horizon,
                    "method": method,
                    "user_index": user,
                    "customer": names[user],
                    "cluster": int(labels[user]),
                    "coverage": float(user_cov[user]),
                    "n": int(user_n[user]),
                })
        gc.collect()
    return window_rows, user_rows


def summarize(window_rows: list[dict]) -> list[dict]:
    frame = pd.DataFrame(window_rows)
    frame["l05"] = 0.5 * (
        frame["macro_user_abs_coverage_gap"] + frame["max_abs_cluster_coverage_gap"]
    )
    keys = ["dataset", "configuration", "coverage", "horizon_hours"]
    global_score = frame[frame.method == "rolling_global_norm"][keys + ["window", "winkler_interval_score"]].rename(
        columns={"winkler_interval_score": "global_score"}
    )
    frame = frame.merge(global_score, on=keys + ["window"], validate="many_to_one")
    frame["relative_score_vs_global"] = frame["winkler_interval_score"] / frame["global_score"] - 1.0
    rows = []
    for group_keys, part in frame.groupby(keys + ["method"], sort=True):
        dataset, configuration, coverage, horizon, method = group_keys
        rows.append({
            "dataset": dataset,
            "configuration": configuration,
            "coverage": coverage,
            "horizon_hours": horizon,
            "method": method,
            "windows": int(len(part)),
            "mean_picp_abs_gap": float(np.mean(np.abs(part.picp - coverage))),
            "mean_macro_user_gap": float(part.macro_user_abs_coverage_gap.mean()),
            "mean_max_segment_gap": float(part.max_abs_cluster_coverage_gap.mean()),
            "mean_l05": float(part.l05.mean()),
            "mean_relative_score_vs_global": float(part.relative_score_vs_global.mean()),
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=tuple(frozen.DATASET_NAMES), required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--prepared", type=Path, default=frozen.LONDON)
    parser.add_argument("--raw-dir", type=Path, default=frozen.EXT / "raw_archive")
    parser.add_argument("--zip", type=Path, default=frozen.V1 / "electricityloaddiagrams20112014.originalmirror.zip")
    parser.add_argument("--coverages", nargs="+", type=float, default=[0.8, 0.9])
    parser.add_argument("--horizons", nargs="+", type=float, default=[1.0, 6.0])
    parser.add_argument("--half-life-days", type=float, default=HALF_LIFE_DAYS)
    args = parser.parse_args()

    started = time.perf_counter()
    out = args.out or HERE / f"weighted_conformal_{args.dataset}"
    out.mkdir(parents=True, exist_ok=True)
    if args.dataset == "london":
        data = frozen.load_london(args.prepared, 3, "kmeans")
    elif args.dataset == "ausgrid":
        data = frozen.load_ausgrid(args.raw_dir, 3, "kmeans")
    else:
        data = frozen.load_uci(args.zip, 3, "kmeans")
    dt, values, names, labels, scales, train0, train1, windows, cadence = data
    window_rows, user_rows = [], []
    for coverage in args.coverages:
        for horizon in args.horizons:
            new_windows, new_users = run_configuration(
                args.dataset, values, dt, names, labels, scales, train0, train1,
                windows, cadence, coverage, horizon, args.half_life_days,
            )
            window_rows.extend(new_windows)
            user_rows.extend(new_users)
    summary_rows = summarize(window_rows)
    write_csv(out / "window_metrics.csv", window_rows)
    write_csv(out / "per_user_window_metrics.csv", user_rows)
    write_csv(out / "summary.csv", summary_rows)
    metadata = {
        "dataset": frozen.DATASET_NAMES[args.dataset],
        "users": len(names),
        "windows": len(windows),
        "half_life_days": args.half_life_days,
        "test_atom_weight": TEST_ATOM_WEIGHT,
        "strict_forward": True,
        "wall_seconds": time.perf_counter() - started,
        "protocol": ["FROZEN_WEIGHTED_CONFORMAL_BASELINE_ADDENDUM.md", "FROZEN_ERW_SENSITIVITY_ADDENDUM.md"],
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    frozen.log(json.dumps(metadata))


if __name__ == "__main__":
    main()
