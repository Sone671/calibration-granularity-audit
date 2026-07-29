"""Frozen persistence-quantile robustness benchmark for London, Ausgrid and UCI."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
LONDON = ROOT / "probabilistic_load_v5_london_final_confirmation_2026-08-02" / "prepared_london"
EXT = ROOT / "probabilistic_load_hierarchical_rolling_cqr_external_2026-07-30"
V1 = ROOT / "probabilistic_load_group_cqr_go_nogo_2026-07-25"
for path in (HERE, EXT, V1 / ".deps", V1):
    sys.path.insert(0, str(path))

import diagnostic_metrics as diag  # noqa: E402
import segmentation_utils as seg  # noqa: E402
import run_external as ext  # noqa: E402
import run_validation as base  # noqa: E402
METHODS = ("raw", "rolling_global_norm", "rolling_group_norm", "rolling_user_norm")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

# Keep the phase-1 operational-segment assignment exactly frozen.
SEED = 20260725
CALIBRATION_DAYS = 56
DATASET_NAMES = {"london": "London", "ausgrid": "Ausgrid", "uci": "UCI Electricity"}


def log(message: str) -> None:
    print(time.strftime("[%H:%M:%S]"), message, flush=True)


def conformal_quantile(scores: np.ndarray, target: float) -> float:
    scores = np.asarray(scores, dtype=float)
    scores = scores[np.isfinite(scores)]
    if scores.size == 0:
        raise RuntimeError("empty conformal score set")
    rank = min(scores.size - 1, math.ceil((scores.size + 1) * target) - 1)
    return float(np.partition(scores, rank)[rank])


def cluster_profiles(values: np.ndarray, dt: pd.DatetimeIndex, train0: int, train1: int, use_uci: bool, segments: int, cluster_method: str):
    train = values[train0:train1]
    if use_uci:
        features, _ = base.user_features(train, dt[train0:train1])
    else:
        features = ext.user_features(train, dt[train0:train1])
    center, spread = np.median(features, axis=0), np.std(features, axis=0)
    standardized = (features - center) / np.where(spread > 1e-9, spread, 1)
    labels = seg.cluster_labels(standardized, segments, cluster_method, SEED, base.kmeans)
    stat = np.column_stack([
        np.nanmean(train, axis=0),
        np.nanstd(train, axis=0),
        np.nanquantile(train, .95, axis=0),
        np.nanmean(train <= 1e-6, axis=0),
    ]).astype(np.float32)
    scales = np.maximum.reduce([stat[:, 1], .1 * stat[:, 2], np.full(values.shape[1], 1e-3)]).astype(float)
    return labels.astype(int), scales


def load_london(prepared: Path, segments: int, cluster_method: str):
    meta = json.loads((prepared / "metadata.json").read_text(encoding="utf-8"))
    values = np.load(prepared / "values.npy").astype(np.float32, copy=False)
    names = list(meta["customers"])
    start = pd.Timestamp(meta["start"]).normalize()
    dt = pd.date_range(start, periods=len(values), freq="30min")
    train0, train1 = 0, int(dt.searchsorted(start + pd.DateOffset(months=12)))
    windows = []
    for off in range(12, 23):
        ws = start + pd.DateOffset(months=off)
        windows.append((ws.strftime("%Y-%m"), ws, start + pd.DateOffset(months=off + 1)))
    labels, scales = cluster_profiles(values, dt, train0, train1, use_uci=False, segments=segments, cluster_method=cluster_method)
    return dt, values, names, labels, scales, train0, train1, windows, 30


def load_ausgrid(raw_dir: Path, segments: int, cluster_method: str):
    dt, raw, all_names, _ = ext.read_ausgrid(raw_dir)
    train0, train1 = ext.slice_bounds(dt, "2010-07-01", "2012-07-01")
    windows = []
    for off in range(12):
        ws = pd.Timestamp("2012-07-01") + pd.DateOffset(months=off)
        windows.append((ws.strftime("%Y-%m"), ws, ws + pd.DateOffset(months=1)))
    eligible = np.mean(np.isfinite(raw[train0:train1]), axis=0) >= .95
    for _, ws, we in windows:
        a, b = int(dt.searchsorted(ws)), int(dt.searchsorted(we))
        eligible &= np.mean(np.isfinite(raw[a:b]), axis=0) >= .95
    train = raw[train0:train1]
    eligible &= np.nanmean(train > 1e-6, axis=0) >= .2
    eligible &= np.nanstd(train, axis=0) > 1e-9
    selected = np.flatnonzero(eligible)
    if selected.size < 100:
        raise RuntimeError(f"too few Ausgrid users: {selected.size}")
    values = raw[:, selected]
    names = [all_names[i] for i in selected]
    labels, scales = cluster_profiles(values, dt, train0, train1, use_uci=False, segments=segments, cluster_method=cluster_method)
    return dt, values, names, labels, scales, train0, train1, windows, 30


def load_uci(zip_path: Path, segments: int, cluster_method: str):
    frame = base.read_dataset(zip_path)
    dt = frame.index
    raw = frame.to_numpy(np.float32, copy=False)
    train0 = int(dt.searchsorted(pd.Timestamp("2012-01-01")))
    train1 = int(dt.searchsorted(pd.Timestamp("2014-01-01")))
    windows = []
    for off in range(12):
        ws = pd.Timestamp("2014-01-01") + pd.DateOffset(months=off)
        windows.append((ws.strftime("%Y-%m"), ws, ws + pd.DateOffset(months=1)))
    eligible = np.mean(np.isfinite(raw[train0:train1]), axis=0) >= .99
    for _, ws, we in windows:
        a, b = int(dt.searchsorted(ws)), int(dt.searchsorted(we))
        eligible &= np.mean(np.isfinite(raw[a:b]), axis=0) >= .99
    train = raw[train0:train1]
    eligible &= np.nanmean(train > 1e-6, axis=0) >= .2
    eligible &= np.nanstd(train, axis=0) > 1e-9
    selected = np.flatnonzero(eligible)
    if selected.size < 100:
        raise RuntimeError(f"too few UCI users: {selected.size}")
    values = raw[:, selected]
    names = [str(frame.columns[i]) for i in selected]
    labels, scales = cluster_profiles(values, dt, train0, train1, use_uci=True, segments=segments, cluster_method=cluster_method)
    return dt, values, names, labels, scales, train0, train1, windows, 15


def residual_quantiles(values: np.ndarray, train0: int, train1: int, lag: int, target: float):
    current = values[max(train0, lag):train1]
    lagged = values[max(train0, lag) - lag:train1 - lag]
    residual = current - lagged
    alpha = (1.0 - target) / 2.0
    return np.nanquantile(residual, alpha, axis=0), np.nanquantile(residual, 1.0 - alpha, axis=0)


def interval_arrays(values: np.ndarray, start: int, stop: int, lag: int, low: np.ndarray, high: np.ndarray):
    start = max(start, lag)
    y2 = values[start:stop]
    lagged = values[start - lag:stop - lag]
    qlo2, qhi2 = lagged + low, lagged + high
    valid = np.isfinite(y2) & np.isfinite(qlo2) & np.isfinite(qhi2)
    times, users = np.nonzero(valid)
    return y2[valid].astype(float), qlo2[valid].astype(float), qhi2[valid].astype(float), users.astype(int)


def calibration_corrections(y, qlo, qhi, users, labels, scales, target):
    scores = np.maximum(qlo - y, y - qhi) / scales[users]
    global_q = conformal_quantile(scores, target)
    n_groups = int(np.max(labels)) + 1
    group_q = np.array([conformal_quantile(scores[labels[users] == group], target) for group in range(n_groups)])
    user_q = np.array([conformal_quantile(scores[users == user], target) for user in range(len(labels))])
    corrections = {
        "raw": np.zeros(len(labels)),
        "rolling_global_norm": np.full(len(labels), global_q),
        "rolling_group_norm": group_q[labels],
        "rolling_user_norm": user_q,
    }
    diagnostics = {
        "abs_global_correction": float(abs(global_q)),
        "user_correction_std": float(np.std(user_q)),
        "group_correction_range": float(np.ptp(group_q)),
        "mean_abs_user_global_disagreement": float(np.mean(np.abs(user_q - global_q))),
    }
    return corrections, diagnostics


def evaluate(y, qlo, qhi, users, labels, scales, correction, target):
    delta = correction[users] * scales[users]
    lo, hi = qlo - delta, qhi + delta
    crossed = hi < lo
    midpoint = .5 * (lo[crossed] + hi[crossed])
    lo[crossed], hi[crossed] = midpoint, midpoint
    covered = (y >= lo) & (y <= hi)
    width = hi - lo
    n_users = len(labels)
    user_n = np.bincount(users, minlength=n_users)
    user_covered = np.bincount(users, weights=covered.astype(float), minlength=n_users)
    if np.any(user_n == 0):
        raise RuntimeError("test window misses at least one frozen user")
    user_cov = user_covered / user_n
    groups = labels[users]
    n_groups = int(np.max(labels)) + 1
    group_n = np.bincount(groups, minlength=n_groups)
    group_covered = np.bincount(groups, weights=covered.astype(float), minlength=n_groups)
    group_cov = group_covered / group_n
    return {
        "n": int(len(y)),
        "picp": float(np.mean(covered)),
        "mpiw": float(np.mean(width)),
        "winkler_interval_score": float(np.mean(width + (2.0 / (1.0-target)) * (lo-y) * (y < lo) + (2.0 / (1.0-target)) * (y-hi) * (y > hi))),
        "macro_user_abs_coverage_gap": float(np.mean(np.abs(user_cov - target))),
        "user_coverage_std": float(np.std(user_cov)),
        "max_abs_cluster_coverage_gap": float(np.max(np.abs(group_cov - target))),
    }, user_cov, user_n


def run_configuration(dataset, values, dt, names, labels, scales, train0, train1, windows, cadence, coverage, horizon, max_windows):
    lag = int(round(horizon * 60 / cadence))
    low, high = residual_quantiles(values, train0, train1, lag, coverage)
    config = f"persistence_qi__c{int(coverage * 100)}__h{horizon:g}"
    selected_windows = windows if max_windows is None else windows[:max_windows]
    window_rows, user_rows, correction_rows = [], [], []
    for window, ts, te in selected_windows:
        log(f"{dataset} coverage={coverage:.2f} horizon={horizon:g}h window={window}")
        cal0 = int(dt.searchsorted(ts - pd.Timedelta(days=CALIBRATION_DAYS)))
        test0, test1 = int(dt.searchsorted(ts)), int(dt.searchsorted(te))
        y, qlo, qhi, users = interval_arrays(values, cal0, test0, lag, low, high)
        corrections, correction_diag = calibration_corrections(y, qlo, qhi, users, labels, scales, coverage)
        y, qlo, qhi, users = interval_arrays(values, test0, test1, lag, low, high)
        raw_metrics = evaluate(y, qlo, qhi, users, labels, scales, corrections["raw"], coverage)[0]
        environment_diag = {"raw_picp_abs_gap": float(abs(raw_metrics["picp"] - coverage)), **correction_diag}
        correction_rows.append({
            "dataset": DATASET_NAMES[dataset], "configuration": config, "window": window,
            "forecaster": "persistence_quantile_interval", "coverage": coverage, "horizon_hours": horizon,
            **environment_diag,
        })
        for method in METHODS:
            metrics, user_cov, user_n = evaluate(y, qlo, qhi, users, labels, scales, corrections[method], coverage)
            window_rows.append({
                "dataset": DATASET_NAMES[dataset], "configuration": config, "window": window,
                "forecaster": "persistence_quantile_interval", "coverage": coverage, "horizon_hours": horizon,
                "method": method, **metrics, **environment_diag,
            })
            for user in range(len(labels)):
                user_rows.append({
                    "dataset": DATASET_NAMES[dataset], "configuration": config, "window": window,
                    "forecaster": "persistence_quantile_interval", "coverage_target": coverage,
                    "horizon_hours": horizon, "method": method, "user_index": user,
                    "customer": names[user], "cluster": int(labels[user]), "coverage": float(user_cov[user]),
                    "n": int(user_n[user]), "coverage_gap": float(abs(user_cov[user] - coverage)),
                })
        gc.collect()
    summary, details = diag.compute_all(window_rows, user_rows, target=coverage)
    summary["configuration"] = {
        "id": config, "dataset": DATASET_NAMES[dataset], "forecaster": "persistence_quantile_interval",
        "coverage": coverage, "horizon_hours": horizon, "horizon_steps": lag,
        "windows": [row[0] for row in selected_windows],
    }
    for row in details["conflict_windows"]:
        row.update({"dataset": DATASET_NAMES[dataset], "configuration": config, "forecaster": "persistence_quantile_interval", "coverage": coverage, "horizon_hours": horizon})
    for row in details["rank_reversal_pairs"]:
        row.update({"dataset": DATASET_NAMES[dataset], "configuration": config, "forecaster": "persistence_quantile_interval", "coverage": coverage, "horizon_hours": horizon})
    for row in summary["temporal_cancellation"]:
        row.update({"dataset": DATASET_NAMES[dataset], "configuration": config, "forecaster": "persistence_quantile_interval", "coverage": coverage, "horizon_hours": horizon})
    for row in summary["routing_oracle_gap"]:
        row.update({"dataset": DATASET_NAMES[dataset], "configuration": config, "forecaster": "persistence_quantile_interval", "coverage": coverage, "horizon_hours": horizon})
    return window_rows, user_rows, correction_rows, details, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=tuple(DATASET_NAMES), required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--prepared", type=Path, default=LONDON)
    parser.add_argument("--raw-dir", type=Path, default=EXT / "raw_archive")
    parser.add_argument("--zip", type=Path, default=V1 / "electricityloaddiagrams20112014.originalmirror.zip")
    parser.add_argument("--coverages", nargs="+", type=float, default=[.8, .9])
    parser.add_argument("--horizons", nargs="+", type=float, default=[1., 6.])
    parser.add_argument("--max-windows", type=int)
    parser.add_argument("--segments", type=int, default=3)
    parser.add_argument("--cluster-method", choices=("kmeans", "ward"), default="kmeans")
    args = parser.parse_args()
    for target in args.coverages:
        if not 0 < target < 1:
            parser.error("coverage must be in (0, 1)")
    started = time.perf_counter()
    default_out = HERE / f"naive_{args.dataset}_full"
    out = args.out or default_out
    out.mkdir(parents=True, exist_ok=True)
    if args.dataset == "london":
        data = load_london(args.prepared, args.segments, args.cluster_method)
    elif args.dataset == "ausgrid":
        data = load_ausgrid(args.raw_dir, args.segments, args.cluster_method)
    else:
        data = load_uci(args.zip, args.segments, args.cluster_method)
    dt, values, names, labels, scales, train0, train1, windows, cadence = data
    log(f"Loaded {DATASET_NAMES[args.dataset]}: users={len(names)}, rows={len(values):,}, windows={len(windows)}")
    all_windows, all_users, all_corrections = [], [], []
    conflict_rows, reversal_rows, tci_rows, rog_rows, summaries = [], [], [], [], []
    for coverage in args.coverages:
        for horizon in args.horizons:
            window_rows, user_rows, correction_rows, details, summary = run_configuration(
                args.dataset, values, dt, names, labels, scales, train0, train1, windows, cadence,
                coverage, horizon, args.max_windows,
            )
            all_windows.extend(window_rows); all_users.extend(user_rows); all_corrections.extend(correction_rows)
            conflict_rows.extend(details["conflict_windows"]); reversal_rows.extend(details["rank_reversal_pairs"])
            tci_rows.extend(summary["temporal_cancellation"]); rog_rows.extend(summary["routing_oracle_gap"])
            summaries.append(summary)
    metadata = {
        "dataset": DATASET_NAMES[args.dataset], "users": len(names), "segments": args.segments, "cluster_method": args.cluster_method, "cluster_sizes": np.bincount(labels, minlength=args.segments).tolist(),
        "cadence_minutes": cadence, "train_start": str(dt[train0]), "train_stop_exclusive": str(dt[train1]),
        "configurations": len(summaries), "environment_observations": len(all_corrections),
        "wall_seconds": time.perf_counter() - started, "max_windows": args.max_windows,
    }
    write_csv(out / "window_metrics.csv", all_windows)
    write_csv(out / "per_user_window_metrics.csv", all_users)
    write_csv(out / "corrections.csv", all_corrections)
    write_csv(out / "conflict_windows.csv", conflict_rows)
    write_csv(out / "rank_reversal_pairs.csv", reversal_rows)
    write_csv(out / "temporal_cancellation.csv", tci_rows)
    write_csv(out / "routing_oracle_gap.csv", rog_rows)
    (out / "diagnostic_summary.json").write_text(json.dumps({"data": metadata, "configurations": summaries}, indent=2), encoding="utf-8")
    log(json.dumps(metadata))


if __name__ == "__main__":
    main()
