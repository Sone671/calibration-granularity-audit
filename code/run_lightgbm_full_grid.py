"""Frozen full 80/90% x 1/6h static-CQR grid for a LightGBM quantile forecaster."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
EXT = ROOT / "probabilistic_load_hierarchical_rolling_cqr_external_2026-07-30"
V1 = ROOT / "probabilistic_load_group_cqr_go_nogo_2026-07-25"
for path in (HERE, EXT, V1 / ".deps", V1):
    sys.path.insert(0, str(path))

import diagnostic_metrics as diag  # noqa: E402
import run_external as ext  # noqa: E402
import run_validation as base  # noqa: E402
import run_naive_robustness as naive  # noqa: E402
METHODS = ("raw", "rolling_global_norm", "rolling_group_norm", "rolling_user_norm")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


SEED = 20260725
CALIBRATION_DAYS = 56
DATASETS = ("london", "ausgrid", "uci")


def log(message: str) -> None:
    print(time.strftime("[%H:%M:%S]"), message, flush=True)


def horizon_steps(cadence_minutes: int, horizon_hours: float) -> int:
    return int(round(horizon_hours * 60 / cadence_minutes))


def external_rows(
    values: np.ndarray,
    dt: pd.DatetimeIndex,
    prefix,
    origins: np.ndarray,
    users: np.ndarray,
    labels: np.ndarray,
    stat: np.ndarray,
    step: int,
):
    x = np.empty((len(origins), len(ext.FEATURE_NAMES)), dtype=np.float32)
    for column, lag in enumerate((1, 2, 3, 48, 96, 336)):
        x[:, column] = values[origins - lag, users]
    x[:, 6], x[:, 7] = ext.rolling_moments(prefix, origins, users, 48)
    x[:, 8], x[:, 9] = ext.rolling_moments(prefix, origins, users, 336)
    target_dt = dt[origins + step]
    minute = target_dt.hour.to_numpy() * 60 + target_dt.minute.to_numpy()
    dow, month = target_dt.dayofweek.to_numpy(), target_dt.month.to_numpy()
    x[:, 10], x[:, 11] = np.sin(2 * np.pi * minute / 1440), np.cos(2 * np.pi * minute / 1440)
    x[:, 12], x[:, 13] = np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7)
    x[:, 14] = dow >= 5
    x[:, 15], x[:, 16] = np.sin(2 * np.pi * (month - 1) / 12), np.cos(2 * np.pi * (month - 1) / 12)
    x[:, 17], x[:, 18] = users, labels[users]
    x[:, 19:23] = stat[users]
    y = values[origins + step, users].astype(np.float32, copy=False)
    valid = np.isfinite(x).all(axis=1) & np.isfinite(y)
    return x[valid], y[valid], users[valid].astype(np.int32), labels[users[valid]].astype(np.int32), origins[valid].astype(np.int32)


def uci_rows(
    values: np.ndarray,
    dt: pd.DatetimeIndex,
    origins: np.ndarray,
    users: np.ndarray,
    labels: np.ndarray,
    stat: np.ndarray,
    step: int,
):
    x = np.empty((len(origins), len(base.FEATURE_NAMES)), dtype=np.float32)
    for column, lag in enumerate((0, 1, 4, 96, 192, 672)):
        x[:, column] = values[origins - lag, users]
    r4 = np.column_stack([values[origins - offset, users] for offset in range(4)])
    r16 = np.column_stack([values[origins - offset, users] for offset in range(16)])
    x[:, 6], x[:, 7] = np.nanmean(r4, axis=1), np.nanstd(r4, axis=1)
    x[:, 8], x[:, 9] = np.nanmean(r16, axis=1), np.nanstd(r16, axis=1)
    target_dt = dt[origins + step]
    minute = target_dt.hour.to_numpy() * 60 + target_dt.minute.to_numpy()
    dow, month = target_dt.dayofweek.to_numpy(), target_dt.month.to_numpy()
    x[:, 10], x[:, 11] = np.sin(2 * np.pi * minute / 1440), np.cos(2 * np.pi * minute / 1440)
    x[:, 12], x[:, 13] = np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7)
    x[:, 14] = dow >= 5
    x[:, 15], x[:, 16] = np.sin(2 * np.pi * (month - 1) / 12), np.cos(2 * np.pi * (month - 1) / 12)
    x[:, 17], x[:, 18] = users, labels[users]
    x[:, 19:23] = stat[users]
    y = values[origins + step, users].astype(np.float32, copy=False)
    valid = np.isfinite(x).all(axis=1) & np.isfinite(y)
    return x[valid], y[valid], users[valid].astype(np.int32), labels[users[valid]].astype(np.int32), origins[valid].astype(np.int32)


def pairs(start: int, stop: int, users: int, step: int, max_lag: int, include_target_at_start: bool):
    # The external 30-minute benchmark indexes its data by prediction origin
    # and includes origin=start-step so that the first target lies exactly at
    # the calibration/test boundary. The UCI frozen benchmark instead starts
    # at its origin boundary. Preserve both definitions exactly.
    first = max(start - step, max_lag) if include_target_at_start else max(start, max_lag)
    origins = np.arange(first, stop - step, dtype=np.int32)
    return np.repeat(origins, users), np.tile(np.arange(users, dtype=np.int32), len(origins))


def sampled_pairs(start: int, stop: int, users: int, step: int, max_lag: int, rows: int):
    low, high = max(start, max_lag), stop - step
    total = max(0, high - low) * users
    if total <= 0:
        raise RuntimeError("empty training range")
    rng = np.random.default_rng(SEED)
    flat = rng.choice(total, size=min(rows, total), replace=False)
    return (low + flat // users).astype(np.int32), (flat % users).astype(np.int32)


def load_data(name: str, args):
    common = dict(segments=args.segments, cluster_method=args.cluster_method)
    if name == "london":
        loaded = naive.load_london(args.prepared, **common)
    elif name == "ausgrid":
        loaded = naive.load_ausgrid(args.raw_dir, **common)
    else:
        loaded = naive.load_uci(args.zip, **common)
    dt, values, names, labels, scales, train0, train1, windows, cadence = loaded
    train = values[train0:train1]
    stat = np.column_stack([
        np.nanmean(train, axis=0), np.nanstd(train, axis=0),
        np.nanquantile(train, .95, axis=0), np.nanmean(train <= 1e-6, axis=0),
    ]).astype(np.float32)
    return {
        "name": name, "dt": dt, "values": values, "names": names, "labels": labels,
        "scales": scales, "train0": train0, "train1": train1, "windows": windows,
        "cadence": cadence, "stat": stat, "prefix": None if name == "uci" else ext.make_prefix(values),
    }


def build_rows(data, origins, users, step):
    if data["name"] == "uci":
        return uci_rows(data["values"], data["dt"], origins, users, data["labels"], data["stat"], step)
    return external_rows(data["values"], data["dt"], data["prefix"], origins, users, data["labels"], data["stat"], step)


def quantile_pair(coverage: float) -> tuple[float, float]:
    alpha = round((1.0 - coverage) / 2.0, 10)
    return alpha, round(1.0 - alpha, 10)


def train_models(data, step: int, args):
    max_lag = base.MAX_LAG if data["name"] == "uci" else ext.MAX_LAG
    origins, users = sampled_pairs(data["train0"], data["train1"], len(data["names"]), step, max_lag, args.train_rows)
    x, y, _, _, _ = build_rows(data, origins, users, step)
    log(f"{naive.DATASET_NAMES[data['name']]} h={args.current_horizon:g} training rows={len(y):,}")
    feature_names = base.FEATURE_NAMES if data["name"] == "uci" else ext.FEATURE_NAMES
    parameters = {
        "verbosity": -1, "learning_rate": .05, "num_leaves": 31, "min_data_in_leaf": 100,
        "feature_fraction": .9, "bagging_fraction": .9, "bagging_freq": 1, "lambda_l2": 1.,
        "seed": SEED, "num_threads": args.threads, "force_col_wise": True,
    }
    dataset = base.lgb.Dataset(x, label=y, feature_name=feature_names, categorical_feature=[17, 18], free_raw_data=False)
    requested = {0.5}
    for coverage in args.coverages:
        requested.update(quantile_pair(coverage))
    models = {}
    for quantile in sorted(requested):
        log(f"{naive.DATASET_NAMES[data['name']]} h={args.current_horizon:g} quantile={quantile}")
        models[quantile] = base.lgb.train(dict(parameters, objective="quantile", alpha=quantile, metric="quantile"), dataset, num_boost_round=args.num_boost_round)
    del dataset, x, y
    gc.collect()
    return models


def predict(data, models, start: int, stop: int, step: int, coverage: float):
    max_lag = base.MAX_LAG if data["name"] == "uci" else ext.MAX_LAG
    origins, users = pairs(
        start, stop, len(data["names"]), step, max_lag,
        include_target_at_start=data["name"] != "uci",
    )
    x, y, users, groups, valid_origins = build_rows(data, origins, users, step)
    # Quantile models can cross, so the lower/median/upper predictions are
    # ordered observation-wise before CQR scoring. The lower and upper models
    # are matched to the requested target coverage.
    lower, upper = quantile_pair(coverage)
    raw = np.column_stack([models[q].predict(x) for q in (lower, 0.5, upper)])
    ordered = np.sort(raw, axis=1)
    qlo, qhi = ordered[:, 0], ordered[:, 2]
    target_times = data["dt"][valid_origins + step].to_numpy()
    return y, qlo, qhi, users, groups, target_times


def run_configuration(data, models, coverage: float, horizon: float):
    step = horizon_steps(data["cadence"], horizon)
    config = f"lightgbm_quantile__c{int(coverage * 100)}__h{horizon:g}"
    windows, user_rows, correction_rows = [], [], []
    conflict_rows, reversal_rows = [], []
    for window, start, stop in data["windows"]:
        log(f"{naive.DATASET_NAMES[data['name']]} c={coverage:.2f} h={horizon:g} window={window}")
        cal0, test0, test1 = (int(data["dt"].searchsorted(value)) for value in (start - pd.Timedelta(days=CALIBRATION_DAYS), start, stop))
        y, qlo, qhi, users, _, _ = predict(data, models, cal0, test0, step, coverage)
        corrections, correction_diag = naive.calibration_corrections(y, qlo, qhi, users, data["labels"], data["scales"], coverage)
        del y, qlo, qhi, users
        y, qlo, qhi, users, _, _ = predict(data, models, test0, test1, step, coverage)
        raw_metrics = naive.evaluate(y, qlo, qhi, users, data["labels"], data["scales"], corrections["raw"], coverage)[0]
        environment_diag = {"raw_picp_abs_gap": float(abs(raw_metrics["picp"] - coverage)), **correction_diag}
        correction_rows.append({"dataset": naive.DATASET_NAMES[data["name"]], "configuration": config, "window": window, "forecaster": "lightgbm_quantile", "coverage": coverage, "horizon_hours": horizon, **environment_diag})
        for method in METHODS:
            metrics, user_cov, user_n = naive.evaluate(y, qlo, qhi, users, data["labels"], data["scales"], corrections[method], coverage)
            windows.append({"dataset": naive.DATASET_NAMES[data["name"]], "configuration": config, "window": window, "forecaster": "lightgbm_quantile", "coverage": coverage, "horizon_hours": horizon, "method": method, **metrics, **environment_diag})
            for user in range(len(data["labels"])):
                user_rows.append({"dataset": naive.DATASET_NAMES[data["name"]], "configuration": config, "window": window, "forecaster": "lightgbm_quantile", "coverage_target": coverage, "horizon_hours": horizon, "method": method, "user_index": user, "customer": data["names"][user], "cluster": int(data["labels"][user]), "coverage": float(user_cov[user]), "n": int(user_n[user]), "coverage_gap": float(abs(user_cov[user] - coverage))})
        del y, qlo, qhi, users
        gc.collect()
    summary, details = diag.compute_all(windows, user_rows, target=coverage)
    for row in details["conflict_windows"]:
        row.update({"dataset": naive.DATASET_NAMES[data["name"]], "configuration": config, "forecaster": "lightgbm_quantile", "coverage": coverage, "horizon_hours": horizon})
    for row in details["rank_reversal_pairs"]:
        row.update({"dataset": naive.DATASET_NAMES[data["name"]], "configuration": config, "forecaster": "lightgbm_quantile", "coverage": coverage, "horizon_hours": horizon})
    for row in summary["temporal_cancellation"]:
        row.update({"dataset": naive.DATASET_NAMES[data["name"]], "configuration": config, "forecaster": "lightgbm_quantile", "coverage": coverage, "horizon_hours": horizon})
    for row in summary["routing_oracle_gap"]:
        row.update({"dataset": naive.DATASET_NAMES[data["name"]], "configuration": config, "forecaster": "lightgbm_quantile", "coverage": coverage, "horizon_hours": horizon})
    return windows, user_rows, correction_rows, details, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--out", type=Path, default=HERE / "lightgbm_full_grid")
    parser.add_argument("--prepared", type=Path, default=naive.LONDON)
    parser.add_argument("--raw-dir", type=Path, default=EXT / "raw_archive")
    parser.add_argument("--zip", type=Path, default=V1 / "electricityloaddiagrams20112014.originalmirror.zip")
    parser.add_argument("--coverages", nargs="+", type=float, default=[.8, .9])
    parser.add_argument("--horizons", nargs="+", type=float, default=[1., 6.])
    parser.add_argument("--train-rows", type=int, default=600000)
    parser.add_argument("--num-boost-round", type=int, default=250)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--segments", type=int, default=3)
    parser.add_argument("--cluster-method", choices=("kmeans", "ward"), default="kmeans")
    args = parser.parse_args()
    if any(not 0 < value < 1 for value in args.coverages):
        parser.error("coverage must be in (0, 1)")
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    all_windows, all_users, all_corrections = [], [], []
    all_conflicts, all_reversals, all_tci, all_rog, summaries = [], [], [], [], []
    data_metadata = []
    for name in args.datasets:
        data = load_data(name, args)
        data_metadata.append({"dataset": naive.DATASET_NAMES[name], "users": len(data["names"]), "windows": [item[0] for item in data["windows"]], "cadence_minutes": data["cadence"], "cluster_sizes": np.bincount(data["labels"], minlength=args.segments).tolist()})
        log(f"Loaded {naive.DATASET_NAMES[name]} users={len(data['names'])} windows={len(data['windows'])}")
        for horizon in args.horizons:
            args.current_horizon = horizon
            models = train_models(data, horizon_steps(data["cadence"], horizon), args)
            for coverage in args.coverages:
                result = run_configuration(data, models, coverage, horizon)
                windows, users, corrections, details, summary = result
                all_windows.extend(windows); all_users.extend(users); all_corrections.extend(corrections)
                all_conflicts.extend(details["conflict_windows"]); all_reversals.extend(details["rank_reversal_pairs"])
                all_tci.extend(summary["temporal_cancellation"]); all_rog.extend(summary["routing_oracle_gap"]); summaries.append(summary)
            del models
            gc.collect()
        del data
        gc.collect()
    metadata = {
        "protocol": "FROZEN_LIGHTGBM_FULL_GRID_PROTOCOL.md", "forecaster": "LightGBM quantile",
        "datasets": data_metadata, "coverages": args.coverages, "horizons_hours": args.horizons,
        "train_rows": args.train_rows, "num_boost_round": args.num_boost_round, "threads": args.threads,
        "segments": args.segments, "cluster_method": args.cluster_method,
        "environment_observations": len(all_corrections), "window_metric_rows": len(all_windows),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "wall_seconds": time.perf_counter() - started,
    }
    write_csv(args.out / "window_metrics.csv", all_windows)
    write_csv(args.out / "per_user_window_metrics.csv", all_users)
    write_csv(args.out / "corrections.csv", all_corrections)
    write_csv(args.out / "conflict_windows.csv", all_conflicts)
    write_csv(args.out / "rank_reversal_pairs.csv", all_reversals)
    write_csv(args.out / "temporal_cancellation.csv", all_tci)
    write_csv(args.out / "routing_oracle_gap.csv", all_rog)
    (args.out / "diagnostic_summary.json").write_text(json.dumps({"data": metadata, "configurations": summaries}, indent=2), encoding="utf-8")
    log(json.dumps(metadata))


if __name__ == "__main__":
    main()
