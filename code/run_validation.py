from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import platform
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except ImportError as exc:
    raise SystemExit(
        "LightGBM is unavailable. Add the project-local .deps directory to PYTHONPATH."
    ) from exc


SEED = 20260725
ALPHA = 0.20
TARGET_COVERAGE = 1.0 - ALPHA
HORIZON = 4  # 4 * 15 minutes = 1 hour
MAX_LAG = 672
FEATURE_NAMES = [
    "lag_0", "lag_1", "lag_4", "lag_96", "lag_192", "lag_672",
    "mean_4", "std_4", "mean_16", "std_16",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "weekend",
    "month_sin", "month_cos", "user_id", "cluster",
    "user_mean", "user_std", "user_q95", "user_zero_frac",
]


def log(msg: str) -> None:
    print(time.strftime("[%H:%M:%S]"), msg, flush=True)


def conformal_quantile(scores: np.ndarray, alpha: float = ALPHA) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    scores = scores[np.isfinite(scores)]
    if scores.size == 0:
        return float("nan")
    rank = min(scores.size - 1, math.ceil((scores.size + 1) * (1 - alpha)) - 1)
    return float(np.partition(scores, rank)[rank])


def pinball(y: np.ndarray, q: np.ndarray, tau: float) -> float:
    e = y - q
    return float(np.mean(np.maximum(tau * e, (tau - 1.0) * e)))


def kmeans(x: np.ndarray, k: int, seed: int, max_iter: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Small deterministic k-means++ implementation to avoid a sklearn dependency."""
    rng = np.random.default_rng(seed)
    n = len(x)
    centers = np.empty((k, x.shape[1]), dtype=np.float64)
    centers[0] = x[rng.integers(n)]
    closest = np.sum((x - centers[0]) ** 2, axis=1)
    for j in range(1, k):
        total = float(closest.sum())
        idx = rng.integers(n) if total <= 0 else rng.choice(n, p=closest / total)
        centers[j] = x[idx]
        closest = np.minimum(closest, np.sum((x - centers[j]) ** 2, axis=1))
    labels = np.zeros(n, dtype=np.int32)
    for _ in range(max_iter):
        dist = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = dist.argmin(axis=1).astype(np.int32)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for j in range(k):
            members = x[labels == j]
            if len(members):
                centers[j] = members.mean(axis=0)
            else:
                centers[j] = x[rng.integers(n)]
    return labels, centers


def read_dataset(zip_path: Path) -> pd.DataFrame:
    log(f"Checking archive CRC: {zip_path}")
    with zipfile.ZipFile(zip_path) as zf:
        bad = zf.testzip()
        if bad is not None:
            raise RuntimeError(f"ZIP CRC failure in {bad}")
        txts = [i.filename for i in zf.infolist() if i.filename.lower().endswith(".txt")]
        if len(txts) != 1:
            raise RuntimeError(f"Expected one text member, found: {txts}")
        log(f"Reading {txts[0]} from archive")
        with zf.open(txts[0]) as fh:
            df = pd.read_csv(fh, sep=";", decimal=",", index_col=0)
    df.index = pd.to_datetime(df.index, errors="raise")
    df = df.sort_index()
    df = df.apply(pd.to_numeric, errors="coerce").astype(np.float32)
    return df


def user_features(values: np.ndarray, dt: pd.DatetimeIndex) -> tuple[np.ndarray, list[str]]:
    eps = 1e-6
    mean = np.nanmean(values, axis=0)
    std = np.nanstd(values, axis=0)
    q95 = np.nanquantile(values, 0.95, axis=0)
    zero = np.nanmean(values <= eps, axis=0)
    load_factor = mean / np.maximum(q95, eps)
    cv = std / np.maximum(mean, eps)
    weekday = dt.dayofweek.to_numpy() < 5
    daytime = (dt.hour.to_numpy() >= 8) & (dt.hour.to_numpy() < 20)
    wd_mean = np.nanmean(values[weekday], axis=0)
    we_mean = np.nanmean(values[~weekday], axis=0)
    day_mean = np.nanmean(values[daytime], axis=0)
    night_mean = np.nanmean(values[~daytime], axis=0)
    weekday_ratio = wd_mean / np.maximum(we_mean, eps)
    day_ratio = day_mean / np.maximum(night_mean, eps)
    if len(values) > 96:
        x0, x1 = values[:-96], values[96:]
        m0, m1 = np.nanmean(x0, axis=0), np.nanmean(x1, axis=0)
        num = np.nansum((x0 - m0) * (x1 - m1), axis=0)
        den = np.sqrt(np.nansum((x0 - m0) ** 2, axis=0) * np.nansum((x1 - m1) ** 2, axis=0))
        ac96 = num / np.maximum(den, eps)
    else:
        ac96 = np.zeros(values.shape[1])
    feats = np.column_stack([
        np.log1p(mean), np.log1p(std), np.log1p(q95), zero,
        load_factor, np.clip(cv, 0, 20), np.clip(weekday_ratio, 0, 10),
        np.clip(day_ratio, 0, 10), np.clip(ac96, -1, 1),
    ])
    names = ["log_mean", "log_std", "log_q95", "zero_frac", "load_factor", "cv",
             "weekday_weekend_ratio", "day_night_ratio", "ac96"]
    return np.nan_to_num(feats, nan=0.0, posinf=20.0, neginf=-20.0), names


def build_rows(
    values: np.ndarray,
    dt: pd.DatetimeIndex,
    origins: np.ndarray,
    users: np.ndarray,
    labels: np.ndarray,
    stat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(origins)
    x = np.empty((n, len(FEATURE_NAMES)), dtype=np.float32)
    for col, lag in enumerate((0, 1, 4, 96, 192, 672)):
        x[:, col] = values[origins - lag, users]
    r4 = np.column_stack([values[origins - j, users] for j in range(4)])
    r16 = np.column_stack([values[origins - j, users] for j in range(16)])
    x[:, 6] = np.nanmean(r4, axis=1)
    x[:, 7] = np.nanstd(r4, axis=1)
    x[:, 8] = np.nanmean(r16, axis=1)
    x[:, 9] = np.nanstd(r16, axis=1)
    target_dt = dt[origins + HORIZON]
    minute_of_day = target_dt.hour.to_numpy() * 60 + target_dt.minute.to_numpy()
    dow = target_dt.dayofweek.to_numpy()
    month = target_dt.month.to_numpy()
    x[:, 10] = np.sin(2 * np.pi * minute_of_day / 1440)
    x[:, 11] = np.cos(2 * np.pi * minute_of_day / 1440)
    x[:, 12] = np.sin(2 * np.pi * dow / 7)
    x[:, 13] = np.cos(2 * np.pi * dow / 7)
    x[:, 14] = dow >= 5
    x[:, 15] = np.sin(2 * np.pi * (month - 1) / 12)
    x[:, 16] = np.cos(2 * np.pi * (month - 1) / 12)
    x[:, 17] = users
    x[:, 18] = labels[users]
    x[:, 19] = stat[users, 0]
    x[:, 20] = stat[users, 1]
    x[:, 21] = stat[users, 2]
    x[:, 22] = stat[users, 3]
    y = values[origins + HORIZON, users].astype(np.float32, copy=False)
    return x, y, users.astype(np.int32), labels[users].astype(np.int32), (origins + HORIZON).astype(np.int32)


def full_pairs(start: int, stop: int, n_users: int) -> tuple[np.ndarray, np.ndarray]:
    origins_1d = np.arange(max(start, MAX_LAG), stop - HORIZON, dtype=np.int32)
    origins = np.repeat(origins_1d, n_users)
    users = np.tile(np.arange(n_users, dtype=np.int32), len(origins_1d))
    return origins, users


def sampled_pairs(start: int, stop: int, n_users: int, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    lo, hi = max(start, MAX_LAG), stop - HORIZON
    total = max(0, hi - lo) * n_users
    n = min(n, total)
    flat = rng.choice(total, size=n, replace=False)
    return (lo + flat // n_users).astype(np.int32), (flat % n_users).astype(np.int32)


def predict_three(models: dict[float, lgb.Booster], x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    t0 = time.perf_counter()
    raw = np.column_stack([models[t].predict(x) for t in (0.1, 0.5, 0.9)])
    elapsed = time.perf_counter() - t0
    ordered = np.sort(raw, axis=1)
    return ordered[:, 0], ordered[:, 1], ordered[:, 2], elapsed


def metric_rows(
    y: np.ndarray,
    q10: np.ndarray,
    q50: np.ndarray,
    q90: np.ndarray,
    users: np.ndarray,
    groups: np.ndarray,
    thresholds: np.ndarray,
    corrections: dict[str, np.ndarray | float],
) -> tuple[list[dict], list[dict], dict[str, np.ndarray]]:
    rows, cluster_rows, intervals = [], [], {}
    base = {
        "mae": float(np.mean(np.abs(y - q50))),
        "rmse": float(np.sqrt(np.mean((y - q50) ** 2))),
        "pinball_mean": float(np.mean([
            pinball(y, q10, 0.1), pinball(y, q50, 0.5), pinball(y, q90, 0.9)
        ])),
    }
    for method in ("raw", "global_cqr", "group_cqr"):
        corr = corrections[method]
        if np.isscalar(corr):
            delta = float(corr)
        else:
            delta = np.asarray(corr)[groups]
        lo, hi = q10 - delta, q90 + delta
        intervals[method] = hi
        covered = (y >= lo) & (y <= hi)
        width = hi - lo
        per_user_cov = np.array([covered[users == u].mean() for u in np.unique(users)])
        cluster_cov = np.array([covered[groups == g].mean() for g in np.unique(groups)])
        peak = y > thresholds[users]
        alarm = hi > thresholds[users]
        tp = int(np.sum(peak & alarm)); fn = int(np.sum(peak & ~alarm))
        fp = int(np.sum(~peak & alarm)); tn = int(np.sum(~peak & ~alarm))
        row = dict(method=method, **base)
        row.update({
            "picp": float(covered.mean()),
            "mpiw": float(width.mean()),
            "nmpiw": float(width.mean() / max(float(np.mean(np.abs(y))), 1e-9)),
            "macro_user_coverage": float(per_user_cov.mean()),
            "macro_user_abs_coverage_gap": float(np.mean(np.abs(per_user_cov - TARGET_COVERAGE))),
            "user_coverage_std": float(np.std(per_user_cov)),
            "max_abs_cluster_coverage_gap": float(np.max(np.abs(cluster_cov - TARGET_COVERAGE))),
            "peak_recall": tp / max(tp + fn, 1),
            "peak_fpr": fp / max(fp + tn, 1),
            "peak_precision": tp / max(tp + fp, 1),
            "true_exceedances": tp + fn,
            "missed_exceedances": fn,
            "false_alarms": fp,
            "decision_cost_fn5_fp1": 5 * fn + fp,
        })
        rows.append(row)
        for g in np.unique(groups):
            mask = groups == g
            cluster_rows.append({
                "method": method, "cluster": int(g), "n": int(mask.sum()),
                "customers": int(len(np.unique(users[mask]))),
                "picp": float(covered[mask].mean()),
                "mpiw": float(width[mask].mean()),
                "coverage_gap": float(covered[mask].mean() - TARGET_COVERAGE),
            })
    return rows, cluster_rows, intervals


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train-rows", type=int, default=600_000)
    parser.add_argument("--num-boost-round", type=int, default=250)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--max-users", type=int, default=0,
                        help="Optional deterministic cap for smoke tests; 0 keeps all eligible users.")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    df = read_dataset(args.zip)
    dt = df.index
    if len(dt) < 1000 or df.shape[1] < 10:
        raise RuntimeError(f"Unexpected dataset shape {df.shape}")
    cadence = pd.Series(dt[1:] - dt[:-1]).mode().iloc[0]
    if cadence != pd.Timedelta(minutes=15):
        raise RuntimeError(f"Unexpected modal cadence: {cadence}")

    train_start_ts = max(pd.Timestamp("2013-01-01"), dt.min() + pd.Timedelta(days=8))
    cal_start_ts = pd.Timestamp("2014-07-01")
    test_start_ts = pd.Timestamp("2014-10-01")
    test_end_ts = dt.max() + pd.Timedelta(minutes=15)
    train_start = int(dt.searchsorted(train_start_ts))
    cal_start = int(dt.searchsorted(cal_start_ts))
    test_start = int(dt.searchsorted(test_start_ts))
    test_end = int(dt.searchsorted(test_end_ts))
    if not (train_start < cal_start < test_start < test_end):
        raise RuntimeError("Dataset does not cover the frozen split")

    raw = df.to_numpy(dtype=np.float32, copy=False)
    finite = np.isfinite(raw)
    nonzero = raw > 1e-6
    split_slices = [slice(train_start, cal_start), slice(cal_start, test_start), slice(test_start, test_end)]
    eligible = np.ones(raw.shape[1], dtype=bool)
    for sl in split_slices:
        eligible &= finite[sl].mean(axis=0) >= 0.99
        eligible &= nonzero[sl].mean(axis=0) >= 0.20
    eligible &= np.nanstd(raw[train_start:cal_start], axis=0) > 1e-6
    selected = np.flatnonzero(eligible)
    if args.max_users and len(selected) > args.max_users:
        selected = selected[: args.max_users]
    if len(selected) < max(12, args.k * 3):
        raise RuntimeError(f"Only {len(selected)} eligible customers")
    customer_names = [str(df.columns[i]) for i in selected]
    values = raw[:, selected]
    del raw, finite, nonzero, df
    gc.collect()
    log(f"Retained {values.shape[1]} customers and {values.shape[0]} timestamps")

    train_values = values[train_start:cal_start]
    uf, uf_names = user_features(train_values, dt[train_start:cal_start])
    center, scale = np.median(uf, axis=0), np.std(uf, axis=0)
    z = (uf - center) / np.where(scale > 1e-9, scale, 1.0)
    labels, centers = kmeans(z, args.k, SEED)
    stat = np.column_stack([
        np.nanmean(train_values, axis=0), np.nanstd(train_values, axis=0),
        np.nanquantile(train_values, 0.95, axis=0), np.nanmean(train_values <= 1e-6, axis=0),
    ]).astype(np.float32)
    thresholds = stat[:, 2].copy()

    cluster_sizes = np.bincount(labels, minlength=args.k)
    log(f"Cluster sizes: {cluster_sizes.tolist()}")
    user_rows = []
    for u, name in enumerate(customer_names):
        row = {"user_index": u, "customer": name, "cluster": int(labels[u])}
        row.update({n: float(v) for n, v in zip(uf_names, uf[u])})
        row["peak_threshold_q95"] = float(thresholds[u])
        user_rows.append(row)
    write_csv(args.out / "customer_clusters.csv", user_rows)

    origins, users = sampled_pairs(train_start, cal_start, len(selected), args.train_rows, SEED)
    x_train, y_train, _, _, _ = build_rows(values, dt, origins, users, labels, stat)
    valid = np.isfinite(x_train).all(axis=1) & np.isfinite(y_train)
    x_train, y_train = x_train[valid], y_train[valid]
    log(f"Training rows: {len(y_train):,}; features: {x_train.shape[1]}")

    params_base = {
        "verbosity": -1, "learning_rate": 0.05, "num_leaves": 31,
        "min_data_in_leaf": 100, "feature_fraction": 0.9,
        "bagging_fraction": 0.9, "bagging_freq": 1,
        "lambda_l2": 1.0, "seed": SEED, "num_threads": args.threads,
        "force_col_wise": True,
    }
    models: dict[float, lgb.Booster] = {}
    fit_seconds = 0.0
    dataset = lgb.Dataset(
        x_train, label=y_train, feature_name=FEATURE_NAMES,
        categorical_feature=[17, 18], free_raw_data=False,
    )
    for tau in (0.1, 0.5, 0.9):
        log(f"Fitting LightGBM quantile tau={tau}")
        t0 = time.perf_counter()
        params = dict(params_base, objective="quantile", alpha=tau, metric="quantile")
        models[tau] = lgb.train(params, dataset, num_boost_round=args.num_boost_round)
        fit_seconds += time.perf_counter() - t0
    model_bytes = sum(len(m.model_to_string().encode("utf-8")) for m in models.values())
    del dataset, x_train, y_train
    gc.collect()

    cal_origins, cal_users = full_pairs(cal_start, test_start, len(selected))
    x_cal, y_cal, cal_users, cal_groups, _ = build_rows(
        values, dt, cal_origins, cal_users, labels, stat
    )
    valid = np.isfinite(x_cal).all(axis=1) & np.isfinite(y_cal)
    x_cal, y_cal, cal_users, cal_groups = x_cal[valid], y_cal[valid], cal_users[valid], cal_groups[valid]
    log(f"Calibration rows: {len(y_cal):,}")
    c10, c50, c90, cal_pred_seconds = predict_three(models, x_cal)
    scores = np.maximum(c10 - y_cal, y_cal - c90)
    global_q = conformal_quantile(scores)
    group_q = np.array([conformal_quantile(scores[cal_groups == g]) for g in range(args.k)])
    cal_counts = np.bincount(cal_groups, minlength=args.k)
    del x_cal, y_cal, c10, c50, c90, scores, cal_origins, cal_users, cal_groups
    gc.collect()
    log(f"Global correction={global_q:.6g}; group corrections={group_q.tolist()}")

    test_origins, test_users = full_pairs(test_start, test_end, len(selected))
    x_test, y_test, test_users, test_groups, target_rows = build_rows(
        values, dt, test_origins, test_users, labels, stat
    )
    valid = np.isfinite(x_test).all(axis=1) & np.isfinite(y_test)
    x_test, y_test = x_test[valid], y_test[valid]
    test_users, test_groups, target_rows = test_users[valid], test_groups[valid], target_rows[valid]
    log(f"Test rows: {len(y_test):,}")
    q10, q50, q90, test_pred_seconds = predict_three(models, x_test)
    del x_test, values
    gc.collect()

    corrections = {"raw": 0.0, "global_cqr": global_q, "group_cqr": group_q}
    rows, cluster_rows, uppers = metric_rows(
        y_test, q10, q50, q90, test_users, test_groups, thresholds, corrections
    )
    write_csv(args.out / "method_metrics.csv", rows)
    write_csv(args.out / "cluster_metrics.csv", cluster_rows)
    correction_rows = [{
        "cluster": g, "customers": int(cluster_sizes[g]),
        "calibration_rows": int(cal_counts[g]), "correction": float(group_q[g]),
        "global_correction": float(global_q),
    } for g in range(args.k)]
    write_csv(args.out / "corrections.csv", correction_rows)

    per_user_rows = []
    for u, name in enumerate(customer_names):
        mask = test_users == u
        peak = y_test[mask] > thresholds[u]
        row = {
            "user_index": u, "customer": name, "cluster": int(labels[u]),
            "n_test": int(mask.sum()), "threshold": float(thresholds[u]),
            "true_exceedances": int(peak.sum()),
        }
        for method, hi in uppers.items():
            delta = 0.0 if method == "raw" else (global_q if method == "global_cqr" else group_q[labels[u]])
            lo = q10[mask] - delta
            cov = (y_test[mask] >= lo) & (y_test[mask] <= hi[mask])
            alarm = hi[mask] > thresholds[u]
            tp = int(np.sum(peak & alarm)); fn = int(np.sum(peak & ~alarm)); fp = int(np.sum(~peak & alarm))
            row[f"{method}_coverage"] = float(cov.mean())
            row[f"{method}_mpiw"] = float(np.mean(hi[mask] - lo))
            row[f"{method}_peak_recall"] = tp / max(int(peak.sum()), 1)
            row[f"{method}_false_alarms"] = fp
            row[f"{method}_missed"] = fn
            row[f"{method}_cost"] = 5 * fn + fp
        per_user_rows.append(row)
    write_csv(args.out / "per_user_metrics.csv", per_user_rows)

    by_method = {r["method"]: r for r in rows}
    glob, group = by_method["global_cqr"], by_method["group_cqr"]
    reduction_cluster = 1 - group["max_abs_cluster_coverage_gap"] / max(glob["max_abs_cluster_coverage_gap"], 1e-12)
    reduction_user = 1 - group["macro_user_abs_coverage_gap"] / max(glob["macro_user_abs_coverage_gap"], 1e-12)
    width_change = group["mpiw"] / max(glob["mpiw"], 1e-12) - 1
    cost_change = group["decision_cost_fn5_fp1"] / max(glob["decision_cost_fn5_fp1"], 1) - 1
    checks = {
        "cluster_gap_reduction_ge_20pct": bool(reduction_cluster >= 0.20),
        "user_gap_reduction_ge_10pct": bool(reduction_user >= 0.10),
        "mpiw_increase_le_15pct": bool(width_change <= 0.15),
        "decision_cost_increase_le_5pct": bool(cost_change <= 0.05),
        "at_least_3_meaningful_clusters": bool(np.sum((cluster_sizes >= 3) & (cal_counts >= 1000)) >= 3),
    }
    passed = sum(checks.values())
    if passed == len(checks):
        verdict = "GO"
    elif checks["cluster_gap_reduction_ge_20pct"] and passed >= 3:
        verdict = "CONDITIONAL_GO"
    else:
        verdict = "NO_GO"
    decision = {
        "verdict": verdict, "checks": checks,
        "effect_sizes": {
            "cluster_gap_reduction": reduction_cluster,
            "user_gap_reduction": reduction_user,
            "mpiw_relative_change": width_change,
            "decision_cost_relative_change": cost_change,
        },
        "data": {
            "source_zip": str(args.zip.resolve()), "shape_original": [len(dt), len(customer_names)],
            "retained_customers": len(customer_names), "cadence": str(cadence),
            "train_start": str(dt[train_start]), "train_end_exclusive": str(dt[cal_start]),
            "calibration_start": str(dt[cal_start]), "calibration_end_exclusive": str(dt[test_start]),
            "test_start": str(dt[test_start]), "test_end": str(dt[test_end - 1]),
            "train_rows_sampled": int(len(origins)), "calibration_rows": int(cal_counts.sum()),
            "test_rows": int(len(y_test)), "cluster_sizes": cluster_sizes.tolist(),
        },
        "efficiency": {
            "fit_seconds_total_3_models": fit_seconds,
            "calibration_prediction_seconds": cal_pred_seconds,
            "test_prediction_seconds": test_pred_seconds,
            "model_bytes_total": model_bytes, "threads": args.threads,
            "python": sys.version, "platform": platform.platform(),
            "lightgbm": lgb.__version__, "wall_seconds": time.perf_counter() - started,
        },
        "limitations": [
            "Single chronological split and one-hour horizon.",
            "Temporal dependence means classical exchangeability coverage is not claimed.",
            "No weather or tariff covariates are available in this dataset.",
            "Go/no-go result validates an empirical lead, not publication readiness.",
        ],
    }
    (args.out / "decision.json").write_text(json.dumps(decision, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Group-conditional CQR validation result", "", f"**Verdict: {verdict}**", "",
        "## Frozen checks", "",
    ]
    for key, ok in checks.items():
        lines.append(f"- {'PASS' if ok else 'FAIL'} — `{key}`")
    lines.extend(["", "## Effect sizes", ""])
    for key, val in decision["effect_sizes"].items():
        lines.append(f"- `{key}`: {val:.3%}")
    lines.extend(["", "## Test metrics", ""])
    for row in rows:
        lines.append(
            f"- **{row['method']}**: PICP={row['picp']:.4f}, MPIW={row['mpiw']:.4g}, "
            f"max cluster gap={row['max_abs_cluster_coverage_gap']:.4f}, "
            f"peak recall={row['peak_recall']:.4f}, FPR={row['peak_fpr']:.4f}, "
            f"cost={row['decision_cost_fn5_fp1']}"
        )
    lines.extend([
        "", "## Interpretation", "",
        "The verdict follows the criteria frozen before the test labels were evaluated. "
        "See `method_metrics.csv`, `cluster_metrics.csv`, `per_user_metrics.csv`, and "
        "`decision.json` for auditable results.", "",
    ])
    (args.out / "VALIDATION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    log(f"Finished with verdict {verdict}; outputs in {args.out}")


if __name__ == "__main__":
    main()
