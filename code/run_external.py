"""Frozen external confirmation for hierarchical rolling CQR on Ausgrid GC load."""

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
V1 = ROOT / "probabilistic_load_group_cqr_go_nogo_2026-07-25"
sys.path.insert(0, str(V1 / ".deps"))
sys.path.insert(0, str(V1))
import run_validation as base  # noqa: E402

SEED = 20260725
TARGET = 0.80
HORIZON = 2
MAX_LAG = 336
METHODS = ("raw", "rolling_global_norm", "rolling_group_norm", "rolling_user_norm", "hierarchical_rolling_cqr")
WINDOWS = (
    ("2012-10", pd.Timestamp("2012-10-01"), pd.Timestamp("2012-11-01")),
    ("2012-11", pd.Timestamp("2012-11-01"), pd.Timestamp("2012-12-01")),
    ("2012-12", pd.Timestamp("2012-12-01"), pd.Timestamp("2013-01-01")),
)
MARGIN_GRID = (-0.50, -0.25, 0.0, 0.25, 0.50)
FEATURE_NAMES = [
    "lag_1", "lag_2", "lag_3", "lag_48", "lag_96", "lag_336",
    "mean_48", "std_48", "mean_336", "std_336",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "weekend", "month_sin", "month_cos",
    "user_id", "cluster", "user_mean", "user_std", "user_q95", "user_zero_frac",
]


def log(message: str) -> None:
    print(time.strftime("[%H:%M:%S]"), message, flush=True)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def interpolate_short_gaps(values: np.ndarray, max_gap: int = 2) -> tuple[np.ndarray, int]:
    output = values.copy()
    filled = 0
    for j in range(output.shape[1]):
        x = output[:, j]
        missing = ~np.isfinite(x)
        if not missing.any():
            continue
        changes = np.diff(np.r_[False, missing, False].astype(np.int8))
        starts, stops = np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)
        for start, stop in zip(starts, stops):
            if stop - start <= max_gap and start > 0 and stop < len(x) and np.isfinite(x[start - 1]) and np.isfinite(x[stop]):
                x[start:stop] = np.linspace(x[start - 1], x[stop], stop - start + 2)[1:-1]
                filled += stop - start
    return output, int(filled)


def read_ausgrid(raw_dir: Path) -> tuple[pd.DatetimeIndex, np.ndarray, list[str], dict]:
    files = [raw_dir / f"Solar home {year}.csv" for year in ("2010-2011", "2011-2012", "2012-2013")]
    if not all(path.exists() for path in files):
        raise FileNotFoundError(f"Missing annual files: {[str(p) for p in files if not p.exists()]}")
    customers = [str(i) for i in range(1, 301)]
    annual_arrays, annual_audit = [], []
    expected_days = (365, 366, 365)
    for path, days in zip(files, expected_days):
        log(f"Reading GC channel from {path.name}")
        frame = pd.read_csv(path, skiprows=1, low_memory=False)
        required = {"Customer", "Consumption Category", "date"}
        if not required.issubset(frame.columns):
            raise RuntimeError(f"Unexpected schema in {path.name}")
        time_columns = list(frame.columns[5:])
        if time_columns and time_columns[-1] == "Row Quality":
            time_columns = time_columns[:-1]
        if len(time_columns) != 48 or time_columns[0] != "0:30" or time_columns[-1] != "0:00":
            raise RuntimeError(f"Unexpected interval columns in {path.name}: {time_columns[:2]}...{time_columns[-2:]}")
        frame = frame[frame["Consumption Category"].astype(str).str.strip().eq("GC")].copy()
        frame["Customer"] = pd.to_numeric(frame["Customer"], errors="raise").astype(int)
        frame["date"] = pd.to_datetime(frame["date"], dayfirst=True, errors="raise")
        for col in time_columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        duplicate_rows = int(frame.duplicated(["date", "Customer"], keep=False).sum())
        grouped = frame.groupby(["date", "Customer"], sort=True)[time_columns].sum(min_count=1)
        dates = pd.date_range(frame["date"].min(), periods=days, freq="D")
        year_values = np.full((days * 48, 300), np.nan, dtype=np.float32)
        for customer in range(1, 301):
            try:
                block = grouped.xs(customer, level="Customer").reindex(dates)
            except KeyError:
                continue
            year_values[:, customer - 1] = block.to_numpy(dtype=np.float32).reshape(-1)
        year_values[year_values < 0] = np.nan
        annual_arrays.append(year_values)
        annual_audit.append({
            "file": path.name, "gc_rows": int(len(frame)), "duplicate_gc_rows": duplicate_rows,
            "date_min": str(frame["date"].min().date()), "date_max": str(frame["date"].max().date()),
            "raw_missing_fraction": float(np.mean(~np.isfinite(year_values))),
        })
        del frame, grouped, year_values
        gc.collect()
    values = np.concatenate(annual_arrays, axis=0)
    dt = pd.date_range("2010-07-01 00:30", "2013-07-01 00:00", freq="30min")
    if len(dt) != len(values):
        raise RuntimeError(f"Datetime/value mismatch: {len(dt)} vs {len(values)}")
    raw_missing = int(np.sum(~np.isfinite(values)))
    values, imputed = interpolate_short_gaps(values, 2)
    audit = {
        "annual": annual_audit, "timestamps": int(len(dt)), "customers_expected": 300,
        "start": str(dt[0]), "end": str(dt[-1]), "raw_missing_cells": raw_missing,
        "short_gap_cells_imputed": imputed, "post_imputation_missing_fraction": float(np.mean(~np.isfinite(values))),
        "units": "kWh per 30-minute interval (original GC values; no unit conversion)",
        "timestamp_convention": "interval ending: 0:30..23:30 and next-day 0:00",
    }
    return dt, values, customers, audit


def slice_bounds(dt: pd.DatetimeIndex, start: str | pd.Timestamp, stop: str | pd.Timestamp) -> tuple[int, int]:
    return int(dt.searchsorted(pd.Timestamp(start))), int(dt.searchsorted(pd.Timestamp(stop)))


def select_users(dt: pd.DatetimeIndex, values: np.ndarray, audit: dict) -> tuple[np.ndarray, dict]:
    train_start, train_end = slice_bounds(dt, "2010-07-01", "2012-07-01")
    segments = [("train", train_start, train_end), ("first_cal", *slice_bounds(dt, "2012-08-06", "2012-10-01"))]
    segments += [(name, *slice_bounds(dt, start, stop)) for name, start, stop in WINDOWS]
    eligible = np.ones(values.shape[1], dtype=bool)
    rates = {}
    for name, start, stop in segments:
        rate = np.mean(np.isfinite(values[start:stop]), axis=0)
        rates[name] = {"min": float(rate.min()), "median": float(np.median(rate)), "max": float(rate.max())}
        eligible &= rate >= 0.95
    train = values[train_start:train_end]
    eligible &= np.nanmean(train > 1e-6, axis=0) >= 0.20
    eligible &= np.nanstd(train, axis=0) > 1e-9
    selected = np.flatnonzero(eligible)
    selection = {
        "eligible_users": int(len(selected)), "excluded_users": int(values.shape[1] - len(selected)),
        "eligible_customer_ids": (selected + 1).tolist(), "segment_validity": rates,
        "minimum_valid_fraction": 0.95, "minimum_train_nonzero_fraction": 0.20,
    }
    audit["selection"] = selection
    if len(selected) < 100:
        raise RuntimeError(f"Only {len(selected)} eligible users; frozen minimum is 100")
    return selected, selection


def user_features(values: np.ndarray, dt: pd.DatetimeIndex) -> np.ndarray:
    eps = 1e-6
    mean, std = np.nanmean(values, axis=0), np.nanstd(values, axis=0)
    q95, zero = np.nanquantile(values, 0.95, axis=0), np.nanmean(values <= eps, axis=0)
    weekday = dt.dayofweek.to_numpy() < 5
    daytime = (dt.hour.to_numpy() >= 8) & (dt.hour.to_numpy() < 20)
    wd, we = np.nanmean(values[weekday], axis=0), np.nanmean(values[~weekday], axis=0)
    day, night = np.nanmean(values[daytime], axis=0), np.nanmean(values[~daytime], axis=0)
    x0, x1 = values[:-48], values[48:]
    m0, m1 = np.nanmean(x0, axis=0), np.nanmean(x1, axis=0)
    numerator = np.nansum((x0 - m0) * (x1 - m1), axis=0)
    denominator = np.sqrt(np.nansum((x0 - m0) ** 2, axis=0) * np.nansum((x1 - m1) ** 2, axis=0))
    feats = np.column_stack([
        np.log1p(mean), np.log1p(std), np.log1p(q95), zero, mean / np.maximum(q95, eps),
        np.clip(std / np.maximum(mean, eps), 0, 20), np.clip(wd / np.maximum(we, eps), 0, 10),
        np.clip(day / np.maximum(night, eps), 0, 10), np.clip(numerator / np.maximum(denominator, eps), -1, 1),
    ])
    return np.nan_to_num(feats, nan=0.0, posinf=20.0, neginf=-20.0)


def make_prefix(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid = np.isfinite(values)
    sums = np.vstack([np.zeros((1, values.shape[1]), dtype=np.float32), np.cumsum(np.where(valid, values, 0), axis=0, dtype=np.float32)])
    squares = np.vstack([np.zeros((1, values.shape[1]), dtype=np.float32), np.cumsum(np.where(valid, values * values, 0), axis=0, dtype=np.float32)])
    counts = np.vstack([np.zeros((1, values.shape[1]), dtype=np.int32), np.cumsum(valid, axis=0, dtype=np.int32)])
    return sums, squares, counts


def rolling_moments(prefix: tuple[np.ndarray, np.ndarray, np.ndarray], origins: np.ndarray, users: np.ndarray, width: int) -> tuple[np.ndarray, np.ndarray]:
    sums, squares, counts = prefix
    start, stop = origins - width + 1, origins + 1
    total = sums[stop, users] - sums[start, users]
    total2 = squares[stop, users] - squares[start, users]
    count = counts[stop, users] - counts[start, users]
    mean = total / np.maximum(count, 1)
    var = np.maximum(total2 / np.maximum(count, 1) - mean * mean, 0)
    mean[count == 0] = np.nan
    var[count == 0] = np.nan
    return mean.astype(np.float32), np.sqrt(var).astype(np.float32)


def build_rows(values, dt, prefix, origins, users, labels, stat):
    x = np.empty((len(origins), len(FEATURE_NAMES)), dtype=np.float32)
    for col, lag in enumerate((1, 2, 3, 48, 96, 336)):
        x[:, col] = values[origins - lag, users]
    x[:, 6], x[:, 7] = rolling_moments(prefix, origins, users, 48)
    x[:, 8], x[:, 9] = rolling_moments(prefix, origins, users, 336)
    target_dt = dt[origins + HORIZON]
    minute = target_dt.hour.to_numpy() * 60 + target_dt.minute.to_numpy()
    dow, month = target_dt.dayofweek.to_numpy(), target_dt.month.to_numpy()
    x[:, 10], x[:, 11] = np.sin(2*np.pi*minute/1440), np.cos(2*np.pi*minute/1440)
    x[:, 12], x[:, 13] = np.sin(2*np.pi*dow/7), np.cos(2*np.pi*dow/7)
    x[:, 14] = dow >= 5
    x[:, 15], x[:, 16] = np.sin(2*np.pi*(month-1)/12), np.cos(2*np.pi*(month-1)/12)
    x[:, 17], x[:, 18] = users, labels[users]
    x[:, 19:23] = stat[users]
    y = values[origins + HORIZON, users].astype(np.float32, copy=False)
    valid = np.isfinite(x).all(axis=1) & np.isfinite(y)
    return x[valid], y[valid], users[valid].astype(np.int32), labels[users[valid]].astype(np.int32)


def sampled_pairs(start, stop, n_users, n, seed):
    rng = np.random.default_rng(seed)
    lo, hi = max(start, MAX_LAG), stop - HORIZON
    total = max(0, hi - lo) * n_users
    flat = rng.choice(total, size=min(n, total), replace=False)
    return (lo + flat // n_users).astype(np.int32), (flat % n_users).astype(np.int32)


def full_pairs(start, stop, n_users):
    origins_1d = np.arange(max(start - HORIZON, MAX_LAG), stop - HORIZON, dtype=np.int32)
    return np.repeat(origins_1d, n_users), np.tile(np.arange(n_users, dtype=np.int32), len(origins_1d))


def conformal_quantile(scores):
    scores = np.asarray(scores, dtype=np.float64)
    scores = scores[np.isfinite(scores)]
    rank = min(len(scores)-1, math.ceil((len(scores)+1)*TARGET)-1)
    return float(np.partition(scores, rank)[rank])


def groups_for_users(users, groups, n_users):
    output = np.full(n_users, -1, dtype=int)
    output[users] = groups
    if np.any(output < 0):
        raise RuntimeError("Calibration window does not contain every retained user")
    return output


def normalized_corrections(scores, users, groups, scales, n_users, n_groups):
    normalized = scores / scales[users]
    global_q = conformal_quantile(normalized)
    group_q = np.array([conformal_quantile(normalized[groups == g]) for g in range(n_groups)])
    user_q = np.array([conformal_quantile(normalized[users == u]) for u in range(n_users)])
    user_groups = groups_for_users(users, groups, n_users)
    group_sizes = np.array([np.unique(users[groups == g]).size for g in range(n_groups)], dtype=float)
    group_weight = group_sizes / (group_sizes + 20.0)
    group_shrunk = group_weight * group_q + (1-group_weight) * global_q
    user_weight = 56.0 / (56.0 + 28.0)
    user_shrunk = user_weight * user_q + (1-user_weight) * group_shrunk[user_groups]
    corrections = {
        "raw": np.zeros(n_users), "rolling_global_norm": np.full(n_users, global_q),
        "rolling_group_norm": group_q[user_groups], "rolling_user_norm": user_q,
        "hierarchical_rolling_cqr": user_shrunk,
    }
    return corrections, global_q, group_q, group_weight, user_weight


def select_margin(y, q90, delta, users, scales, thresholds, min_recall=0.84):
    peak = y > thresholds[users]
    candidates = []
    for factor in MARGIN_GRID:
        alarm = q90 + delta + factor*scales[users] > thresholds[users]
        tp, fn = int(np.sum(peak & alarm)), int(np.sum(peak & ~alarm))
        fp = int(np.sum(~peak & alarm))
        recall, cost = tp/max(tp+fn, 1), 5*fn+fp
        candidates.append((recall < min_recall, cost, abs(factor), factor, recall))
    chosen = min(candidates)
    return chosen[3], chosen[4]


def evaluate(y, q10, q90, users, groups, scales, thresholds, user_correction, margin):
    delta = user_correction[users] * scales[users]
    lo, hi = q10-delta, q90+delta
    crossed = hi < lo
    midpoint = 0.5*(lo[crossed]+hi[crossed])
    lo[crossed], hi[crossed] = midpoint, midpoint
    covered, width = (y >= lo) & (y <= hi), hi-lo
    peak, alarm = y > thresholds[users], hi + margin*scales[users] > thresholds[users]
    tp, fn = int(np.sum(peak & alarm)), int(np.sum(peak & ~alarm))
    fp, tn = int(np.sum(~peak & alarm)), int(np.sum(~peak & ~alarm))
    user_cov = np.array([np.mean(covered[users == u]) for u in np.unique(users)])
    group_cov = np.array([np.mean(covered[groups == g]) for g in np.unique(groups)])
    return {
        "n": int(len(y)), "covered": int(covered.sum()), "width_sum": float(width.sum()),
        "picp": float(covered.mean()), "mpiw": float(width.mean()),
        "macro_user_abs_coverage_gap": float(np.mean(np.abs(user_cov-TARGET))),
        "user_coverage_std": float(np.std(user_cov)),
        "max_abs_cluster_coverage_gap": float(np.max(np.abs(group_cov-TARGET))),
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "peak_recall": tp/max(tp+fn, 1), "peak_fpr": fp/max(fp+tn, 1),
        "peak_precision": tp/max(tp+fp, 1), "decision_cost_fn5_fp1": 5*fn+fp,
        "covered_array": covered, "alarm_array": alarm, "peak_array": peak,
    }


def pooled_metrics(aggregate):
    rows = []
    for method in METHODS:
        a = aggregate[method]
        user_cov = a["user_covered"] / np.maximum(a["user_n"], 1)
        group_cov = a["group_covered"] / np.maximum(a["group_n"], 1)
        tp, fn, fp, tn = a["tp"], a["fn"], a["fp"], a["tn"]
        rows.append({
            "method": method, "n": int(a["n"]), "picp": a["covered"]/a["n"], "mpiw": a["width_sum"]/a["n"],
            "macro_user_abs_coverage_gap": float(np.mean(np.abs(user_cov-TARGET))),
            "user_coverage_std": float(np.std(user_cov)),
            "max_abs_cluster_coverage_gap": float(np.max(np.abs(group_cov-TARGET))),
            "peak_recall": tp/max(tp+fn, 1), "peak_fpr": fp/max(fp+tn, 1), "peak_precision": tp/max(tp+fp, 1),
            "missed_exceedances": int(fn), "false_alarms": int(fp), "decision_cost_fn5_fp1": int(5*fn+fp),
        })
    return rows


def bootstrap_user_differences(user_rows, repetitions=10000):
    frame = pd.DataFrame(user_rows)
    gap = frame.pivot(index="user_index", columns="method", values="coverage_gap")
    cost = frame.pivot(index="user_index", columns="method", values="cost")
    gap_diff = (gap["hierarchical_rolling_cqr"]-gap["rolling_global_norm"]).to_numpy()
    cost_diff = (cost["hierarchical_rolling_cqr"]-cost["rolling_global_norm"]).to_numpy()
    rng, n = np.random.default_rng(20260730), len(gap_diff)
    gap_boot, cost_boot = np.empty(repetitions), np.empty(repetitions)
    for i in range(repetitions):
        idx = rng.integers(0, n, n)
        gap_boot[i], cost_boot[i] = gap_diff[idx].mean(), cost_diff[idx].mean()
    return {
        "gap_mean_difference": float(gap_diff.mean()), "gap_ci95": np.quantile(gap_boot, [0.025,0.975]).tolist(),
        "cost_mean_difference": float(cost_diff.mean()), "cost_ci95": np.quantile(cost_boot, [0.025,0.975]).tolist(),
        "users": n, "bootstrap_repetitions": repetitions,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=HERE/"raw_archive")
    parser.add_argument("--out", type=Path, default=HERE/"external_frozen_full")
    parser.add_argument("--train-rows", type=int, default=600000)
    parser.add_argument("--num-boost-round", type=int, default=250)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    dt, raw_values, customer_names, audit = read_ausgrid(args.raw_dir)
    selected, _ = select_users(dt, raw_values, audit)
    values, customer_names = raw_values[:, selected], [customer_names[i] for i in selected]
    audit["retained_matrix_shape"] = list(values.shape)
    (args.out/"data_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    if args.audit_only:
        log(json.dumps({"audit_only": True, "eligible_users": len(selected), "shape": list(values.shape)}, indent=2))
        return

    train_start, train_end = slice_bounds(dt, "2010-07-01", "2012-07-01")
    train_values = values[train_start:train_end]
    uf = user_features(train_values, dt[train_start:train_end])
    center, spread = np.median(uf, axis=0), np.std(uf, axis=0)
    labels, centers = base.kmeans((uf-center)/np.where(spread>1e-9, spread, 1), 3, SEED)
    stat = np.column_stack([
        np.nanmean(train_values, axis=0), np.nanstd(train_values, axis=0),
        np.nanquantile(train_values, 0.95, axis=0), np.nanmean(train_values <= 1e-6, axis=0),
    ]).astype(np.float32)
    scales = np.maximum.reduce([stat[:,1], 0.1*stat[:,2], np.full(len(selected), 1e-3)]).astype(float)
    thresholds = stat[:,2].astype(float)
    prefix = make_prefix(values)

    origins, users = sampled_pairs(train_start, train_end, len(selected), args.train_rows, SEED)
    x_train, y_train, _, _ = build_rows(values, dt, prefix, origins, users, labels, stat)
    log(f"Training rows after validity filter: {len(y_train):,}")
    params = {"verbosity":-1, "learning_rate":0.05, "num_leaves":31, "min_data_in_leaf":100,
              "feature_fraction":0.9, "bagging_fraction":0.9, "bagging_freq":1, "lambda_l2":1.0,
              "seed":SEED, "num_threads":args.threads, "force_col_wise":True}
    dataset = base.lgb.Dataset(x_train, label=y_train, feature_name=FEATURE_NAMES,
                               categorical_feature=[17,18], free_raw_data=False)
    models = {}
    for tau in (0.1,0.5,0.9):
        log(f"Training quantile model tau={tau}")
        models[tau] = base.lgb.train(dict(params, objective="quantile", alpha=tau, metric="quantile"), dataset,
                                     num_boost_round=args.num_boost_round)
    del dataset, x_train, y_train, raw_values, train_values, uf
    gc.collect()

    n_users, n_groups = len(selected), 3
    aggregate = {m:{"n":0,"covered":0,"width_sum":0.0,"tp":0,"fn":0,"fp":0,"tn":0,
                    "user_n":np.zeros(n_users),"user_covered":np.zeros(n_users),"user_cost":np.zeros(n_users),
                    "group_n":np.zeros(n_groups),"group_covered":np.zeros(n_groups)} for m in METHODS}
    window_rows, correction_rows = [], []
    for name, test_start_ts, test_end_ts in WINDOWS:
        log(f"Evaluating frozen window {name}")
        cal_start, test_start, test_end = (int(dt.searchsorted(x)) for x in (test_start_ts-pd.Timedelta(days=56), test_start_ts, test_end_ts))
        origins, users = full_pairs(cal_start, test_start, n_users)
        x_cal, y_cal, cal_users, cal_groups = build_rows(values, dt, prefix, origins, users, labels, stat)
        c10, _, c90, _ = base.predict_three(models, x_cal)
        scores = np.maximum(c10-y_cal, y_cal-c90)
        corr, global_q, group_q, group_weight, user_weight = normalized_corrections(scores, cal_users, cal_groups, scales, n_users, n_groups)
        margins = {}
        for method in METHODS:
            delta = corr[method][cal_users]*scales[cal_users]
            margins[method], cal_recall = select_margin(y_cal, c90, delta, cal_users, scales, thresholds)
            correction_rows.append({"window":name,"method":method,"mean_normalized_correction":float(np.mean(corr[method])),
                "std_normalized_correction":float(np.std(corr[method])),"alarm_margin":margins[method],
                "calibration_recall_at_margin":cal_recall,"global_q":global_q,"group_q":json.dumps(group_q.tolist()),
                "group_weight":json.dumps(group_weight.tolist()),"user_weight":user_weight})
        del x_cal, y_cal, c10, c90, scores, cal_users, cal_groups
        gc.collect()

        origins, users = full_pairs(test_start, test_end, n_users)
        x_test, y_test, test_users, test_groups = build_rows(values, dt, prefix, origins, users, labels, stat)
        q10, _, q90, _ = base.predict_three(models, x_test)
        for method in METHODS:
            metrics = evaluate(y_test, q10, q90, test_users, test_groups, scales, thresholds, corr[method], margins[method])
            window_rows.append({"window":name,"method":method,**{k:v for k,v in metrics.items() if not k.endswith("_array") and k not in ("n","covered","width_sum","tp","fn","fp","tn")}})
            a = aggregate[method]
            a["n"] += metrics["n"]; a["covered"] += metrics["covered"]; a["width_sum"] += metrics["width_sum"]
            for key in ("tp","fn","fp","tn"): a[key] += metrics[key]
            covered, alarm, peak = metrics["covered_array"], metrics["alarm_array"], metrics["peak_array"]
            for u in range(n_users):
                mask = test_users == u
                a["user_n"][u] += mask.sum(); a["user_covered"][u] += covered[mask].sum()
                a["user_cost"][u] += 5*np.sum(peak[mask]&~alarm[mask]) + np.sum(~peak[mask]&alarm[mask])
            for g in range(n_groups):
                mask = test_groups == g
                a["group_n"][g] += mask.sum(); a["group_covered"][g] += covered[mask].sum()
        del x_test, y_test, q10, q90, test_users, test_groups
        gc.collect()

    pooled = pooled_metrics(aggregate)
    user_rows = []
    for method in METHODS:
        a = aggregate[method]; cov = a["user_covered"]/np.maximum(a["user_n"],1)
        for u, customer in enumerate(customer_names):
            user_rows.append({"user_index":u,"customer":customer,"cluster":int(labels[u]),"method":method,
                              "coverage":float(cov[u]),"coverage_gap":float(abs(cov[u]-TARGET)),"cost":float(a["user_cost"][u])})
    boot = bootstrap_user_differences(user_rows)
    by = {row["method"]:row for row in pooled}
    glob, hier, user = by["rolling_global_norm"], by["hierarchical_rolling_cqr"], by["rolling_user_norm"]
    cluster_reduction = 1-hier["max_abs_cluster_coverage_gap"]/max(glob["max_abs_cluster_coverage_gap"],1e-12)
    user_reduction = 1-hier["macro_user_abs_coverage_gap"]/max(glob["macro_user_abs_coverage_gap"],1e-12)
    width_change = hier["mpiw"]/max(glob["mpiw"],1e-12)-1
    cost_change = hier["decision_cost_fn5_fp1"]/max(glob["decision_cost_fn5_fp1"],1)-1
    wf = pd.DataFrame(window_rows)
    group_windows = int(np.sum(wf[wf.method=="hierarchical_rolling_cqr"].max_abs_cluster_coverage_gap.to_numpy() < wf[wf.method=="rolling_global_norm"].max_abs_cluster_coverage_gap.to_numpy()))
    user_windows = int(np.sum(wf[wf.method=="hierarchical_rolling_cqr"].macro_user_abs_coverage_gap.to_numpy() < wf[wf.method=="rolling_global_norm"].macro_user_abs_coverage_gap.to_numpy()))
    independent_value = hier["macro_user_abs_coverage_gap"] <= user["macro_user_abs_coverage_gap"] or hier["mpiw"] <= 0.95*user["mpiw"]
    checks = {"cluster_gap_reduction_ge_20pct":cluster_reduction>=0.20,"user_gap_reduction_ge_10pct":user_reduction>=0.10,
              "mpiw_increase_le_10pct":width_change<=0.10,"decision_cost_not_increase":cost_change<=0,
              "group_improves_at_least_2_windows":group_windows>=2,"user_improves_at_least_2_windows":user_windows>=2,
              "bootstrap_gap_upper_below_zero":boot["gap_ci95"][1]<0,"hierarchy_has_value_vs_user_cqr":bool(independent_value)}
    verdict = "EXTERNAL_GO" if all(checks.values()) else "EXTERNAL_NO_GO_USER_OR_DECISION" if checks["cluster_gap_reduction_ge_20pct"] else "EXTERNAL_NO_GO"
    decision = {"verdict":verdict,"checks":{k:bool(v) for k,v in checks.items()},
        "effect_sizes":{"cluster_gap_reduction":cluster_reduction,"user_gap_reduction":user_reduction,
        "mpiw_relative_change":width_change,"decision_cost_relative_change":cost_change,
        "group_windows_improved":group_windows,"user_windows_improved":user_windows},"bootstrap":boot,
        "data":{"retained_users":n_users,"cluster_sizes":np.bincount(labels,minlength=3).tolist(),"windows":[x[0] for x in WINDOWS],"test_samples":int(hier["n"])},
        "wall_seconds":time.perf_counter()-started}
    write_csv(args.out/"window_metrics.csv",window_rows); write_csv(args.out/"pooled_metrics.csv",pooled)
    write_csv(args.out/"per_user_metrics.csv",user_rows); write_csv(args.out/"corrections.csv",correction_rows)
    (args.out/"decision.json").write_text(json.dumps(decision,indent=2),encoding="utf-8")
    (args.out/"config.json").write_text(json.dumps({"protocol":"FROZEN_EXTERNAL_PROTOCOL_2026-07-30.md",**vars(args)},indent=2,default=str),encoding="utf-8")
    lines = ["# Independent external confirmation: Ausgrid", "", f"Decision: **{verdict}**", "", "| Method | PICP | MPIW | User gap | Max group gap | Peak cost |", "|---|---:|---:|---:|---:|---:|"]
    for row in pooled:
        lines.append(f"| {row['method']} | {row['picp']:.4f} | {row['mpiw']:.4f} | {row['macro_user_abs_coverage_gap']:.4f} | {row['max_abs_cluster_coverage_gap']:.4f} | {row['decision_cost_fn5_fp1']} |")
    lines += ["", "## Frozen gates", ""] + [f"- `{k}`: **{'pass' if v else 'fail'}**" for k,v in checks.items()]
    (args.out/"EXTERNAL_REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps(decision,indent=2),flush=True)


if __name__ == "__main__":
    main()
