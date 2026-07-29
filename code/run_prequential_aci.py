"""Strictly-forward canonical ACI and conformal-PID baselines for the CQR benchmark.

Each calendar month is an independent deployment environment.  The score set
is frozen from its preceding 56 days; only alpha is updated during the month,
and each update is made after issuing the interval for that timestamp.
"""

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
EXT = ROOT / "probabilistic_load_hierarchical_rolling_cqr_external_2026-07-30"
V1 = ROOT / "probabilistic_load_group_cqr_go_nogo_2026-07-25"
for path in (HERE, EXT, V1):
    sys.path.insert(0, str(path))
if sys.version_info[:2] == (3, 12):
    sys.path.insert(0, str(V1 / ".deps"))
import lightgbm  # noqa: E402,F401  # Pin the interpreter-compatible wheel before importing legacy adapters.

import diagnostic_metrics as diag  # noqa: E402
import run_external as ext  # noqa: E402
import run_naive_robustness as naive  # noqa: E402
import run_validation as base  # noqa: E402
def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


TARGET = .80
INITIAL_ALPHA = .20
ALPHA_MIN, ALPHA_MAX = .01, .50
CALIBRATION_DAYS = 56
STATIC_METHODS = ("raw", "rolling_global_norm", "rolling_group_norm", "rolling_user_norm")
ACI_METHODS = ("aci_global", "aci_segment", "aci_user")
PID_METHODS = ("pid_global", "pid_segment", "pid_user")
DYNAMIC_METHODS = ACI_METHODS + PID_METHODS
ALL_METHODS = STATIC_METHODS + DYNAMIC_METHODS
DATASET_NAMES = naive.DATASET_NAMES
PID_ETA = .05
PID_CSAT = 5.0
PID_KI = 10.0


def log(message: str) -> None:
    print(time.strftime("[%H:%M:%S]"), message, flush=True)


def sorted_quantile(scores: np.ndarray, probability: float) -> float:
    """Finite-sample conformal order statistic, from a sorted score set."""
    if not 0 < probability < 1:
        raise ValueError("probability must lie in (0, 1)")
    if scores.ndim != 1 or scores.size == 0 or not np.isfinite(scores).all():
        raise RuntimeError("invalid fixed ACI score set")
    rank = min(scores.size - 1, math.ceil((scores.size + 1) * probability) - 1)
    return float(scores[rank])


def evaluate(y, qlo, qhi, users, labels, scales, normalized_correction):
    """Return interval-quality and two granularity coverage diagnostics."""
    lo, hi, covered = interval_coverage(y, qlo, qhi, users, scales, normalized_correction)
    width = hi - lo
    n_users = len(labels)
    user_n = np.bincount(users, minlength=n_users)
    user_covered = np.bincount(users, weights=covered.astype(float), minlength=n_users)
    if np.any(user_n == 0):
        raise RuntimeError("a retained user is absent from an evaluation window")
    user_cov = user_covered / user_n
    groups = labels[users]
    n_groups = int(np.max(labels)) + 1
    group_n = np.bincount(groups, minlength=n_groups)
    group_covered = np.bincount(groups, weights=covered.astype(float), minlength=n_groups)
    group_cov = group_covered / group_n
    alpha = 1.0 - TARGET
    interval_score = width + (2.0 / alpha) * (lo - y) * (y < lo) + (2.0 / alpha) * (y - hi) * (y > hi)
    return {
        "n": int(len(y)),
        "picp": float(covered.mean()),
        "mpiw": float(width.mean()),
        "winkler_interval_score": float(interval_score.mean()),
        "macro_user_abs_coverage_gap": float(np.abs(user_cov - TARGET).mean()),
        "user_coverage_std": float(user_cov.std()),
        "max_abs_cluster_coverage_gap": float(np.abs(group_cov - TARGET).max()),
    }, user_cov, user_n, covered


def interval_coverage(y, qlo, qhi, users, scales, normalized_correction):
    """Compute an interval and its coverage without requiring all users."""
    delta = normalized_correction * scales[users]
    lo, hi = qlo - delta, qhi + delta
    crossed = hi < lo
    midpoint = .5 * (lo[crossed] + hi[crossed])
    lo[crossed], hi[crossed] = midpoint, midpoint
    return lo, hi, (y >= lo) & (y <= hi)


def static_corrections(scores, users, labels, n_users):
    """Static 56-day split-CQR corrections using the same fixed score set."""
    normalized = scores / 1.0  # documents that score construction is upstream
    group_count = int(np.max(labels)) + 1
    global_scores = np.sort(normalized)
    group_scores = [np.sort(normalized[labels[users] == g]) for g in range(group_count)]
    user_scores = [np.sort(normalized[users == u]) for u in range(n_users)]
    if any(item.size == 0 for item in group_scores + user_scores):
        raise RuntimeError("the 56-day calibration block misses a group or user")
    global_q = sorted_quantile(global_scores, TARGET)
    group_q = np.array([sorted_quantile(item, TARGET) for item in group_scores])
    user_q = np.array([sorted_quantile(item, TARGET) for item in user_scores])
    return {
        "raw": np.zeros(n_users),
        "rolling_global_norm": np.full(n_users, global_q),
        "rolling_group_norm": group_q[labels],
        "rolling_user_norm": user_q,
    }, {"global": global_scores, "segment": group_scores, "user": user_scores}


def run_batched_aci(y, qlo, qhi, users, labels, scales, target_index, score_sets, cadence):
    """Apply prequential ACI without adding any test labels to score sets."""
    gamma = .005 * cadence / 30.0
    n_users, n_groups = len(labels), int(np.max(labels)) + 1
    alpha_global = INITIAL_ALPHA
    alpha_segment = np.full(n_groups, INITIAL_ALPHA)
    alpha_user = np.full(n_users, INITIAL_ALPHA)
    corrections = {method: np.empty(len(y), dtype=float) for method in ACI_METHODS}
    trace = []
    for index in np.unique(target_index):
        rows = np.flatnonzero(target_index == index)
        row_users = users[rows]
        row_groups = labels[row_users]
        global_q = sorted_quantile(score_sets["global"], 1.0 - alpha_global)
        segment_q = np.array([sorted_quantile(score_sets["segment"][g], 1.0 - alpha_segment[g]) for g in range(n_groups)])
        user_q = np.array([sorted_quantile(score_sets["user"][u], 1.0 - alpha_user[u]) for u in range(n_users)])
        corrections["aci_global"][rows] = global_q
        corrections["aci_segment"][rows] = segment_q[row_groups]
        corrections["aci_user"][rows] = user_q[row_users]
        error_by_method = {}
        for method in ACI_METHODS:
            _, _, covered = interval_coverage(y[rows], qlo[rows], qhi[rows], row_users, scales, corrections[method][rows])
            error_by_method[method] = 1.0 - covered.astype(float)
        global_error = float(error_by_method["aci_global"].mean())
        alpha_global = float(np.clip(alpha_global + gamma * (INITIAL_ALPHA - global_error), ALPHA_MIN, ALPHA_MAX))
        segment_error = error_by_method["aci_segment"]
        for group in np.unique(row_groups):
            mask = row_groups == group
            alpha_segment[group] = np.clip(alpha_segment[group] + gamma * (INITIAL_ALPHA - segment_error[mask].mean()), ALPHA_MIN, ALPHA_MAX)
        user_error = error_by_method["aci_user"]
        for user, error in zip(row_users, user_error, strict=True):
            alpha_user[user] = np.clip(alpha_user[user] + gamma * (INITIAL_ALPHA - error), ALPHA_MIN, ALPHA_MAX)
        trace.append({
            "target_index": int(index), "gamma": gamma,
            "aci_global_alpha_after": alpha_global, "aci_global_error": global_error,
            "aci_segment_mean_alpha_after": float(alpha_segment.mean()), "aci_segment_mean_error": float(segment_error.mean()),
            "aci_user_mean_alpha_after": float(alpha_user.mean()), "aci_user_mean_error": float(user_error.mean()),
        })
    return corrections, trace


def score_scale(scores: np.ndarray) -> float:
    """Robust fixed scale for the proportional PID learning rate."""
    spread = float(np.quantile(scores, .95) - np.quantile(scores, .05))
    return max(spread, 1e-6)


def saturated_integrator(error_sum, steps):
    """Logarithmically saturated integrator from conformal PID control."""
    error_sum = np.asarray(error_sum, dtype=float)
    steps = np.asarray(steps, dtype=float)
    argument = error_sum * np.log(steps + 1.0) / (PID_CSAT * (steps + 1.0))
    argument = np.clip(argument, -np.pi / 2.0 + 1e-9, np.pi / 2.0 - 1e-9)
    return PID_KI * np.tan(argument)


def run_batched_pid(y, qlo, qhi, users, labels, scales, target_index, score_sets):
    """Run the conformal PID PI-controller variant with atomic panel updates."""
    n_users, n_groups = len(labels), int(np.max(labels)) + 1
    alpha = 1.0 - TARGET
    tracker_global = sorted_quantile(score_sets["global"], TARGET)
    tracker_segment = np.array([sorted_quantile(item, TARGET) for item in score_sets["segment"]])
    tracker_user = np.array([sorted_quantile(item, TARGET) for item in score_sets["user"]])
    lr_global = PID_ETA * score_scale(score_sets["global"])
    lr_segment = PID_ETA * np.array([score_scale(item) for item in score_sets["segment"]])
    lr_user = PID_ETA * np.array([score_scale(item) for item in score_sets["user"]])
    integral_global = 0.0
    integral_segment = np.zeros(n_groups)
    integral_user = np.zeros(n_users)
    error_sum_global, steps_global = 0.0, 0.0
    error_sum_segment, steps_segment = np.zeros(n_groups), np.zeros(n_groups)
    error_sum_user, steps_user = np.zeros(n_users), np.zeros(n_users)
    corrections = {method: np.empty(len(y), dtype=float) for method in PID_METHODS}
    trace = []
    for index in np.unique(target_index):
        rows = np.flatnonzero(target_index == index)
        row_users = users[rows]
        row_groups = labels[row_users]
        corrections["pid_global"][rows] = tracker_global + integral_global
        corrections["pid_segment"][rows] = (tracker_segment + integral_segment)[row_groups]
        corrections["pid_user"][rows] = (tracker_user + integral_user)[row_users]
        error_by_method = {}
        for method in PID_METHODS:
            _, _, covered = interval_coverage(
                y[rows], qlo[rows], qhi[rows], row_users, scales, corrections[method][rows]
            )
            error_by_method[method] = 1.0 - covered.astype(float)
        global_error = float(error_by_method["pid_global"].mean())
        tracker_global += lr_global * (global_error - alpha)
        error_sum_global += global_error - alpha
        steps_global += 1.0
        integral_global = float(saturated_integrator(error_sum_global, steps_global))
        segment_error = error_by_method["pid_segment"]
        for group in np.unique(row_groups):
            mask = row_groups == group
            group_error = float(segment_error[mask].mean())
            tracker_segment[group] += lr_segment[group] * (group_error - alpha)
            error_sum_segment[group] += group_error - alpha
            steps_segment[group] += 1.0
            integral_segment[group] = float(saturated_integrator(error_sum_segment[group], steps_segment[group]))
        user_error = error_by_method["pid_user"]
        tracker_user[row_users] += lr_user[row_users] * (user_error - alpha)
        error_sum_user[row_users] += user_error - alpha
        steps_user[row_users] += 1.0
        integral_user[row_users] = saturated_integrator(error_sum_user[row_users], steps_user[row_users])
        trace.append({
            "target_index": int(index), "pid_eta": PID_ETA, "pid_csat": PID_CSAT, "pid_ki": PID_KI,
            "pid_global_threshold_after": float(tracker_global + integral_global),
            "pid_global_error": global_error,
            "pid_segment_mean_threshold_after": float(np.mean(tracker_segment + integral_segment)),
            "pid_segment_mean_error": float(segment_error.mean()),
            "pid_user_mean_threshold_after": float(np.mean(tracker_user + integral_user)),
            "pid_user_mean_error": float(user_error.mean()),
        })
    return corrections, trace


def ext_build_pairs(data, origins, users):
    """External LightGBM feature builder retaining the target time index."""
    values, dt, prefix, labels, stat = data["values"], data["dt"], data["prefix"], data["labels"], data["stat"]
    x = np.empty((len(origins), len(ext.FEATURE_NAMES)), dtype=np.float32)
    for col, lag in enumerate((1, 2, 3, 48, 96, 336)):
        x[:, col] = values[origins - lag, users]
    x[:, 6], x[:, 7] = ext.rolling_moments(prefix, origins, users, 48)
    x[:, 8], x[:, 9] = ext.rolling_moments(prefix, origins, users, 336)
    target = origins + ext.HORIZON
    target_dt = dt[target]
    minute = target_dt.hour.to_numpy() * 60 + target_dt.minute.to_numpy()
    dow, month = target_dt.dayofweek.to_numpy(), target_dt.month.to_numpy()
    x[:, 10], x[:, 11] = np.sin(2 * np.pi * minute / 1440), np.cos(2 * np.pi * minute / 1440)
    x[:, 12], x[:, 13] = np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7)
    x[:, 14] = dow >= 5
    x[:, 15], x[:, 16] = np.sin(2 * np.pi * (month - 1) / 12), np.cos(2 * np.pi * (month - 1) / 12)
    x[:, 17], x[:, 18] = users, labels[users]
    x[:, 19:23] = stat[users]
    y = values[target, users].astype(np.float32, copy=False)
    valid = np.isfinite(x).all(axis=1) & np.isfinite(y)
    return x[valid], y[valid], users[valid].astype(np.int32), labels[users[valid]].astype(np.int32), target[valid].astype(np.int32)


def uci_build_pairs(data, origins, users):
    x, y, uu, gg, target = base.build_rows(data["values"], data["dt"], origins, users, data["labels"], data["stat"])
    valid = np.isfinite(x).all(axis=1) & np.isfinite(y)
    return x[valid], y[valid], uu[valid], gg[valid], target[valid]


def lgbm_prediction_rows(data, start, stop, models):
    n_users = len(data["names"])
    if data["kind"] == "external":
        origins, users = ext.full_pairs(start, stop, n_users)
        x, y, users, groups, target_index = ext_build_pairs(data, origins, users)
    else:
        origins, users = base.full_pairs(start, stop, n_users)
        x, y, users, groups, target_index = uci_build_pairs(data, origins, users)
    qlo, _, qhi, _ = base.predict_three(models, x)
    return y.astype(float), qlo.astype(float), qhi.astype(float), users, groups, target_index


def train_lgbm(data, train_rows, boost_round, threads):
    n_users = len(data["names"])
    if data["kind"] == "external":
        origins, users = ext.sampled_pairs(data["train0"], data["train1"], n_users, train_rows, ext.SEED)
        x, y, _, _, _ = ext_build_pairs(data, origins, users)
        names, seed = ext.FEATURE_NAMES, ext.SEED
    else:
        origins, users = base.sampled_pairs(data["train0"], data["train1"], n_users, train_rows, base.SEED)
        x, y, _, _, _ = uci_build_pairs(data, origins, users)
        names, seed = base.FEATURE_NAMES, base.SEED
    log(f"Training LightGBM rows: {len(y):,}")
    params = {"verbosity": -1, "learning_rate": .05, "num_leaves": 31, "min_data_in_leaf": 100,
              "feature_fraction": .9, "bagging_fraction": .9, "bagging_freq": 1, "lambda_l2": 1.,
              "seed": seed, "num_threads": threads, "force_col_wise": True}
    dataset = base.lgb.Dataset(x, label=y, feature_name=names, categorical_feature=[17, 18], free_raw_data=False)
    models = {tau: base.lgb.train(dict(params, objective="quantile", alpha=tau, metric="quantile"), dataset, num_boost_round=boost_round)
              for tau in (.1, .5, .9)}
    del x, y, dataset
    gc.collect()
    return models


def persistence_prediction_rows(data, start, stop, low, high, lag):
    values = data["values"]
    indexes = np.arange(max(start, lag), stop, dtype=np.int32)
    y_matrix = values[indexes]
    qlo_matrix = values[indexes - lag] + low
    qhi_matrix = values[indexes - lag] + high
    valid = np.isfinite(y_matrix) & np.isfinite(qlo_matrix) & np.isfinite(qhi_matrix)
    time_rows, users = np.nonzero(valid)
    return (y_matrix[valid].astype(float), qlo_matrix[valid].astype(float), qhi_matrix[valid].astype(float),
            users.astype(np.int32), data["labels"][users].astype(np.int32), indexes[time_rows])


def load_data(dataset: str, prepared: Path | None = None, raw_dir: Path | None = None, zip_path: Path | None = None):
    if dataset == "london":
        dt, values, names, labels, scales, train0, train1, windows, cadence = naive.load_london(prepared or naive.LONDON, 3, "kmeans")
        kind = "external"
    elif dataset == "ausgrid":
        dt, values, names, labels, scales, train0, train1, windows, cadence = naive.load_ausgrid(raw_dir or EXT / "raw_archive", 3, "kmeans")
        kind = "external"
    else:
        dt, values, names, labels, scales, train0, train1, windows, cadence = naive.load_uci(
            zip_path or V1 / "electricityloaddiagrams20112014.originalmirror.zip", 3, "kmeans"
        )
        kind = "uci"
    train = values[train0:train1]
    stat = np.column_stack([np.nanmean(train, axis=0), np.nanstd(train, axis=0), np.nanquantile(train, .95, axis=0), np.nanmean(train <= 1e-6, axis=0)]).astype(np.float32)
    demand_order = np.lexsort((np.asarray(names, dtype=str), stat[:, 0]))
    demand_tertile = np.empty(len(names), dtype=int)
    for group, members in enumerate(np.array_split(demand_order, 3)):
        demand_tertile[members] = group
    data = {"dataset": DATASET_NAMES[dataset], "kind": kind, "dt": dt, "values": values, "names": names,
            "labels": labels.astype(int), "scales": scales.astype(float), "train0": train0, "train1": train1,
            "windows": windows, "cadence": cadence, "stat": stat, "demand_tertile": demand_tertile}
    if kind == "external":
        data["prefix"] = ext.make_prefix(values)
    return data


def run(data, forecaster, train_rows, boost_round, threads, max_windows):
    if forecaster == "lgbm":
        models = train_lgbm(data, train_rows, boost_round, threads)
        predictor = lambda start, stop: lgbm_prediction_rows(data, start, stop, models)
        forecaster_name = "lightgbm_quantile"
    else:
        lag = int(round(60 / data["cadence"]))
        low, high = naive.residual_quantiles(data["values"], data["train0"], data["train1"], lag, TARGET)
        predictor = lambda start, stop: persistence_prediction_rows(data, start, stop, low, high, lag)
        forecaster_name = "persistence_quantile_interval"
    window_rows, user_rows, update_rows, pid_update_rows, correction_rows = [], [], [], [], []
    selected = data["windows"] if max_windows is None else data["windows"][:max_windows]
    for window, ts, te in selected:
        log(f"{data['dataset']} {forecaster} forward ACI/PID: {window}")
        cal0 = int(data["dt"].searchsorted(ts - pd.Timedelta(days=CALIBRATION_DAYS)))
        test0, test1 = int(data["dt"].searchsorted(ts)), int(data["dt"].searchsorted(te))
        ycal, loc, hic, ucal, _, _ = predictor(cal0, test0)
        scores = np.maximum(loc - ycal, ycal - hic) / data["scales"][ucal]
        static, score_sets = static_corrections(scores, ucal, data["labels"], len(data["names"]))
        y, qlo, qhi, users, _, target_index = predictor(test0, test1)
        aci, trace = run_batched_aci(y, qlo, qhi, users, data["labels"], data["scales"], target_index, score_sets, data["cadence"])
        pid, pid_trace = run_batched_pid(y, qlo, qhi, users, data["labels"], data["scales"], target_index, score_sets)
        corrections = {**static, **aci, **pid}
        for method in ALL_METHODS:
            method_correction = corrections[method][users] if method in STATIC_METHODS else corrections[method]
            metrics, user_cov, user_n, _ = evaluate(y, qlo, qhi, users, data["labels"], data["scales"], method_correction)
            window_rows.append({"dataset": data["dataset"], "forecaster": forecaster_name, "coverage": TARGET,
                                "horizon_hours": 1., "window": window, "method": method, **metrics})
            for user in range(len(data["names"])):
                user_rows.append({"dataset": data["dataset"], "forecaster": forecaster_name, "coverage_target": TARGET,
                                  "horizon_hours": 1., "window": window, "method": method, "user_index": user,
                                  "customer": data["names"][user], "cluster": int(data["labels"][user]),
                                  "demand_tertile": int(data["demand_tertile"][user]),
                                  "coverage": float(user_cov[user]), "n": int(user_n[user]),
                                  "coverage_gap": float(abs(user_cov[user] - TARGET))})
        for row in trace:
            update_rows.append({"dataset": data["dataset"], "forecaster": forecaster_name, "window": window, **row})
        for row in pid_trace:
            pid_update_rows.append({"dataset": data["dataset"], "forecaster": forecaster_name, "window": window, **row})
        correction_rows.append({"dataset": data["dataset"], "forecaster": forecaster_name, "window": window,
                                "fixed_score_set_days": CALIBRATION_DAYS, "initial_alpha": INITIAL_ALPHA,
                                "gamma": .005 * data["cadence"] / 30., "static_global_correction": float(static["rolling_global_norm"][0]),
                                "static_segment_correction_mean": float(static["rolling_group_norm"].mean()),
                                "static_user_correction_mean": float(static["rolling_user_norm"].mean()),
                                "aci_global_final_alpha": trace[-1]["aci_global_alpha_after"],
                                "aci_segment_final_mean_alpha": trace[-1]["aci_segment_mean_alpha_after"],
                                "aci_user_final_mean_alpha": trace[-1]["aci_user_mean_alpha_after"],
                                "pid_eta": PID_ETA, "pid_csat": PID_CSAT, "pid_ki": PID_KI,
                                "pid_global_final_threshold": pid_trace[-1]["pid_global_threshold_after"],
                                "pid_segment_final_mean_threshold": pid_trace[-1]["pid_segment_mean_threshold_after"],
                                "pid_user_final_mean_threshold": pid_trace[-1]["pid_user_mean_threshold_after"]})
        gc.collect()
    summary, details = diag.compute_all(window_rows, user_rows, target=TARGET)
    aci_window = [{**row, "method": {"aci_global": "rolling_global_norm", "aci_segment": "rolling_group_norm", "aci_user": "rolling_user_norm"}[row["method"]]}
                  for row in window_rows if row["method"] in ACI_METHODS]
    aci_gcr, aci_conflicts = diag.granularity_conflict(pd.DataFrame(aci_window))
    summary["aci_granularity_conflict"] = aci_gcr
    pid_window = [{**row, "method": {"pid_global": "rolling_global_norm", "pid_segment": "rolling_group_norm", "pid_user": "rolling_user_norm"}[row["method"]]}
                  for row in window_rows if row["method"] in PID_METHODS]
    pid_gcr, pid_conflicts = diag.granularity_conflict(pd.DataFrame(pid_window))
    summary["pid_granularity_conflict"] = pid_gcr
    summary["data"] = {"dataset": data["dataset"], "forecaster": forecaster_name, "users": len(data["names"]),
                       "segments": int(np.max(data["labels"]) + 1), "cluster_method": "kmeans",
                       "cluster_sizes": np.bincount(data["labels"], minlength=int(np.max(data["labels"]) + 1)).tolist(),
                       "cadence_minutes": data["cadence"], "windows": [item[0] for item in selected],
                       "score_set_days": CALIBRATION_DAYS, "strict_forward": True,
                       "atomic_timestamp_updates": True,
                       "alpha_clip": [ALPHA_MIN, ALPHA_MAX], "initial_alpha": INITIAL_ALPHA,
                       "pid_variant": "quantile_tracker_plus_log_integrator_no_scorecaster",
                       "pid_eta": PID_ETA, "pid_csat": PID_CSAT, "pid_ki": PID_KI}
    return window_rows, user_rows, update_rows, pid_update_rows, correction_rows, summary, details, aci_conflicts, pid_conflicts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=tuple(DATASET_NAMES), required=True)
    parser.add_argument("--forecaster", choices=("lgbm", "persistence"), required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--train-rows", type=int, default=600000)
    parser.add_argument("--num-boost-round", type=int, default=250)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-windows", type=int)
    parser.add_argument("--prepared", type=Path, help="Prepared London directory")
    parser.add_argument("--raw-dir", type=Path, help="Ausgrid raw archive directory")
    parser.add_argument("--zip", dest="zip_path", type=Path, help="UCI Electricity ZIP archive")
    args = parser.parse_args()
    started = time.perf_counter()
    out = args.out or HERE / f"aci_{args.forecaster}_{args.dataset}"
    out.mkdir(parents=True, exist_ok=True)
    data = load_data(args.dataset, args.prepared, args.raw_dir, args.zip_path)
    rows, users, updates, pid_updates, corrections, summary, details, aci_conflicts, pid_conflicts = run(data, args.forecaster, args.train_rows, args.num_boost_round, args.threads, args.max_windows)
    summary["data"]["wall_seconds"] = time.perf_counter() - started
    write_csv(out / "window_metrics.csv", rows)
    write_csv(out / "per_user_window_metrics.csv", users)
    write_csv(out / "aci_updates.csv", updates)
    write_csv(out / "pid_updates.csv", pid_updates)
    write_csv(out / "corrections.csv", corrections)
    write_csv(out / "conflict_windows.csv", details["conflict_windows"])
    write_csv(out / "aci_conflict_windows.csv", aci_conflicts)
    write_csv(out / "pid_conflict_windows.csv", pid_conflicts)
    write_csv(out / "rank_reversal_pairs.csv", details["rank_reversal_pairs"])
    write_csv(out / "temporal_cancellation.csv", summary["temporal_cancellation"])
    write_csv(out / "routing_oracle_gap.csv", summary["routing_oracle_gap"])
    (out / "diagnostic_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(json.dumps(summary["data"]))


if __name__ == "__main__":
    main()
