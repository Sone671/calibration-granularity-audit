"""External validation of the frozen CSGR rule with LightGBM quantile models."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import generalized_granularity_router as router
import run_generalized_router_validation as common
import run_naive_robustness as persistence


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
EXTENSION = ROOT / "probabilistic_load_hierarchical_rolling_cqr_external_2026-07-30"
V1 = ROOT / "probabilistic_load_group_cqr_go_nogo_2026-07-25"
for dependency_path in (EXTENSION, V1 / ".deps", V1):
    sys.path.insert(0, str(dependency_path))

import run_external as external  # noqa: E402
import run_validation as base  # noqa: E402


DATASETS = ("london", "ausgrid", "uci")
TARGET = 0.8
HORIZON_HOURS = 1.0


def log(message: str) -> None:
    print(time.strftime("[%H:%M:%S]"), message, flush=True)


def external_rows_with_time(values, dt, prefix, origins, users, labels, stat):
    """Exact external feature builder with the target time index retained."""
    x = np.empty((len(origins), len(external.FEATURE_NAMES)), dtype=np.float32)
    for column, lag in enumerate((1, 2, 3, 48, 96, 336)):
        x[:, column] = values[origins - lag, users]
    x[:, 6], x[:, 7] = external.rolling_moments(prefix, origins, users, 48)
    x[:, 8], x[:, 9] = external.rolling_moments(prefix, origins, users, 336)
    target_indices = origins + external.HORIZON
    target_dt = dt[target_indices]
    minute = target_dt.hour.to_numpy() * 60 + target_dt.minute.to_numpy()
    day_of_week = target_dt.dayofweek.to_numpy()
    month = target_dt.month.to_numpy()
    x[:, 10], x[:, 11] = np.sin(2 * np.pi * minute / 1440), np.cos(2 * np.pi * minute / 1440)
    x[:, 12], x[:, 13] = (
        np.sin(2 * np.pi * day_of_week / 7),
        np.cos(2 * np.pi * day_of_week / 7),
    )
    x[:, 14] = day_of_week >= 5
    x[:, 15], x[:, 16] = (
        np.sin(2 * np.pi * (month - 1) / 12),
        np.cos(2 * np.pi * (month - 1) / 12),
    )
    x[:, 17], x[:, 18] = users, labels[users]
    x[:, 19:23] = stat[users]
    y = values[target_indices, users].astype(np.float32, copy=False)
    valid = np.isfinite(x).all(axis=1) & np.isfinite(y)
    return (
        x[valid],
        y[valid],
        users[valid].astype(np.int32),
        labels[users[valid]].astype(np.int32),
        target_indices[valid].astype(np.int32),
    )


def uci_rows_with_time(values, dt, origins, users, labels, stat):
    x, y, retained_users, groups, target_indices = base.build_rows(
        values, dt, origins, users, labels, stat
    )
    valid = np.isfinite(x).all(axis=1) & np.isfinite(y)
    return x[valid], y[valid], retained_users[valid], groups[valid], target_indices[valid]


def load_dataset(name: str, args):
    if name == "london":
        data = persistence.load_london(args.prepared)
    elif name == "ausgrid":
        data = persistence.load_ausgrid(args.raw_dir)
    elif name == "uci":
        data = persistence.load_uci(args.zip)
    else:
        raise ValueError(name)
    dt, values, names, labels, scales, train0, train1, windows, cadence = data
    training = values[train0:train1]
    stat = np.column_stack(
        [
            np.nanmean(training, axis=0),
            np.nanstd(training, axis=0),
            np.nanquantile(training, .95, axis=0),
            np.nanmean(training <= 1e-6, axis=0),
        ]
    ).astype(np.float32)
    prefix = external.make_prefix(values) if name != "uci" else None
    return dt, values, names, labels, scales, train0, train1, windows, cadence, stat, prefix


def build_range(name, values, dt, labels, stat, prefix, start, stop):
    n_users = len(labels)
    if name == "uci":
        origins, users = base.full_pairs(start, stop, n_users)
        return uci_rows_with_time(values, dt, origins, users, labels, stat)
    origins, users = external.full_pairs(start, stop, n_users)
    return external_rows_with_time(values, dt, prefix, origins, users, labels, stat)


def train_models(name, values, dt, labels, stat, prefix, train0, train1, args):
    n_users = len(labels)
    if name == "uci":
        origins, users = base.sampled_pairs(train0, train1, n_users, args.train_rows, base.SEED)
        x, y, _, _, _ = uci_rows_with_time(values, dt, origins, users, labels, stat)
        feature_names = base.FEATURE_NAMES
        seed = base.SEED
    else:
        origins, users = external.sampled_pairs(
            train0, train1, n_users, args.train_rows, external.SEED
        )
        x, y, _, _, _ = external_rows_with_time(
            values, dt, prefix, origins, users, labels, stat
        )
        feature_names = external.FEATURE_NAMES
        seed = external.SEED
    log(f"{persistence.DATASET_NAMES[name]} training rows={len(y):,}")
    parameters = {
        "verbosity": -1,
        "learning_rate": .05,
        "num_leaves": 31,
        "min_data_in_leaf": 100,
        "feature_fraction": .9,
        "bagging_fraction": .9,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "seed": seed,
        "num_threads": args.threads,
        "force_col_wise": True,
    }
    dataset = base.lgb.Dataset(
        x,
        label=y,
        feature_name=feature_names,
        categorical_feature=[17, 18],
        free_raw_data=False,
    )
    models = {}
    for quantile in (.1, .5, .9):
        log(f"{persistence.DATASET_NAMES[name]} training quantile={quantile}")
        models[quantile] = base.lgb.train(
            dict(parameters, objective="quantile", alpha=quantile, metric="quantile"),
            dataset,
            num_boost_round=args.num_boost_round,
        )
    del dataset, x, y
    gc.collect()
    return models


def predict_range(name, values, dt, labels, stat, prefix, models, start, stop):
    x, y, users, groups, target_indices = build_range(
        name, values, dt, labels, stat, prefix, start, stop
    )
    qlo, _, qhi, _ = base.predict_three(models, x)
    del x
    return y, qlo, qhi, users, groups, target_indices


def fold_metrics_from_calibration(
    y, qlo, qhi, users, target_indices, dt, labels, scales, calibration_start
):
    folds = []
    output_rows = []
    for fold_index, fit_days in enumerate(common.FIT_DAYS, start=1):
        fit_stop = calibration_start + pd.Timedelta(days=fit_days)
        validation_stop = fit_stop + pd.Timedelta(days=common.VALIDATION_DAYS)
        fit_boundary = int(dt.searchsorted(fit_stop))
        validation_boundary = int(dt.searchsorted(validation_stop))
        fit_mask = target_indices < fit_boundary
        validation_mask = (target_indices >= fit_boundary) & (target_indices < validation_boundary)
        corrections = router.fit_corrections(
            y[fit_mask], qlo[fit_mask], qhi[fit_mask], users[fit_mask], labels, scales, TARGET
        )
        fold = {}
        for policy in router.POLICIES:
            metrics = router.evaluate_policy(
                y[validation_mask],
                qlo[validation_mask],
                qhi[validation_mask],
                users[validation_mask],
                labels,
                scales,
                corrections[policy],
                TARGET,
            )
            fold[policy] = metrics
            output_rows.append(
                {
                    "fold": fold_index,
                    "fit_days": fit_days,
                    "policy": policy,
                    **metrics,
                }
            )
        folds.append(fold)
    return folds, output_rows


def evaluate_window(name, values, dt, labels, scales, stat, prefix, models, window, test_start, test_stop):
    calibration_start = test_start - pd.Timedelta(days=common.CALIBRATION_DAYS)
    calibration0 = int(dt.searchsorted(calibration_start))
    test0, test1 = int(dt.searchsorted(test_start)), int(dt.searchsorted(test_stop))
    y, qlo, qhi, users, _, target_indices = predict_range(
        name, values, dt, labels, stat, prefix, models, calibration0, test0
    )
    folds, fold_rows = fold_metrics_from_calibration(
        y, qlo, qhi, users, target_indices, dt, labels, scales, calibration_start
    )
    full_corrections = router.fit_corrections(y, qlo, qhi, users, labels, scales, TARGET)
    del y, qlo, qhi, users, target_indices
    y, qlo, qhi, users, _, _ = predict_range(
        name, values, dt, labels, stat, prefix, models, test0, test1
    )
    test_metrics = {
        policy: router.evaluate_policy(
            y, qlo, qhi, users, labels, scales, full_corrections[policy], TARGET
        )
        for policy in router.POLICIES
    }
    del y, qlo, qhi, users
    gc.collect()
    return folds, fold_rows, test_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--out", type=Path, default=HERE / "lightgbm_router_validation")
    parser.add_argument("--prepared", type=Path, default=persistence.LONDON)
    parser.add_argument("--raw-dir", type=Path, default=persistence.EXT / "raw_archive")
    parser.add_argument("--zip", type=Path, default=persistence.V1 / "electricityloaddiagrams20112014.originalmirror.zip")
    parser.add_argument("--train-rows", type=int, default=600000)
    parser.add_argument("--num-boost-round", type=int, default=250)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-windows", type=int)
    parser.add_argument("--user-weights", nargs="+", type=float, default=[0., .25, .5, .75, 1.])
    parser.add_argument("--efficiency-weights", nargs="+", type=float, default=[0., .01])
    parser.add_argument("--bootstrap-reps", type=int, default=5000)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    result_rows = []
    all_fold_rows = []
    configuration = "lightgbm_quantile__c80__h1"
    for name in args.datasets:
        (
            dt,
            values,
            names,
            labels,
            scales,
            train0,
            train1,
            windows,
            _,
            stat,
            prefix,
        ) = load_dataset(name, args)
        log(f"Loaded {persistence.DATASET_NAMES[name]} users={len(names)} windows={len(windows)}")
        models = train_models(name, values, dt, labels, stat, prefix, train0, train1, args)
        selected_windows = windows if args.max_windows is None else windows[: args.max_windows]
        for window_order, (window, test_start, test_stop) in enumerate(selected_windows):
            log(f"{persistence.DATASET_NAMES[name]} LightGBM {window}")
            folds, fold_rows, test_metrics = evaluate_window(
                name,
                values,
                dt,
                labels,
                scales,
                stat,
                prefix,
                models,
                window,
                test_start,
                test_stop,
            )
            for row in fold_rows:
                all_fold_rows.append(
                    {
                        "dataset": persistence.DATASET_NAMES[name],
                        "configuration": configuration,
                        "coverage": TARGET,
                        "horizon_hours": HORIZON_HOURS,
                        "window": window,
                        "window_order": window_order,
                        **row,
                    }
                )
            for user_weight in args.user_weights:
                for efficiency_weight in args.efficiency_weights:
                    decision = router.select_policy(folds, user_weight, efficiency_weight)
                    losses = {
                        policy: router.scalar_loss(test_metrics[policy], user_weight, efficiency_weight)
                        for policy in router.POLICIES
                    }
                    selected_metrics = test_metrics[decision.selected_policy]
                    result_rows.append(
                        {
                            "dataset": persistence.DATASET_NAMES[name],
                            "configuration": configuration,
                            "coverage": TARGET,
                            "horizon_hours": HORIZON_HOURS,
                            "window": window,
                            "window_order": window_order,
                            "user_weight": user_weight,
                            "efficiency_weight": efficiency_weight,
                            "selected_policy": decision.selected_policy,
                            "router_loss": losses[decision.selected_policy],
                            "oracle_loss": min(losses.values()),
                            "oracle_policy": min(losses, key=losses.get),
                            "router_macro_user_abs_coverage_gap": selected_metrics[
                                "macro_user_abs_coverage_gap"
                            ],
                            "router_max_abs_cluster_coverage_gap": selected_metrics[
                                "max_abs_cluster_coverage_gap"
                            ],
                            "router_normalized_interval_score": selected_metrics[
                                "normalized_interval_score"
                            ],
                            **{f"loss__{policy}": loss for policy, loss in losses.items()},
                            **{
                                f"cv_mean_gain_over_global__{policy}": decision.mean_gains_over_global[
                                    policy
                                ]
                                for policy in router.POLICIES
                            },
                            **{
                                f"cv_lower_gain__{policy}": decision.lower_confidence_gains[policy]
                                for policy in router.POLICIES
                            },
                        }
                    )
        del models, values, prefix
        gc.collect()

    result_frame = pd.DataFrame(result_rows)
    fold_frame = pd.DataFrame(all_fold_rows)
    configurations, aggregate = common.summarize(result_frame, args.bootstrap_reps)
    result_frame.to_csv(args.out / "router_window_results.csv", index=False, encoding="utf-8-sig")
    fold_frame.to_csv(args.out / "chronological_fold_metrics.csv", index=False, encoding="utf-8-sig")
    configurations.to_csv(
        args.out / "router_configuration_summary.csv", index=False, encoding="utf-8-sig"
    )
    summary = {
        "method": "Cross-Fitted Stability-Screened Granularity Router (CSGR)",
        "forecaster": "LightGBM quantile",
        "datasets": [persistence.DATASET_NAMES[name] for name in args.datasets],
        "coverage": TARGET,
        "horizon_hours": HORIZON_HOURS,
        "environment_rows": int(
            result_frame[["dataset", "configuration", "window"]].drop_duplicates().shape[0]
        ),
        "decision_rows": len(result_frame),
        "aggregate": aggregate,
        "wall_seconds": time.perf_counter() - started,
    }
    (args.out / "router_validation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    log(json.dumps({"environment_rows": summary["environment_rows"], "wall_seconds": summary["wall_seconds"]}))


if __name__ == "__main__":
    main()
