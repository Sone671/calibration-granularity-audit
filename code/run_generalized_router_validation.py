"""Validate CSGR on the frozen persistence-interval robustness grid."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import generalized_granularity_router as router
import run_naive_robustness as benchmark


HERE = Path(__file__).resolve().parent
DATASETS = ("london", "ausgrid", "uci")
FIT_DAYS = (35, 42, 49)
VALIDATION_DAYS = 7
CALIBRATION_DAYS = 56


def log(message: str) -> None:
    print(time.strftime("[%H:%M:%S]"), message, flush=True)


def load_dataset(name: str, args):
    if name == "london":
        return benchmark.load_london(args.prepared)
    if name == "ausgrid":
        return benchmark.load_ausgrid(args.raw_dir)
    if name == "uci":
        return benchmark.load_uci(args.zip)
    raise ValueError(name)


def interval_slice(values, dt, start, stop, lag, low, high):
    start_index = int(dt.searchsorted(start))
    stop_index = int(dt.searchsorted(stop))
    return benchmark.interval_arrays(values, start_index, stop_index, lag, low, high)


def chronological_fold_metrics(values, dt, test_start, lag, low, high, labels, scales, target):
    calibration_start = test_start - pd.Timedelta(days=CALIBRATION_DAYS)
    rows = []
    metric_frames = []
    for fold_index, fit_days in enumerate(FIT_DAYS, start=1):
        fit_stop = calibration_start + pd.Timedelta(days=fit_days)
        validation_stop = fit_stop + pd.Timedelta(days=VALIDATION_DAYS)
        y_fit, qlo_fit, qhi_fit, users_fit = interval_slice(
            values, dt, calibration_start, fit_stop, lag, low, high
        )
        corrections = router.fit_corrections(
            y_fit, qlo_fit, qhi_fit, users_fit, labels, scales, target
        )
        y_val, qlo_val, qhi_val, users_val = interval_slice(
            values, dt, fit_stop, validation_stop, lag, low, high
        )
        fold = {}
        for policy in router.POLICIES:
            metrics = router.evaluate_policy(
                y_val, qlo_val, qhi_val, users_val, labels, scales, corrections[policy], target
            )
            fold[policy] = metrics
            rows.append(
                {
                    "fold": fold_index,
                    "fit_days": fit_days,
                    "validation_start": str(fit_stop),
                    "validation_stop_exclusive": str(validation_stop),
                    "policy": policy,
                    **metrics,
                }
            )
        metric_frames.append(fold)
    return metric_frames, rows


def full_calibration_and_test_metrics(values, dt, test_start, test_stop, lag, low, high, labels, scales, target):
    calibration_start = test_start - pd.Timedelta(days=CALIBRATION_DAYS)
    y_cal, qlo_cal, qhi_cal, users_cal = interval_slice(
        values, dt, calibration_start, test_start, lag, low, high
    )
    corrections = router.fit_corrections(
        y_cal, qlo_cal, qhi_cal, users_cal, labels, scales, target
    )
    y_test, qlo_test, qhi_test, users_test = interval_slice(
        values, dt, test_start, test_stop, lag, low, high
    )
    return {
        policy: router.evaluate_policy(
            y_test, qlo_test, qhi_test, users_test, labels, scales, corrections[policy], target
        )
        for policy in router.POLICIES
    }


def previous_window_loss(frame: pd.DataFrame) -> float:
    frame = frame.sort_values("window_order")
    policies = list(router.POLICIES)
    selected = router.GLOBAL
    losses = []
    for _, row in frame.iterrows():
        losses.append(float(row[f"loss__{selected}"]))
        selected = min(policies, key=lambda policy: float(row[f"loss__{policy}"]))
    return float(np.mean(losses))


def circular_block_sample(values: np.ndarray, block_length: int, rng: np.random.Generator) -> np.ndarray:
    n = len(values)
    starts = rng.integers(0, n, size=int(np.ceil(n / block_length)))
    indices = np.concatenate(
        [(start + np.arange(block_length)) % n for start in starts]
    )[:n]
    return values[indices]


def moving_block_ci(groups: list[np.ndarray], reps: int, seed: int, block_length: int = 3):
    rng = np.random.default_rng(seed)
    estimates = np.empty(reps, dtype=float)
    for repetition in range(reps):
        estimates[repetition] = np.mean(
            np.concatenate([circular_block_sample(group, block_length, rng) for group in groups])
        )
    return np.quantile(estimates, [.025, .975]).tolist()


def synchronized_hierarchical_block_ci(
    frame: pd.DataFrame,
    column: str,
    reps: int,
    seed: int,
    block_length: int = 3,
    resample_datasets: bool = False,
):
    """Respect shared months across coverage/horizon configurations.

    Every configuration from the same dataset receives the same circular-block
    sample of month indices.  The hierarchical variant additionally resamples
    the three datasets and therefore reflects cross-dataset uncertainty, albeit
    with the unavoidable low resolution of only three datasets.
    """
    rng = np.random.default_rng(seed)
    dataset_names = list(frame.dataset.unique())
    matrices = {}
    for dataset, dataset_frame in frame.groupby("dataset", sort=False):
        matrix = dataset_frame.pivot(
            index="window_order", columns="configuration", values=column
        ).sort_index()
        if matrix.isna().any().any():
            raise ValueError(f"unaligned configuration windows for {dataset}")
        matrices[dataset] = matrix.to_numpy(dtype=float)
    estimates = np.empty(reps, dtype=float)
    for repetition in range(reps):
        sampled_datasets = (
            rng.choice(dataset_names, size=len(dataset_names), replace=True)
            if resample_datasets
            else dataset_names
        )
        dataset_estimates = []
        for dataset in sampled_datasets:
            matrix = matrices[dataset]
            n_windows = matrix.shape[0]
            starts = rng.integers(0, n_windows, size=int(np.ceil(n_windows / block_length)))
            indices = np.concatenate(
                [(start + np.arange(block_length)) % n_windows for start in starts]
            )[:n_windows]
            dataset_estimates.append(float(matrix[indices].mean(axis=0).mean()))
        estimates[repetition] = float(np.mean(dataset_estimates))
    return np.quantile(estimates, [.025, .975]).tolist()


def summarize(window_frame: pd.DataFrame, bootstrap_reps: int):
    configuration_rows = []
    keys = ["dataset", "configuration", "coverage", "horizon_hours", "user_weight", "efficiency_weight"]
    for key, frame in window_frame.groupby(keys, sort=False):
        fixed_means = {
            policy: float(frame[f"loss__{policy}"].mean()) for policy in router.POLICIES
        }
        best_fixed = min(fixed_means, key=fixed_means.get)
        selection_counts = frame.selected_policy.value_counts().to_dict()
        configuration_rows.append(
            {
                **dict(zip(keys, key)),
                "windows": len(frame),
                "csgr_mean_loss": float(frame.router_loss.mean()),
                "global_mean_loss": fixed_means[router.GLOBAL],
                "group_mean_loss": fixed_means[router.GROUP],
                "user_mean_loss": fixed_means[router.USER],
                "best_fixed_policy": best_fixed,
                "best_fixed_mean_loss": fixed_means[best_fixed],
                "previous_window_router_mean_loss": previous_window_loss(frame),
                "oracle_mean_loss": float(frame.oracle_loss.mean()),
                "csgr_minus_best_fixed": float(frame.router_loss.mean() - fixed_means[best_fixed]),
                "csgr_minus_global": float(frame.router_loss.mean() - fixed_means[router.GLOBAL]),
                "selected_global": int(selection_counts.get(router.GLOBAL, 0)),
                "selected_group": int(selection_counts.get(router.GROUP, 0)),
                "selected_user": int(selection_counts.get(router.USER, 0)),
            }
        )
    configurations = pd.DataFrame(configuration_rows)

    aggregate_rows = []
    for (user_weight, efficiency_weight), config_frame in configurations.groupby(
        ["user_weight", "efficiency_weight"], sort=True
    ):
        selected_windows = window_frame[
            (window_frame.user_weight == user_weight)
            & (window_frame.efficiency_weight == efficiency_weight)
        ].copy()
        best_by_config = config_frame.set_index(["dataset", "configuration"])["best_fixed_policy"]
        selected_windows["best_fixed_policy"] = [
            best_by_config.loc[(dataset, configuration)]
            for dataset, configuration in zip(selected_windows.dataset, selected_windows.configuration)
        ]
        selected_windows["best_fixed_window_loss"] = [
            row[f"loss__{row.best_fixed_policy}"] for _, row in selected_windows.iterrows()
        ]
        selected_windows["delta_best"] = (
            selected_windows.router_loss - selected_windows.best_fixed_window_loss
        )
        selected_windows["delta_global"] = (
            selected_windows.router_loss - selected_windows[f"loss__{router.GLOBAL}"]
        )
        best_groups = [
            group.sort_values("window_order").delta_best.to_numpy()
            for _, group in selected_windows.groupby(["dataset", "configuration"], sort=False)
        ]
        global_groups = [
            group.sort_values("window_order").delta_global.to_numpy()
            for _, group in selected_windows.groupby(["dataset", "configuration"], sort=False)
        ]
        aggregate_rows.append(
            {
                "user_weight": float(user_weight),
                "efficiency_weight": float(efficiency_weight),
                "configurations": len(config_frame),
                "windows": len(selected_windows),
                "macro_csgr_mean_loss": float(config_frame.csgr_mean_loss.mean()),
                "macro_best_fixed_mean_loss": float(config_frame.best_fixed_mean_loss.mean()),
                "macro_global_mean_loss": float(config_frame.global_mean_loss.mean()),
                "macro_previous_window_router_mean_loss": float(
                    config_frame.previous_window_router_mean_loss.mean()
                ),
                "macro_oracle_mean_loss": float(config_frame.oracle_mean_loss.mean()),
                "macro_csgr_minus_best_fixed": float(config_frame.csgr_minus_best_fixed.mean()),
                "macro_csgr_minus_global": float(config_frame.csgr_minus_global.mean()),
                "configurations_beating_or_tying_best_fixed": int(
                    (config_frame.csgr_minus_best_fixed <= 0).sum()
                ),
                "csgr_minus_best_fixed_ci95_moving_block": moving_block_ci(
                    best_groups, bootstrap_reps, 20260820 + int(user_weight * 1000) + int(efficiency_weight * 10000)
                ),
                "csgr_minus_global_ci95_moving_block": moving_block_ci(
                    global_groups, bootstrap_reps, 20260821 + int(user_weight * 1000) + int(efficiency_weight * 10000)
                ),
                "csgr_minus_best_fixed_ci95_synchronized_block": synchronized_hierarchical_block_ci(
                    selected_windows,
                    "delta_best",
                    bootstrap_reps,
                    20260822 + int(user_weight * 1000) + int(efficiency_weight * 10000),
                ),
                "csgr_minus_global_ci95_synchronized_block": synchronized_hierarchical_block_ci(
                    selected_windows,
                    "delta_global",
                    bootstrap_reps,
                    20260823 + int(user_weight * 1000) + int(efficiency_weight * 10000),
                ),
                "csgr_minus_best_fixed_ci95_hierarchical_dataset_block": synchronized_hierarchical_block_ci(
                    selected_windows,
                    "delta_best",
                    bootstrap_reps,
                    20260824 + int(user_weight * 1000) + int(efficiency_weight * 10000),
                    resample_datasets=True,
                ),
                "csgr_minus_global_ci95_hierarchical_dataset_block": synchronized_hierarchical_block_ci(
                    selected_windows,
                    "delta_global",
                    bootstrap_reps,
                    20260825 + int(user_weight * 1000) + int(efficiency_weight * 10000),
                    resample_datasets=True,
                ),
            }
        )
    return configurations, aggregate_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--out", type=Path, default=HERE / "generalized_router_validation")
    parser.add_argument("--prepared", type=Path, default=benchmark.LONDON)
    parser.add_argument("--raw-dir", type=Path, default=benchmark.EXT / "raw_archive")
    parser.add_argument("--zip", type=Path, default=benchmark.V1 / "electricityloaddiagrams20112014.originalmirror.zip")
    parser.add_argument("--coverages", nargs="+", type=float, default=[.8, .9])
    parser.add_argument("--horizons", nargs="+", type=float, default=[1., 6.])
    parser.add_argument("--user-weights", nargs="+", type=float, default=[0., .25, .5, .75, 1.])
    parser.add_argument("--efficiency-weights", nargs="+", type=float, default=[0., .01])
    parser.add_argument("--max-windows", type=int)
    parser.add_argument("--bootstrap-reps", type=int, default=5000)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    result_rows = []
    fold_rows = []
    for dataset in args.datasets:
        dt, values, names, labels, scales, train0, train1, windows, cadence = load_dataset(dataset, args)
        log(f"Loaded {benchmark.DATASET_NAMES[dataset]}: users={len(names)}, windows={len(windows)}")
        selected_windows = windows if args.max_windows is None else windows[: args.max_windows]
        for coverage in args.coverages:
            for horizon in args.horizons:
                lag = int(round(horizon * 60 / cadence))
                low, high = benchmark.residual_quantiles(values, train0, train1, lag, coverage)
                configuration = f"persistence_qi__c{int(coverage * 100)}__h{horizon:g}"
                for window_order, (window, test_start, test_stop) in enumerate(selected_windows):
                    log(
                        f"{benchmark.DATASET_NAMES[dataset]} c={coverage:.2f} h={horizon:g} {window}"
                    )
                    folds, raw_fold_rows = chronological_fold_metrics(
                        values, dt, test_start, lag, low, high, labels, scales, coverage
                    )
                    for row in raw_fold_rows:
                        fold_rows.append(
                            {
                                "dataset": benchmark.DATASET_NAMES[dataset],
                                "configuration": configuration,
                                "coverage": coverage,
                                "horizon_hours": horizon,
                                "window": window,
                                "window_order": window_order,
                                **row,
                            }
                        )
                    test_metrics = full_calibration_and_test_metrics(
                        values, dt, test_start, test_stop, lag, low, high, labels, scales, coverage
                    )
                    for user_weight in args.user_weights:
                        for efficiency_weight in args.efficiency_weights:
                            decision = router.select_policy(
                                folds,
                                user_weight=user_weight,
                                efficiency_weight=efficiency_weight,
                            )
                            losses = {
                                policy: router.scalar_loss(
                                    test_metrics[policy], user_weight, efficiency_weight
                                )
                                for policy in router.POLICIES
                            }
                            selected_metrics = test_metrics[decision.selected_policy]
                            result_rows.append(
                                {
                                    "dataset": benchmark.DATASET_NAMES[dataset],
                                    "configuration": configuration,
                                    "coverage": coverage,
                                    "horizon_hours": horizon,
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

    result_frame = pd.DataFrame(result_rows)
    fold_frame = pd.DataFrame(fold_rows)
    configurations, aggregate = summarize(result_frame, args.bootstrap_reps)
    result_frame.to_csv(args.out / "router_window_results.csv", index=False, encoding="utf-8-sig")
    fold_frame.to_csv(args.out / "chronological_fold_metrics.csv", index=False, encoding="utf-8-sig")
    configurations.to_csv(
        args.out / "router_configuration_summary.csv", index=False, encoding="utf-8-sig"
    )
    summary = {
        "method": "Cross-Fitted Stability-Screened Granularity Router (CSGR)",
        "datasets": [benchmark.DATASET_NAMES[name] for name in args.datasets],
        "fit_days": list(FIT_DAYS),
        "validation_days": VALIDATION_DAYS,
        "calibration_days": CALIBRATION_DAYS,
        "environment_rows": int(
            result_frame[["dataset", "configuration", "window"]].drop_duplicates().shape[0]
        ),
        "decision_rows": len(result_frame),
        "bootstrap_reps": args.bootstrap_reps,
        "wall_seconds": time.perf_counter() - started,
        "aggregate": aggregate,
    }
    (args.out / "router_validation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    log(json.dumps({"environment_rows": summary["environment_rows"], "wall_seconds": summary["wall_seconds"]}))


if __name__ == "__main__":
    main()
