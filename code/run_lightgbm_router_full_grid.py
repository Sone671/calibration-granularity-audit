"""Run the prespecified CSGR rule on the complete LightGBM coverage/horizon grid.

The experiment follows FROZEN_LIGHTGBM_ROUTER_FULL_GRID_ADDENDUM_2026-08-02.md.
It reuses the target-matched LightGBM construction from run_lightgbm_full_grid
and the CSGR implementation from generalized_granularity_router without
retuning either component.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import generalized_granularity_router as router
import run_generalized_router_validation as common
import run_lightgbm_full_grid as grid
import run_naive_robustness as benchmark


HERE = Path(__file__).resolve().parent
REPOSITORY_LAYOUT = (HERE.parent / "protocols").is_dir()
OUTPUT_ROOT = HERE.parent / "outputs" if REPOSITORY_LAYOUT else HERE
PROTOCOL = "FROZEN_LIGHTGBM_ROUTER_FULL_GRID_ADDENDUM_2026-08-02.md"
DATASETS = ("london", "ausgrid", "uci")


def log(message: str) -> None:
    print(time.strftime("[%H:%M:%S]"), message, flush=True)


def fold_metrics_from_calibration(
    y: np.ndarray,
    qlo: np.ndarray,
    qhi: np.ndarray,
    users: np.ndarray,
    target_times: np.ndarray,
    labels: np.ndarray,
    scales: np.ndarray,
    calibration_start: pd.Timestamp,
    target: float,
) -> tuple[list[dict], list[dict]]:
    """Fit and evaluate the three prespecified chronological pseudo-future folds."""
    folds: list[dict] = []
    rows: list[dict] = []
    for fold_index, fit_days in enumerate(common.FIT_DAYS, start=1):
        fit_stop = calibration_start + pd.Timedelta(days=fit_days)
        validation_stop = fit_stop + pd.Timedelta(days=common.VALIDATION_DAYS)
        fit_boundary = np.datetime64(fit_stop.to_datetime64())
        validation_boundary = np.datetime64(validation_stop.to_datetime64())
        fit_mask = target_times < fit_boundary
        validation_mask = (target_times >= fit_boundary) & (target_times < validation_boundary)
        if not fit_mask.any() or not validation_mask.any():
            raise RuntimeError(
                f"empty chronological fold {fold_index}: "
                f"fit={int(fit_mask.sum())}, validation={int(validation_mask.sum())}"
            )
        corrections = router.fit_corrections(
            y[fit_mask],
            qlo[fit_mask],
            qhi[fit_mask],
            users[fit_mask],
            labels,
            scales,
            target,
        )
        fold: dict[str, dict] = {}
        for policy in router.POLICIES:
            metrics = router.evaluate_policy(
                y[validation_mask],
                qlo[validation_mask],
                qhi[validation_mask],
                users[validation_mask],
                labels,
                scales,
                corrections[policy],
                target,
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
        folds.append(fold)
    return folds, rows


def evaluate_window(
    data: dict,
    models: dict,
    coverage: float,
    horizon: float,
    test_start: pd.Timestamp,
    test_stop: pd.Timestamp,
) -> tuple[list[dict], list[dict], dict[str, dict]]:
    step = grid.horizon_steps(data["cadence"], horizon)
    calibration_start = test_start - pd.Timedelta(days=common.CALIBRATION_DAYS)
    calibration0, test0, test1 = (
        int(data["dt"].searchsorted(value))
        for value in (calibration_start, test_start, test_stop)
    )

    y, qlo, qhi, users, _, target_times = grid.predict(
        data, models, calibration0, test0, step, coverage
    )
    folds, fold_rows = fold_metrics_from_calibration(
        y,
        qlo,
        qhi,
        users,
        target_times,
        data["labels"],
        data["scales"],
        calibration_start,
        coverage,
    )
    corrections = router.fit_corrections(
        y, qlo, qhi, users, data["labels"], data["scales"], coverage
    )
    del y, qlo, qhi, users, target_times

    y, qlo, qhi, users, _, _ = grid.predict(
        data, models, test0, test1, step, coverage
    )
    test_metrics = {
        policy: router.evaluate_policy(
            y,
            qlo,
            qhi,
            users,
            data["labels"],
            data["scales"],
            corrections[policy],
            coverage,
        )
        for policy in router.POLICIES
    }
    del y, qlo, qhi, users
    gc.collect()
    return folds, fold_rows, test_metrics


def append_decisions(
    result_rows: list[dict],
    folds: list[dict],
    test_metrics: dict[str, dict],
    identity: dict,
    user_weights: list[float],
    efficiency_weights: list[float],
) -> None:
    for user_weight in user_weights:
        for efficiency_weight in efficiency_weights:
            decision = router.select_policy(folds, user_weight, efficiency_weight)
            losses = {
                policy: router.scalar_loss(
                    test_metrics[policy], user_weight, efficiency_weight
                )
                for policy in router.POLICIES
            }
            selected_metrics = test_metrics[decision.selected_policy]
            result_rows.append(
                {
                    **identity,
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
                    **{f"loss__{policy}": value for policy, value in losses.items()},
                    **{
                        f"cv_mean_gain_over_global__{policy}": (
                            decision.mean_gains_over_global[policy]
                        )
                        for policy in router.POLICIES
                    },
                    **{
                        f"cv_lower_gain__{policy}": decision.lower_confidence_gains[policy]
                        for policy in router.POLICIES
                    },
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument(
        "--out", type=Path, default=OUTPUT_ROOT / "lightgbm_router_full_grid_2026-08-02"
    )
    parser.add_argument("--prepared", type=Path, default=benchmark.LONDON)
    parser.add_argument("--raw-dir", type=Path, default=benchmark.EXT / "raw_archive")
    parser.add_argument(
        "--zip",
        type=Path,
        default=benchmark.V1 / "electricityloaddiagrams20112014.originalmirror.zip",
    )
    parser.add_argument("--coverages", nargs="+", type=float, default=[0.8, 0.9])
    parser.add_argument("--horizons", nargs="+", type=float, default=[1.0, 6.0])
    parser.add_argument("--train-rows", type=int, default=600_000)
    parser.add_argument("--num-boost-round", type=int, default=250)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--segments", type=int, default=3)
    parser.add_argument("--cluster-method", choices=("kmeans", "ward"), default="kmeans")
    parser.add_argument("--max-windows", type=int)
    parser.add_argument("--user-weights", nargs="+", type=float, default=[0, 0.25, 0.5, 0.75, 1])
    parser.add_argument("--efficiency-weights", nargs="+", type=float, default=[0, 0.01])
    parser.add_argument("--bootstrap-reps", type=int, default=5_000)
    args = parser.parse_args()
    if any(not 0 < value < 1 for value in args.coverages):
        parser.error("coverage must be in (0, 1)")
    args.out.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    result_rows: list[dict] = []
    all_fold_rows: list[dict] = []
    data_metadata: list[dict] = []

    for name in args.datasets:
        data = grid.load_data(name, args)
        windows = data["windows"] if args.max_windows is None else data["windows"][: args.max_windows]
        data_metadata.append(
            {
                "dataset": benchmark.DATASET_NAMES[name],
                "users": len(data["names"]),
                "windows": [item[0] for item in windows],
                "cadence_minutes": data["cadence"],
                "cluster_sizes": np.bincount(
                    data["labels"], minlength=args.segments
                ).tolist(),
            }
        )
        log(
            f"Loaded {benchmark.DATASET_NAMES[name]} users={len(data['names'])} "
            f"windows={len(windows)}"
        )
        for horizon in args.horizons:
            args.current_horizon = horizon
            step = grid.horizon_steps(data["cadence"], horizon)
            models = grid.train_models(data, step, args)
            for coverage in args.coverages:
                configuration = (
                    f"lightgbm_quantile__c{int(round(coverage * 100))}__h{horizon:g}"
                )
                for window_order, (window, test_start, test_stop) in enumerate(windows):
                    log(
                        f"{benchmark.DATASET_NAMES[name]} c={coverage:.2f} "
                        f"h={horizon:g} window={window}"
                    )
                    folds, raw_fold_rows, test_metrics = evaluate_window(
                        data, models, coverage, horizon, test_start, test_stop
                    )
                    identity = {
                        "dataset": benchmark.DATASET_NAMES[name],
                        "configuration": configuration,
                        "coverage": coverage,
                        "horizon_hours": horizon,
                        "window": window,
                        "window_order": window_order,
                    }
                    for row in raw_fold_rows:
                        all_fold_rows.append({**identity, **row})
                    append_decisions(
                        result_rows,
                        folds,
                        test_metrics,
                        identity,
                        args.user_weights,
                        args.efficiency_weights,
                    )
            del models
            gc.collect()
        del data
        gc.collect()

    result_frame = pd.DataFrame(result_rows)
    fold_frame = pd.DataFrame(all_fold_rows)
    configurations, aggregate = common.summarize(result_frame, args.bootstrap_reps)
    result_frame.to_csv(
        args.out / "router_window_results.csv", index=False, encoding="utf-8-sig"
    )
    fold_frame.to_csv(
        args.out / "chronological_fold_metrics.csv", index=False, encoding="utf-8-sig"
    )
    configurations.to_csv(
        args.out / "router_configuration_summary.csv", index=False, encoding="utf-8-sig"
    )
    summary = {
        "protocol": PROTOCOL,
        "method": "Chronological Stability-Screened Granularity Router (CSGR)",
        "forecaster": "LightGBM quantile",
        "datasets": data_metadata,
        "coverages": args.coverages,
        "horizons_hours": args.horizons,
        "fit_days": list(common.FIT_DAYS),
        "validation_days": common.VALIDATION_DAYS,
        "calibration_days": common.CALIBRATION_DAYS,
        "train_rows": args.train_rows,
        "num_boost_round": args.num_boost_round,
        "threads": args.threads,
        "environment_rows": int(
            result_frame[["dataset", "configuration", "window"]]
            .drop_duplicates()
            .shape[0]
        ),
        "decision_rows": len(result_frame),
        "fold_metric_rows": len(fold_frame),
        "bootstrap_reps": args.bootstrap_reps,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "wall_seconds": time.perf_counter() - started,
        "aggregate": aggregate,
    }
    (args.out / "router_validation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    log(
        json.dumps(
            {
                "environment_rows": summary["environment_rows"],
                "decision_rows": summary["decision_rows"],
                "wall_seconds": summary["wall_seconds"],
            }
        )
    )


if __name__ == "__main__":
    main()
