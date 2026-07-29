"""Strictly-forward 14-day ERW-CQR transfer on the LightGBM 80%/1 h main configuration."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import run_lightgbm_full_grid as lgbm
import run_weighted_conformal_baselines as erw


HERE = Path(__file__).resolve().parent
COVERAGE, HORIZON, HALF_LIFE_DAYS = .8, 1., 14.


def run_dataset(name: str, args) -> tuple[list[dict], list[dict]]:
    data = lgbm.load_data(name, args)
    step = lgbm.horizon_steps(data["cadence"], HORIZON)
    args.current_horizon = HORIZON
    models = lgbm.train_models(data, step, args)
    window_rows, user_rows = [], []
    for window, start, stop in data["windows"]:
        lgbm.log(f"ERW transfer {lgbm.naive.DATASET_NAMES[name]} window={window}")
        cal0, test0, test1 = (
            int(data["dt"].searchsorted(value))
            for value in (start - pd.Timedelta(days=lgbm.CALIBRATION_DAYS), start, stop)
        )
        y, qlo, qhi, users, _, observed_times = lgbm.predict(
            data, models, cal0, test0, step, COVERAGE
        )
        corrections, weight_diag = erw.calibration_corrections(
            y, qlo, qhi, users, observed_times, start, data["labels"], data["scales"],
            COVERAGE, HALF_LIFE_DAYS,
        )
        y, qlo, qhi, users, _, _ = lgbm.predict(data, models, test0, test1, step, COVERAGE)
        for method in erw.METHODS:
            metrics, user_cov, user_n = lgbm.naive.evaluate(
                y, qlo, qhi, users, data["labels"], data["scales"], corrections[method], COVERAGE
            )
            row = {
                "dataset": lgbm.naive.DATASET_NAMES[name],
                "configuration": "lightgbm_quantile__c80__h1",
                "window": window, "forecaster": "lightgbm_quantile",
                "coverage": COVERAGE, "horizon_hours": HORIZON, "method": method,
                "half_life_days": HALF_LIFE_DAYS if method.startswith("erw_") else "",
                **metrics, **weight_diag,
            }
            window_rows.append(row)
            for user in range(len(data["labels"])):
                user_rows.append({
                    "dataset": lgbm.naive.DATASET_NAMES[name],
                    "configuration": row["configuration"], "window": window,
                    "forecaster": "lightgbm_quantile", "coverage_target": COVERAGE,
                    "horizon_hours": HORIZON, "method": method,
                    "user_index": user, "customer": data["names"][user],
                    "cluster": int(data["labels"][user]), "coverage": float(user_cov[user]),
                    "n": int(user_n[user]), "coverage_gap": float(abs(user_cov[user] - COVERAGE)),
                })
        del y, qlo, qhi, users
        gc.collect()
    del models, data
    gc.collect()
    return window_rows, user_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=lgbm.DATASETS, default=list(lgbm.DATASETS))
    parser.add_argument("--out", type=Path, default=HERE / "lightgbm_erw_80_1h")
    parser.add_argument("--prepared", type=Path, default=lgbm.naive.LONDON)
    parser.add_argument("--raw-dir", type=Path, default=lgbm.EXT / "raw_archive")
    parser.add_argument("--zip", type=Path, default=lgbm.V1 / "electricityloaddiagrams20112014.originalmirror.zip")
    parser.add_argument("--train-rows", type=int, default=600000)
    parser.add_argument("--num-boost-round", type=int, default=250)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--segments", type=int, default=3)
    parser.add_argument("--cluster-method", choices=("kmeans", "ward"), default="kmeans")
    args = parser.parse_args()
    args.coverages = [COVERAGE]
    args.horizons = [HORIZON]
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    windows, users = [], []
    for name in args.datasets:
        new_windows, new_users = run_dataset(name, args)
        windows.extend(new_windows); users.extend(new_users)
    expected = sum(11 if name == "london" else 12 for name in args.datasets)
    if len(windows) != expected * len(erw.METHODS):
        raise RuntimeError(f"incomplete LightGBM ERW transfer: {len(windows)} rows")
    lgbm.write_csv(args.out / "window_metrics.csv", windows)
    lgbm.write_csv(args.out / "per_user_window_metrics.csv", users)
    lgbm.write_csv(args.out / "summary.csv", erw.summarize(windows))
    metadata = {
        "protocol": "FROZEN_ERW_SENSITIVITY_ADDENDUM.md", "datasets": args.datasets,
        "coverage": COVERAGE, "horizon_hours": HORIZON, "half_life_days": HALF_LIFE_DAYS,
        "train_rows": args.train_rows, "num_boost_round": args.num_boost_round,
        "strict_forward": True, "wall_seconds": time.perf_counter() - started,
    }
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
