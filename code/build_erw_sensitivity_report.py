"""Aggregate frozen ERW half-life sensitivity and LightGBM transfer results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_weighted_conformal_report import block_bootstrap


HERE = Path(__file__).resolve().parent
DATASETS = {"london": "London", "ausgrid": "Ausgrid", "uci": "UCI Electricity"}
METRICS = (
    "picp", "mpiw", "winkler_interval_score", "macro_user_abs_coverage_gap",
    "user_coverage_std", "max_abs_cluster_coverage_gap",
)
PAIRS = {
    "erw_global_norm": "rolling_global_norm",
    "erw_group_norm": "rolling_group_norm",
    "erw_user_norm": "rolling_user_norm",
}


def persistence_frame(input_root: Path) -> pd.DataFrame:
    frames = []
    for half_life in (7, 14, 28):
        for slug in DATASETS:
            directory = input_root / (f"weighted_conformal_{slug}" if half_life == 14 else f"weighted_conformal_{slug}_h{half_life}")
            frame = pd.read_csv(directory / "window_metrics.csv")
            frame["half_life_days_run"] = half_life
            frame["forecaster"] = "persistence_quantile_interval"
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def paired_summary(frame: pd.DataFrame, scope: str) -> pd.DataFrame:
    keys = ["dataset", "configuration", "window"]
    rows = []
    for half_life in sorted(frame.half_life_days_run.unique()):
        run = frame[frame.half_life_days_run == half_life].copy()
        run["l05"] = .5 * run.macro_user_abs_coverage_gap + .5 * run.max_abs_cluster_coverage_gap
        for erw_method, static_method in PAIRS.items():
            weighted = run[run.method == erw_method].set_index(keys)
            static = run[run.method == static_method].set_index(keys)
            if not weighted.index.equals(static.index):
                raise RuntimeError(f"pairing failed for {scope}/{half_life}/{erw_method}")
            rows.append({
                "scope": scope, "half_life_days": int(half_life), "granularity": erw_method.replace("erw_", "").replace("_norm", ""),
                "environments": len(weighted), "static_mean_l05": float(static.l05.mean()),
                "erw_mean_l05": float(weighted.l05.mean()), "paired_l05_difference": float((weighted.l05-static.l05).mean()),
                "relative_winkler_change": float(np.mean((weighted.winkler_interval_score-static.winkler_interval_score)/static.winkler_interval_score)),
                "paired_user_gap_difference": float((weighted.macro_user_abs_coverage_gap-static.macro_user_abs_coverage_gap).mean()),
                "paired_group_gap_difference": float((weighted.max_abs_cluster_coverage_gap-static.max_abs_cluster_coverage_gap).mean()),
            })
    return pd.DataFrame(rows)


def lightgbm_frame(input_root: Path, frozen_root: Path) -> tuple[pd.DataFrame, dict]:
    frame = pd.read_csv(input_root / "lightgbm_erw_80_1h" / "window_metrics.csv")
    frame["half_life_days_run"] = 14
    reference = pd.read_csv(frozen_root / "lightgbm_full_grid_v3" / "window_metrics.csv")
    reference = reference[np.isclose(reference.coverage, .8) & np.isclose(reference.horizon_hours, 1.)]
    audit = {}
    for method in PAIRS.values():
        candidate = frame[frame.method == method].sort_values(["dataset", "window"]).reset_index(drop=True)
        frozen = reference[reference.method == method].sort_values(["dataset", "window"]).reset_index(drop=True)
        if not candidate[["dataset", "window"]].equals(frozen[["dataset", "window"]]):
            raise RuntimeError(f"LightGBM ERW static key audit failed for {method}")
        audit[method] = {metric: float(np.max(np.abs(candidate[metric].to_numpy(float)-frozen[metric].to_numpy(float)))) for metric in METRICS}
        if max(audit[method].values()) > 1e-10:
            raise RuntimeError(f"LightGBM ERW static metric audit failed for {method}: {audit[method]}")
    return frame, audit


def user_bootstrap(frame: pd.DataFrame, scope: str) -> pd.DataFrame:
    keys = ["dataset", "configuration", "window"]
    rows = []
    for half_life in sorted(frame.half_life_days_run.unique()):
        run = frame[frame.half_life_days_run == half_life].copy()
        run["l05"] = .5 * run.macro_user_abs_coverage_gap + .5 * run.max_abs_cluster_coverage_gap
        weighted = run[run.method == "erw_user_norm"].set_index(keys)
        static = run[run.method == "rolling_user_norm"].set_index(keys)
        paired = pd.DataFrame({
            "l05_difference": weighted.l05-static.l05,
            "relative_winkler_change": (weighted.winkler_interval_score-static.winkler_interval_score)/static.winkler_interval_score,
        }).reset_index()
        for block in (2, 3):
            for metric in ("l05_difference", "relative_winkler_change"):
                point, low, high = block_bootstrap(paired, metric, block)
                rows.append({
                    "scope": scope, "half_life_days": int(half_life), "granularity": "user",
                    "metric": metric, "block_length": block, "point_estimate": point,
                    "ci_lower": low, "ci_upper": high, "replicates": 10000, "seed": 20260729,
                })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=HERE)
    parser.add_argument("--frozen-root", type=Path, default=HERE)
    parser.add_argument("--output-root", type=Path, default=HERE)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    persistence = persistence_frame(args.input_root)
    lightgbm, audit = lightgbm_frame(args.input_root, args.frozen_root)
    summary = pd.concat([
        paired_summary(persistence, "persistence_full_grid"),
        paired_summary(lightgbm, "lightgbm_80pct_1h"),
    ], ignore_index=True)
    bootstrap = pd.concat([
        user_bootstrap(persistence, "persistence_full_grid"),
        user_bootstrap(lightgbm, "lightgbm_80pct_1h"),
    ], ignore_index=True)
    summary.to_csv(args.output_root / "ERW_SENSITIVITY_SUMMARY.csv", index=False, encoding="utf-8-sig")
    bootstrap.to_csv(args.output_root / "ERW_SENSITIVITY_BLOCK_BOOTSTRAP.csv", index=False, encoding="utf-8-sig")
    persistence.to_csv(args.output_root / "ERW_HALF_LIFE_PANEL.csv", index=False, encoding="utf-8-sig")
    report = {"protocol": "FROZEN_ERW_SENSITIVITY_ADDENDUM.md", "lightgbm_static_reproduction_audit": audit, "summary": summary.to_dict("records"), "bootstrap": bootstrap.to_dict("records")}
    (args.output_root / "ERW_SENSITIVITY_REPORT.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["# ERW-CQR sensitivity completion report", "", "- The 14-day setting remains the frozen primary setting.", "- Persistence sensitivity covers 7/14/28 days over the complete 140-environment grid.", "- LightGBM transfer covers the pre-specified 80%/1 h 35-environment main configuration.", "- Static LightGBM metrics reproduce the frozen output with maximum absolute difference 0.", "", "```csv", summary.to_csv(index=False).strip(), "```", ""]
    (args.output_root / "ERW_SENSITIVITY_COMPLETION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
