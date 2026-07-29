"""Aggregate and bootstrap the frozen ERW-CQR baseline extension."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
SEED = 20260729
REPLICATES = 10_000
PAIRS = {
    "ERW-Global - Global-CQR": ("erw_global_norm", "rolling_global_norm"),
    "ERW-Segment - Segment/Mondrian-CQR": ("erw_group_norm", "rolling_group_norm"),
    "ERW-User - User-CQR": ("erw_user_norm", "rolling_user_norm"),
}
METHOD_LABELS = {
    "rolling_global_norm": "Global-CQR",
    "rolling_group_norm": "Segment/Mondrian-CQR",
    "rolling_user_norm": "User-CQR",
    "erw_global_norm": "ERW-Global",
    "erw_group_norm": "ERW-Segment",
    "erw_user_norm": "ERW-User",
}
BASE_METRICS = (
    "picp",
    "mpiw",
    "winkler_interval_score",
    "macro_user_abs_coverage_gap",
    "user_coverage_std",
    "max_abs_cluster_coverage_gap",
)


def circular_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    draws = []
    while len(draws) < n:
        start = int(rng.integers(n))
        draws.extend((start + offset) % n for offset in range(block))
    return np.asarray(draws[:n], dtype=int)


def load_panel(input_root: Path) -> pd.DataFrame:
    frames = []
    for dataset in ("london", "ausgrid", "uci"):
        frame = pd.read_csv(input_root / f"weighted_conformal_{dataset}" / "window_metrics.csv")
        frames.append(frame)
    panel = pd.concat(frames, ignore_index=True)
    panel["picp_abs_gap"] = (panel["picp"] - panel["coverage"]).abs()
    panel["l05"] = 0.5 * (
        panel["macro_user_abs_coverage_gap"] + panel["max_abs_cluster_coverage_gap"]
    )
    return panel


def audit_frozen(panel: pd.DataFrame, frozen_root: Path) -> dict:
    result = {}
    for dataset in ("london", "ausgrid", "uci"):
        new = panel[
            (panel.dataset.str.lower().str.startswith(dataset))
            & panel.method.str.startswith("rolling_")
        ]
        old = pd.read_csv(frozen_root / f"naive_{dataset}_scoring" / "window_metrics.csv")
        keys = ["dataset", "configuration", "window", "coverage", "horizon_hours", "method"]
        merged = new.merge(old, on=keys, suffixes=("_new", "_old"), validate="one_to_one")
        result[dataset] = {
            metric: float(np.max(np.abs(merged[f"{metric}_new"] - merged[f"{metric}_old"])))
            for metric in BASE_METRICS
        }
    return result


def method_summary(panel: pd.DataFrame, scope: str) -> pd.DataFrame:
    if scope == "primary_80_1h":
        panel = panel[(panel.coverage == 0.8) & (panel.horizon_hours == 1.0)]
    elif scope != "full_grid":
        raise ValueError(scope)
    keys = ["dataset", "configuration", "window"]
    global_score = panel[panel.method == "rolling_global_norm"][keys + ["winkler_interval_score"]].rename(
        columns={"winkler_interval_score": "global_score"}
    )
    frame = panel.merge(global_score, on=keys, validate="many_to_one")
    frame["relative_score_vs_global"] = frame.winkler_interval_score / frame.global_score - 1.0
    summary = frame.groupby("method", sort=False).agg(
        observations=("window", "size"),
        mean_picp_abs_gap=("picp_abs_gap", "mean"),
        mean_macro_user_gap=("macro_user_abs_coverage_gap", "mean"),
        mean_max_segment_gap=("max_abs_cluster_coverage_gap", "mean"),
        mean_l05=("l05", "mean"),
        mean_relative_score_vs_global=("relative_score_vs_global", "mean"),
    ).reset_index()
    summary.insert(0, "scope", scope)
    summary["method_label"] = summary.method.map(METHOD_LABELS)
    return summary


def paired_frame(panel: pd.DataFrame, new_method: str, old_method: str, scope: str) -> pd.DataFrame:
    if scope == "primary_80_1h":
        panel = panel[(panel.coverage == 0.8) & (panel.horizon_hours == 1.0)]
    keys = ["dataset", "configuration", "window", "coverage", "horizon_hours"]
    cols = keys + [
        "picp_abs_gap", "macro_user_abs_coverage_gap", "max_abs_cluster_coverage_gap",
        "l05", "winkler_interval_score",
    ]
    new = panel[panel.method == new_method][cols].copy()
    old = panel[panel.method == old_method][cols].copy()
    merged = new.merge(old, on=keys, suffixes=("_new", "_old"), validate="one_to_one")
    merged["picp_abs_gap_diff"] = merged.picp_abs_gap_new - merged.picp_abs_gap_old
    merged["macro_user_gap_diff"] = (
        merged.macro_user_abs_coverage_gap_new - merged.macro_user_abs_coverage_gap_old
    )
    merged["max_segment_gap_diff"] = (
        merged.max_abs_cluster_coverage_gap_new - merged.max_abs_cluster_coverage_gap_old
    )
    merged["l05_diff"] = merged.l05_new - merged.l05_old
    merged["relative_score_change"] = (
        merged.winkler_interval_score_new / merged.winkler_interval_score_old - 1.0
    )
    return merged


def block_bootstrap(frame: pd.DataFrame, metric: str, block: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(SEED + block)
    dataset_parts = []
    for dataset, part in frame.groupby("dataset", sort=True):
        windows = list(dict.fromkeys(part.window.tolist()))
        by_window = {window: part[part.window == window][metric].to_numpy(float) for window in windows}
        dataset_parts.append((windows, by_window))
    values = np.empty(REPLICATES, dtype=float)
    for replicate in range(REPLICATES):
        sampled = []
        for windows, by_window in dataset_parts:
            indices = circular_indices(len(windows), block, rng)
            sampled.extend(by_window[windows[index]] for index in indices)
        values[replicate] = float(np.mean(np.concatenate(sampled)))
    return float(frame[metric].mean()), float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=HERE)
    parser.add_argument("--frozen-root", type=Path, default=HERE)
    parser.add_argument("--output-root", type=Path, default=HERE)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    panel = load_panel(args.input_root)
    audit = audit_frozen(panel, args.frozen_root)
    if any(value != 0.0 for dataset in audit.values() for value in dataset.values()):
        raise RuntimeError(f"frozen metric audit failed: {audit}")
    summaries = pd.concat(
        [method_summary(panel, "full_grid"), method_summary(panel, "primary_80_1h")],
        ignore_index=True,
    )
    bootstrap_rows = []
    metrics = (
        "picp_abs_gap_diff", "macro_user_gap_diff", "max_segment_gap_diff",
        "l05_diff", "relative_score_change",
    )
    for scope in ("full_grid", "primary_80_1h"):
        for pair_label, (new_method, old_method) in PAIRS.items():
            paired = paired_frame(panel, new_method, old_method, scope)
            for block in (2, 3):
                for metric in metrics:
                    estimate, lower, upper = block_bootstrap(paired, metric, block)
                    bootstrap_rows.append({
                        "scope": scope,
                        "pair": pair_label,
                        "metric": metric,
                        "block_length": block,
                        "point_estimate": estimate,
                        "ci_lower": lower,
                        "ci_upper": upper,
                        "replicates": REPLICATES,
                        "seed": SEED,
                    })
    bootstrap = pd.DataFrame(bootstrap_rows)
    panel.to_csv(args.output_root / "WEIGHTED_CONFORMAL_PANEL.csv", index=False, encoding="utf-8-sig")
    summaries.to_csv(args.output_root / "WEIGHTED_CONFORMAL_SUMMARY.csv", index=False, encoding="utf-8-sig")
    bootstrap.to_csv(args.output_root / "WEIGHTED_CONFORMAL_BLOCK_BOOTSTRAP.csv", index=False, encoding="utf-8-sig")

    full = summaries[summaries.scope == "full_grid"].set_index("method")
    primary = summaries[summaries.scope == "primary_80_1h"].set_index("method")
    erw_user_boot = bootstrap[
        (bootstrap.scope == "full_grid")
        & (bootstrap.pair == "ERW-User - User-CQR")
        & (bootstrap.metric == "l05_diff")
        & (bootstrap.block_length == 2)
    ].iloc[0]
    report = f"""# ERW-CQR baseline completion report

> Historical persistence-only report. Its persistence results remain valid, but its transfer scope is superseded by `ERW_SENSITIVITY_COMPLETION_REPORT.md`.

- Protocol: `FROZEN_WEIGHTED_CONFORMAL_BASELINE_ADDENDUM.md`
- Runtime: Python 3.12, NumPy 2.3.5, pandas 2.3.3
- Frozen audit: all six metrics for Global/Segment/User-CQR reproduced with maximum absolute difference 0 on all three datasets.
- Full grid: 140 configuration × window observations; primary setting: 35 unique 80%/1 h windows.

## Main result

Across the full persistence grid, ERW-User reduced mean L0.5 from {full.loc['rolling_user_norm','mean_l05']:.6f} to {full.loc['erw_user_norm','mean_l05']:.6f}. The paired difference was {erw_user_boot.point_estimate:.6f}, with a synchronized circular block-2 interval of [{erw_user_boot.ci_lower:.6f}, {erw_user_boot.ci_upper:.6f}]. Its mean Winkler score changed by {full.loc['erw_user_norm','mean_relative_score_vs_global'] - full.loc['rolling_user_norm','mean_relative_score_vs_global']:.2%} relative to static User-CQR.

In the primary 80%/1 h setting, ERW-User reduced mean L0.5 from {primary.loc['rolling_user_norm','mean_l05']:.6f} to {primary.loc['erw_user_norm','mean_l05']:.6f}.

At this initial stage, ERW-CQR was evaluated only with the persistence backbone. It is not evidence of a distribution-free guarantee under temporal drift. The subsequent LightGBM transfer is reported in `ERW_SENSITIVITY_COMPLETION_REPORT.md`.
"""
    (args.output_root / "WEIGHTED_CONFORMAL_COMPLETION_REPORT.md").write_text(report, encoding="utf-8")
    (args.output_root / "WEIGHTED_CONFORMAL_REPORT.json").write_text(
        json.dumps({"audit": audit, "replicates": REPLICATES, "seed": SEED}, indent=2),
        encoding="utf-8",
    )
    print(report)


if __name__ == "__main__":
    main()
