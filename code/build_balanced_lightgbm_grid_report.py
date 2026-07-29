"""Audit the frozen LightGBM full grid and combine it with the matched persistence grid."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import diagnostic_metrics as diag
import build_robustness_panel as robust


HERE = Path(__file__).resolve().parent
METHODS = ("raw", "rolling_global_norm", "rolling_group_norm", "rolling_user_norm")
METRICS = ("picp", "mpiw", "winkler_interval_score", "macro_user_abs_coverage_gap", "max_abs_cluster_coverage_gap")
DATASETS = ("London", "Ausgrid", "UCI Electricity")
OLD_LGBM = {
    "London": HERE / "lightgbm_london_scoring",
    "Ausgrid": HERE / "lightgbm_ausgrid_scoring",
    "UCI Electricity": HERE / "lightgbm_uci_scoring",
}
PERSISTENCE = {
    "London": HERE / "naive_london_full",
    "Ausgrid": HERE / "naive_ausgrid_full",
    "UCI Electricity": HERE / "naive_uci_full",
}
MATCHED_LGBM_90 = HERE / "lightgbm_matched_90_v1"
REPS, SEED = 10_000, 20260729


def block_indices(n: int, block: int, reps: int, rng: np.random.Generator) -> np.ndarray:
    starts = rng.integers(0, n, size=(reps, int(math.ceil(n / block))))
    return ((starts[:, :, None] + np.arange(block)[None, None, :]) % n).reshape(reps, -1)[:, :n]


def read_long(directory: Path, forecaster: str) -> pd.DataFrame:
    frame = pd.read_csv(directory / "window_metrics.csv")
    required = {"dataset", "coverage", "horizon_hours", "window", "method", *METRICS}
    missing = required - set(frame)
    if missing:
        raise RuntimeError(f"{directory} missing {sorted(missing)}")
    if set(frame.method.unique()) != set(METHODS):
        raise RuntimeError(f"{directory} has incomplete policy set")
    frame = frame.copy()
    frame["forecaster"] = forecaster
    return frame


def read_user_long(directory: Path) -> pd.DataFrame:
    frame = pd.read_csv(directory / "per_user_window_metrics.csv")
    required = {"dataset", "coverage_target", "horizon_hours", "window", "method", "user_index", "coverage", "n"}
    missing = required - set(frame)
    if missing:
        raise RuntimeError(f"{directory} missing per-user fields {sorted(missing)}")
    return frame


def strict_tci_table() -> pd.DataFrame:
    lgbm80 = read_user_long(HERE / "lightgbm_full_grid_v3")
    lgbm80 = lgbm80[np.isclose(lgbm80.coverage_target, .8)].copy()
    lgbm90 = read_user_long(MATCHED_LGBM_90)
    lgbm90 = lgbm90[np.isclose(lgbm90.coverage_target, .9)].copy()
    user = [lgbm80, lgbm90]
    for directory in PERSISTENCE.values():
        user.append(read_user_long(directory))
    frame = pd.concat(user, ignore_index=True)
    rows = []
    keys = ["dataset", "forecaster", "coverage_target", "horizon_hours"]
    for key, subset in frame.groupby(keys, sort=True):
        for item in diag.temporal_cancellation(subset, target=float(key[2])):
            rows.append({
                "dataset": key[0], "forecaster": key[1], "coverage": float(key[2]),
                "horizon_hours": float(key[3]), **item,
            })
    output = pd.DataFrame(rows)
    expected = 2 * 3 * 2 * 2 * 4
    if len(output) != expected or output.temporal_cancellation_absolute.min() < -1e-12:
        raise RuntimeError(f"strict TCI table failed completeness/nonnegativity audit: {len(output)} rows")
    return output


def read_scored_persistence() -> pd.DataFrame:
    """Recover full-grid interval scores from the frozen scored wide panel.

    The original persistence full directories predate interval-score output, while
    `SCORED_ROBUSTNESS_PANEL.csv` is the frozen score augmentation for exactly
    those same configuration-month cells.
    """
    wide = pd.read_csv(HERE / "SCORED_ROBUSTNESS_PANEL.csv")
    wide = wide[wide.forecaster == "persistence_quantile_interval"].copy()
    prefixes = {"raw": "raw", "rolling_global_norm": "global", "rolling_group_norm": "segment", "rolling_user_norm": "user"}
    rows = []
    for _, item in wide.iterrows():
        for method, prefix in prefixes.items():
            rows.append({
                "dataset": item.dataset, "forecaster": "persistence_quantile_interval", "coverage": item.coverage,
                "horizon_hours": item.horizon_hours, "window": item.window, "method": method,
                **{metric: float(item[f"{prefix}_{metric}"]) for metric in METRICS},
            })
    output = pd.DataFrame(rows)
    if len(output) != 140 * 4 or output.isna().any().any():
        raise RuntimeError("frozen scored persistence panel is incomplete")
    return output


def build_panel(long: pd.DataFrame) -> pd.DataFrame:
    keys = ["dataset", "forecaster", "coverage", "horizon_hours", "window"]
    records = []
    for key, frame in long.groupby(keys, sort=True):
        if set(frame.method) != set(METHODS):
            raise RuntimeError(f"missing method for {key}")
        row = dict(zip(keys, key))
        for method, short in (("raw", "raw"), ("rolling_global_norm", "global"), ("rolling_group_norm", "segment"), ("rolling_user_norm", "user")):
            values = frame[frame.method == method].iloc[0]
            for metric in METRICS:
                row[f"{short}_{metric}"] = float(values[metric])
        row["delta_user_gap_user_minus_global"] = row["user_macro_user_abs_coverage_gap"] - row["global_macro_user_abs_coverage_gap"]
        row["delta_segment_gap_user_minus_global"] = row["user_max_abs_cluster_coverage_gap"] - row["global_max_abs_cluster_coverage_gap"]
        row["personalization_conflict"] = int(row["delta_user_gap_user_minus_global"] < 0 and row["delta_segment_gap_user_minus_global"] > 0)
        row["reverse_conflict"] = int(row["delta_user_gap_user_minus_global"] > 0 and row["delta_segment_gap_user_minus_global"] < 0)
        records.append(row)
    panel = pd.DataFrame(records)
    panel["window_order"] = panel.groupby("dataset")["window"].rank(method="dense").astype(int) - 1
    return panel.sort_values(["dataset", "forecaster", "coverage", "horizon_hours", "window"]).reset_index(drop=True)


def reproduction_audit(full: pd.DataFrame) -> dict:
    output = {}
    for dataset, directory in OLD_LGBM.items():
        reference = pd.read_csv(directory / "window_metrics.csv")
        candidate = full[(full.dataset == dataset) & (full.forecaster == "lightgbm_quantile") & (full.coverage == .8) & (full.horizon_hours == 1.)].copy()
        candidate = candidate.melt(id_vars=["dataset", "forecaster", "coverage", "horizon_hours", "window"], value_vars=[col for col in candidate if col.endswith(tuple(METRICS))])
        # Audit the source long table instead of the wide panel; reconstruct policy/metric names deterministically.
        source = pd.read_csv(HERE / "lightgbm_full_grid_v3" / "window_metrics.csv")
        source = source[(source.dataset == dataset) & (source.coverage == .8) & (source.horizon_hours == 1.)].sort_values(["window", "method"]).reset_index(drop=True)
        reference = reference.sort_values(["window", "method"]).reset_index(drop=True)
        if len(source) != len(reference) or not source[["window", "method"]].equals(reference[["window", "method"]]):
            raise RuntimeError(f"LightGBM 80%/1h reproduction keys failed for {dataset}")
        fields = [field for field in METRICS if field in reference and field in source]
        difference = max(float(np.max(np.abs(source[field].to_numpy(float) - reference[field].to_numpy(float)))) for field in fields)
        output[dataset] = {"rows": len(source), "max_absolute_metric_difference": difference, "passed": bool(difference <= 1e-10)}
        if not output[dataset]["passed"]:
            raise RuntimeError(f"LightGBM 80%/1h reproduction audit failed for {dataset}: {difference}")
    return output


def bootstrap_rate(panel: pd.DataFrame, mask: pd.Series, block: int) -> dict:
    subset = panel[mask].copy()
    matrices = []
    for dataset, frame in subset.groupby("dataset", sort=True):
        pivot = frame.pivot(index="window_order", columns=["forecaster", "coverage", "horizon_hours"], values="personalization_conflict").sort_index()
        if pivot.isna().any().any():
            raise RuntimeError(f"missing synchronized grid cell in {dataset}")
        matrices.append(pivot.to_numpy(float))
    rng = np.random.default_rng(SEED + block)
    estimates = np.zeros(REPS)
    weights = 0
    for matrix in matrices:
        indices = block_indices(matrix.shape[0], block, REPS, rng)
        estimates += matrix[indices].sum(axis=(1, 2))
        weights += matrix.size
    point = float(subset.personalization_conflict.mean())
    draws = estimates / weights
    return {"estimate": point, "ci_low": float(np.quantile(draws, .025)), "ci_high": float(np.quantile(draws, .975)), "block_months": block, "repetitions": REPS}


def summarize(panel: pd.DataFrame) -> dict:
    by_grid = []
    for key, frame in panel.groupby(["forecaster", "coverage", "horizon_hours"], sort=True):
        by_grid.append({
            "forecaster": key[0], "coverage": float(key[1]), "horizon_hours": float(key[2]), "environments": len(frame),
            "personalization_conflicts": int(frame.personalization_conflict.sum()), "personalization_conflict_rate": float(frame.personalization_conflict.mean()),
            "reverse_conflicts": int(frame.reverse_conflict.sum()),
            "mean_user_gap_global": float(frame.global_macro_user_abs_coverage_gap.mean()),
            "mean_user_gap_user": float(frame.user_macro_user_abs_coverage_gap.mean()),
            "mean_segment_gap_global": float(frame.global_max_abs_cluster_coverage_gap.mean()),
            "mean_segment_gap_user": float(frame.user_max_abs_cluster_coverage_gap.mean()),
            "mean_winkler_global": float(frame.global_winkler_interval_score.mean()),
            "mean_winkler_user": float(frame.user_winkler_interval_score.mean()),
        })
    return {
        "balanced_panel_rows": len(panel), "unique_dataset_month_windows": int(panel[["dataset", "window"]].drop_duplicates().shape[0]),
        "grid_cells": by_grid,
        "overall": {"personalization_conflicts": int(panel.personalization_conflict.sum()), "personalization_conflict_rate": float(panel.personalization_conflict.mean()), "reverse_conflicts": int(panel.reverse_conflict.sum()), "reverse_conflict_rate": float(panel.reverse_conflict.mean())},
        "gcr_block_bootstrap": {"block_2": bootstrap_rate(panel, pd.Series(True, index=panel.index), 2), "block_3": bootstrap_rate(panel, pd.Series(True, index=panel.index), 3)},
    }


def final_diagnostics(long: pd.DataFrame, panel: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    rank_rows, rog_rows = [], []
    keys = ["dataset", "forecaster", "coverage", "horizon_hours"]
    for key, frame in long.groupby(keys, sort=True):
        rank, _ = diag.rank_reversal(frame)
        rank_rows.append({**dict(zip(keys, key)), **rank})
        for row in diag.routing_oracle_gap(frame):
            rog_rows.append({**dict(zip(keys, key)), **row})
    rank_frame = pd.DataFrame(rank_rows)
    rank_macro = {}
    for forecaster, frame in rank_frame.groupby("forecaster"):
        rank_macro[forecaster] = {
            "rank_reversals": int(frame.rank_reversals.sum()),
            "adjacent_comparisons": int(frame.adjacent_comparisons.sum()),
            "rank_reversal_rate": float(frame.rank_reversals.sum() / frame.adjacent_comparisons.sum()),
        }
    thresholds = []
    for delta in (0., .0025, .005, .01):
        conflict = (panel.delta_user_gap_user_minus_global < -delta) & (panel.delta_segment_gap_user_minus_global > delta)
        thresholds.append({
            "delta_coverage_points": delta * 100, "conflicts": int(conflict.sum()),
            "environments": len(panel), "conflict_rate": float(conflict.mean()),
        })
    material = pd.DataFrame(thresholds)
    return {
        "rank_reversal_by_forecaster": rank_macro,
        "routing_oracle_gap": rog_rows,
        "material_conflict_sensitivity": thresholds,
    }, material


def final_mechanism_analysis(panel: pd.DataFrame) -> dict:
    keys = ["dataset", "forecaster", "coverage", "horizon_hours", "window"]
    fields = ["abs_global_correction", "user_correction_std", "group_correction_range", "mean_abs_user_global_disagreement"]
    frozen = pd.read_csv(HERE / "lightgbm_full_grid_v3" / "corrections.csv")
    matched = pd.read_csv(MATCHED_LGBM_90 / "corrections.csv")
    lgbm = pd.concat([frozen[np.isclose(frozen.coverage, .8)], matched[np.isclose(matched.coverage, .9)]], ignore_index=True)
    lgbm = lgbm[keys + fields]
    old = pd.read_csv(HERE / "ROBUSTNESS_PANEL.csv")
    persistence = old[old.forecaster == "persistence_quantile_interval"][keys + fields]
    corrections = pd.concat([lgbm, persistence], ignore_index=True)
    augmented = panel.merge(corrections, on=keys, how="left", validate="one_to_one")
    if augmented[fields].isna().any().any():
        raise RuntimeError("final mechanism panel has missing correction diagnostics")
    augmented["raw_picp_abs_gap"] = np.abs(augmented.raw_picp - augmented.coverage)
    augmented["delta_group_gap_user_minus_global"] = augmented.delta_segment_gap_user_minus_global
    x, names, scaling = robust.design_matrix(augmented)
    logistic = robust.logistic_irls(x, augmented.personalization_conflict.to_numpy(float), names)
    linear = robust.ols_hc1(x, augmented.delta_group_gap_user_minus_global.to_numpy(float), names)
    output = {"rows": len(augmented), "scaling": scaling, "logistic": logistic, "linear": linear}
    (HERE / "FINAL_MECHANISM_ANALYSIS.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output


def matched_quantile_sensitivity(frozen: pd.DataFrame, matched: pd.DataFrame) -> list[dict]:
    rows = []
    for label, frame in (("shared_0.1_0.9", frozen), ("matched_0.05_0.95", matched)):
        subset = frame[np.isclose(frame.coverage, .9)].copy()
        panel = build_panel(subset)
        for horizon, group in panel.groupby("horizon_hours", sort=True):
            rows.append({
                "construction": label, "coverage": .9, "horizon_hours": float(horizon),
                "environments": len(group), "personalization_conflicts": int(group.personalization_conflict.sum()),
                "personalization_conflict_rate": float(group.personalization_conflict.mean()),
                "reverse_conflicts": int(group.reverse_conflict.sum()),
            })
    pd.DataFrame(rows).to_csv(HERE / "LIGHTGBM_90_QUANTILE_CONSTRUCTION_SENSITIVITY.csv", index=False, encoding="utf-8-sig")
    return rows


def main() -> None:
    frozen = read_long(HERE / "lightgbm_full_grid_v3", "lightgbm_quantile")
    matched = read_long(MATCHED_LGBM_90, "lightgbm_quantile")
    lgbm = pd.concat([
        frozen[np.isclose(frozen.coverage, .8)],
        matched[np.isclose(matched.coverage, .9)],
    ], ignore_index=True)
    expected_lgbm = 140 * 4
    if len(lgbm) != expected_lgbm:
        raise RuntimeError(f"expected {expected_lgbm} LightGBM method rows, got {len(lgbm)}")
    persistence = read_scored_persistence()
    if len(persistence) != expected_lgbm:
        raise RuntimeError(f"expected {expected_lgbm} persistence method rows, got {len(persistence)}")
    long = pd.concat([lgbm, persistence], ignore_index=True)
    panel = build_panel(long)
    if len(panel) != 280 or panel.isna().any().any():
        raise RuntimeError(f"balanced panel failure: rows={len(panel)}, missing={int(panel.isna().sum().sum())}")
    audit = reproduction_audit(panel)
    report = summarize(panel)
    report["lightgbm_90_quantile_construction_sensitivity"] = matched_quantile_sensitivity(frozen, matched)
    diagnostics, material = final_diagnostics(long, panel)
    report.update(diagnostics)
    report["mechanism_analysis"] = final_mechanism_analysis(panel)
    material.to_csv(HERE / "MATERIAL_CONFLICT_SENSITIVITY.csv", index=False, encoding="utf-8-sig")
    tci = strict_tci_table()
    tci.to_csv(HERE / "STRICT_TCI_FULL_GRID.csv", index=False, encoding="utf-8-sig")
    user_tci = tci[tci.method == "rolling_user_norm"].groupby("forecaster").temporal_cancellation_ratio.mean()
    report["protocol"] = ["FROZEN_LIGHTGBM_FULL_GRID_PROTOCOL.md", "FROZEN_MATCHED_LIGHTGBM_QUANTILE_ADDENDUM.md"]
    report["lightgbm_80pct_1h_reproduction_audit"] = audit
    report["strict_user_tci_macro"] = {name: float(value) for name, value in user_tci.items()}
    report["interpretation"] = "The balanced grid strengthens predictor/configuration coverage; block-bootstrap intervals remain descriptive under temporal dependence."
    panel.to_csv(HERE / "BALANCED_FULL_GRID_PANEL.csv", index=False, encoding="utf-8-sig")
    (HERE / "BALANCED_FULL_GRID_REPORT.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["# LightGBM 完整网格完成报告", "", "## 完整性", "", f"- 平衡面板：{len(panel)} 个 forecaster × coverage × horizon × month 环境；每种预测器均为 140 个环境。", f"- 唯一 dataset × calendar-month 窗口：{report['unique_dataset_month_windows']}；不将 280 个配置行视为独立时间重复。", "- LightGBM 80%/1h 与既有冻结输出的逐指标复现审计全部通过。", "- 90% LightGBM 使用目标匹配的 0.05/0.95 基础分位区间；旧 0.1/0.9 共享基础区间结果仅保留为敏感性。", "- TCI 两项均采用用户内有效样本数权重，再对用户宏平均，因此严格非负。", "", "## 主结果", ""]
    for row in report["grid_cells"]:
        lines.append(f"- {row['forecaster']}，{row['coverage']:.0%}、{row['horizon_hours']:g}h：GCR={row['personalization_conflict_rate']:.2%} ({row['personalization_conflicts']}/{row['environments']})。")
    block = report["gcr_block_bootstrap"]["block_2"]
    lines.extend(["", "- 全平衡面板的严格 GCR 为 " + f"{report['overall']['personalization_conflict_rate']:.2%}" + f"；同步月份 block=2 描述性区间为 [{block['ci_low']:.2%}, {block['ci_high']:.2%}]。", "- 详细 PICP、MPIW、Winkler score 和双层级 gap 位于 `BALANCED_FULL_GRID_PANEL.csv`；该报告不从网格中重新选择 ACI 或 CSGR 参数。", ""])
    (HERE / "LIGHTGBM_FULL_GRID_COMPLETION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"rows": len(panel), "audit": audit, "overall": report["overall"]}, indent=2))


if __name__ == "__main__":
    main()
