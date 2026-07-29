"""Aggregate frozen strictly-forward ACI runs without pooling load units."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
DATASETS = ("london", "ausgrid", "uci")
FORECASTERS = ("lgbm", "persistence")
STATIC = {"aci_global": "rolling_global_norm", "aci_segment": "rolling_group_norm", "aci_user": "rolling_user_norm"}
METRICS = ("picp", "mpiw", "winkler_interval_score", "macro_user_abs_coverage_gap", "max_abs_cluster_coverage_gap")


def main():
    frames, reports, pairs = [], [], []
    reproduction = []
    for forecaster in FORECASTERS:
        for dataset in DATASETS:
            directory = HERE / f"aci_{forecaster}_{dataset}"
            frame = pd.read_csv(directory / "window_metrics.csv")
            if set(frame.method) != {"raw", "rolling_global_norm", "rolling_group_norm", "rolling_user_norm", *STATIC}:
                raise RuntimeError(f"incomplete methods in {directory}")
            if not frame[list(METRICS)].apply(pd.to_numeric, errors="coerce").notna().all().all():
                raise RuntimeError(f"non-finite metrics in {directory}")
            summary = json.loads((directory / "diagnostic_summary.json").read_text(encoding="utf-8"))
            reference_name = (f"lightgbm_{dataset}_scoring" if forecaster == "lgbm" else f"naive_{dataset}_scoring")
            reference = pd.read_csv(HERE / reference_name / "window_metrics.csv")
            if forecaster == "persistence":
                reference = reference[(reference.coverage == .8) & (reference.horizon_hours == 1.)].copy()
            current_static = frame[frame.method.isin(set(STATIC.values()))][["window", "method", *METRICS]]
            reference_static = reference[reference.method.isin(set(STATIC.values()))][["window", "method", *METRICS]]
            checked = current_static.merge(reference_static, on=["window", "method"], suffixes=("_aci", "_reference"), validate="one_to_one")
            maximum = max(float((checked[f"{metric}_aci"] - checked[f"{metric}_reference"]).abs().max()) for metric in METRICS)
            if maximum != 0.0:
                raise RuntimeError(f"static reproduction failed in {directory}: {maximum}")
            reproduction.append({"dataset": summary["data"]["dataset"], "forecaster": summary["data"]["forecaster"],
                                 "compared_rows": len(checked), "maximum_absolute_difference": maximum})
            reports.append({"dataset": summary["data"]["dataset"], "forecaster": summary["data"]["forecaster"],
                            "windows": len(summary["data"]["windows"]),
                            "static_gcr": summary["granularity_conflict"]["personalization_conflict_rate"],
                            "aci_gcr": summary["aci_granularity_conflict"]["personalization_conflict_rate"],
                            "static_reverse_gcr": summary["granularity_conflict"]["reverse_conflict_rate"],
                            "aci_reverse_gcr": summary["aci_granularity_conflict"]["reverse_conflict_rate"]})
            for aci, static in STATIC.items():
                left = frame[frame.method == aci].set_index("window")
                right = frame[frame.method == static].set_index("window")
                for window in left.index:
                    row = {"dataset": summary["data"]["dataset"], "forecaster": summary["data"]["forecaster"],
                           "window": window, "aci_method": aci, "static_method": static}
                    for metric in METRICS:
                        row[f"delta_{metric}_aci_minus_static"] = float(left.loc[window, metric] - right.loc[window, metric])
                    pairs.append(row)
            frames.append(frame)
    panel = pd.concat(frames, ignore_index=True)
    per_dataset = panel.groupby(["dataset", "forecaster", "method"], as_index=False)[list(METRICS)].mean()
    pair = pd.DataFrame(pairs)
    pair_summary = pair.groupby(["dataset", "forecaster", "aci_method", "static_method"], as_index=False).mean(numeric_only=True)
    report = pd.DataFrame(reports)
    if len(panel) != 490 or report.windows.sum() != 70:
        raise RuntimeError("ACI panel integrity failure")
    panel.to_csv(HERE / "ACI_FORWARD_PANEL.csv", index=False, encoding="utf-8-sig")
    per_dataset.to_csv(HERE / "ACI_FORWARD_SUMMARY_BY_DATASET.csv", index=False, encoding="utf-8-sig")
    pair_summary.to_csv(HERE / "ACI_FORWARD_PAIR_DIFFERENCES.csv", index=False, encoding="utf-8-sig")
    report.to_csv(HERE / "ACI_FORWARD_CONFLICT_SUMMARY.csv", index=False, encoding="utf-8-sig")
    (HERE / "ACI_FORWARD_REPORT.json").write_text(json.dumps({"panel_rows": len(panel), "configuration_windows": int(report.windows.sum()),
        "conflict_summary": reports, "static_reproduction_audit": reproduction,
        "interpretation": "MPIW and interval-score summaries remain data-set-specific; no cross-unit absolute pooling."}, indent=2), encoding="utf-8")
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
