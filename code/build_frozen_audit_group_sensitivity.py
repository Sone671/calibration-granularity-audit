"""Recompute GCR after changing only the audit grouping of frozen KM3 outputs."""

from __future__ import annotations

import gc
import json
import sys
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

import run_naive_robustness as naive  # noqa: E402

DATASETS = ("London", "Ausgrid", "UCI Electricity")
SCHEMES = {
    "DemandTertile": None,
    "KM2": "sensitivity_persistence_km2_{slug}",
    "KM3": None,
    "KM4": "sensitivity_persistence_km4_{slug}",
    "Ward3": "sensitivity_persistence_ward3_{slug}",
}
SLUG = {"London": "london", "Ausgrid": "ausgrid", "UCI Electricity": "uci"}
MAIN = {
    "lightgbm_quantile": HERE / "lightgbm_full_grid_v3",
    "persistence_quantile_interval": None,
}
PERSISTENCE = {
    "London": HERE / "naive_london_full",
    "Ausgrid": HERE / "naive_ausgrid_full",
    "UCI Electricity": HERE / "naive_uci_full",
}


def read_main(forecaster: str, dataset: str) -> pd.DataFrame:
    directory = MAIN[forecaster] if MAIN[forecaster] is not None else PERSISTENCE[dataset]
    frame = pd.read_csv(directory / "per_user_window_metrics.csv")
    coverage_col = "coverage_target"
    frame = frame[
        (frame.dataset == dataset)
        & np.isclose(frame[coverage_col], .8)
        & np.isclose(frame.horizon_hours, 1.)
        & frame.method.isin(("rolling_global_norm", "rolling_user_norm"))
    ].copy()
    if frame.empty:
        raise RuntimeError(f"missing frozen main rows for {forecaster}/{dataset}")
    frame["customer"] = frame.customer.astype(str)
    return frame


def read_full_grid(forecaster: str, dataset: str) -> pd.DataFrame:
    directory = MAIN[forecaster] if MAIN[forecaster] is not None else PERSISTENCE[dataset]
    frame = pd.read_csv(directory / "per_user_window_metrics.csv")
    frame = frame[
        (frame.dataset == dataset)
        & frame.method.isin(("rolling_global_norm", "rolling_user_norm"))
    ].copy()
    if frame.empty:
        raise RuntimeError(f"missing full-grid rows for {forecaster}/{dataset}")
    frame["customer"] = frame.customer.astype(str)
    return frame


def demand_tertile_mapping(dataset: str) -> pd.DataFrame:
    """Build a deterministic equal-frequency mapping from training-only mean load."""
    if dataset == "London":
        _, values, names, _, _, train0, train1, _, _ = naive.load_london(naive.LONDON, 3, "kmeans")
    elif dataset == "Ausgrid":
        _, values, names, _, _, train0, train1, _, _ = naive.load_ausgrid(EXT / "raw_archive", 3, "kmeans")
    else:
        _, values, names, _, _, train0, train1, _, _ = naive.load_uci(
            V1 / "electricityloaddiagrams20112014.originalmirror.zip", 3, "kmeans"
        )
    training_mean = np.nanmean(values[train0:train1], axis=0)
    if len(training_mean) != len(names) or not np.isfinite(training_mean).all():
        raise RuntimeError(f"invalid training means for {dataset}")
    order = np.lexsort((np.asarray(names, dtype=str), training_mean))
    labels = np.empty(len(names), dtype=int)
    for group, members in enumerate(np.array_split(order, 3)):
        labels[members] = group
    frame = pd.DataFrame({
        "dataset": dataset,
        "customer": np.asarray(names, dtype=str),
        "training_mean_load": training_mean,
        "audit_cluster": labels,
        "audit_group_label": np.asarray(("low", "medium", "high"), dtype=object)[labels],
    })
    sizes = frame.groupby("audit_cluster").size().to_numpy()
    extrema = frame.groupby("audit_cluster").training_mean_load.agg(["min", "max"])
    if sizes.max() - sizes.min() > 1 or not (extrema["max"].iloc[:-1].to_numpy() <= extrema["min"].iloc[1:].to_numpy()).all():
        raise RuntimeError(f"invalid demand-tertile mapping for {dataset}")
    del values
    gc.collect()
    return frame


def mapping(scheme: str, dataset: str, main: pd.DataFrame, demand_maps: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if scheme == "DemandTertile":
        source = demand_maps[dataset][["customer", "audit_cluster"]].copy()
    elif scheme == "KM3":
        source = main[["customer", "cluster"]].drop_duplicates()
    else:
        directory = HERE / SCHEMES[scheme].format(slug=SLUG[dataset])
        frame = pd.read_csv(directory / "per_user_window_metrics.csv")
        source = frame[["customer", "cluster"]].drop_duplicates()
    group_col = "audit_cluster" if "audit_cluster" in source.columns else "cluster"
    counts = source.groupby("customer")[group_col].nunique()
    if len(source) == 0 or counts.max() != 1:
        raise RuntimeError(f"non-unique audit mapping for {scheme}/{dataset}")
    source = source.copy()
    source["customer"] = source.customer.astype(str)
    return source.rename(columns={"cluster": "audit_cluster"})


def group_gaps(frame: pd.DataFrame, target: float = .8) -> tuple[float, float]:
    grouped = frame.groupby("audit_cluster", sort=True).apply(
        lambda z: pd.Series({
            "coverage": np.average(z.coverage.to_numpy(float), weights=z.n.to_numpy(float)),
            "n": z.n.to_numpy(float).sum(),
        }),
        include_groups=False,
    )
    gaps = np.abs(grouped.coverage.to_numpy(float) - target)
    weights = grouped.n.to_numpy(float)
    return float(gaps.max()), float(np.average(gaps, weights=weights))


def main() -> None:
    demand_maps = {dataset: demand_tertile_mapping(dataset) for dataset in DATASETS}
    pd.concat(demand_maps.values(), ignore_index=True).to_csv(
        HERE / "DEMAND_TERTILE_AUDIT_MAPPING.csv", index=False, encoding="utf-8-sig"
    )
    rows = []
    for forecaster in MAIN:
        for dataset in DATASETS:
            frozen = read_main(forecaster, dataset)
            for scheme in SCHEMES:
                merged = frozen.merge(mapping(scheme, dataset, frozen, demand_maps), on="customer", how="left", validate="many_to_one")
                if merged.audit_cluster.isna().any():
                    raise RuntimeError(f"unmapped customer in {forecaster}/{dataset}/{scheme}")
                for window, month in merged.groupby("window", sort=True):
                    metrics = {}
                    for method, subset in month.groupby("method", sort=True):
                        short = "global" if method == "rolling_global_norm" else "user"
                        metrics[f"{short}_macro_user_gap"] = float(np.mean(np.abs(subset.coverage.to_numpy(float) - .8)))
                        max_gap, weighted_gap = group_gaps(subset)
                        metrics[f"{short}_max_audit_group_gap"] = max_gap
                        metrics[f"{short}_weighted_mean_audit_group_gap"] = weighted_gap
                    du = metrics["user_macro_user_gap"] - metrics["global_macro_user_gap"]
                    dg = metrics["user_max_audit_group_gap"] - metrics["global_max_audit_group_gap"]
                    dw = (metrics["user_weighted_mean_audit_group_gap"]
                          - metrics["global_weighted_mean_audit_group_gap"])
                    rows.append({
                        "dataset": dataset, "forecaster": forecaster, "scheme": scheme, "window": window,
                        **metrics, "delta_user_gap_user_minus_global": du,
                        "delta_group_gap_user_minus_global": dg,
                        "delta_weighted_mean_group_gap_user_minus_global": dw,
                        "personalization_conflict": int(du < 0 and dg > 0),
                        "weighted_mean_group_conflict": int(du < 0 and dw > 0),
                    })
    panel = pd.DataFrame(rows).sort_values(["scheme", "forecaster", "dataset", "window"]).reset_index(drop=True)
    counts = panel.groupby(["scheme", "forecaster"]).size()
    if not (counts == 35).all():
        raise RuntimeError(f"incomplete frozen audit panel: {counts.to_dict()}")
    summary = panel.groupby(["scheme", "forecaster"], as_index=False).agg(
        environments=("window", "size"), conflicts=("personalization_conflict", "sum"),
        gcr=("personalization_conflict", "mean"),
        weighted_mean_group_conflicts=("weighted_mean_group_conflict", "sum"),
        weighted_mean_group_gcr=("weighted_mean_group_conflict", "mean"),
        mean_delta_user_gap=("delta_user_gap_user_minus_global", "mean"),
        mean_delta_group_gap=("delta_group_gap_user_minus_global", "mean"),
        mean_delta_weighted_mean_group_gap=("delta_weighted_mean_group_gap_user_minus_global", "mean"),
    )
    main_panel = pd.read_csv(HERE / "BALANCED_FULL_GRID_PANEL.csv")
    reference = main_panel[np.isclose(main_panel.coverage, .8) & np.isclose(main_panel.horizon_hours, 1.)]
    reference = reference.groupby("forecaster").personalization_conflict.sum().astype(int).to_dict()
    km3 = summary[summary.scheme == "KM3"].set_index("forecaster").conflicts.astype(int).to_dict()
    if km3 != reference:
        raise RuntimeError(f"KM3 reproduction failed: frozen={km3}, reference={reference}")
    panel.to_csv(HERE / "FROZEN_AUDIT_GROUP_SENSITIVITY_PANEL.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(HERE / "FROZEN_AUDIT_GROUP_SENSITIVITY_SUMMARY.csv", index=False, encoding="utf-8-sig")
    report = {
        "protocols": ["FROZEN_AUDIT_GROUP_SENSITIVITY_ADDENDUM.md", "FROZEN_P0_DYNAMIC_BASELINE_ADDENDUM.md"],
        "reference_conflicts": reference,
        "rows": summary.to_dict("records"),
    }
    (HERE / "FROZEN_AUDIT_GROUP_SENSITIVITY_REPORT.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))

    full_rows = []
    for forecaster in MAIN:
        for dataset in DATASETS:
            frozen = read_full_grid(forecaster, dataset)
            audit_map = demand_maps[dataset][["customer", "audit_cluster"]]
            merged = frozen.merge(audit_map, on="customer", how="left", validate="many_to_one")
            keys = ["dataset", "forecaster", "coverage_target", "horizon_hours", "window"]
            for values, environment in merged.groupby(keys, sort=True):
                metrics = {}
                for method, subset in environment.groupby("method", sort=True):
                    short = "global" if method == "rolling_global_norm" else "user"
                    target = float(subset.coverage_target.iloc[0])
                    metrics[f"{short}_macro_user_gap"] = float(np.mean(np.abs(subset.coverage.to_numpy(float) - target)))
                    max_gap, weighted_gap = group_gaps(subset, target=target)
                    metrics[f"{short}_max_audit_group_gap"] = max_gap
                    metrics[f"{short}_weighted_mean_audit_group_gap"] = weighted_gap
                du = metrics["user_macro_user_gap"] - metrics["global_macro_user_gap"]
                dg = metrics["user_max_audit_group_gap"] - metrics["global_max_audit_group_gap"]
                dw = metrics["user_weighted_mean_audit_group_gap"] - metrics["global_weighted_mean_audit_group_gap"]
                full_rows.append({**dict(zip(keys, values)), **metrics,
                                  "delta_user_gap_user_minus_global": du,
                                  "delta_group_gap_user_minus_global": dg,
                                  "delta_weighted_mean_group_gap_user_minus_global": dw,
                                  "personalization_conflict": int(du < 0 and dg > 0),
                                  "weighted_mean_group_conflict": int(du < 0 and dw > 0)})
    full_panel = pd.DataFrame(full_rows)
    counts = full_panel.groupby("forecaster").size()
    if not (counts == 140).all():
        raise RuntimeError(f"incomplete demand-tertile full grid: {counts.to_dict()}")
    full_summary = full_panel.groupby("forecaster", as_index=False).agg(
        environments=("window", "size"), conflicts=("personalization_conflict", "sum"),
        gcr=("personalization_conflict", "mean"),
        weighted_mean_group_conflicts=("weighted_mean_group_conflict", "sum"),
        weighted_mean_group_gcr=("weighted_mean_group_conflict", "mean"),
        mean_delta_user_gap=("delta_user_gap_user_minus_global", "mean"),
        mean_delta_group_gap=("delta_group_gap_user_minus_global", "mean"),
        mean_delta_weighted_mean_group_gap=("delta_weighted_mean_group_gap_user_minus_global", "mean"),
    )
    full_panel.to_csv(HERE / "DEMAND_TERTILE_FULL_GRID_PANEL.csv", index=False, encoding="utf-8-sig")
    full_summary.to_csv(HERE / "DEMAND_TERTILE_FULL_GRID_SUMMARY.csv", index=False, encoding="utf-8-sig")
    print("\nDemand-tertile full grid")
    print(full_summary.to_string(index=False))


if __name__ == "__main__":
    main()
