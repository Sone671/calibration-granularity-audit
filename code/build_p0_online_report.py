"""Aggregate the frozen canonical-ACI and conformal-PID P0 extension."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
DATASETS = ("london", "ausgrid", "uci")
FORECASTERS = ("lgbm", "persistence")
STATIC = ("rolling_global_norm", "rolling_group_norm", "rolling_user_norm")
ACI = ("aci_global", "aci_segment", "aci_user")
PID = ("pid_global", "pid_segment", "pid_user")
METHODS = {"raw", *STATIC, *ACI, *PID}
METRICS = (
    "picp", "mpiw", "winkler_interval_score",
    "macro_user_abs_coverage_gap", "max_abs_cluster_coverage_gap",
)
COMPARISONS = (
    ("aci_user_vs_static_user", "aci_user", "rolling_user_norm"),
    ("pid_user_vs_static_user", "pid_user", "rolling_user_norm"),
    ("pid_user_vs_aci_user", "pid_user", "aci_user"),
)
REPS = 10_000
SEED = 20260729


def window_order(frame: pd.DataFrame) -> pd.DataFrame:
    order = frame[["dataset", "window"]].drop_duplicates().sort_values(["dataset", "window"])
    order["window_order"] = order.groupby("dataset").cumcount()
    return frame.merge(order, on=["dataset", "window"], validate="many_to_one")


def conflict(frame: pd.DataFrame, user: str, global_: str) -> pd.DataFrame:
    key = ["dataset", "forecaster_key", "window", "window_order"]
    left = frame[frame.method == user].set_index(key)
    right = frame[frame.method == global_].set_index(key)
    joined = left.join(right, lsuffix="_user", rsuffix="_global", validate="one_to_one")
    output = joined.reset_index()[key].copy()
    output["indicator"] = (
        (joined.macro_user_abs_coverage_gap_user < joined.macro_user_abs_coverage_gap_global)
        & (joined.max_abs_cluster_coverage_gap_user > joined.max_abs_cluster_coverage_gap_global)
    ).astype(int).to_numpy()
    return output


def block_index_matrix(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(REPS, blocks))
    offsets = np.arange(block, dtype=int)
    return ((starts[:, :, None] + offsets[None, None, :]) % n).reshape(REPS, -1)[:, :n]


def synchronized_ci(frame: pd.DataFrame, columns: list[str], block: int, seed: int):
    rng = np.random.default_rng(seed)
    sums = np.zeros((REPS, len(columns)), dtype=float)
    total = 0
    for _, item in frame.groupby("dataset", sort=True):
        ordered = item.sort_values(["window_order", "forecaster_key"])
        n = ordered.window_order.nunique()
        values = ordered[columns].to_numpy(float)
        if len(values) != 2 * n or not np.isfinite(values).all():
            raise RuntimeError("synchronized-month alignment failure")
        values = values.reshape(n, 2, len(columns))
        indices = block_index_matrix(n, block, rng)
        sums += values[indices].sum(axis=(1, 2))
        total += 2 * n
    samples = sums / total
    return frame[columns].mean().to_numpy(float), np.quantile(samples, .025, axis=0), np.quantile(samples, .975, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=HERE)
    parser.add_argument("--frozen-root", type=Path, default=HERE)
    parser.add_argument("--output-root", type=Path, default=HERE)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    frames, user_frames, reproduction = [], [], []
    for forecaster in FORECASTERS:
        for dataset in DATASETS:
            directory = args.input_root / f"pid_{forecaster}_{dataset}"
            frame = pd.read_csv(directory / "window_metrics.csv")
            if set(frame.method) != METHODS:
                raise RuntimeError(f"incomplete P0 methods in {directory}: {set(frame.method)}")
            frame["forecaster_key"] = forecaster
            frames.append(frame)
            users = pd.read_csv(directory / "per_user_window_metrics.csv")
            users["forecaster_key"] = forecaster
            if "demand_tertile" not in users.columns:
                mapping_path = args.input_root / "DEMAND_TERTILE_AUDIT_MAPPING.csv"
                if not mapping_path.exists():
                    mapping_path = args.frozen_root / "DEMAND_TERTILE_AUDIT_MAPPING.csv"
                mapping = pd.read_csv(mapping_path, usecols=["dataset", "customer", "audit_cluster"])
                mapping["customer"] = mapping.customer.astype(str)
                users["customer"] = users.customer.astype(str)
                users = users.merge(mapping.rename(columns={"audit_cluster": "demand_tertile"}),
                                    on=["dataset", "customer"], how="left", validate="many_to_one")
            if users.demand_tertile.isna().any():
                raise RuntimeError(f"missing demand-tertile user mapping in {directory}")
            user_frames.append(users)
            previous_path = args.frozen_root / f"aci_{forecaster}_{dataset}" / "window_metrics.csv"
            if previous_path.exists():
                previous = pd.read_csv(previous_path)
            else:
                previous = pd.read_csv(args.frozen_root / "ACI_FORWARD_PANEL.csv")
                previous = previous[
                    (previous.dataset == frame.dataset.iloc[0])
                    & (previous.forecaster == frame.forecaster.iloc[0])
                ].copy()
            keys = ["dataset", "forecaster", "window", "method"]
            current_static = frame[frame.method.isin({"raw", *STATIC})][keys + list(METRICS)]
            previous_static = previous[previous.method.isin({"raw", *STATIC})][keys + list(METRICS)]
            checked = current_static.merge(previous_static, on=keys, suffixes=("_new", "_old"), validate="one_to_one")
            maximum = max(float((checked[f"{metric}_new"] - checked[f"{metric}_old"]).abs().max()) for metric in METRICS)
            if maximum > 1e-12:
                raise RuntimeError(f"static reproduction failed in {directory}: {maximum}")
            reproduction.append({"dataset": frame.dataset.iloc[0], "forecaster": frame.forecaster.iloc[0],
                                 "rows": len(checked), "maximum_absolute_difference": maximum})
    panel = window_order(pd.concat(frames, ignore_index=True))
    if len(panel) != 700 or panel[list(METRICS)].isna().any().any():
        raise RuntimeError(f"P0 panel integrity failure: {len(panel)}")

    pairs = []
    key = ["dataset", "forecaster_key", "window", "window_order"]
    for label, target, base in COMPARISONS:
        left = panel[panel.method == target].set_index(key)
        right = panel[panel.method == base].set_index(key)
        joined = left.join(right, lsuffix="_target", rsuffix="_base", validate="one_to_one")
        for _, row in joined.reset_index().iterrows():
            output = {column: row[column] for column in key}
            output.update({"comparison": label, "target": target, "base": base})
            for metric in METRICS:
                output[f"delta_{metric}"] = float(row[f"{metric}_target"] - row[f"{metric}_base"])
            output["relative_mpiw_change"] = output["delta_mpiw"] / float(row["mpiw_base"])
            output["relative_winkler_interval_score_change"] = output["delta_winkler_interval_score"] / float(row["winkler_interval_score_base"])
            pairs.append(output)
    pairs = pd.DataFrame(pairs)

    conflict_frames = {
        "static": conflict(panel, "rolling_user_norm", "rolling_global_norm"),
        "aci": conflict(panel, "aci_user", "aci_global"),
        "pid": conflict(panel, "pid_user", "pid_global"),
    }
    conflicts = conflict_frames["static"].rename(columns={"indicator": "static_indicator"})
    for name in ("aci", "pid"):
        conflicts = conflicts.merge(
            conflict_frames[name].rename(columns={"indicator": f"{name}_indicator"}), on=key, validate="one_to_one"
        )
    conflicts["aci_minus_static"] = conflicts.aci_indicator - conflicts.static_indicator
    conflicts["pid_minus_static"] = conflicts.pid_indicator - conflicts.static_indicator
    conflicts["pid_minus_aci"] = conflicts.pid_indicator - conflicts.aci_indicator

    conflict_summary = conflicts.groupby("forecaster_key", as_index=False).agg(
        windows=("window", "size"), static_conflicts=("static_indicator", "sum"),
        aci_conflicts=("aci_indicator", "sum"), pid_conflicts=("pid_indicator", "sum"),
        static_gcr=("static_indicator", "mean"), aci_gcr=("aci_indicator", "mean"), pid_gcr=("pid_indicator", "mean"),
    )

    demand_users = pd.concat(user_frames, ignore_index=True)
    family_methods = {
        "static": ("rolling_global_norm", "rolling_user_norm"),
        "aci": ("aci_global", "aci_user"),
        "pid": ("pid_global", "pid_user"),
    }
    demand_rows = []
    env_keys = ["dataset", "forecaster_key", "window"]
    for values, environment in demand_users.groupby(env_keys, sort=True):
        output = dict(zip(env_keys, values))
        for family, (global_method, user_method) in family_methods.items():
            family_metrics = {}
            for short, method in (("global", global_method), ("user", user_method)):
                subset = environment[environment.method == method]
                user_gap = float(np.mean(np.abs(subset.coverage.to_numpy(float) - .8)))
                grouped = subset.groupby("demand_tertile", sort=True).apply(
                    lambda z: pd.Series({"coverage": np.average(z.coverage.to_numpy(float), weights=z.n.to_numpy(float)),
                                         "n": z.n.to_numpy(float).sum()}), include_groups=False
                )
                gaps = np.abs(grouped.coverage.to_numpy(float) - .8)
                family_metrics[f"{short}_user_gap"] = user_gap
                family_metrics[f"{short}_max_group_gap"] = float(gaps.max())
                family_metrics[f"{short}_weighted_group_gap"] = float(np.average(gaps, weights=grouped.n.to_numpy(float)))
            du = family_metrics["user_user_gap"] - family_metrics["global_user_gap"]
            dm = family_metrics["user_max_group_gap"] - family_metrics["global_max_group_gap"]
            dw = family_metrics["user_weighted_group_gap"] - family_metrics["global_weighted_group_gap"]
            output.update({f"{family}_delta_user_gap": du, f"{family}_delta_max_group_gap": dm,
                           f"{family}_delta_weighted_group_gap": dw,
                           f"{family}_max_conflict": int(du < 0 and dm > 0),
                           f"{family}_weighted_conflict": int(du < 0 and dw > 0)})
        demand_rows.append(output)
    demand_conflicts = window_order(pd.DataFrame(demand_rows))
    if len(demand_conflicts) != 70:
        raise RuntimeError(f"demand-tertile online integrity failure: {len(demand_conflicts)}")
    demand_summary_rows = []
    for forecaster, item in demand_conflicts.groupby("forecaster_key", sort=True):
        row = {"forecaster_key": forecaster, "windows": len(item)}
        for family in family_methods:
            row[f"{family}_max_conflicts"] = int(item[f"{family}_max_conflict"].sum())
            row[f"{family}_max_gcr"] = float(item[f"{family}_max_conflict"].mean())
            row[f"{family}_weighted_conflicts"] = int(item[f"{family}_weighted_conflict"].sum())
            row[f"{family}_weighted_gcr"] = float(item[f"{family}_weighted_conflict"].mean())
        demand_summary_rows.append(row)
    demand_summary = pd.DataFrame(demand_summary_rows)

    inference_rows = []
    pair_columns = [f"delta_{metric}" for metric in METRICS] + ["relative_mpiw_change", "relative_winkler_interval_score_change"]
    for comparison, frame in pairs.groupby("comparison", sort=True):
        for block in (2, 3):
            estimate, low, high = synchronized_ci(frame, pair_columns, block, SEED + block + sum(map(ord, comparison)))
            for index, metric in enumerate(pair_columns):
                inference_rows.append({"kind": "paired_metric", "comparison": comparison, "metric": metric,
                                       "block_months": block, "estimate": estimate[index], "ci_low": low[index], "ci_high": high[index]})
    for comparison in ("aci_minus_static", "pid_minus_static", "pid_minus_aci"):
        for block in (2, 3):
            estimate, low, high = synchronized_ci(conflicts, [comparison], block, SEED + 1000 + block + sum(map(ord, comparison)))
            inference_rows.append({"kind": "gcr_risk_difference", "comparison": comparison, "metric": "delta_gcr",
                                   "block_months": block, "estimate": estimate[0], "ci_low": low[0], "ci_high": high[0]})
    inference = pd.DataFrame(inference_rows)

    panel.to_csv(args.output_root / "P0_ONLINE_PANEL.csv", index=False, encoding="utf-8-sig")
    pairs.to_csv(args.output_root / "P0_ONLINE_PAIR_DIFFERENCES.csv", index=False, encoding="utf-8-sig")
    conflicts.to_csv(args.output_root / "P0_ONLINE_CONFLICT_DIFFERENCES.csv", index=False, encoding="utf-8-sig")
    conflict_summary.to_csv(args.output_root / "P0_ONLINE_CONFLICT_SUMMARY.csv", index=False, encoding="utf-8-sig")
    demand_conflicts.to_csv(args.output_root / "P0_ONLINE_DEMAND_TERTILE_CONFLICTS.csv", index=False, encoding="utf-8-sig")
    demand_summary.to_csv(args.output_root / "P0_ONLINE_DEMAND_TERTILE_SUMMARY.csv", index=False, encoding="utf-8-sig")
    inference.to_csv(args.output_root / "P0_ONLINE_BLOCK_BOOTSTRAP.csv", index=False, encoding="utf-8-sig")
    report = {
        "protocol": "FROZEN_P0_DYNAMIC_BASELINE_ADDENDUM.md",
        "panel_rows": len(panel), "predictor_window_observations": 70,
        "methods": sorted(METHODS), "conflict_summary": conflict_summary.to_dict("records"),
        "demand_tertile_conflict_summary": demand_summary.to_dict("records"),
        "static_reproduction_audit": reproduction, "bootstrap_repetitions": REPS,
    }
    (args.output_root / "P0_ONLINE_REPORT.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# P0 online-baseline completion report", "",
        "Protocol: `FROZEN_P0_DYNAMIC_BASELINE_ADDENDUM.md`.", "",
        "## KM3 audit-group conflict counts", "",
        "| Forecaster | Windows | Static | ACI | PID-PI |", "|---|---:|---:|---:|---:|",
    ]
    for row in conflict_summary.to_dict("records"):
        lines.append(f"| {row['forecaster_key']} | {row['windows']} | {row['static_conflicts']} | {row['aci_conflicts']} | {row['pid_conflicts']} |")
    lines.extend(["", "## Demand-tertile conflict counts", "",
                  "| Forecaster | Static max | ACI max | PID max | Static weighted | ACI weighted | PID weighted |",
                  "|---|---:|---:|---:|---:|---:|---:|"])
    for row in demand_summary.to_dict("records"):
        lines.append(
            f"| {row['forecaster_key']} | {row['static_max_conflicts']} | {row['aci_max_conflicts']} | {row['pid_max_conflicts']} | "
            f"{row['static_weighted_conflicts']} | {row['aci_weighted_conflicts']} | {row['pid_weighted_conflicts']} |"
        )
    lines.extend(["", "## Integrity", "",
                  f"- The released panel contains {len(panel)} method-window rows over 70 predictor-month observations.",
                  f"- The largest static reproduction difference is {max(item['maximum_absolute_difference'] for item in reproduction):.3e}.",
                  "- Both forecasters are retained within synchronized calendar-month bootstrap blocks.", ""])
    (args.output_root / "P0_ONLINE_COMPLETION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(conflict_summary.to_string(index=False))


if __name__ == "__main__":
    main()
