"""Frozen information-regime comparison for static CQR, ACI and CSGR.

ACI and CSGR are intentionally not ranked as peers: the former assumes
within-month label feedback while the latter is a monthly static-policy router.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
DATASETS = ("london", "ausgrid", "uci")
FORECASTERS = ("lgbm", "persistence")
STATIC = ("rolling_global_norm", "rolling_group_norm", "rolling_user_norm")
ACI = ("aci_global", "aci_segment", "aci_user")
METRICS = (
    "picp", "mpiw", "winkler_interval_score",
    "macro_user_abs_coverage_gap", "max_abs_cluster_coverage_gap", "joint_loss",
)
PAIR_COMPARISONS = (
    ("O", "aci_global_vs_static_global", "aci_global", "static_global"),
    ("O", "aci_segment_vs_static_segment", "aci_segment", "static_segment"),
    ("O", "aci_user_vs_static_user", "aci_user", "static_user"),
    ("M", "csgr_vs_static_best_fixed", "csgr", "static_best_fixed"),
    ("M", "csgr_vs_static_global", "csgr", "static_global"),
)
REPS = 10_000
SEED = 20260729


def policy_name(method: str) -> str:
    return {
        "rolling_global_norm": "static_global",
        "rolling_group_norm": "static_segment",
        "rolling_user_norm": "static_user",
    }[method]


def add_loss(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["joint_loss"] = .5 * frame["macro_user_abs_coverage_gap"] + .5 * frame["max_abs_cluster_coverage_gap"]
    return frame


def window_order(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    order = frame[["dataset", "window"]].drop_duplicates().sort_values(["dataset", "window"])
    order["window_order"] = order.groupby("dataset").cumcount()
    return frame.merge(order, on=["dataset", "window"], validate="many_to_one")


def load_aci() -> pd.DataFrame:
    frames = []
    for forecaster in FORECASTERS:
        for dataset in DATASETS:
            path = HERE / f"aci_{forecaster}_{dataset}" / "window_metrics.csv"
            frame = pd.read_csv(path)
            expected = {"raw", *STATIC, *ACI}
            if set(frame.method) != expected:
                raise RuntimeError(f"incomplete ACI methods: {path}")
            frame["forecaster_key"] = forecaster
            frames.append(frame)
    frame = pd.concat(frames, ignore_index=True)
    if len(frame) != 490:
        raise RuntimeError(f"ACI primary panel should have 490 rows, got {len(frame)}")
    if not frame[list(METRICS[:-1])].apply(pd.to_numeric, errors="coerce").notna().all().all():
        raise RuntimeError("non-finite ACI metrics")
    return window_order(add_loss(frame))


def static_event_rows(aci: pd.DataFrame) -> pd.DataFrame:
    static = aci[aci.method.isin(STATIC)].copy()
    static["method_name"] = static.method.map(policy_name)
    return static


def materialize_fixed_oracle(static: pd.DataFrame, scenario: str, prefix: str) -> pd.DataFrame:
    """Materialize hindsight fixed and per-window oracle rows from candidates."""
    keys = ["dataset", "forecaster_key"]
    policy_loss = static.groupby(keys + ["method_name"], as_index=False).joint_loss.mean()
    fixed_choice = policy_loss.loc[policy_loss.groupby(keys).joint_loss.idxmin(), keys + ["method_name"]]
    fixed = static.merge(fixed_choice, on=keys + ["method_name"], validate="many_to_one")
    fixed = fixed.copy(); fixed["method_name"] = f"{prefix}_best_fixed"; fixed["scenario"] = scenario
    oracle = static.loc[static.groupby(["dataset", "forecaster_key", "window"]).joint_loss.idxmin()].copy()
    oracle["method_name"] = f"{prefix}_oracle"; oracle["scenario"] = scenario
    return pd.concat([fixed, oracle], ignore_index=True)


def load_csgr(static: pd.DataFrame) -> pd.DataFrame:
    rows = []
    paths = {
        "persistence": HERE / "generalized_router_validation" / "router_window_results.csv",
        "lgbm": HERE / "lightgbm_router_validation" / "router_window_results.csv",
    }
    static_index = static.set_index(["dataset", "forecaster_key", "window", "method"])
    for forecaster, path in paths.items():
        raw = pd.read_csv(path)
        chosen = raw[(raw.user_weight == .5) & (raw.efficiency_weight == 0.)].copy()
        chosen = chosen[(chosen.coverage == .8) & (chosen.horizon_hours == 1.)].copy()
        if len(chosen) != 35:
            raise RuntimeError(f"CSGR primary rows should be 35 in {path}, got {len(chosen)}")
        for _, row in chosen.iterrows():
            key = (row.dataset, forecaster, row.window, row.selected_policy)
            base = static_index.loc[key].to_dict()
            result = {**base, "dataset": row.dataset, "forecaster_key": forecaster, "window": row.window,
                      "method_name": "csgr", "scenario": "M", "selected_static_policy": row.selected_policy,
                      "router_loss_audit": float(row.router_loss)}
            if abs(result["joint_loss"] - float(row.router_loss)) > 1e-12:
                raise RuntimeError(f"CSGR loss mismatch for {key}")
            rows.append(result)
    return pd.DataFrame(rows)


def build_panel(aci: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    static = static_event_rows(aci)
    base_columns = list(aci.columns)
    monthly_static = static.copy(); monthly_static["scenario"] = "M"
    monthly_extra = materialize_fixed_oracle(static, "M", "static")
    csgr = load_csgr(static)
    online_static = static.copy(); online_static["scenario"] = "O"
    online_aci = aci[aci.method.isin(ACI)].copy(); online_aci["method_name"] = online_aci.method
    online_aci["scenario"] = "O"
    online_extra = materialize_fixed_oracle(online_aci, "O", "aci")
    panel = pd.concat([monthly_static, monthly_extra, csgr, online_static, online_aci, online_extra], ignore_index=True, sort=False)
    columns = ["scenario", "method_name", "dataset", "forecaster_key", "window", "window_order", *METRICS,
               "selected_static_policy", "router_loss_audit"]
    panel = panel[[column for column in columns if column in panel]].copy()
    expected = 70 * (6 + 8)
    if len(panel) != expected or panel[list(METRICS)].isna().any().any():
        raise RuntimeError(f"unified panel integrity failed: rows={len(panel)}, expected={expected}")
    return panel, static


def pair_differences(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    key = ["dataset", "forecaster_key", "window", "window_order"]
    for scenario, label, target, base in PAIR_COMPARISONS:
        subset = panel[panel.scenario == scenario]
        left = subset[subset.method_name == target].set_index(key)
        right = subset[subset.method_name == base].set_index(key)
        joined = left.join(right, lsuffix="_target", rsuffix="_base", how="inner", validate="one_to_one")
        if len(joined) != 70:
            raise RuntimeError(f"unmatched pair {label}: {len(joined)}")
        for index, row in joined.reset_index().iterrows():
            output = {column: row[column] for column in key}
            output.update({"scenario": scenario, "comparison": label, "target": target, "base": base})
            for metric in METRICS:
                output[f"delta_{metric}"] = float(row[f"{metric}_target"] - row[f"{metric}_base"])
            output["relative_mpiw_change"] = float(output["delta_mpiw"] / row["mpiw_base"])
            output["relative_winkler_interval_score_change"] = float(output["delta_winkler_interval_score"] / row["winkler_interval_score_base"])
            rows.append(output)
    return pd.DataFrame(rows)


def conflict_differences(panel: pd.DataFrame) -> pd.DataFrame:
    def conflict(frame: pd.DataFrame, user: str, global_: str) -> pd.DataFrame:
        key = ["dataset", "forecaster_key", "window", "window_order"]
        left = frame[frame.method_name == user].set_index(key)
        right = frame[frame.method_name == global_].set_index(key)
        joined = left.join(right, lsuffix="_user", rsuffix="_global", validate="one_to_one")
        output = joined.reset_index()[key].copy()
        output["indicator"] = ((joined.macro_user_abs_coverage_gap_user < joined.macro_user_abs_coverage_gap_global)
                               & (joined.max_abs_cluster_coverage_gap_user > joined.max_abs_cluster_coverage_gap_global)).astype(int).to_numpy()
        return output
    online = panel[panel.scenario == "O"]
    monthly = panel[panel.scenario == "M"]
    static_online = conflict(online, "static_user", "static_global").rename(columns={"indicator": "static_indicator"})
    aci_online = conflict(online, "aci_user", "aci_global").rename(columns={"indicator": "aci_indicator"})
    static_monthly = conflict(monthly, "static_user", "static_global").rename(columns={"indicator": "static_indicator"})
    csgr_monthly = conflict(monthly, "csgr", "static_global").rename(columns={"indicator": "csgr_indicator"})
    key = ["dataset", "forecaster_key", "window", "window_order"]
    first = static_online.merge(aci_online, on=key, validate="one_to_one")
    first["comparison"] = "aci_gcr_minus_static_gcr"; first["delta_indicator"] = first.aci_indicator - first.static_indicator
    second = static_monthly.merge(csgr_monthly, on=key, validate="one_to_one")
    second["comparison"] = "csgr_gcr_minus_static_user_gcr"; second["delta_indicator"] = second.csgr_indicator - second.static_indicator
    return pd.concat([first[key + ["comparison", "static_indicator", "aci_indicator", "delta_indicator"]],
                      second[key + ["comparison", "static_indicator", "csgr_indicator", "delta_indicator"]]], ignore_index=True, sort=False)


def block_index_matrix(n: int, block: int, reps: int, rng: np.random.Generator) -> np.ndarray:
    """Vectorized circular moving-block indices with one row per bootstrap draw."""
    blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(reps, blocks))
    offsets = np.arange(block, dtype=int)
    return ((starts[:, :, None] + offsets[None, None, :]) % n).reshape(reps, -1)[:, :n]


def vector_ci_synchronized(frame: pd.DataFrame, columns: list[str], block: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Jointly resample calendar months, retaining both forecasters per month."""
    rng = np.random.default_rng(seed)
    sums = np.zeros((REPS, len(columns)), dtype=float)
    total = 0
    for _, item in frame.groupby("dataset", sort=True):
        ordered = item.sort_values(["window_order", "forecaster_key"])
        n = ordered.window_order.nunique()
        matrix = ordered[columns].to_numpy(float)
        if len(matrix) != n * 2 or not np.isfinite(matrix).all():
            raise RuntimeError("synchronous bootstrap alignment failure")
        matrix = matrix.reshape(n, 2, len(columns))
        indices = block_index_matrix(n, block, REPS, rng)
        sums += matrix[indices].sum(axis=(1, 2))
        total += n * 2
    samples = sums / total
    return frame[columns].mean().to_numpy(float), np.quantile(samples, .025, axis=0), np.quantile(samples, .975, axis=0)


def vector_ci_single(frame: pd.DataFrame, columns: list[str], block: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = frame.sort_values("window_order")[columns].to_numpy(float)
    if not np.isfinite(values).all():
        raise RuntimeError("non-finite single-series bootstrap input")
    rng = np.random.default_rng(seed)
    indices = block_index_matrix(len(values), block, REPS, rng)
    samples = values[indices].mean(axis=1)
    return values.mean(axis=0), np.quantile(samples, .025, axis=0), np.quantile(samples, .975, axis=0)


def bootstrap_summaries(pairs: pd.DataFrame, conflicts: pd.DataFrame) -> pd.DataFrame:
    outputs = []
    pair_columns = [f"delta_{metric}" for metric in METRICS] + ["relative_mpiw_change", "relative_winkler_interval_score_change"]
    for comparison, frame in pairs.groupby("comparison", sort=True):
        for block in (2, 3):
            estimate, low, high = vector_ci_synchronized(frame, pair_columns, block, SEED + 100 * block + sum(map(ord, comparison)))
            for index, column in enumerate(pair_columns):
                outputs.append({"kind": "paired_metric", "comparison": comparison, "stratum": "all_synchronized",
                                "metric": column, "block_months": block, "estimate": estimate[index], "ci_low": low[index], "ci_high": high[index]})
        for (dataset, forecaster), item in frame.groupby(["dataset", "forecaster_key"], sort=True):
            for block in (2, 3):
                estimate, low, high = vector_ci_single(item, pair_columns, block, SEED + 1000 * block + sum(map(ord, f"{comparison}|{dataset}|{forecaster}")))
                for index, column in enumerate(pair_columns):
                    outputs.append({"kind": "paired_metric", "comparison": comparison, "stratum": f"{dataset}|{forecaster}",
                                    "metric": column, "block_months": block, "estimate": estimate[index], "ci_low": low[index], "ci_high": high[index]})
    for comparison, frame in conflicts.groupby("comparison", sort=True):
        for block in (2, 3):
            estimate, low, high = vector_ci_synchronized(frame, ["delta_indicator"], block, SEED + 3000 * block + sum(map(ord, comparison)))
            outputs.append({"kind": "gcr_risk_difference", "comparison": comparison, "stratum": "all_synchronized",
                            "metric": "delta_gcr", "block_months": block, "estimate": estimate[0], "ci_low": low[0], "ci_high": high[0]})
        for (dataset, forecaster), item in frame.groupby(["dataset", "forecaster_key"], sort=True):
            for block in (2, 3):
                estimate, low, high = vector_ci_single(item, ["delta_indicator"], block, SEED + 4000 * block + sum(map(ord, f"{comparison}|{dataset}|{forecaster}")))
                outputs.append({"kind": "gcr_risk_difference", "comparison": comparison, "stratum": f"{dataset}|{forecaster}",
                                "metric": "delta_gcr", "block_months": block, "estimate": estimate[0], "ci_low": low[0], "ci_high": high[0]})
    return pd.DataFrame(outputs)


def main() -> None:
    aci = load_aci()
    panel, static = build_panel(aci)
    pairs = pair_differences(panel)
    conflicts = conflict_differences(panel)
    inference = bootstrap_summaries(pairs, conflicts)
    panel.to_csv(HERE / "INFORMATION_REGIME_UNIFIED_PANEL.csv", index=False, encoding="utf-8-sig")
    pairs.to_csv(HERE / "INFORMATION_REGIME_PAIR_DIFFERENCES.csv", index=False, encoding="utf-8-sig")
    conflicts.to_csv(HERE / "INFORMATION_REGIME_CONFLICT_DIFFERENCES.csv", index=False, encoding="utf-8-sig")
    inference.to_csv(HERE / "INFORMATION_REGIME_BLOCK_BOOTSTRAP.csv", index=False, encoding="utf-8-sig")
    report = {
        "protocol": "FROZEN_INFORMATION_REGIME_COMPARISON_PROTOCOL.md",
        "unique_dataset_month_windows": 35,
        "primary_predictor_window_observations": 70,
        "unified_panel_rows": len(panel),
        "pair_rows": len(pairs),
        "bootstrap_repetitions": REPS,
        "block_lengths_months": [2, 3],
        "interpretation": "ACI and CSGR are reported under distinct label-feedback regimes; no cross-regime superiority claim is made.",
    }
    (HERE / "INFORMATION_REGIME_REPORT.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
