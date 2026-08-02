"""Merge, verify, and summarize the complete LightGBM CSGR evaluation grid."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
REPOSITORY_LAYOUT = (HERE.parent / "protocols").is_dir()
REPOSITORY_ROOT = HERE.parent if REPOSITORY_LAYOUT else HERE
OUTPUT_ROOT = REPOSITORY_ROOT / "outputs" if REPOSITORY_LAYOUT else HERE
RESULT_ROOT = REPOSITORY_ROOT / "results" if REPOSITORY_LAYOUT else HERE
PROTOCOL_PATH = (
    REPOSITORY_ROOT / "protocols" / "FROZEN_LIGHTGBM_ROUTER_FULL_GRID_ADDENDUM_2026-08-02.md"
    if REPOSITORY_LAYOUT
    else HERE / "FROZEN_LIGHTGBM_ROUTER_FULL_GRID_ADDENDUM_2026-08-02.md"
)
POLICIES = (
    "rolling_global_norm",
    "rolling_group_norm",
    "rolling_user_norm",
)
GLOBAL, SEGMENT, USER = POLICIES
PRIMARY_USER_WEIGHT = 0.5
PRIMARY_EFFICIENCY_WEIGHT = 0.0
REPS = 10_000
SEED = 20260802
EXPECTED_ENVIRONMENTS = {"London": 44, "Ausgrid": 48, "UCI Electricity": 48}


def forecaster_name(configuration: str) -> str:
    if configuration.startswith("lightgbm_quantile"):
        return "LightGBM"
    if configuration.startswith("persistence_qi"):
        return "Persistence"
    raise RuntimeError(f"unknown forecaster configuration: {configuration}")


def scalar_fold_loss(
    frame: pd.DataFrame,
    user_weight: float = PRIMARY_USER_WEIGHT,
    efficiency_weight: float = PRIMARY_EFFICIENCY_WEIGHT,
) -> pd.Series:
    return (
        user_weight * frame["macro_user_abs_coverage_gap"]
        + (1.0 - user_weight) * frame["max_abs_cluster_coverage_gap"]
        + efficiency_weight * frame["normalized_interval_score"]
    )


def conservative_argmin(values: dict[str, float]) -> str:
    """Choose Global, then Segment, then User on an exact loss tie."""
    return min(POLICIES, key=lambda policy: (float(values[policy]), POLICIES.index(policy)))


def fold_choices(
    folds: pd.DataFrame,
    user_weight: float = PRIMARY_USER_WEIGHT,
    efficiency_weight: float = PRIMARY_EFFICIENCY_WEIGHT,
) -> pd.DataFrame:
    rows = []
    keys = ["dataset", "configuration", "coverage", "horizon_hours", "window", "window_order"]
    work = folds.copy()
    work["fold_loss"] = scalar_fold_loss(work, user_weight, efficiency_weight)
    for key, frame in work.groupby(keys, sort=False):
        pivot = frame.pivot(index="fold", columns="policy", values="fold_loss").reindex(
            columns=POLICIES
        )
        if pivot.shape != (3, 3) or pivot.isna().any().any():
            raise RuntimeError(f"incomplete chronological folds for {key}: {pivot.shape}")
        mean_values = pivot.mean(axis=0).to_dict()
        latest_values = pivot.loc[pivot.index.max()].to_dict()
        rows.append(
            {
                **dict(zip(keys, key)),
                "mean_fold_policy": conservative_argmin(mean_values),
                "latest_fold_policy": conservative_argmin(latest_values),
            }
        )
    return pd.DataFrame(rows)


def primary_events(
    results: pd.DataFrame,
    folds: pd.DataFrame,
    expected_decisions: int,
    user_weight: float = PRIMARY_USER_WEIGHT,
    efficiency_weight: float = PRIMARY_EFFICIENCY_WEIGHT,
) -> pd.DataFrame:
    events = results[
        np.isclose(results.user_weight, user_weight)
        & np.isclose(results.efficiency_weight, efficiency_weight)
    ].copy()
    if len(events) != expected_decisions:
        raise RuntimeError(
            f"expected {expected_decisions} primary decisions, got {len(events)}"
        )
    choices = fold_choices(folds, user_weight, efficiency_weight)
    keys = ["dataset", "configuration", "coverage", "horizon_hours", "window", "window_order"]
    events = events.merge(choices, on=keys, validate="one_to_one")
    events["forecaster"] = events.configuration.map(forecaster_name)
    return events.sort_values(
        ["forecaster", "dataset", "configuration", "window_order"]
    ).reset_index(drop=True)


def selected_loss(row: pd.Series, policy: str) -> float:
    return float(row[f"loss__{policy}"])


def add_sequential_choices(events: pd.DataFrame) -> pd.DataFrame:
    output = events.copy()
    output["previous_window_policy"] = GLOBAL
    output["follow_the_leader_policy"] = GLOBAL
    output["best_fixed_policy"] = ""
    for _, indices in output.groupby(["dataset", "configuration"], sort=False).groups.items():
        ordered = output.loc[indices].sort_values("window_order")
        cumulative = {policy: 0.0 for policy in POLICIES}
        previous = GLOBAL
        for position, (index, row) in enumerate(ordered.iterrows()):
            output.at[index, "previous_window_policy"] = previous
            leader = GLOBAL if position == 0 else conservative_argmin(cumulative)
            output.at[index, "follow_the_leader_policy"] = leader
            month_losses = {policy: selected_loss(row, policy) for policy in POLICIES}
            previous = conservative_argmin(month_losses)
            for policy in POLICIES:
                cumulative[policy] += month_losses[policy]
        fixed_means = {
            policy: float(ordered[f"loss__{policy}"].mean()) for policy in POLICIES
        }
        output.loc[ordered.index, "best_fixed_policy"] = conservative_argmin(fixed_means)
    return output


def long_strategy_frame(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in events.iterrows():
        policy_by_strategy = {
            "CSGR": row.selected_policy,
            "Mean-fold minimum": row.mean_fold_policy,
            "Latest-fold minimum": row.latest_fold_policy,
            "Previous-window winner": row.previous_window_policy,
            "Follow-the-leader": row.follow_the_leader_policy,
            "Global-CQR": GLOBAL,
            "Segment-CQR": SEGMENT,
            "User-CQR": USER,
            "Ex-post best fixed": row.best_fixed_policy,
            "Month-wise oracle": row.oracle_policy,
        }
        identity = {
            "forecaster": row.forecaster,
            "dataset": row.dataset,
            "configuration": row.configuration,
            "coverage": row.coverage,
            "horizon_hours": row.horizon_hours,
            "window": row.window,
            "window_order": row.window_order,
        }
        for strategy, policy in policy_by_strategy.items():
            rows.append(
                {
                    **identity,
                    "strategy": strategy,
                    "selected_policy": policy,
                    "loss": selected_loss(row, policy),
                }
            )
    return pd.DataFrame(rows)


def block_indices(n: int, block: int, reps: int, rng: np.random.Generator) -> np.ndarray:
    starts = rng.integers(0, n, size=(reps, int(np.ceil(n / block))))
    return (
        (starts[:, :, None] + np.arange(block)[None, None, :]) % n
    ).reshape(reps, -1)[:, :n]


def synchronized_ci(
    frame: pd.DataFrame,
    column: str,
    block: int,
    reps: int,
    seed: int,
    resample_datasets: bool,
) -> tuple[float, list[float]]:
    matrices: dict[str, np.ndarray] = {}
    for dataset, item in frame.groupby("dataset", sort=False):
        pivot = item.pivot(
            index="window_order", columns="configuration", values=column
        ).sort_index()
        if pivot.isna().any().any():
            raise RuntimeError(f"unaligned configurations for {dataset}")
        matrices[dataset] = pivot.to_numpy(float)
    names = list(matrices)
    point = float(np.mean([matrix.mean(axis=0).mean() for matrix in matrices.values()]))
    rng = np.random.default_rng(seed)
    if not resample_datasets:
        estimates = []
        for dataset in names:
            matrix = matrices[dataset]
            indices = block_indices(len(matrix), block, reps, rng)
            estimates.append(matrix[indices].mean(axis=(1, 2)))
        samples = np.mean(np.column_stack(estimates), axis=1)
    else:
        dataset_count = len(names)
        chosen = rng.integers(0, dataset_count, size=(reps, dataset_count))
        slot_estimates = np.empty((dataset_count, reps, dataset_count), dtype=float)
        for slot in range(dataset_count):
            for dataset_index, dataset in enumerate(names):
                matrix = matrices[dataset]
                indices = block_indices(len(matrix), block, reps, rng)
                slot_estimates[slot, :, dataset_index] = matrix[indices].mean(
                    axis=(1, 2)
                )
        repetition = np.arange(reps)
        samples = np.mean(
            np.column_stack(
                [
                    slot_estimates[slot, repetition, chosen[:, slot]]
                    for slot in range(dataset_count)
                ]
            ),
            axis=1,
        )
    return point, np.quantile(samples, [0.025, 0.975]).tolist()


def summarize_strategies_one_scope(long: pd.DataFrame, scope: str) -> pd.DataFrame:
    config_means = (
        long.groupby(["strategy", "dataset", "configuration"], sort=False)["loss"]
        .mean()
        .reset_index()
    )
    rows = []
    for strategy, frame in config_means.groupby("strategy", sort=False):
        selections = long[long.strategy == strategy].selected_policy.value_counts().to_dict()
        rows.append(
            {
                "strategy": strategy,
                "scope": scope,
                "configuration_macro_mean_loss": float(frame.loss.mean()),
                "selected_global": int(selections.get(GLOBAL, 0)),
                "selected_segment": int(selections.get(SEGMENT, 0)),
                "selected_user": int(selections.get(USER, 0)),
            }
        )
    return pd.DataFrame(rows).sort_values("configuration_macro_mean_loss")


def summarize_strategies(long: pd.DataFrame) -> pd.DataFrame:
    frames = [
        summarize_strategies_one_scope(long[long.forecaster == "LightGBM"], "LightGBM"),
        summarize_strategies_one_scope(
            long[long.forecaster == "Persistence"], "Persistence"
        ),
        summarize_strategies_one_scope(long, "Combined"),
    ]
    return pd.concat(frames, ignore_index=True)


def summarize_configurations(long: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "forecaster",
        "dataset",
        "configuration",
        "coverage",
        "horizon_hours",
    ]
    means = long.pivot_table(
        index=keys, columns="strategy", values="loss", aggfunc="mean"
    ).reset_index()
    months = (
        long[long.strategy == "CSGR"]
        .groupby(keys, sort=False)
        .size()
        .rename("months")
        .reset_index()
    )
    selections = (
        long[long.strategy == "CSGR"]
        .pivot_table(
            index=keys,
            columns="selected_policy",
            values="window",
            aggfunc="count",
            fill_value=0,
        )
        .reset_index()
        .rename(
            columns={
                GLOBAL: "selected_global",
                SEGMENT: "selected_segment",
                USER: "selected_user",
            }
        )
    )
    output = means.merge(months, on=keys, validate="one_to_one").merge(
        selections, on=keys, validate="one_to_one"
    )
    for column in ("selected_global", "selected_segment", "selected_user"):
        if column not in output:
            output[column] = 0
        output[column] = output[column].astype(int)
    output["CSGR minus Ex-post best fixed"] = (
        output["CSGR"] - output["Ex-post best fixed"]
    )
    return output.sort_values(keys).reset_index(drop=True)


def summarize_strata(long: pd.DataFrame) -> pd.DataFrame:
    config_means = (
        long.groupby(
            [
                "forecaster",
                "coverage",
                "horizon_hours",
                "dataset",
                "configuration",
                "strategy",
            ],
            sort=False,
        )["loss"]
        .mean()
        .reset_index()
    )
    means = (
        config_means.groupby(
            ["forecaster", "coverage", "horizon_hours", "strategy"], sort=False
        )["loss"]
        .mean()
        .reset_index()
    )
    selections = (
        long[long.strategy == "CSGR"]
        .groupby(["forecaster", "coverage", "horizon_hours"])["selected_policy"]
        .value_counts()
        .unstack(fill_value=0)
        .reset_index()
        .rename(
            columns={
                GLOBAL: "selected_global",
                SEGMENT: "selected_segment",
                USER: "selected_user",
            }
        )
    )
    output = means.merge(
        selections,
        on=["forecaster", "coverage", "horizon_hours"],
        how="left",
        validate="many_to_one",
    )
    for column in ("selected_global", "selected_segment", "selected_user"):
        if column not in output:
            output[column] = 0
        output[column] = output[column].astype(int)
    return output.sort_values(
        ["forecaster", "coverage", "horizon_hours", "loss"]
    ).reset_index(drop=True)


def pairwise_inference_one_scope(long: pd.DataFrame, scope: str) -> pd.DataFrame:
    identity = ["dataset", "configuration", "window", "window_order"]
    wide = long.pivot(index=identity, columns="strategy", values="loss").reset_index()
    comparisons = [
        ("CSGR", "Ex-post best fixed"),
        ("CSGR", "Previous-window winner"),
        ("CSGR", "Follow-the-leader"),
        ("CSGR", "Mean-fold minimum"),
        ("CSGR", "Latest-fold minimum"),
        ("CSGR", "Global-CQR"),
    ]
    rows = []
    for number, (left, right) in enumerate(comparisons):
        contrast = wide[identity].copy()
        contrast["delta"] = wide[left] - wide[right]
        for block in (2, 3):
            for hierarchical in (False, True):
                point, interval = synchronized_ci(
                    contrast,
                    "delta",
                    block,
                    REPS,
                    SEED + number * 1000 + block * 100 + int(hierarchical),
                    hierarchical,
                )
                rows.append(
                    {
                        "left": left,
                        "right": right,
                        "scope": scope,
                        "contrast": f"{left} minus {right}",
                        "block_months": block,
                        "resample_datasets": hierarchical,
                        "mean_delta": point,
                        "ci95_low": interval[0],
                        "ci95_high": interval[1],
                    }
                )
    return pd.DataFrame(rows)


def pairwise_inference(long: pd.DataFrame) -> pd.DataFrame:
    frames = [
        pairwise_inference_one_scope(long[long.forecaster == "LightGBM"], "LightGBM"),
        pairwise_inference_one_scope(
            long[long.forecaster == "Persistence"], "Persistence"
        ),
        pairwise_inference_one_scope(long, "Combined"),
    ]
    return pd.concat(frames, ignore_index=True)


def reproduce_existing_cell(results: pd.DataFrame, reference_dir: Path) -> dict:
    reference = pd.read_csv(reference_dir / "router_window_results.csv")
    candidate = results[
        np.isclose(results.coverage, 0.8) & np.isclose(results.horizon_hours, 1.0)
    ].copy()
    keys = [
        "dataset",
        "configuration",
        "window",
        "window_order",
        "user_weight",
        "efficiency_weight",
    ]
    merged = reference.merge(candidate, on=keys, suffixes=("__reference", "__candidate"), validate="one_to_one")
    if len(merged) != 350:
        raise RuntimeError(f"80%/1 h reproduction alignment has {len(merged)} rows")
    selection_mismatches = int(
        (merged.selected_policy__reference != merged.selected_policy__candidate).sum()
    )
    numeric_columns = [
        "router_loss",
        "oracle_loss",
        *[f"loss__{policy}" for policy in POLICIES],
        *[f"cv_mean_gain_over_global__{policy}" for policy in POLICIES],
        *[f"cv_lower_gain__{policy}" for policy in POLICIES],
    ]
    differences = {}
    for column in numeric_columns:
        differences[column] = float(
            np.max(
                np.abs(
                    merged[f"{column}__reference"].to_numpy(float)
                    - merged[f"{column}__candidate"].to_numpy(float)
                )
            )
        )
    maximum = max(differences.values())
    report = {
        "aligned_rows": len(merged),
        "selection_mismatches": selection_mismatches,
        "maximum_absolute_numeric_difference": maximum,
        "column_maximum_absolute_differences": differences,
        "tolerance": 1e-8,
        "passed": selection_mismatches == 0 and maximum <= 1e-8,
    }
    if not report["passed"]:
        raise RuntimeError(f"existing LightGBM cell did not reproduce: {report}")
    return report


def reproduce_static_full_grid(results: pd.DataFrame, static_panel_path: Path) -> dict:
    candidate = results[
        np.isclose(results.user_weight, PRIMARY_USER_WEIGHT)
        & np.isclose(results.efficiency_weight, PRIMARY_EFFICIENCY_WEIGHT)
    ].copy()
    static = pd.read_csv(static_panel_path)
    static = static[static.forecaster == "lightgbm_quantile"].copy()
    keys = ["dataset", "coverage", "horizon_hours", "window", "window_order"]
    merged = static.merge(
        candidate,
        on=keys,
        suffixes=("__static", "__router"),
        validate="one_to_one",
    )
    if len(merged) != 140:
        raise RuntimeError(f"static LightGBM full-grid alignment has {len(merged)} rows")
    metric_prefixes = {
        GLOBAL: "global",
        SEGMENT: "segment",
        USER: "user",
    }
    differences = {}
    for policy, prefix in metric_prefixes.items():
        expected = 0.5 * (
            merged[f"{prefix}_macro_user_abs_coverage_gap"]
            + merged[f"{prefix}_max_abs_cluster_coverage_gap"]
        )
        differences[policy] = float(
            np.max(
                np.abs(
                    expected.to_numpy(float)
                    - merged[f"loss__{policy}"].to_numpy(float)
                )
            )
        )
    maximum = max(differences.values())
    report = {
        "aligned_rows": len(merged),
        "maximum_absolute_loss_difference": maximum,
        "policy_maximum_absolute_loss_differences": differences,
        "tolerance": 1e-8,
        "passed": maximum <= 1e-8,
    }
    if not report["passed"]:
        raise RuntimeError(f"static LightGBM full grid did not reproduce: {report}")
    return report


def write_completion_report(
    path: Path,
    summary: pd.DataFrame,
    inference: pd.DataFrame,
    reproduction: dict,
    static_reproduction: dict,
    width_summary: pd.DataFrame,
    width_inference: pd.DataFrame,
) -> None:
    lines = [
        "# Complete-grid CSGR evaluation report",
        "",
        "## Integrity",
        "",
        f"- Environments: 140 (3 data sets x 2 coverages x 2 horizons x 11/12/12 months).",
        f"- Existing 80%/1 h rows reproduced: {reproduction['aligned_rows']}.",
        f"- Selection mismatches: {reproduction['selection_mismatches']}.",
        f"- Maximum absolute numeric difference: {reproduction['maximum_absolute_numeric_difference']:.3g}.",
        f"- Accepted static LightGBM grid rows reproduced: {static_reproduction['aligned_rows']}.",
        f"- Maximum static-grid loss difference: {static_reproduction['maximum_absolute_loss_difference']:.3g}.",
        "",
    ]
    for scope in ("LightGBM", "Persistence", "Combined"):
        scoped_summary = summary[summary.scope == scope]
        loss = dict(
            zip(scoped_summary.strategy, scoped_summary.configuration_macro_mean_loss)
        )
        primary_ci = inference[
            (inference.scope == scope)
            & (inference.left == "CSGR")
            & (inference.right == "Ex-post best fixed")
            & (inference.block_months == 2)
            & (~inference.resample_datasets)
        ].iloc[0]
        hierarchical_ci = inference[
            (inference.scope == scope)
            & (inference.left == "CSGR")
            & (inference.right == "Ex-post best fixed")
            & (inference.block_months == 2)
            & (inference.resample_datasets)
        ].iloc[0]
        headroom = loss["Ex-post best fixed"] - loss["Month-wise oracle"]
        recovered = (
            100.0 * (loss["Ex-post best fixed"] - loss["CSGR"]) / headroom
            if headroom > 0
            else float("nan")
        )
        lines.extend(
            [
                f"## {scope}: primary equal-preference result",
                "",
                f"- CSGR loss: {loss['CSGR']:.6f}.",
                f"- Ex-post best fixed loss: {loss['Ex-post best fixed']:.6f}.",
                f"- Previous-window winner loss: {loss['Previous-window winner']:.6f}.",
                f"- Follow-the-leader loss: {loss['Follow-the-leader']:.6f}.",
                f"- Mean-fold minimum loss: {loss['Mean-fold minimum']:.6f}.",
                f"- Latest-fold minimum loss: {loss['Latest-fold minimum']:.6f}.",
                f"- Global-CQR loss: {loss['Global-CQR']:.6f}.",
                f"- Month-wise oracle loss: {loss['Month-wise oracle']:.6f}.",
                f"- Best-fixed--oracle headroom recovered by CSGR: {recovered:.1f}%.",
                f"- CSGR minus best fixed, synchronized block-2 95% interval: "
                f"[{primary_ci.ci95_low:.6f}, {primary_ci.ci95_high:.6f}].",
                f"- Data-set hierarchical block-2 interval: "
                f"[{hierarchical_ci.ci95_low:.6f}, {hierarchical_ci.ci95_high:.6f}].",
                "",
            ]
        )
    lines.append(
        "All comparisons were retained regardless of direction; this run does not retune CSGR."
    )
    lines.extend(["", "## Width-aware sensitivity ($eta=0.01$)", ""])
    for scope in ("LightGBM", "Persistence", "Combined"):
        scoped_summary = width_summary[width_summary.scope == scope]
        loss = dict(
            zip(scoped_summary.strategy, scoped_summary.configuration_macro_mean_loss)
        )
        interval = width_inference[
            (width_inference.scope == scope)
            & (width_inference.left == "CSGR")
            & (width_inference.right == "Ex-post best fixed")
            & (width_inference.block_months == 2)
            & (~width_inference.resample_datasets)
        ].iloc[0]
        lines.append(
            f"- {scope}: CSGR {loss['CSGR']:.6f}, best fixed "
            f"{loss['Ex-post best fixed']:.6f}, delta {loss['CSGR'] - loss['Ex-post best fixed']:.6f}, "
            f"interval [{interval.ci95_low:.6f}, {interval.ci95_high:.6f}]."
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=OUTPUT_ROOT / "lightgbm_router_full_grid_2026-08-02",
    )
    parser.add_argument(
        "--reference-dir", type=Path, default=OUTPUT_ROOT / "lightgbm_router_validation"
    )
    parser.add_argument(
        "--persistence-dir", type=Path, default=OUTPUT_ROOT / "generalized_router_validation"
    )
    parser.add_argument(
        "--static-panel", type=Path, default=RESULT_ROOT / "BALANCED_FULL_GRID_PANEL.csv"
    )
    args = parser.parse_args()
    source_dirs = [args.run_dir / name for name in ("london", "ausgrid", "uci")]
    results = pd.concat(
        [pd.read_csv(path / "router_window_results.csv") for path in source_dirs],
        ignore_index=True,
    )
    folds = pd.concat(
        [pd.read_csv(path / "chronological_fold_metrics.csv") for path in source_dirs],
        ignore_index=True,
    )
    environment_counts = (
        results[["dataset", "configuration", "window"]]
        .drop_duplicates()
        .groupby("dataset")
        .size()
        .to_dict()
    )
    if environment_counts != EXPECTED_ENVIRONMENTS:
        raise RuntimeError(f"unexpected environment counts: {environment_counts}")
    if len(results) != 1400 or len(folds) != 1260:
        raise RuntimeError(f"unexpected merged shapes: results={len(results)}, folds={len(folds)}")

    reproduction = reproduce_existing_cell(results, args.reference_dir)
    static_reproduction = reproduce_static_full_grid(results, args.static_panel)
    lightgbm_events = primary_events(results, folds, 140)
    persistence_results = pd.read_csv(args.persistence_dir / "router_window_results.csv")
    persistence_folds = pd.read_csv(args.persistence_dir / "chronological_fold_metrics.csv")
    persistence_events = primary_events(persistence_results, persistence_folds, 140)
    events = add_sequential_choices(
        pd.concat([lightgbm_events, persistence_events], ignore_index=True)
    )
    long = long_strategy_frame(events)
    summary = summarize_strategies(long)
    configuration_summary = summarize_configurations(long)
    stratum_summary = summarize_strata(long)
    inference = pairwise_inference(long)

    width_lightgbm_events = primary_events(results, folds, 140, 0.5, 0.01)
    width_persistence_events = primary_events(
        persistence_results, persistence_folds, 140, 0.5, 0.01
    )
    width_events = add_sequential_choices(
        pd.concat(
            [width_lightgbm_events, width_persistence_events], ignore_index=True
        )
    )
    width_long = long_strategy_frame(width_events)
    width_summary = summarize_strategies(width_long)
    width_inference = pairwise_inference(width_long)

    combined = args.run_dir / "combined"
    combined.mkdir(parents=True, exist_ok=True)
    results.to_csv(combined / "router_window_results.csv", index=False, encoding="utf-8-sig")
    folds.to_csv(combined / "chronological_fold_metrics.csv", index=False, encoding="utf-8-sig")
    events.to_csv(combined / "primary_event_choices.csv", index=False, encoding="utf-8-sig")
    long.to_csv(combined / "primary_strategy_window_results.csv", index=False, encoding="utf-8-sig")
    width_events.to_csv(
        combined / "width_aware_event_choices.csv", index=False, encoding="utf-8-sig"
    )
    width_long.to_csv(
        combined / "width_aware_strategy_window_results.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary.to_csv(
        combined / "FULL_ROUTER_BASELINE_SUMMARY.csv",
        index=False,
        encoding="utf-8-sig",
    )
    configuration_summary.to_csv(
        combined / "FULL_ROUTER_CONFIGURATION_SUMMARY.csv",
        index=False,
        encoding="utf-8-sig",
    )
    stratum_summary.to_csv(
        combined / "FULL_ROUTER_COVERAGE_HORIZON_SUMMARY.csv",
        index=False,
        encoding="utf-8-sig",
    )
    inference.to_csv(
        combined / "FULL_ROUTER_PAIRWISE_CI.csv",
        index=False,
        encoding="utf-8-sig",
    )
    width_summary.to_csv(
        combined / "FULL_ROUTER_WIDTH_AWARE_BASELINE_SUMMARY.csv",
        index=False,
        encoding="utf-8-sig",
    )
    width_inference.to_csv(
        combined / "FULL_ROUTER_WIDTH_AWARE_PAIRWISE_CI.csv",
        index=False,
        encoding="utf-8-sig",
    )
    integrity = {
        "protocol": PROTOCOL_PATH.name,
        "protocol_sha256": hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest(),
        "protocol_modified_time": PROTOCOL_PATH.stat().st_mtime,
        "environment_counts": environment_counts,
        "result_rows": len(results),
        "fold_rows": len(folds),
        "primary_event_rows": len(events),
        "strategy_window_rows": len(long),
        "width_aware_event_rows": len(width_events),
        "width_aware_strategy_window_rows": len(width_long),
        "bootstrap_reps": REPS,
        "seed": SEED,
        "reproduction": reproduction,
        "static_full_grid_reproduction": static_reproduction,
        "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (combined / "LIGHTGBM_ROUTER_FULL_GRID_INTEGRITY.json").write_text(
        json.dumps(integrity, indent=2), encoding="utf-8"
    )
    write_completion_report(
        combined / "LIGHTGBM_ROUTER_FULL_GRID_COMPLETION_REPORT.md",
        summary,
        inference,
        reproduction,
        static_reproduction,
        width_summary,
        width_inference,
    )
    print(json.dumps(integrity, indent=2))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
