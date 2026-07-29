"""Frozen minimal CSGR ablation using existing chronological-fold artifacts.

No base model, calibration block or test-month prediction is rerun.  The script
only applies predeclared selection rules to frozen fold metrics and static test
metrics, retaining every variant specified in the protocol.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
POLICIES = ("rolling_global_norm", "rolling_group_norm", "rolling_user_norm")
GLOBAL = POLICIES[0]
THRESHOLDS = (0.0, .5, 1.0, 2.0)
EFFICIENCY = (0.0, .01)
REPS, SEED = 10_000, 20260729


def loss(frame: pd.DataFrame, eta: float) -> pd.Series:
    return .5 * frame.macro_user_abs_coverage_gap + .5 * frame.max_abs_cluster_coverage_gap + eta * frame.normalized_interval_score


def block_indices(n: int, block: int, reps: int, rng: np.random.Generator) -> np.ndarray:
    starts = rng.integers(0, n, size=(reps, int(np.ceil(n / block))))
    return ((starts[:, :, None] + np.arange(block)[None, None, :]) % n).reshape(reps, -1)[:, :n]


def source_paths(source: str) -> tuple[Path, Path]:
    if source == "persistence":
        return HERE / "generalized_router_validation", HERE / "SCORED_ROBUSTNESS_PANEL.csv"
    if source == "lgbm":
        return HERE / "lightgbm_router_validation", HERE / "SCORED_ROBUSTNESS_PANEL.csv"
    raise ValueError(source)


def static_metrics(source: str) -> pd.DataFrame:
    _, path = source_paths(source)
    wide = pd.read_csv(path)
    forecaster = "persistence_quantile_interval" if source == "persistence" else "lightgbm_quantile"
    wide = wide[wide.forecaster == forecaster].copy()
    if source == "lgbm":
        wide = wide[(wide.coverage == .8) & (wide.horizon_hours == 1.)].copy()
    expected = 140 if source == "persistence" else 35
    if len(wide) != expected:
        raise RuntimeError(f"expected {expected} {source} static rows, got {len(wide)}")
    rows = []
    prefixes = {"rolling_global_norm": "global", "rolling_group_norm": "segment", "rolling_user_norm": "user"}
    for _, item in wide.iterrows():
        for policy, prefix in prefixes.items():
            rows.append({
                "dataset": item.dataset,
                "configuration": f"persistence_qi__c{int(item.coverage * 100)}__h{item.horizon_hours:g}" if source == "persistence" else "lightgbm_quantile__c80__h1",
                "window": item.window,
                "policy": policy, "picp": item[f"{prefix}_picp"], "mpiw": item[f"{prefix}_mpiw"],
                "winkler_interval_score": item[f"{prefix}_winkler_interval_score"],
                "macro_user_abs_coverage_gap": item[f"{prefix}_macro_user_abs_coverage_gap"],
                "max_abs_cluster_coverage_gap": item[f"{prefix}_max_abs_cluster_coverage_gap"],
            })
    return pd.DataFrame(rows)


def load_source(source: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    directory, _ = source_paths(source)
    result = pd.read_csv(directory / "router_window_results.csv")
    folds = pd.read_csv(directory / "chronological_fold_metrics.csv")
    result = result[result.user_weight == .5].copy()
    if source == "lgbm":
        result = result[(result.coverage == .8) & (result.horizon_hours == 1.)].copy()
        folds = folds[(folds.coverage == .8) & (folds.horizon_hours == 1.)].copy()
    expected_events = 140 if source == "persistence" else 35
    if len(result) != expected_events * 2 or len(folds) != expected_events * 3 * 3:
        raise RuntimeError(f"unexpected frozen source shape for {source}: result={len(result)} folds={len(folds)}")
    result["source"] = source; folds["source"] = source
    return result, folds, static_metrics(source)


def fold_choice(folds: pd.DataFrame, eta: float, threshold: float, fallback: str, layout: str) -> str:
    values = folds.copy(); values["loss"] = loss(values, eta)
    losses = values.pivot(index="fold", columns="policy", values="loss").reindex(columns=POLICIES)
    if layout == "recent_7d_direct":
        return str(losses.loc[losses.index.max()].idxmin())
    means = losses.mean(axis=0)
    eligible = []
    for policy in POLICIES[1:]:
        gain = losses[GLOBAL] - losses[policy]
        se = gain.std(ddof=1) / np.sqrt(len(gain))
        if float(gain.mean() - threshold * se) > 0:
            eligible.append((float(gain.mean() - threshold * se), policy))
    if eligible:
        return max(eligible, key=lambda item: (item[0], item[1] == "rolling_group_norm"))[1]
    if fallback == "global":
        return GLOBAL
    if fallback == "history_best":
        return str(means.idxmin())
    raise ValueError(fallback)


def source_ablation(source: str) -> pd.DataFrame:
    result, folds, metrics = load_source(source)
    metric_index = metrics.set_index(["dataset", "configuration", "window", "policy"])
    # Existing router results provide frozen static test losses for eta=0/.01.
    event_rows = []
    event_keys = ["dataset", "configuration", "window", "window_order"]
    unique = result[event_keys].drop_duplicates().sort_values(event_keys)
    variants = []
    for eta in EFFICIENCY:
        for threshold in THRESHOLDS:
            for fallback in ("global", "history_best"):
                variants.append((f"expanding_k{threshold:g}_{fallback}_eta{eta:g}", "expanding", threshold, fallback, eta))
        variants.append((f"recent7_direct_eta{eta:g}", "recent_7d_direct", np.nan, "not_applicable", eta))
    for _, event in unique.iterrows():
        event_fold = folds[(folds.dataset == event.dataset) & (folds.configuration == event.configuration) & (folds.window == event.window)]
        for variant, layout, threshold, fallback, eta in variants:
            selected = fold_choice(event_fold, eta, threshold if np.isfinite(threshold) else 0., fallback, layout)
            frozen = result[(result.dataset == event.dataset) & (result.configuration == event.configuration)
                            & (result.window == event.window) & (result.efficiency_weight == eta)]
            if len(frozen) != 1:
                raise RuntimeError(f"missing frozen loss for {event.to_dict()} eta={eta}")
            frozen = frozen.iloc[0]
            test_losses = {policy: float(frozen[f"loss__{policy}"]) for policy in POLICIES}
            selected_metrics = metric_index.loc[(event.dataset, event.configuration, event.window, selected)].to_dict()
            global_metrics = metric_index.loc[(event.dataset, event.configuration, event.window, GLOBAL)].to_dict()
            event_rows.append({
                **event.to_dict(), "source": source, "variant": variant, "layout": layout, "threshold_se": threshold,
                "fallback": fallback, "efficiency_weight": eta, "selected_policy": selected,
                "router_loss": test_losses[selected], "oracle_loss": min(test_losses.values()),
                **{f"loss__{policy}": test_losses[policy] for policy in POLICIES},
                **{f"router_{key}": selected_metrics[key] for key in ["picp", "mpiw", "winkler_interval_score", "macro_user_abs_coverage_gap", "max_abs_cluster_coverage_gap"]},
                "strict_conflict_vs_global": int(
                    selected_metrics["macro_user_abs_coverage_gap"] < global_metrics["macro_user_abs_coverage_gap"]
                    and selected_metrics["max_abs_cluster_coverage_gap"] > global_metrics["max_abs_cluster_coverage_gap"]
                ),
            })
    return pd.DataFrame(event_rows)


def best_fixed(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    keys = ["source", "variant", "dataset", "configuration", "efficiency_weight"]
    choices = []
    for values, item in output.groupby(keys, sort=False):
        averages = {policy: float(item[f"loss__{policy}"].mean()) for policy in POLICIES}
        choices.append({**dict(zip(keys, values)), "best_fixed_policy": min(averages, key=averages.get),
                        "best_fixed_mean_loss": min(averages.values())})
    choices = pd.DataFrame(choices)
    output = output.merge(choices, on=keys, validate="many_to_one")
    output["best_fixed_window_loss"] = [row[f"loss__{row.best_fixed_policy}"] for _, row in output.iterrows()]
    output["delta_best_fixed"] = output.router_loss - output.best_fixed_window_loss
    output["delta_global"] = output.router_loss - output[f"loss__{GLOBAL}"]
    return output


def block_ci(frame: pd.DataFrame, column: str, block: int) -> tuple[float, list[float]]:
    matrices = []
    for _, item in frame.groupby("dataset", sort=True):
        pivot = item.pivot(index="window_order", columns="configuration", values=column).sort_index()
        if pivot.isna().any().any():
            raise RuntimeError("ablation bootstrap configuration alignment failure")
        matrices.append(pivot.to_numpy(float))
    rng = np.random.default_rng(SEED + block * 100 + sum(map(ord, column)))
    estimates = np.zeros(REPS)
    for matrix in matrices:
        indices = block_indices(len(matrix), block, REPS, rng)
        estimates += matrix[indices].mean(axis=(1, 2))
    samples = estimates / len(matrices)
    point = np.mean([matrix.mean() for matrix in matrices])
    return float(point), np.quantile(samples, [.025, .975]).tolist()


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_keys = ["source", "variant", "layout", "threshold_se", "fallback", "efficiency_weight"]
    for keys, item in frame.groupby(group_keys, dropna=False, sort=True):
        selections = item.selected_policy.value_counts().to_dict()
        by_configuration = item.groupby(["dataset", "configuration"], sort=False)
        macro = lambda column: float(by_configuration[column].mean().mean())
        for block in (2, 3):
            estimate, ci = block_ci(item, "delta_best_fixed", block)
            record = {**dict(zip(group_keys, keys)), "windows": len(item),
                      "mean_router_loss": macro("router_loss"),
                      "mean_best_fixed_loss": macro("best_fixed_window_loss"),
                      "mean_global_loss": macro(f"loss__{GLOBAL}"),
                      "mean_oracle_loss": macro("oracle_loss"),
                      "mean_delta_best_fixed": estimate, "mean_delta_global": macro("delta_global"),
                      "mean_router_user_gap": macro("router_macro_user_abs_coverage_gap"),
                      "mean_router_segment_gap": macro("router_max_abs_cluster_coverage_gap"),
                      "mean_router_interval_score": macro("router_winkler_interval_score"),
                      "strict_conflict_rate_vs_global": macro("strict_conflict_vs_global"),
                      "selected_global": int(selections.get(GLOBAL, 0)), "selected_segment": int(selections.get("rolling_group_norm", 0)),
                      "selected_user": int(selections.get("rolling_user_norm", 0)),
                      "block_months": block, "delta_best_ci_low": ci[0], "delta_best_ci_high": ci[1]}
            rows.append(record)
    return pd.DataFrame(rows)


def main() -> None:
    events = pd.concat([source_ablation("persistence"), source_ablation("lgbm")], ignore_index=True)
    events = best_fixed(events)
    summary = summarize(events)
    if len(events) != (140 + 35) * 18:
        raise RuntimeError(f"unexpected ablation event count: {len(events)}")
    out = HERE / "csgr_minimal_ablation"
    out.mkdir(exist_ok=True)
    events.to_csv(out / "ablation_window_results.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out / "ablation_summary.csv", index=False, encoding="utf-8-sig")
    report = {"protocol": "FROZEN_INFORMATION_REGIME_COMPARISON_PROTOCOL.md", "source_rows": len(events),
              "variants": 18, "primary_rule": "expanding_k1_global_eta0", "block_lengths_months": [2, 3],
              "interpretation": "Ablations test robustness of the frozen rule; they do not select a replacement rule."}
    (out / "ablation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
