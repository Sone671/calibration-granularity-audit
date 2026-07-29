"""Build material-conflict and controlled-simulation evidence for the KBS revision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from diagnostic_metrics import material_conflict_profile


TARGET = 0.80
SEED = 20260729
ALIGNMENTS = (0.0, 0.5, 1.0)
FLIP_PROBABILITIES = (0.0, 0.25, 0.5)
MATERIAL_THRESHOLDS = (0.0, 0.0025, 0.005, 0.01)


def balanced_panel_to_long(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    methods = {
        "global": "rolling_global_norm",
        "segment": "rolling_group_norm",
        "user": "rolling_user_norm",
    }
    for record in panel.itertuples(index=False):
        key = (
            f"{record.dataset}|{record.forecaster}|{record.coverage}|"
            f"{record.horizon_hours}|{record.window}"
        )
        for prefix, method in methods.items():
            rows.append(
                {
                    "window": key,
                    "method": method,
                    "macro_user_abs_coverage_gap": getattr(
                        record, f"{prefix}_macro_user_abs_coverage_gap"
                    ),
                    "max_abs_cluster_coverage_gap": getattr(
                        record, f"{prefix}_max_abs_cluster_coverage_gap"
                    ),
                }
            )
    return pd.DataFrame(rows)


def simulate_panel(
    alignment: float,
    flip_probability: float,
    rng: np.random.Generator,
    users: int = 180,
    environments: int = 24,
    observations: int = 336,
) -> tuple[pd.DataFrame, float, float]:
    groups = np.repeat(np.arange(3), users // 3)
    offsets = np.array([0.012, -0.008, 0.004])
    user_bias = rng.normal(0.0, 0.055, users)
    for group in range(3):
        mask = groups == group
        user_bias[mask] -= user_bias[mask].mean()

    signs = np.ones(environments)
    for environment in range(1, environments):
        signs[environment] = signs[environment - 1]
        if rng.random() < flip_probability:
            signs[environment] *= -1
    realized_flip_rate = float(np.mean(signs[1:] != signs[:-1]))

    global_error = signs[:, None] * (offsets[groups][None, :] + user_bias[None, :])
    user_error = signs[:, None] * (
        0.40 * (offsets[groups][None, :] + user_bias[None, :])
        + alignment * 0.020 * np.sign(offsets[groups])[None, :]
    )
    probabilities = {
        "rolling_global_norm": np.clip(TARGET + global_error, 0.02, 0.98),
        "rolling_user_norm": np.clip(TARGET + user_error, 0.02, 0.98),
    }

    window_rows = []
    user_tci = np.nan
    for method, probability in probabilities.items():
        coverage = rng.binomial(observations, probability) / observations
        for environment in range(environments):
            group_gaps = [
                abs(float(coverage[environment, groups == group].mean()) - TARGET)
                for group in range(3)
            ]
            window_rows.append(
                {
                    "window": f"e{environment:02d}",
                    "method": method,
                    "macro_user_abs_coverage_gap": float(
                        np.mean(np.abs(coverage[environment] - TARGET))
                    ),
                    "max_abs_cluster_coverage_gap": float(max(group_gaps)),
                }
            )
        if method == "rolling_user_norm":
            signed = coverage - TARGET
            mean_window_gap = float(np.mean(np.mean(np.abs(signed), axis=0)))
            pooled_gap = float(np.mean(np.abs(np.mean(signed, axis=0))))
            user_tci = (mean_window_gap - pooled_gap) / max(mean_window_gap, 1e-12)
    return pd.DataFrame(window_rows), float(user_tci), realized_flip_rate


def run_simulation(replicates: int) -> pd.DataFrame:
    seed_sequence = np.random.SeedSequence(SEED)
    child_seeds = seed_sequence.spawn(len(ALIGNMENTS) * len(FLIP_PROBABILITIES) * replicates)
    records = []
    index = 0
    for alignment in ALIGNMENTS:
        for flip_probability in FLIP_PROBABILITIES:
            for replicate in range(replicates):
                rng = np.random.default_rng(child_seeds[index])
                index += 1
                windows, user_tci, realized_flip_rate = simulate_panel(
                    alignment, flip_probability, rng
                )
                magnitude, thresholds, _ = material_conflict_profile(
                    windows, deltas=MATERIAL_THRESHOLDS
                )
                records.append(
                    {
                        "alignment": alignment,
                        "flip_probability": flip_probability,
                        "replicate": replicate,
                        "realized_flip_rate": realized_flip_rate,
                        "strict_gcr": thresholds[0]["conflict_rate"],
                        "material_gcr_0_5pp": thresholds[2]["conflict_rate"],
                        "mean_pareto_conflict_margin": magnitude["mean_margin"],
                        "user_tci": user_tci,
                    }
                )
    return pd.DataFrame(records)


def summarize_simulation(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (alignment, flip_probability), frame in panel.groupby(
        ["alignment", "flip_probability"], sort=True
    ):
        record = {
            "alignment": alignment,
            "flip_probability": flip_probability,
            "replicates": len(frame),
        }
        for metric in (
            "strict_gcr",
            "material_gcr_0_5pp",
            "mean_pareto_conflict_margin",
            "user_tci",
        ):
            record[f"{metric}_mean"] = float(frame[metric].mean())
            record[f"{metric}_q025"] = float(frame[metric].quantile(0.025))
            record[f"{metric}_q975"] = float(frame[metric].quantile(0.975))
        rows.append(record)
    return pd.DataFrame(rows)


def plot_simulation(summary: pd.DataFrame, destination: Path) -> None:
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.65), constrained_layout=True)
    colors = ["#3b6fb6", "#de8f05", "#2a9d6f"]
    for color, flip_probability in zip(colors, FLIP_PROBABILITIES):
        frame = summary[summary.flip_probability == flip_probability].sort_values("alignment")
        axes[0].plot(
            frame.alignment,
            100 * frame.strict_gcr_mean,
            marker="o",
            color=color,
            label=f"flip probability={flip_probability:g}",
        )
        axes[0].fill_between(
            frame.alignment,
            100 * frame.strict_gcr_q025,
            100 * frame.strict_gcr_q975,
            color=color,
            alpha=0.14,
        )
    axes[0].set(xlabel="group-aligned correction strength", ylabel="strict GCR (%)")
    axes[0].set_ylim(bottom=0)
    axes[0].legend(frameon=False, fontsize=7)

    for color, alignment in zip(colors, ALIGNMENTS):
        frame = summary[summary.alignment == alignment].sort_values("flip_probability")
        axes[1].plot(
            frame.flip_probability,
            100 * frame.user_tci_mean,
            marker="s",
            color=color,
            label=f"alignment={alignment:g}",
        )
        axes[1].fill_between(
            frame.flip_probability,
            100 * frame.user_tci_q025,
            100 * frame.user_tci_q975,
            color=color,
            alpha=0.14,
        )
    axes[1].set(xlabel="environment sign-flip probability", ylabel="User-policy TCI (%)")
    axes[1].set_ylim(0, 100)
    axes[1].legend(frameon=False, fontsize=7)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, bbox_inches="tight")
    plt.close(fig)


def write_full_environment_table(conflict_rows: list[dict], destination: Path) -> None:
    lines = [
        r"\scriptsize",
        r"\begin{longtable}{llllrrrc}",
        r"\caption{Complete configuration--environment conflict audit. Differences and margins are in coverage percentage points. Status S denotes a strict conflict and M denotes PCM above 0.5 percentage points.}\label{tab:full-environment-audit}\\",
        r"\toprule",
        r"Data set & Forecaster & Setting & Month & $\Delta_{\rm user}$ & $\Delta_{\rm group}$ & PCM & Status \\",
        r"\midrule",
        r"\endfirsthead",
        r"\multicolumn{8}{c}{Table~\ref{tab:full-environment-audit} continued}\\",
        r"\toprule",
        r"Data set & Forecaster & Setting & Month & $\Delta_{\rm user}$ & $\Delta_{\rm group}$ & PCM & Status \\",
        r"\midrule",
        r"\endhead",
        r"\midrule\multicolumn{8}{r}{Continued on next page}\\",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    for row in conflict_rows:
        dataset, forecaster, coverage, horizon, month = row["window"].split("|")
        short_forecaster = "LightGBM" if forecaster == "lightgbm_quantile" else "Persistence"
        setting = f"{100*float(coverage):.0f}\\%/{float(horizon):g} h"
        user_difference = 100 * row["user_gap_difference_user_minus_global"]
        group_difference = 100 * row["group_gap_difference_user_minus_global"]
        margin = 100 * row["pareto_conflict_margin"]
        status = "M" if row["pareto_conflict_margin"] > 0.005 else ("S" if row["personalization_conflict"] else "--")
        lines.append(
            f"{dataset} & {short_forecaster} & {setting} & {month} & "
            f"{user_difference:.3f} & {group_difference:.3f} & {margin:.3f} & "
            f"{status} \\\\"
        )
    lines.extend([r"\end{longtable}", ""])
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--replicates", type=int, default=200)
    args = parser.parse_args()
    root = args.root.resolve()

    empirical = pd.read_csv(root / "BALANCED_FULL_GRID_PANEL.csv")
    empirical_long = balanced_panel_to_long(empirical)
    magnitude, thresholds, conflict_rows = material_conflict_profile(
        empirical_long, deltas=MATERIAL_THRESHOLDS
    )
    pd.DataFrame(thresholds).to_csv(root / "MATERIAL_CONFLICT_PROFILE.csv", index=False)
    pd.DataFrame(conflict_rows).to_csv(root / "CONFLICT_MAGNITUDE_PANEL.csv", index=False)
    (root / "CONFLICT_MAGNITUDE_SUMMARY.json").write_text(
        json.dumps(magnitude, indent=2), encoding="utf-8"
    )

    simulation = run_simulation(args.replicates)
    summary = summarize_simulation(simulation)
    simulation.to_csv(root / "SYNTHETIC_MECHANISM_PANEL.csv", index=False)
    summary.to_csv(root / "SYNTHETIC_MECHANISM_SUMMARY.csv", index=False)
    plot_simulation(
        summary,
        root / "paper_cn_internal" / "figures" / "fig4_synthetic_mechanism.pdf",
    )
    write_full_environment_table(
        conflict_rows,
        root / "paper_cn_internal" / "sections_en" / "generated_full_environment_table.tex",
    )
    print(
        json.dumps(
            {
                "empirical_conflicts": magnitude["strict_conflicts"],
                "median_margin_percentage_points": 100 * magnitude["median_margin"],
                "simulation_rows": len(simulation),
                "simulation_cells": len(summary),
                "replicates_per_cell": args.replicates,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
