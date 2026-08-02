# Frozen result index

All files below are aggregate or audit outputs approved for public repository release. No table contains raw load measurements or customer identifiers.

## Core static audit

| Claim/output | File |
|---|---|
| Complete balanced 280-row grid | `results/BALANCED_FULL_GRID_PANEL.csv` |
| Grid construction and configuration summaries | `results/BALANCED_FULL_GRID_REPORT.json` |
| Strict/material GCR survival profile | `results/MATERIAL_CONFLICT_PROFILE.csv` |
| Per-environment PCM, trade-off length, and switch threshold | `results/CONFLICT_MAGNITUDE_PANEL.csv` |
| Conflict magnitude summary | `results/CONFLICT_MAGNITUDE_SUMMARY.json` |
| Full common-weight TCI grid | `results/STRICT_TCI_FULL_GRID.csv` |

## Controlled mechanism experiment

| Output | File |
|---|---|
| All 1,800 replicate rows | `results/SYNTHETIC_MECHANISM_PANEL.csv` |
| Nine-cell summary | `results/SYNTHETIC_MECHANISM_SUMMARY.csv` |

## No-immediate-feedback analyses

- Equal-weight/ERW panels and bootstrap: `WEIGHTED_CONFORMAL_*`.
- Half-life sensitivity and LightGBM transfer: `ERW_SENSITIVITY_*`.
- Static CSGR routing and inference: `INFORMATION_REGIME_*`.
- CSGR minimal ablation: `ablation_summary.csv` and `ablation_report.json`.
- Complete LightGBM routing grid, combined two-backbone comparisons, direct selectors, and synchronized inference: `FULL_ROUTER_*` and `LIGHTGBM_ROUTER_FULL_GRID_*`.

## Immediate-feedback analyses

- Primary ACI/PID panel and paired inference: `P0_ONLINE_*`.
- Frozen ACI forward audit retained for traceability: `ACI_FORWARD_*`.

## Audit-partition sensitivity

- Training-only demand tertiles: `DEMAND_TERTILE_FULL_GRID_*`.
- Frozen-predictor KM2/KM3/KM4/Ward3 audit: `FROZEN_AUDIT_GROUP_SENSITIVITY_*`.

## Provenance and invalidation trail

- `LIGHTGBM_ROUTER_FULL_GRID_COMPLETION_REPORT.md` summarizes the 280-decision routing grid and its backward-compatibility checks.
- `LIGHTGBM_FULL_GRID_COMPLETION_REPORT.md` documents the validated target-matched static grid.
- `LIGHTGBM_FULL_GRID_INVALIDATION.md` and `LIGHTGBM_FULL_GRID_V2_ABORTED.md` are retained so invalid preliminary grids cannot be mistaken for final evidence.
- `TCI_WEIGHTING_CORRECTION_REPORT.md` records the common-weight correction.
- `FINAL_REMAINING_REVISIONS_REPORT.md` summarizes final consistency checks.
