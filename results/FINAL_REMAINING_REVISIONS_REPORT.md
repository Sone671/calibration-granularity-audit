# Final remaining-revisions completion report

## Unified final grid and figures

- Final balanced panel: 280 configuration-month environments, 140 per forecaster.
- LightGBM 80% uses 0.1/0.9 base quantiles; 90% uses target-matched 0.05/0.95; 1 h and 6 h are trained independently.
- Final strict GCR: 67/280 (23.93%); LightGBM 37/140, persistence 30/140.
- Figures 2--4 are generated only from `BALANCED_FULL_GRID_PANEL.csv` and `STRICT_TCI_FULL_GRID.csv`; the stale `n=41` and 79.4% labels are absent.

## TCI correction

- Both terms now use within-user effective-sample weights and users are macro-averaged, so TCI is nonnegative by construction.
- User-CQR macro TCI: LightGBM 75.13%, persistence 79.70%.
- Equal-month same-weight sensitivity: 75.16% and 79.66%.

## Segmentation isolation

- End-to-end and frozen-predictor/audit-group-only sensitivities are both released.
- Frozen LightGBM GCR ranges from 20.00% to 42.86% across KM2/KM3/KM4/Ward3.
- Persistence end-to-end and frozen-predictor columns coincide because its predictor and the Global/User policies used by GCR do not use segment labels.

## ERW sensitivity and transfer

- Persistence ERW-User L0.5 at 7/14/28 days: 0.034907/0.036076/0.038370 versus static 0.042042; every block-2 interval for the paired difference is below zero.
- LightGBM 80%/1 h ERW-User: 0.021941 to 0.020430; paired block-2 interval [-0.002283, -0.000814].
- Static LightGBM metrics reproduce the frozen output with maximum absolute difference zero.
- ERW-Global worsens on LightGBM and ERW-Segment is essentially unchanged, so the manuscript reports a granularity-bounded transfer rather than universal superiority.

## Editorial and reproducibility QA

- Removed internal figure-caption instructions and the obsolete appendix TODO.
- Replaced “external validation” with “cross-forecaster transfer validation”.
- Section 8.3 contains an introductory paragraph and a fixed-position deployment table.
- Appendix B precedes its complete two-forecaster TCI table; all four LightGBM and persistence configurations are shown.
- Final PDF: 29 pages, 44 cited/rendered bibliography entries, no LaTeX warnings, unresolved references, or over/underfull boxes.
- Initial data-free package: 66 hashed files; tests: 3 passed; no unintended absolute local path leak.
