# Frozen P0 dynamic-baseline and interpretable-group addendum

Frozen date: 2026-07-29, before inspecting any Conformal PID or demand-tertile results.

## Scope

- The primary setting only: 80% target coverage, 1 h horizon, London, Ausgrid, and UCI Electricity, with both persistence and LightGBM forecasters.
- The existing ACI implementation is retained as canonical ACI with a frozen split-conformal score set. Its panel extension is atomic: all intervals at a timestamp are issued before any label from that timestamp updates a controller.
- Conformal PID uses the quantile-tracker plus logarithmically saturated integrator from Angelopoulos, Candes, and Tibshirani (2023). The optional scorecaster (D term) is disabled to avoid introducing another fitted forecasting model. We label the method PID-PI throughout.
- PID-PI uses the same normalized CQR score and the same 56-day initial calibration block as ACI. The initial threshold is the finite-sample 80% conformal quantile. The official electricity-experiment constants are fixed at `Csat=5` and `KI=10`.
- The proportional learning-rate multiplier is fixed at 0.05, from the original electricity candidate grid. Its scale is the 95th--5th percentile range of the relevant 56-day calibration scores, which is fixed before the target month. No data-set, window, forecaster, or granularity-specific selection is allowed.
- Global, segment, and user controllers are evaluated. Global and segment controllers update from the mean batch miscoverage at a timestamp; user controllers update independently after each user's outcome is observed. All controller states reset at the end of a target month.

## Interpretable audit groups

- Demand tertiles are defined only from each user's mean load in the original training period.
- Users are sorted by `(training_mean_load, stable_customer_identifier)` and divided by stable rank into three equal-frequency groups: low, medium, and high demand. Test-period loads never enter the mapping.
- Frozen Global-CQR and User-CQR per-user monthly outputs are remapped to these groups without retraining either forecaster or recalculating any conformal score.
- For every method and month, report both (i) the maximum absolute group coverage gap and (ii) the observation-count-weighted mean absolute group coverage gap.
- The original GCR remains defined using the maximum group gap. A weighted-mean-group conflict rate is reported as a robustness diagnostic and is not substituted for the primary estimand.

## Acceptance checks

- Each dynamic method/forecaster combination must contain exactly 35 unique data-set months; static rows must exactly reproduce the frozen 80%/1 h reference.
- Every timestamp update must be atomic, and no target-month score may enter the frozen initial score set.
- Each demand-tertile mapping must contain every retained user exactly once, have group sizes differing by at most one, and be monotone in training mean load.
- The existing KM3 frozen audit must still reproduce the balanced-grid reference exactly after the additional metric is introduced.
