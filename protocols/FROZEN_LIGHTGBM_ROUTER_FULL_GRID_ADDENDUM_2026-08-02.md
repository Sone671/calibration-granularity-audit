# LightGBM CSGR full-grid strengthening addendum

Frozen on 2026-08-02 before inspecting any full-grid CSGR result beyond the
previously reported 80% coverage, 1 h transfer cell.

## Purpose

The existing manuscript validates the unchanged CSGR rule on the complete
persistence grid but only on the 35-window LightGBM 80%/1 h transfer cell.  This
addendum closes that evidence gap without changing the router, its candidates,
its loss, or its evidence threshold.

## Frozen design

- Data sets: London, Ausgrid, and UCI Electricity.
- Target coverages: 80% and 90%.
- Horizons: 1 h and 6 h.
- Natural-month environments: all frozen 11/12/12 windows, for 140
  configuration-month decisions.
- Forecaster: the existing target-matched LightGBM quantile specification with
  600,000 sampled training rows, 250 boosting rounds, seed 20260725, and the
  frozen feature set.
- Calibration candidates: equal-weight Global-, Segment-, and User-CQR.
- Router: the unchanged three expanding 35/7, 42/7, and 49/7 chronological
  folds, one-standard-error screen, deterministic Segment tie-break, and
  Global fallback.
- Primary loss: equal user/maximum-segment preference, lambda = 0.5, with no
  interval-score penalty.  The existing lambda and efficiency sensitivities
  are retained but cannot replace the primary setting.

## Predeclared comparisons

1. CSGR versus the ex-post best fixed granularity within each
   data-set/coverage/horizon configuration.
2. CSGR versus Global-CQR and the unattainable month-wise oracle.
3. CSGR versus a feasible previous-window winner, initialized at Global-CQR.
4. CSGR versus a feasible follow-the-leader selector that deploys the policy
   with the lowest cumulative loss over completed earlier months and starts at
   Global-CQR.
5. CSGR versus two within-window selectors using the same chronological folds:
   the lowest mean validation loss without a stability screen and the latest
   seven-day validation winner.

All variants must be retained regardless of direction.  No result from this
grid may be used to change the frozen CSGR rule.

## Inference and integrity checks

- Primary uncertainty: synchronized circular moving blocks of two months;
  three-month blocks and data-set hierarchical resampling are sensitivities.
- Resampling keeps all coverage/horizon views of a data-set month together.
- The 80%/1 h cell must reproduce the existing 35-window LightGBM router
  artifact to numerical tolerance before the other cells are interpreted.
- Run metadata record the code hash, model settings, row counts, and wall time.

