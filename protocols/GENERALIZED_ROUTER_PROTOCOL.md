# Cross-Fitted Safe Granularity Router (CSGR): frozen validation protocol

## Objective

Test whether calibration granularity can be selected without deployment-window
labels and without assuming a particular forecasting backbone.  CSGR treats
global calibration as the stable anchor and adopts segment or user calibration
only when chronological pseudo-future validation shows a stable risk gain.

## Information available at a deployment window

- frozen lower and upper base forecasts;
- labels from the preceding 56 days only;
- frozen user identifiers, operational-segment identifiers, and training-only
  user scales;
- a declared user-versus-segment preference `lambda`.

No label or metric from the deployment month may enter routing.

## Chronological cross-fitting

The preceding 56-day block is split into three expanding folds:

1. fit days 1--35, validate days 36--42;
2. fit days 1--42, validate days 43--49;
3. fit days 1--49, validate days 50--56.

Each fold independently fits Global-, Segment-, and User-CQR corrections and
evaluates them on the following seven days.

## Declared loss and safety rule

For policy `m`, fold loss is

`lambda * macro-user coverage gap + (1-lambda) * worst-segment coverage gap`

with a sensitivity analysis adding `0.01 * normalized interval score`.

For each local policy, calculate its three fold-wise loss gains over Global-CQR.
Select the policy only when mean gain minus one standard error is positive.
When neither local policy passes, retain Global-CQR.  Refit the selected policy
on all 56 days before evaluating the next natural month.

## Frozen validation grid

- datasets: London, Ausgrid, UCI Electricity;
- base forecaster: persistence quantile interval;
- target coverage: 80% and 90%;
- horizon: 1 h and 6 h;
- user weight: 0, 0.25, 0.5, 0.75, 1;
- efficiency weight: 0 and 0.01;
- all 11/12/12 frozen natural-month windows.

Primary comparison: CSGR versus Global-CQR and the hindsight best fixed
granularity within each dataset-coverage-horizon configuration.  Oracle loss is
reported only as an unattainable lower bound.  Results must be retained even if
CSGR does not beat the fixed policies.

