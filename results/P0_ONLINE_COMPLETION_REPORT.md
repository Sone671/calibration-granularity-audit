# P0 online-baseline completion report

Protocol: `FROZEN_P0_DYNAMIC_BASELINE_ADDENDUM.md`.

## KM3 audit-group conflict counts

| Forecaster | Windows | Static | ACI | PID-PI |
|---|---:|---:|---:|---:|
| lgbm | 35 | 11 | 2 | 0 |
| persistence | 35 | 9 | 2 | 0 |

## Demand-tertile conflict counts

| Forecaster | Static max | ACI max | PID max | Static weighted | ACI weighted | PID weighted |
|---|---:|---:|---:|---:|---:|---:|
| lgbm | 16 | 2 | 0 | 13 | 4 | 0 |
| persistence | 11 | 2 | 0 | 11 | 2 | 0 |

## Integrity

- The released panel contains 700 method-window rows over 70 predictor-month observations.
- The largest static reproduction difference is 0.000e+00.
- Both forecasters are retained within synchronized calendar-month bootstrap blocks.
