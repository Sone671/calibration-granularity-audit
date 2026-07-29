# ERW-CQR baseline completion report

> Historical persistence-only report. Its persistence results remain valid, but its transfer scope is superseded by `ERW_SENSITIVITY_COMPLETION_REPORT.md`.

- Protocol: `FROZEN_WEIGHTED_CONFORMAL_BASELINE_ADDENDUM.md`
- Runtime: Python 3.12, NumPy 2.3.5, pandas 2.3.3
- Frozen audit: all six metrics for Global/Segment/User-CQR reproduced with maximum absolute difference 0 on all three datasets.
- Full grid: 140 configuration × window observations; primary setting: 35 unique 80%/1 h windows.

## Main result

Across the full persistence grid, ERW-User reduced mean L0.5 from 0.042042 to 0.036076. The paired difference was -0.005966, with a synchronized circular block-2 interval of [-0.008189, -0.003860]. Its mean Winkler score changed by -1.33% relative to static User-CQR.

In the primary 80%/1 h setting, ERW-User reduced mean L0.5 from 0.036002 to 0.031536.

At this initial stage, ERW-CQR was evaluated only with the persistence backbone. It is not evidence of a distribution-free guarantee under temporal drift. The subsequent LightGBM transfer is reported in `ERW_SENSITIVITY_COMPLETION_REPORT.md`.
