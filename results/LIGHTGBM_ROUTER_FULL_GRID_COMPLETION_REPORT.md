# Complete-grid CSGR evaluation report

## Integrity

- Environments: 140 (3 data sets x 2 coverages x 2 horizons x 11/12/12 months).
- Existing 80%/1 h rows reproduced: 350.
- Selection mismatches: 0.
- Maximum absolute numeric difference: 0.
- Accepted static LightGBM grid rows reproduced: 140.
- Maximum static-grid loss difference: 5.55e-17.

## LightGBM: primary equal-preference result

- CSGR loss: 0.024803.
- Ex-post best fixed loss: 0.024688.
- Previous-window winner loss: 0.025762.
- Follow-the-leader loss: 0.025502.
- Mean-fold minimum loss: 0.024709.
- Latest-fold minimum loss: 0.024847.
- Global-CQR loss: 0.035189.
- Month-wise oracle loss: 0.024319.
- Best-fixed--oracle headroom recovered by CSGR: -31.0%.
- CSGR minus best fixed, synchronized block-2 95% interval: [-0.000031, 0.000331].
- Data-set hierarchical block-2 interval: [-0.000035, 0.000395].

## Persistence: primary equal-preference result

- CSGR loss: 0.041254.
- Ex-post best fixed loss: 0.042024.
- Previous-window winner loss: 0.044604.
- Follow-the-leader loss: 0.044352.
- Mean-fold minimum loss: 0.041680.
- Latest-fold minimum loss: 0.041843.
- Global-CQR loss: 0.057771.
- Month-wise oracle loss: 0.040229.
- Best-fixed--oracle headroom recovered by CSGR: 42.9%.
- CSGR minus best fixed, synchronized block-2 95% interval: [-0.001493, -0.000180].
- Data-set hierarchical block-2 interval: [-0.001878, -0.000032].

## Combined: primary equal-preference result

- CSGR loss: 0.033029.
- Ex-post best fixed loss: 0.033356.
- Previous-window winner loss: 0.035183.
- Follow-the-leader loss: 0.034927.
- Mean-fold minimum loss: 0.033195.
- Latest-fold minimum loss: 0.033345.
- Global-CQR loss: 0.046480.
- Month-wise oracle loss: 0.032274.
- Best-fixed--oracle headroom recovered by CSGR: 30.3%.
- CSGR minus best fixed, synchronized block-2 95% interval: [-0.000696, -0.000008].
- Data-set hierarchical block-2 interval: [-0.000897, 0.000043].

All comparisons were retained regardless of direction; this run does not retune CSGR.

## Width-aware sensitivity ($eta=0.01$)

- LightGBM: CSGR 0.048303, best fixed 0.048286, delta 0.000017, interval [-0.000050, 0.000086].
- Persistence: CSGR 0.085451, best fixed 0.086088, delta -0.000636, interval [-0.001312, -0.000085].
- Combined: CSGR 0.066877, best fixed 0.067187, delta -0.000310, interval [-0.000649, -0.000026].
