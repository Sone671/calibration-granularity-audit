# TCI weighting correction report

The previous implementation mixed an equal user-window average in the first term with effective-sample weighting in the pooled term. That hybrid statistic was descriptive but did not inherit strict nonnegativity from the triangle inequality.

The final implementation uses, for every user, `w_ue = n_ue / sum_e n_ue` in both terms and then macro-averages users. It also reports a same-weight equal-window sensitivity with `w_ue = 1 / E_u`. The implementation asserts nonnegativity for both consistent-weight versions.

## User-CQR macro results on the final matched-quantile grid

| Forecaster | Legacy hybrid | Effective-sample same-weight primary | Equal-window same-weight sensitivity |
|---|---:|---:|---:|
| LightGBM quantile | 75.1358% | 75.1332% | 75.1605% |
| Persistence interval | 79.6996% | 79.6994% | 79.6648% |

The correction changes neither rounded headline value nor interpretation. The complete 96-row method/configuration table is `STRICT_TCI_FULL_GRID.csv`; every primary and equal-window absolute TCI value is nonnegative.
