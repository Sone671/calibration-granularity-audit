# Reproducibility guide

## 1. Raw-data-free verification

Create the pinned Python 3.12 environment and run:

```bash
python -m pip install -r requirements.txt
pytest -q tests
python scripts/verify_repository.py
```

The ten tests cover diagnostic metrics, Pareto conflict materiality, temporal cancellation, routing stability screening, direct selector summaries, full-grid integrity checks, efficiency penalties, and atomic PID panel updates.

## 2. Rebuild revision evidence from frozen results

Use a disposable copy because this command writes result tables and a figure below the supplied root:

```bash
python code/build_revision_evidence.py --root results --replicates 200
```

Expected high-level invariants:

- 67 strict conflicts among 280 environments.
- 39, 19, and 7 conflicts above PCM thresholds 0.25, 0.5, and 1.0 percentage points.
- 1,800 controlled-simulation rows and nine design cells.
- Median empirical PCM about 0.315 percentage points.

## 3. Full raw-data reruns

Follow `data/README.md` and verify source hashes first. The principal runners accept explicit London, Ausgrid, and UCI paths. Full LightGBM experiments use 600,000 training rows, 250 boosting rounds, four CPU threads, and may require substantial memory and compute.

Representative command:

```bash
python code/run_lightgbm_full_grid.py --prepared data/prepared/london --raw-dir data/raw/ausgrid --zip data/raw/uci/ElectricityLoadDiagrams20112014.zip --out outputs/lightgbm_full_grid
```

The complete forward LightGBM routing grid uses the same explicit data paths:

```bash
python code/run_lightgbm_router_full_grid.py --datasets london ausgrid uci --prepared data/prepared/london --raw-dir data/raw/ausgrid --zip data/raw/uci/ElectricityLoadDiagrams20112014.zip --out outputs/lightgbm_router_full_grid_2026-08-02
```

After generating the complete LightGBM grid, the prior 80%/1 h reference, and the persistence routing grid, rebuild the combined summaries with:

```bash
python code/build_lightgbm_router_full_grid_report.py --run-dir outputs/lightgbm_router_full_grid_2026-08-02 --reference-dir outputs/lightgbm_router_validation --persistence-dir outputs/generalized_router_validation --static-panel results/BALANCED_FULL_GRID_PANEL.csv
```

The report builder verifies the 80%/1 h reference cell and the static LightGBM panel before producing combined comparisons.

No target-month labels may enter static, ERW, or CSGR selection for that month. ACI/PID update only after a complete timestamp batch is observed.

## 4. Compare outputs

Compare regenerated aggregate files with `results/` by schema, row count, declared tolerance, and paired differences. The completion reports document the zero-difference checks; invalidated preliminary grids remain available only as provenance records.
