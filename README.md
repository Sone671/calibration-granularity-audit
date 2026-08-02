# Evidence-gated calibration-granularity routing for conformal load forecasts

Public reproducibility release accompanying the manuscript **“Evidence-gated calibration-granularity routing for multi-user conformal load forecasting”** by Jiajun Song and Jun Yao.

This repository contains executable audit code, frozen aggregate results, controlled-simulation outputs, protocols, data-source checksums, and verification tools. The manuscript and supplementary files are intentionally not distributed in this repository. It also contains **no raw household load, customer identifier, credential, trained model binary, or local absolute path**.

## What is reproduced

- Complete 280 configuration–environment static-CQR audit.
- Strict and material granularity conflicts, Pareto conflict margin, rank reversal, temporal cancellation, and routing oracle gap.
- Controlled $3\times3\times200$ simulation separating group-aligned correction from temporal sign switching.
- Equal-weight and recency-weighted CQR, immediate-feedback ACI/PID, segmentation sensitivity, and 280-decision full-grid CSGR routing results.
- A feasible previous-window routing baseline and width-aware routing sensitivity.
- Source-archive SHA-256, ZIP CRC, and selected-member verification without redistributing raw observations.

## Repository layout

| Path | Contents |
|---|---|
| `code/` | Diagnostics, runners, controllers, routers, report builders, and checksum/simulation utilities |
| `tests/` | Ten raw-data-free unit tests |
| `results/` | Prespecified aggregate panels, summaries, bootstrap outputs, and provenance records |
| `protocols/` | Frozen pre-result protocols and addenda |
| `data/` | Download/licensing instructions and archive/member checksums; no raw data |
| `docs/` | Reproduction guide, result index, verification report, and upload checklist |
| `scripts/` | Repository verification and manifest builders |

## Quick verification

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
pytest -q tests
python scripts/verify_repository.py
```

Expected result: **10 tests passed**, manifest verification succeeds, and the repository reports no raw-data archive, forbidden absolute path, secret pattern, or file above 50 MB.

## Data

Raw data are downloaded from the original London and UCI providers and the documented Ausgrid archival source. Do not commit them to GitHub. Follow [`data/README.md`](data/README.md), then compare archive and member hashes with [`data/checksums.json`](data/checksums.json).

## Results and manuscript status

- [`docs/RESULTS_INDEX.md`](docs/RESULTS_INDEX.md) maps manuscript claims to machine-readable files.
- The manuscript and supplementary material are not included in the public repository at this stage.
- Machine-readable aggregate evidence and the complete 280-row environment audit remain available under `results/`.
- In the complete 280-decision forward grid, CSGR has combined loss 0.033029 versus 0.033356 for the ex-post best fixed policy. The gain is concentrated in persistence (0.041254 versus 0.042024); LightGBM is statistically tied and has a small adverse point difference (0.024803 versus 0.024688).
- Customer-level observations, per-user summaries, and identifiers are excluded. The repository contains only aggregate outputs from the cross-sectional and temporal evidence screens.
- In the manuscript, CSGR expands to **Chronological Stability-Screened Granularity Router**. Frozen protocols, code metadata, and result JSON may retain earlier development labels so that the released validation record remains provenance-preserving; these labels do not change the algorithm or numerical results.

## Citation and licenses

Use [`CITATION.cff`](CITATION.cff) for the authors, manuscript title, release version, and repository URL. Add the article or archival DOI when one is assigned.

Repository-authored source code is licensed under the [MIT License](LICENSE). Repository-authored documentation, protocols, metadata, and aggregate result tables are licensed under [CC BY 4.0](LICENSE-CONTENT.md). The scope and exclusions are defined in [`NOTICE.md`](NOTICE.md). External data are not redistributed and remain governed by their original providers.

Public release: `v1.2.0` (2026-08-02).
