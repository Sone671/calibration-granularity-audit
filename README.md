# Calibration granularity audit for rolling conformal load forecasts

Public reproducibility release accompanying the manuscript **“Calibration granularity conflicts in rolling conformal load forecasting: A multi-level, multi-environment audit”** by Jiajun Song and Jun Yao.

This repository contains executable audit code, frozen aggregate results, controlled-simulation outputs, protocols, data-source checksums, and verification tools. The manuscript and supplementary files are intentionally not distributed in this repository. It also contains **no raw household load, customer identifier, credential, trained model binary, or local absolute path**.

## What is reproduced

- Complete 280 configuration–environment static-CQR audit.
- Strict and material granularity conflicts, Pareto conflict margin, rank reversal, temporal cancellation, and routing oracle gap.
- Controlled $3\times3\times200$ simulation separating group-aligned correction from temporal sign switching.
- Equal-weight and recency-weighted CQR, immediate-feedback ACI/PID, segmentation sensitivity, and conservative CSGR routing results.
- Source-archive SHA-256, ZIP CRC, and selected-member verification without redistributing raw observations.

## Repository layout

| Path | Contents |
|---|---|
| `code/` | Diagnostics, runners, controllers, routers, report builders, and checksum/simulation utilities |
| `tests/` | Seven raw-data-free unit tests |
| `results/` | Frozen aggregate panels, summaries, bootstrap outputs, and audit trail |
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

Expected result: **7 tests passed**, manifest verification succeeds, and the repository reports no raw-data archive, forbidden absolute path, secret pattern, or file above 50 MB.

## Data

Raw data are downloaded from the original London and UCI providers and the documented Ausgrid archival source. Do not commit them to GitHub. Follow [`data/README.md`](data/README.md), then compare archive and member hashes with [`data/checksums.json`](data/checksums.json).

## Results and manuscript status

- [`docs/RESULTS_INDEX.md`](docs/RESULTS_INDEX.md) maps manuscript claims to machine-readable files.
- The manuscript and supplementary material are not included in the public repository at this stage.
- Machine-readable aggregate evidence and the complete 280-row environment audit remain available under `results/`.

## Citation and licenses

Use [`CITATION.cff`](CITATION.cff) for the authors, manuscript title, release version, and repository URL. Add the article or archival DOI when one is assigned.

Repository-authored source code is licensed under the [MIT License](LICENSE). Repository-authored documentation, protocols, metadata, and aggregate result tables are licensed under [CC BY 4.0](LICENSE-CONTENT.md). The scope and exclusions are defined in [`NOTICE.md`](NOTICE.md). External data are not redistributed and remain governed by their original providers.

Public release: `v1.1.0` (2026-07-29).
