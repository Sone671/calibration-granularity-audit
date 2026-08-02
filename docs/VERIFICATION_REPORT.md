# Release verification report

Public release: `v1.2.0`
Verification date: `2026-08-02`

## Completed checks

- Pinned Python environment specified in `environment.yml` and `requirements.txt`.
- Ten unit tests passed.
- All 127 release files verified against `MANIFEST.json` and `MANIFEST.sha256`.
- No file exceeds 50 MB; the complete repository is below 5 MB.
- No ZIP/archive, raw household load, trained model binary, local absolute path, user profile path, common secret token, or private key is included.
- `data/checksums.json` contains no raw observations or local paths and records three passing ZIP CRC checks.
- Data-source pages, UCI DOI/license, and London/Ausgrid access records were rechecked for the manuscript release.
- Manuscript and supplementary files are excluded from the public repository.
- MIT and CC BY 4.0 scopes are recorded in `LICENSE`, `LICENSE-CONTENT.md`, and `NOTICE.md`.

## Frozen numerical invariants

- Static grid: 280 configuration–environment rows and 35 unique calendar-month environments.
- Strict GCR: 67/280 (23.93%).
- Material GCR above 0.5 percentage points: 19/280 (6.79%).
- User-CQR TCI macro-average: 75.13% for LightGBM and 79.70% for persistence.
- Controlled simulation: 1,800 replicate rows, nine cells, 200 replicates per cell.
- Forward routing grid: 140 persistence decisions, 140 LightGBM decisions, and 280 combined decisions.
- Combined CSGR loss: 0.033029 versus 0.033356 for the ex-post best fixed policy.
- CSGR minus best fixed: synchronized block-2 interval $[-0.000696,-0.000008]$ and data-set-hierarchical interval $[-0.000897,0.000043]$.
- Backward compatibility: 350 LightGBM 80%/1 h decision rows match with zero selection differences; all 140 static LightGBM environment losses match within $5.55\times10^{-17}$.

Run `python scripts/verify_repository.py` after any edit. Rebuild the manifest with `python scripts/build_manifest.py` before tagging a release.
