# Frozen public release checklist

- [x] Freeze the balanced 280-row environment audit and aggregate result tables.
- [x] Preserve invalidation and aggregation-correction records in `results/`.
- [x] Verify data URLs, archive/member checksums, ZIP CRC outcomes, and dataset licence notes.
- [x] Exclude raw household observations, customer identifiers, trained model binaries, secrets, caches, and local paths.
- [x] Exclude manuscript PDFs, LaTeX sources, figures, and supplementary files from the public repository.
- [x] Run seven raw-data-free unit tests and repository manifest verification.
- [x] Record authors and the public repository URL in `CITATION.cff`.
- [x] Apply MIT to repository-authored code and CC BY 4.0 to repository-authored documentation and aggregate results.
- [ ] Add the article or archival DOI when assigned.

Release: `v1.1.0` (2026-07-29).
