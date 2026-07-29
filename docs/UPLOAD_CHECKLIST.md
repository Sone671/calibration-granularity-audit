# GitHub upload checklist

## Automated checks

- [x] Code, tests, protocols, aggregate results, and data checksums staged.
- [x] Raw archives, prepared user-level data, customer identifiers, model binaries, caches, and local paths excluded.
- [x] Manuscript PDFs, LaTeX sources, figures, and supplementary files excluded.
- [x] Seven unit tests pass in Python 3.12.13.
- [x] Manifest and SHA-256 verification pass.
- [x] Largest file is below GitHub's 100 MB hard limit and the repository is below 5 MB.
- [x] CI workflow and `.gitignore` included.

## Public release status

- [x] Author names, manuscript title, public repository URL, and release version recorded in `CITATION.cff`.
- [x] Manuscript and supplementary files withheld from the public repository.
- [x] Repository name and description aligned with the manuscript.
- [x] MIT code license, CC BY 4.0 content license, and scope notice included.
- [ ] Add the final accepted-manuscript or archival DOI when available.

## Suggested first push

```bash
git init
git branch -M main
git add .
git status --short
python scripts/verify_repository.py
pytest -q tests
git commit -m "Publish reproducibility package v1.1.0"
git remote add origin https://github.com/Sone671/calibration-granularity-audit.git
git push -u origin main
```

Do not use `git add -f` on anything ignored under `data/raw/`, `data/prepared/`, `outputs/`, or archive/model patterns.
