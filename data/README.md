# Data acquisition and verification

Raw user-level load data are not stored in this repository. They are too large for ordinary GitHub hosting and include household/client identifiers that should not be republished without a separate licensing and de-identification decision.

## Sources

| Dataset | Provider/access point | Expected local input |
|---|---|---|
| London | https://data.london.gov.uk/dataset/smartmeter-energy-consumption-data-in-london-households-vqm0d | `Partitioned LCL Data.zip` or equivalent 168-file provider archive |
| Ausgrid | https://www.ausgrid.com.au/about-us/about-ausgrid/research-data-sets; archival copy documented in `checksums.json` | Archive containing the 2010–11, 2011–12, and 2012–13 annual CSV files |
| UCI Electricity | https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014 | Provider ZIP containing `LD2011_2014.txt`; DOI 10.24432/C58C86 |

## Verification

`checksums.json` records archive byte counts, SHA-256 values, ZIP CRC outcomes, member-manifest hashes, and selected member hashes. Verify before preprocessing:

```bash
python code/build_data_checksums.py --london /path/to/london.zip --ausgrid /path/to/ausgrid.zip --uci /path/to/uci.zip --output local_checksums.json
```

Compare `local_checksums.json` with `data/checksums.json`. A mismatch means the source revision or download differs from the frozen analysis. Do not rename a truncated or HTML response to `.zip`; require a successful ZIP CRC check.

## Local directory policy

Recommended locations, all ignored by Git:

```text
data/raw/london/
data/raw/ausgrid/
data/raw/uci/
data/prepared/
outputs/
```

Only aggregated result tables under `results/` are approved for repository release. Dataset licenses remain those of the original providers; the repository's MIT and CC BY 4.0 licenses do not relicense external data. The detailed provider and licensing record is in `DATA_AND_LICENSES.md`.
