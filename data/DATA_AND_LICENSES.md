# Data, access, and licensing record

Access date for the URLs below: 2026-07-28. Raw data are not included in the repository. Users must comply with each provider’s current terms before downloading or processing the data.

| Data set | Provider / source | Access point | Required release check |
|---|---|---|---|
| London smart-meter load | Greater London Authority DataStore; Low Carbon London / UK Power Networks | https://data.london.gov.uk/dataset/smartmeter-energy-consumption-data-in-london-households-vqm0d | Verified 795,722,689-byte split-file ZIP; SHA-256 and the 168-member CRC/manifest identity are in `data_checksums.json`. The provider lists Creative Commons Attribution. |
| Ausgrid Solar Home Electricity Data | Ausgrid research-data catalog; public archival mirror | https://www.ausgrid.com.au/about-us/about-ausgrid/research-data-sets and https://pierreh.eu/downloads/Ausgrid_solar_home_data.zip | Verified enclosing ZIP plus SHA-256 values for the 2010--11, 2011--12, and 2012--13 CSV members. The mirror is an archival copy, not a new official release. |
| ElectricityLoadDiagrams20112014 | UCI Machine Learning Repository | https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014 | Verified the 249.2 MB provider ZIP, its CRC, and `LD2011_2014.txt` member hash. DOI: https://doi.org/10.24432/C58C86; UCI lists CC BY 4.0. |

## Data minimization and anonymization

Do not deposit raw customer-level values, customer IDs, model-prediction arrays, or unaggregated per-user coverage files in a public repository unless the applicable licence explicitly permits it and the de-identification review is complete. The frozen manuscript reports only aggregate diagnostics. Re-runners should keep downloaded raw data outside the repository and provide paths explicitly on the command line.

## Verified release record

`data_checksums.json` contains source URLs, access date, archive byte counts and SHA-256 values, ZIP CRC outcomes, member-manifest hashes, and selected member hashes. It contains no credentials, cookies, customer identifiers, or local absolute paths. Provider pages and the UCI DOI/license were rechecked on 2026-07-29.
