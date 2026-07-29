from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 50 * 1024 * 1024
FORBIDDEN_SUFFIXES = {".zip", ".7z", ".tar", ".gz", ".parquet", ".feather", ".pkl", ".joblib", ".model", ".bin"}
TEXT_SUFFIXES = {".py", ".md", ".txt", ".json", ".yml", ".yaml", ".csv", ".cff", ""}
PATH_PATTERNS = [re.compile(r"[A-Za-z]:\\"), re.compile(r"/Users/"), re.compile(r"/home/[^/\s]+/")]
SECRET_PATTERNS = [re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"), re.compile(r"github_pat_[A-Za-z0-9_]+"), re.compile(r"ghp_[A-Za-z0-9]{20,}"), re.compile(r"AKIA[0-9A-Z]{16}")]
FORBIDDEN_RESULT_COLUMNS = {"customer", "customer_id", "meter_id", "user_id", "user_index", "household_id", "raw_load", "load_value"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


errors = []
manifest = json.loads((ROOT / "MANIFEST.json").read_text(encoding="utf-8"))
manifest_paths = {row["path"] for row in manifest["files"]}
for row in manifest["files"]:
    path = ROOT / row["path"]
    if not path.is_file():
        errors.append(f"missing manifest file: {row['path']}")
        continue
    if path.stat().st_size != row["bytes"] or sha256(path) != row["sha256"]:
        errors.append(f"manifest mismatch: {row['path']}")

for path in ROOT.rglob("*"):
    if not path.is_file() or any(part in {".git", ".pytest_cache", "__pycache__"} for part in path.parts):
        continue
    relative = path.relative_to(ROOT).as_posix()
    if relative not in manifest_paths and path.name not in {"MANIFEST.json", "MANIFEST.sha256"}:
        errors.append(f"unmanifested file: {relative}")
    if path.stat().st_size > MAX_BYTES:
        errors.append(f"file exceeds 50 MB: {relative}")
    if path.suffix.lower() in FORBIDDEN_SUFFIXES:
        errors.append(f"forbidden raw/archive/model suffix: {relative}")
    if path.name in {"MANIFEST.json", "MANIFEST.sha256"} or path.resolve() == Path(__file__).resolve():
        continue
    if path.suffix.lower() in TEXT_SUFFIXES and path.stat().st_size < 5 * 1024 * 1024:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for pattern in PATH_PATTERNS:
            if pattern.search(text):
                errors.append(f"absolute path pattern in {relative}")
        for pattern in SECRET_PATTERNS:
            if pattern.search(text):
                errors.append(f"secret/private-key pattern in {relative}")

checksums = json.loads((ROOT / "data" / "checksums.json").read_text(encoding="utf-8"))
if checksums.get("contains_raw_data") is not False:
    errors.append("data/checksums.json must declare contains_raw_data=false")
if any(item.get("zip_crc_check") != "passed" for item in checksums.get("archives", {}).values()):
    errors.append("one or more source archive CRC checks did not pass")

for path in (ROOT / "results").glob("*.csv"):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle), [])
    overlap = FORBIDDEN_RESULT_COLUMNS.intersection(column.strip().lower() for column in header)
    if overlap:
        errors.append(f"identifier/raw-value columns in {path.relative_to(ROOT).as_posix()}: {sorted(overlap)}")

if errors:
    raise SystemExit("repository verification failed:\n- " + "\n- ".join(errors))
print(f"repository verification passed: {len(manifest['files'])} manifested files")
