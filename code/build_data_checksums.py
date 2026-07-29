"""Create a data-free checksum record for the three source archives."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile


def sha256_stream(handle) -> str:
    digest = hashlib.sha256()
    while chunk := handle.read(1024 * 1024):
        digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return sha256_stream(handle)


def archive_record(path: Path, source_url: str, hash_members: tuple[str, ...]) -> dict:
    with ZipFile(path) as archive:
        bad_member = archive.testzip()
        members = [
            {
                "name": info.filename,
                "uncompressed_bytes": info.file_size,
                "compressed_bytes": info.compress_size,
                "crc32": f"{info.CRC:08x}",
            }
            for info in archive.infolist()
        ]
        manifest_bytes = json.dumps(members, sort_keys=True, separators=(",", ":")).encode()
        selected = []
        for name in hash_members:
            info = archive.getinfo(name)
            with archive.open(info) as handle:
                member_hash = sha256_stream(handle)
            selected.append(
                {
                    "name": name,
                    "uncompressed_bytes": info.file_size,
                    "sha256": member_hash,
                }
            )
    return {
        "source_url": source_url,
        "local_filename": path.name,
        "archive_bytes": path.stat().st_size,
        "archive_sha256": sha256_file(path),
        "zip_crc_check": "passed" if bad_member is None else f"failed:{bad_member}",
        "member_count": len(members),
        "member_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "selected_members": selected,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--london",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--ausgrid",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--uci",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "DATA_CHECKSUMS.json",
    )
    args = parser.parse_args()
    record = {
        "schema_version": 1,
        "access_date": "2026-07-28",
        "contains_raw_data": False,
        "archives": {
            "london": archive_record(
                args.london,
                "https://data.london.gov.uk/dataset/smartmeter-energy-consumption-data-in-london-households-vqm0d",
                (),
            ),
            "ausgrid": archive_record(
                args.ausgrid,
                "https://pierreh.eu/downloads/Ausgrid_solar_home_data.zip",
                (
                    "Solar home 2010-2011.csv",
                    "Solar home 2011-2012.csv",
                    "Solar home 2012-2013.csv",
                ),
            ),
            "uci": archive_record(
                args.uci,
                "https://archive.ics.uci.edu/static/public/321/electricityloaddiagrams20112014.zip",
                ("LD2011_2014.txt",),
            ),
        },
    }
    args.output.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "archives": 3}, indent=2))


if __name__ == "__main__":
    main()
