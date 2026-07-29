from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {"MANIFEST.json", "MANIFEST.sha256"}
IGNORED_PARTS = {".git", ".pytest_cache", "__pycache__"}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


rows = []
for path in sorted(ROOT.rglob("*")):
    if not path.is_file() or path.name in EXCLUDED or any(part in IGNORED_PARTS for part in path.parts):
        continue
    rows.append({"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": digest(path)})
(ROOT / "MANIFEST.json").write_text(json.dumps({"repository": "calibration-granularity-audit", "contains_raw_data": False, "files": rows}, indent=2), encoding="utf-8")
(ROOT / "MANIFEST.sha256").write_text("\n".join(f"{row['sha256']}  {row['path']}" for row in rows) + "\n", encoding="utf-8")
print(f"manifest rebuilt: {len(rows)} files")
