from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Iterator


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_json(path: Path) -> Any:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def verify_manifest(data_root: Path) -> None:
    manifest = data_root / "manifest.jsonl"
    for line in manifest.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        payload = (data_root / item["file"]).read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if digest != item["sha256"]:
            raise ValueError(f"SHA-256 mismatch for {item['file']}")
        n_rows = len(payload.splitlines())
        if n_rows != item["rows"]:
            raise ValueError(f"Row count mismatch for {item['file']}: {n_rows} != {item['rows']}")


def csv_has_complete_row(path: Path, run_id: str, required: str = "val_entity_f1") -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("run_id") != str(run_id):
                continue
            value = (row.get(required) or "").strip()
            if value:
                return True
    return False


def upsert_csv_row(path: Path, fieldnames: list[str] | tuple[str, ...], row: dict[str, Any], key: str = "run_id") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    if path.is_file() and path.stat().st_size > 0:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    rows = [existing for existing in rows if existing.get(key) != str(row.get(key))]
    serialized = {name: row.get(name, "") for name in fieldnames}
    rows.append(serialized)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)
