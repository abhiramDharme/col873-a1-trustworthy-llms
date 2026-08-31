from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib.paths import adapters_dir, results_dir


def select_best(csv_path: Path, dest: Path) -> str:
    with csv_path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"no sweep rows in {csv_path}")
    def key(row):
        return (float(row["val_leak_rate"]), -float(row["val_entity_f1"]), float(row["trainable_pct"]))
    best = min(rows, key=key)
    run_id = best["run_id"]
    src = adapters_dir() / run_id
    if not (src / "adapter_config.json").is_file():
        raise SystemExit(f"missing adapter for {run_id} at {src}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest)
    note = dest.parent / "best_sanitizer_reason.txt"
    note.write_text(
        f"selected {run_id}\n"
        f"val_leak_rate={best['val_leak_rate']}\n"
        f"val_entity_f1={best['val_entity_f1']}\n"
        f"trainable_pct={best['trainable_pct']}\n"
        "tie-break: lowest leak rate, then highest exact entity F1, then fewer trainable params\n",
        encoding="utf-8",
    )
    return run_id


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results_csv", type=Path, default=None)
    p.add_argument("--dest", type=Path, default=None)
    args = p.parse_args()
    csv_path = args.results_csv or (results_dir() / "partA_sweep.csv")
    dest = args.dest or (adapters_dir() / "best_sanitizer")
    run_id = select_best(csv_path, dest)
    print(run_id)


if __name__ == "__main__":
    main()
