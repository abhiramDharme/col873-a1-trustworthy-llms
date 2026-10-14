#!/usr/bin/env python
"""Kaggle GPU entrypoint. Installs assignment deps on top of the image torch, then runs run_all.sh."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

WORKING = Path("/kaggle/working")
INPUT = Path("/kaggle/input")


def pip_install() -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], stdout=subprocess.DEVNULL)
    pkgs = [
        "transformers>=4.44.0,<5",
        "peft==0.14.0",
        "trl>=0.11.0,<0.20",
        "accelerate>=0.33.0",
        "datasets>=2.20.0,<4",
        "bitsandbytes>=0.43.0,<0.46",
        "presidio-analyzer>=2.2.0",
        "spacy>=3.7.0,<3.9",
        "gdown>=5.2.0",
        "matplotlib>=3.8.0",
        "seaborn>=0.13.0",
        "tqdm",
    ]
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *pkgs])
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])


def _has_src(path: Path) -> bool:
    return (path / "src" / "peft_sweep.py").is_file()


def _extract_tars(src: Path, dest: Path) -> None:
    import tarfile

    dest.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        if item.suffix == ".tar" or item.name.endswith(".tar.gz"):
            print("extract", item)
            with tarfile.open(item, "r:*") as archive:
                archive.extractall(dest)
        elif item.is_dir():
            target = dest / item.name
            if not target.exists():
                shutil.copytree(item, target, dirs_exist_ok=True)
        elif item.is_file() and item.name not in {"dataset-metadata.json"}:
            shutil.copy2(item, dest / item.name)


def find_bundle() -> Path:
    env = os.environ.get("A1_ROOT")
    if env and _has_src(Path(env)):
        return Path(env)
    candidates = []
    if WORKING.is_dir():
        candidates.append(WORKING)
    if INPUT.is_dir():
        for child in sorted(INPUT.iterdir()):
            print("kaggle input", child, "contents", [p.name for p in child.iterdir()][:30])
            candidates.append(child)
            unpacked = WORKING / "bundle"
            _extract_tars(child, unpacked)
            candidates.append(unpacked)
    here = Path(__file__).resolve()
    candidates.extend(here.parents)
    for path in candidates:
        if _has_src(path):
            return path
        for match in path.rglob("peft_sweep.py"):
            if match.parent.name == "src":
                return match.parent.parent
    raise FileNotFoundError("could not locate src/peft_sweep.py")


def main() -> None:
    print("python", sys.version)
    try:
        import torch

        print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
    except Exception as exc:
        print("torch missing", exc)
    pip_install()
    bundle = find_bundle()
    print("bundle", bundle)
    out = WORKING if WORKING.is_dir() else bundle
    env = os.environ.copy()
    env["A1_ROOT"] = str(bundle)
    data = bundle / "data"
    if not data.is_dir():
        data = bundle / "trustworthy_llms_a1_bundle" / "data"
    env["A1_DATA"] = str(data)
    env["A1_OUT"] = str(out)
    env["HF_HOME"] = str(out / ".hf")
    env["STAGE"] = env.get("STAGE", "all")
    env["PYTHON"] = sys.executable
    env["SKIP_EXISTING"] = env.get("SKIP_EXISTING", "1")
    script = bundle / "run_all.sh"
    subprocess.check_call(["bash", str(script)], env=env, cwd=str(bundle))
    # Copy required result files to the working directory root for easy download.
    src_results = Path(env["A1_OUT"]) / "results"
    dest = WORKING / "results" if WORKING.is_dir() else src_results
    dest.mkdir(parents=True, exist_ok=True)
    if src_results.resolve() != dest.resolve() and src_results.is_dir():
        for item in src_results.iterdir():
            target = dest / item.name
            if item.is_file():
                shutil.copy2(item, target)
    print("done. results in", dest)


if __name__ == "__main__":
    main()
