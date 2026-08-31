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
    pkgs = [
        "transformers>=4.44.0",
        "peft>=0.12.0",
        "trl>=0.11.0",
        "accelerate>=0.33.0",
        "datasets>=2.20.0",
        "bitsandbytes>=0.43.0",
        "presidio-analyzer>=2.2.0",
        "spacy>=3.7.0",
        "matplotlib>=3.8.0",
        "seaborn>=0.13.0",
        "tqdm",
    ]
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *pkgs])
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])


def find_bundle() -> Path:
    env = os.environ.get("A1_ROOT")
    if env and (Path(env) / "src" / "peft_sweep.py").is_file():
        return Path(env)
    if (WORKING / "src" / "peft_sweep.py").is_file():
        return WORKING
    if INPUT.is_dir():
        for child in sorted(INPUT.iterdir()):
            if (child / "src" / "peft_sweep.py").is_file():
                return child
            nested = child / "A1"
            if (nested / "src" / "peft_sweep.py").is_file():
                return nested
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "src" / "peft_sweep.py").is_file():
            return parent
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
