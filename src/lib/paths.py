from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    env = os.environ.get("A1_ROOT")
    if env:
        return Path(env).resolve()
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "src" / "peft_sweep.py").is_file():
            return parent
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        for child in sorted(kaggle_input.iterdir()):
            if (child / "src" / "peft_sweep.py").is_file():
                return child
    return Path.cwd()


def data_dir() -> Path:
    env = os.environ.get("A1_DATA")
    if env:
        return Path(env).resolve()
    root = project_root()
    for candidate in (root / "data", root / "trustworthy_llms_a1_bundle" / "data"):
        if candidate.is_dir():
            return candidate
    return root / "data"


def output_root() -> Path:
    env = os.environ.get("A1_OUT")
    if env:
        return Path(env).resolve()
    if Path("/kaggle/working").is_dir():
        return Path("/kaggle/working")
    return project_root()


def results_dir() -> Path:
    path = output_root() / "results"
    path.mkdir(parents=True, exist_ok=True)
    return path


def adapters_dir() -> Path:
    path = output_root() / "adapters"
    path.mkdir(parents=True, exist_ok=True)
    return path
