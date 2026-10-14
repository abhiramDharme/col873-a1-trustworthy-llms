"""Fetch submitted LoRA adapters from the public TA Drive folder.

"""

from __future__ import annotations

import logging
from pathlib import Path

LOGGER = logging.getLogger(__name__)

# https://drive.google.com/drive/folders/17dQg4IVXkHNKDPLkotSCd_WeagXzKW1t
DRIVE_FOLDER_ID = "17dQg4IVXkHNKDPLkotSCd_WeagXzKW1t"

# Relative path inside the Drive folder -> Google file id.
DRIVE_FILES: dict[str, str] = {
    "A1/adapter_config.json": "1YQMpHksJ05TwnkVACy6yyyso3T6hqz_5",
    "A1/adapter_model.safetensors": "1-xNEU8SYnntcY2RvtTmA-nU_pqrArZrT",
    "A1/chat_template.jinja": "1EQW-_8ht4fL0GLCAqXeFcIFeET5cUDD3",
    "A1/README.md": "1vTsxeQXYsFiDvZ9qoiNWlttHtOaPmKTW",
    "A1/tokenizer.json": "1f5_aTXiR8uzi-xmybAVB7ROc8dF8qLAs",
    "A1/tokenizer_config.json": "1hxqxMj1A3a6hUcCE1nLFRo4FGE2X1o7m",
    "A2/adapter_config.json": "1wOt7WsAqx13szp3diMRwyIp2quN2K3rS",
    "A2/adapter_model.safetensors": "1tEDYGYqTxH4g7J0LDpZCrytgV7WeKymk",
    "A2/chat_template.jinja": "1lE5OlT-B6ySfW8QLnhR7eElk3xtbDoCW",
    "A2/README.md": "1ISwT9sI5898AYmVBxv4d-ymKwlWbCRIp",
    "A2/tokenizer.json": "1x5xwHviGEwIhWbXT2Scr_UCOvbPt8b8r",
    "A2/tokenizer_config.json": "17oYDgNw0y8c-95g6IDjfEDxy6JSsUAJi",
    "A3/adapter_config.json": "1hRygtR15DB5KynYijlqykegs_uRjzf16",
    "A3/adapter_model.safetensors": "1W_5hAWEmYuUzwJORUxSt1Uj_Di3lL2tq",
    "A3/chat_template.jinja": "1TwBHZG8tToRsTqoQR1m0CAoE-iJz-0LW",
    "A3/README.md": "19BJ3_S1ridf5u3iGODqPuheeAxhB334y",
    "A3/tokenizer.json": "1nBvJHzoiP9vi3mAhYr9gkV-PPf7TppJZ",
    "A3/tokenizer_config.json": "1tH8q5bxTffD8qAUe-hR4pgGQ4c39SFuZ",
    "A4/adapter_config.json": "1Xqd_NYZ6bunK2GLJOuiUFXNGOl7BfinY",
    "A4/adapter_model.safetensors": "1IhbfniBdKENsw3uMqm5JEy8fLHBBFP0o",
    "A4/chat_template.jinja": "1KPSZa6JH_087rNULfr98Rb7MBwYJ8bg2",
    "A4/README.md": "1p5WMjNh-t48ZvlIkeJnKxeti8nkWAG6F",
    "A4/tokenizer.json": "1_KdIpKJDpDpIm4OMu0-16pVd6vtOB4wR",
    "A4/tokenizer_config.json": "1K-EI3ChlBmI3PJoWxOlSh9KxuS6-EePW",
    "A5/adapter_config.json": "1iqlw1vsmdwMHGmKl5Oof2piVuEDgrxo-",
    "A5/adapter_model.safetensors": "1b2v8f1wbEA5Wvny0Ed14Ekh3Hcto619Z",
    "A5/chat_template.jinja": "1rEi2m_RkxVrPXQqBFK1DmfElyiMe7HAh",
    "A5/README.md": "1Ty6aHGRwPLJWyhap8ERXYopKA1jmjGBu",
    "A5/tokenizer.json": "1_HX_kjN-C7HThxJ59QJFHYSzqQbM2KMS",
    "A5/tokenizer_config.json": "1A3PXnA-KuGUTPZi9Ru67mcA4TPIHRZw2",
    "A6/adapter_config.json": "1B9HCcfFoLnwGfnt9WK0VSaaSfnuNLECM",
    "A6/adapter_model.safetensors": "1pUz9iYxnmEahXrIiNy_Uep1G84GSRZ6G",
    "A6/chat_template.jinja": "1v2BT2-B7qtFvFZW5rlz18BNXfhnuWHHS",
    "A6/README.md": "1Yf6MUbWN1eOlJTqwRnd0DOiL4SRtSqGw",
    "A6/tokenizer.json": "1xFaxY6HQZJEkzlI4R4MeGcpT9XBPYpD-",
    "A6/tokenizer_config.json": "1rsU5jQOyceIRELe5We8r80pgE34SqeFz",
    "best_sanitizer/adapter_config.json": "1IG_IDpw9T8OFOYhrn8fcTbzRjiUhzcTD",
    "best_sanitizer/adapter_model.safetensors": "12kFIz7-tH0yid0FlBM6hCT3eWoQSTVqN",
    "best_sanitizer/chat_template.jinja": "1DwGjdlD3Ng-KNNM9a8vxVBjPB6q82E_W",
    "best_sanitizer/README.md": "1NKWRV9Szd4bYMcriIzE65vIZ2AQ6w733",
    "best_sanitizer/tokenizer.json": "13KYrm03fNg-aySOrDw-w1D23EG5eHg_U",
    "best_sanitizer/tokenizer_config.json": "1PirdcB20-HZt4xaZyIuJFKIKGDcy5bqC",
    "victim_raw/adapter_config.json": "1ZOQHNgGERq6DC4WZv93e8EyH04IFTCPQ",
    "victim_raw/adapter_model.safetensors": "1W5XvlXf2krFCyHi1kzPoEOF5w1pQjy8t",
    "victim_raw/chat_template.jinja": "1LVFym9WCiCYs0AbVt0rmB71aDLrag1kN",
    "victim_raw/heldout_ppl.json": "1N11L6F4kiaWBaRVOcpW7lCRCtbQ8y6Uq",
    "victim_raw/README.md": "1E595Qk6xRYc9ix0AHkrJBMYWEVE_1CkR",
    "victim_raw/tokenizer.json": "1kivI5WuwBnirP53Aca4NZNXBqtmH8BdC",
    "victim_raw/tokenizer_config.json": "1blprd3LhAAwo71Dk3IrrbAbJrD0uYKbc",
    "victim_san/adapter_config.json": "1dw_EuUvYQowppiyIdiGRoU1LhWmbIyJI",
    "victim_san/adapter_model.safetensors": "1oMu9evCjlX1UMuCg05KTQzDWsyxwEfLR",
    "victim_san/chat_template.jinja": "1BUA58xFX_5_veuiMLrcdM7ZLM5cTlxUj",
    "victim_san/heldout_ppl.json": "1e6WhteEeqIyiohyd8_9hUjjsj0KPcay6",
    "victim_san/README.md": "1TobjokK_NeO1hktKG1nxd57H-2PjEsI8",
    "victim_san/tokenizer.json": "1vIHyRd3xoBcLG11tTkmMhhiK8Jajq9rn",
    "victim_san/tokenizer_config.json": "1zf72maxKloeOnaHTOx_2D3Z-a-heGRQw",
    "best_sanitizer_reason.txt": "1Qj3333J7HnIx4oY8vMRuhjRT__40quJq",
}


def _download_file(file_id: str, dest: Path) -> None:
    if dest.is_file() and dest.stat().st_size > 0:
        return
    try:
        import gdown
    except ImportError as exc:
        raise SystemExit("Missing gdown. Install dependencies with: pip install -r requirements.txt") from exc

    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".partial")
    if partial.exists():
        partial.unlink()
    LOGGER.info("downloading %s", dest.name)
    saved = gdown.download(id=file_id, output=str(partial), quiet=False, use_cookies=False, retries=3)
    if not saved or not partial.is_file() or partial.stat().st_size <= 0:
        raise RuntimeError(f"Drive download failed for {dest.name} (id={file_id})")
    partial.replace(dest)


def ensure_local_adapter(adapter_path: str | Path | None) -> str | None:
    """Return a local adapter directory, downloading it from Drive when needed."""
    if adapter_path is None:
        return None
    text = str(adapter_path).strip()
    if not text:
        return None
    root = Path(text)
    name = root.name
    wanted = {rel: file_id for rel, file_id in DRIVE_FILES.items() if rel.startswith(name + "/")}
    if not wanted:
        return text
    if _adapter_is_complete(root):
        return str(root)
    LOGGER.info("fetching adapter %s from Drive folder %s", name, DRIVE_FOLDER_ID)
    for rel, file_id in wanted.items():
        _download_file(file_id, root / Path(rel).name)
    if name == "best_sanitizer":
        reason_id = DRIVE_FILES.get("best_sanitizer_reason.txt")
        if reason_id:
            _download_file(reason_id, root.parent / "best_sanitizer_reason.txt")
    if not _adapter_is_complete(root):
        raise FileNotFoundError(f"adapter {name} is still incomplete after Drive download: {root}")
    return str(root)


def _adapter_is_complete(output_dir: Path) -> bool:
    return (output_dir / "adapter_config.json").is_file() and (
        (output_dir / "adapter_model.safetensors").is_file() or (output_dir / "adapter_model.bin").is_file()
    )
