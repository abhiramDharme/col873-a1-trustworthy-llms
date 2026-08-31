from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .constants import LORA_DROPOUT

LOGGER = logging.getLogger(__name__)


def _silence_incompatible_torchao() -> None:
    """PEFT 0.20 raises if torchao is present but older than 0.16 (Kaggle image)."""
    try:
        import peft.import_utils as import_utils
    except Exception:
        return
    fn = getattr(import_utils, "is_torchao_available", None)
    if fn is None:
        return

    def _ok() -> bool:
        try:
            return bool(fn())
        except ImportError:
            return False

    import_utils.is_torchao_available = _ok  # type: ignore[assignment]


_silence_incompatible_torchao()


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def amp_flags() -> tuple[bool, bool]:
    """Return (fp16, bf16). Ampere+ uses bf16; Volta/Turing stay on fp16."""
    if not torch.cuda.is_available():
        return False, False
    if torch.cuda.get_device_capability()[0] >= 8:
        return False, True
    return True, False


def gpu_mem_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.get_device_properties(0).total_memory / (1024**3)


def choose_train_batch_size(quant: str) -> int:
    env = os.environ.get("A1_BATCH")
    if env:
        return max(1, int(env))
    if not torch.cuda.is_available():
        return 1
    gb = gpu_mem_gb()
    if gb >= 70:
        return 16
    if gb >= 30:
        return 8
    if gb >= 20:
        return 4
    if gb >= 14:
        return 2 if quant == "none" else 4
    return 1


def choose_gen_batch_size() -> int:
    env = os.environ.get("A1_GEN_BATCH")
    if env:
        return max(1, int(env))
    gb = gpu_mem_gb()
    if gb >= 40:
        return 16
    if gb >= 14:
        return 8
    return 1 if not torch.cuda.is_available() else 4


def use_grad_checkpoint() -> bool:
    # Checkpointing trades speed for memory; skip it on 40GB+.
    return gpu_mem_gb() < 40


def build_training_args(kwargs: dict) -> "object":
    """Construct TrainingArguments across transformers 4.x and 5.x argument names."""
    import re

    from transformers import TrainingArguments

    pending = dict(kwargs)
    for _ in range(16):
        try:
            return TrainingArguments(**pending)
        except TypeError as exc:
            msg = str(exc)
            match = re.search(r"unexpected keyword argument '([^']+)'", msg)
            if match:
                key = match.group(1)
                if key == "warmup_ratio":
                    pending.pop("warmup_ratio", None)
                    pending.setdefault("warmup_steps", 100)
                    continue
                pending.pop(key, None)
                continue
            if "eval_strategy" in pending:
                pending["evaluation_strategy"] = pending.pop("eval_strategy")
                continue
            raise
    return TrainingArguments(**pending)


def accumulation_steps(batch_size: int, effective: int) -> int:
    return max(1, effective // max(1, batch_size))


def adapter_is_complete(output_dir: Path) -> bool:
    root = Path(output_dir)
    return (root / "adapter_config.json").is_file() and (
        (root / "adapter_model.safetensors").is_file() or (root / "adapter_model.bin").is_file()
    )


def latest_checkpoint(output_dir: Path) -> Path | None:
    """Highest-step Hugging Face checkpoint under output_dir/trainer_state."""
    roots = [Path(output_dir) / "trainer_state", Path(output_dir)]
    ckpts: list[Path] = []
    for root in roots:
        if not root.is_dir():
            continue
        ckpts.extend(p for p in root.glob("checkpoint-*") if p.is_dir())
    if not ckpts:
        return None

    def step_num(path: Path) -> int:
        try:
            return int(path.name.rsplit("-", 1)[-1])
        except ValueError:
            return -1

    return max(ckpts, key=step_num)


def load_tokenizer(model_name: str):
    local = os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get("TRANSFORMERS_OFFLINE") == "1"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, local_files_only=local)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def _quant_config(quant: str):
    if quant in (None, "", "none"):
        return None
    if quant not in {"4bit", "qlora"}:
        raise ValueError(f"Unsupported quant={quant}")
    if not torch.cuda.is_available():
        raise RuntimeError("4-bit QLoRA requires CUDA (bitsandbytes).")
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )


def load_causal_lm(model_name: str, quant: str = "none"):
    qconf = _quant_config(quant)
    local = os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get("TRANSFORMERS_OFFLINE") == "1"
    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "local_files_only": local,
        "attn_implementation": "sdpa",
    }
    if qconf is not None:
        kwargs["quantization_config"] = qconf
        kwargs["device_map"] = "auto"
        kwargs["torch_dtype"] = torch.float16
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    else:
        kwargs["torch_dtype"] = autocast_dtype() if pick_device().type == "cuda" else torch.float32
        try:
            model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        except (TypeError, ValueError):
            kwargs.pop("attn_implementation", None)
            model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        if torch.cuda.is_available():
            model = model.to("cuda")
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_disable") and not use_grad_checkpoint():
        model.gradient_checkpointing_disable()
    model.config.pad_token_id = model.config.pad_token_id or model.config.eos_token_id
    return model


def attach_lora(model, target_modules: list[str], r: int, alpha: int, quant: str = "none"):
    if _quant_config(quant) is not None:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(target_modules),
    )
    model = get_peft_model(model, config)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.print_trainable_parameters()
    return model


def load_with_adapter(model_name: str, adapter_path: str | None, quant: str = "none"):
    model = load_causal_lm(model_name, quant=quant)
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model


def trainable_counts(model) -> tuple[int, int, float]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = 100.0 * trainable / total if total else 0.0
    return trainable, total, pct


def peak_gpu_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024**3)


def reset_peak_gpu() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
