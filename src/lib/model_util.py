from __future__ import annotations

import logging
import os
from typing import Any

import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .constants import LORA_DROPOUT

LOGGER = logging.getLogger(__name__)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def choose_train_batch_size(quant: str) -> int:
    env = os.environ.get("A1_BATCH")
    if env:
        return max(1, int(env))
    if not torch.cuda.is_available():
        return 1
    gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    if gb >= 30:
        return 8
    if gb >= 20:
        return 4
    if gb >= 14:
        return 2 if quant == "none" else 4
    return 1


def accumulation_steps(batch_size: int, effective: int) -> int:
    return max(1, effective // max(1, batch_size))


def load_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
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
    kwargs: dict[str, Any] = {"trust_remote_code": True}
    if qconf is not None:
        kwargs["quantization_config"] = qconf
        kwargs["device_map"] = "auto"
    else:
        kwargs["torch_dtype"] = autocast_dtype() if pick_device().type == "cuda" else torch.float32
        if torch.cuda.is_available():
            kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
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
