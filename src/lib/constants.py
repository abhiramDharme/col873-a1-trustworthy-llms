"""Fixed experiment constants. Changing these without updating run_all.sh breaks comparability."""

from __future__ import annotations

PII_LABELS_300K: tuple[str, ...] = (
    "BOD",
    "BUILDING",
    "CARDISSUER",
    "CITY",
    "COUNTRY",
    "DATE",
    "DRIVERLICENSE",
    "EMAIL",
    "GEOCOORD",
    "GIVENNAME1",
    "GIVENNAME2",
    "IDCARD",
    "IP",
    "LASTNAME1",
    "LASTNAME2",
    "LASTNAME3",
    "PASS",
    "PASSPORT",
    "POSTCODE",
    "SECADDRESS",
    "SEX",
    "SOCIALNUMBER",
    "STATE",
    "STREET",
    "TEL",
    "TIME",
    "TITLE",
    "USERNAME",
)

SANITIZER_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
VICTIM_MODEL = "Qwen/Qwen2.5-0.5B"

# Locked across the Part A sweep (assignment: same data, seed, epochs, schedule, effective batch).
DEFAULT_SEED = 42
SANITIZER_EPOCHS = 2
SANITIZER_LR = 2.0e-4
SANITIZER_WARMUP = 0.03
SANITIZER_MAX_SEQ = 768
SANITIZER_MAX_NEW = 384
LORA_DROPOUT = 0.05
EFFECTIVE_BATCH = 16

VICTIM_EPOCHS = 4
VICTIM_LR = 2.0e-4
VICTIM_MAX_SEQ = 512
VICTIM_MAX_NEW = 256
VICTIM_LORA_R = 32
VICTIM_LORA_ALPHA = 64
VICTIM_TARGET_MODULES: tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

SWEEP: dict[str, dict] = {
    "A1": {
        "method": "LoRA",
        "target_modules": ["q_proj", "v_proj"],
        "lora_r": 8,
        "lora_alpha": 16,
        "quant": "none",
    },
    "A2": {
        "method": "LoRA",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "lora_r": 8,
        "lora_alpha": 16,
        "quant": "none",
    },
    "A3": {
        "method": "LoRA",
        "target_modules": list(VICTIM_TARGET_MODULES),
        "lora_r": 8,
        "lora_alpha": 16,
        "quant": "none",
    },
    "A4": {
        "method": "LoRA",
        "target_modules": list(VICTIM_TARGET_MODULES),
        "lora_r": 32,
        "lora_alpha": 64,
        "quant": "none",
    },
    "A5": {
        "method": "LoRA",
        "target_modules": list(VICTIM_TARGET_MODULES),
        "lora_r": 32,
        "lora_alpha": 16,
        "quant": "none",
    },
    "A6": {
        "method": "QLoRA",
        "target_modules": list(VICTIM_TARGET_MODULES),
        "lora_r": 32,
        "lora_alpha": 64,
        "quant": "4bit",
    },
}

PARTA_CSV_FIELDS: tuple[str, ...] = (
    "run_id",
    "method",
    "target_modules",
    "lora_r",
    "lora_alpha",
    "quant",
    "trainable_params",
    "total_params",
    "trainable_pct",
    "peak_gpu_mem_gb",
    "train_wallclock_s",
    "tokens_per_s",
    "final_train_loss",
    "val_loss",
    "val_entity_f1",
    "val_leak_rate",
)

CANARY_UNIVERSE = 1_000_000
CANARY_SAMPLE = 10_000
UNTARGETED_N = 5000
TARGETED_SAMPLES = 10
