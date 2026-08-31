from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import torch
from transformers import Trainer, TrainingArguments

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib.constants import (
    DEFAULT_SEED,
    EFFECTIVE_BATCH,
    VICTIM_EPOCHS,
    VICTIM_LR,
    VICTIM_LORA_ALPHA,
    VICTIM_LORA_R,
    VICTIM_MAX_SEQ,
    VICTIM_MODEL,
    VICTIM_TARGET_MODULES,
)
from lib.generate import sequence_nll
from lib.io_utils import read_jsonl, write_json
from lib.model_util import (
    accumulation_steps,
    attach_lora,
    choose_train_batch_size,
    load_causal_lm,
    load_tokenizer,
    load_with_adapter,
    peak_gpu_gb,
    reset_peak_gpu,
)
from lib.paths import data_dir, results_dir
from lib.seed import seed_everything
from lib.sft_data import CausalTextDataset, PadCollator

LOGGER = logging.getLogger("train_victim")


def parse_modules(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Part C victim LoRA training.")
    p.add_argument("--model", default=VICTIM_MODEL)
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--heldout", type=Path, default=None)
    p.add_argument("--target_modules", default=",".join(VICTIM_TARGET_MODULES))
    p.add_argument("--lora_r", type=int, default=VICTIM_LORA_R)
    p.add_argument("--lora_alpha", type=int, default=VICTIM_LORA_ALPHA)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--epochs", type=int, default=VICTIM_EPOCHS)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--quant", choices=["none", "4bit"], default="none")
    p.add_argument("--lr", type=float, default=VICTIM_LR)
    p.add_argument("--max_seq_len", type=int, default=VICTIM_MAX_SEQ)
    p.add_argument("--skip_train", action="store_true")
    return p.parse_args()


def heldout_ppl(model, tokenizer, rows: list[dict]) -> float:
    texts = [row["text"] for row in rows]
    nll = sequence_nll(model, tokenizer, texts, batch_size=4 if torch.cuda.is_available() else 1)
    mean = sum(nll) / max(len(nll), 1)
    return float(torch.exp(torch.tensor(mean)).item())


def make_args(output_dir: Path, batch: int, accum: int, epochs: int, lr: float, seed: int) -> TrainingArguments:
    kwargs = dict(
        output_dir=str(output_dir / "trainer_state"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch,
        gradient_accumulation_steps=accum,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=20,
        save_strategy="epoch",
        save_total_limit=1,
        fp16=torch.cuda.is_available(),
        bf16=False,
        gradient_checkpointing=True,
        report_to=[],
        seed=seed,
        data_seed=seed,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        optim="adamw_torch",
        eval_strategy="no",
    )
    try:
        return TrainingArguments(**kwargs)
    except TypeError:
        kwargs.pop("eval_strategy")
        kwargs["evaluation_strategy"] = "no"
        return TrainingArguments(**kwargs)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_args()
    seed_everything(args.seed)
    corpus = read_jsonl(args.corpus)
    texts = [row["text"] for row in corpus]
    tokenizer = load_tokenizer(args.model)
    train_ds = CausalTextDataset(texts, tokenizer, args.max_seq_len)
    LOGGER.info("victim corpus rows=%s tokens=%s", len(train_ds), train_ds.token_count())

    if not args.skip_train:
        reset_peak_gpu()
        model = load_causal_lm(args.model, quant=args.quant)
        model = attach_lora(model, parse_modules(args.target_modules), args.lora_r, args.lora_alpha, quant=args.quant)
        batch = choose_train_batch_size(args.quant)
        accum = accumulation_steps(batch, EFFECTIVE_BATCH)
        t0 = time.time()
        trainer = Trainer(
            model=model,
            args=make_args(args.output_dir, batch, accum, args.epochs, args.lr, args.seed),
            train_dataset=train_ds,
            data_collator=PadCollator(tokenizer),
        )
        trainer.train()
        wall = time.time() - t0
        args.output_dir.mkdir(parents=True, exist_ok=True)
        trainer.model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        LOGGER.info("saved %s wall=%.1fs peak_gb=%.3f", args.output_dir, wall, peak_gpu_gb())
        model = trainer.model
    else:
        model = load_with_adapter(args.model, str(args.output_dir), quant=args.quant)

    heldout_path = args.heldout or (data_dir() / "partC_heldout.jsonl")
    if heldout_path.is_file():
        ppl = heldout_ppl(model, tokenizer, read_jsonl(heldout_path))
        meta = {"heldout_ppl": ppl, "corpus": str(args.corpus), "output_dir": str(args.output_dir)}
        write_json(args.output_dir / "heldout_ppl.json", meta)
        LOGGER.info("heldout ppl=%.4f", ppl)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
