from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import torch
from transformers import Trainer

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
    amp_flags,
    adapter_is_complete,
    attach_lora,
    build_training_args,
    choose_gen_batch_size,
    choose_train_batch_size,
    latest_checkpoint,
    load_causal_lm,
    load_tokenizer,
    load_with_adapter,
    peak_gpu_gb,
    reset_peak_gpu,
    use_grad_checkpoint,
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
    nll = sequence_nll(model, tokenizer, texts, batch_size=choose_gen_batch_size())
    mean = sum(nll) / max(len(nll), 1)
    return float(torch.exp(torch.tensor(mean)).item())


def make_args(output_dir: Path, batch: int, accum: int, epochs: int, lr: float, seed: int) -> TrainingArguments:
    fp16, bf16 = amp_flags()
    workers = 4 if torch.cuda.is_available() else 0
    kwargs = dict(
        output_dir=str(output_dir / "trainer_state"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch,
        gradient_accumulation_steps=accum,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=20,
        save_strategy="steps",
        save_steps=100,
        load_best_model_at_end=False,
        fp16=fp16,
        bf16=bf16,
        gradient_checkpointing=use_grad_checkpoint(),
        report_to=[],
        seed=seed,
        data_seed=seed,
        remove_unused_columns=False,
        dataloader_num_workers=workers,
        dataloader_pin_memory=torch.cuda.is_available(),
        optim="adamw_torch",
        eval_strategy="no",
    )
    return build_training_args(kwargs)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_args()
    seed_everything(args.seed)
    corpus = read_jsonl(args.corpus)
    texts = [row["text"] for row in corpus]
    tokenizer = load_tokenizer(args.model)
    train_ds = CausalTextDataset(texts, tokenizer, args.max_seq_len)
    LOGGER.info("victim corpus rows=%s tokens=%s", len(train_ds), train_ds.token_count())

    skip_train = args.skip_train or adapter_is_complete(args.output_dir)
    resume_ckpt = latest_checkpoint(args.output_dir)

    if skip_train and adapter_is_complete(args.output_dir):
        LOGGER.info("skip train; adapter already at %s", args.output_dir)
        model = load_with_adapter(args.model, str(args.output_dir), quant=args.quant)
    else:
        reset_peak_gpu()
        model = load_causal_lm(args.model, quant=args.quant)
        model = attach_lora(model, parse_modules(args.target_modules), args.lora_r, args.lora_alpha, quant=args.quant)
        batch = choose_train_batch_size(args.quant)
        accum = accumulation_steps(batch, EFFECTIVE_BATCH)
        if resume_ckpt is not None:
            LOGGER.info("resume from %s", resume_ckpt)
        t0 = time.time()
        trainer = Trainer(
            model=model,
            args=make_args(args.output_dir, batch, accum, args.epochs, args.lr, args.seed),
            train_dataset=train_ds,
            data_collator=PadCollator(tokenizer),
        )
        trainer.train(resume_from_checkpoint=str(resume_ckpt) if resume_ckpt else None)
        wall = time.time() - t0
        args.output_dir.mkdir(parents=True, exist_ok=True)
        trainer.model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        LOGGER.info("saved %s wall=%.1fs peak_gb=%.3f", args.output_dir, wall, peak_gpu_gb())
        model = trainer.model

    heldout_path = args.heldout or (data_dir() / "partC_heldout.jsonl")
    ppl_path = args.output_dir / "heldout_ppl.json"
    if ppl_path.is_file():
        LOGGER.info("skip heldout ppl; already at %s", ppl_path)
    elif heldout_path.is_file():
        ppl = heldout_ppl(model, tokenizer, read_jsonl(heldout_path))
        meta = {"heldout_ppl": ppl, "corpus": str(args.corpus), "output_dir": str(args.output_dir)}
        write_json(args.output_dir / "heldout_ppl.json", meta)
        LOGGER.info("heldout ppl=%.4f", ppl)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
