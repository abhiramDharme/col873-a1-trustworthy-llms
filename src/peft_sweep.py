from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from transformers import Trainer

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib.constants import (
    DEFAULT_SEED,
    EFFECTIVE_BATCH,
    PARTA_CSV_FIELDS,
    SANITIZER_EPOCHS,
    SANITIZER_LR,
    SANITIZER_MAX_NEW,
    SANITIZER_MAX_SEQ,
    SANITIZER_MODEL,
    SANITIZER_WARMUP,
    PII_LABELS_300K,
)
from lib.generate import generate_texts
from lib.io_utils import csv_has_complete_row, read_jsonl, upsert_csv_row, verify_manifest
from lib.metrics import score_example
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
    peak_gpu_gb,
    reset_peak_gpu,
    trainable_counts,
    use_grad_checkpoint,
)
from lib.paths import data_dir, results_dir
from lib.prompts import apply_prompt
from lib.seed import seed_everything
from lib.sft_data import CompletionSFTDataset, PadCollator

LOGGER = logging.getLogger("peft_sweep")


def parse_modules(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Part A LoRA/QLoRA sweep runner (one configuration).")
    p.add_argument("--model", default=SANITIZER_MODEL)
    p.add_argument("--train_file", type=Path, default=None)
    p.add_argument("--val_file", type=Path, default=None)
    p.add_argument("--target_modules", default="")
    p.add_argument("--lora_r", type=int, default=0)
    p.add_argument("--lora_alpha", type=int, default=0)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--quant", choices=["none", "4bit"], default="none")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--run_id", default="")
    p.add_argument("--output_dir", type=Path, default=None)
    p.add_argument("--results_csv", type=Path, default=None)
    p.add_argument("--epochs", type=int, default=SANITIZER_EPOCHS)
    p.add_argument("--lr", type=float, default=SANITIZER_LR)
    p.add_argument("--max_seq_len", type=int, default=SANITIZER_MAX_SEQ)
    p.add_argument("--skip_train", action="store_true")
    p.add_argument("--plot_only", action="store_true")
    p.add_argument("--fig_path", type=Path, default=None)
    return p.parse_args()


def make_training_args(output_dir: Path, batch_size: int, accum: int, epochs: int, lr: float, seed: int) -> TrainingArguments:
    fp16, bf16 = amp_flags()
    workers = 4 if torch.cuda.is_available() else 0
    kwargs = dict(
        output_dir=str(output_dir / "trainer_state"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=accum,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=SANITIZER_WARMUP,
        logging_steps=25,
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
        max_grad_norm=1.0,
        optim="adamw_torch",
        eval_strategy="epoch",
    )
    return build_training_args(kwargs)


def evaluate_val_entities(model, tokenizer, rows: list[dict], batch_size: int) -> tuple[float, float]:
    prompts = [apply_prompt(tokenizer, row["source_text"], add_generation_prompt=True) for row in rows]
    texts = generate_texts(model, tokenizer, prompts, max_new_tokens=SANITIZER_MAX_NEW, batch_size=batch_size)
    tp = fp = fn = 0
    leak_n = 0.0
    for row, pred in zip(rows, texts):
        scored = score_example(row["source_text"], pred, row.get("privacy_mask") or [], PII_LABELS_300K)
        tp += scored["exact_tp"]
        fp += scored["exact_fp"]
        fn += scored["exact_fn"]
        leak_n += scored["leak_rate"]
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    leak = leak_n / len(rows) if rows else 0.0
    return f1, leak


def plot_sweep(csv_path: Path, fig_path: Path) -> None:
    import csv

    with csv_path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        LOGGER.warning("No sweep rows to plot")
        return
    xs = [max(float(r["trainable_pct"]), 1e-6) for r in rows]
    ys = [float(r["val_entity_f1"]) for r in rows]
    mem = [float(r["peak_gpu_mem_gb"]) for r in rows]
    ids = [r["run_id"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    axes[0].scatter(xs, ys, c="#1f4e79", s=42, zorder=3)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("trainable parameters (%)")
    axes[0].set_ylabel("val entity F1 (exact)")
    axes[0].grid(True, which="both", ls=":", alpha=0.6)
    for x, y, label in zip(xs, ys, ids):
        axes[0].annotate(label, (x, y), textcoords="offset points", xytext=(5, 4), fontsize=8)
    axes[1].scatter(ys, mem, c="#9c2f2f", s=42, zorder=3)
    for y, m, label in zip(ys, mem, ids):
        axes[1].annotate(label, (y, m), textcoords="offset points", xytext=(5, 4), fontsize=8)
    axes[1].set_xlabel("val entity F1 (exact)")
    axes[1].set_ylabel("peak GPU memory (GB)")
    axes[1].grid(True, ls=":", alpha=0.6)
    fig.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=160)
    fig.savefig(fig_path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_args()
    data_root = data_dir()
    verify_manifest(data_root)
    results_csv = args.results_csv or (results_dir() / "partA_sweep.csv")
    fig_path = args.fig_path or (results_dir() / "partA_fig.png")
    if args.plot_only:
        plot_sweep(results_csv, fig_path)
        return
    if not args.target_modules or not args.run_id or args.output_dir is None or args.lora_r <= 0:
        raise SystemExit("training requires --target_modules --lora_r --lora_alpha --run_id --output_dir")

    seed_everything(args.seed)
    train_file = args.train_file or (data_root / "partA_train.jsonl")
    val_file = args.val_file or (data_root / "partA_validation.jsonl")
    train_rows = read_jsonl(train_file)
    val_rows = read_jsonl(val_file)
    modules = parse_modules(args.target_modules)
    method = "QLoRA" if args.quant == "4bit" else "LoRA"

    tokenizer = load_tokenizer(args.model)
    train_ds = CompletionSFTDataset(train_rows, tokenizer, args.max_seq_len)
    val_ds = CompletionSFTDataset(val_rows, tokenizer, args.max_seq_len)
    LOGGER.info("tokenized train=%s val=%s truncated=%s", len(train_ds), len(val_ds), train_ds.n_truncated)

    adapter_ok = adapter_is_complete(args.output_dir)
    resume_ckpt = latest_checkpoint(args.output_dir)
    skip_train = args.skip_train or adapter_ok
    if args.skip_train and not adapter_ok and resume_ckpt is None:
        raise FileNotFoundError(args.output_dir)

    if skip_train and adapter_ok:
        from lib.model_util import load_with_adapter

        LOGGER.info("skip train; adapter already at %s", args.output_dir)
        model = load_with_adapter(args.model, str(args.output_dir), quant=args.quant)
        trainable, total, pct = trainable_counts(model)
        wall = tps = final_loss = val_loss = peak = 0.0
    else:
        reset_peak_gpu()
        model = load_causal_lm(args.model, quant=args.quant)
        model = attach_lora(model, modules, args.lora_r, args.lora_alpha, quant=args.quant)
        trainable, total, pct = trainable_counts(model)
        batch = choose_train_batch_size(args.quant)
        accum = accumulation_steps(batch, EFFECTIVE_BATCH)
        LOGGER.info("batch=%s accum=%s effective=%s", batch, accum, batch * accum)
        if resume_ckpt is not None:
            LOGGER.info("resume from %s", resume_ckpt)
        t0 = time.time()
        training_args = make_training_args(
            args.output_dir,
            batch,
            accum,
            args.epochs,
            args.lr,
            args.seed,
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=PadCollator(tokenizer),
        )
        train_out = trainer.train(resume_from_checkpoint=str(resume_ckpt) if resume_ckpt else None)
        wall = time.time() - t0
        eval_out = trainer.evaluate()
        args.output_dir.mkdir(parents=True, exist_ok=True)
        trainer.model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        peak = peak_gpu_gb()
        tokens = train_ds.token_count() * args.epochs
        tps = tokens / wall if wall else 0.0
        final_loss = float(train_out.metrics.get("train_loss", 0.0))
        val_loss = float(eval_out.get("eval_loss", 0.0))
        model = trainer.model

    if csv_has_complete_row(results_csv, args.run_id):
        LOGGER.info("skip val decode; %s already in %s", args.run_id, results_csv)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return

    gen_bs = choose_gen_batch_size()
    val_f1, val_leak = evaluate_val_entities(model, tokenizer, val_rows, gen_bs)
    row = {
        "run_id": args.run_id,
        "method": method,
        "target_modules": ",".join(modules),
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "quant": args.quant,
        "trainable_params": trainable,
        "total_params": total,
        "trainable_pct": f"{pct:.6f}",
        "peak_gpu_mem_gb": f"{peak:.4f}",
        "train_wallclock_s": f"{wall:.2f}",
        "tokens_per_s": f"{tps:.2f}",
        "final_train_loss": f"{final_loss:.6f}",
        "val_loss": f"{val_loss:.6f}",
        "val_entity_f1": f"{val_f1:.6f}",
        "val_leak_rate": f"{val_leak:.6f}",
    }
    upsert_csv_row(results_csv, PARTA_CSV_FIELDS, row)
    LOGGER.info("wrote %s row %s f1=%.4f leak=%.4f", results_csv, args.run_id, val_f1, val_leak)
    if results_csv.is_file():
        plot_sweep(results_csv, fig_path)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
