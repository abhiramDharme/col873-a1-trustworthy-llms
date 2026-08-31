from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib.align import Span
from lib.constants import DEFAULT_SEED, PII_LABELS_300K, SANITIZER_MAX_NEW, SANITIZER_MODEL
from lib.generate import generate_texts
from lib.io_utils import iter_jsonl, read_jsonl, write_json, write_jsonl
from lib.metrics import aggregate_reports, score_example
from lib.model_util import load_tokenizer, load_with_adapter, choose_gen_batch_size
from lib.paths import data_dir, results_dir
from lib.prompts import apply_prompt
from lib.regex_baseline import RegexSanitizer
from lib.seed import seed_everything

LOGGER = logging.getLogger("sanitize_eval")


def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Part B sanitizer evaluation and corpus sanitization.")
    p.add_argument("--base_model", default=SANITIZER_MODEL)
    p.add_argument("--adapter_path", default="")
    p.add_argument("--test_file", type=Path, required=True)
    p.add_argument("--system", choices=["finetuned", "regex", "presidio", "zeroshot"], required=True)
    p.add_argument("--output_predictions", type=Path, default=None)
    p.add_argument("--report_file", type=Path, default=None)
    p.add_argument("--quant", choices=["none", "4bit"], default="none")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--sanitize_corpus_out", type=Path, default=None)
    p.add_argument("--batch_size", type=int, default=0)
    return p.parse_args()


def _is_ood(path: Path) -> bool:
    return "ood" in path.name.lower()


def _allowed_labels(rows: list[dict]) -> list[str] | None:
    labels = {item["label"] for row in rows for item in (row.get("privacy_mask") or [])}
    return sorted(labels) if labels else list(PII_LABELS_300K)


def run_finetuned(args, rows: list[dict], sources: list[str]) -> list[str]:
    tokenizer = load_tokenizer(args.base_model)
    adapter = args.adapter_path if args.system == "finetuned" and args.adapter_path else None
    model = load_with_adapter(args.base_model, adapter, quant=args.quant)
    prompts = [apply_prompt(tokenizer, text, add_generation_prompt=True) for text in sources]
    bs = args.batch_size or choose_gen_batch_size()
    texts = generate_texts(model, tokenizer, prompts, max_new_tokens=SANITIZER_MAX_NEW, batch_size=bs)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return texts


def run_rules(system: str, sources: list[str], ood: bool) -> tuple[list[str], list[list[Span]]]:
    if system == "regex":
        engine = RegexSanitizer(ood=ood)
    elif system == "presidio":
        from lib.presidio_baseline import PresidioSanitizer

        engine = PresidioSanitizer(ood=ood)
    else:
        raise ValueError(system)
    texts: list[str] = []
    spans: list[list[Span]] = []
    for text in sources:
        pred, sp = engine.sanitize(text)
        texts.append(pred)
        spans.append(sp)
    return texts, spans


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_args()
    seed_everything(args.seed)
    test_file = args.test_file
    rows = read_jsonl(test_file)
    ood = _is_ood(test_file)
    corpus_mode = args.sanitize_corpus_out is not None or ("text" in rows[0] and "source_text" not in rows[0])

    if corpus_mode and args.sanitize_corpus_out:
        sources = [row["text"] for row in rows]
    else:
        sources = [row["source_text"] for row in rows]

    LOGGER.info("system=%s n=%s ood=%s corpus=%s", args.system, len(rows), ood, corpus_mode)

    pred_path = args.output_predictions or (results_dir() / "partB_predictions.jsonl")
    report_path = args.report_file or (results_dir() / "partB_metrics.json")
    if args.sanitize_corpus_out and args.sanitize_corpus_out.is_file():
        LOGGER.info("skip corpus sanitize; %s exists", args.sanitize_corpus_out)
        return
    if not corpus_mode and pred_path.is_file() and report_path.is_file():
        LOGGER.info("skip eval; %s and %s exist", pred_path, report_path)
        return

    rule_spans: list[list[Span]] | None = None
    t0 = time.time()
    if args.system in {"regex", "presidio"}:
        preds, rule_spans = run_rules(args.system, sources, ood=ood)
    else:
        preds = run_finetuned(args, rows, sources)
    elapsed = time.time() - t0
    per_ms = (elapsed / max(len(rows), 1)) * 1000.0

    if args.sanitize_corpus_out:
        out_rows = []
        cache: dict[str, str] = {}
        for row, pred in zip(rows, preds):
            cache[row["text"]] = pred
            new_row = dict(row)
            new_row["text"] = pred
            out_rows.append(new_row)
        write_jsonl(args.sanitize_corpus_out, out_rows)
        LOGGER.info("wrote sanitized corpus %s", args.sanitize_corpus_out)
        return

    allowed = _allowed_labels(rows)
    pred_rows = []
    scored_rows = []
    for i, (row, pred) in enumerate(zip(rows, preds)):
        scored = score_example(row["source_text"], pred, row.get("privacy_mask") or [], allowed)
        if rule_spans is not None:
            scored["pred_spans"] = rule_spans[i]
            # Recompute with engine spans (regex/presidio already have offsets).
            from lib.metrics import entity_scores, counts_from_scores, totals, leak_rate, over_masking_rate, well_formed_rate
            from lib.align import Span as Sp

            gold = scored["gold_spans"]
            pred_s = rule_spans[i]
            scored["exact"] = entity_scores(pred_s, gold, partial=False)
            scored["overlap"] = entity_scores(pred_s, gold, partial=True)
            scored["exact_counts"] = counts_from_scores(pred_s, gold, False)
            scored["overlap_counts"] = counts_from_scores(pred_s, gold, True)
            scored["exact_tp"], scored["exact_fp"], scored["exact_fn"] = totals(scored["exact_counts"])
            scored["overlap_tp"], scored["overlap_fp"], scored["overlap_fn"] = totals(scored["overlap_counts"])
            leak, leaked = leak_rate(pred, row.get("privacy_mask") or [])
            scored["leak_rate"] = leak
            scored["leaked_values"] = leaked
            scored["over_masking_rate"] = over_masking_rate(row["source_text"], pred, row.get("privacy_mask") or [])
            scored["well_formed"] = well_formed_rate(pred, allowed)
            scored["pred_spans"] = pred_s
        scored_rows.append(scored)
        pred_rows.append(
            {
                "id": row.get("id"),
                "language": row.get("language"),
                "source_text": row["source_text"],
                "predicted_text": pred,
                "predicted_spans": [s.as_dict() for s in scored["pred_spans"]],
                "gold_spans": [
                    {"start": int(m["start"]), "end": int(m["end"]), "label": m["label"]}
                    for m in (row.get("privacy_mask") or [])
                ],
                "leaked_values": scored["leaked_values"],
                "latency_ms": round(per_ms, 3),
            }
        )

    summary = aggregate_reports(scored_rows)
    summary.update(
        {
            "system": args.system,
            "adapter": args.adapter_path or None,
            "test_set": test_file.name,
            "n": len(rows),
        }
    )
    pred_path = args.output_predictions or (results_dir() / "partB_predictions.jsonl")
    report_path = args.report_file or (results_dir() / "partB_metrics.json")
    write_jsonl(pred_path, pred_rows)
    write_json(report_path, summary)
    LOGGER.info(
        "%s exact_f1=%.4f leak=%.4f over=%.4f well=%.4f",
        args.system,
        summary["exact"]["micro_f1"],
        summary["leak_rate"],
        summary["over_masking_rate"],
        summary["well_formed_rate"],
    )


if __name__ == "__main__":
    main()
