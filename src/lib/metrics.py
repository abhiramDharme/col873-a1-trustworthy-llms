from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Iterable

from .align import Span, recover_spans

_WS = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    return _WS.sub(" ", text).strip().lower()


def _index_by_label(spans: Iterable[Span]) -> dict[str, list[Span]]:
    grouped: dict[str, list[Span]] = defaultdict(list)
    for span in spans:
        grouped[span.label].append(span)
    return grouped


def _match_counts(pred: list[Span], gold: list[Span], partial: bool, iou_threshold: float = 0.5) -> tuple[int, int, int]:
    used: set[int] = set()
    tp = 0
    for p in pred:
        for i, g in enumerate(gold):
            if i in used or p.label != g.label:
                continue
            if partial:
                inter = max(0, min(p.end, g.end) - max(p.start, g.start))
                union = (p.end - p.start) + (g.end - g.start) - inter
                score = inter / union if union else 0.0
                hit = score >= iou_threshold
            else:
                hit = p.start == g.start and p.end == g.end
            if hit:
                used.add(i)
                tp += 1
                break
    fp = len(pred) - tp
    fn = len(gold) - tp
    return tp, fp, fn


def _prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"p": prec, "r": rec, "f1": f1}


def entity_scores(pred_spans: list[Span], gold_spans: list[Span], partial: bool = False) -> dict:
    labels = sorted({s.label for s in pred_spans} | {s.label for s in gold_spans})
    per_class: dict[str, dict] = {}
    sum_f1 = 0.0
    n_macro = 0
    tp = fp = fn = 0
    gold_by = _index_by_label(gold_spans)
    pred_by = _index_by_label(pred_spans)
    for label in labels:
        c_tp, c_fp, c_fn = _match_counts(pred_by.get(label, []), gold_by.get(label, []), partial)
        stats = _prf(c_tp, c_fp, c_fn)
        support = len(gold_by.get(label, []))
        per_class[label] = {**stats, "support": support}
        tp += c_tp
        fp += c_fp
        fn += c_fn
        if support > 0:
            sum_f1 += stats["f1"]
            n_macro += 1
    micro = _prf(tp, fp, fn)
    return {
        "micro_p": micro["p"],
        "micro_r": micro["r"],
        "micro_f1": micro["f1"],
        "macro_f1": (sum_f1 / n_macro) if n_macro else 0.0,
        "per_class": per_class,
    }


def leak_rate(predicted_text: str, gold_mask: list[dict]) -> tuple[float, list[str]]:
    folded = normalize_text(predicted_text)
    leaked: list[str] = []
    n = 0
    for item in gold_mask or []:
        value = str(item.get("value", ""))
        if not normalize_text(value):
            continue
        n += 1
        if normalize_text(value) in folded:
            leaked.append(value)
    rate = (len(leaked) / n) if n else 0.0
    return rate, leaked


def over_masking_rate(source: str, predicted: str, gold_mask: list[dict]) -> float:
    from difflib import SequenceMatcher

    pii = [False] * len(source)
    for item in gold_mask or []:
        for i in range(int(item["start"]), int(item["end"])):
            if 0 <= i < len(source):
                pii[i] = True
    preserved = [False] * len(source)
    matcher = SequenceMatcher(a=source, b=predicted, autojunk=False)
    for tag, i1, i2, _j1, _j2 in matcher.get_opcodes():
        if tag == "equal":
            for i in range(i1, i2):
                preserved[i] = True
    non_pii = [i for i, flag in enumerate(pii) if not flag]
    if not non_pii:
        return 0.0
    over = sum(1 for i in non_pii if not preserved[i])
    return over / len(non_pii)


def well_formed_rate(predicted_text: str, allowed_labels: Iterable[str] | None = None) -> bool:
    from .align import PLACEHOLDER_RE

    if "[" in predicted_text and "]" not in predicted_text:
        return False
    labels = PLACEHOLDER_RE.findall(predicted_text)
    if allowed_labels is None:
        return True
    allowed = set(allowed_labels)
    return all(label in allowed for label in labels)


def score_example(source: str, predicted: str, gold_mask: list[dict], allowed_labels: Iterable[str] | None = None) -> dict:
    pred = recover_spans(source, predicted)
    gold = [Span(int(m["start"]), int(m["end"]), str(m["label"])) for m in (gold_mask or [])]
    leak, leaked_values = leak_rate(predicted, gold_mask)
    exact_counts = counts_from_scores(pred, gold, partial=False)
    overlap_counts = counts_from_scores(pred, gold, partial=True)
    exact_tp, exact_fp, exact_fn = totals(exact_counts)
    overlap_tp, overlap_fp, overlap_fn = totals(overlap_counts)
    return {
        "pred_spans": pred,
        "gold_spans": gold,
        "exact": entity_scores(pred, gold, partial=False),
        "overlap": entity_scores(pred, gold, partial=True),
        "exact_counts": exact_counts,
        "overlap_counts": overlap_counts,
        "exact_tp": exact_tp,
        "exact_fp": exact_fp,
        "exact_fn": exact_fn,
        "overlap_tp": overlap_tp,
        "overlap_fp": overlap_fp,
        "overlap_fn": overlap_fn,
        "leak_rate": leak,
        "leaked_values": leaked_values,
        "over_masking_rate": over_masking_rate(source, predicted, gold_mask),
        "well_formed": well_formed_rate(predicted, allowed_labels),
    }


def aggregate_reports(rows: list[dict], top_k_classes: int = 10) -> dict:
    def _sum_prf(key: str) -> dict[str, float]:
        tp = fp = fn = 0
        class_tp: dict[str, int] = defaultdict(int)
        class_fp: dict[str, int] = defaultdict(int)
        class_fn: dict[str, int] = defaultdict(int)
        support: Counter[str] = Counter()
        for row in rows:
            block = row[key]
            per = block["per_class"]
            # Reconstruct tp/fp/fn from p/r/support is unstable. Store raw counts instead.
            for label, stats in per.items():
                support[label] += stats["support"]
                # We keep running micro via stored micro counts on the row.
            tp += row[f"{key}_tp"]
            fp += row[f"{key}_fp"]
            fn += row[f"{key}_fn"]
            for label, stats in row[f"{key}_counts"].items():
                class_tp[label] += stats["tp"]
                class_fp[label] += stats["fp"]
                class_fn[label] += stats["fn"]
        micro = _prf(tp, fp, fn)
        macro_labels = [label for label, n in support.items() if n > 0]
        macro = (
            sum(_prf(class_tp[l], class_fp[l], class_fn[l])["f1"] for l in macro_labels) / len(macro_labels)
            if macro_labels
            else 0.0
        )
        per_class = {
            label: {**_prf(class_tp[label], class_fp[label], class_fn[label]), "support": support[label]}
            for label in support
        }
        top = [label for label, _ in support.most_common(top_k_classes)]
        return {
            "micro_p": micro["p"],
            "micro_r": micro["r"],
            "micro_f1": micro["f1"],
            "macro_f1": macro,
            "per_class": {label: per_class[label] for label in top},
            "per_class_full": per_class,
        }

    n = len(rows)
    leak = sum(r["leak_rate"] for r in rows) / n if n else 0.0
    over = sum(r["over_masking_rate"] for r in rows) / n if n else 0.0
    well = sum(1.0 for r in rows if r["well_formed"]) / n if n else 0.0
    exact = _sum_prf("exact")
    overlap = _sum_prf("overlap")
    return {
        "n": n,
        "exact": exact,
        "overlap_iou50": overlap,
        "per_class": exact["per_class"],
        "leak_rate": leak,
        "over_masking_rate": over,
        "well_formed_rate": well,
    }


def counts_from_scores(pred: list[Span], gold: list[Span], partial: bool) -> dict[str, dict[str, int]]:
    gold_by = _index_by_label(gold)
    pred_by = _index_by_label(pred)
    labels = set(gold_by) | set(pred_by)
    out: dict[str, dict[str, int]] = {}
    for label in labels:
        tp, fp, fn = _match_counts(pred_by.get(label, []), gold_by.get(label, []), partial)
        out[label] = {"tp": tp, "fp": fp, "fn": fn}
    return out


def totals(count_map: dict[str, dict[str, int]]) -> tuple[int, int, int]:
    tp = sum(v["tp"] for v in count_map.values())
    fp = sum(v["fp"] for v in count_map.values())
    fn = sum(v["fn"] for v in count_map.values())
    return tp, fp, fn
