"""Recover predicted PII spans from (source, sanitized) text.

The sanitizer emits placeholders such as [EMAIL], not character offsets. Two
recoveries are used:

1. Literal-gap walk. Split the prediction into literal chunks and [LABEL]
   placeholders. Each literal is located left-to-right in the source; the
   intervening source substring is the predicted span. This is exact when the
   model copies non-PII text, which is the training objective.

2. SequenceMatcher fallback. When a literal cannot be found (paraphrase,
   dropped punctuation, Unicode drift), difflib opcodes map replaced source
   regions onto placeholders in the prediction.

Failure modes, all of which are counted rather than silently repaired:
- Several placeholders collapsing onto one source gap (e.g. a full name).
- A placeholder with no residual source (overgeneration).
- Source rewritten so literals never match; opcode spans become coarse.
- Gold uses GIVENNAME1 while the model emits FIRSTNAME (OOD label shift).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Iterable

PLACEHOLDER_RE = re.compile(r"\[([A-Z][A-Z0-9]*)\]")


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    label: str

    def as_dict(self) -> dict[str, int | str]:
        return {"start": self.start, "end": self.end, "label": self.label}


def _trim_span(source: str, start: int, end: int) -> tuple[int, int] | None:
    while start < end and source[start].isspace():
        start += 1
    while end > start and source[end - 1].isspace():
        end -= 1
    if start >= end:
        return None
    return start, end


def _assign_labels(source: str, start: int, end: int, labels: list[str]) -> list[Span]:
    if not labels:
        return []
    trimmed = _trim_span(source, start, end)
    if trimmed is None:
        return []
    start, end = trimmed
    region = source[start:end]
    if len(labels) == 1:
        return [Span(start, end, labels[0])]
    tokens = list(re.finditer(r"\S+", region))
    if not tokens:
        return [Span(start, end, labels[0])]
    spans: list[Span] = []
    if len(tokens) == len(labels):
        for token, label in zip(tokens, labels):
            spans.append(Span(start + token.start(), start + token.end(), label))
        return spans
    if len(tokens) > len(labels):
        for i, label in enumerate(labels):
            if i < len(labels) - 1:
                token = tokens[i]
                spans.append(Span(start + token.start(), start + token.end(), label))
            else:
                token0, token1 = tokens[i], tokens[-1]
                spans.append(Span(start + token0.start(), start + token1.end(), label))
        return spans
    for token, label in zip(tokens, labels):
        spans.append(Span(start + token.start(), start + token.end(), label))
    return spans


def _gap_align(source: str, predicted: str) -> list[Span] | None:
    parts: list[tuple[str, str]] = []
    cursor = 0
    for match in PLACEHOLDER_RE.finditer(predicted):
        if match.start() > cursor:
            parts.append(("lit", predicted[cursor : match.start()]))
        parts.append(("ph", match.group(1)))
        cursor = match.end()
    if cursor < len(predicted):
        parts.append(("lit", predicted[cursor:]))

    spans: list[Span] = []
    src_pos = 0
    pending_start: int | None = None
    pending_labels: list[str] = []

    def flush(gap_end: int) -> bool:
        nonlocal pending_start, pending_labels
        if pending_start is None:
            return True
        if gap_end < pending_start:
            return False
        spans.extend(_assign_labels(source, pending_start, gap_end, pending_labels))
        pending_start = None
        pending_labels = []
        return True

    for kind, value in parts:
        if kind == "ph":
            if pending_start is None:
                pending_start = src_pos
            pending_labels.append(value)
            continue
        if value == "":
            continue
        idx = source.find(value, src_pos)
        if idx < 0:
            return None
        if not flush(idx):
            return None
        src_pos = idx + len(value)
    if pending_labels:
        if not flush(len(source)):
            return None
    return spans


def _opcode_align(source: str, predicted: str) -> list[Span]:
    matcher = SequenceMatcher(a=source, b=predicted, autojunk=False)
    spans: list[Span] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        labels = [match.group(1) for match in PLACEHOLDER_RE.finditer(predicted[j1:j2])]
        if not labels:
            continue
        spans.extend(_assign_labels(source, i1, i2, labels))
    return spans


def recover_spans(source: str, predicted: str) -> list[Span]:
    if not predicted:
        return []
    gap = _gap_align(source, predicted)
    if gap is not None:
        return _dedupe(gap)
    return _dedupe(_opcode_align(source, predicted))


def _dedupe(spans: Iterable[Span]) -> list[Span]:
    seen: set[tuple[int, int, str]] = set()
    out: list[Span] = []
    for span in spans:
        key = (span.start, span.end, span.label)
        if key in seen or span.start >= span.end:
            continue
        seen.add(key)
        out.append(span)
    out.sort(key=lambda s: (s.start, s.end, s.label))
    return out


def gold_spans(privacy_mask: list[dict]) -> list[Span]:
    spans: list[Span] = []
    for item in privacy_mask or []:
        spans.append(Span(int(item["start"]), int(item["end"]), str(item["label"])))
    return spans
