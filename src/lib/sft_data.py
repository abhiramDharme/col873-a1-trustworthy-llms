from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Dataset

from .prompts import apply_prompt


class CompletionSFTDataset(Dataset):
    """Tokenizes instruction examples and masks prompt tokens in the labels."""

    def __init__(self, rows: list[dict[str, Any]], tokenizer, max_seq_len: int):
        self.examples: list[dict[str, list[int]]] = []
        n_trunc = 0
        for row in rows:
            prompt = apply_prompt(tokenizer, row["source_text"], target_text=None, add_generation_prompt=True)
            full = apply_prompt(tokenizer, row["source_text"], target_text=row["target_text"], add_generation_prompt=False)
            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            full_ids = tokenizer(full, add_special_tokens=False)["input_ids"]
            if full_ids[: len(prompt_ids)] != prompt_ids:
                # Chat template is not a strict prefix; fall back to masking a suffix window.
                prompt_ids = full_ids[: max(1, len(full_ids) // 2)]
            labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
            input_ids = full_ids
            if len(input_ids) > max_seq_len:
                input_ids = input_ids[:max_seq_len]
                labels = labels[:max_seq_len]
                n_trunc += 1
            if all(x == -100 for x in labels):
                labels[-1] = input_ids[-1]
            self.examples.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": [1] * len(input_ids),
                    "labels": labels,
                }
            )
        self.n_truncated = n_trunc

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        return self.examples[idx]

    def token_count(self) -> int:
        return sum(len(ex["input_ids"]) for ex in self.examples)


class CausalTextDataset(Dataset):
    def __init__(self, texts: list[str], tokenizer, max_seq_len: int):
        self.examples: list[dict[str, list[int]]] = []
        for text in texts:
            encoded = tokenizer(
                text,
                add_special_tokens=True,
                truncation=True,
                max_length=max_seq_len,
            )
            ids = encoded["input_ids"]
            self.examples.append(
                {
                    "input_ids": ids,
                    "attention_mask": [1] * len(ids),
                    "labels": list(ids),
                }
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        return self.examples[idx]

    def token_count(self) -> int:
        return sum(len(ex["input_ids"]) for ex in self.examples)


class PadCollator:
    def __init__(self, tokenizer):
        self.pad_id = tokenizer.pad_token_id

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for feat in features:
            pad = max_len - len(feat["input_ids"])
            batch["input_ids"].append(feat["input_ids"] + [self.pad_id] * pad)
            batch["attention_mask"].append(feat["attention_mask"] + [0] * pad)
            batch["labels"].append(feat["labels"] + [-100] * pad)
        return {key: torch.tensor(val, dtype=torch.long) for key, val in batch.items()}
