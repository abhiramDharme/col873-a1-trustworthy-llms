from __future__ import annotations

from .constants import PII_LABELS_300K

SYSTEM_PROMPT = (
    "You are a PII sanitizer. Copy the user's text with every private value replaced "
    "by a placeholder of the form [LABEL]. Keep all non-private characters, whitespace, "
    "and punctuation exactly. Use only these labels: "
    + ", ".join(PII_LABELS_300K)
    + ". Output the sanitized text and nothing else."
)

USER_PREFIX = "Sanitize the following text.\n\n"


def user_message(source_text: str) -> str:
    return USER_PREFIX + source_text


def sanitizer_messages(source_text: str, target_text: str | None = None) -> list[dict[str, str]]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message(source_text)},
    ]
    if target_text is not None:
        messages.append({"role": "assistant", "content": target_text})
    return messages


def apply_prompt(tokenizer, source_text: str, target_text: str | None = None, add_generation_prompt: bool | None = None) -> str:
    messages = sanitizer_messages(source_text, target_text)
    if add_generation_prompt is None:
        add_generation_prompt = target_text is None
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )
