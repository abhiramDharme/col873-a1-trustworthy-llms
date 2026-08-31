from __future__ import annotations

import torch


@torch.no_grad()
def generate_texts(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int,
    batch_size: int = 4,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    num_return_sequences: int = 1,
) -> list[str]:
    was_left = tokenizer.padding_side
    tokenizer.padding_side = "left"
    model.eval()
    device = next(model.parameters()).device
    outputs: list[str] = []
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id
    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start : start + batch_size]
        encoded = tokenizer(
            chunk,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            pad_token_id=pad,
            eos_token_id=eos,
            num_return_sequences=num_return_sequences,
        )
        if do_sample:
            gen_kwargs["temperature"] = max(temperature, 1e-5)
            if top_k and top_k > 0:
                gen_kwargs["top_k"] = top_k
            if top_p < 1.0:
                gen_kwargs["top_p"] = top_p
        sequences = model.generate(**encoded, **gen_kwargs)
        prompt_len = encoded["input_ids"].shape[1]
        for seq in sequences:
            text = tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)
            outputs.append(text.strip())
    tokenizer.padding_side = was_left
    return outputs


@torch.no_grad()
def sequence_nll(model, tokenizer, texts: list[str], batch_size: int = 8) -> list[float]:
    """Mean token NLL (natural log) per sequence, padding ignored."""
    model.eval()
    device = next(model.parameters()).device
    scores: list[float] = []
    was_left = tokenizer.padding_side
    tokenizer.padding_side = "right"
    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        encoded = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=1024)
        input_ids = encoded["input_ids"].to(device)
        attention = encoded["attention_mask"].to(device)
        labels = input_ids.clone()
        labels[attention == 0] = -100
        outputs = model(input_ids=input_ids, attention_mask=attention, labels=labels)
        logits = outputs.logits[:, :-1, :].float()
        target = labels[:, 1:]
        log_probs = torch.log_softmax(logits, dim=-1)
        nll = torch.gather(log_probs, 2, target.clamp(min=0).unsqueeze(-1)).squeeze(-1)
        mask = target.ne(-100)
        nll = -nll * mask
        denom = mask.sum(dim=1).clamp(min=1)
        mean = (nll.sum(dim=1) / denom).tolist()
        scores.extend(float(x) for x in mean)
    tokenizer.padding_side = was_left
    return scores


@torch.no_grad()
def continuation_nll(model, tokenizer, prefixes: list[str], completions: list[str], batch_size: int = 8) -> list[float]:
    """Sum NLL of completion tokens given prefix (not length-normalized)."""
    model.eval()
    device = next(model.parameters()).device
    scores: list[float] = []
    was_left = tokenizer.padding_side
    tokenizer.padding_side = "right"
    for start in range(0, len(prefixes), batch_size):
        pref = prefixes[start : start + batch_size]
        comp = completions[start : start + batch_size]
        full = [p + c for p, c in zip(pref, comp)]
        enc_full = tokenizer(full, return_tensors="pt", padding=True, truncation=True, max_length=1024)
        input_ids = enc_full["input_ids"].to(device)
        attention = enc_full["attention_mask"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention)
        log_probs = torch.log_softmax(outputs.logits[:, :-1, :].float(), dim=-1)
        target = input_ids[:, 1:]
        token_nll = -torch.gather(log_probs, 2, target.unsqueeze(-1)).squeeze(-1)
        for i, (p, c) in enumerate(zip(pref, comp)):
            p_len = len(tokenizer(p, add_special_tokens=False)["input_ids"])
            c_len = len(tokenizer(c, add_special_tokens=False)["input_ids"])
            # With right padding, real tokens occupy [:seq_len]
            seq_len = int(attention[i].sum().item())
            # completion tokens correspond to labels at positions [p_len, p_len+c_len)
            # after the shift, that's indices [p_len-1, p_len-1+c_len) if p_len>=1
            lo = max(p_len - 1, 0)
            hi = min(lo + c_len, seq_len - 1)
            if hi <= lo:
                scores.append(0.0)
            else:
                scores.append(float(token_nll[i, lo:hi].sum().item()))
    tokenizer.padding_side = was_left
    return scores
