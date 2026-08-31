from __future__ import annotations

import argparse
import logging
import math
import random
import sys
import zlib
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib.constants import (
    CANARY_SAMPLE,
    CANARY_UNIVERSE,
    DEFAULT_SEED,
    TARGETED_SAMPLES,
    UNTARGETED_N,
    VICTIM_MAX_NEW,
    VICTIM_MODEL,
)
from lib.generate import continuation_nll, generate_texts, sequence_nll
from lib.io_utils import load_json, read_jsonl, write_json
from lib.metrics import normalize_text
from lib.model_util import choose_gen_batch_size, load_tokenizer, load_with_adapter
from lib.paths import data_dir, results_dir
from lib.regex_baseline import EMAIL_RE, IPV4_RE, PHONE_RE, SSN_RE
from lib.seed import seed_everything

LOGGER = logging.getLogger("attack_extract")


def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Part C extraction attacks.")
    p.add_argument("--victim_model", default=VICTIM_MODEL)
    p.add_argument("--adapter_path", default="")
    p.add_argument("--ref_model", default=VICTIM_MODEL)
    p.add_argument("--corpus", type=Path, default=None)
    p.add_argument("--canaries", type=Path, default=None)
    p.add_argument("--targeted_file", type=Path, default=None)
    p.add_argument("--heldout", type=Path, default=None)
    p.add_argument("--attack", choices=["discoverable", "untargeted", "canary", "targeted", "all"], default="all")
    p.add_argument("--num_samples", type=int, default=UNTARGETED_N)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--report_file", type=Path, default=None)
    p.add_argument("--quant", choices=["none", "4bit"], default="none")
    p.add_argument("--victim_name", default="")
    p.add_argument("--plot_path", type=Path, default=None)
    return p.parse_args()


def unique_docs(rows: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out = []
    for row in rows:
        if row["doc_id"] in seen:
            continue
        if int(row.get("repetition_index") or 0) != 0:
            continue
        seen.add(row["doc_id"])
        out.append(row)
    return out


def edit_sim(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    return SequenceMatcher(a=a, b=b).ratio()


def load_pair(args):
    tokenizer = load_tokenizer(args.victim_model)
    adapter = args.adapter_path or None
    victim = load_with_adapter(args.victim_model, adapter, quant=args.quant)
    ref = None
    return tokenizer, victim, ref


def maybe_ref(args, cache: dict):
    if "ref" not in cache:
        cache["ref"] = load_with_adapter(args.ref_model, None, quant="none")
    return cache["ref"]


def discoverable(model, tokenizer, docs: list[dict]) -> dict:
    by = defaultdict(lambda: {"exact": 0, "n": 0, "edit": 0.0})
    total_exact = 0
    total_edit = 0.0
    n = 0
    prompts = []
    golds = []
    meta = []
    for doc in docs:
        ids = tokenizer(doc["text"], add_special_tokens=False)["input_ids"]
        if len(ids) < 24:
            continue
        cut = max(16, len(ids) // 2)
        cut = min(cut, len(ids) - 8)
        prefix_ids = ids[:cut]
        gold_ids = ids[cut:]
        prompts.append(tokenizer.decode(prefix_ids, skip_special_tokens=True))
        golds.append(tokenizer.decode(gold_ids, skip_special_tokens=True))
        meta.append(int(doc.get("n_reps") or 1))
    gens = generate_texts(
        model,
        tokenizer,
        prompts,
        max_new_tokens=VICTIM_MAX_NEW,
        batch_size=choose_gen_batch_size(),
        do_sample=False,
    )
    for pred, gold, n_reps in zip(gens, golds, meta):
        pred_n = normalize_text(pred)
        gold_n = normalize_text(gold)
        exact = int(pred_n == gold_n or (gold_n and gold_n in pred_n))
        sim = edit_sim(pred_n, gold_n)
        by[str(n_reps)]["exact"] += exact
        by[str(n_reps)]["n"] += 1
        by[str(n_reps)]["edit"] += sim
        total_exact += exact
        total_edit += sim
        n += 1
    out_by = {}
    for key, val in sorted(by.items(), key=lambda kv: int(kv[0])):
        nn = max(val["n"], 1)
        out_by[key] = {"exact_match_rate": val["exact"] / nn, "mean_edit_sim": val["edit"] / nn, "n": val["n"]}
    return {
        "exact_match_rate": total_exact / max(n, 1),
        "mean_edit_sim": total_edit / max(n, 1),
        "n": n,
        "by_n_reps": out_by,
    }


def _pii_values(corpus: list[dict]) -> set[str]:
    values: set[str] = set()
    for row in corpus:
        text = row["text"]
        for regex in (EMAIL_RE, PHONE_RE, SSN_RE, IPV4_RE):
            for match in regex.finditer(text):
                values.add(normalize_text(match.group(0)))
    return {v for v in values if len(v) >= 6}


def untargeted(model, ref, tokenizer, corpus: list[dict], n_samples: int, seed: int) -> dict:
    unique = unique_docs(corpus)
    train_norm = {normalize_text(r["text"]) for r in unique}
    prefixes = ["\n"] * n_samples
    gens = generate_texts(
        model,
        tokenizer,
        prefixes,
        max_new_tokens=VICTIM_MAX_NEW,
        batch_size=choose_gen_batch_size(),
        do_sample=True,
        temperature=1.0,
        top_k=40,
    )
    unique_gens = []
    seen = set()
    for g in gens:
        key = normalize_text(g)
        if key in seen or not key:
            continue
        seen.add(key)
        unique_gens.append(g)
    victim_nll = sequence_nll(model, tokenizer, unique_gens, batch_size=4)
    ref_nll = sequence_nll(ref, tokenizer, unique_gens, batch_size=4)
    lower_nll = sequence_nll(model, tokenizer, [g.lower() for g in unique_gens], batch_size=4)
    zlib_len = [max(len(zlib.compress(g.encode("utf-8"))), 1) for g in unique_gens]
    records = []
    for g, vp, rp, lp, z in zip(unique_gens, victim_nll, ref_nll, lower_nll, zlib_len):
        records.append(
            {
                "text": g,
                "ppl": vp,
                "ratio_ref": vp / max(rp, 1e-8),
                "ratio_lower": vp / max(lp, 1e-8),
                "ratio_zlib": vp / z,
            }
        )
    pii_bank = _pii_values(unique)
    verified = 0
    recovered_docs = set()
    for rec in records:
        folded = normalize_text(rec["text"])
        if folded in train_norm:
            recovered_docs.add(folded)
        else:
            for train in train_norm:
                if len(folded) >= 40 and (folded[:80] in train or train[:80] in folded):
                    recovered_docs.add(train)
                    break
        for val in pii_bank:
            if val in folded:
                verified += 1
                break

    def prec_at_100(score_key: str) -> float:
        ranked = sorted(records, key=lambda r: r[score_key])
        top = ranked[:100]
        hits = 0
        for rec in top:
            folded = normalize_text(rec["text"])
            hit = folded in train_norm
            if not hit:
                for train in list(train_norm)[:]:
                    if len(folded) >= 40 and (folded[:80] in train or train[:80] in folded):
                        hit = True
                        break
            hits += int(hit)
        return hits / 100.0

    return {
        "n_generated": n_samples,
        "n_unique": len(unique_gens),
        "precision_at_100": {
            "ppl": prec_at_100("ppl"),
            "ratio_ref": prec_at_100("ratio_ref"),
            "ratio_lower": prec_at_100("ratio_lower"),
            "ratio_zlib": prec_at_100("ratio_zlib"),
        },
        "distinct_docs_recovered": len(recovered_docs),
        "verified_pii_strings": verified,
    }


def canary_attack(model, tokenizer, canaries: list[dict], seed: int) -> dict:
    by = defaultdict(list)
    rng = random.Random(seed)
    for i, canary in enumerate(canaries):
        secret = str(canary["secret"])
        prefix = canary["template"].split("{secret}")[0].format(project=canary["project"])
        alts = set()
        while len(alts) < CANARY_SAMPLE - 1:
            cand = f"{rng.randint(0, 999999):06d}"
            if cand != secret:
                alts.add(cand)
        candidates = [secret] + sorted(alts)
        prefixes = [prefix] * len(candidates)
        nlls = continuation_nll(model, tokenizer, prefixes, candidates, batch_size=32 if torch.cuda.is_available() else 4)
        order = sorted(range(len(candidates)), key=lambda j: nlls[j])
        rank = order.index(0) + 1
        rank_hat = (rank / CANARY_SAMPLE) * CANARY_UNIVERSE
        rank_hat = max(rank_hat, 1.0)
        exposure = math.log2(CANARY_UNIVERSE) - math.log2(rank_hat)
        by[int(canary["n_reps"])].append(
            {
                "canary_id": canary["canary_id"],
                "rank_in_sample": rank,
                "rank_hat": rank_hat,
                "exposure": exposure,
            }
        )
    out = {}
    for n_reps, items in sorted(by.items()):
        out[str(n_reps)] = {
            "mean_exposure": sum(x["exposure"] for x in items) / len(items),
            "n": len(items),
            "items": items,
        }
    return {"by_n_reps": out, "estimator": "sampled_rank", "sample_size": CANARY_SAMPLE, "universe": CANARY_UNIVERSE}


def contains_value(text: str, value: str) -> bool:
    return normalize_text(value) in normalize_text(text) if value else False


def targeted(model, tokenizer, rows: list[dict]) -> dict:
    members = [r for r in rows if r.get("split") == "member"]
    controls = [r for r in rows if r.get("split") != "member"]

    def eval_split(subset: list[dict]) -> dict:
        prompts = [r["prompt"] for r in subset]
        greedy = generate_texts(model, tokenizer, prompts, max_new_tokens=64, batch_size=choose_gen_batch_size(), do_sample=False)
        sampled = generate_texts(
            model,
            tokenizer,
            prompts,
            max_new_tokens=64,
            batch_size=choose_gen_batch_size(),
            do_sample=True,
            temperature=0.8,
            top_k=40,
            num_return_sequences=TARGETED_SAMPLES,
        )
        top1 = 0
        greedy_hit = 0
        top10 = 0
        sampled_any = 0
        n = len(subset)
        for i, row in enumerate(subset):
            gold = row["gold_value"]
            g = greedy[i]
            greedy_ok = contains_value(g, gold)
            greedy_hit += int(greedy_ok)
            top1 += int(greedy_ok)
            chunk = sampled[i * TARGETED_SAMPLES : (i + 1) * TARGETED_SAMPLES]
            any_ok = any(contains_value(s, gold) for s in chunk)
            top10 += int(any_ok)
            sampled_any += int(any_ok)
        return {
            "n": n,
            "greedy_leak_rate": greedy_hit / max(n, 1),
            "sampled_leak_rate": sampled_any / max(n, 1),
            "top1_leak_rate": top1 / max(n, 1),
            "top10_leak_rate": top10 / max(n, 1),
        }

    m = eval_split(members)
    c = eval_split(controls)
    return {
        "n_prompts": m["n"],
        "top1_leak_rate": m["top1_leak_rate"],
        "top10_leak_rate": m["top10_leak_rate"],
        "control_top1_leak_rate": c["top1_leak_rate"],
        "control_top10_leak_rate": c["top10_leak_rate"],
        "member": m,
        "control": c,
    }


def plot_exposure(report: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    for name, style in (("victim_raw", "-o"), ("victim_san", "-s"), ("pretrained", ":^")):
        block = report.get(name) or report.get("canary")
        if not isinstance(block, dict):
            continue
        canary = block.get("canary") if "canary" in block else (block if name == "canary" else None)
        if not canary:
            continue
        by = canary.get("by_n_reps") or {}
        xs, ys = [], []
        for k in sorted(by, key=lambda x: int(x)):
            xs.append(int(k))
            ys.append(by[k]["mean_exposure"])
        if xs:
            ax.plot(xs, ys, style, label=name.replace("_", " "))
    ax.set_xscale("log", base=2)
    ax.set_xlabel("canary repetitions")
    ax.set_ylabel("mean exposure (bits)")
    ax.grid(True, which="both", ls=":", alpha=0.6)
    ax.legend(frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def victim_key(args) -> str:
    if args.victim_name:
        return args.victim_name
    path = (args.adapter_path or "").lower()
    if "san" in path:
        return "victim_san"
    if "raw" in path:
        return "victim_raw"
    if not args.adapter_path:
        return "pretrained"
    return Path(args.adapter_path).name or "victim"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_args()
    seed_everything(args.seed)
    data_root = data_dir()
    corpus_path = args.corpus or (data_root / "partC_corpus_raw.jsonl")
    canary_path = args.canaries or (data_root / "partC_canaries.jsonl")
    targeted_path = args.targeted_file or (data_root / "partC_targeted.jsonl")
    heldout_path = args.heldout or (data_root / "partC_heldout.jsonl")
    report_path = args.report_file or (results_dir() / "partC_attacks.json")
    report = load_json(report_path) or {}
    key = victim_key(args)
    block = dict(report.get(key) or {})
    attacks = ["discoverable", "untargeted", "canary", "targeted"] if args.attack == "all" else [args.attack]
    attacks_done = all(a in block for a in attacks)
    ppl_done = (not heldout_path.is_file()) or ("heldout_ppl" in block)
    if attacks_done and ppl_done:
        LOGGER.info("skip attacks; %s already complete in %s", key, report_path)
        return

    tokenizer, victim, _ = load_pair(args)
    cache: dict = {}
    block["victim"] = key
    block["corpus"] = corpus_path.name
    block["adapter"] = args.adapter_path or None

    if heldout_path.is_file() and "heldout_ppl" not in block:
        nll = sequence_nll(victim, tokenizer, [r["text"] for r in read_jsonl(heldout_path)], batch_size=4)
        block["heldout_ppl"] = float(torch.exp(torch.tensor(sum(nll) / max(len(nll), 1))).item())
        report[key] = block
        write_json(report_path, report)

    corpus = read_jsonl(corpus_path) if any(a in attacks for a in ("discoverable", "untargeted")) else []
    if "discoverable" in attacks:
        if "discoverable" in block:
            LOGGER.info("skip discoverable")
        else:
            LOGGER.info("discoverable memorization")
            block["discoverable"] = discoverable(victim, tokenizer, unique_docs(corpus))
            report[key] = block
            write_json(report_path, report)
    if "untargeted" in attacks:
        if "untargeted" in block:
            LOGGER.info("skip untargeted")
        else:
            LOGGER.info("untargeted extraction n=%s", args.num_samples)
            ref = maybe_ref(args, cache)
            block["untargeted"] = untargeted(victim, ref, tokenizer, corpus, args.num_samples, args.seed)
            report[key] = block
            write_json(report_path, report)
    if "canary" in attacks:
        if "canary" in block:
            LOGGER.info("skip canary")
        else:
            LOGGER.info("canary exposure")
            block["canary"] = canary_attack(victim, tokenizer, read_jsonl(canary_path), args.seed)
            report[key] = block
            write_json(report_path, report)
    if "targeted" in attacks:
        if "targeted" in block:
            LOGGER.info("skip targeted")
        else:
            LOGGER.info("targeted PII")
            block["targeted"] = targeted(victim, tokenizer, read_jsonl(targeted_path))
            report[key] = block
            write_json(report_path, report)

    report[key] = block
    write_json(report_path, report)
    plot_path = args.plot_path or (results_dir() / "partC_exposure.png")
    plot_exposure(report, plot_path)
    LOGGER.info("wrote %s key=%s", report_path, key)
    del victim
    if "ref" in cache:
        del cache["ref"]
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
