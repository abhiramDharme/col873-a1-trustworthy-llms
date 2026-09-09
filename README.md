# COL873 Assignment 1 — Fine-Tune, Sanitize, Extract

Qwen2.5-1.5B-Instruct is adapted with LoRA/QLoRA as a PII sanitizer. The selected adapter is then scored as a sanitization system (span recovery, leak rate, over-masking) against regex and Presidio, and used to build a sanitized victim corpus. Two Qwen2.5-0.5B LoRA victims (raw vs sanitized) plus the untuned base model are attacked with discoverable memorization, untargeted membership ranking, canary exposure, and targeted PII prompting.

The four CLI entry points required by the handout live in `src/`. Shared code is in `src/lib/`. Do not train on `partC_canaries.jsonl`. Do not publish adapters or extraction dumps.

## Data

Use the frozen JSONL under `data/` (or `trustworthy_llms_a1_bundle/data/`). `run_all.sh` verifies `data/manifest.jsonl` before any training. Do not reshuffle or subsample.

## Where this runs

| Machine | Verdict |
|---|---|
| MacBook Air M3, 8 GB | Fits tokenizer/metrics/regex only. 1.5B LoRA training will page to death. bitsandbytes QLoRA is CUDA-only. |
| Kaggle GPU (Tesla T4, 16 GB) | Intended run. Enable GPU + Internet. One 12 h session is tight but feasible with LoRA rank as specified, `fp16`, gradient checkpointing, and effective batch 16. |
| IITD HPC `scai_q` A100 80 GB | Machine used for the reported numbers. `hpc/submit.pbs` requests project `vision.cse.spons`, job name `default`. |

Hyperparameters are identical across machines (`src/lib/constants.py`). Only batch size and gradient accumulation change with GPU memory.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

On Kaggle, do not reinstall `torch`. `kaggle/kernel.py` installs the remaining packages on top of the image.

## Commands

```bash
bash run_all.sh
```

`STAGE=A|B|C` runs one part. `SKIP_EXISTING=1` (default) reuses adapters that already have `adapter_config.json`.

The TAs invoke:

```bash
python src/peft_sweep.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --train_file data/partA_train.jsonl --val_file data/partA_validation.jsonl \
  --target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --lora_r 32 --lora_alpha 64 --lora_dropout 0.05 \
  --quant none --seed 42 \
  --run_id A5 --output_dir adapters/A5 --results_csv results/partA_sweep.csv

python src/sanitize_eval.py \
  --base_model Qwen/Qwen2.5-1.5B-Instruct \
  --adapter_path adapters/best_sanitizer \
  --test_file data/partB_test.jsonl \
  --system finetuned \
  --output_predictions results/partB_predictions.jsonl \
  --report_file results/partB_metrics.json

python src/train_victim.py \
  --model Qwen/Qwen2.5-0.5B --corpus data/partC_corpus_raw.jsonl \
  --heldout data/partC_heldout.jsonl \
  --target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --lora_r 32 --lora_alpha 64 --epochs 4 --seed 42 \
  --output_dir adapters/victim_raw

python src/attack_extract.py \
  --victim_model Qwen/Qwen2.5-0.5B --adapter_path adapters/victim_raw \
  --ref_model Qwen/Qwen2.5-0.5B \
  --corpus data/partC_corpus_raw.jsonl --canaries data/partC_canaries.jsonl \
  --targeted_file data/partC_targeted.jsonl \
  --attack all --num_samples 5000 --seed 42 \
  --report_file results/partC_attacks.json
```

`--system` also accepts `regex`, `presidio`, and `zeroshot`.

## Method notes

- Sanitizer loss is completion-only: prompt tokens are `-100` in the labels.
- Predicted spans are recovered by a literal-gap walk against `[LABEL]` placeholders, with `difflib.SequenceMatcher` as fallback. See `src/lib/align.py`.
- Best Part A adapter is the lowest validation leak rate, then highest exact entity F1. That choice is written to `adapters/best_sanitizer_reason.txt`.
- Canary exposure uses the sampled-rank estimator in the handout: 9,999 alternatives, `|R|=10^6`. Rank-1 in the sample saturates at about 13.3 bits, not `log2(10^6) ≈ 19.9`. That ceiling is a property of the estimator, not of the model.

## Kaggle

1. Create a private dataset containing this repository (code + `data/`).
2. Create a private GPU script kernel with Internet on, attach the dataset, and set the code file to `kaggle/kernel.py`.
3. The kernel writes `results/` and `adapters/` under `/kaggle/working`.

## Layout

```
README.md
requirements.txt
run_all.sh
src/peft_sweep.py
src/sanitize_eval.py
src/train_victim.py
src/attack_extract.py
src/lib/
data/
results/
kaggle/kernel.py
hpc/submit.pbs
```
