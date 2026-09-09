#!/usr/bin/env bash
# End-to-end assignment run. Paths are overridable so the same script
# executes on a local checkout, Kaggle (/kaggle/working), or IITD HPC.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export A1_ROOT="${A1_ROOT:-$ROOT}"
export A1_DATA="${A1_DATA:-$ROOT/data}"
export A1_OUT="${A1_OUT:-$ROOT}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$A1_OUT/.hf}"

PYTHON="${PYTHON:-python}"
STAGE="${STAGE:-all}"
SEED="${SEED:-42}"
EPOCHS_A="${EPOCHS_A:-2}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

ADAPTERS="$A1_OUT/adapters"
RESULTS="$A1_OUT/results"
mkdir -p "$ADAPTERS" "$RESULTS" "$HF_HOME"

MODULES_ATTN="q_proj,v_proj"
MODULES_ATTN_FULL="q_proj,k_proj,v_proj,o_proj"
MODULES_ALL="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"

run_cfg() {
  local id="$1" modules="$2" r="$3" alpha="$4" quant="$5"
  local out="$ADAPTERS/$id"
  local skip=()
  if [[ "$SKIP_EXISTING" == "1" && -f "$out/adapter_config.json" ]]; then
    skip=(--skip_train)
  fi
  "$PYTHON" "$ROOT/src/peft_sweep.py" \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --train_file "$A1_DATA/partA_train.jsonl" \
    --val_file "$A1_DATA/partA_validation.jsonl" \
    --target_modules "$modules" \
    --lora_r "$r" --lora_alpha "$alpha" --lora_dropout 0.05 \
    --quant "$quant" --seed "$SEED" --epochs "$EPOCHS_A" \
    --run_id "$id" --output_dir "$out" \
    --results_csv "$RESULTS/partA_sweep.csv" \
    "${skip[@]}"
}

part_a() {
  run_cfg A1 "$MODULES_ATTN" 8 16 none
  run_cfg A2 "$MODULES_ATTN_FULL" 8 16 none
  run_cfg A3 "$MODULES_ALL" 8 16 none
  run_cfg A4 "$MODULES_ALL" 32 16 none
  run_cfg A5 "$MODULES_ALL" 32 64 none
  run_cfg A6 "$MODULES_ALL" 32 64 4bit
  "$PYTHON" "$ROOT/src/peft_sweep.py" --plot_only \
    --results_csv "$RESULTS/partA_sweep.csv" \
    --fig_path "$RESULTS/partA_fig.png"
  "$PYTHON" "$ROOT/src/select_best.py" \
    --results_csv "$RESULTS/partA_sweep.csv" \
    --dest "$ADAPTERS/best_sanitizer"
}

eval_system() {
  local system="$1" adapter="$2" test="$3" pred="$4" report="$5"
  "$PYTHON" "$ROOT/src/sanitize_eval.py" \
    --base_model Qwen/Qwen2.5-1.5B-Instruct \
    --adapter_path "$adapter" \
    --test_file "$test" \
    --system "$system" \
    --output_predictions "$pred" \
    --report_file "$report"
}

part_b() {
  local best="$ADAPTERS/best_sanitizer"
  eval_system finetuned "$best" "$A1_DATA/partB_test.jsonl" \
    "$RESULTS/partB_predictions.jsonl" "$RESULTS/partB_metrics.json"
  eval_system regex "" "$A1_DATA/partB_test.jsonl" \
    "$RESULTS/partB_predictions_regex.jsonl" "$RESULTS/partB_metrics_regex.json"
  eval_system presidio "" "$A1_DATA/partB_test.jsonl" \
    "$RESULTS/partB_predictions_presidio.jsonl" "$RESULTS/partB_metrics_presidio.json"
  eval_system zeroshot "" "$A1_DATA/partB_test.jsonl" \
    "$RESULTS/partB_predictions_zeroshot.jsonl" "$RESULTS/partB_metrics_zeroshot.json"
  eval_system finetuned "$best" "$A1_DATA/partB_ood_test.jsonl" \
    "$RESULTS/partB_predictions_ood.jsonl" "$RESULTS/partB_metrics_ood.json"
  eval_system regex "" "$A1_DATA/partB_ood_test.jsonl" \
    "$RESULTS/partB_predictions_ood_regex.jsonl" "$RESULTS/partB_metrics_ood_regex.json"
  eval_system presidio "" "$A1_DATA/partB_ood_test.jsonl" \
    "$RESULTS/partB_predictions_ood_presidio.jsonl" "$RESULTS/partB_metrics_ood_presidio.json"
}

part_c() {
  local best="$ADAPTERS/best_sanitizer"
  local san_corpus="$A1_OUT/data_derived/partC_corpus_sanitized.jsonl"
  mkdir -p "$A1_OUT/data_derived"
  if [[ ! -f "$san_corpus" ]]; then
    "$PYTHON" "$ROOT/src/sanitize_eval.py" \
      --base_model Qwen/Qwen2.5-1.5B-Instruct \
      --adapter_path "$best" \
      --test_file "$A1_DATA/partC_corpus_raw.jsonl" \
      --system finetuned \
      --sanitize_corpus_out "$san_corpus"
  fi

  local skip_raw=() skip_san=()
  if [[ "$SKIP_EXISTING" == "1" && -f "$ADAPTERS/victim_raw/adapter_config.json" ]]; then
    skip_raw=(--skip_train)
  fi
  if [[ "$SKIP_EXISTING" == "1" && -f "$ADAPTERS/victim_san/adapter_config.json" ]]; then
    skip_san=(--skip_train)
  fi

  "$PYTHON" "$ROOT/src/train_victim.py" \
    --model Qwen/Qwen2.5-0.5B \
    --corpus "$A1_DATA/partC_corpus_raw.jsonl" \
    --heldout "$A1_DATA/partC_heldout.jsonl" \
    --target_modules "$MODULES_ALL" \
    --lora_r 32 --lora_alpha 64 --epochs 4 --seed "$SEED" \
    --output_dir "$ADAPTERS/victim_raw" "${skip_raw[@]}"

  "$PYTHON" "$ROOT/src/train_victim.py" \
    --model Qwen/Qwen2.5-0.5B \
    --corpus "$san_corpus" \
    --heldout "$A1_DATA/partC_heldout.jsonl" \
    --target_modules "$MODULES_ALL" \
    --lora_r 32 --lora_alpha 64 --epochs 4 --seed "$SEED" \
    --output_dir "$ADAPTERS/victim_san" "${skip_san[@]}"

  "$PYTHON" "$ROOT/src/attack_extract.py" \
    --victim_model Qwen/Qwen2.5-0.5B --adapter_path "$ADAPTERS/victim_raw" \
    --ref_model Qwen/Qwen2.5-0.5B \
    --corpus "$A1_DATA/partC_corpus_raw.jsonl" \
    --canaries "$A1_DATA/partC_canaries.jsonl" \
    --targeted_file "$A1_DATA/partC_targeted.jsonl" \
    --attack all --num_samples 5000 --seed "$SEED" \
    --victim_name victim_raw \
    --report_file "$RESULTS/partC_attacks.json"

  "$PYTHON" "$ROOT/src/attack_extract.py" \
    --victim_model Qwen/Qwen2.5-0.5B --adapter_path "$ADAPTERS/victim_san" \
    --ref_model Qwen/Qwen2.5-0.5B \
    --corpus "$A1_DATA/partC_corpus_raw.jsonl" \
    --canaries "$A1_DATA/partC_canaries.jsonl" \
    --targeted_file "$A1_DATA/partC_targeted.jsonl" \
    --attack all --num_samples 5000 --seed "$SEED" \
    --victim_name victim_san \
    --report_file "$RESULTS/partC_attacks.json"

  "$PYTHON" "$ROOT/src/attack_extract.py" \
    --victim_model Qwen/Qwen2.5-0.5B --adapter_path "" \
    --ref_model Qwen/Qwen2.5-0.5B \
    --corpus "$A1_DATA/partC_corpus_raw.jsonl" \
    --canaries "$A1_DATA/partC_canaries.jsonl" \
    --targeted_file "$A1_DATA/partC_targeted.jsonl" \
    --attack all --num_samples 5000 --seed "$SEED" \
    --victim_name pretrained \
    --report_file "$RESULTS/partC_attacks.json"
}

"$PYTHON" - <<'PY'
from pathlib import Path
import sys
sys.path.insert(0, str(Path("src") if Path("src").exists() else Path(".")))
import os
from pathlib import Path as P
sys.path.insert(0, str(P(os.environ["A1_ROOT"]) / "src"))
from lib.io_utils import verify_manifest
from lib.paths import data_dir
verify_manifest(data_dir())
print("dataset bundle OK", data_dir())
PY

case "$STAGE" in
  a|A|partA) part_a ;;
  b|B|partB) part_b ;;
  c|C|partC) part_c ;;
  bc|BC|partBC)
    part_b
    part_c
    ;;
  all)
    part_a
    part_b
    part_c
    ;;
  *)
    echo "unknown STAGE=$STAGE" >&2
    exit 1
    ;;
esac
