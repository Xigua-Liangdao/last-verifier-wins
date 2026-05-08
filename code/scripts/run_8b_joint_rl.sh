#!/usr/bin/env bash
set -eo pipefail

MODEL="${MODEL:-meta-llama/Llama-3.1-8B}"
STOP_AFTER="${STOP_AFTER:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"

SFT_INFO="${SFT_INFO:-logs/8b_sft/checkpoint_info.json}"
GSM8K_LOCAL="${GSM8K_LOCAL:-}"
TULU_LOCAL="${TULU_LOCAL:-}"

JOINT_ITERS="${JOINT_ITERS:-25}"
JOINT_SAVE_EVERY="${JOINT_SAVE_EVERY:-5}"
JOINT_MATH_BATCH="${JOINT_MATH_BATCH:-64}"
JOINT_IF_BATCH="${JOINT_IF_BATCH:-64}"
JOINT_GROUP="${JOINT_GROUP:-8}"
JOINT_LR="${JOINT_LR:-2e-5}"
JOINT_LR_SCHEDULE="${JOINT_LR_SCHEDULE:-cosine}"
JOINT_MAX_TOKENS="${JOINT_MAX_TOKENS:-512}"
JOINT_LOG_DIR="${JOINT_LOG_DIR:-./logs/8b_joint_rl}"
JOINT_CHECKPOINT_NAME="${JOINT_CHECKPOINT_NAME:-8b_joint_rl}"

SUMMARY_PATH="${SUMMARY_PATH:-results/summary_metrics.json}"

require_file() {
    if [ ! -f "$1" ]; then
        echo "ERROR: required file not found: $1"
        exit 1
    fi
}

json_field() {
    "$PYTHON_BIN" - "$1" "$2" <<'PY'
import json
import sys

path = sys.argv[1]
field = sys.argv[2]
with open(path) as f:
    data = json.load(f)
value = data
for part in field.split('.'):
    value = value[part]
print(value or "")
PY
}

checkpoint_path_for_iteration() {
    "$PYTHON_BIN" - "$1" "$2" <<'PY'
import json
import sys

info_path = sys.argv[1]
target_iter = int(sys.argv[2])
with open(info_path) as f:
    data = json.load(f)
matches = [cp for cp in data.get("checkpoints", []) if int(cp.get("iteration", -1)) == target_iter]
matches.sort(key=lambda cp: (cp.get("tag") != "final", cp.get("tag") != "mid"))
print((matches[0].get("path") if matches else "") or "")
PY
}

metric_triplet() {
    "$PYTHON_BIN" - "$1" <<'PY'
import json
import sys

with open(sys.argv[1]) as f:
    d = json.load(f)

ifeval = float(d["ifeval"]["metrics"]["google/IFEval/final_acc"]) * 100
gsm8k = float(d["gsm8k"]["metrics"]["openai/gsm8k/accuracy"]) * 100
humaneval = float(d["humaneval"]["metrics"]["openai/openai_humaneval/accuracy"]) * 100
avg = (ifeval + gsm8k + humaneval) / 3
print(f"{ifeval:.2f} {gsm8k:.2f} {humaneval:.2f} {avg:.2f}")
PY
}

if [ -z "${TINKER_API_KEY:-}" ]; then
    echo "ERROR: set TINKER_API_KEY in your environment."
    exit 1
fi

require_file "$SFT_INFO"
mkdir -p "$JOINT_LOG_DIR" evaluation results
export LLAMA3_TOKENIZER_DIR="${LLAMA3_TOKENIZER_DIR:-$(pwd)/evaluation/local_tokenizers/meta-llama-3-instruct-tokenizer}"

SFT_STATE="${SFT_STATE:-$(json_field "$SFT_INFO" final_state_path)}"
if [ -z "$SFT_STATE" ]; then
    echo "ERROR: failed to resolve final_state_path from $SFT_INFO"
    exit 1
fi

echo "========================================"
echo "STEP 0: reuse existing 8B SFT state"
echo "========================================"
echo "SFT info:  $SFT_INFO"
echo "SFT state: $SFT_STATE"

if [ "$STOP_AFTER" = "resolve_sft" ]; then
    echo "[stop] STOP_AFTER=resolve_sft"
    exit 0
fi

echo "========================================"
echo "STEP 1: joint RL from SFT state"
echo "========================================"
JOINT_ARGS=(
    --load_from_sft "$SFT_STATE"
    --model "$MODEL"
    --num_iterations "$JOINT_ITERS"
    --save_every "$JOINT_SAVE_EVERY"
    --math_batch_size "$JOINT_MATH_BATCH"
    --if_batch_size "$JOINT_IF_BATCH"
    --group_size "$JOINT_GROUP"
    --learning_rate "$JOINT_LR"
    --lr_schedule "$JOINT_LR_SCHEDULE"
    --max_tokens "$JOINT_MAX_TOKENS"
    --log_dir "$JOINT_LOG_DIR"
    --checkpoint_name "$JOINT_CHECKPOINT_NAME"
)
if [ -d "$GSM8K_LOCAL" ]; then
    JOINT_ARGS+=(--gsm8k_path "$GSM8K_LOCAL")
fi
if [ -d "$TULU_LOCAL" ]; then
    JOINT_ARGS+=(--tulu_path "$TULU_LOCAL")
fi
"$PYTHON_BIN" code/ablations/train/multitask_rl.py "${JOINT_ARGS[@]}" 2>&1 | tee "$JOINT_LOG_DIR/train.log"

JOINT_INFO="$JOINT_LOG_DIR/checkpoint_info.json"
require_file "$JOINT_INFO"

if [ "$STOP_AFTER" = "joint_train" ]; then
    echo "[stop] STOP_AFTER=joint_train"
    exit 0
fi

echo "========================================"
echo "STEP 2: full eval at iter 5/10/15/20/25"
echo "========================================"
for iter in 5 10 15 20 25; do
    ckpt_path="$(checkpoint_path_for_iteration "$JOINT_INFO" "$iter")"
    if [ -z "$ckpt_path" ]; then
        echo "ERROR: missing checkpoint path for iteration $iter in $JOINT_INFO"
        exit 1
    fi
    out_path="evaluation/submission_8b_joint_rl_iter$(printf '%02d' "$iter").json"
    log_path="$JOINT_LOG_DIR/full_eval_iter$(printf '%02d' "$iter").log"
    echo "[eval] iter=$iter checkpoint=$ckpt_path"
    "$PYTHON_BIN" evaluation/eval_all.py --checkpoint_path "$ckpt_path" --base_model "$MODEL" --output_path "$out_path" 2>&1 | tee "$log_path"
    read cur_if cur_gsm cur_he cur_avg < <(metric_triplet "$out_path")
    echo "[iter $iter] IFEval=$cur_if GSM8K=$cur_gsm HumanEval=$cur_he Avg=$cur_avg"
done

echo "DONE 8B joint RL pipeline."