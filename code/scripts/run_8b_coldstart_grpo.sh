#!/usr/bin/env bash
: <<'DOC'
Final release script for the 8B cold-start control.

Stage: start from the 8B SFT final state and run GSM8K GRPO directly, skipping
IF-RLVR. Key settings: 25 GRPO iterations, batch 128, group 16, lr 3e-5,
with full evaluations at iter 5/10/15/20/25.
DOC
set -eo pipefail

MODEL="${MODEL:-meta-llama/Llama-3.1-8B}"
STOP_AFTER="${STOP_AFTER:-}"

SFT_INFO="${SFT_INFO:-logs/8b_sft/checkpoint_info.json}"
GSM8K_LOCAL="${GSM8K_LOCAL:-}"

GRPO_ITERS="${GRPO_ITERS:-25}"
GRPO_BATCH="${GRPO_BATCH:-128}"
GRPO_GROUP="${GRPO_GROUP:-16}"
GRPO_LR="${GRPO_LR:-3e-5}"
GRPO_MAX_TOKENS="${GRPO_MAX_TOKENS:-320}"
GRPO_SAVE_EVERY="${GRPO_SAVE_EVERY:-5}"
GRPO_LOG_DIR="${GRPO_LOG_DIR:-./logs/8b_coldstart_grpo}"
GRPO_CHECKPOINT_NAME="${GRPO_CHECKPOINT_NAME:-8b_coldstart_grpo}"

SUMMARY_PATH="${SUMMARY_PATH:-results/summary_metrics.json}"
SUMMARY_SEED_PATH="${SUMMARY_SEED_PATH:-report_extension_bundle/results/summary_metrics.json}"

require_file() {
    if [ ! -f "$1" ]; then
        echo "ERROR: required file not found: $1"
        exit 1
    fi
}

json_field() {
    python - "$1" "$2" <<'PY'
import json
import sys

path = sys.argv[1]
field = sys.argv[2]
with open(path) as f:
    data = json.load(f)
value = data
for part in field.split('.'):
    value = value[part]
if value is None:
    print("")
else:
    print(value)
PY
}

checkpoint_path_for_iteration() {
    python - "$1" "$2" <<'PY'
import json
import sys

info_path = sys.argv[1]
target_iter = int(sys.argv[2])
with open(info_path) as f:
    data = json.load(f)

checkpoints = data.get("checkpoints", [])
matches = [cp for cp in checkpoints if int(cp.get("iteration", -1)) == target_iter]
matches.sort(key=lambda cp: (cp.get("tag") != "final", cp.get("tag") != "mid"))

if matches:
    print(matches[0].get("path") or "")
    sys.exit(0)

if int(data.get("iterations_run", -1)) == target_iter:
    print(data.get("final_path") or "")
    sys.exit(0)

print("")
PY
}

metric_triplet() {
    python - "$1" <<'PY'
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
mkdir -p "$GRPO_LOG_DIR" evaluation "$(dirname "$SUMMARY_PATH")"
export LLAMA3_TOKENIZER_DIR="${LLAMA3_TOKENIZER_DIR:-$(pwd)/evaluation/local_tokenizers/meta-llama-3-instruct-tokenizer}"

SFT_STATE="${SFT_STATE:-$(json_field "$SFT_INFO" final_state_path)}"
if [ -z "$SFT_STATE" ]; then
    echo "ERROR: failed to resolve final_state_path from $SFT_INFO"
    exit 1
fi

echo "========================================"
echo "STEP 1: reuse existing 8B SFT state"
echo "========================================"
echo "SFT info:  $SFT_INFO"
echo "SFT state: $SFT_STATE"
echo "Pipeline:  cold-start GSM8K GRPO only"

if [ "$STOP_AFTER" = "resolve_sft" ]; then
    echo "[stop] STOP_AFTER=resolve_sft"
    exit 0
fi

echo "========================================"
echo "STEP 2: GSM8K GRPO directly from SFT state"
echo "========================================"
GRPO_ARGS=(
    --model "$MODEL"
    --load_from_sft "$SFT_STATE"
    --batch_size "$GRPO_BATCH"
    --group_size "$GRPO_GROUP"
    --learning_rate "$GRPO_LR"
    --max_tokens "$GRPO_MAX_TOKENS"
    --num_iterations "$GRPO_ITERS"
    --save_every "$GRPO_SAVE_EVERY"
    --log_dir "$GRPO_LOG_DIR"
    --checkpoint_name "$GRPO_CHECKPOINT_NAME"
)
if [ -d "$GSM8K_LOCAL" ]; then
    GRPO_ARGS+=(--gsm8k_path "$GSM8K_LOCAL")
fi
python evaluation/grpo_gsm8k.py "${GRPO_ARGS[@]}" 2>&1 | tee "$GRPO_LOG_DIR/train.log"

GRPO_INFO="$GRPO_LOG_DIR/checkpoint_info.json"
require_file "$GRPO_INFO"

if [ "$STOP_AFTER" = "grpo_train" ]; then
    echo "[stop] STOP_AFTER=grpo_train"
    exit 0
fi

echo "========================================"
echo "STEP 3: full eval at iter 5/10/15/20/25"
echo "========================================"
for iter in 5 10 15 20 25; do
    ckpt_path="$(checkpoint_path_for_iteration "$GRPO_INFO" "$iter")"
    if [ -z "$ckpt_path" ]; then
        echo "ERROR: missing checkpoint path for iteration $iter in $GRPO_INFO"
        exit 1
    fi

    out_path="evaluation/submission_8b_coldstart_grpo_iter$(printf '%02d' "$iter").json"
    log_path="$GRPO_LOG_DIR/full_eval_iter$(printf '%02d' "$iter").log"
    echo "[eval] iter=$iter checkpoint=$ckpt_path"
    python evaluation/eval_all.py \
        --checkpoint_path "$ckpt_path" \
        --base_model "$MODEL" \
        --output_path "$out_path" 2>&1 | tee "$log_path"

    read cur_if cur_gsm cur_he cur_avg < <(metric_triplet "$out_path")
    echo "[iter $iter] IFEval=$cur_if GSM8K=$cur_gsm HumanEval=$cur_he Avg=$cur_avg"
done

echo "========================================"
echo "STEP 4: update summary metrics"
echo "========================================"
python - "$SUMMARY_PATH" "$SUMMARY_SEED_PATH" <<'PY'
import json
import os
import sys

summary_path = sys.argv[1]
seed_path = sys.argv[2]

if os.path.exists(summary_path):
    with open(summary_path) as f:
        summary = json.load(f)
elif os.path.exists(seed_path):
    with open(seed_path) as f:
        summary = json.load(f)
else:
    summary = []

def score(path):
    with open(path) as f:
        d = json.load(f)
    if_acc = float(d["ifeval"]["metrics"]["google/IFEval/final_acc"]) * 100
    gsm8k = float(d["gsm8k"]["metrics"]["openai/gsm8k/accuracy"]) * 100
    he = float(d["humaneval"]["metrics"]["openai/openai_humaneval/accuracy"]) * 100
    avg = (if_acc + gsm8k + he) / 3
    return {
        "IFEval": round(if_acc, 2),
        "GSM8K": round(gsm8k, 2),
        "HumanEval": round(he, 2),
        "Avg": round(avg, 2),
    }

new_entries = []
for iteration in (5, 10, 15, 20, 25):
    path = f"evaluation/submission_8b_coldstart_grpo_iter{iteration:02d}.json"
    metrics = score(path)
    new_entries.append({
        "name": f"8B Cold-start GRPO iter{iteration}",
        "path": path,
        **metrics,
    })

new_names = {entry["name"] for entry in new_entries}
summary = [entry for entry in summary if entry.get("name") not in new_names]
summary.extend(new_entries)

os.makedirs(os.path.dirname(summary_path), exist_ok=True)
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)
    f.write("\n")

for entry in new_entries:
    print(
        f"{entry['name']}: IFEval={entry['IFEval']:.2f} "
        f"GSM8K={entry['GSM8K']:.2f} HumanEval={entry['HumanEval']:.2f} "
        f"Avg={entry['Avg']:.2f}"
    )
print(f"updated {summary_path}")
PY

echo "========================================"
echo "DONE 8B cold-start GRPO pipeline."
echo "  evaluation/submission_8b_coldstart_grpo_iter05.json"
echo "  evaluation/submission_8b_coldstart_grpo_iter10.json"
echo "  evaluation/submission_8b_coldstart_grpo_iter15.json"
echo "  evaluation/submission_8b_coldstart_grpo_iter20.json"
echo "  evaluation/submission_8b_coldstart_grpo_iter25.json"
echo "  $SUMMARY_PATH"
echo "========================================"