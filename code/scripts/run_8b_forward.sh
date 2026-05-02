#!/usr/bin/env bash
: <<'DOC'
Final release script for the forward 8B pipeline.

Stages: multi-task SFT -> IF-RLVR -> GSM8K GRPO.
Key settings: SFT rank 128 / batch 128 / 2 epochs, IF-RLVR 40 iters at lr 2e-5,
GRPO 25 iters at lr 3e-5 with full-eval checkpoint selection.
DOC
set -eo pipefail

MODEL="${MODEL:-meta-llama/Llama-3.1-8B}"
DATA_JSONL="${DATA_JSONL:-evaluation/train_3b_pipeline.jsonl}"
STOP_AFTER="${STOP_AFTER:-}"

SFT_RANK=128
SFT_BATCH=128
SFT_EPOCHS=2
SFT_MAX_LEN=2048
SFT_SAVE_EVERY=150
SFT_LOG_DIR="./logs/8b_sft"

IF_RLVR_ITERS=40
IF_RLVR_BATCH=64
IF_RLVR_GROUP=8
IF_RLVR_LR=2e-5
IF_RLVR_MAX_TOKENS=512
IF_RLVR_SAVE_EVERY=10
IF_RLVR_LOG_DIR="./logs/8b_if_rlvr"

GRPO_ITERS=25
GRPO_BATCH=128
GRPO_GROUP=16
GRPO_LR=3e-5
GRPO_MAX_TOKENS=320
GRPO_SAVE_EVERY=5
GRPO_LOG_DIR="./logs/8b_grpo"

mkdir -p "$SFT_LOG_DIR" "$IF_RLVR_LOG_DIR" "$GRPO_LOG_DIR"
export LLAMA3_TOKENIZER_DIR="${LLAMA3_TOKENIZER_DIR:-$(pwd)/evaluation/local_tokenizers/meta-llama-3-instruct-tokenizer}"
rm -f \
    "$SFT_LOG_DIR/train.log" "$SFT_LOG_DIR/checkpoint_info.json" \
    "$IF_RLVR_LOG_DIR/train.log" "$IF_RLVR_LOG_DIR/checkpoint_info.json" \
    "$GRPO_LOG_DIR/train.log" "$GRPO_LOG_DIR/checkpoint_info.json" \
    evaluation/submission_8b_sft_quick.json \
    evaluation/submission_8b_if_rlvr.json \
    evaluation/submission_8b_final.json

require_file() {
    if [ ! -f "$1" ]; then
        echo "ERROR: required file not found: $1"
        exit 1
    fi
}

wait_for_pattern() {
    local file="$1"
    local pattern="$2"
    local max_wait="${3:-1800}"
    local waited=0
    while [ "$waited" -lt "$max_wait" ]; do
        if [ -f "$file" ] && grep -qE "$pattern" "$file"; then
            return 0
        fi
        sleep 5
        waited=$((waited + 5))
    done
    return 1
}

parse_last_float() {
    python - "$1" "$2" <<'PY'
import pathlib
import re
import sys

path = pathlib.Path(sys.argv[1])
pattern = re.compile(sys.argv[2])
text = path.read_text() if path.exists() else ""
matches = pattern.findall(text)
if not matches:
    sys.exit(1)
value = matches[-1]
if isinstance(value, tuple):
    value = value[-1]
print(value)
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

require_file "$DATA_JSONL"

echo "========================================"
echo "STEP 1: reusing existing train jsonl"
echo "========================================"
echo "[skip] $DATA_JSONL already exists"

echo "========================================"
echo "STEP 2: 8B SFT with guardrails"
echo "========================================"

(
    HF_HUB_OFFLINE=1 python sft_train.py \
        --data "$DATA_JSONL" \
        --model "$MODEL" \
        --rank $SFT_RANK \
        --batch_size $SFT_BATCH \
        --num_epochs $SFT_EPOCHS \
        --max_length $SFT_MAX_LEN \
        --save_every $SFT_SAVE_EVERY \
        --log_dir "$SFT_LOG_DIR" \
        --checkpoint_name "8b_sft" 2>&1 | tee "$SFT_LOG_DIR/train.log"
) &
SFT_PIPE_PID=$!

if ! wait_for_pattern "$SFT_LOG_DIR/train.log" 'using cookbook get_lr' 300; then
    echo "ERROR: timed out waiting for SFT LR line."
    kill "$SFT_PIPE_PID" || true
    exit 1
fi
SFT_BASE_LR=$(parse_last_float "$SFT_LOG_DIR/train.log" 'using cookbook get_lr\(.*\)\s*=\s*([0-9.eE+-]+)')
if ! python - "$SFT_BASE_LR" <<'PY'
import sys
lr = float(sys.argv[1])
ok = 2.5e-4 <= lr <= 3.2e-4
print(f"[guard] sft cookbook lr = {lr:.8f}")
sys.exit(0 if ok else 1)
PY
then
    echo "ERROR: SFT cookbook LR out of expected 8B range; stopping."
    kill "$SFT_PIPE_PID" || true
    exit 1
fi

if ! wait_for_pattern "$SFT_LOG_DIR/train.log" 'step\s+50/' 7200; then
    echo "ERROR: timed out waiting for SFT step 50."
    kill "$SFT_PIPE_PID" || true
    exit 1
fi
SFT_STEP50_AVG=$(parse_last_float "$SFT_LOG_DIR/train.log" 'step\s+50/\d+\s+loss=[0-9.]+\s+avg50=([0-9.]+)')
if ! python - "$SFT_STEP50_AVG" <<'PY'
import sys
avg = float(sys.argv[1])
print(f"[guard] sft step50 avg50 = {avg:.4f}")
sys.exit(0 if avg <= 1.0 else 1)
PY
then
    echo "ERROR: SFT step 50 avg50 > 1.0; stopping."
    kill "$SFT_PIPE_PID" || true
    exit 1
fi

if ! wait_for_pattern "$SFT_LOG_DIR/train.log" 'step\s+100/' 10800; then
    echo "ERROR: timed out waiting for SFT step 100."
    kill "$SFT_PIPE_PID" || true
    exit 1
fi
SFT_STEP100_AVG=$(parse_last_float "$SFT_LOG_DIR/train.log" 'step\s+100/\d+\s+loss=[0-9.]+\s+avg50=([0-9.]+)')
if ! python - "$SFT_STEP100_AVG" <<'PY'
import sys
avg = float(sys.argv[1])
print(f"[guard] sft step100 avg50 = {avg:.4f}")
sys.exit(0 if avg < 0.85 else 1)
PY
then
    echo "ERROR: SFT step 100 avg50 >= 0.85; stopping."
    kill "$SFT_PIPE_PID" || true
    exit 1
fi

if ! wait_for_pattern "$SFT_LOG_DIR/train.log" 'step\s+300/' 21600; then
    echo "ERROR: timed out waiting for SFT step 300."
    kill "$SFT_PIPE_PID" || true
    exit 1
fi
SFT_STEP300_AVG=$(parse_last_float "$SFT_LOG_DIR/train.log" 'step\s+300/\d+\s+loss=[0-9.]+\s+avg50=([0-9.]+)')
if ! python - "$SFT_STEP300_AVG" <<'PY'
import sys
avg = float(sys.argv[1])
print(f"[guard] sft step300 avg50 = {avg:.4f}")
sys.exit(0 if avg <= 0.85 else 1)
PY
then
    echo "ERROR: SFT step 300 avg50 > 0.85; stopping."
    kill "$SFT_PIPE_PID" || true
    exit 1
fi

wait "$SFT_PIPE_PID"

SFT_FINAL=$(python -c "import json; d=json.load(open('$SFT_LOG_DIR/checkpoint_info.json')); print(d['final_path'])")
SFT_STATE=$(python -c "import json; d=json.load(open('$SFT_LOG_DIR/checkpoint_info.json')); print(d.get('final_state_path') or '')")
if [ -z "$SFT_STATE" ]; then
    echo "ERROR: SFT did not produce final_state_path."
    exit 1
fi

echo "========================================"
echo "STEP 3: quick eval of SFT checkpoint"
echo "========================================"
python evaluation/eval_all.py \
    --checkpoint_path "$SFT_FINAL" \
    --base_model "$MODEL" \
    --limit 50 \
    --output_path evaluation/submission_8b_sft_quick.json

read SFT_IF SFT_GSM SFT_HE SFT_AVG < <(metric_triplet evaluation/submission_8b_sft_quick.json)
echo "[quick] IFEval=$SFT_IF GSM8K=$SFT_GSM HumanEval=$SFT_HE Avg=$SFT_AVG"

if [ "$STOP_AFTER" = "sft_quick" ]; then
    echo "[stop] STOP_AFTER=sft_quick"
    exit 0
fi

echo "========================================"
echo "STEP 4: IF-RLVR on SFT checkpoint"
echo "========================================"
python evaluation/if_rlvr.py \
    --model "$MODEL" \
    --load_from_sft "$SFT_STATE" \
    --batch_size $IF_RLVR_BATCH \
    --group_size $IF_RLVR_GROUP \
    --learning_rate $IF_RLVR_LR \
    --max_tokens $IF_RLVR_MAX_TOKENS \
    --num_iterations $IF_RLVR_ITERS \
    --save_every $IF_RLVR_SAVE_EVERY \
    --log_dir "$IF_RLVR_LOG_DIR" \
    --checkpoint_name "8b_if_rlvr" 2>&1 | tee "$IF_RLVR_LOG_DIR/train.log"

IF_RLVR_FINAL=$(python -c "import json; d=json.load(open('$IF_RLVR_LOG_DIR/checkpoint_info.json')); print(d['final_path'])")
IF_RLVR_STATE=$(python -c "import json; d=json.load(open('$IF_RLVR_LOG_DIR/checkpoint_info.json')); print(d.get('final_state_path') or '')")
if [ -z "$IF_RLVR_STATE" ]; then
    echo "ERROR: IF-RLVR did not produce final_state_path."
    exit 1
fi

echo "========================================"
echo "STEP 5: full eval of IF-RLVR checkpoint"
echo "========================================"
python evaluation/eval_all.py \
    --checkpoint_path "$IF_RLVR_FINAL" \
    --base_model "$MODEL" \
    --output_path evaluation/submission_8b_if_rlvr.json

read IF_IF IF_GSM IF_HE IF_AVG < <(metric_triplet evaluation/submission_8b_if_rlvr.json)
echo "[if-rlvr] IFEval=$IF_IF GSM8K=$IF_GSM HumanEval=$IF_HE Avg=$IF_AVG"

if [ "$STOP_AFTER" = "if_rlvr_eval" ]; then
    echo "[stop] STOP_AFTER=if_rlvr_eval"
    exit 0
fi

echo "========================================"
echo "STEP 6: GSM8K GRPO on IF-RLVR checkpoint"
echo "========================================"
python evaluation/grpo_gsm8k.py \
    --model "$MODEL" \
    --load_from_sft "$IF_RLVR_STATE" \
    --batch_size $GRPO_BATCH \
    --group_size $GRPO_GROUP \
    --learning_rate $GRPO_LR \
    --max_tokens $GRPO_MAX_TOKENS \
    --num_iterations $GRPO_ITERS \
    --save_every $GRPO_SAVE_EVERY \
    --log_dir "$GRPO_LOG_DIR" \
    --checkpoint_name "8b_grpo" 2>&1 | tee "$GRPO_LOG_DIR/train.log"

GRPO_FINAL=$(python -c "import json; d=json.load(open('$GRPO_LOG_DIR/checkpoint_info.json')); print(d['final_path'])")

echo "========================================"
echo "STEP 7: final full eval"
echo "========================================"
python evaluation/eval_all.py \
    --checkpoint_path "$GRPO_FINAL" \
    --base_model "$MODEL" \
    --output_path evaluation/submission_8b_final.json

python - <<'PY'
import json
import shutil

def score(path):
    with open(path) as f:
        d = json.load(f)
    if_acc = float(d["ifeval"]["metrics"]["google/IFEval/final_acc"]) * 100
    gsm8k = float(d["gsm8k"]["metrics"]["openai/gsm8k/accuracy"]) * 100
    he = float(d["humaneval"]["metrics"]["openai/openai_humaneval/accuracy"]) * 100
    avg = (if_acc + gsm8k + he) / 3
    return avg, if_acc, gsm8k, he

files = [
    "evaluation/submission_8b_if_rlvr.json",
    "evaluation/submission_8b_final.json",
]
results = []
for path in files:
    avg, if_acc, gsm8k, he = score(path)
    results.append((path, avg, if_acc, gsm8k, he))
    print(f"{path}: IFEval={if_acc:.1f} GSM8K={gsm8k:.1f} HumanEval={he:.1f} Avg={avg:.2f}")
best = max(results, key=lambda x: x[1])
shutil.copy(best[0], "evaluation/submission.json")
print(f">>> Best: {best[0]} with Avg {best[1]:.2f}, copied to evaluation/submission.json")
PY

echo "========================================"
echo "DONE 8B pipeline."
echo "  evaluation/submission_8b_sft_quick.json"
echo "  evaluation/submission_8b_if_rlvr.json"
echo "  evaluation/submission_8b_final.json"
echo "  evaluation/submission.json"
echo "========================================"
