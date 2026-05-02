#!/usr/bin/env bash
: <<'DOC'
Final release script for the reverse-order 8B pipeline.

Stages: 8B SFT -> GSM8K GRPO -> IF-RLVR. Key settings: GRPO 25 iterations at
lr 3e-5, followed by IF-RLVR 40 iterations at lr 2e-5, with full evaluations
at IF-RLVR iter 10/20/30/40.
DOC
set -eo pipefail

MODEL="${MODEL:-meta-llama/Llama-3.1-8B}"
STOP_AFTER="${STOP_AFTER:-}"

SFT_INFO="${SFT_INFO:-logs/8b_sft/checkpoint_info.json}"
COLDSTART_INFO="${COLDSTART_INFO:-logs/8b_coldstart_grpo/checkpoint_info.json}"
TULU_LOCAL="${TULU_LOCAL:-}"
GSM8K_LOCAL="${GSM8K_LOCAL:-}"

GRPO_ITERS="${GRPO_ITERS:-25}"
GRPO_BATCH="${GRPO_BATCH:-128}"
GRPO_GROUP="${GRPO_GROUP:-16}"
GRPO_LR="${GRPO_LR:-3e-5}"
GRPO_MAX_TOKENS="${GRPO_MAX_TOKENS:-320}"
GRPO_SAVE_EVERY="${GRPO_SAVE_EVERY:-5}"
GRPO_LOG_DIR="${GRPO_LOG_DIR:-./logs/8b_reverse_grpo}"
GRPO_CHECKPOINT_NAME="${GRPO_CHECKPOINT_NAME:-8b_reverse_grpo}"

IF_RLVR_ITERS="${IF_RLVR_ITERS:-40}"
IF_RLVR_BATCH="${IF_RLVR_BATCH:-64}"
IF_RLVR_GROUP="${IF_RLVR_GROUP:-8}"
IF_RLVR_LR="${IF_RLVR_LR:-2e-5}"
IF_RLVR_MAX_TOKENS="${IF_RLVR_MAX_TOKENS:-512}"
IF_RLVR_SAVE_EVERY="${IF_RLVR_SAVE_EVERY:-10}"
IF_RLVR_LOG_DIR="${IF_RLVR_LOG_DIR:-./logs/8b_reverse_ifrlvr}"
IF_RLVR_CHECKPOINT_NAME="${IF_RLVR_CHECKPOINT_NAME:-8b_reverse_ifrlvr}"

SUMMARY_PATH="${SUMMARY_PATH:-results/summary_metrics.json}"
REPORT_PATH="${REPORT_PATH:-results/reverse_vs_forward.md}"

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

matches = [
    cp for cp in data.get("checkpoints", [])
    if int(cp.get("iteration", -1)) == target_iter
]
matches.sort(key=lambda cp: (cp.get("tag") != "final", cp.get("tag") != "mid"))
print((matches[0].get("path") if matches else "") or "")
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
mkdir -p "$GRPO_LOG_DIR" "$IF_RLVR_LOG_DIR" evaluation results
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

GRPO_INFO=""
GRPO_STATE=""
GRPO_FINAL=""

if [ -f "$COLDSTART_INFO" ]; then
    GRPO_STATE="$(json_field "$COLDSTART_INFO" final_state_path 2>/dev/null || true)"
    GRPO_FINAL="$(json_field "$COLDSTART_INFO" final_path 2>/dev/null || true)"
    if [ -n "$GRPO_STATE" ]; then
        GRPO_INFO="$COLDSTART_INFO"
    fi
fi

if [ -n "$GRPO_INFO" ]; then
    echo "========================================"
    echo "STEP 1: reuse existing cold-start GRPO state"
    echo "========================================"
    echo "GRPO info:  $GRPO_INFO"
    echo "GRPO state: $GRPO_STATE"
else
    echo "========================================"
    echo "STEP 1: GSM8K GRPO on SFT checkpoint"
    echo "========================================"
    GRPO_ARGS=(
        --model "$MODEL"
        --load_from_sft "$SFT_STATE"
        --rank 32
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
    GRPO_STATE="$(json_field "$GRPO_INFO" final_state_path)"
    GRPO_FINAL="$(json_field "$GRPO_INFO" final_path)"
    if [ -z "$GRPO_STATE" ]; then
        echo "ERROR: GRPO did not produce final_state_path."
        exit 1
    fi
fi

if [ ! -f evaluation/submission_8b_coldstart_grpo_iter25.json ] && [ -n "$GRPO_FINAL" ]; then
    python evaluation/eval_all.py \
        --checkpoint_path "$GRPO_FINAL" \
        --base_model "$MODEL" \
        --output_path evaluation/submission_8b_coldstart_grpo_iter25.json
fi

if [ "$STOP_AFTER" = "grpo_ready" ]; then
    echo "[stop] STOP_AFTER=grpo_ready"
    exit 0
fi

echo "========================================"
echo "STEP 2: IF-RLVR on top of GRPO state"
echo "========================================"
IF_ARGS=(
    --model "$MODEL"
    --load_from_sft "$GRPO_STATE"
    --rank 32
    --batch_size "$IF_RLVR_BATCH"
    --group_size "$IF_RLVR_GROUP"
    --learning_rate "$IF_RLVR_LR"
    --max_tokens "$IF_RLVR_MAX_TOKENS"
    --num_iterations "$IF_RLVR_ITERS"
    --save_every "$IF_RLVR_SAVE_EVERY"
    --log_dir "$IF_RLVR_LOG_DIR"
    --checkpoint_name "$IF_RLVR_CHECKPOINT_NAME"
)
if [ -d "$TULU_LOCAL" ]; then
    IF_ARGS+=(--tulu_path "$TULU_LOCAL")
fi
python evaluation/if_rlvr.py "${IF_ARGS[@]}" 2>&1 | tee "$IF_RLVR_LOG_DIR/train.log"

IF_INFO="$IF_RLVR_LOG_DIR/checkpoint_info.json"
require_file "$IF_INFO"

if [ "$STOP_AFTER" = "if_train" ]; then
    echo "[stop] STOP_AFTER=if_train"
    exit 0
fi

echo "========================================"
echo "STEP 3: full eval at reverse iter 10/20/30/40"
echo "========================================"
for iter in 10 20 30 40; do
    ckpt_path="$(checkpoint_path_for_iteration "$IF_INFO" "$iter")"
    if [ -z "$ckpt_path" ]; then
        echo "ERROR: missing checkpoint path for iteration $iter in $IF_INFO"
        exit 1
    fi
    out_path="evaluation/submission_8b_reverse_iter$(printf '%02d' "$iter").json"
    log_path="$IF_RLVR_LOG_DIR/full_eval_iter$(printf '%02d' "$iter").log"
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
python - "$SUMMARY_PATH" <<'PY'
import json
import os
import sys

summary_path = sys.argv[1]
with open(summary_path) as f:
    summary = json.load(f)

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
for iteration in (10, 20, 30, 40):
    path = f"evaluation/submission_8b_reverse_iter{iteration:02d}.json"
    metrics = score(path)
    new_entries.append({
        "name": f"8B Reverse (math→IF) iter{iteration}",
        "path": path,
        **metrics,
    })

new_names = {entry["name"] for entry in new_entries}
summary = [entry for entry in summary if entry.get("name") not in new_names]
summary.extend(new_entries)

with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)
    f.write("\n")

for entry in new_entries:
    print(
        f"{entry['name']}: IFEval={entry['IFEval']:.2f} "
        f"GSM8K={entry['GSM8K']:.2f} HumanEval={entry['HumanEval']:.2f} "
        f"Avg={entry['Avg']:.2f}"
    )
PY

echo "========================================"
echo "STEP 5: write reverse_vs_forward report"
echo "========================================"
python - "$REPORT_PATH" <<'PY'
import json
import os
import sys

report_path = sys.argv[1]

rows = [
    ("8B SFT (baseline)", "evaluation/submission_8b_sft_full.json"),
    ("8B IF-RLVR-only", "evaluation/submission_8b_if_rlvr.json"),
    ("8B Cold-start GRPO iter25", "evaluation/submission_8b_coldstart_grpo_iter25.json"),
    ("8B Forward (IF→math) final", "evaluation/submission_8b_final.json"),
    ("8B Reverse (math→IF) final", "evaluation/submission_8b_reverse_iter40.json"),
]

def score(path):
    with open(path) as f:
        d = json.load(f)
    if_acc = float(d["ifeval"]["metrics"]["google/IFEval/final_acc"]) * 100
    gsm8k = float(d["gsm8k"]["metrics"]["openai/gsm8k/accuracy"]) * 100
    he = float(d["humaneval"]["metrics"]["openai/openai_humaneval/accuracy"]) * 100
    avg = (if_acc + gsm8k + he) / 3
    return [round(if_acc, 2), round(gsm8k, 2), round(he, 2), round(avg, 2)]

table = []
for name, path in rows:
    table.append((name, path, score(path)))

maxima = [max(row[2][col] for row in table) for col in range(4)]

def fmt(value, max_value):
    text = f"{value:.2f}"
    return f"**{text}**" if value == max_value else text

lines = [
    "# Reverse vs Forward 8B RLVR",
    "",
    "| Run | IFEval | GSM8K | HumanEval | Avg |",
    "| --- | ---: | ---: | ---: | ---: |",
]
for name, path, metrics in table:
    metric_cells = [fmt(value, maxima[idx]) for idx, value in enumerate(metrics)]
    lines.append(f"| {name} | {metric_cells[0]} | {metric_cells[1]} | {metric_cells[2]} | {metric_cells[3]} |")

lines.extend([
    "",
    "Files:",
])
for name, path, _ in table:
    lines.append(f"- {name}: {path}")

os.makedirs(os.path.dirname(report_path), exist_ok=True)
with open(report_path, "w") as f:
    f.write("\n".join(lines) + "\n")

print(f"wrote {report_path}")
PY

echo "========================================"
echo "DONE 8B reverse pipeline."
echo "  evaluation/submission_8b_reverse_iter10.json"
echo "  evaluation/submission_8b_reverse_iter20.json"
echo "  evaluation/submission_8b_reverse_iter30.json"
echo "  evaluation/submission_8b_reverse_iter40.json"
echo "  $SUMMARY_PATH"
echo "  $REPORT_PATH"
echo "========================================"