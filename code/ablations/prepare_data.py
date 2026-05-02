"""
Prepare multi-task SFT training data from the three allowed datasets:
    - openai/gsm8k (train split)
    - allenai/tulu-3-sft-mixture (filtered by source)
    - nvidia/OpenCodeInstruct (filtered for function-completion style)

Output: a single JSONL file of chat conversations, one conversation per line.
Each line: {"messages": [{"role": "user", "content": ...}, {"role": "assistant", "content": ...}], "task": "math|if|code"}

This is the ONLY data-prep step you need. The trainer (sft_train.py) reads
this JSONL directly — no more keyword scoring, no more mixture_presets.

Compliance:
    - Never trains on IFEval / GSM8K test / HumanEval. We filter by source
      and also do an n-gram decontamination pass against GSM8K test questions.
    - Only uses the three approved training datasets listed above.

Usage:
    python prepare_data.py \
        --out train.jsonl \
        --if_samples 20000 \
        --math_samples 15000 \
        --code_samples 8000 \
        --gsm8k_path   /path/to/gsm8k_train \
        --tulu_path    /path/to/tulu3_sft_train \
        --code_path    /path/to/opencodeinstruct_train

All local paths are optional — if omitted the script downloads from HF.
"""

import argparse
import json
import os
import random
import re
from collections import Counter, defaultdict

from datasets import load_dataset, load_from_disk


# -------------------- Decontamination helpers --------------------

def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9 ]", "", text)
    return text.strip()


def _ngrams(text: str, n: int = 10) -> set:
    toks = _normalize(text).split()
    if len(toks) < n:
        return {" ".join(toks)} if toks else set()
    return {" ".join(toks[i : i + n]) for i in range(len(toks) - n + 1)}


def build_decontamination_set(gsm8k_path: str | None) -> set:
    """Build a set of 10-grams from GSM8K TEST questions; any training sample
    whose question overlaps will be dropped. We never touch test answers."""
    print("[decontam] Loading GSM8K TEST split to build n-gram blocklist...")
    try:
        if gsm8k_path and os.path.isdir(gsm8k_path):
            # local cache might only have train split — fall back to HF for test
            raise FileNotFoundError("local path is train-only; using HF for test")
        ds = load_dataset("openai/gsm8k", "main", split="test")
    except Exception as e:
        print(f"[decontam] WARNING: could not load GSM8K test split ({e}). "
              f"Skipping decontamination — results may include leaked paraphrases.")
        return set()
    blocklist = set()
    for row in ds:
        blocklist |= _ngrams(row["question"], n=10)
    print(f"[decontam] built blocklist of {len(blocklist)} 10-grams from GSM8K test")
    return blocklist


def is_contaminated(text: str, blocklist: set) -> bool:
    if not blocklist:
        return False
    return bool(_ngrams(text, n=10) & blocklist)


# -------------------- Dataset loaders --------------------

def load_hf_or_local(hf_name: str, local_path: str | None, split: str = "train",
                    config: str | None = None, streaming: bool = False):
    if local_path and os.path.exists(local_path):
        print(f"[data] loading {hf_name} from local: {local_path}")
        ds = load_from_disk(local_path)
        if hasattr(ds, "keys") and split in ds:
            return ds[split]
        return ds
    print(f"[data] downloading {hf_name} from HuggingFace (streaming={streaming})")
    if config:
        return load_dataset(hf_name, config, split=split, streaming=streaming)
    return load_dataset(hf_name, split=split, streaming=streaming)


# -------------------- GSM8K (math) --------------------

def build_gsm8k_samples(num_samples: int, local_path: str | None) -> list:
    ds = load_hf_or_local("openai/gsm8k", local_path, split="train", config="main")
    out = []
    for row in ds:
        q, a = row["question"].strip(), row["answer"].strip()
        if not q or not a or "####" not in a:
            continue
        out.append({
            "messages": [
                {"role": "user", "content": q},
                {"role": "assistant", "content": a},
            ],
            "task": "math",
            "source": "gsm8k_train",
        })
    random.shuffle(out)
    return out[:num_samples]


# -------------------- Tulu 3 (IF + a bit of math/code persona) --------------------

# Source-based allocation (by fraction of the IF-target budget).
# Priorities are based on the Tülu 3 paper: persona_if > no_robots > flan_v2 > misc.
TULU_IF_QUOTA = {
    # strong IFEval signal — 拉到 55%
    "personahub_ifdata":      0.55,
    "tulu_hard_coded":        0.05,
    "no_robots":              0.15,
    # general instruction diversity
    "oasst1":                 0.05,
    "flan_v2":                0.08,
    "aya":                    0.02,
    # 保留一点 persona 用于泛化
    "personahub_math":        0.04,
    "personahub_code":        0.03,
    "numinamath":             0.03,
    # 砍掉 personas-math-grade 和 algebra,给 persona_if 腾空间
}
# Sources we explicitly exclude (low quality or risk).
TULU_DROP = {"wildjailbreak", "wildchat", "table_gpt", "hard_coded_repeated",
             "synthetic_finalresp", "sciriff", "evol_codealpaca"}

FORBIDDEN_NAME_PATTERNS = re.compile(
    r"ifeval|google/ifeval|gsm8k|openai/gsm8k|humaneval|openai_humaneval|human-eval",
    re.IGNORECASE,
)


def source_tag(raw: str) -> str:
    """Pick the first quota bucket that the raw source name contains."""
    r = (raw or "").lower()
    for tag in TULU_IF_QUOTA:
        if tag in r:
            return tag
    return "other"


def build_tulu_samples(num_samples: int, local_path: str | None,
                       blocklist: set) -> list:
    ds = load_hf_or_local("allenai/tulu-3-sft-mixture", local_path, split="train",
                          streaming=local_path is None)

    # Target counts per bucket
    targets = {k: max(1, int(round(num_samples * v))) for k, v in TULU_IF_QUOTA.items()}
    buckets: dict[str, list] = defaultdict(list)
    scanned = 0
    scan_cap = max(600_000, num_samples * 40)

    for row in ds:
        scanned += 1
        if scanned >= scan_cap:
            break
        src_raw = str(row.get("source", row.get("dataset", "")))
        lo = src_raw.lower()

        # drops
        if any(bad in lo for bad in TULU_DROP):
            continue
        if FORBIDDEN_NAME_PATTERNS.search(src_raw):
            continue

        tag = source_tag(src_raw)
        if tag == "other":
            continue
        if len(buckets[tag]) >= targets[tag]:
            continue

        msgs = row.get("messages") or []
        if len(msgs) < 2:
            continue
        # normalize to simple user/assistant turns
        conv = []
        for m in msgs:
            role = m.get("role", "")
            content = (m.get("content") or "").strip()
            if role in {"user", "assistant"} and content:
                conv.append({"role": role, "content": content})
        if len(conv) < 2 or conv[0]["role"] != "user" or conv[-1]["role"] != "assistant":
            continue

        user_text = conv[0]["content"]
        asst_text = conv[-1]["content"]
        if FORBIDDEN_NAME_PATTERNS.search(user_text + "\n" + asst_text):
            continue
        # decontaminate against GSM8K test
        if is_contaminated(user_text, blocklist):
            continue

        # sanity length filter
        if not (20 <= len(user_text) <= 4000) or not (20 <= len(asst_text) <= 6000):
            continue

        buckets[tag].append({
            "messages": conv,
            "task": "if",
            "source": f"tulu:{tag}",
        })

        if all(len(buckets[t]) >= targets[t] for t in targets):
            break

    out = []
    for tag, items in buckets.items():
        out.extend(items[:targets[tag]])
    random.shuffle(out)
    print(f"[tulu] scanned={scanned}, selected per source: {Counter(x['source'] for x in out)}")
    return out[:num_samples]


# -------------------- OpenCodeInstruct (code) --------------------

# Prefer function-completion style because that's what HumanEval scores.
CODE_PROMPT_HINTS = ("def ", "docstring", "complete the function", "implement the function",
                     "write a python function", "function signature", "pass@")


def is_function_completion_style(prompt: str, response: str) -> bool:
    p = prompt.lower()
    # must look like a function spec
    if not any(h in p for h in CODE_PROMPT_HINTS):
        return False
    # response should contain a def
    if "def " not in response:
        return False
    # avoid very long essays
    if len(response) > 5000 or len(prompt) > 3000:
        return False
    return True


def build_code_samples(num_samples: int, local_path: str | None,
                       blocklist: set) -> list:
    ds = load_hf_or_local("nvidia/OpenCodeInstruct", local_path, split="train",
                          streaming=local_path is None)
    out = []
    scanned = 0
    scan_cap = max(400_000, num_samples * 80)
    for row in ds:
        scanned += 1
        if scanned >= scan_cap or len(out) >= num_samples:
            break
        prompt = (row.get("input") or row.get("prompt") or row.get("question") or "").strip()
        response = (row.get("output") or row.get("response") or row.get("solution") or "").strip()
        if not prompt or not response:
            continue
        if FORBIDDEN_NAME_PATTERNS.search(prompt + "\n" + response):
            continue
        if is_contaminated(prompt, blocklist):
            continue
        if not is_function_completion_style(prompt, response):
            continue
        # Lightweight quality filter using the judgement_scores if present
        js = row.get("judgement") or row.get("judgement_scores")
        if isinstance(js, str):
            try:
                js = json.loads(js)
            except Exception:
                js = None
        if isinstance(js, dict):
            rc = js.get("requirement_conformance", {})
            if isinstance(rc, dict) and isinstance(rc.get("score"), (int, float)):
                if rc["score"] < 4:  # 5-point scale; skip weak ones
                    continue
        out.append({
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ],
            "task": "code",
            "source": "opencodeinstruct",
        })
    print(f"[code] scanned={scanned}, kept {len(out)}")
    return out


# -------------------- Main --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, required=True, help="Output JSONL path")
    ap.add_argument("--if_samples",   type=int, default=20000)
    ap.add_argument("--math_samples", type=int, default=15000)
    ap.add_argument("--code_samples", type=int, default=8000)
    ap.add_argument("--gsm8k_path", type=str, default=None)
    ap.add_argument("--tulu_path",  type=str, default=None)
    ap.add_argument("--code_path",  type=str, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    blocklist = build_decontamination_set(args.gsm8k_path)

    print("\n== GSM8K (math) ==")
    math_rows = build_gsm8k_samples(args.math_samples, args.gsm8k_path)
    print(f"  kept {len(math_rows)}")

    print("\n== Tulu3 (if + persona bleed) ==")
    if_rows = build_tulu_samples(args.if_samples, args.tulu_path, blocklist)
    print(f"  kept {len(if_rows)}")

    print("\n== OpenCodeInstruct (code) ==")
    code_rows = build_code_samples(args.code_samples, args.code_path, blocklist)
    print(f"  kept {len(code_rows)}")

    rows = math_rows + if_rows + code_rows
    random.shuffle(rows)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n[done] wrote {len(rows)} conversations to {args.out}")
    print(f"  task mix: {Counter(r['task'] for r in rows)}")


if __name__ == "__main__":
    main()
