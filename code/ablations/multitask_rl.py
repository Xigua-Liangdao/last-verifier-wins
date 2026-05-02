"""
Train a multi-task SFT model using only allowed training data.

This script keeps the original Tinker LoRA workflow intact while upgrading the
training recipe with stronger data filtering, configurable mixture presets,
curriculum-aware batching, checkpoint saving, and optional automatic evaluation
for checkpoint selection.
"""

import argparse
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from collections import defaultdict

import numpy as np
import tinker
from datasets import load_dataset, load_from_disk
from tinker import types

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(EVAL_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from evaluation.tokenizer_bootstrap import configure_local_tokenizers
from tinker_cookbook import model_info, renderers
from tinker_cookbook.hyperparam_utils import get_lr as cookbook_get_lr
from tinker_cookbook.supervised.data import conversation_to_datum
from tinker_cookbook.tokenizer_utils import get_tokenizer

LOCAL_LLAMA3_TOKENIZER_DIR = configure_local_tokenizers()

DEFAULT_CONFIG = {
    "model": "meta-llama/Llama-3.2-3B",
    "math_samples": 500,
    "if_samples": 1000,
    "code_samples": 500,
    "num_steps": 10,
    "batch_size": 4,
    "lr": 1e-4,
    "rank": 32,
    "max_length": 1024,
    "save_every": 0,
    "checkpoint_name": "real_sft",
    "mixture_preset": "balanced",
    "quick_eval_limit": 0,
    "full_eval_top_k": 0,
    "quick_eval_min_humaneval": 0.30,
    "results_path": os.path.join(EVAL_DIR, "experiment_results.json"),
    "lr_schedule": "cosine",
    "warmup_ratio": 0.1,
    "warmup_steps": None,
    "grad_accum_steps": 1,
    "use_cookbook_lr": False,
    "train_mlp": True,
    "train_attn": True,
    "train_unembed": True,
}

AUTO_LOCAL_DATA_PATHS = {
    "gsm8k": os.environ.get("GSM8K_LOCAL_PATH"),
    "if": os.environ.get("TULU_LOCAL_PATH"),
    "code": os.environ.get("CODE_LOCAL_PATH"),
}

PRESETS = {
    "dev": {
        "model": "meta-llama/Llama-3.2-3B",
        "math_samples": 500,
        "if_samples": 1000,
        "code_samples": 500,
        "num_steps": 300,
        "batch_size": 4,
        "lr": 2e-4,
        "rank": 64,
        "max_length": 1024,
        "save_every": 100,
        "checkpoint_name": "dev",
        "quick_eval_limit": 40,
        "full_eval_top_k": 0,
    },
    "medium": {
        "model": "meta-llama/Llama-3.2-3B",
        "math_samples": 3000,
        "if_samples": 6000,
        "code_samples": 3000,
        "num_steps": 3000,
        "batch_size": 4,
        "lr": 2e-4,
        "rank": 128,
        "max_length": 1536,
        "save_every": 300,
        "checkpoint_name": "medium",
        "quick_eval_limit": 50,
        "full_eval_top_k": 2,
    },
    "final": {
        "model": "meta-llama/Llama-3.2-3B",
        "math_samples": 7400,
        "if_samples": 15000,
        "code_samples": 8000,
        "num_steps": 6000,
        "batch_size": 4,
        "lr": 2e-4,
        "rank": 128,
        "max_length": 2048,
        "save_every": 500,
        "checkpoint_name": "final",
        "quick_eval_limit": 50,
        "full_eval_top_k": 2,
    },
}

MIXTURE_PRESETS = {
    "balanced": {
        "sample_multipliers": {"math": 1.0, "if": 1.0, "code": 1.2},
        "math_pool_transfer_fraction": 0.2,
        "batch_stages": [
            {"until": 1.0, "pattern": ["math", "if", "code", "if", "math", "code", "if", "code"]},
        ],
        "instruction": {
            "scan_multiplier": 30,
            "min_scan": 250000,
            "min_score": 6,
            "preferred_cap_ratio": 0.22,
            "neutral_cap_ratio": 0.12,
            "discouraged_cap_ratio": 0.06,
            "preferred_fraction": 0.8,
            "if_specialized_fraction": 0.45,
            "general_if_fraction": 0.25,
            "if_specialized_cap_ratio": 0.5,
            "general_if_cap_ratio": 0.18,
            "math_transfer_cap_ratio": 0.12,
        },
        "code": {
            "scan_multiplier": 120,
            "min_scan": 300000,
            "min_average_test_score": 1.0,
            "min_pass_rate": 1.0,
            "min_score": 13,
        },
    },
    "math_if_heavy": {
        "sample_multipliers": {"math": 1.3, "if": 1.2, "code": 1.25},
        "math_pool_transfer_fraction": 0.25,
        "batch_stages": [
            {"until": 1.0, "pattern": ["math", "if", "if", "math", "if", "code", "math", "if"]},
        ],
        "instruction": {
            "scan_multiplier": 40,
            "min_scan": 750000,
            "min_score": 9,
            "preferred_cap_ratio": 0.24,
            "neutral_cap_ratio": 0.10,
            "discouraged_cap_ratio": 0.05,
            "preferred_fraction": 0.85,
            "if_specialized_fraction": 0.50,
            "general_if_fraction": 0.25,
            "if_specialized_cap_ratio": 0.55,
            "general_if_cap_ratio": 0.18,
            "math_transfer_cap_ratio": 0.12,
        },
        "code": {
            "scan_multiplier": 140,
            "min_scan": 350000,
            "min_average_test_score": 1.0,
            "min_pass_rate": 1.0,
            "min_score": 15,
        },
    },
    "curriculum_math_if_then_mix_code": {
        "sample_multipliers": {"math": 1.25, "if": 1.15, "code": 1.35},
        "math_pool_transfer_fraction": 0.25,
        "batch_stages": [
            {"until": 0.35, "pattern": ["math", "if", "math", "if", "if", "math", "if", "code"]},
            {"until": 0.75, "pattern": ["math", "if", "if", "math", "code", "if", "math", "code"]},
            {"until": 1.0, "pattern": ["math", "if", "code", "if", "math", "code", "if", "code"]},
        ],
        "instruction": {
            "scan_multiplier": 45,
            "min_scan": 750000,
            "min_score": 9,
            "preferred_cap_ratio": 0.24,
            "neutral_cap_ratio": 0.10,
            "discouraged_cap_ratio": 0.05,
            "preferred_fraction": 0.85,
            "if_specialized_fraction": 0.50,
            "general_if_fraction": 0.25,
            "if_specialized_cap_ratio": 0.55,
            "general_if_cap_ratio": 0.18,
            "math_transfer_cap_ratio": 0.10,
        },
        "code": {
            "scan_multiplier": 150,
            "min_scan": 120000,
            "min_average_test_score": 1.0,
            "min_pass_rate": 1.0,
            "min_score": 15,
        },
    },
}

FORBIDDEN_BENCHMARK_PATTERNS = (
    r"ifeval",
    r"google/ifeval",
    r"openai/gsm8k",
    r"gsm8k",
    r"openai/openai_humaneval",
    r"openai_humaneval",
    r"humaneval",
    r"human-eval",
)

IF_KEYWORDS = [
    "exactly",
    "at least",
    "at most",
    "paragraphs",
    "sentences",
    "words",
    "bullet",
    "numbered",
    "list",
    "format",
    "include",
    "keyword",
    "do not",
    "must",
    "always",
    "never",
    "uppercase",
    "lowercase",
    "json",
    "markdown",
    "bold",
    "italic",
    "title",
    "heading",
    "end with",
    "start with",
    "contain",
    "repeat",
    "constraint",
    "requirement",
    "instruction",
    "follow",
]

IF_SOURCE_BONUSES = {
    "personahub_ifdata": 10,
    "personas-math-grade": 9,
    "personahub_math": 8,
    "algebra": 8,
    "python": 7,
    "math": 5,
    "sciriff": 4,
    "coconot": 3,
    "no_robots": 3,
    "flan": 2,
    "oasst1": 2,
    "aya": 1,
}

IF_SOURCE_PENALTIES = {
    "wildjailbreak": -12,
    "hard_coded_repeated": -12,
    "synthetic_finalresp": -7,
    "wildchat": -5,
    "table_gpt": -4,
    "evol_codealpaca": -5,
    "personahub_code": -4,
}

IF_EXCLUDED_SOURCE_PATTERNS = {
    "wildjailbreak",
    "hard_coded_repeated",
    "synthetic_finalresp",
    "wildchat",
    "table_gpt",
    "evol_codealpaca",
}

HIGH_VALUE_INSTRUCTION_SOURCE_PATTERNS = (
    "personahub_ifdata",
    "instruction_following",
    "persona_if",
    "personahub_math",
    "persona_math",
    "personas-math-grade",
    "algebra",
    "numinamath",
    "sciriff",
    "persona_code",
    "tulu_hard_coded",
    "no_robots",
    "flan_v2",
)

IF_SPECIALIZED_SOURCE_PATTERNS = (
    "personahub_ifdata",
    "instruction_following",
    "persona_if",
)

GENERAL_IF_SOURCE_PATTERNS = (
    "no_robots",
    "flan_v2",
    "oasst1",
    "aya",
)

MATH_TRANSFER_SOURCE_PATTERNS = (
    "personahub_math",
    "persona_math",
    "personas-math-grade",
    "algebra",
    "numinamath",
    "sciriff",
    "coconot",
)

CODE_DOMAIN_BONUSES = {
    "generic": 6,
    "python": 6,
    "algorithmic": 5,
    "coding": 4,
    "code": 4,
}

CODE_PROMPT_KEYWORDS = [
    "python function",
    "implement",
    "write a function",
    "complete the function",
    "return",
    "given",
    "you are given",
    "docstring",
    "list",
    "string",
    "integer",
    "tree",
    "linked list",
]

HUMANEVAL_STYLE_PROMPT_PATTERNS = [
    "def ",
    '"""',
    "'''",
    "docstring",
    "complete the function",
    "python function",
]


def is_offline_mode() -> bool:
    return any(
        os.environ.get(name, "").lower() in {"1", "true", "yes"}
        for name in ["HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"]
    )


def dataset_cache_kwargs() -> dict:
    cache_dir = os.environ.get("HF_HOME")
    return {"cache_dir": cache_dir} if cache_dir else {}


def load_dataset_split(path, split, *, name=None, streaming=False):
    kwargs = dataset_cache_kwargs()
    try:
        return load_dataset(path, name, split=split, streaming=streaming, **kwargs)
    except Exception as exc:
        mode = "offline cache" if is_offline_mode() else "Hugging Face"
        raise RuntimeError(
            f"Failed to load dataset {path} ({split}) from {mode}. "
            f"If you are offline, provide a local path or pre-cache the dataset. Original error: {exc}"
        ) from exc


def load_local_dataset(local_path):
    if os.path.isdir(local_path):
        dataset = load_from_disk(local_path)
        if hasattr(dataset, "keys") and "train" in dataset:
            return dataset["train"]
        return dataset
    if local_path.endswith((".json", ".jsonl")):
        return load_dataset(
            "json",
            data_files=local_path,
            split="train",
            **dataset_cache_kwargs(),
        )
    raise ValueError(
        f"Unsupported local dataset path: {local_path}. Expected a load_from_disk directory or a .json/.jsonl file."
    )


def maybe_shuffle_dataset(dataset, seed=42):
    shuffle_fn = getattr(dataset, "shuffle", None)
    if not callable(shuffle_fn):
        return dataset
    try:
        return shuffle_fn(seed=seed)
    except TypeError:
        return shuffle_fn()


def resolve_local_dataset_path(explicit_path, key):
    if explicit_path:
        return explicit_path
    auto_path = AUTO_LOCAL_DATA_PATHS.get(key)
    if auto_path and os.path.exists(auto_path):
        return auto_path
    return None


def contains_forbidden_benchmark_reference(*texts) -> bool:
    combined = "\n".join(text for text in texts if text).lower()
    return any(re.search(pattern, combined) for pattern in FORBIDDEN_BENCHMARK_PATTERNS)


def safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_test_statuses(value):
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).lower() for item in value]
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = json.loads(stripped)
                if isinstance(parsed, list):
                    return [str(item).lower() for item in parsed]
            except json.JSONDecodeError:
                pass
        return [stripped.lower()]
    return [str(value).lower()]


def compute_pass_rate(statuses):
    if not statuses:
        return None
    passed = sum(status == "pass" for status in statuses)
    return passed / max(len(statuses), 1)


def strip_code_fences(text):
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if len(lines) >= 3 and lines[0].startswith("```") and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return stripped


def parse_judgement_scores(value):
    if not value:
        return {}
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return {}
    if not isinstance(value, dict):
        return {}

    scores = {}
    for key in ["requirement_conformance", "logical_correctness", "readability", "efficiency"]:
        entry = value.get(key)
        if isinstance(entry, dict):
            numeric = safe_float(entry.get("score"))
            if numeric is not None:
                scores[key] = numeric
    return scores


def score_math_example(question, answer):
    score = 0
    if "####" in answer:
        score += 5
    score += min(answer.count("\n"), 5)
    if "<<" in answer and ">>" in answer:
        score += 2
    if 80 <= len(question) <= 420:
        score += 2
    if 80 <= len(answer) <= 650:
        score += 2
    if any(char.isdigit() for char in question):
        score += 1
    return score


def load_gsm8k_ranked_examples(local_path=None):
    dataset = (
        load_local_dataset(local_path)
        if local_path
        else load_dataset_split("openai/gsm8k", "train", name="main")
    )

    ranked = []
    for item in dataset:
        question = item["question"].strip()
        answer = item["answer"].strip()
        if not question or not answer or "####" not in answer:
            continue
        ranked.append(
            (
                score_math_example(question, answer),
                random.random(),
                [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer},
                ],
            )
        )
    return ranked


def score_math_transfer_conversation(conversation, source):
    question = conversation[0]["content"]
    answer = conversation[-1]["content"]
    score = 0
    lowered_source = source.lower()
    if any(pattern in lowered_source for pattern in MATH_TRANSFER_SOURCE_PATTERNS):
        score += 8
    if "####" in answer:
        score += 4
    if any(token in answer for token in ["Therefore", "Thus", "So ", "Answer"]):
        score += 2
    if any(char.isdigit() for char in question):
        score += 2
    if 80 <= len(question) <= 600:
        score += 2
    if 80 <= len(answer) <= 1800:
        score += 2
    if len(conversation) >= 2:
        score += 1
    return score


def load_math_transfer_examples(num_samples=1200, local_path=None, scan_limit=None):
    if num_samples <= 0 or not local_path:
        return []
    dataset = maybe_shuffle_dataset(load_local_dataset(local_path), seed=123)
    ranked = []
    scanned = 0
    max_scan = scan_limit if scan_limit is not None else max(750000, num_samples * 60)
    for item in dataset:
        if scanned >= max_scan:
            break
        scanned += 1
        source = item.get("source", item.get("dataset", "unknown"))
        lowered_source = source.lower()
        if not any(pattern in lowered_source for pattern in MATH_TRANSFER_SOURCE_PATTERNS):
            continue
        messages = item.get("messages", [])
        if not messages or len(messages) < 2:
            continue
        conversation = []
        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")
            if role in {"user", "assistant"} and content.strip():
                conversation.append({"role": role, "content": content.strip()})
        if len(conversation) < 2:
            continue
        if conversation[0]["role"] != "user" or conversation[-1]["role"] != "assistant":
            continue
        if contains_forbidden_benchmark_reference(source, conversation[0]["content"], conversation[-1]["content"]):
            continue
        quality = score_math_transfer_conversation(conversation, source)
        ranked.append((quality, random.random(), conversation))
    ranked.sort(key=lambda entry: (entry[0], entry[1]), reverse=True)
    conversations = [conversation for _, _, conversation in ranked[:num_samples]]
    print(f"  -> {len(conversations)} math-transfer examples loaded (from {scanned} scanned)")
    return conversations


def load_math_examples(num_samples=7400, gsm8k_local_path=None, if_local_path=None, mixture_preset="balanced"):
    print(f"[DATA] Loading math data (target: {num_samples})...")
    settings = MIXTURE_PRESETS[mixture_preset]
    transfer_target = int(round(num_samples * settings.get("math_pool_transfer_fraction", 0.0)))
    transfer_target = min(max(0, transfer_target), num_samples)
    gsm8k_target = max(0, num_samples - transfer_target)

    gsm8k_ranked = load_gsm8k_ranked_examples(gsm8k_local_path)
    gsm8k_ranked.sort(key=lambda entry: (entry[0], entry[1]), reverse=True)
    conversations = [conversation for _, _, conversation in gsm8k_ranked[:gsm8k_target]]
    instruction_scan_limit = MIXTURE_PRESETS[mixture_preset]["instruction"]["min_scan"]
    transfer_examples = load_math_transfer_examples(transfer_target, if_local_path, scan_limit=instruction_scan_limit)
    conversations.extend(transfer_examples)
    if len(conversations) < num_samples:
        backfill_needed = num_samples - len(conversations)
        used_keys = {(conv[0]["content"], conv[-1]["content"]) for conv in conversations}
        for _, _, conversation in gsm8k_ranked[gsm8k_target:]:
            key = (conversation[0]["content"], conversation[-1]["content"])
            if key in used_keys:
                continue
            conversations.append(conversation)
            used_keys.add(key)
            backfill_needed -= 1
            if backfill_needed <= 0:
                break
    print(f"  -> {len(conversations)} total math examples loaded ({len(conversations) - len(transfer_examples)} GSM8K + {len(transfer_examples)} transfer)")
    return conversations[:num_samples]


def instruction_source_score(source):
    lowered = source.lower()
    score = 0
    for key, value in IF_SOURCE_BONUSES.items():
        if key in lowered:
            score += value
    for key, value in IF_SOURCE_PENALTIES.items():
        if key in lowered:
            score += value
    return score


def instruction_source_priority(source):
    lowered = source.lower()
    if any(pattern in lowered for pattern in HIGH_VALUE_INSTRUCTION_SOURCE_PATTERNS):
        return 2
    if any(pattern in lowered for pattern in ("oasst1", "coconot", "aya")):
        return 1
    return 0


def instruction_source_bucket(source):
    lowered = source.lower()
    if any(pattern in lowered for pattern in IF_SPECIALIZED_SOURCE_PATTERNS):
        return "if_specialized"
    if any(pattern in lowered for pattern in GENERAL_IF_SOURCE_PATTERNS):
        return "general_if"
    if any(pattern in lowered for pattern in MATH_TRANSFER_SOURCE_PATTERNS):
        return "math_transfer"
    return "fallback"


def instruction_score(conversation, source=""):
    user_text = conversation[0]["content"].lower()
    assistant_text = conversation[-1]["content"]
    priority = instruction_source_priority(source)
    source_score = instruction_source_score(source)
    score = sum(1 for keyword in IF_KEYWORDS if keyword in user_text)
    if len(user_text) > 120:
        score += 2
    if len(user_text) > 260:
        score += 2
    if len(user_text) > 500:
        score += 1
    if 80 <= len(assistant_text) <= 2200:
        score += 2
    elif len(assistant_text) < 40:
        score -= 4
    elif len(assistant_text) > 3500:
        score -= 2
    if len(conversation) >= 4:
        score += 1
    if any(token in user_text for token in ["json", "markdown", "bullet", "numbered", "exactly"]):
        score += 3
    if priority >= 2:
        score += 6
    elif priority == 1:
        score += 2
    if source_score >= 8:
        score += 3
    if not any(keyword in user_text for keyword in IF_KEYWORDS) and len(user_text) < 60 and priority == 0:
        score -= 5
    score += source_score
    return score


def should_exclude_instruction_source(source):
    lowered = source.lower()
    return any(pattern in lowered for pattern in IF_EXCLUDED_SOURCE_PATTERNS)


def instruction_source_cap(num_samples, source, settings):
    source_score_value = instruction_source_score(source)
    bucket = instruction_source_bucket(source)
    if bucket == "if_specialized":
        ratio = settings["if_specialized_cap_ratio"]
    elif bucket == "general_if":
        ratio = settings["general_if_cap_ratio"]
    elif bucket == "math_transfer":
        ratio = settings["math_transfer_cap_ratio"]
    elif source_score_value >= 8:
        ratio = settings["preferred_cap_ratio"]
    elif source_score_value <= -4:
        ratio = settings["discouraged_cap_ratio"]
    else:
        ratio = settings["neutral_cap_ratio"]
    return max(12, int(num_samples * ratio))


def load_instruction_examples(num_samples=15000, local_path=None, mixture_preset="balanced"):
    print(f"[DATA] Loading instruction-following data (target: {num_samples})...")
    dataset = (
        load_local_dataset(local_path)
        if local_path
        else load_dataset_split(
            "allenai/tulu-3-sft-mixture",
            "train",
            streaming=not is_offline_mode(),
        )
    )
    dataset = maybe_shuffle_dataset(dataset, seed=42)

    settings = MIXTURE_PRESETS[mixture_preset]["instruction"]
    scan_limit = max(settings["min_scan"], num_samples * settings["scan_multiplier"])
    candidates_by_source = defaultdict(list)
    scanned = 0

    for item in dataset:
        if scanned >= scan_limit:
            break
        scanned += 1

        messages = item.get("messages", [])
        if not messages or len(messages) < 2:
            continue

        source = item.get("source", item.get("dataset", "unknown"))
        if contains_forbidden_benchmark_reference(source) or should_exclude_instruction_source(source):
            continue

        conversation = []
        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")
            if role in {"user", "assistant"} and content.strip():
                conversation.append({"role": role, "content": content.strip()})

        if len(conversation) < 2:
            continue
        if conversation[0]["role"] != "user" or conversation[-1]["role"] != "assistant":
            continue
        if contains_forbidden_benchmark_reference(
            source,
            conversation[0]["content"],
            conversation[-1]["content"],
        ):
            continue

        quality = instruction_score(conversation, source)
        if quality < settings["min_score"]:
            continue
        candidates_by_source[source].append((quality, random.random(), conversation))

    if not candidates_by_source:
        raise RuntimeError("No instruction examples passed filtering; relax IF source filters or increase scan budget.")

    selected = []
    selected_source_counts = {}
    preferred_selected = []
    fallback_selected = []
    for source, items in candidates_by_source.items():
        items.sort(key=lambda entry: (entry[0], entry[1]), reverse=True)
        cap = instruction_source_cap(num_samples, source, settings)
        chosen = items[:cap]
        bucket = preferred_selected if instruction_source_priority(source) >= 2 else fallback_selected
        bucket.extend((score, tie_breaker, source, conversation) for score, tie_breaker, conversation in chosen)
        selected_source_counts[source] = len(chosen)

    preferred_selected.sort(key=lambda entry: (entry[0], entry[1]), reverse=True)
    fallback_selected.sort(key=lambda entry: (entry[0], entry[1]), reverse=True)

    bucketed_preferred = defaultdict(list)
    for entry in preferred_selected:
        bucketed_preferred[instruction_source_bucket(entry[2])].append(entry)

    if_specialized_target = max(1, int(num_samples * settings["if_specialized_fraction"]))
    general_if_target = max(1, int(num_samples * settings["general_if_fraction"]))

    selected.extend(bucketed_preferred["if_specialized"][:if_specialized_target])
    selected.extend(bucketed_preferred["general_if"][:general_if_target])

    selected_keys = {(entry[2], id(entry[3])) for entry in selected}
    remaining_preferred = [
        entry for entry in preferred_selected if (entry[2], id(entry[3])) not in selected_keys
    ]
    preferred_target = min(
        len(preferred_selected),
        max(1, int(num_samples * settings["preferred_fraction"])),
    )
    if len(selected) < preferred_target:
        selected.extend(remaining_preferred[: preferred_target - len(selected)])
        selected_keys = {(entry[2], id(entry[3])) for entry in selected}

    if len(selected) < num_samples:
        selected.extend(
            entry for entry in remaining_preferred if (entry[2], id(entry[3])) not in selected_keys
        )
        selected = selected[:num_samples]
        selected_keys = {(entry[2], id(entry[3])) for entry in selected}

    if len(selected) < num_samples:
        selected.extend(
            entry for entry in fallback_selected if (entry[2], id(entry[3])) not in selected_keys
        )
        selected = selected[:num_samples]

    selected.sort(key=lambda entry: (entry[0], entry[1]), reverse=True)
    conversations = [conversation for _, _, _, conversation in selected[:num_samples]]
    final_source_counts = defaultdict(int)
    for _, _, source, _ in selected[:num_samples]:
        final_source_counts[source] += 1

    print(f"  -> {len(conversations)} instruction examples loaded after scanning {scanned}")
    top_sources = sorted(final_source_counts.items(), key=lambda item: item[1], reverse=True)[:10]
    print(f"  Selected sources: {dict(top_sources)}")
    return conversations


def code_prompt_score(question):
    lowered = question.lower()
    score = sum(2 for keyword in CODE_PROMPT_KEYWORDS if keyword in lowered)
    if any(pattern in question for pattern in HUMANEVAL_STYLE_PROMPT_PATTERNS):
        score += 8
    if "def " in question and ('"""' in question or "'''" in question):
        score += 8
    if "python" in lowered:
        score += 2
    if "class" in lowered:
        score += 1
    return score


def is_function_completion_style(question, answer):
    lowered = question.lower()
    stripped_answer = answer.lstrip()
    if not stripped_answer.startswith(("def ", "class ")):
        return False
    if any(
        lowered.startswith(prefix)
        for prefix in (
            "explain",
            "describe",
            "what does",
            "analyze",
            "review",
            "debug",
            "why does",
        )
    ):
        return False
    if "```" in answer:
        return False
    if any(phrase in answer.lower() for phrase in ["here is", "here's", "explanation", "solution:"]):
        return False
    return (
        "def " in question
        or '"""' in question
        or "'''" in question
        or "write a function" in lowered
        or "implement" in lowered
        or "complete the function" in lowered
    )


def code_quality_score(question, answer, domain, average_test_score, pass_rate, judgement_scores, generation_algorithm):
    score = code_prompt_score(question)
    score += CODE_DOMAIN_BONUSES.get(domain, -8)
    if is_function_completion_style(question, answer):
        score += 10
    if answer.lstrip().startswith("def "):
        score += 6
    if answer.count("def ") == 1:
        score += 4
    elif answer.count("def ") > 1:
        score += 1
    if "```" in answer:
        score -= 4
    if any(phrase in answer.lower() for phrase in ["here is", "here's", "explanation", "solution:"]):
        score -= 3
    if "class " in answer:
        score += 1
    if 60 <= len(answer) <= 1800:
        score += 3
    elif len(answer) > 2600:
        score -= 3
    if average_test_score is not None:
        score += int(round(average_test_score * 10))
    if pass_rate is not None:
        score += int(round(pass_rate * 8))
    if generation_algorithm in {"self-instruct", "evol-instruct"}:
        score += 2
    if judgement_scores:
        score += int(round(sum(judgement_scores.values()) / len(judgement_scores)))
    return score


def load_code_examples(num_samples=8000, local_path=None, mixture_preset="balanced"):
    print(f"[DATA] Loading code data (target: {num_samples})...")
    dataset = (
        load_local_dataset(local_path)
        if local_path
        else load_dataset_split(
            "nvidia/OpenCodeInstruct",
            "train",
            streaming=not is_offline_mode(),
        )
    )

    settings = MIXTURE_PRESETS[mixture_preset]["code"]
    max_scan = max(settings["min_scan"], num_samples * settings["scan_multiplier"])
    ranked = []
    scanned = 0

    for item in dataset:
        if scanned >= max_scan:
            break
        scanned += 1

        question = item.get("question", item.get("instruction", item.get("prompt", item.get("input", ""))))
        answer = item.get("answer", item.get("response", item.get("output", "")))
        domain = str(item.get("domain", "")).lower()
        average_test_score = safe_float(item.get("average_test_score"))
        tests_execution_status = normalize_test_statuses(item.get("tests_execution_status"))
        pass_rate = compute_pass_rate(tests_execution_status)
        generation_algorithm = str(item.get("generation_algorithm", "")).lower()
        judgement_scores = parse_judgement_scores(item.get("llm_judgement"))

        if not question or not answer:
            continue

        question = question.strip()
        answer = strip_code_fences(answer)
        if contains_forbidden_benchmark_reference(item.get("source", ""), item.get("dataset", ""), question):
            continue
        if len(answer) < 40 or len(answer) > 4000:
            continue
        if "def " not in answer and "class " not in answer:
            continue
        if domain not in CODE_DOMAIN_BONUSES:
            continue
        if not is_function_completion_style(question, answer):
            continue
        if average_test_score is not None and average_test_score < settings["min_average_test_score"]:
            continue
        if pass_rate is not None and pass_rate < settings["min_pass_rate"]:
            continue

        quality = code_quality_score(
            question=question,
            answer=answer,
            domain=domain,
            average_test_score=average_test_score,
            pass_rate=pass_rate,
            judgement_scores=judgement_scores,
            generation_algorithm=generation_algorithm,
        )
        if quality < settings["min_score"]:
            continue

        ranked.append(
            (
                quality,
                random.random(),
                [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer},
                ],
            )
        )

    ranked.sort(key=lambda entry: (entry[0], entry[1]), reverse=True)
    conversations = [conversation for _, _, conversation in ranked[:num_samples]]
    print(f"  -> {len(conversations)} code examples loaded (from {scanned} scanned)")
    return conversations


def build_training_conversations(math_examples, instruction_examples, code_examples):
    return {
        "math": math_examples,
        "if": instruction_examples,
        "code": code_examples,
    }


def prepare_training_data(conversation_groups, renderer, max_length):
    print("[PREP] Tokenizing and organizing training data...")
    all_items = []
    stats = {"math": 0, "if": 0, "code": 0, "skipped": 0}

    for label, conversations in conversation_groups.items():
        for conversation in conversations:
            try:
                datum = conversation_to_datum(
                    conversation,
                    renderer,
                    max_length=max_length,
                    train_on_what=renderers.TrainOnWhat.ALL_ASSISTANT_MESSAGES,
                )
                all_items.append((datum, label))
                stats[label] += 1
            except Exception:
                stats["skipped"] += 1

    print(
        "  -> "
        f"math={stats['math']} if={stats['if']} code={stats['code']} skipped={stats['skipped']}"
    )
    return all_items


def get_stage_pattern(mixture_preset, step, num_steps):
    schedule = MIXTURE_PRESETS[mixture_preset]["batch_stages"]
    progress = (step + 1) / max(num_steps, 1)
    for stage in schedule:
        if progress <= stage["until"]:
            return stage["pattern"]
    return schedule[-1]["pattern"]


def get_balanced_batch(data_by_task, cursors, batch_size, step, num_steps, mixture_preset):
    preferred_order = get_stage_pattern(mixture_preset, step, num_steps)
    available_labels = [label for label, items in data_by_task.items() if items]
    if not available_labels:
        raise ValueError("No task data available for batching")

    selected_labels = []
    offset = step % len(preferred_order)
    index = 0
    while len(selected_labels) < batch_size:
        label = preferred_order[(offset + index) % len(preferred_order)]
        if label in available_labels:
            selected_labels.append(label)
        index += 1
        if index > len(preferred_order) * 6 and len(selected_labels) < batch_size:
            selected_labels.extend(available_labels)

    selected_labels = selected_labels[:batch_size]
    batch = []
    for label in selected_labels:
        items = data_by_task[label]
        cursor = cursors[label]
        batch.append(items[cursor % len(items)])
        cursors[label] = (cursor + 1) % len(items)
        if cursors[label] == 0:
            random.shuffle(items)

    return batch, selected_labels


def compute_loss(forward_backward_result, batch):
    logprobs = np.concatenate(
        [output["logprobs"].tolist() for output in forward_backward_result.loss_fn_outputs]
    )
    weights = np.concatenate([datum.loss_fn_inputs["weights"].tolist() for datum in batch])
    return -np.dot(logprobs, weights) / max(weights.sum(), 1)


def summarize_submission(submission):
    ifeval = safe_float(submission.get("ifeval", {}).get("metrics", {}).get("google/IFEval/final_acc")) or 0.0
    gsm8k = safe_float(submission.get("gsm8k", {}).get("metrics", {}).get("openai/gsm8k/accuracy")) or 0.0
    humaneval = safe_float(submission.get("humaneval", {}).get("metrics", {}).get("openai/openai_humaneval/accuracy")) or 0.0
    return {
        "ifeval": ifeval,
        "gsm8k": gsm8k,
        "humaneval": humaneval,
        "avg": (ifeval + gsm8k + humaneval) / 3.0,
    }


def summarize_metrics(submission):
    metrics = {}
    for task_name in ["ifeval", "gsm8k", "humaneval"]:
        metrics.update(submission.get(task_name, {}).get("metrics", {}))
    return metrics


def print_results_table(results, title):
    if not results:
        return
    print(f"\n{title}")
    print("| stage | step | IFEval | GSM8K | HumanEval | Avg |")
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    for result in results:
        scores = result["scores"]
        print(
            "| {stage} | {step} | {ifeval:.1f} | {gsm8k:.1f} | {humaneval:.1f} | {avg:.1f} |".format(
                stage=result["stage"],
                step=result["step"],
                ifeval=scores["ifeval"] * 100,
                gsm8k=scores["gsm8k"] * 100,
                humaneval=scores["humaneval"] * 100,
                avg=scores["avg"] * 100,
            )
        )


def evaluate_checkpoint(*, checkpoint_path, base_model, step, stage, limit, checkpoint_name, temperature=0.0, top_p=1.0):
    results_dir = os.path.join(EVAL_DIR, "auto_eval_results")
    os.makedirs(results_dir, exist_ok=True)
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{checkpoint_name}_{stage}_step{step}")
    output_path = os.path.join(results_dir, f"{safe_name}.json")

    command = [
        sys.executable,
        os.path.join(EVAL_DIR, "eval_all.py"),
        "--checkpoint_path",
        checkpoint_path,
        "--base_model",
        base_model,
        "--temperature",
        str(temperature),
        "--top_p",
        str(top_p),
        "--output_path",
        output_path,
    ]
    if limit is not None:
        command.extend(["--limit", str(limit)])

    print(f"\n[EVAL] {' '.join(command)}")
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    with open(output_path) as handle:
        submission = json.load(handle)

    return {
        "checkpoint": checkpoint_path,
        "step": step,
        "stage": stage,
        "limit": limit,
        "temperature": temperature,
        "top_p": top_p,
        "output_path": output_path,
        "scores": summarize_submission(submission),
        "metrics": summarize_metrics(submission),
    }


def evaluate_saved_checkpoints(model_name, checkpoint_records, config):
    if not checkpoint_records:
        return None
    if config["quick_eval_limit"] <= 0 and config["full_eval_top_k"] <= 0:
        return None

    results_payload = {
        "config": {
            "model": model_name,
            "checkpoint_name": config["checkpoint_name"],
            "mixture_preset": config["mixture_preset"],
            "quick_eval_limit": config["quick_eval_limit"],
            "full_eval_top_k": config["full_eval_top_k"],
            "quick_eval_min_humaneval": config["quick_eval_min_humaneval"],
        },
        "checkpoints": checkpoint_records,
        "quick_results": [],
        "full_results": [],
        "best_checkpoint_by_eval": None,
    }

    if config["quick_eval_limit"] > 0:
        for record in checkpoint_records:
            results_payload["quick_results"].append(
                evaluate_checkpoint(
                    checkpoint_path=record["path"],
                    base_model=model_name,
                    step=record["step"],
                    stage="quick",
                    limit=config["quick_eval_limit"],
                    checkpoint_name=config["checkpoint_name"],
                )
            )
        results_payload["quick_results"].sort(
            key=lambda result: (
                result["scores"]["avg"],
                result["scores"]["humaneval"],
                result["scores"]["gsm8k"],
                result["scores"]["ifeval"],
            ),
            reverse=True,
        )
        print_results_table(results_payload["quick_results"], "[EVAL] Quick screening results")

    if config["full_eval_top_k"] > 0:
        quick_candidates = results_payload["quick_results"] or [
            {
                "checkpoint": record["path"],
                "step": record["step"],
                "scores": {"ifeval": 0.0, "gsm8k": 0.0, "humaneval": 0.0, "avg": 0.0},
            }
            for record in checkpoint_records
        ]
        viable_candidates = [
            result
            for result in quick_candidates
            if result["scores"]["humaneval"] >= config["quick_eval_min_humaneval"]
        ]
        shortlist = (viable_candidates or quick_candidates)[: config["full_eval_top_k"]]
        for result in shortlist:
            results_payload["full_results"].append(
                evaluate_checkpoint(
                    checkpoint_path=result["checkpoint"],
                    base_model=model_name,
                    step=result["step"],
                    stage="full",
                    limit=None,
                    checkpoint_name=config["checkpoint_name"],
                )
            )
        results_payload["full_results"].sort(
            key=lambda result: (
                result["scores"]["avg"],
                result["scores"]["humaneval"],
                result["scores"]["gsm8k"],
                result["scores"]["ifeval"],
            ),
            reverse=True,
        )
        print_results_table(results_payload["full_results"], "[EVAL] Full evaluation results")

    best_pool = results_payload["full_results"] or results_payload["quick_results"]
    if best_pool:
        results_payload["best_checkpoint_by_eval"] = best_pool[0]

    with open(config["results_path"], "w") as handle:
        json.dump(results_payload, handle, indent=2)
    print(f"[EVAL] Structured results saved to {config['results_path']}")
    return results_payload


def write_training_metadata(
    *,
    model_name,
    renderer_name,
    config,
    seed,
    offline_mode,
    gsm8k_train_path,
    if_data_path,
    code_data_path,
    checkpoint_records,
    loss_history,
    experiment_results=None,
):
    if not checkpoint_records:
        return

    final_checkpoint = checkpoint_records[-1]
    best_checkpoint = min(checkpoint_records, key=lambda record: record["loss"])

    checkpoint_info = {
        "checkpoint_path": final_checkpoint["path"],
        "state_path": final_checkpoint.get("state_path"),
        "base_model": model_name,
        "renderer_name": renderer_name,
        "training": {
            "num_steps": config["num_steps"],
            "batch_size": config["batch_size"],
            "learning_rate": config["lr"],
            "lora_rank": config["rank"],
            "max_length": config["max_length"],
            "math_samples": config["math_samples"],
            "if_samples": config["if_samples"],
            "code_samples": config["code_samples"],
            "save_every": config["save_every"],
            "seed": seed,
            "mixture_preset": config["mixture_preset"],
            "train_mlp": config["train_mlp"],
            "train_attn": config["train_attn"],
            "train_unembed": config["train_unembed"],
        },
        "published": final_checkpoint["published"],
        "checkpoints": checkpoint_records,
        "best_checkpoint_by_loss": best_checkpoint,
    }
    if experiment_results and experiment_results.get("best_checkpoint_by_eval"):
        checkpoint_info["best_checkpoint_by_eval"] = experiment_results["best_checkpoint_by_eval"]

    checkpoint_info_path = os.path.join(EVAL_DIR, "checkpoint_info.json")
    with open(checkpoint_info_path, "w") as handle:
        json.dump(checkpoint_info, handle, indent=2)

    training_info = {
        "config": {
            **config,
            "seed": seed,
            "offline_mode": offline_mode,
            "gsm8k_train_path": gsm8k_train_path,
            "if_data_path": if_data_path,
            "code_data_path": code_data_path,
        },
        "checkpoints": checkpoint_records,
        "final_loss": float(sum(loss_history[-100:]) / len(loss_history[-100:])) if loss_history else None,
    }
    if experiment_results:
        training_info["evaluation"] = experiment_results

    training_info_path = os.path.join(EVAL_DIR, "training_info.json")
    with open(training_info_path, "w") as handle:
        json.dump(training_info, handle, indent=2)


def train(
    model_name,
    renderer_name,
    config,
    seed,
    offline_mode,
    gsm8k_train_path,
    if_data_path,
    code_data_path,
    training_data,
    num_steps,
    batch_size,
    lr,
    rank,
    grad_accum_steps,
    checkpoint_name,
    save_every,
    publish,
    publish_intermediate,
    resume_from_state=None,
    train_mlp=True,
    train_attn=True,
    train_unembed=True,
):
    service_client = tinker.ServiceClient()
    if resume_from_state:
        if "/sampler_weights/" in resume_from_state:
            raise ValueError(
                "resume_from_state must be a training state path, not a sampler_weights path. "
                "Use the saved state_path from checkpoint metadata."
            )
        print(f"Creating training client from state: {resume_from_state}")
        training_client = service_client.create_training_client_from_state_with_optimizer(path=resume_from_state)
    else:
        print(
            f"Creating LoRA training client (rank={rank}, "
            f"train_attn={train_attn}, train_mlp={train_mlp}, train_unembed={train_unembed})..."
        )
        training_client = service_client.create_lora_training_client(
            base_model=model_name,
            rank=rank,
            train_mlp=train_mlp,
            train_attn=train_attn,
            train_unembed=train_unembed,
        )
    print("  Training client ready")

    data_by_task = {}
    for datum, label in training_data:
        data_by_task.setdefault(label, []).append(datum)
    for items in data_by_task.values():
        random.shuffle(items)
    cursors = {label: 0 for label in data_by_task}

    checkpoint_every = save_every if save_every and save_every > 0 else num_steps
    checkpoint_records = []
    loss_history = []

    print(
        f"\nTraining for {num_steps} steps "
        f"(batch_size={batch_size}, grad_accum={grad_accum_steps}, "
        f"effective_batch={batch_size * max(grad_accum_steps, 1)}, lr={lr}, schedule={config['lr_schedule']}, "
        f"save_every={checkpoint_every}, mixture={config['mixture_preset']})..."
    )
    start_time = time.time()

    for step in range(num_steps):
        warmup_steps = config["warmup_steps"]
        if warmup_steps is None:
            warmup_steps = min(100, max(1, int(num_steps * config["warmup_ratio"])))
        if config["lr_schedule"] == "constant":
            current_lr = lr
        else:
            if step < warmup_steps:
                current_lr = lr * (step + 1) / max(warmup_steps, 1)
            else:
                progress = (step - warmup_steps) / max(num_steps - warmup_steps, 1)
                current_lr = lr * 0.5 * (1 + math.cos(math.pi * progress))
        adam_params = types.AdamParams(learning_rate=current_lr, beta1=0.9, beta2=0.95, eps=1e-8)

        micro_losses = []
        micro_labels = []
        for micro_step in range(max(grad_accum_steps, 1)):
            batch, batch_labels = get_balanced_batch(
                data_by_task,
                cursors,
                batch_size,
                step * max(grad_accum_steps, 1) + micro_step,
                num_steps * max(grad_accum_steps, 1),
                config["mixture_preset"],
            )
            forward_backward_result = training_client.forward_backward(batch, loss_fn="cross_entropy").result()
            micro_losses.append(compute_loss(forward_backward_result, batch))
            micro_labels.extend(batch_labels)

        optim_future = training_client.optim_step(adam_params)
        optim_future.result()

        loss = float(sum(micro_losses) / len(micro_losses))
        loss_history.append(float(loss))

        if (step + 1) % 10 == 0 or step == 0:
            elapsed = time.time() - start_time
            avg_loss = sum(loss_history[-50:]) / len(loss_history[-50:])
            steps_per_second = (step + 1) / max(elapsed, 1e-6)
            print(
                f"  Step {step + 1}/{num_steps} | Loss: {loss:.4f} | Avg50: {avg_loss:.4f} "
                f"| LR: {current_lr:.2e} | Speed: {steps_per_second:.2f} steps/s | Batch: {micro_labels}"
            )

        if (step + 1) % checkpoint_every == 0 or (step + 1) == num_steps:
            current_checkpoint_name = f"{checkpoint_name}_step{step + 1}"
            print(f"\nSaving checkpoint '{current_checkpoint_name}'...")
            checkpoint = training_client.save_weights_for_sampler(name=current_checkpoint_name).result()
            checkpoint_state = training_client.save_state(name=f"{current_checkpoint_name}_state").result()
            record = {
                "step": step + 1,
                "path": checkpoint.path,
                "state_path": checkpoint_state.path,
                "loss": float(sum(loss_history[-checkpoint_every:]) / len(loss_history[-checkpoint_every:])),
                "published": False,
            }

            should_publish = publish and (publish_intermediate or (step + 1) == num_steps)
            if should_publish:
                print("Publishing checkpoint...")
                rest_client = service_client.create_rest_client()
                rest_client.publish_checkpoint_from_tinker_path(checkpoint.path).result()
                record["published"] = True
                print("  Published successfully")
            elif not publish:
                print("Skipping publish (--no_publish).")
            else:
                print("Skipping intermediate publish (only final checkpoint will be published).")

            checkpoint_records.append(record)
            write_training_metadata(
                model_name=model_name,
                renderer_name=renderer_name,
                config=config,
                seed=seed,
                offline_mode=offline_mode,
                gsm8k_train_path=gsm8k_train_path,
                if_data_path=if_data_path,
                code_data_path=code_data_path,
                checkpoint_records=checkpoint_records,
                loss_history=loss_history,
            )

    return checkpoint_records, loss_history


def parse_args():
    parser = argparse.ArgumentParser(description="Train, save, and publish a compliant multi-task checkpoint")
    parser.add_argument("--mode", choices=["dev", "medium", "final"], default=None)
    parser.add_argument("--model", type=str, default=None, help="Override the preset/base model")
    parser.add_argument("--math_samples", type=int, default=None)
    parser.add_argument("--if_samples", type=int, default=None)
    parser.add_argument("--code_samples", type=int, default=None)
    parser.add_argument("--num_steps", type=int, default=None, help="Number of training steps")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--rank", type=int, default=None, help="LoRA rank")
    parser.add_argument("--max_length", type=int, default=None, help="Maximum rendered sequence length")
    parser.add_argument("--save_every", type=int, default=None, help="Save checkpoint every N steps (0 = final only)")
    parser.add_argument("--checkpoint_name", type=str, default=None, help="Checkpoint name prefix")
    parser.add_argument("--mixture_preset", choices=sorted(MIXTURE_PRESETS.keys()), default=None)
    parser.add_argument("--quick_eval_limit", type=int, default=None, help="Per-task sample count for quick checkpoint screening (0 disables)")
    parser.add_argument("--full_eval_top_k", type=int, default=None, help="Run full eval on the top K quick-screened checkpoints (0 disables)")
    parser.add_argument("--quick_eval_min_humaneval", type=float, default=None, help="Minimum quick HumanEval accuracy required before promotion to full eval")
    parser.add_argument("--results_path", type=str, default=None, help="Path to structured experiment results JSON")
    parser.add_argument("--lr_schedule", choices=["constant", "cosine"], default=None, help="Learning-rate schedule")
    parser.add_argument("--warmup_ratio", type=float, default=None, help="Warmup ratio for cosine schedule")
    parser.add_argument("--warmup_steps", type=int, default=None, help="Override warmup steps")
    parser.add_argument("--grad_accum_steps", type=int, default=None, help="Gradient accumulation steps per optimizer step")
    parser.add_argument("--use_cookbook_lr", action="store_true", help="Use tinker-cookbook get_lr(model_name) instead of manual lr")
    parser.add_argument("--gsm8k_train_path", type=str, default=None, help="Optional local GSM8K train data path")
    parser.add_argument("--if_data_path", type=str, default=None, help="Optional local instruction data path")
    parser.add_argument("--code_data_path", type=str, default=None, help="Optional local code data path")
    parser.add_argument("--publish_intermediate", action="store_true", help="Publish every saved checkpoint, not just the final one")
    parser.add_argument("--no_publish", action="store_true", help="Skip publishing")
    parser.add_argument("--resume_from_state", type=str, default=None, help="Optional checkpoint/state path to continue training from")
    parser.add_argument("--freeze_mlp", action="store_true", help="Disable LoRA training on MLP blocks")
    parser.add_argument("--freeze_attn", action="store_true", help="Disable LoRA training on attention blocks")
    parser.add_argument("--freeze_unembed", action="store_true", help="Disable LoRA training on unembedding/output head")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve_config(args):
    config = DEFAULT_CONFIG.copy()
    if args.mode:
        config.update(PRESETS[args.mode])

    for key in [
        "model",
        "math_samples",
        "if_samples",
        "code_samples",
        "num_steps",
        "batch_size",
        "lr",
        "rank",
        "max_length",
        "save_every",
        "checkpoint_name",
        "mixture_preset",
        "quick_eval_limit",
        "full_eval_top_k",
        "quick_eval_min_humaneval",
        "results_path",
        "lr_schedule",
        "warmup_ratio",
        "warmup_steps",
        "grad_accum_steps",
    ]:
        value = getattr(args, key)
        if value is not None:
            config[key] = value

    if args.freeze_mlp:
        config["train_mlp"] = False
    if args.freeze_attn:
        config["train_attn"] = False
    if args.freeze_unembed:
        config["train_unembed"] = False
    if args.use_cookbook_lr:
        config["use_cookbook_lr"] = True

    sample_override_flags = {
        "math": args.math_samples is not None,
        "if": args.if_samples is not None,
        "code": args.code_samples is not None,
    }
    multipliers = MIXTURE_PRESETS[config["mixture_preset"]]["sample_multipliers"]
    if not sample_override_flags["math"]:
        config["math_samples"] = max(1, int(round(config["math_samples"] * multipliers["math"])))
    if not sample_override_flags["if"]:
        config["if_samples"] = max(1, int(round(config["if_samples"] * multipliers["if"])))
    if not sample_override_flags["code"]:
        config["code_samples"] = max(1, int(round(config["code_samples"] * multipliers["code"])))
    return config


def main():
    args = parse_args()
    config = resolve_config(args)

    random.seed(args.seed)
    np.random.seed(args.seed)

    model_name = config["model"]
    if config.get("use_cookbook_lr"):
        config["lr"] = cookbook_get_lr(model_name, is_lora=True)
    print(f"Model: {model_name}")
    if LOCAL_LLAMA3_TOKENIZER_DIR:
        print(f"Using local Llama 3 tokenizer from {LOCAL_LLAMA3_TOKENIZER_DIR}")
    print(f"Offline mode: {is_offline_mode()}")
    print(f"Mixture preset: {config['mixture_preset']}")
    print(
        "LoRA target surface: "
        f"attn={config['train_attn']} mlp={config['train_mlp']} unembed={config['train_unembed']}"
    )
    if config.get("use_cookbook_lr"):
        print(f"Using cookbook LoRA LR: {config['lr']:.6g}")

    tokenizer = get_tokenizer(model_name)
    renderer_name = model_info.get_recommended_renderer_name(model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    print(f"Renderer: {renderer_name}")

    print("Preparing real training data...")
    gsm8k_train_path = resolve_local_dataset_path(args.gsm8k_train_path, "gsm8k")
    if_data_path = resolve_local_dataset_path(args.if_data_path, "if")
    code_data_path = resolve_local_dataset_path(args.code_data_path, "code")
    print(f"Local GSM8K path: {gsm8k_train_path}")
    print(f"Local IF path: {if_data_path}")
    print(f"Local code path: {code_data_path}")

    math_examples = load_math_examples(
        config["math_samples"],
        gsm8k_local_path=gsm8k_train_path,
        if_local_path=if_data_path,
        mixture_preset=config["mixture_preset"],
    )
    instruction_examples = load_instruction_examples(
        config["if_samples"],
        local_path=if_data_path,
        mixture_preset=config["mixture_preset"],
    )
    code_examples = load_code_examples(
        config["code_samples"],
        local_path=code_data_path,
        mixture_preset=config["mixture_preset"],
    )
    conversation_groups = build_training_conversations(math_examples, instruction_examples, code_examples)
    training_data = prepare_training_data(conversation_groups, renderer, config["max_length"])
    print(f"  {len(training_data)} training examples prepared")

    checkpoint_records, loss_history = train(
        model_name=model_name,
        renderer_name=renderer_name,
        config=config,
        seed=args.seed,
        offline_mode=is_offline_mode(),
        gsm8k_train_path=gsm8k_train_path,
        if_data_path=if_data_path,
        code_data_path=code_data_path,
        training_data=training_data,
        num_steps=config["num_steps"],
        batch_size=config["batch_size"],
        lr=config["lr"],
        rank=config["rank"],
        grad_accum_steps=config["grad_accum_steps"],
        checkpoint_name=config["checkpoint_name"],
        save_every=config["save_every"],
        publish=not args.no_publish,
        publish_intermediate=args.publish_intermediate,
        resume_from_state=args.resume_from_state,
        train_mlp=config["train_mlp"],
        train_attn=config["train_attn"],
        train_unembed=config["train_unembed"],
    )

    experiment_results = evaluate_saved_checkpoints(model_name, checkpoint_records, config)

    write_training_metadata(
        model_name=model_name,
        renderer_name=renderer_name,
        config=config,
        seed=args.seed,
        offline_mode=is_offline_mode(),
        gsm8k_train_path=gsm8k_train_path,
        if_data_path=if_data_path,
        code_data_path=code_data_path,
        checkpoint_records=checkpoint_records,
        loss_history=loss_history,
        experiment_results=experiment_results,
    )

    checkpoint_info_path = os.path.join(EVAL_DIR, "checkpoint_info.json")
    training_info_path = os.path.join(EVAL_DIR, "training_info.json")
    print(f"\nCheckpoint info saved to {checkpoint_info_path}")
    print(f"Training info saved to {training_info_path}")

    if experiment_results and experiment_results.get("best_checkpoint_by_eval"):
        best_checkpoint = experiment_results["best_checkpoint_by_eval"]
        print("\nBest checkpoint selected by evaluation")
        print_results_table([best_checkpoint], "[EVAL] Best checkpoint")
        best_path = best_checkpoint["checkpoint"]
    else:
        best_checkpoint = min(checkpoint_records, key=lambda record: record["loss"])
        best_path = best_checkpoint["path"]
        print("\nBest checkpoint selected by training loss")

    print("\nEvaluation command for the selected checkpoint:")
    print(
        f"  python evaluation/eval_all.py --checkpoint_path \"{best_path}\" "
        f"--base_model {model_name}"
    )


if __name__ == "__main__":
    main()
