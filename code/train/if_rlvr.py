"""
IF-RLVR — GRPO for Instruction Following with Verifiable Rewards.

Adapted directly from grpo_gsm8k.py, same GRPO infrastructure, different
reward function. Following the Allen AI 2025 paper "Generalizing Verifiable
Instruction Following" (IF-RLVR):
    - Pull user instructions from tulu-3-sft-mixture
    - Randomly append 1-6 verifiable constraints per prompt
    - Reward = fraction of constraints satisfied (rule-based, no LLM judge)
    - GRPO with group size 16, reward centering within group

Key design choices (from the paper):
    - Training with up to 5-6 constraints per instance yields better IFEval
      than training with only 1-3 (even though test has at most 3).
    - Variable ranges wider than test range (e.g. "20-40 sentences" train vs
      "1-20" test) improves generalization.
    - Constraints are drawn from the IFEval taxonomy (Apache 2.0 license, NOT
      test data), plus a few extra easy ones.

COMPLIANCE:
    - No IFEval test prompts used. We only use the constraint taxonomy
      (check names and parameter ranges), which is categorical metadata, not
      test data. The user instructions come from tulu-3-sft-mixture (allowed).
    - No LLM teacher. All verifiers are pure Python rule-based functions.

Usage:
    python evaluation/if_rlvr.py \\
        --model meta-llama/Llama-3.2-3B \\
        --load_from_sft "tinker://<your-sft-run>/sampler_weights/sft_final_stepNNNNNN" \\
        --tulu_path /path/to/local/tulu \\
        --num_iterations 50 \\
        --log_dir ./logs/if_rlvr
"""

import argparse
import json
import logging
import os
import random
import re
import string
import time
from concurrent.futures import Future
from typing import Callable

import tinker
import torch
from datasets import load_dataset, load_from_disk
from tinker import types
from tinker.types.tensor_data import TensorData
from tqdm import tqdm

# tokenizer bootstrap (same pattern as grpo_gsm8k)
try:
    from evaluation.tokenizer_bootstrap import configure_local_tokenizers
    configure_local_tokenizers()
except Exception:
    try:
        from tokenizer_bootstrap import configure_local_tokenizers
        configure_local_tokenizers()
    except Exception:
        pass

from tinker_cookbook import model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("if_rlvr")
logging.getLogger("httpx").setLevel(logging.WARN)


# ======================================================================
# Constraint library — pure Python verifiers.
# Each entry: (name, render_fn -> (instruction_text, verify_fn))
# verify_fn(response_text: str) -> bool
#
# We implement ~10 constraint types. This is deliberately a *subset* of the
# IFEval taxonomy — fewer types means the model learns them more reliably,
# and IFEval's overall score is the average of prompt_level + instruction_level
# (strict + loose), so saturating a few types is enough.
# ======================================================================

def _count_words(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def _count_sentences(text: str) -> int:
    # simple but robust-enough
    return len([s for s in re.split(r"[.!?]+", text) if s.strip()])


def _count_paragraphs(text: str) -> int:
    return len([p for p in re.split(r"\n\s*\n", text.strip()) if p.strip()])


def _count_bullets(text: str) -> int:
    # bullet = line starting with *, -, +, or "N."
    n = 0
    for line in text.splitlines():
        s = line.strip()
        if re.match(r"^[\*\-\+]\s+\S", s):
            n += 1
        elif re.match(r"^\d+[\.\)]\s+\S", s):
            n += 1
    return n


# Each builder returns (constraint_text, verifier)
def c_min_words(rng: random.Random):
    n = rng.choice([30, 50, 80, 120, 200, 300])
    text = f"Your response must contain at least {n} words."
    return text, lambda r: _count_words(r) >= n


def c_max_words(rng: random.Random):
    n = rng.choice([30, 50, 80, 120, 200])
    text = f"Your response must contain at most {n} words."
    return text, lambda r: _count_words(r) <= n


def c_num_sentences(rng: random.Random):
    # wider range than test; IFEval test uses 1-10
    n = rng.choice([3, 5, 8, 12, 15, 20])
    text = f"Your response must contain exactly {n} sentences."
    return text, lambda r: _count_sentences(r) == n


def c_num_paragraphs(rng: random.Random):
    n = rng.choice([2, 3, 4, 5, 6])
    text = f"Your response must contain exactly {n} paragraphs, separated by blank lines."
    return text, lambda r: _count_paragraphs(r) == n


def c_num_bullets(rng: random.Random):
    n = rng.choice([3, 4, 5, 6, 8])
    text = f"Your answer must contain exactly {n} bullet points. Use markdown bullets like `* item`."
    return text, lambda r: _count_bullets(r) == n


def c_all_uppercase(rng: random.Random):
    text = "Your entire response must be in English, and in all capital letters. No lowercase letters allowed."
    def verify(r: str) -> bool:
        letters = [c for c in r if c.isalpha()]
        return bool(letters) and all(c.isupper() for c in letters)
    return text, verify


def c_all_lowercase(rng: random.Random):
    text = "Your entire response must be in English, and in all lowercase letters. No capital letters allowed."
    def verify(r: str) -> bool:
        letters = [c for c in r if c.isalpha()]
        return bool(letters) and all(c.islower() for c in letters)
    return text, verify


# A small bank of keywords to inject
_KEYWORD_BANK = [
    "notable", "illumination", "chronicle", "tangential", "serendipity",
    "fortitude", "quintessential", "paradigm", "resonance", "meticulous",
    "algorithm", "equilibrium", "labyrinth", "momentum", "harbinger",
    "peculiar", "tapestry", "abundant", "pristine", "eloquent",
]


def c_keyword_required(rng: random.Random):
    k = rng.choice(_KEYWORD_BANK)
    text = f'Include the keyword "{k}" in your response.'
    return text, lambda r: k.lower() in r.lower()


def c_keyword_forbidden(rng: random.Random):
    k = rng.choice(["the", "and", "you", "is", "a"])
    text = f'Do not include the word "{k}" anywhere in your response.'
    # match as whole word (case-insensitive)
    pat = re.compile(r"\b" + re.escape(k) + r"\b", re.IGNORECASE)
    return text, lambda r: pat.search(r) is None


def c_keyword_frequency(rng: random.Random):
    k = rng.choice(_KEYWORD_BANK)
    n = rng.choice([2, 3, 4, 5])
    text = f'The word "{k}" must appear exactly {n} times in your response.'
    pat = re.compile(r"\b" + re.escape(k) + r"\b", re.IGNORECASE)
    return text, lambda r: len(pat.findall(r)) == n


def c_no_comma(rng: random.Random):
    text = "Your response must not contain any commas."
    return text, lambda r: "," not in r


def c_start_with(rng: random.Random):
    prefix = rng.choice(["Sure", "Certainly", "Of course", "Well", "Let me explain",
                         "Here is", "In summary"])
    text = f'Your response must start with the exact phrase "{prefix}".'
    return text, lambda r: r.lstrip().startswith(prefix)


def c_end_with(rng: random.Random):
    suffix = rng.choice(["Is there anything else?", "Let me know if this helps.",
                         "I hope this is useful.", "That is all."])
    text = f'Your response must end with the exact phrase "{suffix}".'
    return text, lambda r: r.rstrip().endswith(suffix)


def c_quote_wrapped(rng: random.Random):
    text = "Wrap your entire response in double quotation marks."
    def verify(r: str) -> bool:
        s = r.strip()
        return s.startswith('"') and s.endswith('"') and len(s) >= 2
    return text, verify


def c_title_tag(rng: random.Random):
    text = ('Your answer must contain a title, wrapped in double angular brackets, '
            'such as <<poem of joy>>.')
    pat = re.compile(r"<<[^<>]{2,}>>")
    return text, lambda r: pat.search(r) is not None


def c_highlighted_sections(rng: random.Random):
    n = rng.choice([2, 3, 4])
    text = f"Highlight at least {n} sections of your answer using markdown bold (**like this**)."
    pat = re.compile(r"\*\*[^\*\n]{1,}\*\*")
    return text, lambda r: len(pat.findall(r)) >= n


def c_placeholder_brackets(rng: random.Random):
    n = rng.choice([2, 3, 4])
    text = (f"The response must contain at least {n} placeholders represented by "
            f"square brackets, such as [address] or [name].")
    pat = re.compile(r"\[[A-Za-z][A-Za-z0-9 _\-]{0,30}\]")
    return text, lambda r: len(pat.findall(r)) >= n


def c_postscript(rng: random.Random):
    tag = rng.choice(["P.S.", "P.P.S."])
    text = f'At the end of your response, include a postscript starting with "{tag}".'
    return text, lambda r: tag in r and r.rfind(tag) > len(r) * 0.5


# Full registry
CONSTRAINTS: list[Callable[[random.Random], tuple[str, Callable[[str], bool]]]] = [
    c_min_words, c_max_words, c_num_sentences, c_num_paragraphs, c_num_bullets,
    c_all_uppercase, c_all_lowercase,
    c_keyword_required, c_keyword_forbidden, c_keyword_frequency,
    c_no_comma, c_start_with, c_end_with, c_quote_wrapped,
    c_title_tag, c_highlighted_sections, c_placeholder_brackets, c_postscript,
]

# Pairs that conflict — don't put them on the same prompt.
_CONFLICTS = [
    ("c_all_uppercase", "c_all_lowercase"),
    ("c_min_words", "c_max_words"),          # not strictly conflicting but risky
    ("c_quote_wrapped", "c_postscript"),     # postscript after close-quote is weird
    ("c_quote_wrapped", "c_title_tag"),
    ("c_all_uppercase", "c_keyword_required"),   # keyword case-matching gets weird
    ("c_all_lowercase", "c_keyword_required"),
    ("c_no_comma", "c_num_sentences"),       # sentences often want commas inside
]
_CONFLICT_SET = set()
for a, b in _CONFLICTS:
    _CONFLICT_SET.add((a, b))
    _CONFLICT_SET.add((b, a))


def sample_constraints(rng: random.Random, max_n: int = 6):
    """Pick 1..max_n non-conflicting constraints."""
    n = rng.randint(1, max_n)
    pool = list(CONSTRAINTS)
    rng.shuffle(pool)
    chosen = []
    for fn in pool:
        if any((fn.__name__, c.__name__) in _CONFLICT_SET for c in chosen):
            continue
        chosen.append(fn)
        if len(chosen) >= n:
            break
    rendered = [fn(rng) for fn in chosen]
    texts = [t for (t, _) in rendered]
    verifiers = [v for (_, v) in rendered]
    return texts, verifiers


# ======================================================================
# Prompt assembly: instruction + constraint block
# ======================================================================

def build_full_prompt(instruction: str, constraint_texts: list[str]) -> str:
    """Combine instruction with constraints in a natural way."""
    if not constraint_texts:
        return instruction
    constraints_block = "\n".join(f"- {c}" for c in constraint_texts)
    return (
        f"{instruction.strip()}\n\n"
        f"You must follow these additional constraints when answering:\n"
        f"{constraints_block}"
    )


def reward_fn(response: str, verifiers: list[Callable[[str], bool]]) -> float:
    """Reward = fraction of constraints satisfied. All pure Python."""
    if not verifiers:
        return 0.0
    passed = 0
    for v in verifiers:
        try:
            if v(response):
                passed += 1
        except Exception:
            pass
    return passed / len(verifiers)


# ======================================================================
# Pull instructions from tulu-3-sft-mixture, one of the approved training sources.
# ======================================================================

_FORBIDDEN = re.compile(
    r"ifeval|google/ifeval|gsm8k|openai/gsm8k|humaneval|openai_humaneval|human-eval",
    re.IGNORECASE,
)


def load_instructions(tulu_path: str | None, n: int, seed: int = 0) -> list[str]:
    """Load user instructions from tulu-3-sft-mixture. We only keep the FIRST
    user turn (no multi-turn), and drop any that mention benchmark names."""
    if tulu_path and os.path.exists(tulu_path):
        ds = load_from_disk(tulu_path)
        if hasattr(ds, "keys") and "train" in ds:
            ds = ds["train"]
    else:
        ds = load_dataset("allenai/tulu-3-sft-mixture", split="train",
                          streaming=True)

    out = []
    rng = random.Random(seed)
    scanned = 0
    for row in ds:
        scanned += 1
        if scanned > max(500_000, n * 30):
            break
        msgs = row.get("messages") or []
        if not msgs:
            continue
        first = msgs[0]
        if first.get("role") != "user":
            continue
        text = (first.get("content") or "").strip()
        if not text or len(text) < 20 or len(text) > 2000:
            continue
        if _FORBIDDEN.search(text):
            continue
        # Prefer instructions that do not already contain formatting constraints,
        # which gives a cleaner signal for the appended constraints.
        lowered = text.lower()
        if any(k in lowered for k in ["exactly", "at least", "at most", "bullet",
                                       "paragraph", "word", "uppercase"]):
            # Retain a small fraction to preserve diversity.
            if rng.random() > 0.1:
                continue
        out.append(text)
        if len(out) >= n:
            break
    log.info(f"loaded {len(out)} base instructions (scanned {scanned})")
    return out


# ======================================================================
# Training loop — essentially identical to grpo_gsm8k.py
# ======================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="meta-llama/Llama-3.2-3B")
    ap.add_argument("--load_from_sft", type=str, default=None,
                    help="tinker:// training state checkpoint (/weights/...) from SFT stage")
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--batch_size", type=int, default=64,
                    help="Problems per iteration. Smaller than gsm8k because "
                         "IF prompts + responses are longer.")
    ap.add_argument("--group_size", type=int, default=8,
                    help="Rollouts per problem (GRPO group size).")
    ap.add_argument("--learning_rate", type=float, default=3e-5)
    ap.add_argument("--max_tokens", type=int, default=512,
                    help="Max new tokens per rollout. IF prompts need more room.")
    ap.add_argument("--num_iterations", type=int, default=50)
    ap.add_argument("--save_every", type=int, default=10)
    ap.add_argument("--log_dir", type=str, default="./logs/if_rlvr")
    ap.add_argument("--checkpoint_name", type=str, default="if_rlvr")
    ap.add_argument("--tulu_path", type=str, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_publish", action="store_true")
    ap.add_argument("--max_constraints", type=int, default=6,
                    help="Max constraints per prompt (IFEval test has ≤3, but "
                         "the paper shows training with up to 5-6 is better).")
    args = ap.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    rng = random.Random(args.seed)

    # --- tokenizer / renderer
    tokenizer = get_tokenizer(args.model)
    renderer_name = model_info.get_recommended_renderer_name(args.model)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    log.info(f"renderer: {renderer_name}")

    # --- data: enough instructions for num_iterations * batch_size
    needed = args.num_iterations * args.batch_size + 500
    instructions = load_instructions(args.tulu_path, needed, seed=args.seed)
    if len(instructions) < args.num_iterations * args.batch_size:
        log.warning(f"Only have {len(instructions)} instructions, reducing "
                    f"num_iterations to {len(instructions) // args.batch_size}")
        args.num_iterations = len(instructions) // args.batch_size

    # --- client
    service = tinker.ServiceClient()
    rest = service.create_rest_client()

    if args.load_from_sft:
        log.info(f"loading SFT weights from {args.load_from_sft}")
        if "/sampler_weights/" in args.load_from_sft:
            raise ValueError(
                "--load_from_sft must be a training state path (/weights/...), "
                "not a sampler checkpoint (/sampler_weights/...)."
            )
        tc = service.create_training_client_from_state(path=args.load_from_sft)
        log.info("loaded training state with fresh RL optimizer state")
    else:
        log.info("starting from base model (no SFT init)")
        tc = service.create_lora_training_client(base_model=args.model, rank=args.rank)

    adam = types.AdamParams(
        learning_rate=args.learning_rate, beta1=0.9, beta2=0.95, eps=1e-8,
    )
    sampling_params = tinker.types.SamplingParams(
        max_tokens=args.max_tokens, stop=renderer.get_stop_sequences(),
    )

    saved = []
    reward_history = []

    def _write_info():
        info = {
            "model": args.model,
            "load_from_sft": args.load_from_sft,
            "rank": args.rank,
            "batch_size": args.batch_size,
            "group_size": args.group_size,
            "learning_rate": args.learning_rate,
            "max_constraints": args.max_constraints,
            "iterations_run": len(reward_history),
            "reward_history": reward_history,
            "checkpoints": saved,
            "final_path": saved[-1]["path"] if saved else None,
            "final_sampler_path": saved[-1]["sampler_path"] if saved else None,
            "final_state_path": saved[-1]["state_path"] if saved else None,
        }
        with open(os.path.join(args.log_dir, "checkpoint_info.json"), "w") as f:
            json.dump(info, f, indent=2)
        with open("checkpoint_info.json", "w") as f:
            json.dump(info, f, indent=2)

    def save_and_publish(it: int, tag: str):
        name = f"{args.checkpoint_name}_{tag}_iter{it:04d}"
        log.info(f"[save] {name}")
        state_result = tc.save_state(name=name).result()
        sampler_result = tc.save_weights_for_sampler(name=name).result()
        rec = {
            "iteration": it,
            "tag": tag,
            "name": name,
            "path": sampler_result.path,
            "sampler_path": sampler_result.path,
            "state_path": state_result.path,
            "published": False,
        }
        if not args.no_publish:
            try:
                rest.publish_checkpoint_from_tinker_path(sampler_result.path).result()
                rec["published"] = True
                log.info(f"[publish] {sampler_result.path}")
            except Exception as e:
                log.warning(f"publish failed: {e}")
        saved.append(rec)
        _write_info()

    log.info(f"running {args.num_iterations} iterations, batch_size={args.batch_size}, "
             f"group_size={args.group_size}")

    for it in range(args.num_iterations):
        t0 = time.time()

        sampling_client = tc.save_weights_and_get_sampling_client()

        # pick batch of (instruction, constraints, verifiers)
        batch_data = []
        for i in range(args.batch_size):
            inst = instructions[it * args.batch_size + i]
            texts, verifiers = sample_constraints(rng, max_n=args.max_constraints)
            full_prompt = build_full_prompt(inst, texts)
            batch_data.append((full_prompt, verifiers))

        # submit all rollouts
        futures: list[Future] = []
        prompts: list = []
        for (full_prompt, _) in batch_data:
            mi = renderer.build_generation_prompt([
                {"role": "user", "content": full_prompt}
            ])
            fut = sampling_client.sample(
                prompt=mi, num_samples=args.group_size,
                sampling_params=sampling_params,
            )
            futures.append(fut)
            prompts.append(mi)

        # collect + score
        datums: list[types.Datum] = []
        mean_rewards = []
        skipped = 0
        for fut, prompt, (_, verifiers) in tqdm(
            zip(futures, prompts, batch_data),
            total=len(futures), desc=f"iter {it} sampling",
        ):
            res = fut.result()
            rewards_g, toks_g, lp_g = [], [], []
            for seq in res.sequences:
                tok = seq.tokens
                lp = seq.logprobs
                assert lp is not None
                toks_g.append(tok)
                lp_g.append(lp)
                parsed, _ = renderer.parse_response(tok)
                content = renderers.get_text_content(parsed)
                rewards_g.append(reward_fn(content, verifiers))

            mean_r = sum(rewards_g) / len(rewards_g)
            mean_rewards.append(mean_r)
            adv_g = [r - mean_r for r in rewards_g]

            if all(abs(a) < 1e-9 for a in adv_g):
                skipped += 1
                continue
            for tok, lp, adv in zip(toks_g, lp_g, adv_g):
                ob_len = prompt.length - 1
                mi = prompt.append(types.EncodedTextChunk(tokens=tok[:-1]))
                target = [0] * ob_len + tok
                padded_lp = [0.0] * ob_len + lp
                padded_adv = [0.0] * ob_len + [adv] * (mi.length - ob_len)
                datums.append(types.Datum(
                    model_input=mi,
                    loss_fn_inputs={
                        "target_tokens": TensorData.from_torch(torch.tensor(target)),
                        "logprobs": TensorData.from_torch(torch.tensor(padded_lp)),
                        "advantages": TensorData.from_torch(torch.tensor(padded_adv)),
                    },
                ))

        batch_mean_reward = sum(mean_rewards) / len(mean_rewards) if mean_rewards else 0.0
        reward_history.append(batch_mean_reward)

        if not datums:
            log.warning(f"iter {it}: all advantages zero, skipping optim_step")
        else:
            fb = tc.forward_backward(datums, loss_fn="importance_sampling")
            op = tc.optim_step(adam)
            fb.result()
            op.result()

        log.info(
            f"iter {it+1}/{args.num_iterations} mean_reward={batch_mean_reward:.3f} "
            f"n_datums={len(datums)} skipped={skipped} "
            f"time={time.time()-t0:.1f}s"
        )

        if args.save_every > 0 and (it + 1) % args.save_every == 0:
            save_and_publish(it + 1, "mid")

    save_and_publish(args.num_iterations, "final")

    log.info("=" * 60)
    log.info(f"IF-RLVR DONE — final checkpoint: {saved[-1]['path']}")
    log.info(f"final batch reward: {reward_history[-1]:.3f}")
    log.info("=" * 60)
    log.info("Evaluate with:")
    log.info(
        f"  python evaluation/eval_all.py "
        f"--checkpoint_path \"{saved[-1]['path']}\" "
        f"--base_model {args.model}"
    )


if __name__ == "__main__":
    main()
