"""Joint RL baseline that mixes GSM8K math reward and IF-RLVR reward per iteration.

This script starts from an SFT training state and, within each iteration, runs
one GSM8K GRPO batch and one IF-RLVR batch through the same Tinker training
client and optimizer state. The result is a simultaneous multi-objective RL
baseline for comparison against staged sequential RLVR.
"""

import argparse
import json
import logging
import math
import os
import random
import sys
import time
from concurrent.futures import Future

import datasets
import tinker
import torch
from tinker import types
from tinker.types.tensor_data import TensorData
from tqdm import tqdm

try:
    from evaluation.tokenizer_bootstrap import configure_local_tokenizers
except ModuleNotFoundError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from evaluation.tokenizer_bootstrap import configure_local_tokenizers

from evaluation.grpo_gsm8k import get_reward as get_math_reward
from evaluation.if_rlvr import (
    build_full_prompt,
    load_instructions,
    reward_fn as get_if_reward,
    sample_constraints,
)
from tinker_cookbook import model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("joint_rl")
logging.getLogger("httpx").setLevel(logging.WARN)

LOCAL_LLAMA3_TOKENIZER_DIR = configure_local_tokenizers()

QUESTION_SUFFIX = " Provide a numerical answer without units, written inside \\boxed{}."
CONVO_PREFIX = [
    {
        "role": "user",
        "content": "How many r's are in strawberry?" + QUESTION_SUFFIX,
    },
    {
        "role": "assistant",
        "content": (
            "Let's spell the word out and number all the letters: "
            "1) s 2) t 3) r 4) a 5) w 6) b 7) e 8) r 9) r 10) y. "
            "We have r's at positions 3, 8, and 9. \\boxed{3}"
        ),
    },
]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--gsm8k_path", type=str, default=None)
    ap.add_argument("--tulu_path", type=str, default=None)
    ap.add_argument("--load_from_sft", type=str, required=True)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--math_batch_size", type=int, default=64)
    ap.add_argument("--if_batch_size", type=int, default=64)
    ap.add_argument("--group_size", type=int, default=8)
    ap.add_argument("--learning_rate", type=float, default=2e-5)
    ap.add_argument("--lr_schedule", choices=["constant", "cosine"], default="cosine")
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--num_iterations", type=int, default=25)
    ap.add_argument("--save_every", type=int, default=5)
    ap.add_argument("--log_dir", type=str, default="./logs/joint_rl")
    ap.add_argument("--checkpoint_name", type=str, default="joint_rl")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_constraints", type=int, default=6)
    ap.add_argument("--no_publish", action="store_true")
    return ap.parse_args()


def resolve_math_train(path: str | None):
    if path and os.path.exists(path):
        ds = datasets.load_from_disk(path)
        train = ds["train"] if hasattr(ds, "keys") and "train" in ds else ds
        log.info(f"loading GSM8K from local path: {path}")
        return train
    ds = datasets.load_dataset("openai/gsm8k", "main")
    return ds["train"]


def current_lr(args, iteration_idx: int) -> float:
    if args.lr_schedule == "constant":
        return args.learning_rate
    warmup_iters = max(1, int(args.num_iterations * args.warmup_ratio))
    if iteration_idx < warmup_iters:
        return args.learning_rate * (iteration_idx + 1) / warmup_iters
    progress = (iteration_idx - warmup_iters) / max(args.num_iterations - warmup_iters, 1)
    return args.learning_rate * 0.5 * (1 + math.cos(math.pi * progress))


def make_datum(prompt, tok, lp, adv):
    ob_len = prompt.length - 1
    model_input = prompt.append(types.EncodedTextChunk(tokens=tok[:-1]))
    target = [0] * ob_len + tok
    padded_lp = [0.0] * ob_len + lp
    padded_adv = [0.0] * ob_len + [adv] * (model_input.length - ob_len)
    return types.Datum(
        model_input=model_input,
        loss_fn_inputs={
            "target_tokens": TensorData.from_torch(torch.tensor(target)),
            "logprobs": TensorData.from_torch(torch.tensor(padded_lp)),
            "advantages": TensorData.from_torch(torch.tensor(padded_adv)),
        },
    )


def collect_math_datums(tc, renderer, sampling_params, batch, group_size):
    sampling_client = tc.save_weights_and_get_sampling_client()
    futures: list[Future] = []
    prompts = []
    for question in batch["question"]:
        convo = [*CONVO_PREFIX, {"role": "user", "content": question + QUESTION_SUFFIX}]
        prompt = renderer.build_generation_prompt(convo)
        futures.append(
            sampling_client.sample(
                prompt=prompt,
                num_samples=group_size,
                sampling_params=sampling_params,
            )
        )
        prompts.append(prompt)

    datums = []
    mean_rewards = []
    skipped = 0
    for fut, prompt, answer in tqdm(
        zip(futures, prompts, batch["answer"]),
        total=len(futures),
        desc="math sampling",
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
            rewards_g.append(get_math_reward(content, answer))
        mean_r = sum(rewards_g) / len(rewards_g)
        mean_rewards.append(mean_r)
        adv_g = [reward - mean_r for reward in rewards_g]
        if all(abs(adv) < 1e-9 for adv in adv_g):
            skipped += 1
            continue
        for tok, lp, adv in zip(toks_g, lp_g, adv_g):
            datums.append(make_datum(prompt, tok, lp, adv))
    mean_reward = sum(mean_rewards) / len(mean_rewards) if mean_rewards else 0.0
    return datums, mean_reward, skipped


def collect_if_datums(tc, renderer, sampling_params, instructions, group_size, rng, max_constraints):
    sampling_client = tc.save_weights_and_get_sampling_client()
    futures: list[Future] = []
    prompts = []
    batch_data = []
    for instruction in instructions:
        texts, verifiers = sample_constraints(rng, max_n=max_constraints)
        full_prompt = build_full_prompt(instruction, texts)
        prompt = renderer.build_generation_prompt([{"role": "user", "content": full_prompt}])
        futures.append(
            sampling_client.sample(
                prompt=prompt,
                num_samples=group_size,
                sampling_params=sampling_params,
            )
        )
        prompts.append(prompt)
        batch_data.append(verifiers)

    datums = []
    mean_rewards = []
    skipped = 0
    for fut, prompt, verifiers in tqdm(
        zip(futures, prompts, batch_data),
        total=len(futures),
        desc="if sampling",
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
            rewards_g.append(get_if_reward(content, verifiers))
        mean_r = sum(rewards_g) / len(rewards_g)
        mean_rewards.append(mean_r)
        adv_g = [reward - mean_r for reward in rewards_g]
        if all(abs(adv) < 1e-9 for adv in adv_g):
            skipped += 1
            continue
        for tok, lp, adv in zip(toks_g, lp_g, adv_g):
            datums.append(make_datum(prompt, tok, lp, adv))
    mean_reward = sum(mean_rewards) / len(mean_rewards) if mean_rewards else 0.0
    return datums, mean_reward, skipped


def main():
    args = parse_args()
    os.makedirs(args.log_dir, exist_ok=True)
    rng = random.Random(args.seed)

    tokenizer = get_tokenizer(args.model)
    renderer_name = model_info.get_recommended_renderer_name(args.model)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    log.info(f"renderer: {renderer_name}")
    if LOCAL_LLAMA3_TOKENIZER_DIR:
        log.info(f"local tokenizer: {LOCAL_LLAMA3_TOKENIZER_DIR}")

    math_train = resolve_math_train(args.gsm8k_path)
    log.info(f"GSM8K train size: {len(math_train)}")
    needed_instructions = args.num_iterations * args.if_batch_size + 500
    instructions = load_instructions(args.tulu_path, needed_instructions, seed=args.seed)

    n_math_iters = len(math_train) // args.math_batch_size
    n_if_iters = len(instructions) // args.if_batch_size
    n_iters = min(args.num_iterations, n_math_iters, n_if_iters)
    if n_iters < args.num_iterations:
        log.warning(
            f"reducing num_iterations from {args.num_iterations} to {n_iters} due to data limits"
        )

    service = tinker.ServiceClient()
    rest = service.create_rest_client()
    if "/sampler_weights/" in args.load_from_sft:
        raise ValueError("--load_from_sft must be a training state path (/weights/...), not sampler_weights")
    log.info(f"loading SFT state from {args.load_from_sft}")
    tc = service.create_training_client_from_state(path=args.load_from_sft)

    saved = []
    total_reward_history = []
    math_reward_history = []
    if_reward_history = []

    def write_info():
        info = {
            "model": args.model,
            "load_from_sft": args.load_from_sft,
            "rank": args.rank,
            "math_batch_size": args.math_batch_size,
            "if_batch_size": args.if_batch_size,
            "group_size": args.group_size,
            "learning_rate": args.learning_rate,
            "lr_schedule": args.lr_schedule,
            "max_tokens": args.max_tokens,
            "max_constraints": args.max_constraints,
            "iterations_run": len(total_reward_history),
            "reward_history": total_reward_history,
            "math_reward_history": math_reward_history,
            "if_reward_history": if_reward_history,
            "checkpoints": saved,
            "final_path": saved[-1]["path"] if saved else None,
            "final_sampler_path": saved[-1]["sampler_path"] if saved else None,
            "final_state_path": saved[-1]["state_path"] if saved else None,
        }
        with open(os.path.join(args.log_dir, "checkpoint_info.json"), "w") as f:
            json.dump(info, f, indent=2)

    def save_and_publish(iteration: int, tag: str):
        name = f"{args.checkpoint_name}_{tag}_iter{iteration:04d}"
        log.info(f"[save] {name}")
        state_result = tc.save_state(name=name).result()
        sampler_result = tc.save_weights_for_sampler(name=name).result()
        record = {
            "iteration": iteration,
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
                record["published"] = True
                log.info(f"[publish] {sampler_result.path}")
            except Exception as exc:
                log.warning(f"publish failed: {exc}")
        saved.append(record)
        write_info()

    sampling_params = tinker.types.SamplingParams(
        max_tokens=args.max_tokens,
        stop=renderer.get_stop_sequences(),
    )
    log.info(
        f"running {n_iters} joint iterations (math_batch={args.math_batch_size}, if_batch={args.if_batch_size}, group={args.group_size})"
    )

    for it in range(n_iters):
        t0 = time.time()
        adam = types.AdamParams(
            learning_rate=current_lr(args, it),
            beta1=0.9,
            beta2=0.95,
            eps=1e-8,
        )
        math_batch = math_train.select(
            range(it * args.math_batch_size, (it + 1) * args.math_batch_size)
        )
        if_batch = instructions[it * args.if_batch_size : (it + 1) * args.if_batch_size]

        math_datums, math_mean_reward, math_skipped = collect_math_datums(
            tc, renderer, sampling_params, math_batch, args.group_size
        )
        if_datums, if_mean_reward, if_skipped = collect_if_datums(
            tc, renderer, sampling_params, if_batch, args.group_size, rng, args.max_constraints
        )
        datums = math_datums + if_datums
        joint_mean_reward = (math_mean_reward + if_mean_reward) / 2.0
        math_reward_history.append(math_mean_reward)
        if_reward_history.append(if_mean_reward)
        total_reward_history.append(joint_mean_reward)

        if not datums:
            log.warning(f"iter {it}: all advantages zero, skipping optim_step")
        else:
            fb = tc.forward_backward(datums, loss_fn="importance_sampling")
            op = tc.optim_step(adam)
            fb.result()
            op.result()

        log.info(
            f"iter {it+1}/{n_iters} joint_reward={joint_mean_reward:.3f} math_reward={math_mean_reward:.3f} "
            f"if_reward={if_mean_reward:.3f} n_datums={len(datums)} "
            f"skipped_math={math_skipped} skipped_if={if_skipped} lr={current_lr(args, it):.2e} "
            f"time={time.time()-t0:.1f}s"
        )

        if args.save_every > 0 and (it + 1) % args.save_every == 0:
            save_and_publish(it + 1, "mid")

    save_and_publish(n_iters, "final")
    log.info("=" * 60)
    log.info(f"JOINT RL DONE — final checkpoint: {saved[-1]['path']}")
    log.info(
        f"final reward snapshot: joint={total_reward_history[-1]:.3f} math={math_reward_history[-1]:.3f} if={if_reward_history[-1]:.3f}"
    )
    log.info("=" * 60)


if __name__ == "__main__":
    main()