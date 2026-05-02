"""
GSM8K GRPO training, starting from a SFT checkpoint.

Directly adapted from tinker-cookbook/recipes/rl_loop.py, which uses
GRPO-style advantage centering + importance-sampling policy gradient.

The only significant changes vs. the stock rl_loop.py:
    1. Accept --load_from_sft to initialize from your SFT sampler checkpoint
       (weights only; optimizer starts fresh — this is the recommended recipe
       for RLVR on top of SFT).
    2. Periodically save+publish sampler checkpoints so eval_all.py can use them.
    3. argparse flags instead of chz config for simpler standalone execution.

Expected behavior (per Tinker docs): ~60-70% GSM8K after 30-50 iterations,
starting from a decent SFT init. Each iteration ≈ 1-2 min on Tinker.

Compliance:
    - Uses ONLY GSM8K train split (openai/gsm8k, 'main' config).
    - Verifiable reward = final numeric answer correct. No test data touched.

Usage:
    python grpo_gsm8k.py \
        --load_from_sft "tinker://<your-run>/sampler_weights/sft_final_step002000" \
        --model meta-llama/Llama-3.1-8B \
        --num_iterations 50 \
        --save_every 10 \
        --log_dir ./logs/grpo_gsm8k
"""

import argparse
import json
import logging
import os
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
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from evaluation.tokenizer_bootstrap import configure_local_tokenizers
from tinker_cookbook import model_info, renderers
from tinker_cookbook.recipes.math_rl.math_env import extract_gsm8k_final_answer
from tinker_cookbook.recipes.math_rl.math_grading import extract_boxed, grade_answer
from tinker_cookbook.tokenizer_utils import get_tokenizer


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("grpo")
logging.getLogger("httpx").setLevel(logging.WARN)

LOCAL_LLAMA3_TOKENIZER_DIR = configure_local_tokenizers()


def get_reward(response: str, answer: str) -> float:
    """Reward = 1 iff model's \\boxed{} matches GSM8K final '####' answer."""
    try:
        given = extract_boxed(response)
        truth = extract_gsm8k_final_answer(answer)
        return 1.0 if grade_answer(given, truth) else 0.0
    except ValueError:
        return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--gsm8k_path", type=str, default=None,
                    help="Optional local path for GSM8K train data (load_from_disk).")
    ap.add_argument("--load_from_sft", type=str, default=None,
                    help="tinker:// sampler checkpoint from SFT stage. "
                         "If omitted, starts from the base model (Tülu-3 style cold-start).")
    ap.add_argument("--rank", type=int, default=32,
                    help="LoRA rank for RL (32 is plenty per Tinker docs)")
    ap.add_argument("--batch_size", type=int, default=128,
                    help="Number of distinct problems per iteration")
    ap.add_argument("--group_size", type=int, default=16,
                    help="Rollouts per problem (GRPO group size)")
    ap.add_argument("--learning_rate", type=float, default=4e-5)
    ap.add_argument("--max_tokens", type=int, default=320,
                    help="Max new tokens per rollout")
    ap.add_argument("--num_iterations", type=int, default=50,
                    help="Number of RL iterations (each = 1 batch of problems)")
    ap.add_argument("--save_every", type=int, default=10,
                    help="Save+publish sampler checkpoint every N iterations")
    ap.add_argument("--log_dir", type=str, default="./logs/grpo_gsm8k")
    ap.add_argument("--checkpoint_name", type=str, default="grpo")
    ap.add_argument("--no_publish", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)

    # --- tokenizer / renderer
    tokenizer = get_tokenizer(args.model)
    renderer_name = model_info.get_recommended_renderer_name(args.model)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    log.info(f"renderer: {renderer_name}")
    if LOCAL_LLAMA3_TOKENIZER_DIR:
        log.info(f"local tokenizer: {LOCAL_LLAMA3_TOKENIZER_DIR}")

    # --- data
    if args.gsm8k_path and os.path.exists(args.gsm8k_path):
        ds = datasets.load_from_disk(args.gsm8k_path)
        train = ds["train"] if hasattr(ds, "keys") and "train" in ds else ds
        log.info(f"loading GSM8K from local path: {args.gsm8k_path}")
    else:
        ds = datasets.load_dataset("openai/gsm8k", "main")
        train = ds["train"]
    log.info(f"GSM8K train size: {len(train)}")

    # Few-shot prefix, copied from tinker-cookbook/rl_loop.py — proven to help
    question_suffix = (
        " Provide a numerical answer without units, written inside \\boxed{}."
    )
    convo_prefix = [
        {"role": "user",
         "content": "How many r's are in strawberry?" + question_suffix},
        {"role": "assistant",
         "content": "Let's spell the word out and number all the letters: "
                    "1) s 2) t 3) r 4) a 5) w 6) b 7) e 8) r 9) r 10) y. "
                    "We have r's at positions 3, 8, and 9. \\boxed{3}"},
    ]

    # --- client
    service = tinker.ServiceClient()
    rest = service.create_rest_client()

    if args.load_from_sft:
        log.info(f"loading SFT init from {args.load_from_sft}")
        if "/sampler_weights/" in args.load_from_sft:
            raise ValueError(
                "--load_from_sft must be a training state path (/weights/...), "
                "not a sampler checkpoint (/sampler_weights/...)."
            )
        # Use create_training_client_from_state (weights-only) — we want a fresh
        # optimizer for the new RL objective.
        tc = service.create_training_client_from_state(path=args.load_from_sft)
    else:
        log.info("starting from base model (no SFT init)")
        tc = service.create_lora_training_client(base_model=args.model, rank=args.rank)

    adam = types.AdamParams(
        learning_rate=args.learning_rate, beta1=0.9, beta2=0.95, eps=1e-8
    )

    # --- log tracking
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

    # --- RL loop (directly from rl_loop.py, slightly trimmed)
    sampling_params = tinker.types.SamplingParams(
        max_tokens=args.max_tokens,
        stop=renderer.get_stop_sequences(),
    )

    n_avail_batches = len(train) // args.batch_size
    n_iters = min(args.num_iterations, n_avail_batches)
    log.info(f"running {n_iters} iterations (batch_size={args.batch_size}, "
             f"group_size={args.group_size})")

    for it in range(n_iters):
        t0 = time.time()

        # sample rollouts with the CURRENT policy weights
        sampling_client = tc.save_weights_and_get_sampling_client()

        batch = train.select(range(it * args.batch_size,
                                   (it + 1) * args.batch_size))

        futures: list[Future] = []
        prompts: list = []
        for q in batch["question"]:
            convo = [*convo_prefix,
                     {"role": "user", "content": q + question_suffix}]
            prompt = renderer.build_generation_prompt(convo)
            fut = sampling_client.sample(
                prompt=prompt,
                num_samples=args.group_size,
                sampling_params=sampling_params,
            )
            futures.append(fut)
            prompts.append(prompt)

        # collect + score
        datums: list[types.Datum] = []
        mean_rewards = []
        skipped = 0
        for fut, prompt, answer in tqdm(
            zip(futures, prompts, batch["answer"]),
            total=len(futures),
            desc=f"iter {it} sampling",
        ):
            res = fut.result()
            rewards_g = []
            toks_g = []
            lp_g = []
            for seq in res.sequences:
                tok = seq.tokens
                lp = seq.logprobs
                assert lp is not None
                toks_g.append(tok)
                lp_g.append(lp)
                parsed, _ = renderer.parse_response(tok)
                content = renderers.get_text_content(parsed)
                rewards_g.append(get_reward(content, answer))
            mean_r = sum(rewards_g) / len(rewards_g)
            mean_rewards.append(mean_r)
            adv_g = [r - mean_r for r in rewards_g]
            if all(a == 0.0 for a in adv_g):
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

        batch_mean_reward = sum(mean_rewards) / len(mean_rewards)
        reward_history.append(batch_mean_reward)

        if not datums:
            log.warning(f"iter {it}: all advantages zero, skipping optim_step")
        else:
            fb = tc.forward_backward(datums, loss_fn="importance_sampling")
            op = tc.optim_step(adam)
            fb.result()
            op.result()

        log.info(
            f"iter {it+1}/{n_iters} mean_reward={batch_mean_reward:.3f} "
            f"n_datums={len(datums)} skipped={skipped} "
            f"time={time.time()-t0:.1f}s"
        )

        if args.save_every > 0 and (it + 1) % args.save_every == 0:
            save_and_publish(it + 1, "mid")

    save_and_publish(n_iters, "final")

    log.info("=" * 60)
    log.info(f"GRPO DONE — final checkpoint: {saved[-1]['path']}")
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
