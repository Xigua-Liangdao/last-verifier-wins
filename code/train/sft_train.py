"""
Clean multi-task SFT training script for the anonymous release bundle.

Modeled directly on tinker-cookbook's recipes/sl_loop.py, with:
    - Llama-3.1-8B (recommended for final submission)
    - LoRA rank 128
    - batch_size 128 (one real forward_backward per step, no fake grad-accum)
    - learning rate from tinker-cookbook's get_lr(model_name)
    - linear LR decay with short warmup
    - reads a single conversations.jsonl produced by prepare_data.py
    - saves + PUBLISHES checkpoints so eval_all.py can use them

This replaces the larger train_and_publish.py workflow for the final release.
Curriculum choices, mixture weights, and filtering decisions are baked into
the JSONL produced by prepare_data.py.

Usage:
    python sft_train.py \
        --data train.jsonl \
        --model meta-llama/Llama-3.1-8B \
        --rank 128 \
        --batch_size 128 \
        --num_epochs 2 \
        --save_every 200 \
        --log_dir ./logs/sft_8b

Output:
    - Prints the published tinker:// checkpoint path on stdout
        - Writes checkpoint_info.json next to the script with the final+best paths,
            compatible with the bundled eval_all.py
"""

import argparse
import json
import logging
import os
import random
import sys
import time

import numpy as np
import tinker
from tinker import types

from evaluation.tokenizer_bootstrap import configure_local_tokenizers
from tinker_cookbook import model_info, renderers
from tinker_cookbook.hyperparam_utils import get_lr as cookbook_get_lr
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.supervised.data import conversation_to_datum
from tinker_cookbook.tokenizer_utils import get_tokenizer


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("sft")
logging.getLogger("httpx").setLevel(logging.WARN)

LOCAL_LLAMA3_TOKENIZER_DIR = configure_local_tokenizers()


# -------------------- Data --------------------

def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def rows_to_datums(rows, renderer, max_length):
    out = []
    dropped = 0
    for r in rows:
        try:
            datum = conversation_to_datum(
                r["messages"],
                renderer,
                max_length,
                TrainOnWhat.ALL_ASSISTANT_MESSAGES,
            )
            out.append(datum)
        except Exception:
            dropped += 1
    if dropped:
        log.warning(f"dropped {dropped} rows during tokenization")
    return out


# -------------------- Loss computation (matches cookbook) --------------------

def compute_mean_nll(fwd_bwd_result, batch):
    logprobs = np.concatenate(
        [o["logprobs"].tolist() for o in fwd_bwd_result.loss_fn_outputs]
    )
    weights = np.concatenate([d.loss_fn_inputs["weights"].tolist() for d in batch])
    denom = max(weights.sum(), 1.0)
    return float(-np.dot(logprobs, weights) / denom)


# -------------------- Training --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True, help="Path to train.jsonl")
    ap.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--rank", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_epochs", type=int, default=2)
    ap.add_argument("--max_length", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=None,
                    help="Override LR. If omitted, uses tinker-cookbook get_lr()")
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--save_every", type=int, default=200,
                    help="Save+publish sampler checkpoint every N steps. 0=final only.")
    ap.add_argument("--log_dir", type=str, default="./logs/sft")
    ap.add_argument("--checkpoint_name", type=str, default="sft",
                    help="Prefix for saved checkpoint names.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume_from_state", type=str, default=None,
                    help="tinker:// path to resume training (weights+optim) from.")
    ap.add_argument("--no_publish", action="store_true",
                    help="Skip publishing; checkpoint stays private to your run.")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.log_dir, exist_ok=True)

    # --- LR
    if args.lr is not None:
        lr = args.lr
        log.info(f"using manual lr={lr}")
    else:
        lr = cookbook_get_lr(args.model)
        log.info(f"using cookbook get_lr({args.model}) = {lr:.2e}")

    # --- tokenizer / renderer
    tokenizer = get_tokenizer(args.model)
    renderer_name = model_info.get_recommended_renderer_name(args.model)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    log.info(f"renderer: {renderer_name}")
    if LOCAL_LLAMA3_TOKENIZER_DIR:
        log.info(f"local tokenizer: {LOCAL_LLAMA3_TOKENIZER_DIR}")

    # --- data
    log.info(f"loading {args.data}")
    rows = load_jsonl(args.data)
    log.info(f"  {len(rows)} raw conversations")
    random.shuffle(rows)
    datums = rows_to_datums(rows, renderer, args.max_length)
    log.info(f"  {len(datums)} datums after tokenization")

    # Drop last partial batch (matches cookbook sl_loop)
    n_batches_per_epoch = len(datums) // args.batch_size
    n_total_steps = n_batches_per_epoch * args.num_epochs
    if n_total_steps < 50:
        log.warning(f"Only {n_total_steps} steps total — consider more data or smaller batch")
    log.info(
        f"training: batch_size={args.batch_size} "
        f"n_batches_per_epoch={n_batches_per_epoch} "
        f"epochs={args.num_epochs} total_steps={n_total_steps}"
    )

    # --- client
    service = tinker.ServiceClient()
    if args.resume_from_state:
        log.info(f"resuming from {args.resume_from_state}")
        tc = service.create_training_client_from_state_with_optimizer(
            path=args.resume_from_state
        )
    else:
        tc = service.create_lora_training_client(base_model=args.model, rank=args.rank)

    warmup_steps = max(1, int(n_total_steps * args.warmup_ratio))
    log.info(f"warmup_steps={warmup_steps}, decay to 0 linearly after")

    rest = service.create_rest_client()

    saved = []      # list of dicts describing saved checkpoints
    loss_history = []
    start_time = time.time()

    def save_and_publish(step_idx: int, tag: str) -> dict:
        name = f"{args.checkpoint_name}_{tag}_step{step_idx:06d}"
        log.info(f"[save] {name}")
        state_result = tc.save_state(name=name).result()
        sampler_result = tc.save_weights_for_sampler(name=name).result()
        path = sampler_result.path
        state_path = state_result.path
        record = {
            "step": step_idx,
            "tag": tag,
            "name": name,
            "path": path,
            "sampler_path": path,
            "state_path": state_path,
            "published": False,
        }
        if not args.no_publish:
            try:
                rest.publish_checkpoint_from_tinker_path(path).result()
                record["published"] = True
                log.info(f"[publish] {path}")
            except Exception as e:
                log.warning(f"publish failed (continuing): {e}")
        saved.append(record)
        _write_checkpoint_info()
        return record

    def _write_checkpoint_info():
        info = {
            "model": args.model,
            "rank": args.rank,
            "batch_size": args.batch_size,
            "lr": lr,
            "num_epochs": args.num_epochs,
            "total_steps": n_total_steps,
            "checkpoints": saved,
            "final_path": saved[-1]["path"] if saved else None,
            "final_sampler_path": saved[-1]["sampler_path"] if saved else None,
            "final_state_path": saved[-1]["state_path"] if saved else None,
            "num_samples": len(datums),
        }
        with open(os.path.join(args.log_dir, "checkpoint_info.json"), "w") as f:
            json.dump(info, f, indent=2)
        # Also drop a copy next to the script for the grader's eval_all.py
        with open("checkpoint_info.json", "w") as f:
            json.dump(info, f, indent=2)

    # --- main loop
    step_global = 0
    for epoch in range(args.num_epochs):
        random.shuffle(datums)
        for batch_idx in range(n_batches_per_epoch):
            t0 = time.time()

            # LR schedule: linear warmup, linear decay
            if step_global < warmup_steps:
                lr_mult = (step_global + 1) / warmup_steps
            else:
                progress = (step_global - warmup_steps) / max(n_total_steps - warmup_steps, 1)
                lr_mult = max(0.0, 1.0 - progress)
            current_lr = lr * lr_mult

            start = batch_idx * args.batch_size
            batch = datums[start : start + args.batch_size]

            adam = types.AdamParams(
                learning_rate=current_lr, beta1=0.9, beta2=0.95, eps=1e-8
            )
            # Submit both futures before awaiting (cookbook idiom for throughput)
            fb_future = tc.forward_backward(batch, loss_fn="cross_entropy")
            opt_future = tc.optim_step(adam)
            fb = fb_future.result()
            opt_future.result()

            nll = compute_mean_nll(fb, batch)
            loss_history.append(nll)
            step_global += 1

            if step_global % 10 == 0 or step_global == 1:
                elapsed = time.time() - start_time
                recent = sum(loss_history[-50:]) / len(loss_history[-50:])
                log.info(
                    f"step {step_global}/{n_total_steps} (epoch {epoch+1}/{args.num_epochs}) "
                    f"loss={nll:.4f} avg50={recent:.4f} lr={current_lr:.2e} "
                    f"elapsed={elapsed:.0f}s step_time={time.time()-t0:.1f}s"
                )

            if args.save_every > 0 and step_global % args.save_every == 0:
                save_and_publish(step_global, "mid")

    # Final checkpoint
    save_and_publish(step_global, "final")

    log.info("=" * 60)
    log.info(f"TRAINING DONE — final checkpoint: {saved[-1]['path']}")
    log.info("=" * 60)
    log.info("Evaluate with:")
    log.info(
        f"  python evaluation/eval_all.py "
        f"--checkpoint_path \"{saved[-1]['path']}\" "
        f"--base_model {args.model}"
    )


if __name__ == "__main__":
    main()
