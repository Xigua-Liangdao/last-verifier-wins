# Last Verifier Wins

Code and experiment artifacts for a sequential RLVR study on `meta-llama/Llama-3.1-8B`, comparing multi-task SFT, IF-RLVR, GSM8K GRPO, a cold-start GRPO control, and a reverse-order math->IF pipeline.

This repository is organized as a compact reproduction bundle: the primary training code, the main ablation scripts, five final 8B evaluation JSONs, and the summary analyses used in the write-up.

## Main Results

| Run | IFEval | GSM8K | HumanEval | Avg |
| --- | ---: | ---: | ---: | ---: |
| 8B SFT | 68.82 | 58.23 | 49.39 | 58.81 |
| 8B IF-RLVR-only | 82.03 | 65.43 | 49.39 | 65.62 |
| 8B cold-start GRPO best | 70.46 | 78.62 | 55.49 | 68.19 |
| 8B forward final | 81.60 | 78.39 | 56.71 | 72.23 |
| 8B reverse final | 83.58 | 77.10 | 56.71 | 72.46 |

Key findings:

- The forward staged pipeline is the primary final method and remains the strongest math-heavy configuration in this bundle.
- The reverse-order pipeline slightly wins on average and IFEval, showing that math-first then IF can recover instruction following surprisingly well.
- Cold-start GRPO on 8B does improve over SFT, but it remains materially worse than the staged RL pipelines on alignment-sensitive behavior.
- The 3B runs are retained as ablations and negative controls rather than as final methods.

## Repository Layout

- `code/train/`: final training code used for the main 8B pipeline.
- `code/ablations/`: 3B screening and earlier multitask ablation code kept for citation and negative-result context.
- `code/scripts/`: reproducible driver scripts for forward, cold-start, reverse, and format-analysis workflows.
- `evaluation/`: renamed copies of the five best 8B submission JSONs.
- `results/`: compact metrics, reverse-vs-forward comparison, and format-analysis outputs.
- `checkpoints.md`: checkpoint provenance, stage configs, and 3B summary.

## Quick Start

Assumptions:

- Run from the repository root, not from inside `final/`.
- Python environment with the dependencies installed.
- Llama tokenizer already prepared under `evaluation/local_tokenizers/` or `LLAMA3_TOKENIZER_DIR`.
- Tinker API key exported in the shell.

### 1. Build the training mix

```bash
python prepare_data.py \
  --out evaluation/train_3b_pipeline.jsonl \
  --gsm8k_path /path/to/gsm8k_train \
  --tulu_path /path/to/tulu3_sft_train \
  --code_path /path/to/opencodeinstruct_train
```

### 2. Run the primary pipeline

```bash
bash final/code/scripts/run_8b_forward.sh
```

### 3. Run the control pipelines

```bash
bash final/code/scripts/run_8b_coldstart_grpo.sh
bash final/code/scripts/run_8b_reverse.sh
```

### 4. Regenerate the format-analysis bundle

```bash
python final/code/scripts/analyze_output_format.py
```

## Data

Training sources used in the final pipeline:

- GSM8K train: https://huggingface.co/datasets/openai/gsm8k
- Tulu 3 SFT mixture: https://huggingface.co/datasets/allenai/tulu-3-sft-mixture
- OpenCodeInstruct: https://huggingface.co/datasets/nvidia/OpenCodeInstruct

Optional local snapshots can be supplied through CLI arguments or environment variables such as `GSM8K_LOCAL`, `TULU_LOCAL`, and `CODE_LOCAL_PATH`.

No IFEval, GSM8K test, or HumanEval examples are used for training.

## Additional Notes

- `checkpoints.md` is the source of truth for checkpoint provenance.
- `results/summary_metrics.json` is the source of truth for compact score reporting.
- `results/format_analysis.csv`, `results/format_correct_vs_wrong.csv`, and `results/format_analysis_summary.md` are reproducible from `code/scripts/analyze_output_format.py`.
- Two historical GRPO runs only preserve sampler checkpoints, not state checkpoints; this is documented explicitly in `checkpoints.md`.

## Citation Placeholder

Placeholder BibTeX for camera-ready release:

```bibtex
@misc{anonymous_sequential_rlvr_2026,
  title  = {Sequential RLVR on Llama-3.1-8B: IF-RLVR, GSM8K GRPO, and Order-Sensitivity Controls},
  author = {Anonymous},
  year   = {2026},
  note   = {Anonymous release bundle}
}
```

## Compute Budget

All reported training runs were executed through Tinker-hosted jobs rather than local GPUs. A practical estimate for the 8B study in this bundle is about `$240` total cloud cost, counting the primary forward pipeline plus the cold-start and reverse-order controls and their associated full evaluations.