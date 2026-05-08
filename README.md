# Last Verifier Wins

Anonymous reproduction bundle for the paper "Last Verifier Wins the Format, Verifier Set Wins the Score: Decoupling Surface Style from Capability in Sequential RLVR."

The bundle is organized around the six 8B runs used in the paper:

- 8B SFT baseline
- 8B IF-RLVR only
- 8B Math-RLVR only / cold-start GRPO
- 8B Joint RL
- 8B Forward sequential IF->Math
- 8B Reverse sequential Math->IF

Precomputed evaluation outputs are included, so reviewers can regenerate the paper tables and Figure 1 without rerunning the training jobs.

## Main Results

| Run | IFEval | GSM8K | HumanEval | Avg |
| --- | ---: | ---: | ---: | ---: |
| 8B SFT baseline | 68.82 | 58.23 | 49.39 | 58.81 |
| 8B IF-RLVR only | 82.03 | 65.43 | 49.39 | 65.62 |
| 8B Math-RLVR only | 70.46 | 78.62 | 55.49 | 68.19 |
| 8B Joint RL | 76.35 | 76.72 | 54.27 | 69.11 |
| 8B Forward IF->Math | 81.60 | 78.39 | 56.71 | 72.23 |
| 8B Reverse Math->IF | 83.58 | 77.10 | 56.71 | 72.46 |

## Directory Map

- `code/train/`: training code for the main SFT and staged RLVR pipelines.
- `code/ablations/train/`: joint RL baseline implementation.
- `code/scripts/`: reviewer-facing scripts for reproducing tables and figures.
- `evaluation/`: included evaluation JSONs and GSM8K inspect logs for the six paper runs.
- `results/`: generated tables, summaries, and figure outputs.
- `checkpoints.md`: checkpoint provenance for the six paper runs.
- `REPRODUCIBILITY.md`: step-by-step commands and expected outputs.

## Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Some training scripts additionally require:

- a valid `TINKER_API_KEY`
- a local Llama tokenizer under `evaluation/local_tokenizers/` or `LLAMA3_TOKENIZER_DIR`
- local dataset snapshots passed through environment variables such as `GSM8K_LOCAL`, `TULU_LOCAL`, and `CODE_LOCAL_PATH`

## Reproduce Paper Artifacts

### Table 1

```bash
python code/scripts/build_table1.py
```

Expected outputs:

- `results/summary_metrics.json`
- `results/table1_main_results.md`

### Table 2

```bash
python code/scripts/analyze_output_format.py
```

Expected outputs:

- `results/format_analysis_summary.md`
- `results/format_analysis.csv`
- `results/format_correct_vs_wrong.csv`
- `results/table2_gsm8k_format.md`

### Figure 1

```bash
python code/scripts/plot_figure1_score_format_decoupling.py
```

Expected outputs:

- `results/figures/figure1_score_format_decoupling.png`
- `results/figures/figure1_score_format_decoupling.pdf`

## Training Pipelines Included

The repository keeps the scripts needed to reproduce the six 8B methods:

- `code/scripts/run_8b_forward.sh`
- `code/scripts/run_8b_coldstart_grpo.sh`
- `code/scripts/run_8b_reverse.sh`
- `code/scripts/run_8b_joint_rl.sh`

These scripts write fresh `submission_*.json` files under `evaluation/` when rerun. The included paper-ready files in `evaluation/` are lightweight renamed copies of the best checkpoints used in the submitted paper.

## Notes

- `results/summary_metrics.json` and `results/table1_main_results.md` are generated from the included evaluation JSONs.
- `results/table2_gsm8k_format.md` and `results/format_analysis_summary.md` are generated from the included GSM8K inspect logs.
- The included figure script reads those generated result files and reproduces Figure 1.

See `REPRODUCIBILITY.md` for more detailed reproduction notes and expected reviewer outputs.