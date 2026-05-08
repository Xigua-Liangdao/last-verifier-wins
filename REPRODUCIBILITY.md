# Reproducibility Guide

This file lists the minimal commands needed to reproduce the paper-facing artifacts from the anonymous bundle.

## Assumptions

- Run from the repository root.
- Use a Python environment with `requirements.txt` installed.
- The included evaluation JSONs and inspect logs are already present under `evaluation/`.

## Fast Path: Regenerate Tables and Figure Without Training

### Table 1: main 8B comparison

```bash
python code/scripts/build_table1.py
```

Outputs:

- `results/summary_metrics.json`
- `results/table1_main_results.md`

### Table 2: GSM8K format analysis

```bash
python code/scripts/analyze_output_format.py
```

Outputs:

- `results/format_analysis_summary.md`
- `results/format_analysis.csv`
- `results/format_correct_vs_wrong.csv`
- `results/table2_gsm8k_format.md`

### Figure 1: Score-Format Decoupling

```bash
python code/scripts/plot_figure1_score_format_decoupling.py
```

Outputs:

- `results/figures/figure1_score_format_decoupling.png`
- `results/figures/figure1_score_format_decoupling.pdf`

## Optional: Re-run Training Pipelines

The following scripts are included for completeness:

- `bash code/scripts/run_8b_forward.sh`
- `bash code/scripts/run_8b_coldstart_grpo.sh`
- `bash code/scripts/run_8b_reverse.sh`
- `bash code/scripts/run_8b_joint_rl.sh`

These scripts require additional external resources:

- `TINKER_API_KEY`
- `LLAMA3_TOKENIZER_DIR` or a tokenizer prepared under `evaluation/local_tokenizers/`
- local training data snapshots exposed through variables such as `GSM8K_LOCAL`, `TULU_LOCAL`, and `CODE_LOCAL_PATH`

Use generic local paths such as `data/gsm8k_train` rather than machine-specific paths.

## Paper Consistency Targets

The generated outputs should contain the following values:

- SFT Avg: 58.81
- IF-RLVR only Avg: 65.62
- Math-RLVR only Avg: 68.19
- Joint RL Avg: 69.11
- Forward IF->Math Avg: 72.23
- Reverse Math->IF Avg: 72.46
- Joint RL GSM8K avg words: 152.11
- Forward GSM8K avg words: 72.07
- Reverse GSM8K avg words: 67.39