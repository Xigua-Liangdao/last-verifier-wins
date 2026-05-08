# Checkpoint Inventory

This file summarizes the six 8B checkpoints used in the submitted paper and the corresponding lightweight evaluation files included in this anonymous bundle.

| Alias | Stage | Included evaluation file | Scores (IFEval / GSM8K / HumanEval / Avg) |
| --- | --- | --- | --- |
| 8B-SFT | Multi-task SFT baseline | `evaluation/8b_sft.json` | 68.82 / 58.23 / 49.39 / 58.81 |
| 8B-IF-RLVR-only | SFT -> IF-RLVR | `evaluation/8b_if_rlvr_only.json` | 82.03 / 65.43 / 49.39 / 65.62 |
| 8B-Math-RLVR-only | SFT -> cold-start GSM8K GRPO | `evaluation/8b_coldstart_grpo_best.json` | 70.46 / 78.62 / 55.49 / 68.19 |
| 8B-Joint-RL | simultaneous GSM8K + IF reward optimization | `evaluation/8b_joint_rl_best.json` | 76.35 / 76.72 / 54.27 / 69.11 |
| 8B-Forward-Final | SFT -> IF-RLVR -> GSM8K GRPO | `evaluation/8b_forward_final.json` | 81.60 / 78.39 / 56.71 / 72.23 |
| 8B-Reverse-Final | SFT -> GSM8K GRPO -> IF-RLVR | `evaluation/8b_reverse_final.json` | 83.58 / 77.10 / 56.71 / 72.46 |

## Provenance Notes

- The included evaluation JSONs are lightweight copies of the paper checkpoints selected for the final tables and figure.
- The GSM8K format analysis additionally depends on six inspect logs stored under `evaluation/inspect-logs/`.
- Re-running the training scripts in `code/scripts/` will generate fresh `submission_*.json` files under `evaluation/`; the included renamed files above are the reviewer-facing copies used by the paper tables.