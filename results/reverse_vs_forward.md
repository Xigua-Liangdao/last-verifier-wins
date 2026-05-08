# Reverse vs Forward 8B RLVR

| Run | IFEval | GSM8K | HumanEval | Avg |
| --- | ---: | ---: | ---: | ---: |
| 8B SFT (baseline) | 68.82 | 58.23 | 49.39 | 58.81 |
| 8B IF-RLVR-only | 82.03 | 65.43 | 49.39 | 65.62 |
| 8B Cold-start GRPO iter25 | 70.52 | **79.45** | 54.27 | 68.08 |
| 8B Forward (IF→math) final | 81.60 | 78.39 | **56.71** | 72.23 |
| 8B Reverse (math→IF) final | **83.58** | 77.10 | **56.71** | **72.46** |

Files:
- 8B SFT (baseline): evaluation/8b_sft.json
- 8B IF-RLVR-only: evaluation/8b_if_rlvr_only.json
- 8B Cold-start GRPO best: evaluation/8b_coldstart_grpo_best.json
- 8B Forward (IF→math) final: evaluation/8b_forward_final.json
- 8B Reverse (math→IF) final: evaluation/8b_reverse_final.json
