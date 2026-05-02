# Reverse vs Forward 8B RLVR

| Run | IFEval | GSM8K | HumanEval | Avg |
| --- | ---: | ---: | ---: | ---: |
| 8B SFT (baseline) | 68.82 | 58.23 | 49.39 | 58.81 |
| 8B IF-RLVR-only | 82.03 | 65.43 | 49.39 | 65.62 |
| 8B Cold-start GRPO iter25 | 70.52 | **79.45** | 54.27 | 68.08 |
| 8B Forward (IF→math) final | 81.60 | 78.39 | **56.71** | 72.23 |
| 8B Reverse (math→IF) final | **83.58** | 77.10 | **56.71** | **72.46** |

Files:
- 8B SFT (baseline): evaluation/submission_8b_sft_full.json
- 8B IF-RLVR-only: evaluation/submission_8b_if_rlvr.json
- 8B Cold-start GRPO iter25: evaluation/submission_8b_coldstart_grpo_iter25.json
- 8B Forward (IF→math) final: evaluation/submission_8b_final.json
- 8B Reverse (math→IF) final: evaluation/submission_8b_reverse_iter40.json
