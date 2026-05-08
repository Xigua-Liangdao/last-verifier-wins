# 8B Joint RL vs Sequential RL

| Run | IFEval | GSM8K | HumanEval | Avg |
| --- | ---: | ---: | ---: | ---: |
| 8B SFT baseline | 68.82 | 58.23 | 49.39 | 58.81 |
| 8B IF-RLVR-only | 82.03 | 65.43 | 49.39 | 65.62 |
| 8B Cold-start GRPO best | 70.46 | **78.62** | 55.49 | 68.19 |
| 8B Joint RL best | 76.35 | 76.72 | 54.27 | 69.11 |
| 8B Forward sequential final (IF->math) | 81.60 | 78.39 | **56.71** | 72.23 |
| 8B Reverse sequential final (math->IF) | **83.58** | 77.10 | **56.71** | **72.46** |

## Takeaways

- Joint RL improves over the math-only cold-start control on Avg because it recovers substantially more instruction-following.
- Joint RL still underperforms both sequential pipelines on overall capability.
- The reverse sequential control remains the strongest all-around checkpoint in this bundle.

## Files

- 8B SFT baseline: evaluation/8b_sft.json
- 8B IF-RLVR-only: evaluation/8b_if_rlvr_only.json
- 8B Cold-start GRPO best: evaluation/8b_coldstart_grpo_best.json
- 8B Joint RL best: evaluation/8b_joint_rl_best.json
- 8B Forward sequential final: evaluation/8b_forward_final.json
- 8B Reverse sequential final: evaluation/8b_reverse_final.json