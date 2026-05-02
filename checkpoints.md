# Checkpoint Inventory

This file lists the checkpoints and submission files that matter for the final paper/release bundle.

For the 8B runs below, the preferred source is the corresponding `logs/*/checkpoint_info.json` file. Three runs expose a true `final_state_path` in that metadata. Two older GRPO runs were executed before `evaluation/grpo_gsm8k.py` saved state checkpoints, so only the original published sampler checkpoint path is available; those exceptions are marked explicitly instead of being reconstructed.

## 8B Final Checkpoints

| Alias | Stage | Key config | Tinker checkpoint path | Scores (IFEval / GSM8K / HumanEval / Avg) | Evidence |
| --- | --- | --- | --- | --- | --- |
| 8B-SFT | Multi-task SFT baseline | rank 128, batch 128, epochs 2, lr 2.856415e-4, total steps 526 | `tinker://35c2859c-c919-562f-b4c6-a743562ce61c:train:0/weights/8b_sft_final_step000526` | 68.82 / 58.23 / 49.39 / 58.81 | logs/8b_sft/checkpoint_info.json, final/evaluation/8b_sft.json |
| 8B-IF-RLVR-only | SFT -> IF-RLVR | rank 32, batch 64, group 8, lr 2e-5, max constraints 6, iter 40 | `tinker://abc7d04e-cc29-5c1c-9210-42c963b63871:train:0/weights/8b_if_rlvr_final_iter0040` | 82.03 / 65.43 / 49.39 / 65.62 | logs/8b_if_rlvr/checkpoint_info.json, final/evaluation/8b_if_rlvr_only.json |
| 8B-Cold-start-GRPO-best | SFT -> GSM8K GRPO, skip IF-RLVR | rank 32, batch 128, group 16, lr 3e-5, iter 20 selected by full eval | `tinker://f37f76c2-d4b0-5f62-bd37-40a8ad2d5397:train:0/sampler_weights/8b_coldstart_grpo_mid_iter0020` | 70.46 / 78.62 / 55.49 / 68.19 | logs/8b_coldstart_grpo/checkpoint_info.json, final/evaluation/8b_coldstart_grpo_best.json |
| 8B-Forward-Final | SFT -> IF-RLVR -> GSM8K GRPO | IF-RLVR as above, then GRPO rank 32, batch 128, group 16, lr 3e-5, iter 25 selected by full eval | `tinker://1bfba26c-9e28-59b6-9109-e7d97efdb364:train:0/sampler_weights/8b_grpo_final_iter0025` | 81.60 / 78.39 / 56.71 / 72.23 | logs/8b_grpo/checkpoint_info.json, final/evaluation/8b_forward_final.json |
| 8B-Reverse-Final | SFT -> GSM8K GRPO -> IF-RLVR | GRPO rank 32, batch 128, group 16, lr 3e-5, iter 25; then IF-RLVR rank 32, batch 64, group 8, lr 2e-5, iter 40 | `tinker://3ada1dde-1494-5935-9670-eeeebfb105a8:train:0/weights/8b_reverse_ifrlvr_final_iter0040` | 83.58 / 77.10 / 56.71 / 72.46 | logs/8b_reverse_ifrlvr/checkpoint_info.json, final/evaluation/8b_reverse_final.json |

### Notes on checkpoint provenance

- 8B-SFT, 8B-IF-RLVR-only, and 8B-Reverse-Final expose real `final_state_path` values in `logs/*/checkpoint_info.json`.
- 8B-Cold-start-GRPO-best and 8B-Forward-Final come from older GRPO runs that only saved sampler checkpoints. Their original published sampler paths are preserved above and in the matching evaluation JSONs.
- 8B-Cold-start-GRPO-best is the iter20 checkpoint because it had the best full-eval average among iter5/10/15/20/25.
- 8B-Forward-Final is the iter25 checkpoint because full eval beat iter20 even though reward peaked earlier.

## 3B Summary

These 3B runs are retained as screening, ablation, and negative-result references rather than as final methods.

| Run | Role | Tinker checkpoint path | Scores (IFEval / GSM8K / HumanEval / Avg) |
| --- | --- | --- | --- |
| 3B earlier SFT | early baseline | `tinker://61996fea-a462-5b33-a094-a6e78ed9eaae:train:0/sampler_weights/sft_final_step000526` | 57.64 / 39.95 / 37.80 / 45.13 |
| 3B pipeline SFT | cleaned pipeline SFT | `tinker://8c1631ad-b440-5904-8f6a-3add93509c91:train:0/sampler_weights/pipeline3b_sft_final_step000526` | 57.06 / 37.38 / 35.98 / 43.47 |
| 3B IF-RLVR v2 | instruction-following RL ablation | `tinker://8d78d35e-510d-5eb5-8066-2956455bde66:train:0/sampler_weights/ifrlvr_v1sft_final_iter0050` | 73.90 / 40.11 / 30.49 / 48.16 |
| 3B cold-start GRPO | negative control; collapsed on code | `tinker://bc3391c8-2158-528a-b8af-ffe60bc97655:train:0/sampler_weights/grpo3b_fresh_final_iter0040` | 23.65 / 42.15 / 0.61 / 22.14 |
| 3B joint RL v6 | screening-era multitask/curriculum reference | `tinker://37f37ef1-3b16-58d5-b167-5e7cf082cf1f:train:0/sampler_weights/screening_curriculum_lr2e5_sched_step600` | 34.93 / 34.80 / 32.32 / 34.02 |

## Files Included in final/evaluation

- final/evaluation/8b_sft.json
- final/evaluation/8b_if_rlvr_only.json
- final/evaluation/8b_coldstart_grpo_best.json
- final/evaluation/8b_forward_final.json
- final/evaluation/8b_reverse_final.json