# Table 2. GSM8K Format Analysis

| Run | Avg words | Step-like % | Numbered-step % | Code-block % | Final-marker % |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8B SFT baseline | 196.80 | 94.77 | 94.16 | 78.32 | 72.18 |
| 8B IF-RLVR-only | 124.77 | 64.52 | 64.29 | 18.42 | 7.13 |
| 8B Math-RLVR-only | 83.03 | 3.79 | 3.34 | 0.23 | 0.00 |
| 8B Forward sequential final | 72.07 | 15.92 | 14.56 | 0.00 | 0.00 |
| 8B Reverse sequential final | 67.39 | 20.62 | 20.47 | 0.00 | 0.00 |
| 8B Joint RL iter25 | 152.11 | 94.92 | 94.24 | 0.08 | 0.15 |

Joint RL remains much more verbose and explicitly step-structured than the sequential forward/reverse pipelines. This supports the paper's claim that verifier order and composition affect surface format, not only aggregate score.
