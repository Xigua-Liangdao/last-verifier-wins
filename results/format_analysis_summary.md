# GSM8K Format Analysis Summary

Primary metrics are computed from the raw GSM8K inspect logs for the six paper runs.

Metric definitions:
- Avg words: average number of word tokens per GSM8K response.
- Avg non-empty lines: average number of non-blank lines per response.
- Step-like %: response contains the word step/steps or at least two numbered step lines.
- Numbered-step %: response contains at least one explicit numbered step line.
- Equation %: response contains equation-like formatting, inline math, or explicit arithmetic equalities.
- Code-block %: response contains a fenced code block.
- Final-marker %: response contains an explicit final answer marker like \boxed{} or ####.

| Run | Avg words | Avg non-empty lines | Step-like % | Numbered-step % | Equation % | Code-block % | Final-marker % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A. 8B SFT baseline | 196.80 | 29.54 | 94.77 | 94.16 | 99.70 | 78.32 | 72.18 |
| B. 8B IF-RLVR-only | 124.77 | 14.82 | 64.52 | 64.29 | 98.64 | 18.42 | 7.13 |
| C. 8B Math-RLVR-only | 83.03 | 8.07 | 3.79 | 3.34 | 99.92 | 0.23 | 0.00 |
| D. 8B Forward final | 72.07 | 7.50 | 15.92 | 14.56 | 99.92 | 0.00 | 0.00 |
| E. 8B Reverse final | 67.39 | 5.34 | 20.62 | 20.47 | 99.92 | 0.00 | 0.00 |
| F. 8B Joint RL iter25 | 152.11 | 18.95 | 94.92 | 94.24 | 99.01 | 0.08 | 0.15 |

Auxiliary observations:
- Explicit final-answer marker rate falls from 72.18% in the SFT baseline to 0.00% / 0.00% / 0.15% in forward / reverse / joint RL.
- Forward / reverse / joint average numbered lines per response: 0.69 / 0.98 / 3.76.
- Forward / reverse / joint discourse-marker rate: 73.84% / 87.41% / 83.93%.

The sequential forward and reverse pipelines both converge to compact GSM8K answer styles, with average response lengths of 72.07 and 67.39 words, respectively. Reverse remains slightly shorter and slightly more explicitly step-structured than forward. By contrast, the joint RL baseline is much more verbose (152.11 words) and strongly preserves overt step formatting (94.92% step-like; 94.24% numbered-step). This supports the paper's central decoupling claim: aggregate score and surface format do not move together, and the final verifier stage strongly shapes the response style seen on GSM8K.

Source logs:
- 8B SFT baseline: evaluation/inspect-logs/2026-04-20T23-05-43+00-00_task_GTm82dC9Xwmc2KjJ4iGDkw.eval
- 8B IF-RLVR-only: evaluation/inspect-logs/2026-04-20T23-37-53+00-00_task_C6rgsWqeiBU6HvVC7LYCH4.eval
- 8B Math-RLVR-only: evaluation/inspect-logs/2026-04-30T22-26-38+00-00_task_iQ2aiQLET8oidXTTLMx76U.eval
- 8B Forward sequential final: evaluation/inspect-logs/2026-04-21T00-03-49+00-00_task_Fk5WdQvmrY2AANaR6z9xmu.eval
- 8B Reverse sequential final: evaluation/inspect-logs/2026-04-30T23-56-27+00-00_task_KjYfSS8rBpY8gdfUY6rvPe.eval
- 8B Joint RL iter25: evaluation/inspect-logs/2026-05-02T20-48-57+00-00_task_bwVEnJmAnoTo2wqExGmi83.eval
