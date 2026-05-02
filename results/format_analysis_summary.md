# GSM8K Format Analysis Summary

Primary metrics are computed from the raw GSM8K inspect logs for each checkpoint.

Metric definitions:
- Avg words: average number of word tokens per GSM8K response.
- Avg non-empty lines: average number of non-blank lines per response.
- Step-like %: response contains the word step/steps or at least two numbered step lines.
- Numbered-step %: response contains at least one explicit numbered step line.
- Equation %: response contains equation-like formatting, inline math, or explicit arithmetic equalities.
- Code-block %: response contains a fenced code block.

| Run | Avg words | Avg non-empty lines | Step-like % | Numbered-step % | Equation % | Code-block % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A. 8B SFT baseline | 196.80 | 29.54 | 94.77 | 94.16 | 99.70 | 78.32 |
| B. 8B IF-RLVR-only | 124.77 | 14.82 | 64.52 | 64.29 | 98.64 | 18.42 |
| C. 8B Cold-start GRPO iter25 | 83.03 | 8.07 | 3.79 | 3.34 | 99.92 | 0.23 |
| D. 8B Forward final | 72.07 | 7.50 | 15.92 | 14.56 | 99.92 | 0.00 |
| E. 8B Reverse final | 67.39 | 5.34 | 20.62 | 20.47 | 99.92 | 0.00 |

Auxiliary observations:
- Explicit final-answer marker rate (\boxed{} or ####) falls from 72.18% in A to 0.00% in D and E.
- D vs E average numbered lines per response: 0.69 vs 0.98.
- D vs E discourse-marker rate (first/then/next/therefore/so): 73.84% vs 87.41%.

Forward vs Reverse Final Model 的格式比较：D 和 E 在粗粒度上属于同一类 GSM8K 输出风格，二者的 equation 触发率都接近满格（D 99.92% vs E 99.92%），code block 使用率都为 0%，显式最终答案标记率也都为 0.00% / 0.00%。但它们并非格式上等价：Reverse final 更短、更紧凑，平均词数从 72.07 降到 67.39，平均非空行数从 7.50 降到 5.34；同时 Reverse 更偏向显式步骤化表达，step-like 比例从 15.92% 升到 20.62%，numbered-step 比例从 14.56% 升到 20.47%，而平均等号数从 6.06 降到 4.39。因此，这组结果更支持“外显分数接近但内部策略仍不同”的解释，而不是强版本的 commutative composition produces the same internal policy。

Source logs:
- A. 8B SFT baseline: evaluation/inspect-logs/2026-04-20T23-05-43+00-00_task_GTm82dC9Xwmc2KjJ4iGDkw.eval
- B. 8B IF-RLVR-only: evaluation/inspect-logs/2026-04-20T23-37-53+00-00_task_C6rgsWqeiBU6HvVC7LYCH4.eval
- C. 8B Cold-start GRPO iter25: evaluation/inspect-logs/2026-04-30T22-26-38+00-00_task_iQ2aiQLET8oidXTTLMx76U.eval
- D. 8B Forward final: evaluation/inspect-logs/2026-04-21T00-03-49+00-00_task_Fk5WdQvmrY2AANaR6z9xmu.eval
- E. 8B Reverse final: evaluation/inspect-logs/2026-04-30T23-56-27+00-00_task_KjYfSS8rBpY8gdfUY6rvPe.eval
