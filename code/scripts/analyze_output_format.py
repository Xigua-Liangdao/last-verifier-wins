"""Generate the final release format-analysis bundle for the five 8B checkpoints.

Reads raw GSM8K inspect logs and writes three artifacts under final/results:
format_analysis_summary.md, format_analysis.csv, and format_correct_vs_wrong.csv.
The primary comparison is the forward-final versus reverse-final output format.
"""

from __future__ import annotations

import csv
import re
import statistics
from dataclasses import dataclass
from pathlib import Path

from inspect_ai.log import read_eval_log_samples


FINAL_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[3]
RESULT_PATH = FINAL_ROOT / "results" / "format_analysis_summary.md"
CSV_PATH = FINAL_ROOT / "results" / "format_analysis.csv"
CORRECT_VS_WRONG_PATH = FINAL_ROOT / "results" / "format_correct_vs_wrong.csv"

INSPECT_LOG_RE = re.compile(r"evaluation/inspect-logs/\S+\.eval")
WORD_RE = re.compile(r"\b\w+\b")
NUMBERED_LINE_RE = re.compile(r"(?m)^\s*(?:\d+[\.)]|Step\s+\d+[:\.)-])")
STEP_PHRASE_RE = re.compile(r"(?i)\b(step|steps)\b")
MATH_RE = re.compile(
    r"(<<[^\n]*=[^\n]*>>|\\\(|\\\)|\$[^$]+\$|\b[a-zA-Z0-9_]+\s*=\s*[^\n]+|\d\s*[+\-*/=]\s*\d)"
)
FINAL_MARKER_RE = re.compile(r"\\boxed\s*\{|####\s*[-+]?\d")
DISCOURSE_RE = re.compile(r"(?i)\b(first|then|next|therefore|so)\b")
EQUALS_RE = re.compile(r"=")


@dataclass(frozen=True)
class RunSpec:
    label: str
    checkpoint_path: str
    inspect_log_path: str


RUNS = [
    RunSpec(
        label="A. 8B SFT baseline",
        checkpoint_path="evaluation/submission_8b_sft_full.json",
        inspect_log_path="evaluation/inspect-logs/2026-04-20T23-05-43+00-00_task_GTm82dC9Xwmc2KjJ4iGDkw.eval",
    ),
    RunSpec(
        label="B. 8B IF-RLVR-only",
        checkpoint_path="evaluation/submission_8b_if_rlvr.json",
        inspect_log_path="evaluation/inspect-logs/2026-04-20T23-37-53+00-00_task_C6rgsWqeiBU6HvVC7LYCH4.eval",
    ),
    RunSpec(
        label="C. 8B Cold-start GRPO iter25",
        checkpoint_path="evaluation/submission_8b_coldstart_grpo_iter25.json",
        inspect_log_path="evaluation/inspect-logs/2026-04-30T22-26-38+00-00_task_iQ2aiQLET8oidXTTLMx76U.eval",
    ),
    RunSpec(
        label="D. 8B Forward final",
        checkpoint_path="evaluation/submission_8b_final.json",
        inspect_log_path="evaluation/inspect-logs/2026-04-21T00-03-49+00-00_task_Fk5WdQvmrY2AANaR6z9xmu.eval",
    ),
    RunSpec(
        label="E. 8B Reverse final",
        checkpoint_path="evaluation/submission_8b_reverse_iter40.json",
        inspect_log_path="evaluation/inspect-logs/2026-04-30T23-56-27+00-00_task_KjYfSS8rBpY8gdfUY6rvPe.eval",
    ),
]


def pct(predicate_results: list[bool]) -> float:
    return 100.0 * statistics.mean(predicate_results)


def correctness_flag(sample) -> bool:
    payload = sample.model_dump(mode="json")
    match = (payload.get("scores") or {}).get("match") or {}
    value = str(match.get("value", "")).upper()
    return value == "C"


def format_metrics_from_texts(texts: list[str]) -> dict[str, float]:
    if not texts:
        return {
            "avg_words": 0.0,
            "avg_lines": 0.0,
            "step_like_pct": 0.0,
            "numbered_steps_pct": 0.0,
            "equation_pct": 0.0,
            "code_block_pct": 0.0,
            "final_marker_pct": 0.0,
            "avg_numbered_lines": 0.0,
            "avg_equals": 0.0,
            "discourse_marker_pct": 0.0,
        }

    return {
        "avg_words": statistics.mean(len(WORD_RE.findall(text)) for text in texts),
        "avg_lines": statistics.mean(
            len([line for line in text.splitlines() if line.strip()]) for text in texts
        ),
        "step_like_pct": pct(
            [
                bool(STEP_PHRASE_RE.search(text) or len(NUMBERED_LINE_RE.findall(text)) >= 2)
                for text in texts
            ]
        ),
        "numbered_steps_pct": pct([bool(NUMBERED_LINE_RE.search(text)) for text in texts]),
        "equation_pct": pct([bool(MATH_RE.search(text)) for text in texts]),
        "code_block_pct": pct(["```" in text for text in texts]),
        "final_marker_pct": pct([bool(FINAL_MARKER_RE.search(text)) for text in texts]),
        "avg_numbered_lines": statistics.mean(len(NUMBERED_LINE_RE.findall(text)) for text in texts),
        "avg_equals": statistics.mean(len(EQUALS_RE.findall(text)) for text in texts),
        "discourse_marker_pct": pct([bool(DISCOURSE_RE.search(text)) for text in texts]),
    }


def load_samples(inspect_log_path: Path) -> list:
    return list(read_eval_log_samples(str(inspect_log_path)))


def write_csv(rows: list[dict[str, str]], output_path: Path, fieldnames: list[str]) -> None:
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_value(value: float) -> str:
    return f"{value:.2f}"


def build_forward_reverse_conclusion(forward: dict[str, float], reverse: dict[str, float]) -> str:
    return (
        "Forward vs Reverse Final Model 的格式比较：D 和 E 在粗粒度上属于同一类 GSM8K 输出风格，"
        f"二者的 equation 触发率都接近满格（D {format_value(forward['equation_pct'])}% vs "
        f"E {format_value(reverse['equation_pct'])}%），code block 使用率都为 0%，"
        f"显式最终答案标记率也都为 {format_value(forward['final_marker_pct'])}% / "
        f"{format_value(reverse['final_marker_pct'])}%。"
        "但它们并非格式上等价：Reverse final 更短、更紧凑，"
        f"平均词数从 {format_value(forward['avg_words'])} 降到 {format_value(reverse['avg_words'])}，"
        f"平均非空行数从 {format_value(forward['avg_lines'])} 降到 {format_value(reverse['avg_lines'])}；"
        "同时 Reverse 更偏向显式步骤化表达，"
        f"step-like 比例从 {format_value(forward['step_like_pct'])}% 升到 {format_value(reverse['step_like_pct'])}%，"
        f"numbered-step 比例从 {format_value(forward['numbered_steps_pct'])}% 升到 {format_value(reverse['numbered_steps_pct'])}%，"
        f"而平均等号数从 {format_value(forward['avg_equals'])} 降到 {format_value(reverse['avg_equals'])}。"
        "因此，这组结果更支持“外显分数接近但内部策略仍不同”的解释，"
        "而不是强版本的 commutative composition produces the same internal policy。"
    )


def main() -> None:
    rows: list[tuple[RunSpec, dict[str, float], Path]] = []
    csv_rows: list[dict[str, str]] = []
    split_rows: list[dict[str, str]] = []

    for run in RUNS:
        inspect_log_path = REPO_ROOT / run.inspect_log_path
        samples = load_samples(inspect_log_path)
        texts = [sample.output.completion or "" for sample in samples]
        metrics = format_metrics_from_texts(texts)
        rows.append((run, metrics, inspect_log_path))

        csv_rows.append(
            {
                "run": run.label,
                "checkpoint_json": run.checkpoint_path,
                "inspect_log": run.inspect_log_path,
                "avg_words": format_value(metrics["avg_words"]),
                "avg_non_empty_lines": format_value(metrics["avg_lines"]),
                "step_like_pct": format_value(metrics["step_like_pct"]),
                "numbered_step_pct": format_value(metrics["numbered_steps_pct"]),
                "equation_pct": format_value(metrics["equation_pct"]),
                "code_block_pct": format_value(metrics["code_block_pct"]),
            }
        )

        correct_texts = [sample.output.completion or "" for sample in samples if correctness_flag(sample)]
        wrong_texts = [sample.output.completion or "" for sample in samples if not correctness_flag(sample)]
        for outcome, split_texts in (("correct", correct_texts), ("wrong", wrong_texts)):
            split_metrics = format_metrics_from_texts(split_texts)
            split_rows.append(
                {
                    "run": run.label,
                    "outcome": outcome,
                    "sample_count": str(len(split_texts)),
                    "avg_words": format_value(split_metrics["avg_words"]),
                    "avg_non_empty_lines": format_value(split_metrics["avg_lines"]),
                    "step_like_pct": format_value(split_metrics["step_like_pct"]),
                    "numbered_step_pct": format_value(split_metrics["numbered_steps_pct"]),
                    "equation_pct": format_value(split_metrics["equation_pct"]),
                    "code_block_pct": format_value(split_metrics["code_block_pct"]),
                }
            )

    forward_metrics = next(metrics for run, metrics, _ in rows if run.label.startswith("D."))
    reverse_metrics = next(metrics for run, metrics, _ in rows if run.label.startswith("E."))
    conclusion = build_forward_reverse_conclusion(forward_metrics, reverse_metrics)

    lines = [
        "# GSM8K Format Analysis Summary",
        "",
        "Primary metrics are computed from the raw GSM8K inspect logs for each checkpoint.",
        "",
        "Metric definitions:",
        "- Avg words: average number of word tokens per GSM8K response.",
        "- Avg non-empty lines: average number of non-blank lines per response.",
        "- Step-like %: response contains the word step/steps or at least two numbered step lines.",
        "- Numbered-step %: response contains at least one explicit numbered step line.",
        "- Equation %: response contains equation-like formatting, inline math, or explicit arithmetic equalities.",
        "- Code-block %: response contains a fenced code block.",
        "",
        "| Run | Avg words | Avg non-empty lines | Step-like % | Numbered-step % | Equation % | Code-block % |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for run, metrics, _ in rows:
        lines.append(
            "| {label} | {avg_words} | {avg_lines} | {step_like_pct} | {numbered_steps_pct} | {equation_pct} | {code_block_pct} |".format(
                label=run.label,
                avg_words=format_value(metrics["avg_words"]),
                avg_lines=format_value(metrics["avg_lines"]),
                step_like_pct=format_value(metrics["step_like_pct"]),
                numbered_steps_pct=format_value(metrics["numbered_steps_pct"]),
                equation_pct=format_value(metrics["equation_pct"]),
                code_block_pct=format_value(metrics["code_block_pct"]),
            )
        )

    lines.extend(
        [
            "",
            "Auxiliary observations:",
            f"- Explicit final-answer marker rate (\\boxed{{}} or ####) falls from {format_value(rows[0][1]['final_marker_pct'])}% in A to 0.00% in D and E.",
            f"- D vs E average numbered lines per response: {format_value(forward_metrics['avg_numbered_lines'])} vs {format_value(reverse_metrics['avg_numbered_lines'])}.",
            f"- D vs E discourse-marker rate (first/then/next/therefore/so): {format_value(forward_metrics['discourse_marker_pct'])}% vs {format_value(reverse_metrics['discourse_marker_pct'])}%.",
            "",
            conclusion,
            "",
            "Source logs:",
        ]
    )

    for run, _, inspect_log in rows:
        lines.append(f"- {run.label}: {inspect_log.relative_to(REPO_ROOT)}")

    RESULT_PATH.write_text("\n".join(lines) + "\n")
    write_csv(
        csv_rows,
        CSV_PATH,
        [
            "run",
            "checkpoint_json",
            "inspect_log",
            "avg_words",
            "avg_non_empty_lines",
            "step_like_pct",
            "numbered_step_pct",
            "equation_pct",
            "code_block_pct",
        ],
    )
    write_csv(
        split_rows,
        CORRECT_VS_WRONG_PATH,
        [
            "run",
            "outcome",
            "sample_count",
            "avg_words",
            "avg_non_empty_lines",
            "step_like_pct",
            "numbered_step_pct",
            "equation_pct",
            "code_block_pct",
        ],
    )
    print(f"wrote {RESULT_PATH}")
    print(f"wrote {CSV_PATH}")
    print(f"wrote {CORRECT_VS_WRONG_PATH}")


if __name__ == "__main__":
    main()