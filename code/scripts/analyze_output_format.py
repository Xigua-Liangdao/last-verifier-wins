"""Generate the GSM8K format-analysis bundle for the six 8B paper runs."""

from __future__ import annotations

import csv
import re
import statistics
from dataclasses import dataclass
from pathlib import Path

from inspect_ai.log import read_eval_log_samples


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = REPO_ROOT / "results" / "format_analysis_summary.md"
CSV_PATH = REPO_ROOT / "results" / "format_analysis.csv"
CORRECT_VS_WRONG_PATH = REPO_ROOT / "results" / "format_correct_vs_wrong.csv"
TABLE2_PATH = REPO_ROOT / "results" / "table2_gsm8k_format.md"

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
    short_label: str
    checkpoint_path: str
    inspect_log_path: str


RUNS = [
    RunSpec("A. 8B SFT baseline", "8B SFT baseline", "evaluation/8b_sft.json", "evaluation/inspect-logs/2026-04-20T23-05-43+00-00_task_GTm82dC9Xwmc2KjJ4iGDkw.eval"),
    RunSpec("B. 8B IF-RLVR-only", "8B IF-RLVR-only", "evaluation/8b_if_rlvr_only.json", "evaluation/inspect-logs/2026-04-20T23-37-53+00-00_task_C6rgsWqeiBU6HvVC7LYCH4.eval"),
    RunSpec("C. 8B Math-RLVR-only", "8B Math-RLVR-only", "evaluation/8b_coldstart_grpo_best.json", "evaluation/inspect-logs/2026-04-30T22-26-38+00-00_task_iQ2aiQLET8oidXTTLMx76U.eval"),
    RunSpec("D. 8B Forward final", "8B Forward sequential final", "evaluation/8b_forward_final.json", "evaluation/inspect-logs/2026-04-21T00-03-49+00-00_task_Fk5WdQvmrY2AANaR6z9xmu.eval"),
    RunSpec("E. 8B Reverse final", "8B Reverse sequential final", "evaluation/8b_reverse_final.json", "evaluation/inspect-logs/2026-04-30T23-56-27+00-00_task_KjYfSS8rBpY8gdfUY6rvPe.eval"),
    RunSpec("F. 8B Joint RL iter25", "8B Joint RL iter25", "evaluation/8b_joint_rl_best.json", "evaluation/inspect-logs/2026-05-02T20-48-57+00-00_task_bwVEnJmAnoTo2wqExGmi83.eval"),
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
        "avg_lines": statistics.mean(len([line for line in text.splitlines() if line.strip()]) for text in texts),
        "step_like_pct": pct([bool(STEP_PHRASE_RE.search(text) or len(NUMBERED_LINE_RE.findall(text)) >= 2) for text in texts]),
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


def build_conclusion(forward: dict[str, float], reverse: dict[str, float], joint: dict[str, float]) -> str:
    return (
        "The sequential forward and reverse pipelines both converge to compact GSM8K answer styles, "
        f"with average response lengths of {format_value(forward['avg_words'])} and {format_value(reverse['avg_words'])} words, respectively. "
        "Reverse remains slightly shorter and slightly more explicitly step-structured than forward. "
        f"By contrast, the joint RL baseline is much more verbose ({format_value(joint['avg_words'])} words) and strongly preserves overt step formatting "
        f"({format_value(joint['step_like_pct'])}% step-like; {format_value(joint['numbered_steps_pct'])}% numbered-step). "
        "This supports the paper's central decoupling claim: aggregate score and surface format do not move together, and the final verifier stage strongly shapes the response style seen on GSM8K."
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
                "run": run.short_label,
                "checkpoint_json": run.checkpoint_path,
                "inspect_log": run.inspect_log_path,
                "avg_words": format_value(metrics["avg_words"]),
                "avg_non_empty_lines": format_value(metrics["avg_lines"]),
                "step_like_pct": format_value(metrics["step_like_pct"]),
                "numbered_step_pct": format_value(metrics["numbered_steps_pct"]),
                "equation_pct": format_value(metrics["equation_pct"]),
                "code_block_pct": format_value(metrics["code_block_pct"]),
                "final_marker_pct": format_value(metrics["final_marker_pct"]),
            }
        )

        correct_texts = [sample.output.completion or "" for sample in samples if correctness_flag(sample)]
        wrong_texts = [sample.output.completion or "" for sample in samples if not correctness_flag(sample)]
        for outcome, split_texts in (("correct", correct_texts), ("wrong", wrong_texts)):
            split_metrics = format_metrics_from_texts(split_texts)
            split_rows.append(
                {
                    "run": run.short_label,
                    "outcome": outcome,
                    "sample_count": str(len(split_texts)),
                    "avg_words": format_value(split_metrics["avg_words"]),
                    "avg_non_empty_lines": format_value(split_metrics["avg_lines"]),
                    "step_like_pct": format_value(split_metrics["step_like_pct"]),
                    "numbered_step_pct": format_value(split_metrics["numbered_steps_pct"]),
                    "equation_pct": format_value(split_metrics["equation_pct"]),
                    "code_block_pct": format_value(split_metrics["code_block_pct"]),
                    "final_marker_pct": format_value(split_metrics["final_marker_pct"]),
                }
            )

    forward_metrics = next(metrics for run, metrics, _ in rows if run.label.startswith("D."))
    reverse_metrics = next(metrics for run, metrics, _ in rows if run.label.startswith("E."))
    joint_metrics = next(metrics for run, metrics, _ in rows if run.label.startswith("F."))

    lines = [
        "# GSM8K Format Analysis Summary",
        "",
        "Primary metrics are computed from the raw GSM8K inspect logs for the six paper runs.",
        "",
        "Metric definitions:",
        "- Avg words: average number of word tokens per GSM8K response.",
        "- Avg non-empty lines: average number of non-blank lines per response.",
        "- Step-like %: response contains the word step/steps or at least two numbered step lines.",
        "- Numbered-step %: response contains at least one explicit numbered step line.",
        "- Equation %: response contains equation-like formatting, inline math, or explicit arithmetic equalities.",
        "- Code-block %: response contains a fenced code block.",
        "- Final-marker %: response contains an explicit final answer marker like \\boxed{} or ####.",
        "",
        "| Run | Avg words | Avg non-empty lines | Step-like % | Numbered-step % | Equation % | Code-block % | Final-marker % |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run, metrics, _ in rows:
        lines.append(
            f"| {run.label} | {format_value(metrics['avg_words'])} | {format_value(metrics['avg_lines'])} | {format_value(metrics['step_like_pct'])} | {format_value(metrics['numbered_steps_pct'])} | {format_value(metrics['equation_pct'])} | {format_value(metrics['code_block_pct'])} | {format_value(metrics['final_marker_pct'])} |"
        )

    lines.extend(
        [
            "",
            "Auxiliary observations:",
            f"- Explicit final-answer marker rate falls from {format_value(rows[0][1]['final_marker_pct'])}% in the SFT baseline to {format_value(forward_metrics['final_marker_pct'])}% / {format_value(reverse_metrics['final_marker_pct'])}% / {format_value(joint_metrics['final_marker_pct'])}% in forward / reverse / joint RL.",
            f"- Forward / reverse / joint average numbered lines per response: {format_value(forward_metrics['avg_numbered_lines'])} / {format_value(reverse_metrics['avg_numbered_lines'])} / {format_value(joint_metrics['avg_numbered_lines'])}.",
            f"- Forward / reverse / joint discourse-marker rate: {format_value(forward_metrics['discourse_marker_pct'])}% / {format_value(reverse_metrics['discourse_marker_pct'])}% / {format_value(joint_metrics['discourse_marker_pct'])}%.",
            "",
            build_conclusion(forward_metrics, reverse_metrics, joint_metrics),
            "",
            "Source logs:",
        ]
    )
    for run, _, inspect_log in rows:
        lines.append(f"- {run.short_label}: {inspect_log.relative_to(REPO_ROOT)}")

    RESULT_PATH.write_text("\n".join(lines) + "\n")

    table2_lines = [
        "# Table 2. GSM8K Format Analysis",
        "",
        "| Run | Avg words | Step-like % | Numbered-step % | Code-block % | Final-marker % |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run, metrics, _ in rows:
        table2_lines.append(
            f"| {run.short_label} | {format_value(metrics['avg_words'])} | {format_value(metrics['step_like_pct'])} | {format_value(metrics['numbered_steps_pct'])} | {format_value(metrics['code_block_pct'])} | {format_value(metrics['final_marker_pct'])} |"
        )
    table2_lines.extend(
        [
            "",
            "Joint RL remains much more verbose and explicitly step-structured than the sequential forward/reverse pipelines. This supports the paper's claim that verifier order and composition affect surface format, not only aggregate score.",
        ]
    )
    TABLE2_PATH.write_text("\n".join(table2_lines) + "\n")

    write_csv(csv_rows, CSV_PATH, [
        "run",
        "checkpoint_json",
        "inspect_log",
        "avg_words",
        "avg_non_empty_lines",
        "step_like_pct",
        "numbered_step_pct",
        "equation_pct",
        "code_block_pct",
        "final_marker_pct",
    ])
    write_csv(split_rows, CORRECT_VS_WRONG_PATH, [
        "run",
        "outcome",
        "sample_count",
        "avg_words",
        "avg_non_empty_lines",
        "step_like_pct",
        "numbered_step_pct",
        "equation_pct",
        "code_block_pct",
        "final_marker_pct",
    ])
    print(f"wrote {RESULT_PATH}")
    print(f"wrote {TABLE2_PATH}")
    print(f"wrote {CSV_PATH}")
    print(f"wrote {CORRECT_VS_WRONG_PATH}")


if __name__ == "__main__":
    main()