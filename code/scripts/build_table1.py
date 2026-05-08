from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = REPO_ROOT / "evaluation"
SUMMARY_PATH = REPO_ROOT / "results" / "summary_metrics.json"
TABLE_PATH = REPO_ROOT / "results" / "table1_main_results.md"

RUNS = [
    ("8B SFT baseline", "evaluation/8b_sft.json"),
    ("8B IF-RLVR only", "evaluation/8b_if_rlvr_only.json"),
    ("8B Math-RLVR only", "evaluation/8b_coldstart_grpo_best.json"),
    ("8B Joint RL", "evaluation/8b_joint_rl_best.json"),
    ("8B Forward IF->Math", "evaluation/8b_forward_final.json"),
    ("8B Reverse Math->IF", "evaluation/8b_reverse_final.json"),
]

EXPECTED = {
    "8B SFT baseline": 58.81,
    "8B IF-RLVR only": 65.62,
    "8B Math-RLVR only": 68.19,
    "8B Joint RL": 69.11,
    "8B Forward IF->Math": 72.23,
    "8B Reverse Math->IF": 72.46,
}


def score(path: Path) -> dict[str, float]:
    with path.open() as handle:
        data = json.load(handle)
    ifeval = float(data["ifeval"]["metrics"]["google/IFEval/final_acc"]) * 100
    gsm8k = float(data["gsm8k"]["metrics"]["openai/gsm8k/accuracy"]) * 100
    humaneval = float(data["humaneval"]["metrics"]["openai/openai_humaneval/accuracy"]) * 100
    avg = (ifeval + gsm8k + humaneval) / 3
    return {
        "IFEval": round(ifeval, 2),
        "GSM8K": round(gsm8k, 2),
        "HumanEval": round(humaneval, 2),
        "Avg": round(avg, 2),
    }


def main() -> None:
    rows = []
    for name, rel_path in RUNS:
        metrics = score(REPO_ROOT / rel_path)
        expected = EXPECTED[name]
        if abs(metrics["Avg"] - expected) > 1e-2:
            raise ValueError(f"{name} Avg mismatch: expected {expected}, got {metrics['Avg']}")
        rows.append({"name": name, "path": rel_path, **metrics})

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(rows, indent=2) + "\n")

    lines = [
        "# Table 1. Main 8B Results",
        "",
        "| Run | IFEval | GSM8K | HumanEval | Avg |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['name']} | {row['IFEval']:.2f} | {row['GSM8K']:.2f} | {row['HumanEval']:.2f} | {row['Avg']:.2f} |"
        )
    TABLE_PATH.write_text("\n".join(lines) + "\n")

    print(f"wrote {SUMMARY_PATH}")
    print(f"wrote {TABLE_PATH}")


if __name__ == "__main__":
    main()