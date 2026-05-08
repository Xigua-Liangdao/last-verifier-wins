from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SUMMARY_PATH = REPO_ROOT / "results" / "summary_metrics.json"
FORMAT_TABLE_PATH = REPO_ROOT / "results" / "table2_gsm8k_format.md"
OUTPUT_STEM = REPO_ROOT / "results" / "figures" / "figure1_score_format_decoupling"

PALETTE = {
    "sft": "#CFCECE",
    "if_only": "#AADCA9",
    "math_only": "#8BCF8B",
    "joint": "#E9A6A1",
    "forward": "#0F4D92",
    "reverse": "#3775BA",
}

DISPLAY_LABELS = [
    "SFT",
    "IF-RLVR\nonly",
    "Math-RLVR\nonly",
    "Joint\nRL",
    "Forward\nIF→\nMath",
    "Reverse\nMath→\nIF",
]

COLOR_ORDER = [
    PALETTE["sft"],
    PALETTE["if_only"],
    PALETTE["math_only"],
    PALETTE["joint"],
    PALETTE["forward"],
    PALETTE["reverse"],
]

SCORE_RUN_ORDER = [
    "8B SFT baseline",
    "8B IF-RLVR only",
    "8B Math-RLVR only",
    "8B Joint RL",
    "8B Forward IF->Math",
    "8B Reverse Math->IF",
]

FORMAT_RUN_ORDER = [
    "8B SFT baseline",
    "8B IF-RLVR-only",
    "8B Math-RLVR-only",
    "8B Joint RL iter25",
    "8B Forward sequential final",
    "8B Reverse sequential final",
]


def apply_publication_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 15,
            "font.family": "DejaVu Sans",
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 2.2,
            "xtick.major.width": 1.8,
            "ytick.major.width": 1.8,
            "xtick.major.size": 6,
            "ytick.major.size": 6,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_avg_scores() -> list[float]:
    rows = json.loads(SUMMARY_PATH.read_text())
    by_name = {row["name"]: row for row in rows}
    return [float(by_name[name]["Avg"]) for name in SCORE_RUN_ORDER]


def load_avg_words() -> list[float]:
    rows: dict[str, float] = {}
    for line in FORMAT_TABLE_PATH.read_text().splitlines():
        if not line.startswith("| "):
            continue
        if line.startswith("| Run ") or line.startswith("| ---"):
            continue
        parts = [part.strip() for part in line.strip().split("|")[1:-1]]
        if len(parts) != 6:
            continue
        rows[parts[0]] = float(parts[1])
    return [rows[name] for name in FORMAT_RUN_ORDER]


def annotate_bars(ax: plt.Axes, bars, values: list[float], fmt: str, offset: float) -> None:
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + offset,
            fmt.format(value),
            ha="center",
            va="bottom",
            fontsize=10.5,
            color="#222222",
        )


def make_panel(
    ax: plt.Axes,
    values: list[float],
    ylabel: str,
    title: str,
    panel_label: str,
    ylim: tuple[float, float],
    note: str,
    fmt: str,
    offset: float,
    note_x: float = 0.02,
    note_y: float = 0.98,
    note_ha: str = "left",
) -> None:
    x = np.arange(len(DISPLAY_LABELS))
    bars = ax.bar(x, values, width=0.72, color=COLOR_ORDER, edgecolor="#333333", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(DISPLAY_LABELS)
    ax.tick_params(axis="x", labelsize=11, pad=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=12, fontsize=18)
    ax.set_ylim(*ylim)
    ax.margins(x=0.06)
    ax.grid(axis="y", color="#E6E6E6", linewidth=1.0)
    ax.set_axisbelow(True)
    ax.text(-0.13, 1.05, panel_label, transform=ax.transAxes, fontsize=24, fontweight="bold")
    ax.text(note_x, note_y, note, transform=ax.transAxes, ha=note_ha, va="top", fontsize=11, color="#555555")
    annotate_bars(ax, bars, values, fmt=fmt, offset=offset)


def main() -> None:
    apply_publication_style()
    avg_scores = load_avg_scores()
    avg_words = load_avg_words()

    fig, axes = plt.subplots(1, 2, figsize=(14.8, 6.2), constrained_layout=True)

    make_panel(axes[0], avg_scores, "Avg accuracy", "Average task score", "A", (56, 75), "Higher is better", "{:.2f}", 0.32)
    make_panel(axes[1], avg_words, "Avg words on GSM8K", "GSM8K output format", "B", (55, 210), "Lower is better", "{:.2f}", 3.5, note_x=0.98, note_ha="right")

    OUTPUT_STEM.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        fig.savefig(OUTPUT_STEM.with_suffix(suffix), dpi=400, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"wrote {OUTPUT_STEM.with_suffix('.png')}")
    print(f"wrote {OUTPUT_STEM.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()