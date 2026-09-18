"""One matplotlib style for every figure in the round-2 reports."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import FIGURES

INK = "#1b1f24"
MUTED = "#6b7280"
GRID = "#e5e7eb"
ACCENT = "#2563eb"
SERIES = ["#2563eb", "#ea7317", "#0f9d58", "#b23a48", "#7c4dff", "#00897b"]
SEQ = "Blues"

plt.rcParams.update({
    "figure.dpi": 160,
    "savefig.dpi": 160,
    "savefig.bbox": "tight",
    "font.family": "DejaVu Sans",
    "font.size": 8.5,
    "axes.titlesize": 9.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 8.5,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.frameon": False,
    "legend.fontsize": 7.5,
})


def despine(ax, keep=("left", "bottom")):
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep)
    return ax


def save(fig, name: str) -> str:
    """Write a figure to reports/figures and return its filename."""
    path = FIGURES / name
    fig.savefig(path)
    plt.close(fig)
    print(f"    figure: {name}")
    return name
