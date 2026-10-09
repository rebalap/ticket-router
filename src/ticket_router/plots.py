"""Matplotlib figures shared by the eval report and the Streamlit app."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from ticket_router.labels import INTENTS
from ticket_router.metrics import CalibrationBin, ThresholdPoint


def cost_confusion_figure(matrix: np.ndarray, title: str) -> Figure:
    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.imshow(matrix, cmap="Reds")
    ax.set_xticks(range(len(INTENTS)), INTENTS, rotation=45, ha="right")
    ax.set_yticks(range(len(INTENTS)), INTENTS)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    for i in range(len(INTENTS)):
        for j in range(len(INTENTS)):
            if matrix[i, j]:
                ax.text(j, i, f"{matrix[i, j]:.0f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    return fig


def threshold_sweep_figure(points: list[ThresholdPoint], chosen: float, title: str) -> Figure:
    t = [p.threshold for p in points]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(t, [p.cost_per_ticket for p in points], lw=2, label="total cost / ticket")
    ax.plot(t, [p.misroute_cost / p.n for p in points], "--", label="misroute cost / ticket")
    ax.plot(t, [p.escalation_cost / p.n for p in points], ":", label="human triage cost / ticket")
    ax.axvline(chosen, color="black", lw=0.8)
    ax.set_xlabel("confidence threshold (escalate below)")
    ax.set_ylabel("cost per ticket")
    ax.set_title(title)
    ax2 = ax.twinx()
    ax2.plot(t, [p.coverage for p in points], color="grey", alpha=0.5, label="auto-routed share")
    ax2.set_ylabel("auto-routed share")
    ax2.set_ylim(0, 1.05)
    lines = ax.get_legend_handles_labels()
    lines2 = ax2.get_legend_handles_labels()
    ax.legend(lines[0] + lines2[0], lines[1] + lines2[1], loc="upper left", fontsize=8)
    fig.tight_layout()
    return fig


def reliability_figure(curves: dict[str, list[CalibrationBin]], title: str) -> Figure:
    """Confidence vs actual accuracy per model; the diagonal is perfect calibration."""
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], color="grey", lw=0.8, ls="--", label="perfectly calibrated")
    for label, bins in curves.items():
        ax.plot(
            [b.mean_confidence for b in bins],
            [b.accuracy for b in bins],
            marker="o",
            label=label,
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("model confidence")
    ax.set_ylabel("actual accuracy")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    return fig
