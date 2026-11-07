"""Report figures (matplotlib). One consistent, colour-blind-checked style for every chart."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import matplotlib.ticker

if "ipykernel" not in sys.modules:  # headless backend for CLI/CI; keep inline plots in notebooks
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve

# Categorical slots in fixed order (validated for colour-vision deficiency on adjacent pairs).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.linewidth": 2,
            "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
            "axes.prop_cycle": matplotlib.cycler(color=SERIES),
        }
    )


def _save(fig: plt.Figure, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def roc_pr_curves(y: np.ndarray, probs: dict[str, np.ndarray], path: str | Path) -> Path:
    apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    for i, (name, p) in enumerate(probs.items()):
        fpr, tpr, _ = roc_curve(y, p)
        prec, rec, _ = precision_recall_curve(y, p)
        ax1.plot(fpr, tpr, color=SERIES[i], label=name)
        ax2.plot(rec, prec, color=SERIES[i], label=name)
    ax1.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--", label="random")
    ax2.axhline(np.mean(y), color=MUTED, lw=1, ls="--", label="base rate")
    ax1.set(title="ROC curve (test set)", xlabel="False positive rate", ylabel="True positive rate")
    ax2.set(title="Precision-recall (test set)", xlabel="Recall", ylabel="Precision")
    ax1.legend(loc="lower right")
    ax2.legend(loc="upper right")
    return _save(fig, path)


def calibration_plot(y: np.ndarray, probs: dict[str, np.ndarray], path: str | Path) -> Path:
    apply_style()
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--", label="perfectly calibrated")
    for i, (name, p) in enumerate(probs.items()):
        frac_pos, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac_pos, marker="o", ms=5, color=SERIES[i], label=name)
    ax.set(
        title="Calibration (test set)",
        xlabel="Mean predicted probability of default",
        ylabel="Observed default rate",
    )
    ax.legend(loc="upper left")
    return _save(fig, path)


def cost_curve_plot(curve: pd.DataFrame, threshold: float, path: str | Path) -> Path:
    apply_style()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(curve["threshold"], curve["expected_cost"], color=SERIES[0])
    best = curve.loc[(curve["threshold"] - threshold).abs().idxmin()]
    ax.scatter([best["threshold"]], [best["expected_cost"]], s=60, color=SERIES[0], zorder=3)
    ax.annotate(
        f"cost-optimal threshold = {threshold:.2f}",
        (best["threshold"], best["expected_cost"]),
        xytext=(12, 18),
        textcoords="offset points",
        color=INK,
        fontsize=9,
    )
    ax.axvline(0.5, color=MUTED, lw=1, ls="--")
    ax.text(0.505, ax.get_ylim()[1] * 0.95, "default 0.5 cut-off", color=MUTED, fontsize=8)
    ax.set(
        title="Expected cost per applicant vs. decision threshold (validation)",
        xlabel="Decline if PD ≥ threshold",
        ylabel="Expected cost (FP = 1 unit)",
    )
    return _save(fig, path)


def importance_plot(mean_abs_contrib: pd.Series, path: str | Path, top: int = 15) -> Path:
    apply_style()
    s = mean_abs_contrib.sort_values().tail(top)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.barh(s.index, s.values, color=SERIES[0], height=0.6)
    ax.grid(axis="y", visible=False)
    ax.set(title="Global feature importance", xlabel="Mean |SHAP value| (log-odds)")
    return _save(fig, path)


def fairness_plot(reports: dict[str, pd.DataFrame], path: str | Path) -> Path:
    apply_style()
    fig, axes = plt.subplots(1, len(reports), figsize=(4 * len(reports), 4.4), sharey=True)
    for ax, (name, rep) in zip(np.atleast_1d(axes), reports.items(), strict=True):
        x = np.arange(len(rep))
        w = 0.38
        bar = {"width": w, "edgecolor": SURFACE, "linewidth": 2}
        ax.bar(x - w / 2, rep["approval_rate"], color=SERIES[0], label="approval rate", **bar)
        ax.bar(
            x + w / 2,
            1 - rep["default_rate"],
            color=SERIES[1],
            label="observed non-default rate",
            **bar,
        )
        ax.set_xticks(x, rep["group"])
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
        ax.grid(axis="x", visible=False)
        ax.set_title(name.replace("_", " "))
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=2, bbox_to_anchor=(0.99, 0.99))
    fig.suptitle("Approval rate by protected group (test set)", x=0.01, ha="left", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
