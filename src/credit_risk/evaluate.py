"""Model evaluation: discrimination, calibration and business cost."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
    roc_curve,
)


def ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Kolmogorov-Smirnov statistic: max separation between the score CDFs of goods and bads.

    A standard discrimination metric in credit scoring (equivalently max(TPR - FPR)).
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


def classification_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 1e-6, 1 - 1e-6)
    auc = roc_auc_score(y_true, y_prob)
    return {
        "roc_auc": float(auc),
        "gini": float(2 * auc - 1),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "ks": ks_statistic(y_true, y_prob),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, y_prob)),
    }


def expected_cost(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float, cost_fn: float, cost_fp: float
) -> float:
    """Average cost per applicant when declining everyone with PD >= threshold."""
    y_true = np.asarray(y_true)
    decline = np.asarray(y_prob) >= threshold
    fn = np.sum(~decline & (y_true == 1))  # approved, then defaulted
    fp = np.sum(decline & (y_true == 0))  # declined a good customer
    return float((cost_fn * fn + cost_fp * fp) / len(y_true))


def cost_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cost_fn: float,
    cost_fp: float,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    if thresholds is None:
        thresholds = np.round(np.arange(0.01, 1.0, 0.01), 2)
    rows = []
    y_true = np.asarray(y_true)
    for t in thresholds:
        decline = np.asarray(y_prob) >= t
        approved = ~decline
        rows.append(
            {
                "threshold": float(t),
                "expected_cost": expected_cost(y_true, y_prob, t, cost_fn, cost_fp),
                "approval_rate": float(approved.mean()),
                "bad_rate_approved": float(y_true[approved].mean()) if approved.any() else 0.0,
                "defaulters_caught": float(decline[y_true == 1].mean()),
            }
        )
    return pd.DataFrame(rows)


def optimal_threshold(
    y_true: np.ndarray, y_prob: np.ndarray, cost_fn: float, cost_fp: float
) -> tuple[float, pd.DataFrame]:
    """Threshold minimising expected cost (selected on validation data, never on test)."""
    curve = cost_curve(y_true, y_prob, cost_fn, cost_fp)
    best = curve.loc[curve["expected_cost"].idxmin()]
    return float(best["threshold"]), curve


def decision_summary(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float, cost_fn: float, cost_fp: float
) -> dict[str, float]:
    y_true = np.asarray(y_true)
    decline = np.asarray(y_prob) >= threshold
    approved = ~decline
    return {
        "threshold": float(threshold),
        "expected_cost": expected_cost(y_true, y_prob, threshold, cost_fn, cost_fp),
        "approval_rate": float(approved.mean()),
        "bad_rate_approved": float(y_true[approved].mean()) if approved.any() else 0.0,
        "defaulters_caught": float(decline[y_true == 1].mean()),
        "goods_declined": float(decline[y_true == 0].mean()),
    }
