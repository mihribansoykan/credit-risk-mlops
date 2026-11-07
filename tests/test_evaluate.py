import numpy as np
import pytest

from credit_risk.evaluate import (
    classification_metrics,
    decision_summary,
    expected_cost,
    ks_statistic,
    optimal_threshold,
)


def test_ks_perfect_and_random():
    y = np.array([0, 0, 1, 1])
    assert ks_statistic(y, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)
    assert ks_statistic(y, np.array([0.5, 0.5, 0.5, 0.5])) == pytest.approx(0.0)


def test_classification_metrics_keys():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    m = classification_metrics(y, rng.random(200))
    assert set(m) == {"roc_auc", "gini", "pr_auc", "ks", "brier", "log_loss"}
    assert m["gini"] == pytest.approx(2 * m["roc_auc"] - 1)


def test_expected_cost_hand_computed():
    y = np.array([1, 1, 0, 0])
    p = np.array([0.9, 0.2, 0.7, 0.1])
    # threshold 0.5: declines rows 0 and 2 -> 1 FN (row 1), 1 FP (row 2)
    assert expected_cost(y, p, 0.5, cost_fn=5, cost_fp=1) == pytest.approx((5 + 1) / 4)


def test_optimal_threshold_lowers_with_expensive_false_negatives():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 2000)
    p = np.clip(0.3 * y + rng.random(2000) * 0.7, 0, 1)
    t_cheap, _ = optimal_threshold(y, p, cost_fn=1, cost_fp=1)
    t_expensive, _ = optimal_threshold(y, p, cost_fn=10, cost_fp=1)
    assert t_expensive <= t_cheap


def test_decision_summary_approve_all():
    y = np.array([0, 1, 0, 0])
    s = decision_summary(y, np.zeros(4), 1.0, cost_fn=5, cost_fp=1)
    assert s["approval_rate"] == 1.0
    assert s["bad_rate_approved"] == pytest.approx(0.25)
    assert s["expected_cost"] == pytest.approx(5 / 4)
