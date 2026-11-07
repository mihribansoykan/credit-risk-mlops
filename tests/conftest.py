"""Shared fixtures. Tests use synthetic data with a known signal, so CI needs no network access."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from credit_risk.data import BILL_COLS, PAY_AMT_COLS, PAY_STATUS_COLS, TARGET, split
from credit_risk.model import CreditRiskModel
from credit_risk.train import build_model


def make_synthetic(n: int = 3000, seed: int = 0) -> pd.DataFrame:
    """Data in the cleaned schema; default risk driven by delays and utilisation."""
    rng = np.random.default_rng(seed)
    limit = rng.choice([10_000, 50_000, 100_000, 200_000, 500_000], size=n).astype(float)
    delay = rng.choice([-2, -1, 0, 1, 2, 3], size=n, p=[0.1, 0.2, 0.45, 0.12, 0.1, 0.03])
    util = rng.beta(2, 3, size=n)

    df = pd.DataFrame(
        {
            "ID": np.arange(1, n + 1),
            "LIMIT_BAL": limit,
            "SEX": rng.integers(1, 3, n),
            "EDUCATION": rng.integers(1, 5, n),
            "MARRIAGE": rng.integers(1, 4, n),
            "AGE": rng.integers(21, 70, n),
        }
    )
    for i, col in enumerate(PAY_STATUS_COLS):
        df[col] = np.clip(delay - (i > 2) * rng.integers(0, 2, n), -2, 9)
    for i, col in enumerate(BILL_COLS):
        df[col] = np.round(limit * np.clip(util + rng.normal(0, 0.05, n) - 0.02 * i, 0, 1.2))
    for i, col in enumerate(PAY_AMT_COLS):
        df[col] = np.round(df[BILL_COLS[min(i + 1, 5)]] * rng.uniform(0, 0.3, n))

    logit = -2.5 + 1.4 * np.clip(delay, 0, None) + 2.5 * util - 0.5 * (limit > 100_000)
    df[TARGET] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return df


@pytest.fixture(scope="session")
def synthetic_df() -> pd.DataFrame:
    return make_synthetic()


@pytest.fixture(scope="session")
def splits(synthetic_df):
    return split(synthetic_df, test_size=0.2, valid_size=0.25, seed=0)


@pytest.fixture(scope="session")
def trained_model(splits) -> CreditRiskModel:
    train, valid, _ = splits
    model, _, _ = build_model(
        train,
        valid,
        n_trials=3,
        cv_folds=3,
        calibration="isotonic",
        cost_fn=5.0,
        cost_fp=1.0,
        seed=0,
    )
    model.metadata = {"model_version": "test"}
    return model
