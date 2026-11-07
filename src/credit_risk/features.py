"""Feature engineering shared by training and serving.

Keeping a single ``build_features`` function used in both places removes training/serving skew.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from credit_risk.data import BILL_COLS, PAY_AMT_COLS, PAY_STATUS_COLS

# Raw attributes the model is allowed to use. Protected attributes (SEX, MARRIAGE, AGE)
# are intentionally absent: they are only used to audit the model (see fairness.py).
RAW_FEATURES = ["LIMIT_BAL", "EDUCATION", *PAY_STATUS_COLS, *BILL_COLS, *PAY_AMT_COLS]

ENGINEERED_FEATURES = [
    "util_recent",
    "util_mean",
    "util_max",
    "bill_trend",
    "pay_ratio_recent",
    "pay_ratio_mean",
    "total_paid_to_billed",
    "months_delayed",
    "max_delay",
    "delay_trend",
    "months_no_consumption",
]

FEATURES = RAW_FEATURES + ENGINEERED_FEATURES


def _safe_ratio(
    num: pd.Series | np.ndarray, den: pd.Series | np.ndarray, fill: float
) -> np.ndarray:
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    out = np.full_like(num, fill, dtype=float)
    mask = den > 0
    out[mask] = num[mask] / den[mask]
    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return the model matrix (``FEATURES`` columns, in order) for raw applicant rows."""
    missing = [c for c in RAW_FEATURES if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    x = df[RAW_FEATURES].astype(float).copy()
    limit = x["LIMIT_BAL"].to_numpy()
    bills = x[BILL_COLS].to_numpy()
    paid = x[PAY_AMT_COLS].to_numpy()
    status = x[PAY_STATUS_COLS].to_numpy()

    # Credit utilisation: how much of the limit is drawn.
    util = bills / limit[:, None]
    x["util_recent"] = util[:, 0]
    x["util_mean"] = util.mean(axis=1)
    x["util_max"] = util.max(axis=1)
    x["bill_trend"] = (bills[:, 0] - bills[:, 5]) / limit

    # Repayment behaviour: PAY_AMT in month k settles the statement of month k+1.
    ratios = np.column_stack(
        [_safe_ratio(paid[:, k], bills[:, k + 1], fill=1.0) for k in range(5)]
    ).clip(0, 5)
    x["pay_ratio_recent"] = ratios[:, 0]
    x["pay_ratio_mean"] = ratios.mean(axis=1)
    x["total_paid_to_billed"] = _safe_ratio(
        paid.sum(axis=1), bills[:, 1:].clip(min=0).sum(axis=1), fill=1.0
    ).clip(0, 5)

    # Delinquency history: PAY_k >= 1 means k months of payment delay.
    x["months_delayed"] = (status >= 1).sum(axis=1)
    x["max_delay"] = status.max(axis=1)
    x["delay_trend"] = status[:, 0] - status[:, 5]
    x["months_no_consumption"] = (status == -2).sum(axis=1)

    return x[FEATURES]
