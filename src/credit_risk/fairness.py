"""Fairness audit across protected groups.

Protected attributes are excluded from the model inputs, but proxies (e.g. credit limit)
can still produce disparate outcomes, so outcomes are audited explicitly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

SEX_LABELS = {1: "male", 2: "female"}
MARRIAGE_LABELS = {1: "married", 2: "single", 3: "other"}


def age_band(age: pd.Series) -> pd.Series:
    """Ordered categorical age bands (keeps a natural order in reports and charts)."""
    return pd.cut(age, bins=[0, 29, 39, 49, 200], labels=["<30", "30-39", "40-49", "50+"])


def group_report(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float, groups: pd.Series
) -> pd.DataFrame:
    """Per-group outcome metrics.

    * approval_rate     - share approved (basis of the disparate impact ratio)
    * default_rate      - observed base rate in the group
    * tpr               - share of actual defaulters that were declined
    * fpr               - share of good customers that were wrongly declined
    """
    df = pd.DataFrame(
        {
            "y": np.asarray(y_true),
            "p": np.asarray(y_prob),
            "g": pd.Series(groups).reset_index(drop=True),  # keeps categorical order if present
        }
    )
    df["decline"] = df["p"] >= threshold
    rows = []
    for g, part in df.groupby("g", sort=True, observed=True):
        bads, goods = part[part.y == 1], part[part.y == 0]
        rows.append(
            {
                "group": str(g),
                "n": len(part),
                "default_rate": part.y.mean(),
                "approval_rate": 1 - part.decline.mean(),
                "tpr": bads.decline.mean() if len(bads) else np.nan,
                "fpr": goods.decline.mean() if len(goods) else np.nan,
                "auc": roc_auc_score(part.y, part.p) if part.y.nunique() == 2 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def disparate_impact_ratio(report: pd.DataFrame) -> float:
    """min / max approval rate across groups. The "four-fifths rule" flags values below 0.8."""
    rates = report["approval_rate"]
    return float(rates.min() / rates.max())


def audit(
    df: pd.DataFrame, y_prob: np.ndarray, threshold: float, target: str = "default"
) -> dict[str, pd.DataFrame]:
    groups = {
        "sex": df["SEX"].map(SEX_LABELS),
        "marriage": df["MARRIAGE"].map(MARRIAGE_LABELS),
        "age_band": age_band(df["AGE"]),
    }
    return {
        name: group_report(df[target].to_numpy(), y_prob, threshold, g)
        for name, g in groups.items()
    }
