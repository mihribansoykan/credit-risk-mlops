"""Data acquisition, cleaning and splitting.

Dataset: "Default of Credit Card Clients" (Yeh & Lien, 2009), UCI ML Repository, id 350.
30,000 Taiwanese credit card holders, observed April-September 2005. Licence: CC BY 4.0.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

TARGET = "default"
ID_COL = "ID"

# The UCI CSV ships with anonymous column names (X1..X23, Y). Map them to the documented meaning.
# PAY_1 / BILL_AMT1 / PAY_AMT1 refer to September 2005 (most recent), ..._6 to April 2005.
COLUMN_MAP: dict[str, str] = {
    "X1": "LIMIT_BAL",
    "X2": "SEX",
    "X3": "EDUCATION",
    "X4": "MARRIAGE",
    "X5": "AGE",
    **{f"X{i + 6}": f"PAY_{i + 1}" for i in range(6)},
    **{f"X{i + 12}": f"BILL_AMT{i + 1}" for i in range(6)},
    **{f"X{i + 18}": f"PAY_AMT{i + 1}" for i in range(6)},
    "Y": TARGET,
}

PAY_STATUS_COLS = [f"PAY_{i}" for i in range(1, 7)]
BILL_COLS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAY_AMT_COLS = [f"PAY_AMT{i}" for i in range(1, 7)]


def download_raw(url: str, path: str | Path, force: bool = False) -> Path:
    """Download the raw CSV once; subsequent calls are no-ops unless ``force``."""
    path = Path(path)
    if path.exists() and not force:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    with urllib.request.urlopen(url, timeout=60) as resp, open(tmp, "wb") as f:  # noqa: S310
        f.write(resp.read())
    tmp.replace(path)
    return path


def load_raw(path: str | Path) -> pd.DataFrame:
    """Read the raw CSV and give columns their documented names."""
    df = pd.read_csv(path)
    # Some mirrors of the data already use the descriptive names (with PAY_0 instead of PAY_1).
    df = df.rename(columns={**COLUMN_MAP, "PAY_0": "PAY_1", "default payment next month": TARGET})
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Fix undocumented category codes.

    * EDUCATION: documented values are 1-4 (graduate school, university, high school, other);
      the data also contains 0, 5 and 6, which are merged into 4 ("other").
    * MARRIAGE: documented values are 1-3; the undocumented 0 is merged into 3 ("other").
    """
    out = df.copy()
    out["EDUCATION"] = out["EDUCATION"].where(out["EDUCATION"].between(1, 4), 4)
    out["MARRIAGE"] = out["MARRIAGE"].where(out["MARRIAGE"].between(1, 3), 3)
    out["LIMIT_BAL"] = out["LIMIT_BAL"].astype(float)
    return out


def split(
    df: pd.DataFrame, test_size: float, valid_size: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified train / validation / test split.

    The validation set is used only for probability calibration and threshold selection,
    so the test set stays untouched until the final evaluation.
    """
    train_valid, test = train_test_split(
        df, test_size=test_size, stratify=df[TARGET], random_state=seed
    )
    train, valid = train_test_split(
        train_valid, test_size=valid_size, stratify=train_valid[TARGET], random_state=seed
    )
    return train.reset_index(drop=True), valid.reset_index(drop=True), test.reset_index(drop=True)
