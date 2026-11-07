"""Data contract for the cleaned dataset, enforced with pandera.

Failing fast on schema violations (e.g. a new, unexpected PAY status code) is cheaper than
silently training or scoring on corrupted data.
"""

from __future__ import annotations

try:  # pandera >= 0.24 exposes the pandas API under ``pandera.pandas``
    import pandera.pandas as pa
except ImportError:  # pragma: no cover
    import pandera as pa

from credit_risk.data import BILL_COLS, ID_COL, PAY_AMT_COLS, PAY_STATUS_COLS, TARGET

_pay_status = {c: pa.Column(int, pa.Check.in_range(-2, 9)) for c in PAY_STATUS_COLS}
_bill = {c: pa.Column(float, coerce=True) for c in BILL_COLS}
_pay_amt = {c: pa.Column(float, pa.Check.ge(0), coerce=True) for c in PAY_AMT_COLS}

CLEAN_SCHEMA = pa.DataFrameSchema(
    {
        ID_COL: pa.Column(int, unique=True),
        "LIMIT_BAL": pa.Column(float, pa.Check.gt(0), coerce=True),
        "SEX": pa.Column(int, pa.Check.isin([1, 2])),
        "EDUCATION": pa.Column(int, pa.Check.isin([1, 2, 3, 4])),
        "MARRIAGE": pa.Column(int, pa.Check.isin([1, 2, 3])),
        "AGE": pa.Column(int, pa.Check.in_range(18, 100)),
        **_pay_status,
        **_bill,
        **_pay_amt,
        TARGET: pa.Column(int, pa.Check.isin([0, 1])),
    },
    strict=True,
)


def validate(df):
    """Validate (and coerce) a cleaned dataframe; raises ``pandera.errors.SchemaErrors``."""
    return CLEAN_SCHEMA.validate(df, lazy=True)
