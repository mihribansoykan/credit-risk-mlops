import pandas as pd
import pandera.errors
import pytest

from credit_risk.data import COLUMN_MAP, TARGET, clean, load_raw, split
from credit_risk.schema import validate


def test_column_map_covers_all_raw_columns():
    assert len(COLUMN_MAP) == 24  # X1..X23 + Y
    assert COLUMN_MAP["X6"] == "PAY_1" and COLUMN_MAP["X23"] == "PAY_AMT6"


def test_load_raw_renames(tmp_path, synthetic_df):
    inverse = {v: k for k, v in COLUMN_MAP.items()}
    path = tmp_path / "raw.csv"
    synthetic_df.rename(columns=inverse).to_csv(path, index=False)
    df = load_raw(path)
    assert list(df.columns) == list(synthetic_df.columns)


def test_clean_merges_undocumented_codes(synthetic_df):
    df = synthetic_df.head(4).copy()
    df["EDUCATION"] = [0, 5, 6, 2]
    df["MARRIAGE"] = [0, 1, 2, 3]
    out = clean(df)
    assert out["EDUCATION"].tolist() == [4, 4, 4, 2]
    assert out["MARRIAGE"].tolist() == [3, 1, 2, 3]


def test_schema_accepts_clean_data(synthetic_df):
    validate(clean(synthetic_df))


def test_schema_rejects_bad_pay_status(synthetic_df):
    df = clean(synthetic_df)
    df.loc[0, "PAY_1"] = 12
    with pytest.raises(pandera.errors.SchemaErrors):
        validate(df)


def test_split_is_stratified_and_disjoint(synthetic_df):
    train, valid, test = split(synthetic_df, test_size=0.2, valid_size=0.25, seed=1)
    assert len(train) + len(valid) + len(test) == len(synthetic_df)
    ids = [set(x["ID"]) for x in (train, valid, test)]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    rates = pd.Series([x[TARGET].mean() for x in (train, valid, test)])
    assert rates.max() - rates.min() < 0.02
