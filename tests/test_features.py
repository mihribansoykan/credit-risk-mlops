import numpy as np
import pytest

from credit_risk.features import FEATURES, build_features


def test_build_features_shape_and_order(synthetic_df):
    x = build_features(synthetic_df)
    assert list(x.columns) == FEATURES
    assert len(x) == len(synthetic_df)
    assert np.isfinite(x.to_numpy()).all()


def test_protected_attributes_not_used(synthetic_df):
    assert not {"SEX", "MARRIAGE", "AGE"} & set(build_features(synthetic_df).columns)


def test_missing_column_raises(synthetic_df):
    with pytest.raises(KeyError):
        build_features(synthetic_df.drop(columns=["PAY_1"]))


def test_engineered_values(synthetic_df):
    row = synthetic_df.head(1).copy()
    row[["PAY_1", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]] = [2, 1, 0, -1, -2, -2]
    row["LIMIT_BAL"] = 1000.0
    row["BILL_AMT1"] = 500.0
    x = build_features(row).iloc[0]
    assert x["months_delayed"] == 2
    assert x["max_delay"] == 2
    assert x["months_no_consumption"] == 2
    assert x["util_recent"] == pytest.approx(0.5)


def test_zero_bill_gives_neutral_pay_ratio(synthetic_df):
    row = synthetic_df.head(1).copy()
    row["BILL_AMT2"] = 0.0
    assert build_features(row).iloc[0]["pay_ratio_recent"] == 1.0
