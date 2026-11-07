import numpy as np
import pandas as pd
import pytest

from credit_risk.fairness import age_band, audit, disparate_impact_ratio, group_report
from credit_risk.features import build_features
from credit_risk.monitoring import (
    build_reference_profile,
    drift_report,
    load_profile,
    psi,
    psi_status,
    save_profile,
)


def test_psi_identical_is_zero_and_shift_is_large():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 5000)
    assert psi(a, a) == pytest.approx(0.0, abs=1e-9)
    assert psi(a, rng.normal(1.0, 1, 5000)) > 0.25


@pytest.mark.parametrize(
    ("value", "status"), [(0.05, "stable"), (0.2, "moderate"), (0.4, "significant")]
)
def test_psi_status(value, status):
    assert psi_status(value) == status


def test_drift_report_detects_shift(tmp_path, synthetic_df):
    x = build_features(synthetic_df)
    profile = build_reference_profile(x)
    save_profile(profile, tmp_path / "p.json")
    profile = load_profile(tmp_path / "p.json")

    assert drift_report(profile, x)["status"].eq("stable").all()
    shifted = x.assign(util_recent=x["util_recent"] + 0.5)
    report = drift_report(profile, shifted).set_index("feature")
    assert report.loc["util_recent", "status"] == "significant"


def test_age_band():
    assert age_band(pd.Series([25, 35, 45, 65])).tolist() == ["<30", "30-39", "40-49", "50+"]


def test_group_report_and_di_ratio():
    y = np.array([0, 1, 0, 1, 0, 0])
    p = np.array([0.1, 0.9, 0.2, 0.8, 0.9, 0.1])
    g = pd.Series(["a", "a", "a", "b", "b", "b"])
    rep = group_report(y, p, 0.5, g).set_index("group")
    assert rep.loc["a", "approval_rate"] == pytest.approx(2 / 3)
    assert rep.loc["b", "approval_rate"] == pytest.approx(1 / 3)
    assert disparate_impact_ratio(rep.reset_index()) == pytest.approx(0.5)


def test_audit_returns_all_groups(synthetic_df):
    out = audit(synthetic_df, np.full(len(synthetic_df), 0.3), threshold=0.5)
    assert set(out) == {"sex", "marriage", "age_band"}
    assert out["sex"]["approval_rate"].eq(1.0).all()
