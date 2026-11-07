import numpy as np
from sklearn.metrics import roc_auc_score

from credit_risk.data import TARGET
from credit_risk.model import Calibrator, CreditRiskModel


def test_quality_gate_on_holdout(trained_model, splits):
    """The model must clearly beat random ranking on unseen data."""
    _, _, test = splits
    auc = roc_auc_score(test[TARGET], trained_model.predict_pd(test))
    assert auc > 0.70


def test_probabilities_are_valid(trained_model, splits):
    pd_ = trained_model.predict_pd(splits[2])
    assert ((pd_ > 0) & (pd_ < 1)).all()


def test_threshold_and_decisions(trained_model, splits):
    assert 0 < trained_model.threshold < 1
    decisions = trained_model.decide(np.array([0.0, 0.999]))
    assert decisions.tolist() == ["approve", "decline"]


def test_reason_codes_are_risk_increasing(trained_model, splits):
    reasons = trained_model.reason_codes(splits[2].head(5), top_k=3)
    assert len(reasons) == 5
    for r in reasons:
        assert len(r) <= 3
        assert all(item["contribution"] > 0 for item in r)


def test_contributions_sum_to_raw_logit(trained_model, splits):
    sample = splits[2].head(20)
    contrib = trained_model.contributions(sample)
    raw = trained_model.classifier.booster_.predict(trained_model.features(sample), raw_score=True)
    bias = trained_model.classifier.booster_.predict(
        trained_model.features(sample), pred_contrib=True
    )[:, -1]
    np.testing.assert_allclose(contrib.sum(axis=1) + bias, raw, rtol=1e-5, atol=1e-6)


def test_save_and_load_roundtrip(tmp_path, trained_model, splits):
    path = trained_model.save(tmp_path / "m.joblib")
    loaded = CreditRiskModel.load(path)
    np.testing.assert_allclose(loaded.predict_pd(splits[2]), trained_model.predict_pd(splits[2]))


def test_sigmoid_calibrator_is_monotonic():
    rng = np.random.default_rng(0)
    s = rng.random(500)
    y = (rng.random(500) < s).astype(int)
    out = Calibrator("sigmoid").fit(s, y).transform(np.array([0.1, 0.5, 0.9]))
    assert out[0] < out[1] < out[2]


def test_auto_calibrator_fixes_miscalibrated_scores():
    """Scores that are systematically too high should not be left uncalibrated."""
    rng = np.random.default_rng(0)
    true_p = rng.random(4000) * 0.5
    y = (rng.random(4000) < true_p).astype(int)
    raw = np.clip(true_p * 1.8, 0, 0.999)  # over-confident model
    cal = Calibrator("auto").fit(raw, y)
    assert set(cal.cv_log_loss) == {"none", "sigmoid", "isotonic"}
    assert cal.selected != "none"
    assert abs(cal.transform(raw).mean() - y.mean()) < 0.02
