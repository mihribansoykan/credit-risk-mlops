"""End-to-end smoke test of the training pipeline on synthetic data (no network, no MLflow)."""

import json

from credit_risk.config import TrainConfig
from credit_risk.data import COLUMN_MAP
from credit_risk.model import CreditRiskModel
from credit_risk.train import run


def test_full_pipeline(tmp_path, synthetic_df):
    raw = tmp_path / "raw.csv"
    synthetic_df.rename(columns={v: k for k, v in COLUMN_MAP.items()}).to_csv(raw, index=False)

    cfg = TrainConfig.model_validate(
        {
            "data": {"url": "http://unused", "raw_path": raw, "test_size": 0.2, "valid_size": 0.25},
            "protected_attributes": ["SEX", "MARRIAGE", "AGE"],
            "model": {"n_trials": 2, "cv_folds": 3, "calibration": "isotonic"},
            "costs": {"false_negative": 5, "false_positive": 1},
            "tracking": {"enabled": False},
            "artifacts": {
                "model_path": tmp_path / "models/model.joblib",
                "metrics_path": tmp_path / "reports/metrics.json",
                "figures_dir": tmp_path / "reports/figures",
                "reference_profile_path": tmp_path / "models/profile.json",
            },
            "seed": 0,
        }
    )
    metrics = run(cfg, tracking=False)

    assert CreditRiskModel.load(cfg.artifacts.model_path).threshold > 0
    saved = json.loads(cfg.artifacts.metrics_path.read_text())
    assert saved["test_metrics"]["lightgbm_final"]["roc_auc"] > 0.7
    assert set(saved["decisions_test"]) == {"approve_all", "threshold_0.5", "cost_optimal"}
    # The cost-optimal policy must not be worse than approving everyone.
    d = metrics["decisions_test"]
    assert d["cost_optimal"]["expected_cost"] <= d["approve_all"]["expected_cost"]
    assert len(list(cfg.artifacts.figures_dir.glob("*.png"))) == 5
