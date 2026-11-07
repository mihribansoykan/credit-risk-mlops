"""Training pipeline.

    uv run credit-risk-train                      # full run with configs/train.yaml
    uv run credit-risk-train --n-trials 5         # quick run
    uv run credit-risk-train --no-tracking        # skip MLflow logging

Steps: download -> clean -> validate schema -> split -> baseline -> tune LightGBM (CV) ->
calibrate (validation) -> pick cost-optimal threshold (validation) -> evaluate once on test ->
fairness audit -> drift reference profile -> figures, metrics.json, model artefact, MLflow run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from credit_risk import __version__
from credit_risk.config import TrainConfig, load_config
from credit_risk.data import TARGET, clean, download_raw, load_raw, split
from credit_risk.evaluate import classification_metrics, decision_summary, optimal_threshold
from credit_risk.features import FEATURES, build_features
from credit_risk.model import Calibrator, CreditRiskModel

log = logging.getLogger("credit_risk.train")


# --------------------------------------------------------------------------- models
def _signed_log1p(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


def make_baseline(seed: int = 42) -> Pipeline:
    """Regularised logistic regression on the same engineered features (the benchmark to beat)."""
    numeric = [f for f in FEATURES if f != "EDUCATION"]
    pre = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [("log", FunctionTransformer(_signed_log1p)), ("scale", StandardScaler())]
                ),
                numeric,
            ),
            ("edu", OneHotEncoder(handle_unknown="ignore"), ["EDUCATION"]),
        ]
    )
    return Pipeline(
        [("pre", pre), ("clf", LogisticRegression(C=0.5, max_iter=5000, random_state=seed))]
    )


def _lgbm(params: dict[str, Any], seed: int) -> LGBMClassifier:
    return LGBMClassifier(
        objective="binary", subsample_freq=1, random_state=seed, verbosity=-1, n_jobs=-1, **params
    )


def tune_lightgbm(
    x: pd.DataFrame, y: pd.Series, n_trials: int, cv_folds: int, seed: int
) -> dict[str, Any]:
    """Bayesian hyper-parameter search (Optuna TPE) maximising cross-validated ROC-AUC."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 8, 64),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 300),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        }
        return cross_val_score(_lgbm(params, seed), x, y, cv=cv, scoring="roc_auc").mean()

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    log.info("Best CV ROC-AUC %.4f with %s", study.best_value, study.best_params)
    return {**study.best_params, "cv_roc_auc": study.best_value}


def build_model(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    *,
    n_trials: int,
    cv_folds: int,
    calibration: str,
    cost_fn: float,
    cost_fp: float,
    seed: int,
) -> tuple[CreditRiskModel, dict[str, Any], pd.DataFrame]:
    """Tune + fit LightGBM on train, calibrate and choose the threshold on validation."""
    x_tr, y_tr = build_features(train), train[TARGET]
    best = tune_lightgbm(x_tr, y_tr, n_trials=n_trials, cv_folds=cv_folds, seed=seed)
    params = {k: v for k, v in best.items() if k != "cv_roc_auc"}
    clf = _lgbm(params, seed).fit(x_tr, y_tr)

    raw_valid = clf.predict_proba(build_features(valid))[:, 1]
    calibrator = Calibrator(calibration).fit(raw_valid, valid[TARGET].to_numpy(), seed=seed)
    log.info("Calibration: %s (CV log-loss %s)", calibrator.selected, calibrator.cv_log_loss)
    pd_valid = calibrator.transform(raw_valid)
    threshold, curve = optimal_threshold(valid[TARGET].to_numpy(), pd_valid, cost_fn, cost_fp)

    model = CreditRiskModel(classifier=clf, calibrator=calibrator, threshold=threshold)
    return model, best, curve


# ------------------------------------------------------------------------- pipeline
def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(df.round(4).to_json(orient="records"))


def run(cfg: TrainConfig, n_trials: int | None = None, tracking: bool = True) -> dict[str, Any]:
    from credit_risk import plots
    from credit_risk.fairness import audit, disparate_impact_ratio
    from credit_risk.monitoring import build_reference_profile, drift_report, save_profile
    from credit_risk.schema import validate

    n_trials = n_trials or cfg.model.n_trials
    c_fn, c_fp = cfg.costs.false_negative, cfg.costs.false_positive

    # 1. Data --------------------------------------------------------------------------
    raw_path = download_raw(cfg.data.url, cfg.data.raw_path)
    df = validate(clean(load_raw(raw_path)))
    train, valid, test = split(df, cfg.data.test_size, cfg.data.valid_size, cfg.seed)
    log.info("Rows: train=%d valid=%d test=%d", len(train), len(valid), len(test))
    y_test = test[TARGET].to_numpy()

    # 2. Baseline ----------------------------------------------------------------------
    baseline = make_baseline(cfg.seed).fit(build_features(train), train[TARGET])
    p_base = baseline.predict_proba(build_features(test))[:, 1]

    # 3. LightGBM + calibration + threshold ----------------------------------------------
    model, best_params, curve = build_model(
        train,
        valid,
        n_trials=n_trials,
        cv_folds=cfg.model.cv_folds,
        calibration=cfg.model.calibration,
        cost_fn=c_fn,
        cost_fp=c_fp,
        seed=cfg.seed,
    )
    p_raw = model.raw_score(test)
    p_cal = model.predict_pd(test)

    # 4. Evaluation on the untouched test set -------------------------------------------
    test_metrics = {
        "logistic_regression": classification_metrics(y_test, p_base),
        "lightgbm_raw": classification_metrics(y_test, p_raw),
        "lightgbm_final": classification_metrics(y_test, p_cal),
    }
    decisions = {
        "approve_all": decision_summary(y_test, np.zeros_like(p_cal), 1.0, c_fn, c_fp),
        "threshold_0.5": decision_summary(y_test, p_cal, 0.5, c_fn, c_fp),
        "cost_optimal": decision_summary(y_test, p_cal, model.threshold, c_fn, c_fp),
    }

    # 5. Explainability, fairness, drift reference --------------------------------------
    contrib = model.contributions(test)
    importance = contrib.abs().mean().sort_values(ascending=False)
    fairness = audit(test, p_cal, model.threshold)
    x_train = build_features(train)
    profile = build_reference_profile(x_train)
    save_profile(profile, cfg.artifacts.reference_profile_path)
    drift = drift_report(profile, build_features(test))

    # 6. Artefacts ---------------------------------------------------------------------
    figs = cfg.artifacts.figures_dir
    plots.roc_pr_curves(
        y_test,
        {"LightGBM (final model)": p_cal, "Logistic regression (baseline)": p_base},
        figs / "roc_pr_curves.png",
    )
    plots.calibration_plot(
        y_test,
        {"LightGBM (final model)": p_cal, "Logistic regression (baseline)": p_base},
        figs / "calibration.png",
    )
    plots.cost_curve_plot(curve, model.threshold, figs / "cost_curve.png")
    plots.importance_plot(importance, figs / "feature_importance.png")
    plots.fairness_plot(fairness, figs / "fairness.png")

    metrics: dict[str, Any] = {
        "model_version": __version__,
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "data_sha256": _sha256(Path(raw_path)),
        "dataset": {
            "n_train": len(train),
            "n_valid": len(valid),
            "n_test": len(test),
            "default_rate": round(float(df[TARGET].mean()), 4),
        },
        "costs": {"false_negative": c_fn, "false_positive": c_fp},
        "best_params": best_params,
        "calibration": {
            "selected": model.calibrator.selected,
            "cv_log_loss_validation": model.calibrator.cv_log_loss,
        },
        "test_metrics": test_metrics,
        "decisions_test": decisions,
        "fairness_test": {k: _records(v) for k, v in fairness.items()},
        "disparate_impact_ratio": {k: disparate_impact_ratio(v) for k, v in fairness.items()},
        "drift_test_vs_train": {
            "max_psi": float(drift["psi"].max()),
            "n_significant": int((drift["status"] == "significant").sum()),
        },
        "top_features": {k: round(float(v), 4) for k, v in importance.head(10).items()},
    }
    model.metadata = {
        k: metrics[k] for k in ("model_version", "trained_at", "data_sha256", "best_params")
    }
    model.metadata["test_roc_auc"] = test_metrics["lightgbm_final"]["roc_auc"]
    model.save(cfg.artifacts.model_path)

    cfg.artifacts.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.artifacts.metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    log.info(
        "Test ROC-AUC: baseline %.4f | LightGBM %.4f | threshold %.2f",
        test_metrics["logistic_regression"]["roc_auc"],
        test_metrics["lightgbm_final"]["roc_auc"],
        model.threshold,
    )

    if tracking and cfg.tracking.enabled:
        _log_to_mlflow(cfg, metrics, n_trials)
    return metrics


def _log_to_mlflow(cfg: TrainConfig, metrics: dict[str, Any], n_trials: int) -> None:
    try:
        import mlflow
    except ImportError:  # pragma: no cover
        log.warning("mlflow not installed - skipping experiment tracking")
        return
    mlflow.set_tracking_uri(cfg.tracking.tracking_uri)
    mlflow.set_experiment(cfg.tracking.experiment_name)
    with mlflow.start_run(run_name=f"lgbm-{metrics['trained_at']}"):
        mlflow.log_params({f"lgbm_{k}": v for k, v in metrics["best_params"].items()})
        mlflow.log_params(
            {
                "n_trials": n_trials,
                "calibration": cfg.model.calibration,
                "cost_fn": cfg.costs.false_negative,
                "cost_fp": cfg.costs.false_positive,
                "data_sha256": metrics["data_sha256"][:12],
            }
        )
        for model_name, m in metrics["test_metrics"].items():
            mlflow.log_metrics({f"test_{model_name}_{k}": v for k, v in m.items()})
        for policy, m in metrics["decisions_test"].items():
            mlflow.log_metrics({f"decision_{policy}_{k}": v for k, v in m.items()})
        mlflow.log_artifact(str(cfg.artifacts.metrics_path))
        mlflow.log_artifacts(str(cfg.artifacts.figures_dir), artifact_path="figures")
        mlflow.log_artifact(str(cfg.artifacts.model_path), artifact_path="model")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train the credit default model")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--n-trials", type=int, default=None, help="override Optuna trials")
    parser.add_argument("--no-tracking", action="store_true", help="disable MLflow logging")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(load_config(args.config), n_trials=args.n_trials, tracking=not args.no_tracking)


if __name__ == "__main__":
    main()
