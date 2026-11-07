"""Deployable model object: features -> LightGBM score -> calibrated PD -> decision + reasons."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold

from credit_risk.features import FEATURES, build_features

_EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


CalibrationMethod = Literal["auto", "none", "sigmoid", "isotonic"]


class Calibrator:
    """Maps raw model scores to well-calibrated probabilities of default.

    Fitted on a held-out validation set so that a predicted PD of 0.2 really means
    ~20% of such applicants default, which matters for pricing and expected-loss estimates.

    ``method="auto"`` picks between no calibration, Platt (sigmoid) and isotonic regression by
    cross-validated log-loss *on the validation set*, so the test set is never used for the choice.
    """

    CANDIDATES = ("none", "sigmoid", "isotonic")

    def __init__(self, method: CalibrationMethod = "auto") -> None:
        self.method = method
        self.selected: str | None = None
        self.cv_log_loss: dict[str, float] = {}
        self._model: IsotonicRegression | LogisticRegression | None = None

    @staticmethod
    def _fit_one(method: str, raw: np.ndarray, y: np.ndarray):
        if method == "isotonic":
            return IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(raw, y)
        if method == "sigmoid":
            return LogisticRegression().fit(_logit(raw).reshape(-1, 1), y)
        return None

    @staticmethod
    def _apply(model, raw: np.ndarray) -> np.ndarray:
        if model is None:
            out = raw
        elif isinstance(model, IsotonicRegression):
            out = model.predict(raw)
        else:
            out = model.predict_proba(_logit(raw).reshape(-1, 1))[:, 1]
        return np.clip(out, _EPS, 1 - _EPS)

    def fit(self, raw_scores: np.ndarray, y: np.ndarray, seed: int = 0) -> Calibrator:
        raw, y = np.asarray(raw_scores, dtype=float), np.asarray(y)
        if self.method == "auto":
            folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            for m in self.CANDIDATES:
                losses = [
                    log_loss(y[te], self._apply(self._fit_one(m, raw[tr], y[tr]), raw[te]))
                    for tr, te in folds.split(raw, y)
                ]
                self.cv_log_loss[m] = float(np.mean(losses))
            self.selected = min(self.cv_log_loss, key=self.cv_log_loss.get)
        else:
            self.selected = self.method
        self._model = self._fit_one(self.selected, raw, y)
        return self

    def transform(self, raw_scores: np.ndarray) -> np.ndarray:
        if self.selected is None:
            raise RuntimeError("Calibrator is not fitted")
        return self._apply(self._model, np.asarray(raw_scores, dtype=float))


@dataclass
class CreditRiskModel:
    classifier: LGBMClassifier
    calibrator: Calibrator
    threshold: float
    feature_names: list[str] = field(default_factory=lambda: list(FEATURES))
    metadata: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ scoring
    def features(self, df: pd.DataFrame) -> pd.DataFrame:
        return build_features(df)[self.feature_names]

    def raw_score(self, df: pd.DataFrame) -> np.ndarray:
        return self.classifier.predict_proba(self.features(df))[:, 1]

    def predict_pd(self, df: pd.DataFrame) -> np.ndarray:
        """Calibrated probability of default for each row."""
        return self.calibrator.transform(self.raw_score(df))

    def decide(self, pd_: np.ndarray) -> np.ndarray:
        """Decline applicants whose PD reaches the cost-optimal threshold."""
        return np.where(pd_ >= self.threshold, "decline", "approve")

    # ------------------------------------------------------------ explanations
    def contributions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Per-feature SHAP contributions (log-odds) computed natively by LightGBM."""
        x = self.features(df)
        contrib = self.classifier.booster_.predict(x, pred_contrib=True)
        return pd.DataFrame(contrib[:, :-1], columns=self.feature_names, index=df.index)

    def reason_codes(self, df: pd.DataFrame, top_k: int = 3) -> list[list[dict[str, float]]]:
        """Top risk-increasing features per applicant (similar to adverse-action reason codes)."""
        contrib = self.contributions(df)
        reasons: list[list[dict[str, float]]] = []
        for _, row in contrib.iterrows():
            top = row[row > 0].sort_values(ascending=False).head(top_k)
            reasons.append(
                [{"feature": str(k), "contribution": round(float(v), 4)} for k, v in top.items()]
            )
        return reasons

    # -------------------------------------------------------------- persistence
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path: str | Path) -> CreditRiskModel:
        obj = joblib.load(path)
        if not isinstance(obj, CreditRiskModel):
            raise TypeError(f"{path} does not contain a CreditRiskModel")
        return obj
