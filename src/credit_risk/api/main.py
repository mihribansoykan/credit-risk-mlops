"""FastAPI scoring service.

    uv run uvicorn credit_risk.api.main:app --reload
    # Swagger UI: http://127.0.0.1:8000/docs

The model path is read from the MODEL_PATH environment variable (default: models/model.joblib).
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException

from credit_risk.api.schemas import (
    Applicant,
    BatchRequest,
    BatchResponse,
    Health,
    Prediction,
    PredictionResponse,
)
from credit_risk.model import CreditRiskModel

log = logging.getLogger("credit_risk.api")
_state: dict[str, CreditRiskModel | None] = {"model": None}


def load_model(path: str | None = None) -> CreditRiskModel | None:
    path = path or os.getenv("MODEL_PATH", "models/model.joblib")
    try:
        model = CreditRiskModel.load(path)
    except FileNotFoundError:
        log.warning("No model found at %s - API starts in degraded mode", path)
        return None
    log.info("Loaded model %s from %s", model.metadata.get("model_version"), path)
    return model


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    if _state["model"] is None:
        _state["model"] = load_model()
    yield


app = FastAPI(
    title="Credit Default Risk API",
    version="1.0.0",
    description="Scores credit card applicants with a calibrated LightGBM model and returns "
    "the probability of default, a cost-optimal decision and the top risk factors.",
    lifespan=lifespan,
)


def _model() -> CreditRiskModel:
    model = _state["model"]
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return model


def _score(model: CreditRiskModel, applicants: list[Applicant]) -> list[Prediction]:
    df = pd.DataFrame([a.model_dump() for a in applicants])
    pds = model.predict_pd(df)
    decisions = model.decide(pds)
    reasons = model.reason_codes(df, top_k=3)
    return [
        Prediction(
            probability_of_default=round(float(p), 4),
            decision=d,
            threshold=model.threshold,
            top_risk_factors=r,
        )
        for p, d, r in zip(pds, decisions, reasons, strict=True)
    ]


def _version(model: CreditRiskModel) -> str:
    return str(model.metadata.get("model_version", "unknown"))


@app.get("/health", response_model=Health)
def health() -> Health:
    model = _state["model"]
    if model is None:
        return Health(status="degraded", model_loaded=False)
    return Health(status="ok", model_loaded=True, model_version=_version(model))


@app.post("/predict", response_model=PredictionResponse)
def predict(applicant: Applicant) -> PredictionResponse:
    model = _model()
    pred = _score(model, [applicant])[0]
    return PredictionResponse(**pred.model_dump(), model_version=_version(model))


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(request: BatchRequest) -> BatchResponse:
    model = _model()
    return BatchResponse(
        model_version=_version(model), predictions=_score(model, request.applicants)
    )
