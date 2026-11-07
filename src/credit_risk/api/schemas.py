"""Request / response contracts of the scoring API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

PayStatus = Field(
    ...,
    ge=-2,
    le=9,
    description="-2 no consumption, -1 paid in full, 0 revolving credit, 1-9 months of delay",
)


class Applicant(BaseModel):
    """Account history for one card holder (amounts in NT dollars).

    Protected attributes (sex, marital status, age) are deliberately NOT accepted:
    the model does not use them and extra fields are rejected.
    """

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "LIMIT_BAL": 50000,
                "EDUCATION": 2,
                "PAY_1": 2,
                "PAY_2": 2,
                "PAY_3": 0,
                "PAY_4": 0,
                "PAY_5": 0,
                "PAY_6": 0,
                "BILL_AMT1": 48000,
                "BILL_AMT2": 47000,
                "BILL_AMT3": 45500,
                "BILL_AMT4": 30000,
                "BILL_AMT5": 28000,
                "BILL_AMT6": 27000,
                "PAY_AMT1": 0,
                "PAY_AMT2": 1500,
                "PAY_AMT3": 1500,
                "PAY_AMT4": 1200,
                "PAY_AMT5": 1100,
                "PAY_AMT6": 1000,
            }
        },
    )

    LIMIT_BAL: float = Field(..., gt=0, description="Credit limit")
    EDUCATION: int = Field(
        ..., ge=1, le=4, description="1 grad school, 2 university, 3 high school, 4 other"
    )
    PAY_1: int = PayStatus
    PAY_2: int = PayStatus
    PAY_3: int = PayStatus
    PAY_4: int = PayStatus
    PAY_5: int = PayStatus
    PAY_6: int = PayStatus
    BILL_AMT1: float
    BILL_AMT2: float
    BILL_AMT3: float
    BILL_AMT4: float
    BILL_AMT5: float
    BILL_AMT6: float
    PAY_AMT1: float = Field(..., ge=0)
    PAY_AMT2: float = Field(..., ge=0)
    PAY_AMT3: float = Field(..., ge=0)
    PAY_AMT4: float = Field(..., ge=0)
    PAY_AMT5: float = Field(..., ge=0)
    PAY_AMT6: float = Field(..., ge=0)


class RiskFactor(BaseModel):
    feature: str
    contribution: float = Field(..., description="SHAP contribution to the log-odds of default")


class Prediction(BaseModel):
    probability_of_default: float
    decision: Literal["approve", "decline"]
    threshold: float
    top_risk_factors: list[RiskFactor]


class PredictionResponse(Prediction):
    model_version: str


class BatchRequest(BaseModel):
    applicants: list[Applicant] = Field(..., min_length=1, max_length=1000)


class BatchResponse(BaseModel):
    model_version: str
    predictions: list[Prediction]


class Health(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_version: str | None = None
