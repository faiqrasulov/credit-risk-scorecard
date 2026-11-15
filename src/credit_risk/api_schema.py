"""Pydantic request/response models of the scoring API (levels generated from ``schema``)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from credit_risk.schema import CATEGORICAL_LEVELS as L

EXAMPLE_APPLICANT = {
    "checking_status": "<0", "duration": 24, "credit_history": "existing paid",
    "purpose": "radio/tv", "credit_amount": 3500, "savings_status": "<100",
    "employment": "1<=X<4", "installment_commitment": 4, "personal_status": "male single",
    "other_parties": "none", "residence_since": 2, "property_magnitude": "car", "age": 31,
    "other_payment_plans": "none", "housing": "own", "existing_credits": 1, "job": "skilled",
    "num_dependents": 1, "own_telephone": "none", "foreign_worker": "yes",
}


class Applicant(BaseModel):
    """The 20 fields of a German Credit application (Statlog labels as shipped by OpenML).

    age, personal_status and foreign_worker are accepted (they are on the application form and
    are needed for fairness monitoring) but are not used by the model.
    """

    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [EXAMPLE_APPLICANT]})

    checking_status: Literal[L["checking_status"]]  # type: ignore[valid-type]
    duration: int = Field(ge=1, le=120, description="Loan duration in months")
    credit_history: Literal[L["credit_history"]]  # type: ignore[valid-type]
    purpose: Literal[L["purpose"]]  # type: ignore[valid-type]
    credit_amount: float = Field(gt=0, le=1_000_000, description="Loan amount (DM)")
    savings_status: Literal[L["savings_status"]]  # type: ignore[valid-type]
    employment: Literal[L["employment"]]  # type: ignore[valid-type]
    installment_commitment: Literal[1, 2, 3, 4]
    personal_status: Literal[L["personal_status"]]  # type: ignore[valid-type]
    other_parties: Literal[L["other_parties"]]  # type: ignore[valid-type]
    residence_since: Literal[1, 2, 3, 4]
    property_magnitude: Literal[L["property_magnitude"]]  # type: ignore[valid-type]
    age: int = Field(ge=18, le=100, description="Age in years (not used by the model)")
    other_payment_plans: Literal[L["other_payment_plans"]]  # type: ignore[valid-type]
    housing: Literal[L["housing"]]  # type: ignore[valid-type]
    existing_credits: Literal[1, 2, 3, 4]
    job: Literal[L["job"]]  # type: ignore[valid-type]
    num_dependents: Literal[1, 2]
    own_telephone: Literal[L["own_telephone"]]  # type: ignore[valid-type]
    foreign_worker: Literal[L["foreign_worker"]]  # type: ignore[valid-type]


class ReasonCode(BaseModel):
    feature: str
    value: str | float
    bin: str
    points: float
    best_bin_points: float
    points_lost: float = Field(description="Points below the feature's best bin")


class ScoreResponse(BaseModel):
    probability_of_default: float
    score: float
    decision: Literal["approve", "reject"]
    threshold_pd: float = Field(description="Reject when PD >= this cost-optimal cutoff")
    cutoff_score: float = Field(description="Equivalent score cutoff (reject at or below)")
    reason_codes: list[ReasonCode] = Field(
        description="Top-3 features costing the most points vs their best bin")


class Health(BaseModel):
    status: str
    model: str
    n_features: int
    threshold_pd: float
    cutoff_score: float
    model_version: str
