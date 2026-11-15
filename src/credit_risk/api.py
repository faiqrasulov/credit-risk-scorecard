"""FastAPI service: ``GET /health`` and ``POST /score``.

The model artifact is read from ``$CREDIT_RISK_MODEL`` (default ``models/scorecard.joblib``);
create it with ``credit-risk train``.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from credit_risk import __version__
from credit_risk.api_schema import Applicant, Health, ScoreResponse
from credit_risk.data import clean_frame
from credit_risk.scorecard import Scorecard
from credit_risk.train import DEFAULT_MODEL_PATH, load_artifact


def create_app(artifact: dict[str, Any] | None = None,
               model_path: str | Path | None = None) -> FastAPI:
    """Build the app around an in-memory artifact, or load one from disk at startup."""
    state: dict[str, Any] = {"artifact": None, "scorecard": None}

    def install(a: dict[str, Any]) -> None:
        state["artifact"] = a
        state["scorecard"] = Scorecard(a["pipeline"], a["scaler"])

    if artifact is not None:
        install(artifact)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if state["artifact"] is None:
            path = model_path or os.environ.get("CREDIT_RISK_MODEL", str(DEFAULT_MODEL_PATH))
            try:
                install(load_artifact(path))
            except FileNotFoundError:
                pass  # /health reports 503 until a model exists
        yield

    app = FastAPI(title="Credit-risk scorecard", version=__version__, lifespan=lifespan,
                  description="WoE scorecard PD, score points, cost-optimal decision and "
                              "adverse-action reason codes for German Credit applicants.")

    @app.get("/health", response_model=Health)
    def health() -> Any:
        a = state["artifact"]
        if a is None:
            return JSONResponse(status_code=503, content={
                "status": "model not loaded: run `credit-risk train`"})
        return Health(status="ok", model="woe_scorecard",
                      n_features=len(state["scorecard"].features),
                      threshold_pd=round(a["threshold_pd"], 6),
                      cutoff_score=round(a["cutoff_score"], 2),
                      model_version=a["versions"]["credit_risk"])

    @app.post("/score", response_model=ScoreResponse)
    def score(applicant: Applicant) -> ScoreResponse:
        a, sc = state["artifact"], state["scorecard"]
        if a is None:
            raise HTTPException(status_code=503, detail="model not loaded")
        X = clean_frame(pd.DataFrame([applicant.model_dump()]))
        pd_ = float(sc.predict_pd(X)[0])
        return ScoreResponse(
            probability_of_default=round(pd_, 6),
            score=round(float(sc.score(X)[0]), 2),
            decision="reject" if pd_ >= a["threshold_pd"] else "approve",
            threshold_pd=round(a["threshold_pd"], 6),
            cutoff_score=round(a["cutoff_score"], 2),
            reason_codes=sc.reason_codes(X, k=3)[0],
        )

    return app


app = create_app()
