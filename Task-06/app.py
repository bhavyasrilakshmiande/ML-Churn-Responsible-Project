"""FastAPI inference service for the saved Task-04 churn pipeline."""

import os
from pathlib import Path
from typing import Any, Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field


FEATURE_NAMES = [
    "tenure_months",
    "support_tickets",
    "monthly_spend_inr",
    "last_login_days",
    "plan_type",
]
EXPECTED_CLASSES = {0, 1}
PLAN_TYPES = Literal["Basic", "Pro", "Standard"]
APP_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = APP_DIRECTORY.parent
DEFAULT_MODEL_PATH = REPOSITORY_ROOT / "Task-04" / "champion_model.joblib"
MODEL_PATH = Path(os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH)).resolve()


class PredictionRequest(BaseModel):
    """Validated raw feature values required by the fitted pipeline."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    tenure_months: int = Field(ge=0)
    support_tickets: int = Field(ge=0)
    monthly_spend_inr: float = Field(ge=0)
    last_login_days: int = Field(ge=0)
    plan_type: PLAN_TYPES


class PredictionResponse(BaseModel):
    predicted_class: int
    class_probabilities: dict[str, float] | None


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model_loaded: bool


def load_champion_model(path: Path) -> tuple[Any | None, str]:
    """Load and validate the saved pipeline without exposing loading details."""
    if not path.is_file():
        return None, "missing"

    try:
        loaded_model = joblib.load(path)
    except Exception:
        return None, "unreadable"

    if not callable(getattr(loaded_model, "predict", None)):
        return None, "incompatible"

    saved_features = list(getattr(loaded_model, "feature_names_in_", []))
    if saved_features != FEATURE_NAMES:
        return None, "incompatible"

    saved_classes = getattr(loaded_model, "classes_", None)
    if saved_classes is None or {int(label) for label in saved_classes} != EXPECTED_CLASSES:
        return None, "incompatible"

    return loaded_model, "ready"


model, model_status = load_champion_model(MODEL_PATH)
app = FastAPI(
    title="Customer Churn Inference API",
    version="1.0.0",
    description="Inference using the saved Task-04 preprocessing and classification pipeline.",
)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Return field-level validation details without echoing submitted values."""
    sanitized_errors = [
        {"loc": list(error["loc"]), "msg": error["msg"], "type": error["type"]}
        for error in exc.errors()
    ]
    return JSONResponse(status_code=422, content={"detail": sanitized_errors})


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="The prediction model is unavailable.",
        )
    return HealthResponse(status="ok", model_loaded=True)


@app.post("/predict", response_model=PredictionResponse)
def predict(request_data: PredictionRequest) -> PredictionResponse:
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="The prediction model is unavailable.",
        )

    try:
        row = pd.DataFrame(
            [request_data.model_dump()],
            columns=FEATURE_NAMES,
        )
        predicted_class = int(model.predict(row)[0])
        probabilities: dict[str, float] | None = None

        predict_proba = getattr(model, "predict_proba", None)
        if callable(predict_proba):
            probability_values = predict_proba(row)[0]
            model_classes = model.classes_
            if len(probability_values) != len(model_classes):
                raise ValueError("Probability output does not match the model classes.")
            probabilities = {
                str(int(class_label)): float(probability)
                for class_label, probability in zip(model_classes, probability_values)
            }

        return PredictionResponse(
            predicted_class=predicted_class,
            class_probabilities=probabilities,
        )
    except Exception as exc:
        if isinstance(exc, HTTPException):
            raise
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check the submitted feature values and try again.",
        ) from None
