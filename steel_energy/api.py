"""Internal demonstration API for the frozen historical benchmark."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from steel_energy.bundle import BundleStore, BundleUnavailable
from steel_energy.config import ROOT, output_root, read_json
from steel_energy.serving import predict_request

store = BundleStore(output_root() / "end_of_day")


@asynccontextmanager
async def lifespan(app):
    try:
        store.get()
    except BundleUnavailable:
        pass  # The dashboard remains accessible before the first pipeline run.
    yield


app = FastAPI(
    title="Steel Energy Forecast - Assignment 2", version="1.1.0", lifespan=lifespan
)
app.mount("/static", StaticFiles(directory=ROOT / "steel_energy/static"), name="static")


class Reading(BaseModel):
    """An interval-end energy reading, in the documented benchmark convention."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    interval_end: str
    energy: float


class ForecastRequest(BaseModel):
    """Target and issue times plus strictly historical readings."""

    model_config = ConfigDict(extra="forbid")
    target_start: str
    issued_at: str
    history: list[Reading] = Field(min_length=672, max_length=672)


@app.get("/health")
def health() -> dict:
    """Report ready only when the frozen local model can be verified and loaded."""
    try:
        _, frozen = store.get()
        return {
            "status": "ready",
            "model_version": frozen["registered_version"],
            **store.status(),
        }
    except BundleUnavailable as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.post("/predict")
def forecast(request: ForecastRequest) -> dict:
    """Validate and score a historical-replay request; never control equipment."""
    try:
        model, frozen = store.get()
        return predict_request(model, frozen, request.model_dump())
    except BundleUnavailable as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except (ValueError, KeyError, TypeError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except OSError as error:
        raise HTTPException(
            status_code=503, detail="Run the pipeline before serving"
        ) from error


@app.get("/", include_in_schema=False)
def dashboard():
    return FileResponse(ROOT / "steel_energy/static/index.html")


@app.get("/evidence")
def evidence():
    try:
        return read_json(output_root() / "engineering/dashboard.json")
    except (OSError, ValueError) as error:
        raise HTTPException(
            503, "Run the pipeline and analyze stage to prepare dashboard evidence"
        ) from error


@app.get("/replay-request")
def replay_request():
    try:
        return read_json(output_root() / "demo-request.json")
    except (OSError, ValueError) as error:
        raise HTTPException(
            503, "Run the verify stage to prepare historical replay"
        ) from error
