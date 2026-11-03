"""FastAPI backend: model inference + log insights + AI root-cause analysis.

Run locally:   uvicorn api.main:app --reload --port 8000      (from the project root)
Interactive docs:  http://localhost:8000/docs
"""
import json
import math
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent import IncidentAssistant  # noqa: E402
from features import FORECAST_TARGETS  # noqa: E402
from telemetry import Telemetry  # noqa: E402

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    tel = Telemetry()                      # loads data + trained models once at startup
    state["tel"] = tel
    state["assistant"] = IncidentAssistant(tel)
    yield


app = FastAPI(title="Digital Bank AI Observability API", version="1.0", lifespan=lifespan)


def clean(obj):
    """Make pandas / numpy values JSON-safe (NaN -> None, Timestamp -> str)."""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, (pd.Timestamp,)):
        return str(obj)
    if isinstance(obj, (np.floating, float)):
        return None if math.isnan(obj) or math.isinf(obj) else float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def frame_to_records(df: pd.DataFrame) -> list[dict]:
    d = df.reset_index()
    d.iloc[:, 0] = d.iloc[:, 0].astype(str)
    return clean(d.to_dict(orient="records"))


# ------------------------------------------------------------------------------------------ schemas
class IncidentInput(BaseModel):
    cpu_utilization: float = Field(..., ge=0, le=100, examples=[88.0])
    memory_usage: float = Field(..., ge=0, le=100, examples=[85.0])
    db_connection_pool_usage: float = Field(..., ge=0, le=100, examples=[90.0])
    network_throughput_mbps: float = Field(..., ge=0, examples=[250.0])
    application_error_rate: float = Field(..., ge=0, examples=[12])
    neft_success_rate: float = Field(..., ge=0, le=100, examples=[88.0])
    imps_success_rate: float = Field(..., ge=0, le=100, examples=[99.0])
    rtgs_success_rate: float = Field(..., ge=0, le=100, examples=[99.5])
    avg_transaction_processing_time_ms: float = Field(..., ge=0, examples=[300.0])


class LogInput(BaseModel):
    message: str = Field(..., examples=["ERROR: Database connection pool exhausted. Connections: 495"])


class RCARequest(BaseModel):
    question: str = Field(..., examples=["Why is the system degraded right now? Give root cause and actions"])
    as_of: Optional[str] = Field(None, examples=["2025-10-20 11:30"], description="Replay time; default = end of data")


# ------------------------------------------------------------------------------------------ endpoints
@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": "tel" in state}


@app.get("/meta")
def meta():
    tel: Telemetry = state["tel"]
    return {"data_range": tel.data_range(), "forecast_targets": FORECAST_TARGETS,
            "incident_model": tel.clf["name"], "llm_enabled": state["assistant"].llm_enabled}


@app.get("/kpis")
def kpis(start: Optional[str] = None, end: Optional[str] = None, freq: str = "15min"):
    return frame_to_records(state["tel"].kpi_series(start, end, freq))


@app.get("/snapshot")
def snapshot(as_of: Optional[str] = None):
    return clean(state["tel"].snapshot(as_of))


@app.get("/forecast")
def forecast(metric: str = Query("cpu_utilization"), horizon_hours: int = Query(24, ge=1, le=72), as_of: Optional[str] = None):
    if metric not in FORECAST_TARGETS:
        raise HTTPException(404, f"metric must be one of {list(FORECAST_TARGETS)}")
    return frame_to_records(state["tel"].forecast(metric, horizon_hours, as_of))


@app.get("/risk/hourly")
def risk_hourly(start: Optional[str] = None, end: Optional[str] = None):
    return frame_to_records(state["tel"].hourly_risk(start, end))


@app.post("/incident/predict")
def incident_predict(body: IncidentInput):
    tel: Telemetry = state["tel"]
    row = body.model_dump()
    res = tel.classify_incident(row)
    devs = tel.deviations(row, pd.Timestamp("2025-10-22 11:00"))  # compare against a weekday business-hour baseline
    return clean({**res, "top_deviations_vs_weekday": devs[:3]})


@app.get("/logs/summary")
def logs_summary(as_of: Optional[str] = None, window_minutes: int = Query(60, ge=5, le=1440)):
    return clean(state["tel"].log_insights(as_of, window_minutes))


@app.get("/logs/timeline")
def logs_timeline(start: Optional[str] = None, end: Optional[str] = None, freq: str = "1h"):
    return frame_to_records(state["tel"].log_timeline(start, end, freq))


@app.get("/logs/clusters")
def logs_clusters():
    from common import OUT
    return json.loads((OUT / "log_cluster_summary.json").read_text())


@app.post("/logs/classify")
def logs_classify(body: LogInput):
    return clean(state["tel"].classify_log(body.message))


@app.post("/rca")
def rca(body: RCARequest):
    return clean(state["assistant"].ask(body.question, body.as_of))


@app.get("/model-metrics")
def model_metrics():
    from common import OUT
    return json.loads((OUT / "model_metrics.json").read_text())
