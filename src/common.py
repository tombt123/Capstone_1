"""Shared helpers: paths, data loading and cleaning.

Every other script imports from here so the cleaning rules live in ONE place.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = ROOT / "models"
OUT = ROOT / "outputs"
FIGS = OUT / "figures"
for _p in (MODELS, OUT, FIGS):
    _p.mkdir(parents=True, exist_ok=True)

# Column groups used throughout the project
SUCCESS_COLS = ["neft_success_rate", "imps_success_rate", "rtgs_success_rate", "auth_success_rate"]
INFRA_COLS = ["cpu_utilization", "memory_usage", "db_connection_pool_usage", "network_throughput_mbps"]
KPI_COLS = SUCCESS_COLS + ["avg_transaction_processing_time_ms", "application_error_rate",
                           "transaction_volume_per_hour", "service_uptime"]
ALL_METRICS = INFRA_COLS + KPI_COLS


def load_metrics(name: str = "infrastructure_metrics.csv") -> pd.DataFrame:
    """Load minute-level infrastructure + business KPI metrics, cleaned."""
    df = pd.read_csv(DATA / name)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).drop_duplicates("timestamp").sort_values("timestamp")
    for c in ALL_METRICS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # Fill any gaps by time interpolation (the shipped file has none, but be safe)
    df = df.set_index("timestamp")
    df[ALL_METRICS] = df[ALL_METRICS].interpolate(method="time").ffill().bfill()
    # Convenience KPI: average payment-rail success rate (NEFT/IMPS/RTGS)
    df["overall_success_rate"] = df[["neft_success_rate", "imps_success_rate", "rtgs_success_rate"]].mean(axis=1)
    return df


def load_transactions() -> pd.DataFrame:
    df = pd.read_csv(DATA / "transactions_fraud_train.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df["processing_time_ms"] = pd.to_numeric(df["processing_time_ms"], errors="coerce")
    df = df.dropna(subset=["timestamp", "amount", "processing_time_ms"]).drop_duplicates("transaction_id")
    for c in ["transaction_type", "merchant_category", "device_type", "location"]:
        df[c] = df[c].astype("category")
    return df.sort_values("timestamp").reset_index(drop=True)


def load_logs() -> pd.DataFrame:
    df = pd.read_csv(DATA / "application_logs.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["log_level"] = df["log_level"].str.upper().str.strip()
    df["log_message"] = df["log_message"].astype(str).str.strip()
    df = df.dropna(subset=["timestamp"]).drop_duplicates()  # 9 exact duplicates in raw file
    return df.sort_values("timestamp").reset_index(drop=True)


def load_incidents() -> pd.DataFrame:
    df = pd.read_csv(DATA / "incident_reports.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def add_incident_labels(metrics: pd.DataFrame) -> pd.Series:
    """Hourly critical-incident flag aligned to the minute-level metrics index."""
    inc = load_incidents().set_index("timestamp")["is_critical"]
    hour = metrics.index.floor("h")
    return pd.Series(inc.reindex(hour).fillna(0).astype(int).to_numpy(), index=metrics.index, name="is_critical")
