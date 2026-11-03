"""Feature helpers shared by training scripts and the API (so both always use identical features)."""
import numpy as np
import pandas as pd

CALENDAR_FEATURES = ["hour_frac", "dayofweek", "is_weekend", "is_business_hours", "hour_sin", "hour_cos"]

# Columns the incident classifier / anomaly detector use (all exist in incident_reports.csv)
INCIDENT_FEATURES = [
    "cpu_utilization", "memory_usage", "db_connection_pool_usage", "network_throughput_mbps",
    "application_error_rate", "neft_success_rate", "imps_success_rate", "rtgs_success_rate",
    "avg_transaction_processing_time_ms",
]

FORECAST_TARGETS = {
    "overall_success_rate": "Transaction success rate (%) - mean of NEFT/IMPS/RTGS",
    "avg_transaction_processing_time_ms": "API / transaction latency (ms)",
    "cpu_utilization": "CPU utilization (%)",
    "memory_usage": "Memory usage (%)",
    "db_connection_pool_usage": "DB connection-pool usage (%)",
}


def calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Time-of-day / day-of-week features. They are known in advance, so they work for ANY forecast horizon."""
    idx = pd.DatetimeIndex(index)
    hour_frac = idx.hour + idx.minute / 60.0
    out = pd.DataFrame(index=idx)
    out["hour_frac"] = hour_frac
    out["dayofweek"] = idx.dayofweek
    out["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    out["is_business_hours"] = ((idx.hour >= 9) & (idx.hour <= 17)).astype(int)
    out["hour_sin"] = np.sin(2 * np.pi * hour_frac / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour_frac / 24)
    return out[CALENDAR_FEATURES]
