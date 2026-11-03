"""Telemetry service: ONE place that loads the data + trained models and answers questions about them.

The FastAPI backend, the LangGraph assistant and the dashboard (through the API) all use this class.

About "real time": the project data covers 2025-10-20 .. 2025-10-26. There is no live feed, so the system REPLAYS the data:
every function takes an `as_of` timestamp meaning "pretend it is now this time" (default = end of the data).
"""
from functools import cached_property

import joblib
import numpy as np
import pandas as pd

from common import MODELS, OUT, load_incidents, load_metrics
from features import FORECAST_TARGETS, INCIDENT_FEATURES, calendar_features
from log_nlp import normalize_log

# metric -> short phrase used to query the knowledge base
METRIC_QUERY = {
    "cpu_utilization": "high CPU utilization",
    "memory_usage": "memory usage spike",
    "db_connection_pool_usage": "database connection pool exhausted",
    "neft_success_rate": "NEFT transaction failures gateway timeout",
    "application_error_rate": "application error rate",
    "avg_transaction_processing_time_ms": "slow transaction processing latency",
    "network_throughput_mbps": "network timeout latency",
}


class Telemetry:
    def __init__(self):
        self.metrics = load_metrics()
        self.incidents = load_incidents().set_index("timestamp")
        self.logs = pd.read_csv(OUT / "logs_clustered.csv", parse_dates=["timestamp"])
        self.clf = joblib.load(MODELS / "incident_classifier.joblib")
        self.iso = joblib.load(MODELS / "anomaly_isolation_forest.joblib")["model"]
        self.zcfg = joblib.load(MODELS / "anomaly_zscore_config.joblib")
        self.forecasters = {t: joblib.load(MODELS / f"forecast_{t}.joblib") for t in FORECAST_TARGETS}
        self.log_model = joblib.load(MODELS / "log_clusterer.joblib")
        self.end = self.metrics.index.max()
        self.start = self.metrics.index.min()

    # ------------------------------------------------------------------ helpers
    def _ts(self, as_of) -> pd.Timestamp:
        ts = pd.Timestamp(as_of) if as_of else self.end
        return min(max(ts, self.start), self.end)

    def data_range(self):
        return {"start": str(self.start), "end": str(self.end)}

    # ------------------------------------------------------------------ KPI series
    def kpi_series(self, start=None, end=None, freq="15min", columns=None) -> pd.DataFrame:
        start, end = (pd.Timestamp(start) if start else self.start), self._ts(end)
        df = self.metrics.loc[start:end]
        cols = columns or (list(FORECAST_TARGETS) + ["transaction_volume_per_hour", "application_error_rate", "network_throughput_mbps"])
        return df[cols].resample(freq).mean()

    # ------------------------------------------------------------------ forecasting
    def forecast(self, metric: str, horizon_hours: int = 24, as_of=None) -> pd.DataFrame:
        f = self.forecasters[metric]
        ts = self._ts(as_of)
        idx = pd.date_range(ts.floor("15min") + pd.Timedelta(minutes=15), periods=horizon_hours * 4, freq="15min")
        yhat = f["model"].predict(calendar_features(idx))
        band = 1.96 * f["resid_std"]
        return pd.DataFrame({"timestamp": idx, "forecast": yhat, "lower": yhat - band, "upper": yhat + band}).set_index("timestamp")

    # ------------------------------------------------------------------ incident risk + anomalies
    def classify_incident(self, row: dict) -> dict:
        X = pd.DataFrame([{k: row[k] for k in INCIDENT_FEATURES}])
        p = float(self.clf["model"].predict_proba(X)[0, 1])
        return {"critical_probability": round(p, 4), "is_critical": bool(p >= 0.5),
                "risk_level": "Critical" if p >= 0.5 else ("Elevated" if p >= 0.2 else "Normal"), "model": self.clf["name"]}

    def deviations(self, row: dict, ts: pd.Timestamp) -> list[dict]:
        """How far is each metric from what is normal for this time of day? (z-score; >4 = anomalous)"""
        out = []
        cal = calendar_features(pd.DatetimeIndex([ts]))
        for t in self.zcfg["resid_targets"]:
            f = self.forecasters[t]
            exp = float(f["model"].predict(cal)[0])
            out.append({"metric": t, "value": float(row[t]), "expected": exp, "z": (float(row[t]) - exp) / f["raw_resid_std"]})
        for t in ["neft_success_rate", "application_error_rate"]:
            mu, sd = self.metrics[t].mean(), self.metrics[t].std()
            out.append({"metric": t, "value": float(row[t]), "expected": float(mu), "z": (float(row[t]) - mu) / sd})
        thr = self.zcfg["threshold"]
        for d in out:
            d["anomalous"] = bool(abs(d["z"]) > thr)
            d["z"] = round(float(d["z"]), 2)
        return sorted(out, key=lambda d: -abs(d["z"]))

    def hour_record(self, as_of=None) -> tuple[dict, pd.Timestamp, str]:
        """The hourly operational record for the hour containing `as_of`.
        Uses incident_reports.csv (the only source containing incident-hour spikes); falls back to minute averages."""
        ts = self._ts(as_of)
        h = ts.floor("h")
        if h in self.incidents.index:
            return self.incidents.loc[h].to_dict(), h, "incident_reports.csv (hourly)"
        return self.metrics.loc[h:h + pd.Timedelta(minutes=59)].mean(numeric_only=True).to_dict(), h, "infrastructure_metrics.csv (hourly mean)"

    def snapshot(self, as_of=None) -> dict:
        ts = self._ts(as_of)
        rec, h, source = self.hour_record(ts)
        latest = self.metrics.loc[ts - pd.Timedelta(minutes=14):ts].mean(numeric_only=True)
        risk = self.classify_incident(rec)
        iso_flag = bool(self.iso.predict(pd.DataFrame([{k: rec[k] for k in INCIDENT_FEATURES}]))[0] == -1)
        devs = self.deviations(rec, h)
        return {
            "as_of": str(ts), "hour": str(h), "record_source": source,
            "latest_15min_kpis": {k: round(float(v), 2) for k, v in latest.items() if k in
                                  list(FORECAST_TARGETS) + ["transaction_volume_per_hour", "application_error_rate", "network_throughput_mbps",
                                                            "neft_success_rate", "imps_success_rate", "rtgs_success_rate", "auth_success_rate", "service_uptime"]},
            "hour_record": {k: round(float(rec[k]), 2) for k in INCIDENT_FEATURES},
            "incident_risk": risk, "isolation_forest_anomaly": iso_flag,
            "deviations": devs, "anomalous_metrics": [d["metric"] for d in devs if d["anomalous"]],
            "labelled_incident": (self.incidents.loc[h, ["incident_type", "severity"]].to_dict() if h in self.incidents.index else None),
        }

    def hourly_risk(self, start=None, end=None) -> pd.DataFrame:
        """Risk probability + anomaly flag for every hour with a record (for the dashboard timeline)."""
        inc = self.incidents.loc[(pd.Timestamp(start) if start else self.start):self._ts(end)]
        X = inc[INCIDENT_FEATURES]
        out = inc[["cpu_utilization", "memory_usage", "db_connection_pool_usage", "neft_success_rate", "application_error_rate"]].copy()
        out["critical_probability"] = self.clf["model"].predict_proba(X)[:, 1]
        out["isolation_forest_anomaly"] = (self.iso.predict(X) == -1)
        out["labelled_severity"] = inc["severity"]
        return out

    # ------------------------------------------------------------------ logs
    def log_insights(self, as_of=None, window_minutes: int = 60) -> dict:
        ts = self._ts(as_of)
        w = self.logs[(self.logs.timestamp > ts - pd.Timedelta(minutes=window_minutes)) & (self.logs.timestamp <= ts)]
        # baseline = typical count per window over the full week
        hours = max((self.logs.timestamp.max() - self.logs.timestamp.min()).total_seconds() / 60 / window_minutes, 1)
        base = self.logs.groupby("issue_group").size() / hours
        counts = w.groupby("issue_group").size()
        groups = {}
        for g in base.index:
            groups[g] = {"count": int(counts.get(g, 0)), "typical_per_window": round(float(base[g]), 1)}
        top = (w[w.issue_group != "Informational / Normal"].groupby(["cluster_name", "issue_group"]).size().sort_values(ascending=False).head(5))
        return {"window_minutes": window_minutes, "window_end": str(ts), "total_logs": int(len(w)), "groups": groups,
                "top_issue_clusters": [{"pattern": n, "group": g, "count": int(c)} for (n, g), c in top.items()],
                "levels": w.log_level.value_counts().to_dict()}

    def log_timeline(self, start=None, end=None, freq="1h") -> pd.DataFrame:
        d = self.logs.set_index("timestamp").loc[(pd.Timestamp(start) if start else self.start):self._ts(end)]
        return d.groupby("issue_group").resample(freq).size().unstack(0).fillna(0).astype(int)

    def classify_log(self, message: str, min_similarity: float = 0.5) -> dict:
        """Assign a log line to its nearest cluster, but say "Unknown" when it does not really resemble any cluster.

        Similarity = cosine between the message's TF-IDF vector and the cluster centre. A message sharing no words
        with the training logs has an all-zero vector (similarity 0) -> Unknown. (Euclidean distance is misleading
        here: a zero vector is 'close' to the smallest centroid.)"""
        vec, km, summ = self.log_model["vectorizer"], self.log_model["kmeans"], self.log_model["summary"]
        clean = normalize_log(message)
        X = vec.transform([clean])
        centres = km.cluster_centers_
        norms = np.linalg.norm(centres, axis=1) * (np.sqrt(X.multiply(X).sum()) or 1.0)
        sims = np.asarray(X @ centres.T).ravel() / np.where(norms == 0, 1, norms)
        c = int(np.argmax(sims))
        sim = float(sims[c])
        known = sim >= min_similarity
        s = summ[c] if c in summ else summ[str(c)]
        return {"cluster": c if known else None, "cluster_name": s["name"] if known else None,
                "issue_group": s["group"] if known else "Unknown / new pattern", "normalized": clean,
                "similarity": round(sim, 3), "known_pattern": known, "closest_cluster": s["name"]}
