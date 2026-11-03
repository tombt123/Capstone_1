"""Smoke tests. Run from the project root:  python -m pytest tests -q   (pip install pytest)"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient  # noqa: E402

from api.main import app  # noqa: E402

CRITICAL = dict(cpu_utilization=90, memory_usage=85, db_connection_pool_usage=90, network_throughput_mbps=250,
                application_error_rate=12, neft_success_rate=88, imps_success_rate=99, rtgs_success_rate=99.5,
                avg_transaction_processing_time_ms=300)
NORMAL = dict(CRITICAL, cpu_utilization=40, memory_usage=36, db_connection_pool_usage=30, application_error_rate=2, neft_success_rate=98)


def test_api():
    with TestClient(app) as c:
        assert c.get("/health").json()["status"] == "ok"
        assert c.post("/incident/predict", json=CRITICAL).json()["is_critical"] is True
        assert c.post("/incident/predict", json=NORMAL).json()["is_critical"] is False
        f = c.get("/forecast", params={"metric": "cpu_utilization", "horizon_hours": 3}).json()
        assert len(f) == 12 and f[0]["lower"] < f[0]["forecast"] < f[0]["upper"]
        assert c.get("/forecast", params={"metric": "nope"}).status_code == 404
        r = c.post("/logs/classify", json={"message": "ERROR: Disk I/O bottleneck detected on srv-3"}).json()
        assert r["issue_group"] == "Infrastructure issue"
        rca = c.post("/rca", json={"question": "Why is the system degraded right now?", "as_of": "2025-10-20 11:30"}).json()
        assert "Critical" in rca["answer"] and rca["sources"]
        gen = c.post("/rca", json={"question": "What causes database connection pool exhaustion?"}).json()
        assert gen["intent"] == "general"
