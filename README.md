# AI-Driven Observability & Predictive Analytics for a Digital Bank

End-to-end capstone: data exploration → ML models → log NLP + RAG assistant → FastAPI + Streamlit + Docker.

## 1. Quick start

**Option A – Docker (recommended):**
```bash
docker compose up --build
# Dashboard: http://localhost:8501     API docs: http://localhost:8000/docs
```
**Option B – plain Python (3.12+):**
```bash
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
./run_pipeline.sh            # optional: re-creates all models/figures from the raw data (models are already included)
uvicorn api.main:app --port 8000            # terminal 1
streamlit run dashboard/app.py              # terminal 2  (opens http://localhost:8501)
python -m pytest tests -q                   # smoke tests (pip install pytest)
```
> Docker note: the compose files were validated (`docker compose config`) and the API's dependency list was tested in a clean
> virtual environment, but the images themselves could not be built in the sandbox where this was developed (no Docker daemon).
> If a build fails on your machine, the first thing to check is the Python / package versions in `requirements-api.txt`.

## 2. Project layout
```
data/                 raw CSV/TXT files (copied from the provided dataset)
src/common.py         paths + loading/cleaning (one place for all cleaning rules)
src/01_eda.py         TASK 1  cleaning, exploration, correlations, figures  -> outputs/figures, outputs/eda_findings.json
src/02_train_models.py TASK 2 forecasting, incident classifier, anomaly detection -> models/*.joblib, outputs/model_metrics.json
src/03_log_analysis.py TASK 3 log preprocessing (TF-IDF) + KMeans clustering -> models/log_clusterer.joblib
src/04_build_knowledge_base.py + src/rag.py   knowledge base + TF-IDF retriever (RAG)
src/agent.py          TASK 3 LangGraph assistant (route → gather evidence → retrieve → generate)
src/telemetry.py      service layer used by API and agent (loads data + models, answers questions)
api/main.py           TASK 4 FastAPI endpoints
dashboard/app.py      TASK 4 Streamlit dashboard (talks to the API only)
Dockerfile.api, Dockerfile.dashboard, docker-compose.yml
```

## 3. What each task produced

### Task 1 – Data preparation & exploration
* Cleaning: timestamps parsed, numeric types enforced, 9 exact-duplicate log rows dropped, gaps would be time-interpolated (the files had no missing values).
* Business hours (09:00–17:59) vs off-hours: CPU 64% vs 40%, latency 343 ms vs 114 ms, volume 6,416 vs 4,001 txn/h. Weekends are lighter (CPU 43.8% vs 51.3%).
* **Success rates do not move with load** (correlations ≈ 0). NEFT/IMPS/RTGS success rates are noise around ~98% / 99% / 99.5% in the minute-level file; they only collapse during labelled critical incidents (NEFT 98.0% → 87.3%).
* Latency and resource metrics all rise and fall together with the business-hours cycle (the data was clearly generated with that pattern).
* Transactions file (4,000 rows, 4.78% fraud): processing time is ~1,000–1,050 ms for every transaction type; only 313 transactions fall inside the infra window, and their processing time is uncorrelated with infra load (|r| < 0.05).

### Task 2 – Models (test period = provided `infrastructure_metrics_test.csv`)
| Target (15-min data) | Best baseline RMSE | Deployed model RMSE (gradient boosting on calendar features) |
|---|---|---|
| Latency (ms) | 73.5 (naive) | **5.7** |
| CPU (%) | 7.9 | **0.85** |
| Memory (%) | 7.8 | **0.53** |
| DB pool (%) | 7.3 (seasonal naive) | **1.46** |
| Success rate (%) | 0.107 | 0.104 (≈ no gain: the series is pure noise) |

* **Incident classifier** (`is_critical`): logistic regression, random forest and gradient boosting all score accuracy = F1 = 1.0 in 5-fold CV and on a chronological hold-out. **This is a property of the data, not proof of a great model**: critical hours follow clean thresholds (CPU > 85, memory > 80, DB pool > 85, errors ≥ 10, NEFT < 90). `severity` and `incident_type` are excluded as inputs because they would leak the label.
* **Anomaly detection**: Isolation Forest trained on normal minute-level data catches all 26 critical hours (precision 0.96, F1 0.98). A second detector (forecast-residual z-score > 4) scores F1 = 1.0. Both are evaluated on the hourly incident table.
* Models are saved in `models/`.

### Task 3 – Logs, knowledge base, assistant
* 4,991 logs → TF-IDF → KMeans. Silhouette peaks at k = 17, which is exactly the number of message templates in the file (cluster purity 1.0), so the clustering recovers the templates perfectly. Clusters are grouped into **Infrastructure issue** (CPU, memory, disk, network, container OOM, DB pool), **Business KPI issue** (NEFT/IMPS/RTGS failures, duplicate transactions, auth failures, gateway SLA, API rate limit) and **Informational**.
* **Logs are not informative about incidents**: the mix of log families is the same in critical and normal hours. The assistant therefore reports the log mix honestly and relies on the metrics for evidence.
* Knowledge base (56 documents): 5 runbooks from `incident_knowledge_base.txt`, 29 historical incident documents built from `incident_reports.csv`, 5 pattern/baseline summaries, 17 log patterns. Retrieval = TF-IDF cosine similarity, queried per document type.
* Assistant = LangGraph graph `route → gather_evidence → retrieve → generate`. With no API key it writes the analysis from a deterministic template (works offline). Set `ANTHROPIC_API_KEY` (and `pip install langchain-anthropic`) or `OPENAI_API_KEY` (and `langchain-openai`) and the `generate` step uses an LLM grounded on the same evidence and documents. **The LLM path is written but has not been run** (no key available during development).

### Task 4 – API, dashboard, containers
Endpoints (see `/docs`): `/health`, `/meta`, `/kpis`, `/snapshot`, `/forecast`, `/risk/hourly`, `POST /incident/predict`, `/logs/summary`, `/logs/timeline`, `/logs/clusters`, `POST /logs/classify`, `POST /rca`, `/model-metrics`.
Dashboard tabs: KPIs & forecast · Incidents & anomalies (with a what-if predictor) · Log intelligence · AI assistant · Model performance.

## 4. Limitations to mention in your write-up
1. **"Real time" is a replay.** The data covers one week (20–26 Oct 2025); the sidebar clock replays it. Forecasts are time-of-day patterns and cannot capture weekly trends with 7 days of history.
2. **Two data sources disagree during incidents.** `incident_reports.csv` (hourly) equals the minute-level file except in its 26 critical hours, where CPU/memory/DB spikes exist only in the incident file. The dashboard therefore uses the minute file for KPI tiles/forecasts and the incident table for risk and anomalies.
3. Near-perfect classifier/anomaly scores come from rule-generated labels; real incidents would be messier.
4. The test period is a weekend (lower load), so the forecasting comparison partly measures how well each model handles the weekend pattern.
5. Fraud data (`transactions_fraud_train.csv`) is only explored in Task 1; fraud detection was not part of the brief.
6. Docker images were not built in the development sandbox (see note in section 1).
