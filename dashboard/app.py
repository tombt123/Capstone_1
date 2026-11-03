"""Streamlit dashboard for the Digital Bank AI Observability platform.

Run locally (API must be running on port 8000):   streamlit run dashboard/app.py
The dashboard only talks to the FastAPI backend (API_URL env variable), never to the models directly.

"Real time" = replay: pick the simulated current time in the sidebar (or press Play to advance it automatically).
"""
import os
import time
from datetime import timedelta

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

API = os.getenv("API_URL", "http://localhost:8000")
st.set_page_config(page_title="Bank AI Observability", page_icon="🏦", layout="wide")


# ------------------------------------------------------------------------------------------ API helpers
@st.cache_data(ttl=60, show_spinner=False)
def get(path: str, **params):
    r = requests.get(f"{API}{path}", params={k: v for k, v in params.items() if v is not None}, timeout=60)
    r.raise_for_status()
    return r.json()


def post(path: str, body: dict):
    r = requests.post(f"{API}{path}", json=body, timeout=120)
    r.raise_for_status()
    return r.json()


try:
    meta = get("/meta")
except Exception as e:
    st.error(f"Cannot reach the API at {API}. Start it with `uvicorn api.main:app --port 8000`.\n\n{e}")
    st.stop()

start, end = pd.Timestamp(meta["data_range"]["start"]), pd.Timestamp(meta["data_range"]["end"])
targets = meta["forecast_targets"]
risk_df = pd.DataFrame(get("/risk/hourly"))
risk_df["timestamp"] = pd.to_datetime(risk_df["timestamp"])
critical_hours = risk_df[risk_df.labelled_severity == "Critical"]["timestamp"].tolist()

# ------------------------------------------------------------------------------------------ sidebar: replay clock
st.sidebar.title("🏦 Bank AI Observability")
st.sidebar.caption(f"Data replay: {start:%b %d} → {end:%b %d %Y}")
if "now" not in st.session_state:
    st.session_state.now = (critical_hours[0] + timedelta(minutes=30)).to_pydatetime()

jump = st.sidebar.selectbox("Jump to a critical incident", ["—"] + [f"{t:%a %b %d %H:%M}" for t in critical_hours])
if jump != "—" and st.session_state.get("last_jump") != jump:
    st.session_state.now = critical_hours[[f"{t:%a %b %d %H:%M}" for t in critical_hours].index(jump)].to_pydatetime() + timedelta(minutes=30)
    st.session_state.last_jump = jump
now = st.sidebar.slider("Simulated current time", min_value=start.to_pydatetime(), max_value=end.to_pydatetime(),
                        value=st.session_state.now, step=timedelta(minutes=15), format="ddd MMM DD, HH:mm")
st.session_state.now = now
play = st.sidebar.toggle("▶ Live replay (+15 min every 3 s)")
history_h = st.sidebar.selectbox("History window", [6, 12, 24, 48], index=2, format_func=lambda h: f"{h} hours")
horizon_h = st.sidebar.selectbox("Forecast horizon", [6, 12, 24, 48], index=2, format_func=lambda h: f"{h} hours")
metric = st.sidebar.selectbox("Metric to chart", list(targets), index=list(targets).index("cpu_utilization"), format_func=lambda m: targets[m])
st.sidebar.caption(f"API: {API}  •  Incident model: {meta['incident_model']}  •  LLM: {'on' if meta['llm_enabled'] else 'offline template'}")

as_of = pd.Timestamp(now).strftime("%Y-%m-%d %H:%M:%S")
snap = get("/snapshot", as_of=as_of)

st.title("Digital Banking – AI Observability")
st.caption(f"Simulated time: **{pd.Timestamp(now):%A %d %b %Y, %H:%M}**  •  hourly record source: {snap['record_source']}")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 KPIs & forecast", "🚨 Incidents & anomalies", "📜 Log intelligence", "🤖 AI assistant", "🧪 Model performance"])

# ------------------------------------------------------------------------------------------ TAB 1: KPIs
with tab1:
    risk = snap["incident_risk"]
    k = snap["latest_15min_kpis"]
    c = st.columns(7)
    c[0].metric("Txn success rate", f"{k['overall_success_rate']:.2f}%")
    c[1].metric("Latency", f"{k['avg_transaction_processing_time_ms']:.0f} ms")
    c[2].metric("CPU", f"{k['cpu_utilization']:.0f}%")
    c[3].metric("Memory", f"{k['memory_usage']:.0f}%")
    c[4].metric("DB pool", f"{k['db_connection_pool_usage']:.0f}%")
    c[5].metric("Volume / hour", f"{k['transaction_volume_per_hour']:.0f}")
    c[6].metric("Incident risk (this hour)", f"{risk['critical_probability']:.0%}", risk["risk_level"],
                delta_color="inverse" if risk["risk_level"] != "Normal" else "off")
    st.caption("Tiles = average of the last 15 minutes (minute-level file). Incident risk uses the hourly incident record, "
               "which is the only source containing incident-hour spikes.")

    hist = pd.DataFrame(get("/kpis", start=str(pd.Timestamp(now) - timedelta(hours=history_h)), end=as_of, freq="15min"))
    fc = pd.DataFrame(get("/forecast", metric=metric, horizon_hours=horizon_h, as_of=as_of))
    fig = go.Figure()
    if not hist.empty:
        hist["timestamp"] = pd.to_datetime(hist["timestamp"])
        fig.add_scatter(x=hist["timestamp"], y=hist[metric], name="actual", line=dict(color="#1f77b4"))
    fc["timestamp"] = pd.to_datetime(fc["timestamp"])
    fig.add_scatter(x=fc["timestamp"], y=fc["upper"], line=dict(width=0), showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=fc["timestamp"], y=fc["lower"], fill="tonexty", fillcolor="rgba(255,127,14,0.18)", line=dict(width=0),
                    name="95% band", hoverinfo="skip")
    fig.add_scatter(x=fc["timestamp"], y=fc["forecast"], name="forecast", line=dict(color="#ff7f0e", dash="dash"))
    fig.add_vline(x=pd.Timestamp(now), line_dash="dot", line_color="grey")
    fig.update_layout(title=f"{targets[metric]} – actual and {horizon_h}h forecast", height=420, margin=dict(t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Forecast = gradient-boosting model on time-of-day / weekday features. The data covers one week, so it can learn "
               "the business-hours pattern but not week-over-week trends. Success-rate forecasts are essentially flat: that KPI is random noise around ~98.8%.")

# ------------------------------------------------------------------------------------------ TAB 2: incidents
with tab2:
    left, right = st.columns([3, 2])
    with left:
        win = risk_df[(risk_df.timestamp <= pd.Timestamp(now).floor("h")) & (risk_df.timestamp > pd.Timestamp(now) - timedelta(hours=72))]
        fig = go.Figure()
        fig.add_bar(x=win["timestamp"], y=win["critical_probability"], name="predicted critical probability",
                    marker_color=["#d62728" if p >= 0.5 else "#9ecae1" for p in win["critical_probability"]])
        crit = win[win.labelled_severity == "Critical"]
        fig.add_scatter(x=crit["timestamp"], y=[1.05] * len(crit), mode="markers", marker=dict(symbol="triangle-down", color="black", size=9),
                        name="labelled critical incident")
        an = win[win.isolation_forest_anomaly]
        fig.add_scatter(x=an["timestamp"], y=[1.12] * len(an), mode="markers", marker=dict(symbol="x", color="orange", size=8), name="Isolation Forest flag")
        fig.update_layout(title="Hourly incident risk (last 72 h)", height=380, yaxis_range=[0, 1.2], margin=dict(t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.subheader(f"Why is this hour {risk['risk_level'].lower()}?")
        st.write(f"Hour: **{snap['hour']}**  •  labelled: **{(snap['labelled_incident'] or {}).get('incident_type', 'n/a')}**")
        dev = pd.DataFrame(snap["deviations"])
        dev["status"] = dev["anomalous"].map({True: "🔴 anomalous", False: "🟢 normal"})
        st.dataframe(dev[["metric", "value", "expected", "z", "status"]].round(2), hide_index=True, use_container_width=True)
        st.caption("z = distance from what is expected at this time of day, in typical-error units. |z| > 4 is flagged.")

    st.subheader("What-if: predict incident criticality from KPIs")
    rec = snap["hour_record"]
    cols = st.columns(5)
    inputs = {}
    spec = [("cpu_utilization", 0.0, 100.0), ("memory_usage", 0.0, 100.0), ("db_connection_pool_usage", 0.0, 100.0),
            ("network_throughput_mbps", 0.0, 600.0), ("application_error_rate", 0.0, 30.0), ("neft_success_rate", 50.0, 100.0),
            ("imps_success_rate", 50.0, 100.0), ("rtgs_success_rate", 50.0, 100.0), ("avg_transaction_processing_time_ms", 0.0, 1000.0)]
    for i, (name, lo, hi) in enumerate(spec):
        inputs[name] = cols[i % 5].number_input(name, lo, hi, float(min(max(rec[name], lo), hi)), key=f"wi_{name}_{snap['hour']}")
    if st.button("Predict criticality"):
        res = post("/incident/predict", inputs)
        (st.error if res["is_critical"] else st.success)(f"{res['risk_level']} – critical probability {res['critical_probability']:.1%}")

# ------------------------------------------------------------------------------------------ TAB 3: logs
with tab3:
    logs = get("/logs/summary", as_of=as_of, window_minutes=60)
    c = st.columns(3)
    for col, (grp, v) in zip(c, logs["groups"].items()):
        col.metric(grp, v["count"], f"typical {v['typical_per_window']}", delta_color="off")
    tl = pd.DataFrame(get("/logs/timeline", start=str(pd.Timestamp(now) - timedelta(hours=history_h * 2)), end=as_of, freq="1h"))
    if not tl.empty:
        tl["timestamp"] = pd.to_datetime(tl["timestamp"])
        palette = {"Infrastructure issue": "#d62728", "Business KPI issue": "#1f77b4", "Informational / Normal": "#7f7f7f"}
        fig = go.Figure()
        for g in [c for c in tl.columns if c != "timestamp"]:
            fig.add_bar(x=tl["timestamp"], y=tl[g], name=g, marker_color=palette.get(g))
        fig.update_layout(barmode="stack", title="Log volume per hour by issue family (unsupervised clusters)", height=330, margin=dict(t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)
    a, b = st.columns(2)
    with a:
        st.subheader("Top issue clusters (last hour)")
        st.dataframe(pd.DataFrame(logs["top_issue_clusters"]), hide_index=True, use_container_width=True)
    with b:
        st.subheader("Classify a log line")
        msg = st.text_input("Log message", "ERROR: Disk I/O bottleneck detected on srv-7")
        if msg:
            res = post("/logs/classify", {"message": msg})
            if res["known_pattern"]:
                st.write(f"**{res['issue_group']}** — cluster *{res['cluster_name']}* (similarity {res['similarity']:.2f})")
            else:
                st.warning(f"**Unknown / new pattern** – no known cluster matches (closest: *{res['closest_cluster']}*, "
                           f"similarity {res['similarity']:.2f}). Possibly a new issue type worth investigating.")
    cl = get("/logs/clusters")
    with st.expander(f"All {cl['best_k']} clusters (silhouette {cl['silhouette']}, purity {cl['purity']})"):
        st.dataframe(pd.DataFrame(cl["clusters"])[["cluster", "group", "size", "dominant_level", "example_message"]], hide_index=True, use_container_width=True)

# ------------------------------------------------------------------------------------------ TAB 4: assistant
with tab4:
    st.write("Ask about the system **at the simulated time** or about incident handling in general. "
             "The agent gathers live evidence → retrieves runbooks and similar past incidents (RAG) → writes the analysis.")
    examples = ["Why is the system degraded right now? Give root cause and actions", "What is the status right now?",
                "What causes database connection pool exhaustion?", "How do I fix NEFT transaction failures?"]
    q = st.selectbox("Example questions", [""] + examples)
    question = st.text_input("Your question", q)
    if st.button("Analyse", type="primary") and question:
        with st.spinner("Gathering evidence and retrieving knowledge…"):
            res = post("/rca", {"question": question, "as_of": as_of})
        st.markdown(res["answer"])
        st.caption(f"engine: {res['engine']}  •  routed as: {res['intent']}")
        with st.expander("Sources retrieved"):
            st.dataframe(pd.DataFrame(res["sources"]), hide_index=True, use_container_width=True)

# ------------------------------------------------------------------------------------------ TAB 5: model performance
with tab5:
    mm = get("/model-metrics")
    st.subheader("Forecasting – test-set RMSE / MAE")
    rows = [{"target": t, "model": m, **s} for t, d in mm["forecasting"]["scores"].items() for m, s in d.items()]
    f = pd.DataFrame(rows)
    st.dataframe(f.pivot(index="model", columns="target", values="RMSE").round(3), use_container_width=True)
    st.caption(mm["forecasting"]["setup"])
    st.subheader("Incident criticality classifier (5-fold CV)")
    st.dataframe(pd.DataFrame(mm["classification"]["cv_5fold"]).T, use_container_width=True)
    st.warning(mm["classification"]["caveat"])
    st.subheader("Anomaly detection vs labelled incidents")
    st.dataframe(pd.DataFrame({k: v for k, v in mm["anomaly_detection"].items() if isinstance(v, dict)}).T, use_container_width=True)
    st.caption(mm["anomaly_detection"]["note"])

# ------------------------------------------------------------------------------------------ auto-advance
if play:
    time.sleep(3)
    nxt = pd.Timestamp(now) + timedelta(minutes=15)
    st.session_state.now = (start if nxt > end else nxt).to_pydatetime()
    st.rerun()
