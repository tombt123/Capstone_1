"""Knowledge base + retrieval (the 'R' in RAG).

The knowledge base is built ONLY from the project's own data:
  - runbook      : the 5 articles in incident_knowledge_base.txt
  - incident     : one document per abnormal hour in incident_reports.csv (29 hours)
  - pattern      : an aggregate summary of historical incidents + normal operating baselines
  - log_pattern  : one document per log cluster found in Task 3

Retrieval uses TF-IDF cosine similarity. It needs no downloads/API keys, so it works offline and in Docker.
(To upgrade: replace TfidfVectorizer by sentence-embeddings + a vector DB; the search() interface stays the same.)
"""
import json
import re
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from common import DATA, MODELS, OUT, load_incidents, load_metrics

KB_PATH = MODELS / "knowledge_base.joblib"

# thresholds observed in the labelled incident data (see model_metrics.json caveat)
FLAG_RULES = [
    ("cpu_utilization", 85, ">", "high CPU utilization"),
    ("memory_usage", 80, ">", "memory usage spike"),
    ("db_connection_pool_usage", 85, ">", "database connection pool exhaustion"),
    ("neft_success_rate", 90, "<", "NEFT transaction failures gateway timeout"),
    ("application_error_rate", 10, ">=", "high application error rate"),
]


def flags_for_row(row) -> list[str]:
    out = []
    for col, thr, op, text in FLAG_RULES:
        v = row[col]
        if (op == ">" and v > thr) or (op == "<" and v < thr) or (op == ">=" and v >= thr):
            out.append(text)
    return out


def parse_runbooks() -> list[dict]:
    raw = (DATA / "incident_knowledge_base.txt").read_text()
    docs = []
    for i, block in enumerate([b for b in re.split(r"\n=+\n", raw) if b.strip()], 1):
        m = re.search(r"TITLE:\s*(.+?)\nCONTENT:\s*(.*)", block, re.S)
        if m:
            docs.append({"id": f"RB-{i:02d}", "type": "runbook", "title": m.group(1).strip(), "text": m.group(2).strip()})
    return docs


def build_documents() -> list[dict]:
    docs = parse_runbooks()

    # ---- historical incidents
    inc = load_incidents()
    abnormal = inc[inc.incident_type != "Normal"]
    for _, r in abnormal.iterrows():
        flags = flags_for_row(r)
        text = (f"{r.incident_type.replace('_', ' ')} incident ({r.severity} severity) at {r.timestamp:%Y-%m-%d %H:%M}. "
                f"CPU {r.cpu_utilization:.1f}%, memory {r['memory_usage']:.1f}%, DB connection pool {r.db_connection_pool_usage:.1f}%, "
                f"network {r.network_throughput_mbps:.0f} Mbps, application error rate {r.application_error_rate}, "
                f"NEFT success {r.neft_success_rate:.1f}%, IMPS {r.imps_success_rate:.1f}%, RTGS {r.rtgs_success_rate:.1f}%, "
                f"avg processing time {r.avg_transaction_processing_time_ms:.0f} ms. "
                + ("Symptoms: " + "; ".join(flags) + "." if flags else "Symptoms: elevated but below critical thresholds."))
        docs.append({"id": r.incident_id, "type": "incident", "title": f"{r.incident_type} {r.timestamp:%Y-%m-%d %H:%M}", "text": text,
                     "meta": {"timestamp": str(r.timestamp), "severity": r.severity, "flags": flags}})

    # ---- aggregate pattern summary
    crit = inc[inc.is_critical == 1]
    summ = (f"Historical incident pattern: {len(crit)} critical hours out of {len(inc)} observed hours ({len(crit)/len(inc):.0%}). "
            f"Median metrics during critical hours: CPU {crit.cpu_utilization.median():.0f}%, memory {crit['memory_usage'].median():.0f}%, "
            f"DB pool {crit.db_connection_pool_usage.median():.0f}%, error rate {crit.application_error_rate.median():.0f}, "
            f"NEFT success {crit.neft_success_rate.median():.0f}%. In every critical hour CPU, memory and DB pool were all elevated "
            f"together (resource exhaustion) while IMPS and RTGS success rates stayed normal - the payment impact was concentrated on NEFT. "
            f"Critical hours occurred at all times of day, not only during business peak, so peak load is not the only trigger.")
    docs.append({"id": "PAT-01", "type": "pattern", "title": "Resource exhaustion pattern: all critical incidents", "text": summ})

    # ---- normal operating baselines (from the minute-level file)
    m = load_metrics()
    m["bh"] = m.index.hour.to_series(index=m.index).between(9, 17)
    m["we"] = m.index.dayofweek >= 5
    for (we, bh), g in m.groupby(["we", "bh"]):
        label = f"{'weekend' if we else 'weekday'} {'business hours (09-18)' if bh else 'off-hours'}"
        docs.append({"id": f"BASE-{int(we)}{int(bh)}", "type": "pattern", "title": f"Normal operating baseline: {label}",
                     "text": (f"Normal baseline on {label}: CPU {g.cpu_utilization.mean():.0f}% (max {g.cpu_utilization.max():.0f}), "
                              f"memory {g['memory_usage'].mean():.0f}%, DB pool {g.db_connection_pool_usage.mean():.0f}%, network "
                              f"{g.network_throughput_mbps.mean():.0f} Mbps, latency {g.avg_transaction_processing_time_ms.mean():.0f} ms, "
                              f"{g.transaction_volume_per_hour.mean():.0f} transactions/hour, NEFT success {g.neft_success_rate.mean():.1f}%.")})

    # ---- log patterns from Task 3
    ls = json.loads((OUT / "log_cluster_summary.json").read_text())
    total = sum(c["size"] for c in ls["clusters"])
    for c in ls["clusters"]:
        docs.append({"id": f"LOG-{c['cluster']:02d}", "type": "log_pattern", "title": f"Log pattern: {c['name']}",
                     "text": (f"Log pattern ({c['group']}), dominant level {c['dominant_level']}: \"{c['example_message']}\". "
                              f"Seen {c['size']} times in the week ({c['size']/total:.1%} of logs). Key terms: {', '.join(c['top_terms'])}."),
                     "meta": {"group": c["group"]}})
    return docs


@dataclass
class KnowledgeBase:
    docs: list
    vectorizer: TfidfVectorizer
    matrix: object

    @classmethod
    def build(cls):
        docs = build_documents()
        vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
        mat = vec.fit_transform([d["title"] + ". " + d["text"] for d in docs])
        kb = cls(docs, vec, mat)
        joblib.dump(kb, KB_PATH)
        return kb

    @classmethod
    def load(cls):
        return joblib.load(KB_PATH)

    def search(self, query: str, k: int = 3, types: list[str] | None = None) -> list[dict]:
        sims = linear_kernel(self.vectorizer.transform([query]), self.matrix).ravel()
        order = np.argsort(sims)[::-1]
        hits = []
        for i in order:
            d = self.docs[i]
            if types and d["type"] not in types:
                continue
            if sims[i] <= 0:
                break
            hits.append({**d, "score": round(float(sims[i]), 3)})
            if len(hits) == k:
                break
        return hits
