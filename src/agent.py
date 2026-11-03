"""Task 3 (part 3) - AI incident-analysis assistant built with LangGraph.

Graph:   route ──► gather_evidence ──► retrieve ──► generate ──► END
                └─────────────────────► retrieve (general questions skip live evidence)

* route            : decides if the question is about the live system state or a general knowledge question
* gather_evidence  : calls Telemetry for the KPI snapshot, anomaly flags, incident risk, and log mix at the chosen time
* retrieve         : RAG - fetch runbooks, similar past incidents and log patterns from the knowledge base
* generate         : writes the root-cause analysis. Uses an LLM through LangChain when an API key is configured,
                     otherwise a deterministic template, so the system also works offline.

Environment variables (all optional):
  ANTHROPIC_API_KEY (+ pip install langchain-anthropic)  or  OPENAI_API_KEY (+ pip install langchain-openai)
  LLM_MODEL  - model name for the chosen provider
"""
import os
import re
from typing import Optional, TypedDict

from langgraph.graph import END, StateGraph

from rag import KnowledgeBase
from telemetry import METRIC_QUERY, Telemetry

# Words that mean "tell me about the system at this moment". Generic words like "cause" or "issue" are NOT here,
# otherwise "What causes pool exhaustion?" would be treated as a live-system question.
LIVE_WORDS = ("now", "current", "currently", "happening", "wrong", "degraded", "degradation", "status", "rca", "outage",
              "alert", "anomal", "today", "latest", "this hour", "at this time", "going on", "health", "why is", "why are")


class AgentState(TypedDict, total=False):
    question: str
    as_of: Optional[str]
    intent: str                # "live" | "general"
    snapshot: dict
    log_insights: dict
    retrieved: dict            # {"runbook": [...], "incident": [...], "log_pattern": [...], "pattern": [...]}
    answer: str
    sources: list
    engine: str


# ----------------------------------------------------------------------------------------- LLM (optional)
def get_llm():
    try:
        if os.getenv("ANTHROPIC_API_KEY"):
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(model=os.getenv("LLM_MODEL", "claude-sonnet-5-5"), temperature=0, max_tokens=900)
        if os.getenv("OPENAI_API_KEY"):
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model=os.getenv("LLM_MODEL", "gpt-4o-mini"), temperature=0)
    except Exception as e:  # missing package etc. -> fall back to template
        print("LLM unavailable, using template generator:", e)
    return None


# ----------------------------------------------------------------------------------------- helpers
def _steps(text: str, header: str) -> list[str]:
    """Pull the numbered/bulleted lines that follow a header like 'Resolution Steps:' out of a runbook."""
    m = re.search(header + r".*?\n((?:\s*[-\d].*\n?)+)", text, re.S | re.I)
    return [re.sub(r"^\s*(?:[-]|\d+\.)\s*", "", l).strip() for l in m.group(1).splitlines() if l.strip()] if m else []


def build_graph(tel: Telemetry, kb: KnowledgeBase):
    llm = get_llm()

    def route(state: AgentState) -> AgentState:
        q = state["question"].lower()
        return {"intent": "live" if any(w in q for w in LIVE_WORDS) else "general"}

    def gather_evidence(state: AgentState) -> AgentState:
        snap = tel.snapshot(state.get("as_of"))
        logs = tel.log_insights(state.get("as_of"), 60)
        return {"snapshot": snap, "log_insights": logs}

    def retrieve(state: AgentState) -> AgentState:
        q = state["question"]
        snap = state.get("snapshot")
        if snap:  # enrich the search with what the metrics say is wrong
            q += " " + " ".join(METRIC_QUERY.get(m, "") for m in snap["anomalous_metrics"])
        out = {
            "runbook": kb.search(q, min(4, max(2, len(snap["anomalous_metrics"]) - 1)) if snap else 2, ["runbook"]),
            "incident": kb.search(q, 3, ["incident"]),
            "log_pattern": kb.search(q, 2, ["log_pattern"]),
            "pattern": kb.search(q, 1, ["pattern"]),
        }
        return {"retrieved": out}

    def generate(state: AgentState) -> AgentState:
        r, snap, logs = state["retrieved"], state.get("snapshot"), state.get("log_insights")
        sources = [{"id": d["id"], "type": d["type"], "title": d["title"], "score": d["score"]} for v in r.values() for d in v]

        # ---------- LLM path (RAG prompt grounded in evidence + retrieved documents)
        if llm is not None:
            ctx = "\n\n".join(f"[{d['id']}] {d['title']}\n{d['text']}" for v in r.values() for d in v)
            live = ""
            if snap:
                devs = "; ".join(f"{d['metric']}={d['value']:.1f} (expected {d['expected']:.1f}, z={d['z']})" for d in snap["deviations"][:6])
                live = (f"LIVE EVIDENCE at {snap['hour']}: incident risk {snap['incident_risk']['critical_probability']:.0%} "
                        f"({snap['incident_risk']['risk_level']}); anomalous metrics: {snap['anomalous_metrics']}; deviations: {devs}; "
                        f"logs last hour: {logs['groups']}")
            prompt = ("You are an SRE assistant for a digital bank. Answer ONLY from the evidence and documents below; cite document ids "
                      "in square brackets. Structure: Summary, Likely root causes, Evidence, Recommended actions. If evidence is weak, say so.\n\n"
                      f"QUESTION: {state['question']}\n\n{live}\n\nDOCUMENTS:\n{ctx}")
            try:
                return {"answer": llm.invoke(prompt).content, "sources": sources, "engine": "llm"}
            except Exception as e:
                print("LLM call failed, falling back to template:", e)

        # ---------- Template path (deterministic, offline)
        lines = []
        if snap:
            risk = snap["incident_risk"]
            lines.append(f"### Summary\nAt **{snap['hour']}** the incident model rates the risk as **{risk['risk_level']}** "
                         f"({risk['critical_probability']:.0%} probability of a critical incident); "
                         f"the Isolation Forest {'also flags' if snap['isolation_forest_anomaly'] else 'does not flag'} this hour as anomalous.")
            bad = [d for d in snap["deviations"] if d["anomalous"]]
            if bad:
                lines.append("### Evidence\n" + "\n".join(
                    f"- **{d['metric']}** = {d['value']:.1f} vs ~{d['expected']:.1f} expected ({'+' if d['z'] > 0 else ''}{d['z']:.1f}σ)" for d in bad))
            else:
                lines.append("### Evidence\n- All monitored metrics are within their normal range for this time of day.")
            g = logs["groups"]
            unusual = [k for k, v in g.items() if k != "Informational / Normal" and v["count"] > 1.5 * v["typical_per_window"]]
            lines[-1] += "\n" + ("- Logs (last 60 min): " + ", ".join(f"{k}: {v['count']} (typical {v['typical_per_window']})" for k, v in g.items())
                         + (f". Elevated: {', '.join(unusual)}." if unusual else ". No unusual log surge, so the metrics (not the logs) carry the signal."))
        else:
            lines.append("### Summary\nNo live metrics were requested; answering from the knowledge base.")

        if snap and snap["incident_risk"]["risk_level"] == "Normal" and not snap["anomalous_metrics"]:
            lines.append("### Recommended actions\n- No action needed. Continue monitoring; check the forecast panel for upcoming load.")
        else:
            lines.append("### Likely root causes (from runbooks)")
            for d in r["runbook"]:
                causes = _steps(d["text"], r"(?:causes|indicate|occurs|errors occur|can be caused)[^\n]*:")[:4] or _steps(d["text"], r"\n")[:4]
                lines.append(f"**{d['title']}** [{d['id']}]\n" + "\n".join(f"- {c}" for c in causes))
            lines.append("### Recommended actions")
            for d in r["runbook"]:
                steps = _steps(d["text"], "Resolution Steps:")[:4]
                lines.append(f"**{d['title']}** [{d['id']}]\n" + "\n".join(f"- {s}" for s in steps))
        if r["incident"] and not (snap and snap["incident_risk"]["risk_level"] == "Normal" and not snap["anomalous_metrics"]):
            lines.append("### Similar past incidents\n" + "\n".join(f"- [{d['id']}] {d['title']}: {d['text'][:170]}…" for d in r["incident"][:2]))
        if r["pattern"]:
            lines.append(f"### Context\n- [{r['pattern'][0]['id']}] {r['pattern'][0]['text']}")
        return {"answer": "\n\n".join(lines), "sources": sources, "engine": "template"}

    g = StateGraph(AgentState)
    g.add_node("route", route)
    g.add_node("gather_evidence", gather_evidence)
    g.add_node("retrieve", retrieve)
    g.add_node("generate", generate)
    g.set_entry_point("route")
    g.add_conditional_edges("route", lambda s: s["intent"], {"live": "gather_evidence", "general": "retrieve"})
    g.add_edge("gather_evidence", "retrieve")
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", END)
    return g.compile()


class IncidentAssistant:
    def __init__(self, tel: Telemetry | None = None, kb: KnowledgeBase | None = None):
        self.tel = tel or Telemetry()
        self.kb = kb or KnowledgeBase.load()
        self.graph = build_graph(self.tel, self.kb)
        self.llm_enabled = get_llm() is not None

    def ask(self, question: str, as_of: str | None = None) -> dict:
        out = self.graph.invoke({"question": question, "as_of": as_of})
        return {"question": question, "as_of": as_of, "intent": out["intent"], "answer": out["answer"],
                "sources": out["sources"], "engine": out["engine"], "anomalous_metrics": out.get("snapshot", {}).get("anomalous_metrics", [])}


if __name__ == "__main__":
    a = IncidentAssistant()
    for q, t in [("Why is the system degraded right now? Give root cause and actions", "2025-10-20 11:30"),
                 ("What causes database connection pool exhaustion?", None),
                 ("What is the status right now?", "2025-10-21 10:30")]:
        r = a.ask(q, t)
        print("=" * 90, f"\nQ: {q} (as_of={t}) intent={r['intent']} engine={r['engine']}\n", r["answer"][:1800])
