"""Log text preprocessing + cluster naming rules (shared by training script and API)."""
import re

# Words that tell us a cluster is about infrastructure vs. a business KPI (payments / customers / API)
INFRA_WORDS = {"cpu", "memory", "disk", "network", "container", "oom", "database", "connection", "pool", "latency",
               "srv", "io", "bottleneck", "restarting", "timeout", "utilization", "threshold", "cache"}
BUSINESS_WORDS = {"neft", "imps", "rtgs", "transaction", "payment", "gateway", "authentication", "login", "customer",
                  "rate", "limit", "api", "duplicate", "sla", "beneficiary", "funds", "account"}


def normalize_log(msg: str) -> str:
    """'ERROR: NEFT transaction failed. Transaction ID: TXN399. Reason: Gateway timeout'
        -> 'neft transaction failed transaction id txnid reason gateway timeout'
    Steps: drop the level prefix, replace IDs / numbers by placeholders, lowercase, strip punctuation.
    (Numbers are removed on purpose - every log line has different ones, they would hide the shared pattern.)"""
    m = re.sub(r"^(INFO|WARNING|ERROR|CRITICAL):\s*", "", str(msg), flags=re.I)
    m = re.sub(r"\b(TXN|USR)\d+\b", r"\1id", m)
    m = re.sub(r"\bsrv-\d+\b", "srv", m)
    m = re.sub(r"\d+(\.\d+)?\s*(ms|%)?", " ", m)
    m = re.sub(r"[^a-zA-Z/ ]+", " ", m)
    return re.sub(r"\s+", " ", m).strip().lower()


def group_for_terms(terms, info_share: float) -> str:
    """Decide the cluster family from its most important words + share of INFO-level lines."""
    if info_share >= 0.8:
        return "Informational / Normal"
    t = {w for term in terms for w in term.split()}
    infra, biz = len(t & INFRA_WORDS), len(t & BUSINESS_WORDS)
    return "Infrastructure issue" if infra > biz else "Business KPI issue"
