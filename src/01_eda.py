"""Task 1 - Data preparation and exploration.

Run:  python src/01_eda.py
Output: figures in outputs/figures/ and a findings file outputs/eda_findings.json
"""
import json

import matplotlib
matplotlib.use("Agg")  # draw to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from common import (FIGS, INFRA_COLS, OUT, SUCCESS_COLS, load_incidents, load_logs,
                    load_metrics, load_transactions)

sns.set_theme(style="whitegrid")
findings = {}


def save(name):
    plt.tight_layout()
    plt.savefig(FIGS / f"{name}.png", dpi=130)
    plt.close()


# ---------------------------------------------------------------- 1. load + clean
metrics, txns, logs, incidents = load_metrics(), load_transactions(), load_logs(), load_incidents()
findings["rows"] = {"metrics": len(metrics), "transactions": len(txns), "logs": len(logs), "incidents": len(incidents)}
findings["missing_values_after_cleaning"] = int(metrics.isna().sum().sum() + txns.isna().sum().sum() + logs.isna().sum().sum())
print("Rows:", findings["rows"])

metrics["hour"] = metrics.index.hour
metrics["is_business_hours"] = metrics["hour"].between(9, 17)
metrics["is_weekend"] = metrics.index.dayofweek >= 5

# ---------------------------------------------------------------- 2. business KPI distributions
fig, ax = plt.subplots(1, 4, figsize=(18, 3.8))
for a, c in zip(ax, SUCCESS_COLS):
    sns.histplot(metrics[c], bins=40, ax=a, kde=True)
    a.set_title(c)
fig.suptitle("Success-rate distributions (minute level)")
save("kpi_success_rate_distributions")

fig, ax = plt.subplots(1, 2, figsize=(13, 4))
sns.histplot(metrics["avg_transaction_processing_time_ms"], bins=50, ax=ax[0])
ax[0].set_title("Avg processing time (ms) - infra file (bimodal: calm vs busy)")
sns.boxplot(data=txns, x="transaction_type", y="processing_time_ms", ax=ax[1])
ax[1].set_title("Processing time by transaction type (transactions file)")
ax[1].tick_params(axis="x", rotation=30)
save("kpi_processing_time")

findings["success_rate_summary"] = metrics[SUCCESS_COLS].describe().round(2).to_dict()
findings["processing_time_by_txn_type_ms"] = txns.groupby("transaction_type", observed=True)["processing_time_ms"].mean().round(1).to_dict()
findings["fraud_rate_by_txn_type_pct"] = (txns.groupby("transaction_type", observed=True)["is_fraud"].mean() * 100).round(2).to_dict()
findings["overall_fraud_rate_pct"] = round(txns["is_fraud"].mean() * 100, 2)

# ---------------------------------------------------------------- 3. infrastructure metrics
# NOTE: incident_reports.csv is an HOURLY table. Its normal rows equal the minute file's value at the top of
# the hour, but its 26 critical hours contain spike values (e.g. CPU ~90%) that do NOT appear in the minute
# file (minute-file CPU never exceeds ~78%). So incidents are analysed from the incident table itself.
findings["data_note"] = ("incident_reports.csv hourly rows match the minute file except during incident hours, "
                         "where CPU/memory/DB-pool/error-rate spikes exist only in the incident file.")
inc = incidents.set_index("timestamp")
hourly = metrics.resample("1h").mean(numeric_only=True)

fig, ax = plt.subplots(5, 1, figsize=(14, 12), sharex=True)
for a, c in zip(ax, INFRA_COLS + ["application_error_rate"]):
    a.plot(inc.index, inc[c], lw=1) if c in inc else a.plot(hourly.index, hourly[c], lw=1)
    crit = inc["is_critical"] == 1
    if c in inc:
        a.scatter(inc.index[crit], inc.loc[crit, c], color="red", s=16, label="critical incident hour", zorder=3)
    a.set_ylabel(c, fontsize=8)
ax[0].legend(loc="upper right")
fig.suptitle("Hourly infrastructure metrics (incident_reports.csv) - critical incident hours in red")
save("infra_timeseries")

hr = metrics.groupby("hour")[["cpu_utilization", "memory_usage", "avg_transaction_processing_time_ms", "transaction_volume_per_hour"]].mean()
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
hr[["cpu_utilization", "memory_usage"]].plot(ax=ax[0], marker="o")
ax[0].set_title("Average CPU / memory by hour of day")
hr["transaction_volume_per_hour"].plot(ax=ax[1], marker="o", color="tab:green")
ax[1].set_title("Average transaction volume by hour of day")
save("daily_pattern")

findings["business_hours_vs_off_hours"] = metrics.groupby("is_business_hours")[
    ["cpu_utilization", "memory_usage", "avg_transaction_processing_time_ms", "transaction_volume_per_hour"]].mean().round(1).to_dict()
findings["weekday_vs_weekend_cpu"] = metrics.groupby("is_weekend")["cpu_utilization"].mean().round(1).to_dict()

# ---------------------------------------------------------------- 4. correlation: transaction performance vs infra load
corr_cols = INFRA_COLS + ["application_error_rate", "transaction_volume_per_hour", "avg_transaction_processing_time_ms",
                          "neft_success_rate", "imps_success_rate", "rtgs_success_rate", "auth_success_rate"]
corr = metrics[corr_cols].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, annot_kws={"size": 7})
plt.title("Correlation: infrastructure load vs transaction KPIs (minute level)")
save("correlation_heatmap")

# Hourly view is more informative: incidents are labelled per hour
hcorr = hourly[corr_cols].corr()
findings["strongest_hourly_correlations_with_processing_time"] = (
    hcorr["avg_transaction_processing_time_ms"].drop("avg_transaction_processing_time_ms").sort_values(key=abs, ascending=False).head(5).round(2).to_dict())
findings["strongest_hourly_correlations_with_neft_success"] = (
    hcorr["neft_success_rate"].drop("neft_success_rate").sort_values(key=abs, ascending=False).head(5).round(2).to_dict())

# Individual card/UPI/NEFT transactions overlap the infra window only for a few days -> join by hour
t = txns[(txns.timestamp >= metrics.index.min()) & (txns.timestamp <= metrics.index.max())].copy()
t["hour_ts"] = t.timestamp.dt.floor("h")
joined = t.merge(hourly[INFRA_COLS + ["transaction_volume_per_hour"]], left_on="hour_ts", right_index=True, how="left")
findings["txns_overlapping_infra_window"] = len(joined)
txn_corr = joined[["processing_time_ms", "is_fraud"] + INFRA_COLS].corr(numeric_only=True)
findings["txn_processing_time_vs_infra_corr"] = txn_corr.loc["processing_time_ms", INFRA_COLS].round(3).to_dict()

# Under incident vs normal hours
by_state = incidents.groupby("is_critical")[["avg_transaction_processing_time_ms", "neft_success_rate", "application_error_rate",
                                             "cpu_utilization", "memory_usage", "db_connection_pool_usage"]].mean().round(2)
findings["normal_vs_critical_means"] = by_state.to_dict()
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for a, c in zip(ax, ["neft_success_rate", "application_error_rate", "cpu_utilization"]):
    sns.boxplot(data=incidents, x="is_critical", y=c, ax=a)
    a.set_xticks([0, 1]); a.set_xticklabels(["Normal", "Critical incident"])
    a.set_title(c)
save("normal_vs_critical")

# ---------------------------------------------------------------- 5. bottlenecks / anomalies (simple z-score on hourly data)
zcols = INFRA_COLS + ["application_error_rate", "neft_success_rate"]
z = (inc[zcols] - inc[zcols].median()) / inc[zcols].std()
inc["max_abs_z"] = z.abs().max(axis=1)
findings["hours_with_abs_z_gt_3"] = int((inc["max_abs_z"] > 3).sum())
findings["critical_hours_total"] = int(incidents.is_critical.sum())
findings["critical_hours_caught_by_z3"] = int(((inc["max_abs_z"] > 3) & (inc["is_critical"] == 1)).sum())
plt.figure(figsize=(14, 3.5))
plt.plot(inc.index, inc["max_abs_z"], lw=1)
plt.scatter(inc.index[inc.is_critical == 1], inc.loc[inc.is_critical == 1, "max_abs_z"], color="red", s=14, label="critical", zorder=3)
plt.axhline(3, color="red", ls="--", label="|z| = 3")
plt.legend(); plt.title("Anomaly score per hour (largest |z-score| across infra metrics)")
save("anomaly_zscore")

# Incident mix and logs
fig, ax = plt.subplots(1, 3, figsize=(16, 4))
incidents["incident_type"].value_counts().plot.bar(ax=ax[0], title="Incident types")
logs["log_level"].value_counts().plot.bar(ax=ax[1], title="Log levels")
logs[logs.log_level != "INFO"].set_index("timestamp").resample("6h").size().plot(ax=ax[2], title="WARNING+ERROR logs per 6h")
save("incidents_and_logs")
findings["log_level_counts"] = logs["log_level"].value_counts().to_dict()
findings["critical_incident_hours_by_hour_of_day"] = incidents[incidents.is_critical == 1].timestamp.dt.hour.value_counts().sort_index().to_dict()

# Fraud context
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
sns.histplot(data=txns, x="amount", hue="is_fraud", bins=50, log_scale=True, ax=ax[0], common_norm=False, stat="density")
ax[0].set_title("Transaction amount (log scale) by fraud flag")
txns.groupby("location", observed=True)["is_fraud"].mean().sort_values().plot.barh(ax=ax[1], title="Fraud rate by location")
save("fraud_overview")

with open(OUT / "eda_findings.json", "w") as f:
    json.dump(findings, f, indent=2, default=str)
print(json.dumps({k: findings[k] for k in ["txns_overlapping_infra_window", "txn_processing_time_vs_infra_corr",
                                            "strongest_hourly_correlations_with_neft_success",
                                            "strongest_hourly_correlations_with_processing_time",
                                            "hours_with_abs_z_gt_3", "critical_hours_total", "critical_hours_caught_by_z3", "weekday_vs_weekend_cpu"]}, indent=2, default=str))
print("Figures saved to", FIGS)
