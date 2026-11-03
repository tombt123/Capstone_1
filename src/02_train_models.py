"""Task 2 - Machine learning and predictive modelling.

  A. Time-series forecasting (success rate, latency, CPU, memory, DB pool)
  B. Incident-criticality classification
  C. Anomaly detection
Run:  python src/02_train_models.py
Saves trained models to models/ and scores to outputs/model_metrics.json
"""
import json
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, mean_absolute_error, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from common import FIGS, MODELS, OUT, load_incidents, load_metrics
from features import FORECAST_TARGETS, INCIDENT_FEATURES, calendar_features

warnings.filterwarnings("ignore")
RNG = 42
results = {}


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


# =============================================================================================
# A. FORECASTING
# =============================================================================================
# The raw data is 1 row per minute. We average to 15-minute buckets: still detailed, but much less noise.
full = load_metrics()
train_raw, test_raw = load_metrics("infrastructure_metrics_train.csv"), load_metrics("infrastructure_metrics_test.csv")
for d in (train_raw, test_raw):
    d["overall_success_rate"] = d[["neft_success_rate", "imps_success_rate", "rtgs_success_rate"]].mean(axis=1)
r15 = lambda d: d.resample("15min").mean(numeric_only=True)
train15, test15, full15 = r15(train_raw), r15(test_raw), r15(full)
print(f"Forecast data: train={len(train15)} test={len(test15)} points (15-min)")

SEASON = 96  # 96 x 15 min = 1 day
forecast_scores, forecast_preds, deployed = {}, {}, {}

for target in FORECAST_TARGETS:
    y_tr, y_te = train15[target], test15[target]
    preds = {}

    # 1) Naive: tomorrow looks like the last value we saw
    preds["naive_last_value"] = np.repeat(y_tr.iloc[-1], len(y_te))

    # 2) Seasonal naive: same time-of-day yesterday (copy the last 96 points forward)
    last_day = y_tr.iloc[-SEASON:].to_numpy()
    preds["seasonal_naive"] = np.resize(last_day, len(y_te))

    # 3) Holt-Winters exponential smoothing with daily seasonality
    hw = ExponentialSmoothing(y_tr, trend=None, seasonal="add", seasonal_periods=SEASON,
                              initialization_method="estimated").fit()
    preds["holt_winters"] = hw.forecast(len(y_te)).to_numpy()

    # 4) Gradient boosting on calendar features (hour, weekday/weekend, business hours)
    gbr = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, min_samples_leaf=20, random_state=RNG)
    gbr.fit(calendar_features(y_tr.index), y_tr)
    preds["gradient_boosting_calendar"] = gbr.predict(calendar_features(y_te.index))

    forecast_scores[target] = {m: {"RMSE": round(rmse(y_te, p), 4), "MAE": round(float(mean_absolute_error(y_te, p)), 4)}
                               for m, p in preds.items()}
    forecast_preds[target] = preds

    # --- Deployed model: refit GBM on ALL data (train+test) so forecasts start from the latest data
    final = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, min_samples_leaf=20, random_state=RNG)
    final.fit(calendar_features(full15.index), full15[target])
    # Typical forecast error must come from HELD-OUT data (in-sample error is unrealistically small)
    resid_std = float(forecast_scores[target]["gradient_boosting_calendar"]["RMSE"])
    # Single-minute observations are noisier than 15-min averages -> separate spread for point-in-time anomaly checks
    raw_resid_std = float(np.std(test_raw[target] - gbr.predict(calendar_features(test_raw.index))))
    deployed[target] = {"model": final, "resid_std": resid_std, "raw_resid_std": raw_resid_std, "target": target}
    joblib.dump(deployed[target], MODELS / f"forecast_{target}.joblib")

results["forecasting"] = {
    "setup": "15-minute averages; chronological split from provided train/test files "
             f"(train {train15.index.min()} -> {train15.index.max()}, test {test15.index.min()} -> {test15.index.max()})",
    "scores": forecast_scores,
}

# plot test forecasts
fig, axes = plt.subplots(len(FORECAST_TARGETS), 1, figsize=(13, 14), sharex=True)
for ax, target in zip(axes, FORECAST_TARGETS):
    ax.plot(test15.index, test15[target], color="black", lw=1.2, label="actual")
    for m, p in forecast_preds[target].items():
        ax.plot(test15.index, p, lw=1, label=m, alpha=0.8)
    ax.set_ylabel(target, fontsize=8)
axes[0].legend(ncol=5, fontsize=8)
fig.suptitle("Forecast vs actual on the held-out test period")
plt.tight_layout(); plt.savefig(FIGS / "forecast_test_comparison.png", dpi=120); plt.close()

# =============================================================================================
# B. INCIDENT CRITICALITY CLASSIFICATION
# =============================================================================================
inc = load_incidents().sort_values("timestamp").reset_index(drop=True)
X, y = inc[INCIDENT_FEATURES], inc["is_critical"]
# NOTE: 'severity' and 'incident_type' are NOT used as inputs - they would leak the answer.

models = {
    "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")),
    "random_forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=RNG),
    "gradient_boosting": GradientBoostingClassifier(random_state=RNG),
}
cv = StratifiedKFold(5, shuffle=True, random_state=RNG)
clf_scores = {}
for name, m in models.items():
    cvres = cross_validate(m, X, y, cv=cv, scoring=["accuracy", "f1", "precision", "recall", "roc_auc"])
    clf_scores[name] = {k.replace("test_", ""): round(float(v.mean()), 4) for k, v in cvres.items() if k.startswith("test_")}

# Chronological hold-out as a second check: train on first 70% of hours, test on the last 30%
cut = int(len(inc) * 0.7)
holdout = {}
for name, m in models.items():
    m.fit(X.iloc[:cut], y.iloc[:cut])
    p = m.predict(X.iloc[cut:])
    holdout[name] = {"accuracy": round(accuracy_score(y.iloc[cut:], p), 4), "f1": round(f1_score(y.iloc[cut:], p, zero_division=0), 4),
                     "n_test": int(len(p)), "n_critical_in_test": int(y.iloc[cut:].sum())}

best_name = max(clf_scores, key=lambda k: (clf_scores[k]["f1"], clf_scores[k]["roc_auc"]))
best = models[best_name].fit(X, y)  # final fit on all labelled hours
joblib.dump({"model": best, "features": INCIDENT_FEATURES, "name": best_name}, MODELS / "incident_classifier.joblib")
if hasattr(best, "feature_importances_"):
    imp = dict(zip(INCIDENT_FEATURES, best.feature_importances_.round(3)))
else:
    imp = dict(zip(INCIDENT_FEATURES, np.abs(best[-1].coef_[0]).round(3)))
results["classification"] = {"target": "is_critical", "cv_5fold": clf_scores, "chronological_holdout": holdout,
                             "deployed_model": best_name, "feature_importance": dict(sorted(imp.items(), key=lambda kv: -kv[1])),
                             "caveat": "Scores are near-perfect because the labels follow clear thresholds "
                                       "(critical hours have CPU>85, memory>80, DB pool>85, error rate>=10, NEFT<90). "
                                       "Expect lower scores on real, messier incident labels."}
print("Classifier CV:", json.dumps(clf_scores, indent=1))

# =============================================================================================
# C. ANOMALY DETECTION
# =============================================================================================
# Model 1 - Isolation Forest: trained on 'normal' minute-level operation, flags unusual combinations of KPIs.
Xn = full[INCIDENT_FEATURES]
iso = make_pipeline(StandardScaler(), IsolationForest(n_estimators=300, contamination=0.005, random_state=RNG))
iso.fit(Xn)
joblib.dump({"model": iso, "features": INCIDENT_FEATURES}, MODELS / "anomaly_isolation_forest.joblib")

pred_iso = (iso.predict(X) == -1).astype(int)
score_iso = -iso.decision_function(X)

# Model 2 - forecast-residual z-score: how far is the metric from what we expect at this time of day?
resid_targets = ["cpu_utilization", "memory_usage", "db_connection_pool_usage", "avg_transaction_processing_time_ms"]
zs = pd.DataFrame(index=inc.index)
for t in resid_targets:
    d = deployed[t]
    expected = d["model"].predict(calendar_features(pd.DatetimeIndex(inc["timestamp"])))
    zs[t] = (inc[t].to_numpy() - expected) / d["raw_resid_std"]
# also neft success rate & error rate vs their overall mean/std (they have no daily pattern)
for t in ["neft_success_rate", "application_error_rate"]:
    zs[t] = (inc[t] - full[t].mean()) / full[t].std()
inc_z = zs.abs().max(axis=1)
pred_z = (inc_z > 4).astype(int)


def det_scores(pred):
    return {"precision": round(precision_score(y, pred, zero_division=0), 3), "recall": round(recall_score(y, pred), 3),
            "f1": round(f1_score(y, pred), 3), "accuracy": round(accuracy_score(y, pred), 3), "flagged": int(pred.sum())}


results["anomaly_detection"] = {
    "isolation_forest": {**det_scores(pred_iso), "roc_auc_vs_critical_label": round(roc_auc_score(y, score_iso), 4)},
    "residual_zscore_gt4": {**det_scores(pred_z), "roc_auc_vs_critical_label": round(roc_auc_score(y, inc_z), 4)},
    "note": "Evaluated on the 168 hourly rows of incident_reports.csv; 26 are labelled critical. The model is trained on "
            "minute-level normal operation only, so critical hours are unseen.",
}
joblib.dump({"threshold": 4.0, "resid_targets": resid_targets}, MODELS / "anomaly_zscore_config.joblib")

# =============================================================================================
with open(OUT / "model_metrics.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n=== FORECAST (test RMSE) ===")
print(pd.DataFrame({t: {m: v["RMSE"] for m, v in s.items()} for t, s in forecast_scores.items()}).round(3).to_string())
print("\n=== CLASSIFIER ===", best_name, clf_scores[best_name], "\nholdout", holdout)
print("\n=== ANOMALY ===", json.dumps(results["anomaly_detection"], indent=1))
