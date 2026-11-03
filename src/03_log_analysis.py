"""Task 3 (part 1) - NLP on application logs: TF-IDF + unsupervised clustering.

Run:  python src/03_log_analysis.py
Outputs: models/log_clusterer.joblib, outputs/logs_clustered.csv, outputs/log_cluster_summary.json
"""
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

from common import FIGS, MODELS, OUT, load_logs
from log_nlp import group_for_terms, normalize_log

logs = load_logs()
logs["clean"] = logs["log_message"].map(normalize_log)
print(f"{len(logs)} logs, {logs['clean'].nunique()} distinct patterns after removing IDs/numbers")

# 1) TF-IDF: turn each message into a vector of weighted word / word-pair counts
vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2)
X = vec.fit_transform(logs["clean"])

# 2) Pick k (number of clusters) with the silhouette score
scores = {}
for k in range(3, min(25, logs["clean"].nunique() + 1)):  # cannot have more clusters than distinct patterns
    labels = KMeans(k, n_init=10, random_state=42).fit_predict(X)
    scores[k] = silhouette_score(X, labels)
best_k = max(scores, key=scores.get)
print("silhouette by k:", {k: round(v, 3) for k, v in scores.items()}, "-> best k =", best_k)
plt.figure(figsize=(7, 3.5)); plt.plot(list(scores), list(scores.values()), marker="o"); plt.axvline(best_k, color="r", ls="--")
plt.xlabel("k (clusters)"); plt.ylabel("silhouette"); plt.title("Choosing the number of log clusters")
plt.tight_layout(); plt.savefig(FIGS / "log_cluster_silhouette.png", dpi=120); plt.close()

# 3) Final clustering
km = KMeans(best_k, n_init=20, random_state=42).fit(X)
logs["cluster"] = km.labels_

# 4) Name each cluster from its centroid's top terms + assign issue family
terms = np.array(vec.get_feature_names_out())
summary = {}
for c in range(best_k):
    sub = logs[logs.cluster == c]
    top = terms[np.argsort(km.cluster_centers_[c])[::-1][:6]].tolist()
    info_share = float((sub.log_level == "INFO").mean())
    group = group_for_terms(top, info_share)
    example = sub["clean"].value_counts().index[0]
    summary[c] = {"cluster": c, "size": int(len(sub)), "top_terms": top, "group": group,
                  "dominant_level": sub.log_level.mode().iat[0], "example_pattern": example,
                  "example_message": sub.loc[sub.clean == example, "log_message"].iat[0]}
# readable names: the cluster's most common cleaned message, shortened
for c, s in summary.items():
    s["name"] = " ".join(s["example_pattern"].split()[:7])

logs["cluster_name"] = logs.cluster.map(lambda c: summary[c]["name"])
logs["issue_group"] = logs.cluster.map(lambda c: summary[c]["group"])

# Purity check: do clusters correspond to a single message pattern?
purity = logs.groupby("cluster")["clean"].agg(lambda s: s.value_counts(normalize=True).iloc[0]).mean()
print(f"Average cluster purity (share of the cluster's dominant message pattern): {purity:.3f}")
print(logs.groupby("issue_group").size())
for s in sorted(summary.values(), key=lambda s: (s["group"], -s["size"])):
    print(f"  [{s['group']:<22}] #{s['cluster']:<2} n={s['size']:<4} {s['name']}")

joblib.dump({"vectorizer": vec, "kmeans": km, "summary": summary}, MODELS / "log_clusterer.joblib")
logs[["timestamp", "log_level", "log_message", "cluster", "cluster_name", "issue_group"]].to_csv(OUT / "logs_clustered.csv", index=False)
with open(OUT / "log_cluster_summary.json", "w") as f:
    json.dump({"best_k": best_k, "silhouette": round(scores[best_k], 4), "purity": round(float(purity), 4),
               "group_counts": logs.issue_group.value_counts().to_dict(), "clusters": list(summary.values())}, f, indent=2)

fig, ax = plt.subplots(1, 2, figsize=(15, 5))
order = pd.DataFrame(summary.values()).sort_values(["group", "size"])
colors = {"Infrastructure issue": "tab:red", "Business KPI issue": "tab:blue", "Informational / Normal": "tab:gray"}
ax[0].barh(order["name"], order["size"], color=order["group"].map(colors)); ax[0].set_title("Log clusters (colour = issue family)")
ax[0].tick_params(axis="y", labelsize=7)
logs.set_index("timestamp").groupby("issue_group").resample("6h").size().unstack(0).plot(ax=ax[1], color=[colors.get(c) for c in sorted(logs.issue_group.unique())])
ax[1].set_title("Log volume per 6h by issue family")
plt.tight_layout(); plt.savefig(FIGS / "log_clusters.png", dpi=120); plt.close()
