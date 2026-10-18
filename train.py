import json, os, time, joblib, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve
from common import synth, featurize

CSV = "data/paysim.csv"   # optional: real PaySim file renamed to this
if os.path.exists(CSV):
    df = pd.read_csv(CSV); print("Using PaySim:", len(df))
else:
    df = synth(); print("PaySim not found -> using synthetic data:", len(df))

df = df.sort_values("step").reset_index(drop=True)          # time-based split
n = len(df); a, b = int(n * .70), int(n * .85)
tr, va, te = df[:a], df[a:b], df[b:]
Xtr, Xva, Xte = featurize(tr), featurize(va), featurize(te)

w = np.where(tr.isFraud == 1, (tr.isFraud == 0).sum() / max((tr.isFraud == 1).sum(), 1) ** 1.0 * 0.2, 1.0)
model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=0)
t = time.time(); model.fit(Xtr, tr.isFraud, sample_weight=w); print("trained in %.1fs" % (time.time() - t))

p_va = model.predict_proba(Xva)[:, 1]
pr, rc, th = precision_recall_curve(va.isFraud, p_va)
f1 = 2 * pr * rc / (pr + rc + 1e-9)
block = float(th[np.argmax(f1[:-1])])
verify = max(0.05, block * 0.4)

p_te = model.predict_proba(Xte)[:, 1]
pred = p_te >= block
tp = int(((pred == 1) & (te.isFraud == 1)).sum())
metrics = {"pr_auc_test": float(average_precision_score(te.isFraud, p_te)),
           "precision_at_block": tp / max(int(pred.sum()), 1),
           "recall_at_block": tp / max(int((te.isFraud == 1).sum()), 1),
           "block_threshold": block, "verify_threshold": verify,
           "test_fraud_rate": float(te.isFraud.mean())}
print(json.dumps(metrics, indent=2))
joblib.dump({"model": model, "block": block, "verify": verify}, "models/model.joblib")
json.dump(metrics, open("models/metrics.json", "w"), indent=2)
