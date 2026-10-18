import time, joblib, numpy as np
from common import synth, featurize
art = joblib.load("models/model.joblib"); m = art["model"]
X = featurize(synth(n=6000, seed=99))
for i in range(100): m.predict_proba(X.iloc[[i]])          # warm-up
lat = []
for i in range(5000):
    t = time.perf_counter(); m.predict_proba(X.iloc[[i]]); lat.append((time.perf_counter() - t) * 1000)
lat = np.array(lat)
print("single-transaction scoring  p50=%.2f ms  p99=%.2f ms  throughput=%.0f tx/s" % (np.percentile(lat, 50), np.percentile(lat, 99), 1000 / lat.mean()))
