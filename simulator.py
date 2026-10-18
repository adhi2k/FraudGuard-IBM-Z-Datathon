import json, sys, time, urllib.request, random
from common import synth

rate = float(sys.argv[1]) if len(sys.argv) > 1 else 10      # transactions per second
df = synth(n=60_000, seed=7)
fr, lg = df[df.isFraud == 1], df[df.isFraud == 0]
print("streaming at %.0f tx/s ... Ctrl+C to stop" % rate)
i = 0
while True:
    row = (fr if random.random() < 0.08 else lg).sample(1).iloc[0]
    payload = {"user": "U%05d" % random.randint(1, 99999), "type": row["type"], "amount": float(row["amount"]),
               "oldbalanceOrg": float(row["oldbalanceOrg"]), "newbalanceOrig": float(row["newbalanceOrig"]),
               "oldbalanceDest": float(row["oldbalanceDest"]), "newbalanceDest": float(row["newbalanceDest"]),
               "step": int(row["step"]), "label": int(row["isFraud"])}
    req = urllib.request.Request("http://127.0.0.1:8000/score", json.dumps(payload).encode(), {"Content-Type": "application/json"})
    urllib.request.urlopen(req).read(); i += 1
    time.sleep(1.0 / rate)
