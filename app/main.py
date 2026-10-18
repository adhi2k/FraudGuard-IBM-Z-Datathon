import sys, os, time, sqlite3, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import joblib, pandas as pd
from typing import Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel
from common import featurize, reasons

# Network containment engine is included in this repository.
from network.engine import generate_scenario, greedy_containment

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
art = joblib.load(os.path.join(ROOT, "models/model.joblib"))
model, BLOCK, VERIFY = art["model"], art["block"], art["verify"]
db = sqlite3.connect(os.path.join(ROOT, "audit.db"), check_same_thread=False)
db.execute("CREATE TABLE IF NOT EXISTS audit(ts REAL, user TEXT, type TEXT, amount REAL, risk REAL, action TEXT, reasons TEXT)")

app = FastAPI(title="FraudGuard")
clients = set()
stats = {"total": 0, "approve": 0, "verify": 0, "block": 0, "prevented_amount": 0.0, "lat": []}


class Tx(BaseModel):
    user: str = "user"
    type: str
    amount: float
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float = 0
    newbalanceDest: float = 0
    step: int = 0
    label: Optional[int] = None   # demo only


@app.post("/score")
async def score(tx: Tx):
    t0 = time.perf_counter()
    df = pd.DataFrame([tx.dict()])
    X = featurize(df)
    risk = float(model.predict_proba(X)[0, 1])
    action = "block" if risk >= BLOCK else "verify" if risk >= VERIFY else "approve"
    lat = (time.perf_counter() - t0) * 1000
    res = {"user": tx.user, "type": tx.type, "amount": tx.amount, "risk": round(risk, 4),
           "action": action, "reasons": reasons(X.iloc[0]), "latency_ms": round(lat, 2), "label": tx.label}
    stats["total"] += 1; stats[action] += 1; stats["lat"] = (stats["lat"] + [lat])[-2000:]
    if action != "approve":
        stats["prevented_amount"] += tx.amount if action == "block" else 0
    db.execute("INSERT INTO audit VALUES(?,?,?,?,?,?,?)", (time.time(), tx.user, tx.type, tx.amount, risk, action, json.dumps(res["reasons"])))
    db.commit()
    for ws in list(clients):
        try:
            await ws.send_json(res)
        except Exception:
            clients.discard(ws)
    return res



class ContainmentRequest(BaseModel):
    seed: int = 42
    accounts: int = 80
    edges: int = 240
    max_holds: int = 3
    lambda_count: float = 250.0
    lambda_value: float = 0.03
    candidate_limit: int = 80


@app.post("/containment/recommend")
def containment_recommend(req: ContainmentRequest):
    """Return synthetic network containment recommendations; never execute holds."""
    if not 8 <= req.accounts <= 200:
        from fastapi import HTTPException
        raise HTTPException(status_code=422, detail="accounts must be between 8 and 200")
    if not 1 <= req.edges <= 1000:
        from fastapi import HTTPException
        raise HTTPException(status_code=422, detail="edges must be between 1 and 1000")
    if not 0 <= req.max_holds <= 10:
        from fastapi import HTTPException
        raise HTTPException(status_code=422, detail="max_holds must be between 0 and 10")
    if req.lambda_count < 0 or req.lambda_value < 0:
        from fastapi import HTTPException
        raise HTTPException(status_code=422, detail="penalties must be non-negative")
    if not 1 <= req.candidate_limit <= 1000:
        from fastapi import HTTPException
        raise HTTPException(status_code=422, detail="candidate_limit must be between 1 and 1000")

    scenario = generate_scenario(
        n_accounts=req.accounts,
        n_edges=req.edges,
        seed=req.seed,
    )
    result = greedy_containment(
        scenario,
        k=req.max_holds,
        lambda_count=req.lambda_count,
        lambda_value=req.lambda_value,
        seed=req.seed,
        mode="estimated",
        candidate_limit=req.candidate_limit,
    )

    return {
        "mode": "synthetic_simulation",
        "real_holds_executed": False,
        "warning": "Synthetic demonstration only. No live accounts or transactions were blocked.",
        "config": {
            "seed": req.seed,
            "accounts": req.accounts,
            "requested_edges": req.edges,
            "max_holds": req.max_holds,
            "lambda_count": req.lambda_count,
            "lambda_value": req.lambda_value,
            "candidate_limit": req.candidate_limit,
        },
        "recommendation": result.to_dict(),
        "recommended_edges": [
            {
                "edge_id": edge.edge_id,
                "source": edge.source,
                "target": edge.target,
                "amount": edge.amount,
                "risk": edge.risk,
                "hold_seconds": edge.hold_seconds,
            }
            for edge in scenario.edges
            if edge.edge_id in set(result.held_edge_ids)
        ],
        "scenario_metadata": scenario.metadata,
    }


@app.get("/history")
def get_history(limit: int = 100):
    limit = max(1, min(limit, 500))
    rows = db.execute(
        "SELECT ts, user, type, amount, risk, action, reasons "
        "FROM audit ORDER BY ts DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [
        {
            "timestamp": row[0],
            "user": row[1],
            "type": row[2],
            "amount": row[3],
            "risk": row[4],
            "action": row[5],
            "reasons": json.loads(row[6]) if row[6] else [],
        }
        for row in reversed(rows)
    ]


@app.get("/stats")
def get_stats():
    l = sorted(stats["lat"]) or [0]
    return {**{k: v for k, v in stats.items() if k != "lat"},
            "p50_ms": round(l[len(l) // 2], 2), "p99_ms": round(l[int(len(l) * .99) - 1], 2)}


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept(); clients.add(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        clients.discard(ws)


@app.get("/")
def index():
    return FileResponse(os.path.join(ROOT, "app/static/index.html"))
