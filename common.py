import numpy as np, pandas as pd

TYPES = ["PAYMENT", "CASH_IN", "CASH_OUT", "TRANSFER", "DEBIT"]
FEATURES = ["type_code", "log_amount", "drain_ratio", "new_balance_zero", "dest_was_empty", "hour"]


def synth(n=200_000, fraud_rate=0.012, seed=42):
    """Synthetic PaySim-style transactions (same columns as PaySim)."""
    rng = np.random.default_rng(seed)
    step = np.sort(rng.integers(1, 744, n))
    ttype = rng.choice(TYPES, n, p=[.35, .2, .30, .1, .05])
    amount = rng.lognormal(7.5, 1.2, n)
    old = rng.lognormal(9.0, 1.3, n)
    dest_old = rng.lognormal(9.0, 1.3, n)
    dest_old[rng.random(n) < 0.30] = 0
    fraud = np.zeros(n, int)

    cand = np.where(np.isin(ttype, ["TRANSFER", "CASH_OUT"]))[0]
    f_idx = rng.choice(cand, int(n * fraud_rate), replace=False)
    fraud[f_idx] = 1
    drain = rng.random(len(f_idx)) < 0.8
    amount[f_idx] = np.where(drain, old[f_idx] * rng.uniform(0.95, 1.0, len(f_idx)),
                             old[f_idx] * rng.uniform(0.3, 0.8, len(f_idx)))
    dest_old[f_idx[rng.random(len(f_idx)) < 0.7]] = 0

    # hard negatives: some genuine users also empty their accounts
    legit = np.setdiff1d(cand, f_idx)
    h_idx = rng.choice(legit, int(len(legit) * 0.02), replace=False)
    amount[h_idx] = old[h_idx] * rng.uniform(0.9, 1.0, len(h_idx))

    amount = np.minimum(amount, old)
    new = np.where(ttype == "CASH_IN", old + amount, old - amount)
    dest_new = dest_old + amount
    return pd.DataFrame({"step": step, "type": ttype, "amount": amount.round(2),
                         "oldbalanceOrg": old.round(2), "newbalanceOrig": new.round(2),
                         "oldbalanceDest": dest_old.round(2), "newbalanceDest": dest_new.round(2),
                         "isFraud": fraud})


def featurize(df):
    X = pd.DataFrame(index=df.index)
    X["type_code"] = df["type"].map({t: i for i, t in enumerate(TYPES)}).fillna(0).astype(int)
    X["log_amount"] = np.log1p(df["amount"])
    X["drain_ratio"] = (df["amount"] / (df["oldbalanceOrg"] + 1)).clip(0, 1.5)
    X["new_balance_zero"] = ((df["newbalanceOrig"] == 0) & (df["oldbalanceOrg"] > 0)).astype(int)
    X["dest_was_empty"] = (df["oldbalanceDest"] == 0).astype(int)
    X["hour"] = df["step"] % 24
    return X[FEATURES]


def reasons(row):
    """Plain-language reasons from feature values (transparent, rule-based explanations)."""
    out = []
    if row["drain_ratio"] >= 0.9:
        out.append("Transfer empties ~%d%% of the account balance" % round(row["drain_ratio"] * 100))
    elif row["drain_ratio"] >= 0.5:
        out.append("Unusually large share of balance (%d%%)" % round(row["drain_ratio"] * 100))
    if row["dest_was_empty"]:
        out.append("Receiver account was empty / newly active")
    if row["type_code"] in (2, 3):
        out.append("High-risk transaction type (transfer / cash-out)")
    if row["hour"] < 5:
        out.append("Night-time activity")
    return out or ["No strong risk signals"]
