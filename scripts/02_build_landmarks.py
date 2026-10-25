"""Landmark (at-risk) samples for two transitions on the CNEMC weekly grid.
onset   : station compliant at week t  -> weeks until first observed Class-III breach
recovery: station in breach at week t  -> weeks until first observed compliant week
Strict right-censoring at the first unobserved week (monitoring gap) or record end.
Features use information observed at or before t only.
"""
import pandas as pd, numpy as np
from pathlib import Path
OUT = Path(__file__).resolve().parents[1] / "results"
K = 12
w = pd.read_parquet(OUT / "cnemc_weekly.parquet")
IND = ["NH4N", "DO", "CODMn", "pH"]
rows = []
for s, g in w.groupby("site"):
    g = g.set_index("date").sort_index()
    idx = pd.date_range(g.index.min(), g.index.max(), freq="W-MON")
    g = g.reindex(idx)
    g["site"] = s
    g["lon"] = g["lon"].ffill().bfill(); g["lat"] = g["lat"].ffill().bfill()
    obs = g[["NH4N", "DO", "CODMn"]].notna().all(axis=1)  # status determinable
    ex = (g.NH4N > 1.0) | (g.DO < 5.0) | (g.CODMn > 6.0)
    state = pd.Series(np.where(obs, ex.astype(float), np.nan), index=idx)
    g["obs"] = obs; g["state"] = state
    for v in ["NH4N", "CODMn"]:
        g[f"log_{v}"] = np.log1p(g[v])
    feats = {}
    for v in ["log_NH4N", "DO", "log_CODMn", "pH"]:
        x = g[v]
        feats[f"{v}_0"] = x
        for L in (1, 2, 4):
            feats[f"{v}_l{L}"] = x.shift(L)
        feats[f"{v}_d1"] = x - x.shift(1)
        for W in (4, 12, 52):
            r = x.rolling(W, min_periods=max(2, W // 4))
            feats[f"{v}_m{W}"] = r.mean()
            feats[f"{v}_s{W}"] = r.std()
        feats[f"{v}_max12"] = x.rolling(12, min_periods=3).max()
        feats[f"{v}_min12"] = x.rolling(12, min_periods=3).min()
    # margins to Class-III limits (positive = safe)
    feats["mNH4N"] = 1.0 - g.NH4N
    feats["mDO"] = g.DO - 5.0
    feats["mCOD"] = 6.0 - g.CODMn
    feats["margin_min"] = pd.concat([feats["mNH4N"] / 1.0, feats["mDO"] / 5.0, feats["mCOD"] / 6.0], axis=1).min(axis=1)
    st = state.copy()
    feats["ex_rate52"] = st.rolling(52, min_periods=8).mean()
    feats["ex_rate_all"] = st.expanding(min_periods=8).mean()
    for v, e in [("NH4N", g.NH4N > 1.0), ("DO", g.DO < 5.0), ("COD", g.CODMn > 6.0)]:
        feats[f"ex_rate52_{v}"] = e.where(g[["NH4N", "DO", "CODMn"]].notna().all(axis=1)).rolling(52, min_periods=8).mean()
    # weeks since last breach / last compliance
    last_ex = pd.Series(np.where(st == 1, np.arange(len(st)), np.nan), index=idx).ffill()
    last_ok = pd.Series(np.where(st == 0, np.arange(len(st)), np.nan), index=idx).ffill()
    feats["wk_since_ex"] = np.arange(len(st)) - last_ex
    feats["wk_since_ok"] = np.arange(len(st)) - last_ok
    # current run length of same state
    feats["obs_frac12"] = obs.rolling(12, min_periods=1).mean()
    F = pd.DataFrame(feats, index=idx)
    woy = idx.isocalendar().week.values.astype(float)
    F["woy_sin"] = np.sin(2 * np.pi * woy / 52.18); F["woy_cos"] = np.cos(2 * np.pi * woy / 52.18)
    F["lon"] = g.lon.values; F["lat"] = g.lat.values
    F["site"] = s; F["date"] = idx
    # outcomes for landmarks
    sv = state.values; n = len(sv)
    T = np.full(n, np.nan); E = np.zeros(n, dtype=int); C = np.full(n, np.nan)
    for i in range(n):
        if np.isnan(sv[i]):
            continue
        cur = sv[i]
        for k in range(1, K + 1):
            j = i + k
            if j >= n or np.isnan(sv[j]):
                T[i] = k - 1; E[i] = 0; break   # censored after k-1 fully observed weeks
            if sv[j] != cur:
                T[i] = k; E[i] = 1; break
        else:
            T[i] = K; E[i] = 0                   # administratively censored at horizon
        # monitoring-censoring time (first unobserved week ahead, capped at K) known for all
        cc = K
        for k in range(1, K + 1):
            j = i + k
            if j >= n or np.isnan(sv[j]):
                cc = k - 1; break
        C[i] = cc
    F["state"] = sv; F["T"] = T; F["E"] = E; F["C"] = C
    rows.append(F[~np.isnan(sv)])
L = pd.concat(rows, ignore_index=True)
L["trans"] = np.where(L.state == 0, "onset", "recovery")
L["year"] = L.date.dt.year
print(L.groupby("trans").agg(n=("T", "size"), events=("E", "sum"), sites=("site", "nunique"), meanT=("T", "mean"), cens0=("T", lambda t: (t == 0).mean())))
print("incomplete 12-wk windows (C<12) share by period:")
L["post"] = L.date >= "2015-01-01"
print(L.assign(inc=L.C < K).groupby(["trans", "post"]).inc.mean().round(3))
L.to_parquet(OUT / "landmarks.parquet")
