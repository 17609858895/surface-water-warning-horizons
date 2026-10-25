"""Post-recovery vulnerability window.
(1) Empirical next-week breach probability for compliant weeks vs weeks since the last recovery (1..52) and for
    long-compliant weeks (no breach in >=52 observed-or-not weeks), per indicator; hysteresis-band sensitivity.
(2) DML (station-grouped cross-fitting, cluster SE): effect of being in the relapse window (recovered <= 8 weeks ago)
    on next-week breach, under nested adjustment sets:
      S0 season + location + station long-run climatology (training-free: station mean/sd up to t-52)
      S1 S0 + 52-week breach rate
      S2 S1 + current margins to the Class-III limits and 4-week concentration means (proximity)
      S3 S2 + full short-term dynamics (lags, slopes, rolling sd)
    The drop from S1 to S2 measures how much of the relapse excess is explained by proximity to the threshold.
"""
import numpy as np, pandas as pd, lightgbm as lgb
from pathlib import Path
from sklearn.model_selection import GroupKFold
OUT = Path(__file__).resolve().parents[1] / "results"
w = pd.read_parquet(OUT / "cnemc_weekly.parquet")
L = pd.read_parquet(OUT / "landmarks.parquet")
Om = np.load(OUT / "obs_windows.npy")
L["next"] = Om[:, 1]


def series_rows(band=0.0):
    rows = []
    for s, g in w.groupby("site"):
        g = g.set_index("date").sort_index(); idx = pd.date_range(g.index.min(), g.index.max(), freq="W-MON"); g = g.reindex(idx)
        ob = g[["NH4N", "DO", "CODMn"]].notna().all(axis=1)
        defs = {"any": ((g.NH4N > 1) | (g.DO < 5) | (g.CODMn > 6), (g.NH4N <= 1 - band) & (g.DO >= 5 + 5 * band) & (g.CODMn <= 6 - 6 * band)),
                "NH4N": (g.NH4N > 1, g.NH4N <= 1 - band), "DO": (g.DO < 5, g.DO >= 5 + 5 * band), "CODMn": (g.CODMn > 6, g.CODMn <= 6 - 6 * band)}
        for nm, (brk, ok) in defs.items():
            # hysteresis: state switches to breach on brk, back to compliant only when ok
            st = np.full(len(idx), np.nan); cur = np.nan
            for i in range(len(idx)):
                if not ob.iloc[i]: cur = np.nan; continue
                if brk.iloc[i]: cur = 1.0
                elif ok.iloc[i] or np.isnan(cur): cur = 0.0
                st[i] = cur
            st = pd.Series(st, index=idx)
            # weeks since last recovery (first compliant week after a breach) using all weeks (gaps count as time)
            last_rec = pd.Series(np.where((st == 0) & (st.shift(1) == 1), np.arange(len(st)), np.nan), index=idx).ffill()
            last_brk = pd.Series(np.where(st == 1, np.arange(len(st)), np.nan), index=idx).ffill()
            since_rec = np.arange(len(st)) - last_rec.values + 1
            since_brk = np.arange(len(st)) - last_brk.values
            d = pd.DataFrame(dict(site=s, ind=nm, date=idx, cur=st.values, nxt=st.shift(-1).values,
                                  since_rec=since_rec, since_brk=since_brk))
            rows.append(d[(d.cur == 0) & d.nxt.notna()])
    return pd.concat(rows)


out = []
for band in (0.0, 0.1):
    R = series_rows(band)
    R["win"] = np.where(R.since_brk.isna() | (R.since_brk > 52), "long", np.clip(R.since_rec, 1, 53).astype("Int64").astype(str))
    for ind, g in R.groupby("ind"):
        base = g[g.win == "long"].nxt.mean()
        for j in list(range(1, 27)) + [52]:
            gg = g[(g.since_brk <= 52) & (g.since_rec == j)]
            if len(gg) < 30: continue
            out.append(dict(band=band, ind=ind, weeks_since_recovery=j, p_breach=gg.nxt.mean(), n=len(gg), p_long=base,
                            n_long=(g.win == "long").sum(), ratio=gg.nxt.mean() / base))
V = pd.DataFrame(out); V.to_csv(OUT / "relapse_window_empirical.csv", index=False)
print(V[(V.band == 0) & V.weeks_since_recovery.isin([1, 2, 4, 8, 12, 26])].round(4).to_string())
print(V[(V.band == 0.1) & V.weeks_since_recovery.isin([1, 2, 4, 8, 12, 26])].round(4).to_string())

# ---------- DML on onset landmarks (compliant now) ----------
D = L[(L.state == 0) & ~np.isnan(L["next"]) & (L.date.dt.year >= 2008)].reset_index(drop=True)
D["treat"] = (D.wk_since_ex <= 9).astype(float)       # breach within the previous 9 weeks <=> recovered <= 8 weeks ago
D["y"] = D["next"].astype(float)
# station climatology up to t-52 (no future information)
clim = []
for s, g in D.groupby("site"):
    g = g.sort_values("date")
    for v in ["log_NH4N_0", "DO_0", "log_CODMn_0"]:
        g[f"clim_{v}"] = g[v].shift(52).expanding(min_periods=10).mean()
    clim.append(g)
D = pd.concat(clim).sort_index()
S0 = ["woy_sin", "woy_cos", "lon", "lat", "clim_log_NH4N_0", "clim_DO_0", "clim_log_CODMn_0"]
S1 = S0 + ["ex_rate52", "ex_rate52_NH4N", "ex_rate52_DO", "ex_rate52_COD", "log_NH4N_m52", "DO_m52", "log_CODMn_m52"]
S2 = S1 + ["mNH4N", "mDO", "mCOD", "margin_min", "log_NH4N_m4", "DO_m4", "log_CODMn_m4"]
S3 = S2 + [c for c in D.columns if any(c.startswith(p) for p in ("log_NH4N_", "DO_", "log_CODMn_", "pH_")) and c not in S2 and not c.startswith("clim")]
prm = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=200, feature_fraction=0.8,
           bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=24, seed=7)
res = []
for nm, S in [("S0 season+location+climatology", S0), ("S1 +52-wk breach history", S1),
              ("S2 +current proximity to limits", S2), ("S3 +full short-term dynamics", S3)]:
    Xm = D[S].values.astype(float); Y = D.y.values; T = D.treat.values
    ry = np.zeros_like(Y); rt = np.zeros_like(T)
    for a, b in GroupKFold(5).split(Xm, Y, D.site.values):
        my = lgb.train(prm, lgb.Dataset(Xm[a], Y[a]), 300); ry[b] = Y[b] - my.predict(Xm[b])
        mt = lgb.train(prm, lgb.Dataset(Xm[a], T[a]), 300); rt[b] = T[b] - np.clip(mt.predict(Xm[b]), 0.01, 0.99)
    theta = (rt @ ry) / (rt @ rt); eps = ry - theta * rt
    meat = sum((rt[D.site.values == s] @ eps[D.site.values == s]) ** 2 for s in np.unique(D.site.values))
    se = np.sqrt(meat) / (rt @ rt)
    base = Y[T == 0].mean()
    res.append(dict(adjust=nm, theta=theta, se=se, lo=theta - 1.96 * se, hi=theta + 1.96 * se, base_long=base,
                    raw_diff=Y[T == 1].mean() - base, n=len(Y), n_treat=int(T.sum())))
    print(res[-1], flush=True)
pd.DataFrame(res).to_csv(OUT / "relapse_dml.csv", index=False)
