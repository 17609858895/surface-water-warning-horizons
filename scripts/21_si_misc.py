"""Supplementary analyses:
(1) Relapse-window length (4/8/12 weeks): pooled weekly relapse vs onset probability and PLR-DML theta under S0, S1, S2a.
(2) Conditional coverage of the weighted-conformal safe-week bound (alpha = 0.10) by start state, chronic status and year.
(3) Breach-occupancy forecasts of 3S-Boost+FT versus a direct regression of the observed 12-week breach share."""
import sys, numpy as np, pandas as pd, lightgbm as lgb
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, str(Path(__file__).parent))
from survlib import *
from tristate import start_state

OUT = Path(__file__).resolve().parents[1] / "results"; SD = OUT / "si"; SD.mkdir(exist_ok=True)
L = pd.read_parquet(OUT / "landmarks.parquet").reset_index(drop=True); O = np.load(OUT / "obs_windows.npy")
state = L.state.values.astype(int)

# ---------------- (1) window length ----------------
w = pd.read_parquet(OUT / "cnemc_weekly.parquet"); w = w[w.date.dt.year >= 2008]
pairs = []
for s, g in w.groupby("site"):
    g = g.set_index("date").sort_index(); idx = pd.date_range(g.index.min(), g.index.max(), freq="W-MON"); g = g.reindex(idx)
    ob = g[["NH4N", "DO", "CODMn"]].notna().all(axis=1)
    st = ((g.NH4N > 1) | (g.DO < 5) | (g.CODMn > 6)).astype(float).where(ob)
    last = pd.Series(np.where(st == 1, np.arange(len(st)), np.nan), index=idx).ffill()
    pairs.append(pd.DataFrame(dict(site=s, cur=st.values, nxt=st.shift(-1).values, since=np.arange(len(st)) - last.values)))
P = pd.concat(pairs); P = P[(P.cur == 0) & P.nxt.notna()]
rng = np.random.default_rng(21); win = []
D = L[(L.state == 0) & ~np.isnan(O[:, 1]) & (L.date.dt.year >= 2008)].copy(); D["y"] = O[D.index, 1]
clim = []
for s, g in D.groupby("site"):
    g = g.sort_values("date")
    for v in ["log_NH4N_0", "DO_0", "log_CODMn_0"]: g[f"clim_{v}"] = g[v].shift(52).expanding(min_periods=10).mean()
    clim.append(g)
D = pd.concat(clim).sort_values(["site", "date"]).reset_index(drop=True)
S0 = ["woy_sin", "woy_cos", "lon", "lat", "clim_log_NH4N_0", "clim_DO_0", "clim_log_CODMn_0"]
S1 = S0 + ["ex_rate52", "ex_rate52_NH4N", "ex_rate52_DO", "ex_rate52_COD", "log_NH4N_m52", "DO_m52", "log_CODMn_m52"]
S2a = S1 + ["mNH4N", "mDO", "mCOD", "margin_min"]
PRM = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8,
           bagging_freq=1, verbose=-1, num_threads=16, seed=2)


def folds(groups):
    us = np.unique(groups); perm = np.random.default_rng(0).permutation(len(us)); fid = dict(zip(us[perm], np.arange(len(us)) % 5))
    f = np.array([fid[g] for g in groups]); return [(np.where(f != i)[0], np.where(f == i)[0]) for i in range(5)]


def plr(d, S, tcol):
    Xm = d[S].values.astype(float); Y = d.y.values; T = d[tcol].values; ry = np.zeros(len(d)); rt = np.zeros(len(d))
    for a, b in folds(d.site.values):
        ry[b] = Y[b] - lgb.train(PRM, lgb.Dataset(Xm[a], Y[a]), 300).predict(Xm[b])
        rt[b] = T[b] - np.clip(lgb.train(PRM, lgb.Dataset(Xm[a], T[a]), 300).predict(Xm[b]), 0.01, 0.99)
    th = (rt @ ry) / (rt @ rt); eps = ry - th * rt; sv = d.site.values
    se = np.sqrt(sum((rt[sv == s] @ eps[sv == s]) ** 2 for s in np.unique(sv))) / (rt @ rt); return th, se


for M in (4, 8, 12):
    rel = P[P.since.between(1, M)]; ons = P[P.since.isna() | (P.since > M)]
    sites = P.site.unique(); tr_ = rel.groupby("site").nxt.agg(["sum", "count"]); to_ = ons.groupby("site").nxt.agg(["sum", "count"])
    bs = []
    for _ in range(300):
        sm = rng.choice(sites, len(sites)); a = tr_.reindex(sm).fillna(0); b = to_.reindex(sm).fillna(0)
        bs.append((a["sum"].sum() / a["count"].sum(), b["sum"].sum() / b["count"].sum()))
    bs = np.array(bs)
    D[f"T{M}"] = ((D.wk_since_ex >= 1) & (D.wk_since_ex <= M)).astype(float)
    row = dict(window=M, p_relapse=rel.nxt.mean(), p_relapse_lo=np.quantile(bs[:, 0], 0.025), p_relapse_hi=np.quantile(bs[:, 0], 0.975),
               p_onset=ons.nxt.mean(), p_onset_lo=np.quantile(bs[:, 1], 0.025), p_onset_hi=np.quantile(bs[:, 1], 0.975), n_relapse=len(rel))
    for nm, S in [("S0", S0), ("S1", S1), ("S2a", S2a)]:
        th, se = plr(D, S, f"T{M}"); row[f"theta_{nm}"] = th; row[f"se_{nm}"] = se
    win.append(row); print(row, flush=True)
pd.DataFrame(win).to_csv(SD / "window_sensitivity.csv", index=False)

# ---------------- (2) conditional coverage ----------------
H = pd.read_pickle(OUT / "core" / "hazards.pkl")["onset"]; CF = pd.read_csv(OUT / "fig7" / "conformal.csv")
beta = CF[(CF.alpha == 0.10) & (CF.method == "Weighted conformal")].beta.iloc[0]
te = (L.date >= "2015-01-01").values & (L.date.dt.year <= 2016).values
mte = (L.date >= "2015-01-01").values & (state == 0); Lte = L[mte].reset_index(drop=True); hte = H["BRR-SL"]["te"]
keepyr = Lte.date.dt.year.values <= 2016; Lte, hte = Lte[keepyr].reset_index(drop=True), hte[keepyr]
F = 1 - surv_from_h(hte); q = (F >= beta).argmax(1) + 1; q[F[:, -1] < beta] = K + 1; lpb = np.maximum(np.minimum(q, K) - 1, 0)
truth = np.where(Lte.E.values == 1, Lte["T"].values, K + 1); ev = Lte.C.values >= K
s3 = start_state(Lte.state.values.astype(int), Lte.wk_since_ex.values)
groups = {"All": np.ones(len(Lte), bool), "A start": s3 == 0, "R start": s3 >= 2,
          "Chronic (52-wk share > 0.10)": Lte.ex_rate52.fillna(0).values > 0.10, "Non-chronic (≤ 0.10)": Lte.ex_rate52.fillna(0).values <= 0.10,
          "2015": Lte.date.dt.year.values == 2015, "2016": Lte.date.dt.year.values == 2016}
cc = []
for g, m in groups.items():
    mm = m & ev; k_ = truth[mm] > lpb[mm]; n = mm.sum()
    lo, hi = (np.quantile([rng.choice(k_, n).mean() for _ in range(500)], [0.025, 0.975]) if n > 20 else (np.nan, np.nan))
    cc.append(dict(group=g, coverage=k_.mean() if n else np.nan, lo=lo, hi=hi, n_eval=int(n), mean_safe_weeks=lpb[m].mean(), n_all=int(m.sum())))
CC = pd.DataFrame(cc); CC.to_csv(SD / "conditional_coverage.csv", index=False); print(CC.round(3).to_string(), flush=True)

# ---------------- (3) occupancy ----------------
occ = np.load(OUT / "core" / "occ_3s_test.npy"); FFt = np.load(OUT / "forecast_features.npy")
FC = feat_cols(L); XH = np.column_stack([L[FC].values.astype(float), state, FFt])
tr = (L.date <= "2013-09-30").values; va = ((L.date >= "2014-01-01") & (L.date <= "2014-09-30")).values; tt = (L.date >= "2015-01-01").values
def frac(m):
    ob = ~np.isnan(O[m][:, 1:]); n = ob.sum(1); return np.where(n > 0, np.where(ob, O[m][:, 1:], 0).sum(1) / np.maximum(n, 1), np.nan), n, ob
yt, nt, _ = frac(tr); yv, nv, _ = frac(va); ye, ne, obe = frac(tt)
reg = lgb.train(dict(LGB_PARAMS, objective="cross_entropy"), lgb.Dataset(XH[tr][nt > 0], yt[nt > 0], weight=nt[nt > 0]), 3000,
                valid_sets=[lgb.Dataset(XH[va][nv > 0], yv[nv > 0], weight=nv[nv > 0])], callbacks=[lgb.early_stopping(100, verbose=False)])
pd_ = reg.predict(XH[tt]); p3 = np.where(obe, occ, 0).sum(1) / np.maximum(ne, 1)
s3t = start_state(state[tt], L.wk_since_ex.values[tt]); keep = ne >= 4; out = []
for g, m in {"All": keep, "A start": keep & (s3t == 0), "B start": keep & (s3t == 1), "R start": keep & (s3t >= 2)}.items():
    for nm, p in [("3S-Boost+FT", p3), ("Direct regression", pd_)]:
        out.append(dict(group=g, model=nm, MAE=np.mean(np.abs(p[m] - ye[m])), rho=spearmanr(p[m], ye[m])[0], bias=np.mean(p[m] - ye[m]), n=int(m.sum())))
OC = pd.DataFrame(out); OC.to_csv(SD / "occupancy.csv", index=False); print(OC.round(4).to_string())
