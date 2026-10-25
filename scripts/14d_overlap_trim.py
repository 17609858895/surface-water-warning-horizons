"""Overlap diagnostics for the nested DML (Fig. 5): cross-fitted propensity per adjustment set, share in common support
(0.05 < e < 0.95), DML re-estimated on the trimmed sample (Crump et al. 2009 rule), a global-permutation placebo,
and a DR-learner CATE under S1 (chronic history only), which has better overlap than S2."""
import numpy as np, pandas as pd, lightgbm as lgb
from pathlib import Path
OUT = Path(__file__).resolve().parents[1] / "results"; RD = OUT / "relapse"
rng = np.random.default_rng(144); MEM = 8
L = pd.read_parquet(OUT / "landmarks.parquet"); Om = np.load(OUT / "obs_windows.npy"); L["next"] = Om[:, 1]
D = L[(L.state == 0) & ~np.isnan(L["next"]) & (L.date.dt.year >= 2008)].copy(); D["y"] = D["next"].astype(float)
clim = []
for s, g in D.groupby("site"):
    g = g.sort_values("date")
    for v in ["log_NH4N_0", "DO_0", "log_CODMn_0"]:
        g[f"clim_{v}"] = g[v].shift(52).expanding(min_periods=10).mean()
    clim.append(g)
D = pd.concat(clim).sort_values(["site", "date"]).reset_index(drop=True)
D["treat"] = ((D.wk_since_ex >= 1) & (D.wk_since_ex <= MEM)).astype(float)
SET0 = ["woy_sin", "woy_cos", "lon", "lat", "clim_log_NH4N_0", "clim_DO_0", "clim_log_CODMn_0"]
SET1 = SET0 + ["ex_rate52", "ex_rate52_NH4N", "ex_rate52_DO", "ex_rate52_COD", "log_NH4N_m52", "DO_m52", "log_CODMn_m52"]
SET2a = SET1 + ["mNH4N", "mDO", "mCOD", "margin_min"]
SET2 = SET2a + ["log_NH4N_m4", "DO_m4", "log_CODMn_m4"]
SET3 = SET2 + [c for c in D.columns if any(c.startswith(p) for p in ("log_NH4N_", "DO_", "log_CODMn_", "pH_")) and c not in SET2 and not c.startswith("clim")]
PRM = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=200, feature_fraction=0.8,
           bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=6, seed=1)


def folds(groups, seed=0):
    us = np.unique(groups); perm = np.random.default_rng(seed).permutation(len(us)); fid = dict(zip(us[perm], np.arange(len(us)) % 5))
    f = np.array([fid[g] for g in groups]); return [(np.where(f != i)[0], np.where(f == i)[0]) for i in range(5)]


def crossfit(d, S, target):
    X = d[S].values.astype(float); y = d[target].values.astype(float); p = np.zeros(len(d))
    for a, b in folds(d.site.values):
        p[b] = lgb.train(PRM, lgb.Dataset(X[a], y[a]), 300).predict(X[b])
    return p


def plr(d, S, treat="treat"):
    e = np.clip(crossfit(d, S, treat), 0.01, 0.99); m = crossfit(d, S, "y")
    rt = d[treat].values - e; ry = d.y.values - m
    th = (rt @ ry) / (rt @ rt); eps = ry - th * rt; sv = d.site.values
    se = np.sqrt(sum((rt[sv == s] @ eps[sv == s]) ** 2 for s in np.unique(sv))) / (rt @ rt)
    return th, se, e


rows = []; props = {}
for nm, S in [("S0", SET0), ("S1", SET1), ("S2a (no 4-wk means)", SET2a), ("S2", SET2), ("S3", SET3)]:
    th, se, e = plr(D, S); props[nm] = e
    keep = (e > 0.05) & (e < 0.95); Dk = D[keep].reset_index(drop=True)
    th_t, se_t, _ = plr(Dk, S) if keep.sum() > 2000 and Dk.treat.nunique() == 2 else (np.nan, np.nan, None)
    rows.append(dict(adjust=nm, theta=th, se=se, overlap_share=keep.mean(), n_trim=int(keep.sum()), n_treat_trim=int(Dk.treat.sum()),
                     theta_trim=th_t, se_trim=se_t, lo_trim=th_t - 1.96 * se_t, hi_trim=th_t + 1.96 * se_t))
    print(rows[-1], flush=True)
OV = pd.DataFrame(rows); OV.to_csv(RD / "dml_overlap_trim.csv", index=False)
pd.DataFrame({k: v for k, v in props.items()}).assign(treat=D.treat.values).to_csv(RD / "propensity_by_set.csv", index=False)

# global-permutation placebo (true null, S2) and within-station-year placebo for comparison already in dml_robustness.csv
P = D.copy(); P["treat_g"] = rng.permutation(P.treat.values)
th, se, _ = plr(P, SET2, "treat_g")
pd.DataFrame([dict(check="Placebo: global permutation", theta=th, se=se, lo=th - 1.96 * se, hi=th + 1.96 * se, n=len(P))]).to_csv(RD / "dml_placebo_global.csv", index=False)
print("global placebo", th, se, flush=True)

# DR-learner CATE under S1 adjustment
Y = D.y.values; T = D.treat.values; Xm = D[SET1].values.astype(float)
mu0 = np.zeros(len(D)); mu1 = np.zeros(len(D)); e = np.zeros(len(D))
for a, b in folds(D.site.values):
    a1 = a[T[a] == 1]; a0 = a[T[a] == 0]
    mu1[b] = lgb.train(PRM, lgb.Dataset(Xm[a1], Y[a1]), 300).predict(Xm[b])
    mu0[b] = lgb.train(PRM, lgb.Dataset(Xm[a0], Y[a0]), 300).predict(Xm[b])
    e[b] = np.clip(lgb.train(PRM, lgb.Dataset(Xm[a], T[a]), 300).predict(Xm[b]), 0.05, 0.95)
D["phi"] = mu1 - mu0 + T * (Y - mu1) / e - (1 - T) * (Y - mu0) / (1 - e)
cate = []
for mod, bins in [("ex_rate52", [0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0]), ("margin_min", [-np.inf, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, np.inf])]:
    d = D[[mod, "phi", "site"]].dropna(); d["bin"] = pd.cut(d[mod], bins, include_lowest=True)
    for bn, g in d.groupby("bin", observed=True):
        if len(g) < 200: continue
        sites = g.site.unique(); sm = g.groupby("site").phi.agg(["sum", "count"])
        bs = [sm.loc[rng.choice(sites, len(sites))].pipe(lambda t: t["sum"].sum() / t["count"].sum()) for _ in range(300)]
        cate.append(dict(moderator=mod, bin=str(bn), x=g[mod].median(), cate=g.phi.mean(), lo=np.quantile(bs, 0.025), hi=np.quantile(bs, 0.975), n=len(g)))
C = pd.DataFrame(cate); C.to_csv(RD / "dr_cate_S1.csv", index=False); print(C.round(4).to_string())
print("S1 overlap share:", ((e > 0.05) & (e < 0.95)).mean().round(3))
