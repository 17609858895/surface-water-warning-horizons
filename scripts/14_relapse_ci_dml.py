"""Fig. 2 and Fig. 5 analyses: episode KM, relapse-window curves with station-bootstrap CI, hysteresis bands,
nested DML with repeated cross-fitting (treatment = recovered <= 8 weeks ago), robustness checks and DR-learner CATE."""
import numpy as np, pandas as pd, lightgbm as lgb
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
OUT = Path(__file__).resolve().parents[1] / "results"; RD = OUT / "relapse"; RD.mkdir(exist_ok=True)
rng = np.random.default_rng(14)
MEM = 8
w = pd.read_parquet(OUT / "cnemc_weekly.parquet")


def state_series(g, ob, brk, ok):
    st = np.full(len(g), np.nan); cur = np.nan
    for i in range(len(g)):
        if not ob[i]: cur = np.nan; continue
        if brk[i]: cur = 1.0
        elif ok[i] or np.isnan(cur): cur = 0.0
        st[i] = cur
    return st


def all_series(band=0.0):
    rows = []
    for s, g in w.groupby("site"):
        g = g.set_index("date").sort_index(); idx = pd.date_range(g.index.min(), g.index.max(), freq="W-MON"); g = g.reindex(idx)
        ob = g[["NH4N", "DO", "CODMn"]].notna().all(axis=1).values
        N, D, C = g.NH4N.values, g.DO.values, g.CODMn.values
        defs = {"any": ((N > 1) | (D < 5) | (C > 6), (N <= 1 - band) & (D >= 5 * (1 + band)) & (C <= 6 * (1 - band))),
                "NH4N": (N > 1, N <= 1 - band), "DO": (D < 5, D >= 5 * (1 + band)), "CODMn": (C > 6, C <= 6 * (1 - band))}
        for nm, (brk, ok) in defs.items():
            st = state_series(idx, ob, brk, ok); ss = pd.Series(st)
            last_rec = pd.Series(np.where((ss == 0) & (ss.shift(1) == 1), np.arange(len(st)), np.nan)).ffill().values
            last_brk = pd.Series(np.where(ss == 1, np.arange(len(st)), np.nan)).ffill().values
            rows.append(pd.DataFrame(dict(site=s, ind=nm, date=idx, cur=st, nxt=ss.shift(-1).values,
                                          since_rec=np.arange(len(st)) - last_rec + 1, since_brk=np.arange(len(st)) - last_brk)))
    return pd.concat(rows, ignore_index=True)


SKIP = (RD / "episode_km.csv").exists() and (RD / "relapse_curves_ci.csv").exists()
# ---------- (1) episode KM (time to recovery from onset; censored at first gap or record end) ----------
S0 = all_series(0.0)
km = []
for ind, d in ([] if SKIP else S0.groupby("ind")):
    durs, evs = [], []
    for s, g in d.groupby("site"):
        st = g.cur.values
        for i in range(1, len(st)):
            if st[i] == 1 and st[i - 1] == 0:
                j = i; ev = 0
                while True:
                    if j + 1 >= len(st) or np.isnan(st[j + 1]): ev = 0; break
                    if st[j + 1] == 0: ev = 1; break
                    j += 1
                durs.append(j - i + 1); evs.append(ev)
    durs, evs = np.array(durs), np.array(evs)
    surv = 1.0
    km.append(dict(ind=ind, week=0, S=1.0, n_risk=len(durs)))
    for t in range(1, 27):
        at = (durs >= t).sum(); e = ((durs == t) & (evs == 1)).sum()
        surv *= (1 - e / at) if at > 0 else 1
        km.append(dict(ind=ind, week=t, S=surv, n_risk=int(at)))
    print(ind, "episodes", len(durs), "events", evs.sum(), "median-dur KM:", next((r["week"] for r in km if r["ind"] == ind and r["S"] <= 0.5), None))
if not SKIP: pd.DataFrame(km).to_csv(RD / "episode_km.csv", index=False)


# ---------- (2) relapse curves with station-bootstrap CI; hysteresis bands ----------
def curve(Sr, ind, B=300, js=tuple(range(1, 27))):
    d = Sr[(Sr.ind == ind) & (Sr.cur == 0) & Sr.nxt.notna()].copy()
    d["grp"] = np.where(d.since_brk.isna() | (d.since_brk > 52), 0, np.where(d.since_rec.between(1, 26), d.since_rec, -1))
    d = d[d.grp >= 0]
    tab = d.groupby(["site", "grp"]).nxt.agg(["sum", "count"]).unstack(fill_value=0)
    sites = tab.index.values
    def est(t):
        s = t["sum"].sum(0); c = t["count"].sum(0); p = s / c.replace(0, np.nan)
        return p
    p0 = est(tab); bs = []
    for _ in range(B):
        bs.append(est(tab.loc[rng.choice(sites, len(sites))]).values)
    bs = np.array(bs); out = []
    base_b = bs[:, list(p0.index).index(0)]
    for j in js:
        if j not in p0.index: continue
        k = list(p0.index).index(j)
        out.append(dict(ind=ind, j=j, p=p0[j], lo=np.nanquantile(bs[:, k], 0.025), hi=np.nanquantile(bs[:, k], 0.975),
                        p_long=p0[0], fold=p0[j] / p0[0], fold_lo=np.nanquantile(bs[:, k] / base_b, 0.025),
                        fold_hi=np.nanquantile(bs[:, k] / base_b, 0.975), n=int(tab["count"][j].sum())))
    return out


if SKIP:
    Sb10 = all_series(0.10)
else:
    rel = []
    for ind in ["any", "NH4N", "DO", "CODMn"]:
        rel += [dict(band=0.0, **r) for r in curve(S0, ind)]
    for band in (0.05, 0.10, 0.20):
        Sb = all_series(band)
        rel += [dict(band=band, **r) for r in curve(Sb, "any", B=200)]
        if band == 0.10: Sb10 = Sb
    pd.DataFrame(rel).to_csv(RD / "relapse_curves_ci.csv", index=False)

# ---------- (3) DML ----------
L = pd.read_parquet(OUT / "landmarks.parquet"); Om = np.load(OUT / "obs_windows.npy"); L["next"] = Om[:, 1]
D = L[(L.state == 0) & ~np.isnan(L["next"]) & (L.date.dt.year >= 2008)].copy()
D["y"] = D["next"].astype(float)
clim = []
for s, g in D.groupby("site"):
    g = g.sort_values("date")
    for v in ["log_NH4N_0", "DO_0", "log_CODMn_0"]:
        g[f"clim_{v}"] = g[v].shift(52).expanding(min_periods=10).mean()
    clim.append(g)
D = pd.concat(clim).sort_values(["site", "date"]).reset_index(drop=True)
SET0 = ["woy_sin", "woy_cos", "lon", "lat", "clim_log_NH4N_0", "clim_DO_0", "clim_log_CODMn_0"]
SET1 = SET0 + ["ex_rate52", "ex_rate52_NH4N", "ex_rate52_DO", "ex_rate52_COD", "log_NH4N_m52", "DO_m52", "log_CODMn_m52"]
SET2 = SET1 + ["mNH4N", "mDO", "mCOD", "margin_min", "log_NH4N_m4", "DO_m4", "log_CODMn_m4"]
SET3 = SET2 + [c for c in D.columns if any(c.startswith(p) for p in ("log_NH4N_", "DO_", "log_CODMn_", "pH_")) and c not in SET2 and not c.startswith("clim")]
SETS = [("Raw", None), ("S0", SET0), ("S1", SET1), ("S2", SET2), ("S3", SET3)]
PRM = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=200, feature_fraction=0.8,
           bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=6)


def folds(groups, seed):
    us = np.unique(groups); perm = np.random.default_rng(seed).permutation(len(us)); fid = dict(zip(us[perm], np.arange(len(us)) % 5))
    f = np.array([fid[g] for g in groups]); return [(np.where(f != i)[0], np.where(f == i)[0]) for i in range(5)]


def dml(d, S, treat, seed=0, learner="lgb", outcome="y"):
    Y = d[outcome].values.astype(float); T = d[treat].values.astype(float)
    if S is None:
        th = Y[T == 1].mean() - Y[T == 0].mean(); return th, np.nan
    Xm = d[S].values.astype(float); ry = np.zeros_like(Y); rt = np.zeros_like(T)
    for a, b in folds(d.site.values, seed):
        if learner == "lgb":
            my = lgb.train(dict(PRM, seed=seed), lgb.Dataset(Xm[a], Y[a]), 300); mt = lgb.train(dict(PRM, seed=seed), lgb.Dataset(Xm[a], T[a]), 300)
            ry[b] = Y[b] - my.predict(Xm[b]); rt[b] = T[b] - np.clip(mt.predict(Xm[b]), 0.01, 0.99)
        else:
            Xa = np.nan_to_num(Xm[a], nan=-999); Xb = np.nan_to_num(Xm[b], nan=-999)
            my = ExtraTreesRegressor(200, min_samples_leaf=50, n_jobs=8, random_state=seed).fit(Xa, Y[a])
            mt = ExtraTreesClassifier(200, min_samples_leaf=50, n_jobs=8, random_state=seed).fit(Xa, T[a])
            ry[b] = Y[b] - my.predict(Xb); rt[b] = T[b] - np.clip(mt.predict_proba(Xb)[:, 1], 0.01, 0.99)
    th = (rt @ ry) / (rt @ rt); eps = ry - th * rt
    sv = d.site.values; meat = sum((rt[sv == s] @ eps[sv == s]) ** 2 for s in np.unique(sv))
    return th, np.sqrt(meat) / (rt @ rt)


D["treat"] = ((D.wk_since_ex >= 1) & (D.wk_since_ex <= MEM)).astype(float)
nest = []
for nm, S in SETS:
    reps = [dml(D, S, "treat", seed=r) for r in range(1 if S is None else 10)]
    th = np.array([r[0] for r in reps]); se = np.array([r[1] for r in reps])
    med = np.median(th); se_m = np.median(np.sqrt(se ** 2 + (th - med) ** 2)) if S is not None else np.nan   # Chernozhukov median aggregation
    nest.append(dict(adjust=nm, theta=med, se=se_m, lo=med - 1.96 * se_m, hi=med + 1.96 * se_m, reps=";".join(f"{t:.5f}" for t in th),
                     n=len(D), n_treat=int(D.treat.sum()), base_long=D.y[D.treat == 0].mean()))
    print(nest[-1]["adjust"], round(med, 4), round(se_m, 4) if S is not None else "", flush=True)
NEST = pd.DataFrame(nest); NEST.to_csv(RD / "dml_nested.csv", index=False)

# robustness (S2)
rob = []
def add(label, th, se, n):
    rob.append(dict(check=label, theta=th, se=se, lo=th - 1.96 * se, hi=th + 1.96 * se, n=n))
th, se = dml(D, SET2, "treat"); add("Main (S2)", th, se, len(D))
P = D.copy(); P["treat_p"] = P.groupby([P.site, P.date.dt.year]).treat.transform(lambda x: rng.permutation(x.values))
th, se = dml(P, SET2, "treat_p"); add("Placebo: treatment permuted within station-year", th, se, len(P))
P = D.copy(); P["rcc"] = rng.normal(size=len(P)); th, se = dml(P, SET2 + ["rcc"], "treat"); add("Random common cause added", th, se, len(P))
P = D[D.date.dt.year <= 2014]; th, se = dml(P, SET2, "treat"); add("Subset 2008–2014", th, se, len(P))
for win in (4, 12):
    P = D.copy(); P["treat_w"] = ((P.wk_since_ex >= 1) & (P.wk_since_ex <= win)).astype(float)
    th, se = dml(P, SET2, "treat_w"); add(f"Relapse window {win} weeks", th, se, len(P))
th, se = dml(D, SET2, "treat", learner="et"); add("Extra-trees nuisance models", th, se, len(D))
# hysteresis 10% and per-indicator definitions joined to landmark covariates
def joined(Sr, ind):
    d = Sr[(Sr.ind == ind) & (Sr.cur == 0) & Sr.nxt.notna()][["site", "date", "nxt", "since_brk"]]
    J = D.drop(columns=["y", "treat"]).merge(d, on=["site", "date"], how="inner")
    J["y_i"] = J.nxt.astype(float); J["treat_i"] = ((J.since_brk >= 1) & (J.since_brk <= MEM)).astype(float)
    return J
J = joined(Sb10, "any"); th, se = dml(J, SET2, "treat_i", outcome="y_i"); add("Hysteresis band 10%", th, se, len(J))
for ind in ["NH4N", "DO", "CODMn"]:
    J = joined(S0, ind); th, se = dml(J, SET2, "treat_i", outcome="y_i"); add(f"Indicator: {ind}", th, se, len(J))
ROB = pd.DataFrame(rob); ROB.to_csv(RD / "dml_robustness.csv", index=False); print(ROB.round(4).to_string(), flush=True)

# DR-learner CATE (S2 nuisance), smoothed over moderators
Y = D.y.values; T = D.treat.values; Xm = D[SET2].values.astype(float)
mu0 = np.zeros(len(D)); mu1 = np.zeros(len(D)); e = np.zeros(len(D))
for a, b in folds(D.site.values, 0):
    a1 = a[T[a] == 1]; a0 = a[T[a] == 0]
    mu1[b] = lgb.train(PRM, lgb.Dataset(Xm[a1], Y[a1]), 300).predict(Xm[b])
    mu0[b] = lgb.train(PRM, lgb.Dataset(Xm[a0], Y[a0]), 300).predict(Xm[b])
    e[b] = np.clip(lgb.train(PRM, lgb.Dataset(Xm[a], T[a]), 300).predict(Xm[b]), 0.02, 0.98)
phi = mu1 - mu0 + T * (Y - mu1) / e - (1 - T) * (Y - mu0) / (1 - e)
D["phi"] = phi
cate = []
for mod, bins in [("ex_rate52", [0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0]), ("margin_min", [-np.inf, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, np.inf])]:
    d = D[[mod, "phi", "site"]].dropna(); d["bin"] = pd.cut(d[mod], bins, include_lowest=True)
    for bn, g in d.groupby("bin", observed=True):
        if len(g) < 200: continue
        sites = g.site.unique(); sm = g.groupby("site").phi.agg(["sum", "count"])
        bs = [sm.loc[rng.choice(sites, len(sites))].pipe(lambda t: t["sum"].sum() / t["count"].sum()) for _ in range(300)]
        cate.append(dict(moderator=mod, bin=str(bn), x=g[mod].median(), cate=g.phi.mean(), lo=np.quantile(bs, 0.025), hi=np.quantile(bs, 0.975), n=len(g)))
CATE = pd.DataFrame(cate); CATE.to_csv(RD / "dr_cate.csv", index=False); print(CATE.round(4).to_string())
pd.DataFrame(dict(e=e, treat=T)).to_csv(RD / "propensity_S2.csv", index=False)
print("overlap: propensity quantiles", np.quantile(e, [0.01, 0.05, 0.5, 0.95, 0.99]).round(3))
