"""Station-level aggregates for the map panels and headline numbers.
Headlines: share of breach onsets that are relapses (onset from R, i.e. <=8 weeks after a recovery), by year;
Kaplan-Meier probability of re-breach within 4/8/12 weeks after a recovery (censored at monitoring gaps).
Station maps: empirical-Bayes relapse probability; S1-adjusted relapse excess (DR-learner pseudo-outcomes, shrunk);
4-week Brier skill of BRR-SL vs population KM (2015-2016); surveillance priority (share of weeks in the weekly top 20%
of 4-week risk); median Mondrian safe weeks."""
import sys, numpy as np, pandas as pd, lightgbm as lgb
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from survlib import *

OUT = Path(__file__).resolve().parents[1] / "results"; SP = OUT / "spatial"; SP.mkdir(exist_ok=True)
w = pd.read_parquet(OUT / "cnemc_weekly.parquet"); w = w[w.date.dt.year >= 2008]
coords = w.groupby("site")[["lon", "lat"]].first()
MEM = 8; rng = np.random.default_rng(24)

# ---------- headline numbers ----------
on_rows, rec_T = [], []
for s, g in w.groupby("site"):
    g = g.set_index("date").sort_index(); idx = pd.date_range(g.index.min(), g.index.max(), freq="W-MON"); g = g.reindex(idx)
    ob = g[["NH4N", "DO", "CODMn"]].notna().all(axis=1).values
    st = np.where(ob, ((g.NH4N > 1) | (g.DO < 5) | (g.CODMn > 6)).values.astype(float), np.nan)
    last_brk = -999
    for i in range(1, len(st)):
        if np.isnan(st[i]) or np.isnan(st[i - 1]):
            if not np.isnan(st[i]) and st[i] == 1: last_brk = i
            continue
        if st[i] == 1 and st[i - 1] == 0:
            on_rows.append(dict(site=s, year=idx[i].year, relapse=int(1 <= (i - 1) - last_brk <= MEM)))
        if st[i] == 0 and st[i - 1] == 1:          # recovery at i: weeks until next breach
            k = 1; ev = 0
            while i + k < len(st):
                if np.isnan(st[i + k]): break
                if st[i + k] == 1: ev = 1; break
                k += 1
            rec_T.append((k if ev else k - 1, ev))
        if st[i] == 1: last_brk = i
ON = pd.DataFrame(on_rows)
share = ON.groupby("year").relapse.agg(["mean", "size"]).rename(columns={"mean": "relapse_share", "size": "n_onsets"})
sites = ON.site.unique(); tab = ON.groupby("site").relapse.agg(["sum", "count"])
bs = [tab.loc[rng.choice(sites, len(sites))].pipe(lambda t: t["sum"].sum() / t["count"].sum()) for _ in range(500)]
T = np.array([t for t, e in rec_T]); E = np.array([e for t, e in rec_T]); S = 1.0; km = {}
for t in range(1, 27):
    at = (T >= t).sum(); d = ((T == t) & (E == 1)).sum(); S *= (1 - d / at) if at else 1; km[t] = 1 - S
H = dict(relapse_share_all=ON.relapse.mean(), relapse_share_lo=np.quantile(bs, 0.025), relapse_share_hi=np.quantile(bs, 0.975),
         n_onsets=len(ON), n_recoveries=len(T), rebreach_4wk=km[4], rebreach_8wk=km[8], rebreach_12wk=km[12], rebreach_26wk=km[26])
pd.Series(H).to_csv(SP / "headlines.csv"); share.to_csv(SP / "relapse_share_by_year.csv")
print(pd.Series(H).round(3).to_string()); print(share.round(3).to_string(), flush=True)

# ---------- station aggregates ----------
EB = pd.read_csv(OUT / "check_station_relapse_eb.csv", index_col=0)
L = pd.read_parquet(OUT / "landmarks.parquet").reset_index(drop=True); O = np.load(OUT / "obs_windows.npy"); state = L.state.values.astype(int)
# (a) DR pseudo-outcomes under S1, station means shrunk toward the global mean
D = L[(L.state == 0) & ~np.isnan(O[:, 1]) & (L.date.dt.year >= 2008)].copy(); D["y"] = O[D.index, 1]
clim = []
for s, g in D.groupby("site"):
    g = g.sort_values("date")
    for v in ["log_NH4N_0", "DO_0", "log_CODMn_0"]: g[f"clim_{v}"] = g[v].shift(52).expanding(min_periods=10).mean()
    clim.append(g)
D = pd.concat(clim).sort_values(["site", "date"]).reset_index(drop=True)
S1 = ["woy_sin", "woy_cos", "lon", "lat", "clim_log_NH4N_0", "clim_DO_0", "clim_log_CODMn_0", "ex_rate52", "ex_rate52_NH4N",
      "ex_rate52_DO", "ex_rate52_COD", "log_NH4N_m52", "DO_m52", "log_CODMn_m52"]
PRM = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8,
           bagging_freq=1, verbose=-1, num_threads=16, seed=3)
D["T"] = ((D.wk_since_ex >= 1) & (D.wk_since_ex <= MEM)).astype(float)
X = D[S1].values.astype(float); Y = D.y.values; Tt = D["T"].values
us = np.unique(D.site.values); fid = dict(zip(np.random.default_rng(0).permutation(us), np.arange(len(us)) % 5)); f = np.array([fid[s] for s in D.site.values])
mu0, mu1, e = np.zeros(len(D)), np.zeros(len(D)), np.zeros(len(D))
for i in range(5):
    a, b = np.where(f != i)[0], np.where(f == i)[0]; a1, a0 = a[Tt[a] == 1], a[Tt[a] == 0]
    mu1[b] = lgb.train(PRM, lgb.Dataset(X[a1], Y[a1]), 300).predict(X[b]); mu0[b] = lgb.train(PRM, lgb.Dataset(X[a0], Y[a0]), 300).predict(X[b])
    e[b] = np.clip(lgb.train(PRM, lgb.Dataset(X[a], Tt[a]), 300).predict(X[b]), 0.05, 0.95)
D["phi"] = mu1 - mu0 + Tt * (Y - mu1) / e - (1 - Tt) * (Y - mu0) / (1 - e)
g = D.groupby("site").agg(phi=("phi", "mean"), n=("phi", "size"), n_treat=("T", "sum"))
gm = D.phi.mean(); k0 = 200
g["cate_eb"] = (g.phi * g.n + gm * k0) / (g.n + k0)
# (b) 4-week Brier skill on the 2015-2016 test (known labels)
Hz = pd.read_pickle(OUT / "core" / "hazards.pkl")["onset"]
te = (L.date >= "2015-01-01").values & (state == 0); Lte = L[te].reset_index(drop=True)
F4 = (1 - surv_from_h(Hz["BRR-SL"]["te"]))[:, 3]; P4 = (1 - surv_from_h(Hz["Population KM"]["te"]))[:, 3]
y4, kn = horizon_labels(Lte, 4); dense = (Lte.date.dt.year.values <= 2016) & kn
B = pd.DataFrame(dict(site=Lte.site.values, f=F4, p=P4, y=y4))[dense]
bs_ = B.assign(bm=(B.f - B.y) ** 2, br=(B.p - B.y) ** 2).groupby("site").agg(bm=("bm", "mean"), br=("br", "mean"), n=("y", "size"), ev=("y", "sum"))
bs_["bss"] = 1 - bs_.bm / bs_.br
# (c) surveillance priority: share of weeks in the weekly top 20 % of 4-week risk (2015-2016, compliant landmarks)
Q = pd.DataFrame(dict(site=Lte.site.values, date=Lte.date.values, f=F4))[Lte.date.dt.year.values <= 2016]
Q["rank_pct"] = Q.groupby("date").f.rank(pct=True, ascending=True); Q["top20"] = Q.rank_pct > 0.8
pri = Q.groupby("site").agg(priority=("top20", "mean"), weeks=("top20", "size"), mean_risk4=("f", "mean"))
onsets = ON[ON.year.between(2015, 2016)].groupby("site").size().rename("onsets_2015_16")
# (d) Mondrian safe weeks
SW = pd.read_csv(OUT / "fig7" / "safe_weeks_by_landmark.csv"); swm = SW.groupby("site").safe_mondrian.median().rename("safe_weeks_median")
A = coords.join(EB[["relapse", "onset", "n_R"]]).join(g[["cate_eb", "n", "n_treat"]]).join(bs_[["bss", "n", "ev"]].rename(columns={"n": "n_bss", "ev": "ev_bss"})) \
    .join(pri).join(onsets).join(swm)
occ = w[(w.date.dt.year <= 2014)].groupby("site").ex_any.mean().rename("breach_share_0814"); A = A.join(occ)
A.to_csv(SP / "station_aggregates.csv"); print(A.describe().round(3).to_string())
