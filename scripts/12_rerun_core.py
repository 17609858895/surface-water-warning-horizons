"""Core re-run for Figs 3-4 and Table 1 (temporal split; year-stratified IPCW evaluation).
Members (each with base features X and forecast-augmented XH): DH-GBM, XGB-AFT, DH-NN (first-passage, per transition)
and 3S-Boost (joint three-state with R_1..R_8 memory; start state from weeks since last breach).
BRR-SL = censored-likelihood convex stack of the four XH members. Ablations: -FT, -stacking, -R state, -covariates."""
import sys, json, numpy as np, pandas as pd, lightgbm as lgb
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from survlib import *
from nnhazard import NNHazard
from tristate import TriStateBoost, start_state

OUT = Path(__file__).resolve().parents[1] / "results"; CORE = OUT / "core"; CORE.mkdir(exist_ok=True)
rng = np.random.default_rng(12)
L = pd.read_parquet(OUT / "landmarks.parquet").reset_index(drop=True)
O = np.load(OUT / "obs_windows.npy"); FF = np.load(OUT / "forecast_features.npy")
FC = feat_cols(L); state = L.state.values.astype(int)
X = np.column_stack([L[FC].values.astype(float), state]); XH = np.column_stack([X, FF])
ws, wc = L.woy_sin.values, L.woy_cos.values
SPL = {"tr": (L.date <= "2013-09-30").values,
       "va": ((L.date >= "2014-01-01") & (L.date <= "2014-09-30")).values,
       "te": (L.date >= "2015-01-01").values}
TR = {"onset": 0, "recovery": 1}
s3 = start_state(state, L.wk_since_ex.values); s2 = start_state(state, L.wk_since_ex.values, two_state=True)
print("start states (A,B,R):", (s3 == 0).sum(), (s3 == 1).sum(), (s3 >= 2).sum(), flush=True)

H = {t: {} for t in TR}           # H[trans][model][split] -> n x K hazards
def h_from_F(F):
    S = 1 - F; Sp = np.column_stack([np.ones(len(S)), S[:, :-1]])
    return np.clip(1 - S / np.maximum(Sp, 1e-12), 1e-6, 1 - 1e-6)

for tag, XX in [("", X), ("+FT", XH)]:
    for trans, st in TR.items():
        m = {k: v & (state == st) for k, v in SPL.items()}
        Ls = {k: L[v].reset_index(drop=True) for k, v in m.items()}; Ws = {k: (ws[v], wc[v]) for k, v in m.items()}
        g = LGBHazard().fit(Ls["tr"], XX[m["tr"]], Ws["tr"], Ls["va"], XX[m["va"]], Ws["va"])
        H[trans]["DH-GBM" + tag] = {k: g.hazard(XX[m[k]], *Ws[k]) for k in ("va", "te")}
        a = XGBAFT().fit(Ls["tr"], XX[m["tr"]], Ws["tr"], Ls["va"], XX[m["va"]], Ws["va"])
        H[trans]["XGB-AFT" + tag] = {k: a.hazard(XX[m[k]], *Ws[k]) for k in ("va", "te")}
        nn_ = NNHazard().fit(Ls["tr"], XX[m["tr"]], Ws["tr"], Ls["va"], XX[m["va"]], Ws["va"])
        H[trans]["DH-NN" + tag] = {k: nn_.hazard(XX[m[k]]) for k in ("va", "te")}
        print(tag or "base", trans, "first-passage members done", flush=True)
    for name, s0, two in [("3S-Boost" + tag, s3, False), ("2S-Boost" + tag, s2, True)]:
        tb = TriStateBoost(two_state=two).fit(O[SPL["tr"]], s0[SPL["tr"]], XX[SPL["tr"]], (ws[SPL["tr"]], wc[SPL["tr"]]),
                                              O[SPL["va"]], s0[SPL["va"]], XX[SPL["va"]], (ws[SPL["va"]], wc[SPL["va"]]))
        for k in ("va", "te"):
            mk = SPL[k]; Hs = tb.all_hazards(XX[mk], ws[mk], wc[mk]); Fon, Frc, occ = tb.propagate(Hs, s0[mk])
            for trans, st in TR.items():
                sel = state[mk] == st
                H[trans].setdefault(name, {})[k] = h_from_F((Fon if st == 0 else Frc)[sel])
            if k == "te" and name == "3S-Boost+FT":
                np.save(CORE / "occ_3s_test.npy", occ)
        if name == "3S-Boost+FT":
            (CORE / "tristate_FT.txt").write_text(tb.bst.model_to_string(), encoding="utf-8")
        print(name, "trees", tb.bst.best_iteration, flush=True)

# baselines: station KM, population KM, complete-case horizon classifiers (XH), forecast-then-threshold
pw_te = np.load(OUT / "ft_pweek_test.npy")
CCv = {}
for trans, st in TR.items():
    m = {k: v & (state == st) for k, v in SPL.items()}; Ls = {k: L[v].reset_index(drop=True) for k, v in m.items()}
    XX, idx_k, y, idx = None, *person_period(Ls["tr"], X[m["tr"]])[1:]
    hk = np.array([y[idx_k == k].mean() for k in range(1, K + 1)])
    df = pd.DataFrame(dict(site=Ls["tr"].site.values[idx], k=idx_k, y=y)).groupby(["site", "k"]).y.agg(["sum", "count"]).reset_index()
    df["h"] = (df["sum"] + 20 * hk[df.k - 1]) / (df["count"] + 20); tab = df.pivot(index="site", columns="k", values="h")
    def site_h(v):
        out = np.tile(hk, (len(v), 1)); mm = v.site.isin(tab.index).values
        out[mm] = tab.reindex(v.site[mm]).fillna(pd.Series(hk, index=range(1, K + 1))).values; return out
    H[trans]["Station KM"] = {k: site_h(Ls[k]) for k in ("va", "te")}
    H[trans]["Population KM"] = {k: np.tile(hk, (len(Ls[k]), 1)) for k in ("va", "te")}
    Fcc = np.zeros((len(Ls["te"]), K))
    for hz in range(1, K + 1):
        ytr, ktr = horizon_labels(Ls["tr"], hz); yva, kva = horizon_labels(Ls["va"], hz)
        mm = lgb.train(LGB_PARAMS, lgb.Dataset(XH[m["tr"]][ktr], ytr[ktr]), 3000, valid_sets=[lgb.Dataset(XH[m["va"]][kva], yva[kva])],
                       callbacks=[lgb.early_stopping(100, verbose=False)])
        Fcc[:, hz - 1] = mm.predict(XH[m["te"]], num_iteration=mm.best_iteration)
    CCv[trans] = Fcc
    pw = pw_te[state[SPL["te"]] == st]; pw = pw if st == 0 else 1 - pw
    CCv[trans + "_FTmax"] = np.maximum.accumulate(pw, 1); CCv[trans + "_FTind"] = 1 - np.cumprod(1 - pw, 1)

# stacks
WEIGHTS = {}
for trans, st in TR.items():
    Lva = L[SPL["va"] & (state == st)].reset_index(drop=True)
    for sname, members in [("BRR-SL", ["DH-GBM+FT", "XGB-AFT+FT", "DH-NN+FT", "3S-Boost+FT"]),
                           ("BRR-SL -FT", ["DH-GBM", "XGB-AFT", "DH-NN", "3S-Boost"])]:
        w = stack_weights([H[trans][b]["va"] for b in members], Lva)
        H[trans][sname] = {k: sum(wi * H[trans][b][k] for wi, b in zip(w, members)) for k in ("va", "te")}
        WEIGHTS[f"{trans}|{sname}"] = dict(zip(members, np.round(w, 4).tolist()))
    vnll = {b: nll_rc(H[trans][b]["va"], Lva) for b in ["DH-GBM+FT", "XGB-AFT+FT", "DH-NN+FT", "3S-Boost+FT"]}
    best = min(vnll, key=vnll.get); WEIGHTS[f"{trans}|best_single"] = best
    H[trans]["BRR-SL -stack"] = H[trans][best]
json.dump(WEIGHTS, open(CORE / "stack_weights.json", "w"), indent=1)
print(json.dumps(WEIGHTS, indent=1), flush=True)
pd.to_pickle(H, CORE / "hazards.pkl"); pd.to_pickle(CCv, CORE / "baseline_F_test.pkl")

# ---------------- evaluation ----------------
def F_of(trans, model):
    if model == "Horizon classifiers": return CCv[trans]
    if model == "Forecast-threshold (max)": return CCv[trans + "_FTmax"]
    if model == "Forecast-threshold (indep.)": return CCv[trans + "_FTind"]
    return 1 - surv_from_h(H[trans][model]["te"])

MODELS = ["BRR-SL", "BRR-SL -FT", "BRR-SL -stack", "DH-GBM+FT", "XGB-AFT+FT", "DH-NN+FT", "3S-Boost+FT", "2S-Boost+FT",
          "DH-GBM", "XGB-AFT", "DH-NN", "3S-Boost", "2S-Boost", "Station KM", "Population KM",
          "Forecast-threshold (max)", "Forecast-threshold (indep.)", "Horizon classifiers"]
HZ = (1, 2, 4, 8, 12)
rows, calib, rel = [], [], []
for trans, st in TR.items():
    Lte = L[SPL["te"] & (state == st)].reset_index(drop=True); yr = Lte.date.dt.year.values
    for model in MODELS:
        F = F_of(trans, model)
        nll = nll_rc(H[trans][model]["te"], Lte) if model in H[trans] else np.nan
        for per, msk in [("2015-2016", yr <= 2016), ("2017-2018", yr >= 2017), ("2015-2018", np.ones(len(yr), bool))]:
            r = eval_ipcw_strat(F[msk], Lte[msk], yr[msk], horizons=HZ)
            rows.append(dict(trans=trans, model=model, period=per, NLL=nll_rc(H[trans][model]["te"][msk], Lte[msk]) if model in H[trans] else np.nan, **r))
        if model in ["BRR-SL", "DH-GBM+FT", "Forecast-threshold (max)", "Forecast-threshold (indep.)", "Horizon classifiers", "Station KM"]:
            msk = yr <= 2016
            r = eval_ipcw_strat(F[msk], Lte[msk], yr[msk], horizons=tuple(range(1, K + 1)))
            for k in range(1, K + 1):
                calib.append(dict(trans=trans, model=model, k=k, pred=r[f"Pred@{k}"], obs=r[f"ObsIPCW@{k}"], ratio=r[f"Pred@{k}"] / r[f"ObsIPCW@{k}"]))
    # reliability deciles (stratified IPCW observed within each decile)
    for model in ["BRR-SL", "Forecast-threshold (max)"]:
        F = F_of(trans, model)
        for hz, per_mask in [(4, np.ones(len(yr), bool)), (12, yr <= 2016)]:
            f = F[per_mask, hz - 1]; q = pd.qcut(f, 10, labels=False, duplicates="drop")
            for d in np.unique(q):
                mm = np.where(per_mask)[0][q == d]
                r = eval_ipcw_strat(F[mm], Lte.iloc[mm].reset_index(drop=True), yr[mm], horizons=(hz,), min_G=0.0)
                rel.append(dict(trans=trans, model=model, k=hz, decile=int(d), pred=f[q == d].mean(), obs=r.get(f"ObsIPCW@{hz}", np.nan), n=len(mm)))
R = pd.DataFrame(rows); R.to_csv(CORE / "metrics.csv", index=False)
pd.DataFrame(calib).to_csv(CORE / "calibration_ratio.csv", index=False)
pd.DataFrame(rel).to_csv(CORE / "reliability.csv", index=False)
print(R[R.period == "2015-2016"][["trans", "model", "NLL", "IBS", "AUCw@1", "AUCw@4", "AUCw@12", "Pred@12", "ObsIPCW@12"]].round(4).to_string(), flush=True)

# horizon-classifier incoherence
viol = []
for trans in TR:
    F = CCv[trans]
    for a, b in [(1, 2), (2, 4), (4, 8), (8, 12)]:
        viol.append(dict(trans=trans, pair=f"{a}→{b}", share=float((F[:, b - 1] < F[:, a - 1] - 1e-9).mean())))
pd.DataFrame(viol).to_csv(CORE / "cc_monotonic_violations.csv", index=False)

# ---------------- ablation with station-block bootstrap (2015-2016, IBS over HZ) ----------------
def ibs(F, Lsub, yr):
    T = Lsub["T"].values.astype(int); E = Lsub["E"].values; C = Lsub["C"].values; tot = []
    for hz in HZ:
        w = np.zeros(len(T)); y = ((E == 1) & (T <= hz)).astype(float)
        for s in np.unique(yr):
            m = yr == s; G = np.array([(C[m] >= k).mean() for k in range(K + 1)])
            if G[hz] < 0.05: continue
            ev = m & (E == 1) & (T <= hz); ne = m & ((T > hz) | ((T == hz) & (E == 0)))
            w[ev] = 1 / G[T[ev]]; w[ne] = 1 / G[hz]
        tot.append(np.mean(w * (F[:, hz - 1] - y) ** 2))
    return np.mean(tot)

ABL = {"Forecast features removed": "BRR-SL -FT", "Stacking removed (best single)": "BRR-SL -stack",
       "Relapse state removed (3S→2S)": ("3S-Boost+FT", "2S-Boost+FT"), "Covariates removed (station KM)": "Station KM",
       "Concentration forecast (max)": "Forecast-threshold (max)"}
ab = []
for trans, st in TR.items():
    Lte = L[SPL["te"] & (state == st)].reset_index(drop=True); yr = Lte.date.dt.year.values
    keep = yr <= 2016; Lk = Lte[keep].reset_index(drop=True); yk = yr[keep]
    sites = Lk.site.values; us = np.unique(sites); grp = {s: np.where(sites == s)[0] for s in us}
    Fs = {m: F_of(trans, m)[keep] for m in ["BRR-SL", "BRR-SL -FT", "BRR-SL -stack", "3S-Boost+FT", "2S-Boost+FT", "Station KM", "Forecast-threshold (max)"]}
    for lab, mdl in ABL.items():
        ref, alt = (mdl if isinstance(mdl, tuple) else ("BRR-SL", mdl))
        d0 = ibs(Fs[alt], Lk, yk) - ibs(Fs[ref], Lk, yk); bs = []
        for b in range(200):
            ii = np.concatenate([grp[s] for s in rng.choice(us, len(us))])
            Lb = Lk.iloc[ii].reset_index(drop=True)
            bs.append(ibs(Fs[alt][ii], Lb, yk[ii]) - ibs(Fs[ref][ii], Lb, yk[ii]))
        ab.append(dict(trans=trans, ablation=lab, reference=ref, variant=alt, dIBS=d0, lo=np.quantile(bs, 0.025), hi=np.quantile(bs, 0.975),
                       rel_pct=100 * d0 / ibs(Fs[ref], Lk, yk)))
        print(trans, lab, round(d0, 5), flush=True)
pd.DataFrame(ab).to_csv(CORE / "ablation_bootstrap.csv", index=False)

# ---------------- persistence vs LightGBM NSE at all horizons ----------------
w_ = pd.read_parquet(OUT / "cnemc_weekly.parquet"); te = SPL["te"]; out = []
f = {"NH4N": np.log1p, "DO": lambda x: x, "CODMn": np.log1p}
for v in f:
    Y = np.full((len(L), K + 1), np.nan)
    for s, g in w_.groupby("site"):
        g = g.set_index("date").sort_index(); ix = pd.date_range(g.index.min(), g.index.max(), freq="W-MON"); g = g.reindex(ix)
        pos = pd.Series(np.arange(len(ix)), index=ix); li = np.where(L.site.values == s)[0]; p = pos.reindex(L.date.values[li]).values.astype(int)
        sv = np.concatenate([f[v](g[v].values), np.full(K, np.nan)]); Y[li] = np.stack([sv[p + k] for k in range(K + 1)], 1)
    for k in range(1, K + 1):
        y = Y[te, k]; x0 = Y[te, 0]; m = ~np.isnan(y) & ~np.isnan(x0)
        out.append(dict(ind=v, k=k, NSE_persistence=1 - np.sum((y[m] - x0[m]) ** 2) / np.sum((y[m] - y[m].mean()) ** 2)))
NS = pd.DataFrame(out).merge(pd.read_csv(OUT / "forecast_nse.csv").rename(columns={"NSE": "NSE_LightGBM"}), on=["ind", "k"])
NS.to_csv(CORE / "nse_all_horizons.csv", index=False); print(NS.round(3).to_string())
