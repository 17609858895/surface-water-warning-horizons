"""Per-indicator first-passage models (Fig. S6): breach defined by NH4-N, DO or CODMn alone (Class III).
Same landmarks, features (base + cross-fitted forecast features) and temporal split as the composite endpoint.
Members: DH-GBM+FT, XGB-AFT+FT, DH-NN+FT; stack by censored likelihood; baselines: station KM, population KM.
Evaluation: year-stratified IPCW, 2015-2016 (all horizons) and 2017-2018 (<=4 weeks)."""
import sys, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from survlib import *
from nnhazard import NNHazard
import torch
torch.set_num_threads(6)
LGB_PARAMS["num_threads"] = 6

OUT = Path(__file__).resolve().parents[1] / "results"; PD = Path(__file__).resolve().parents[1] / "results" / "per_indicator_corrected"; PD.mkdir(exist_ok=True)
L0 = pd.read_parquet(OUT / "landmarks.parquet").reset_index(drop=True); FF = np.load(OUT / "forecast_features.npy")
w = pd.read_parquet(OUT / "cnemc_weekly.parquet")
FC = feat_cols(L0)
SPL = {"tr": (L0.date <= "2013-09-30").values, "va": ((L0.date >= "2014-01-01") & (L0.date <= "2014-09-30")).values,
       "te": (L0.date >= "2015-01-01").values}
DEF = {"NH4N": lambda g: g.NH4N > 1.0, "DO": lambda g: g.DO < 5.0, "CODMn": lambda g: g.CODMn > 6.0}
rows = []
for ind, f in DEF.items():
    Oi = np.full((len(L0), K + 1), np.nan)
    for s, g in w.groupby("site"):
        g = g.set_index("date").sort_index(); idx = pd.date_range(g.index.min(), g.index.max(), freq="W-MON"); g = g.reindex(idx)
        ob = g[["NH4N", "DO", "CODMn"]].notna().all(axis=1)
        st = f(g).astype(float).where(ob).values; sv = np.concatenate([st, np.full(K, np.nan)])
        pos = pd.Series(np.arange(len(idx)), index=idx); li = np.where(L0.site.values == s)[0]; p = pos.reindex(L0.date.values[li]).values.astype(int)
        Oi[li] = np.stack([sv[p + k] for k in range(K + 1)], 1)
    s0 = Oi[:, 0].astype(int)
    T = np.full(len(L0), K); E = np.zeros(len(L0), int); C = np.full(len(L0), K)
    for i in range(len(L0)):
        missing = np.flatnonzero(np.isnan(Oi[i, 1:]))
        C[i] = int(missing[0]) if len(missing) else K
        for k in range(1, K + 1):
            o = Oi[i, k]
            if np.isnan(o): T[i] = k - 1; break
            if o != s0[i]: T[i] = k; E[i] = 1; break
    L = L0.copy(); L["T"] = T; L["E"] = E; L["C"] = C; L["state"] = s0
    X = np.column_stack([L[FC].values.astype(float), s0]); XH = np.column_stack([X, FF]); ws, wc = L.woy_sin.values, L.woy_cos.values
    for trans, st in [("onset", 0), ("recovery", 1)]:
        m = {k: v & (s0 == st) for k, v in SPL.items()}
        Ls = {k: L[v].reset_index(drop=True) for k, v in m.items()}; Ws = {k: (ws[v], wc[v]) for k, v in m.items()}
        H = {}
        g_ = LGBHazard().fit(Ls["tr"], XH[m["tr"]], Ws["tr"], Ls["va"], XH[m["va"]], Ws["va"]); H["DH-GBM+FT"] = {k: g_.hazard(XH[m[k]], *Ws[k]) for k in ("va", "te")}
        a_ = XGBAFT().fit(Ls["tr"], XH[m["tr"]], Ws["tr"], Ls["va"], XH[m["va"]], Ws["va"]); H["XGB-AFT+FT"] = {k: a_.hazard(XH[m[k]], *Ws[k]) for k in ("va", "te")}
        n_ = NNHazard().fit(Ls["tr"], XH[m["tr"]], Ws["tr"], Ls["va"], XH[m["va"]], Ws["va"]); H["DH-NN+FT"] = {k: n_.hazard(XH[m[k]]) for k in ("va", "te")}
        mem = list(H); wt = stack_weights([H[q]["va"] for q in mem], Ls["va"])
        H["BRR-SL"] = {k: sum(wi * H[q][k] for wi, q in zip(wt, mem)) for k in ("va", "te")}
        pp = person_period(Ls["tr"], X[m["tr"]]); hk = np.array([pp[2][pp[1] == k].mean() if (pp[1] == k).any() else 0 for k in range(1, K + 1)])
        df = pd.DataFrame(dict(site=Ls["tr"].site.values[pp[3]], k=pp[1], y=pp[2])).groupby(["site", "k"]).y.agg(["sum", "count"]).reset_index()
        df["h"] = (df["sum"] + 20 * hk[df.k - 1]) / (df["count"] + 20); tab = df.pivot(index="site", columns="k", values="h")
        def site_h(v):
            out = np.tile(hk, (len(v), 1)); mm = v.site.isin(tab.index).values
            out[mm] = tab.reindex(v.site[mm]).fillna(pd.Series(hk, index=range(1, K + 1))).values; return out
        H["Station KM"] = {"te": site_h(Ls["te"])}; H["Population KM"] = {"te": np.tile(hk, (len(Ls["te"]), 1))}
        yr = Ls["te"].date.dt.year.values
        for nm, hh in H.items():
            F = 1 - surv_from_h(hh["te"])
            for per, msk in [("2015-2016", yr <= 2016), ("2017-2018", yr >= 2017)]:
                r = eval_ipcw_strat(F[msk], Ls["te"][msk], yr[msk], horizons=(1, 2, 4, 8, 12))
                rows.append(dict(ind=ind, trans=trans, model=nm, period=per, n_te=int(msk.sum()), events=int(Ls["te"].E.values[msk].sum()),
                                 NLL=nll_rc(hh["te"][msk], Ls["te"][msk]), **r))
        pd.to_pickle(dict(H=H,Lte=Ls["te"]), PD / f"hazards_{ind}_{trans}.pkl")
        pd.DataFrame(rows).to_csv(PD / "per_indicator_metrics.csv",index=False)
        print(ind, trans, "stack", dict(zip(mem, np.round(wt, 2))), flush=True)
R = pd.DataFrame(rows); R.to_csv(PD / "per_indicator_metrics.csv", index=False)
print(R[R.period == "2015-2016"][["ind", "trans", "model", "n_te", "events", "IBS", "AUCw@1", "AUCw@4", "AUCw@12", "Pred@12", "ObsIPCW@12"]].round(4).to_string())
