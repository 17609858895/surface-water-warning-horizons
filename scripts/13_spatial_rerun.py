"""Spatiotemporal hold-out (Fig. 4d, Fig. S7): 5 station-grouped folds; forecast features recomputed inside each fold
(in-fold training stations only); members + BRR-SL stack fit on in-fold stations (train <=2013-09, val 2014);
evaluation on held-out stations in 2015-2016 (all horizons) and 2017-2018 (<=4 weeks), year-stratified IPCW."""
import sys, numpy as np, pandas as pd, lightgbm as lgb
from pathlib import Path
from scipy.stats import norm
from sklearn.model_selection import GroupKFold
sys.path.insert(0, str(Path(__file__).parent))
from survlib import *
from nnhazard import NNHazard
from tristate import TriStateBoost, start_state

OUT = Path(__file__).resolve().parents[1] / "results"; CORE = OUT / "core"
L = pd.read_parquet(OUT / "landmarks.parquet").reset_index(drop=True)
O = np.load(OUT / "obs_windows.npy"); w = pd.read_parquet(OUT / "cnemc_weekly.parquet")
FC = feat_cols(L); state = L.state.values.astype(int)
X = np.column_stack([L[FC].values.astype(float), state]); ws, wc = L.woy_sin.values, L.woy_cos.values
tr = (L.date <= "2013-09-30").values; va = ((L.date >= "2014-01-01") & (L.date <= "2014-09-30")).values; te = (L.date >= "2015-01-01").values
s3 = start_state(state, L.wk_since_ex.values)
HZ = (1, 2, 4, 8, 12)
targets = {"NH4N": (np.log1p, np.log1p(1.0), +1), "DO": (lambda x: x, 5.0, -1), "CODMn": (np.log1p, np.log1p(6.0), +1)}
Y = {v: np.full((len(L), K + 1), np.nan) for v in targets}
for s, g in w.groupby("site"):
    g = g.set_index("date").sort_index(); idx = pd.date_range(g.index.min(), g.index.max(), freq="W-MON"); g = g.reindex(idx)
    pos = pd.Series(np.arange(len(idx)), index=idx); li = np.where(L.site.values == s)[0]; p = pos.reindex(L.date.values[li]).values.astype(int)
    for v, (f, _, _) in targets.items():
        sv = np.concatenate([f(g[v].values), np.full(K, np.nan)]); Y[v][li] = np.stack([sv[p + k] for k in range(K + 1)], 1)
year = np.clip(L.date.dt.year.values, 2008, None); PRM = dict(LGB_PARAMS, objective="regression")


def forecast_features(train_mask, apply_mask):
    FF = np.full((len(L), len(targets) * len(HZ) * 2 + len(HZ)), np.nan); col = 0; pe_ = {}
    for v, (f, thr, sgn) in targets.items():
        for k in HZ:
            y = Y[v][:, k]; mu = np.full(len(L), np.nan)
            for yr in np.unique(year[train_mask]):
                fit = train_mask & (year != yr) & ~np.isnan(y); hold = train_mask & (year == yr)
                mu[hold] = lgb.train(PRM, lgb.Dataset(X[fit], y[fit]), 400).predict(X[hold])
            fit = train_mask & ~np.isnan(y)
            mu[apply_mask] = lgb.train(PRM, lgb.Dataset(X[fit], y[fit]), 400).predict(X[apply_mask])
            sig = np.nanstd((y - mu)[train_mask]); z = (thr - mu) / sig
            pe = norm.sf(z) if sgn > 0 else norm.cdf(z)
            FF[:, col] = mu; FF[:, col + 1] = pe; col += 2; pe_[(v, k)] = pe
    for j, k in enumerate(HZ):
        FF[:, col + j] = 1 - np.prod([1 - pe_[(v, k)] for v in targets], axis=0)
    return FF


def h_from_F(F):
    S = 1 - F; Sp = np.column_stack([np.ones(len(S)), S[:, :-1]]); return np.clip(1 - S / np.maximum(Sp, 1e-12), 1e-6, 1 - 1e-6)


sites = np.array(sorted(L.site.unique())); rows = []
for fold, (a, b) in enumerate(GroupKFold(5).split(sites, groups=sites)):
    inm = L.site.isin(sites[a]).values; outm = L.site.isin(sites[b]).values
    FF = forecast_features(tr & inm, (va & inm) | (te & outm)); XH = np.column_stack([X, FF])
    m_tr, m_va, m_te = tr & inm, va & inm, te & outm
    tb = TriStateBoost().fit(O[m_tr], s3[m_tr], XH[m_tr], (ws[m_tr], wc[m_tr]), O[m_va], s3[m_va], XH[m_va], (ws[m_va], wc[m_va]))
    TS = {}
    for k, mk in [("va", m_va), ("te", m_te)]:
        Fon, Frc, _ = tb.propagate(tb.all_hazards(XH[mk], ws[mk], wc[mk]), s3[mk]); TS[k] = (Fon, Frc, state[mk])
    for trans, st in [("onset", 0), ("recovery", 1)]:
        mm = {k: v & (state == st) for k, v in [("tr", m_tr), ("va", m_va), ("te", m_te)]}
        Ls = {k: L[v].reset_index(drop=True) for k, v in mm.items()}; Wd = {k: (ws[v], wc[v]) for k, v in mm.items()}
        Hh = {}
        g = LGBHazard().fit(Ls["tr"], XH[mm["tr"]], Wd["tr"], Ls["va"], XH[mm["va"]], Wd["va"]); Hh["DH-GBM+FT"] = {k: g.hazard(XH[mm[k]], *Wd[k]) for k in ("va", "te")}
        g0 = LGBHazard().fit(Ls["tr"], X[mm["tr"]], Wd["tr"], Ls["va"], X[mm["va"]], Wd["va"]); Hh["DH-GBM"] = {k: g0.hazard(X[mm[k]], *Wd[k]) for k in ("va", "te")}
        x_ = XGBAFT().fit(Ls["tr"], XH[mm["tr"]], Wd["tr"], Ls["va"], XH[mm["va"]], Wd["va"]); Hh["XGB-AFT+FT"] = {k: x_.hazard(XH[mm[k]], *Wd[k]) for k in ("va", "te")}
        n_ = NNHazard().fit(Ls["tr"], XH[mm["tr"]], Wd["tr"], Ls["va"], XH[mm["va"]], Wd["va"]); Hh["DH-NN+FT"] = {k: n_.hazard(XH[mm[k]]) for k in ("va", "te")}
        Hh["3S-Boost+FT"] = {k: h_from_F((TS[k][0] if st == 0 else TS[k][1])[TS[k][2] == st]) for k in ("va", "te")}
        pp = person_period(Ls["tr"], X[mm["tr"]]); hk = np.array([pp[2][pp[1] == k].mean() for k in range(1, K + 1)])
        Hh["Population KM"] = {k: np.tile(hk, (mm[k].sum(), 1)) for k in ("va", "te")}
        mem = ["DH-GBM+FT", "XGB-AFT+FT", "DH-NN+FT", "3S-Boost+FT"]
        wt = stack_weights([Hh[q]["va"] for q in mem], Ls["va"])
        Hh["BRR-SL"] = {k: sum(wi * Hh[q][k] for wi, q in zip(wt, mem)) for k in ("va", "te")}
        yr = Ls["te"].date.dt.year.values
        for nm, hh in Hh.items():
            F = 1 - surv_from_h(hh["te"])
            for per, msk in [("2015-2016", yr <= 2016), ("2017-2018", yr >= 2017)]:
                r = eval_ipcw_strat(F[msk], Ls["te"][msk], yr[msk], horizons=HZ)
                rows.append(dict(fold=fold, trans=trans, model=nm, period=per, n_sites=len(b), **r))
    print("fold", fold, "done", flush=True)
SP = pd.DataFrame(rows); SP.to_csv(CORE / "spatial_holdout.csv", index=False)
print(SP.groupby(["trans", "period", "model"])[["IBS", "AUCw@1", "AUCw@4"]].agg(["mean", "std"]).round(4).to_string())
