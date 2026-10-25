"""Forecast-then-threshold baseline: direct multi-horizon concentration forecasts (LightGBM) -> breach risk.
Reports concentration NSE (the metric used by forecasting papers on this network) next to breach-warning skill."""
import sys, numpy as np, pandas as pd, lightgbm as lgb
from pathlib import Path
from scipy.stats import norm
sys.path.insert(0, str(Path(__file__).parent))
from survlib import *

OUT = Path(__file__).resolve().parents[1] / "results"
L = pd.read_parquet(OUT / "landmarks.parquet").reset_index(drop=True)
w = pd.read_parquet(OUT / "cnemc_weekly.parquet")
FC = feat_cols(L)
s0 = L.state.values.astype(int)
X = np.column_stack([L[FC].values.astype(float), s0])
tr = (L.date <= "2013-09-30").values
va = ((L.date >= "2014-01-01") & (L.date <= "2014-09-30")).values
te = (L.date >= "2015-01-01").values

# future concentrations Y[v][i,k]
targets = {"NH4N": (lambda x: np.log1p(x), np.log1p(1.0), +1), "DO": (lambda x: x, 5.0, -1), "CODMn": (lambda x: np.log1p(x), np.log1p(6.0), +1)}
Y = {v: np.full((len(L), K + 1), np.nan) for v in targets}
for s, g in w.groupby("site"):
    g = g.set_index("date").sort_index(); idx = pd.date_range(g.index.min(), g.index.max(), freq="W-MON"); g = g.reindex(idx)
    pos = pd.Series(np.arange(len(idx)), index=idx); li = np.where(L.site.values == s)[0]
    p = pos.reindex(L.date.values[li]).values.astype(int)
    for v, (f, _, _) in targets.items():
        sv = np.concatenate([f(g[v].values), np.full(K, np.nan)])
        Y[v][li] = np.stack([sv[p + k] for k in range(K + 1)], 1)

P = {v: np.zeros((te.sum(), K)) for v in targets}; SIG = {}; nse = []
for v, (f, thr, sgn) in targets.items():
    for k in range(1, K + 1):
        yt, yv, ye = Y[v][tr, k], Y[v][va, k], Y[v][te, k]
        mt, mv = ~np.isnan(yt), ~np.isnan(yv)
        m = lgb.train(dict(LGB_PARAMS, objective="regression"), lgb.Dataset(X[tr][mt], yt[mt]), 3000,
                      valid_sets=[lgb.Dataset(X[va][mv], yv[mv])], callbacks=[lgb.early_stopping(100, verbose=False)])
        pe = m.predict(X[te]); pv = m.predict(X[va][mv])
        sig = np.std(yv[mv] - pv); SIG[(v, k)] = sig
        me = ~np.isnan(ye)
        nse.append(dict(ind=v, k=k, NSE=1 - np.sum((ye[me] - pe[me]) ** 2) / np.sum((ye[me] - ye[me].mean()) ** 2)))
        z = (thr - pe) / sig
        P[v][:, k - 1] = norm.sf(z) if sgn > 0 else norm.cdf(z)
    print(v, "NSE k=1:", round(nse[-K]["NSE"], 3), "k=4:", round(nse[-K + 3]["NSE"], 3), "k=12:", round(nse[-1]["NSE"], 3), flush=True)
pd.DataFrame(nse).to_csv(OUT / "forecast_nse.csv", index=False)
pweek = 1 - np.prod([1 - P[v] for v in targets], axis=0)        # weekly breach prob (indicators independent)
Lte = L[te].reset_index(drop=True)
rows = []
for trans, st in [("onset", 0), ("recovery", 1)]:
    mk = s0[te] == st
    pw = pweek[mk] if st == 0 else 1 - pweek[mk]              # recovery = first compliant week
    variants = {"FT-indep": 1 - np.cumprod(1 - pw, 1), "FT-max": np.maximum.accumulate(pw, 1)}
    for nm, F in variants.items():
        rows.append(dict(trans=trans, model=nm, **eval_ipcw(F, Lte[mk])))
R = pd.DataFrame(rows); R.to_csv(OUT / "forecast_threshold_eval.csv", index=False)
print(R[["trans", "model", "IBS", "AUCw@1", "AUCw@4", "AUCw@12", "ObsIPCW@12", "Pred@12"]].round(4).to_string())
np.save(OUT / "ft_pweek_test.npy", pweek)
