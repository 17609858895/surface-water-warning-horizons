"""Shared helpers: discrete-time hazard learning, AFT, neural hazard, stacking, metrics."""
import numpy as np, pandas as pd, lightgbm as lgb, xgboost as xgb
from sklearn.metrics import roc_auc_score
from scipy.stats import norm
from scipy.optimize import minimize
K = 12
NONFEAT = {"site", "date", "state", "T", "E", "C", "trans", "year", "post", "inc"}

def feat_cols(df):
    return [c for c in df.columns if c not in NONFEAT]

def person_period(L, X, kmax=K):
    """Expand landmarks into person-period rows for right-censored discrete hazard."""
    T = L["T"].values.astype(int); E = L["E"].values.astype(int)
    rep = np.clip(T, 0, kmax)
    idx = np.repeat(np.arange(len(L)), rep)
    k = np.concatenate([np.arange(1, r + 1) for r in rep]) if rep.sum() else np.array([], int)
    y = ((k == T[idx]) & (E[idx] == 1)).astype(int)
    XX = X[idx]
    return XX, k, y, idx

def add_k(X, k, woy_sin, woy_cos):
    ang = np.arctan2(woy_sin, woy_cos) + 2 * np.pi * k / 52.18
    return np.column_stack([X, k, np.sin(ang), np.cos(ang)])

def expand_all_k(X, ws, wc, kmax=K):
    n = len(X)
    Xr = np.repeat(X, kmax, axis=0); kr = np.tile(np.arange(1, kmax + 1), n)
    return add_k(Xr, kr, np.repeat(ws, kmax), np.repeat(wc, kmax))

LGB_PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=31, min_data_in_leaf=100,
                  feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0,
                  verbose=-1, num_threads=24, seed=7)

class LGBHazard:
    name = "DH-GBM"
    def __init__(self, params=None, rounds=3000):
        self.p = dict(LGB_PARAMS, **(params or {})); self.rounds = rounds
    def _pp(self, L, X, ws, wc):
        XX, k, y, idx = person_period(L, X)
        return add_k(XX, k, ws[idx], wc[idx]), y
    def fit(self, Ltr, Xtr, wtr, Lva, Xva, wva, sample_weight=None):
        A, y = self._pp(Ltr, Xtr, *wtr); B, yb = self._pp(Lva, Xva, *wva)
        dtr = lgb.Dataset(A, y, weight=sample_weight); dva = lgb.Dataset(B, yb)
        self.m = lgb.train(self.p, dtr, self.rounds, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
        return self
    def hazard(self, X, ws, wc):
        h = self.m.predict(expand_all_k(X, ws, wc), num_iteration=self.m.best_iteration)
        return h.reshape(len(X), K)

class XGBAFT:
    name = "XGB-AFT"
    def __init__(self, dist="logistic", scale=1.0):
        self.dist = dist; self.scale = scale
    def _bounds(self, L):
        T = L["T"].values.astype(float); E = L["E"].values
        lo = np.where(E == 1, np.maximum(T - 1, 0) + 1e-3, np.maximum(T, 1e-3))
        hi = np.where(E == 1, T, np.inf)
        return lo, hi
    def fit(self, Ltr, Xtr, wtr, Lva, Xva, wva):
        keep = ~((Ltr["E"].values == 0) & (Ltr["T"].values == 0))
        lo, hi = self._bounds(Ltr[keep]); d = xgb.DMatrix(Xtr[keep]); d.set_float_info("label_lower_bound", lo); d.set_float_info("label_upper_bound", hi)
        kv = ~((Lva["E"].values == 0) & (Lva["T"].values == 0))
        lv, hv = self._bounds(Lva[kv]); dv = xgb.DMatrix(Xva[kv]); dv.set_float_info("label_lower_bound", lv); dv.set_float_info("label_upper_bound", hv)
        p = dict(objective="survival:aft", eval_metric="aft-nloglik", aft_loss_distribution=self.dist,
                 aft_loss_distribution_scale=self.scale, tree_method="hist", learning_rate=0.03, max_depth=6,
                 min_child_weight=20, subsample=0.8, colsample_bytree=0.7, reg_lambda=5.0, nthread=24, seed=7)
        self.m = xgb.train(p, d, 3000, evals=[(dv, "va")], early_stopping_rounds=100, verbose_eval=False)
        return self
    def surv(self, X):
        mu = np.log(self.m.predict(xgb.DMatrix(X), iteration_range=(0, self.m.best_iteration + 1)))
        ks = np.arange(1, K + 1)
        z = (np.log(ks)[None, :] - mu[:, None]) / self.scale
        F = norm.cdf(z) if self.dist == "normal" else 1 / (1 + np.exp(-z))
        return 1 - F
    def hazard(self, X, ws, wc):
        S = self.surv(X); Sp = np.column_stack([np.ones(len(S)), S[:, :-1]])
        return np.clip(1 - S / np.maximum(Sp, 1e-12), 1e-6, 1 - 1e-6)

def surv_from_h(h):
    return np.cumprod(1 - h, axis=1)

def nll_rc(h, L, reduce=True):
    """Right-censored discrete-time negative log-likelihood per landmark."""
    T = L["T"].values.astype(int); E = L["E"].values
    h = np.clip(h, 1e-7, 1 - 1e-7)
    lS = np.column_stack([np.zeros(len(h)), np.cumsum(np.log(1 - h), axis=1)])  # lS[:,j] = log S(j)
    m = T > 0
    r = np.arange(len(T))
    ll = np.where(E == 1, lS[r, np.maximum(T - 1, 0)] + np.log(h[r, np.maximum(T - 1, 0)]), lS[r, T])
    return -ll[m].mean() if reduce else -ll

def horizon_labels(L, hz):
    T = L["T"].values; E = L["E"].values
    pos = (E == 1) & (T <= hz)
    known = pos | (T >= hz)
    return pos.astype(int), known

def eval_risk(F, L, horizons=(1, 2, 4, 8, 12)):
    """F: n x K cumulative risk. Known-label AUC / Brier at horizons."""
    out = {}
    for hz in horizons:
        y, kn = horizon_labels(L, hz)
        r = F[kn, hz - 1]; yy = y[kn]
        out[f"AUC@{hz}"] = roc_auc_score(yy, r) if 0 < yy.sum() < len(yy) else np.nan
        out[f"Brier@{hz}"] = np.mean((r - yy) ** 2)
        out[f"Obs@{hz}"] = yy.mean(); out[f"Pred@{hz}"] = r.mean()
    return out

def stack_weights(hlist, L):
    """Convex weights on hazards minimising right-censored NLL (on validation)."""
    m = len(hlist)
    def f(z):
        w = np.exp(z) / np.exp(z).sum()
        return nll_rc(sum(wi * hi for wi, hi in zip(w, hlist)), L)
    r = minimize(f, np.zeros(m), method="Nelder-Mead", options=dict(maxiter=400, xatol=1e-4, fatol=1e-6))
    w = np.exp(r.x) / np.exp(r.x).sum()
    return w

def ipcw_G(L):
    """P(C >= k), k=0..K, from the monitoring-censoring time known for every landmark."""
    C = L["C"].values
    return np.array([(C >= k).mean() for k in range(0, K + 1)])

def eval_ipcw(F, L, horizons=(1, 2, 4, 8, 12)):
    """IPCW Brier, IPCW observed risk, mean predicted risk, calibration-in-the-large ratio, IPCW AUC."""
    T = L["T"].values.astype(int); E = L["E"].values; G = ipcw_G(L)
    out = {}
    for hz in horizons:
        ev = (E == 1) & (T <= hz); nonev = (T > hz) | ((T == hz) & (E == 0))
        w = np.zeros(len(T)); w[ev] = 1 / G[T[ev]]; w[nonev] = 1 / G[hz]
        r = F[:, hz - 1]
        if np.isnan(r).all(): continue
        y = ev.astype(float)
        out[f"IBS_h{hz}"] = np.sum(w * (r - y) ** 2) / len(T)
        out[f"ObsIPCW@{hz}"] = np.sum(w * y) / len(T)
        out[f"Pred@{hz}"] = np.nanmean(r)
        m = ev | nonev
        out[f"AUCw@{hz}"] = roc_auc_score(y[m], r[m], sample_weight=w[m])
    out["IBS"] = np.mean([out[f"IBS_h{h}"] for h in horizons if f"IBS_h{h}" in out])
    return out

def eval_ipcw_strat(F, L, strata, horizons=(1, 2, 4, 8, 12), min_G=0.05):
    """IPCW metrics with the monitoring-censoring distribution estimated within strata (e.g., calendar year),
    because whole-network reporting gaps make C depend on time. Horizons whose stratum G(h) < min_G are skipped
    for that stratum (labels unidentifiable)."""
    T = L["T"].values.astype(int); E = L["E"].values; C = L["C"].values; strata = np.asarray(strata)
    out = {}
    for hz in horizons:
        w = np.zeros(len(T)); use = np.zeros(len(T), bool)
        for s in np.unique(strata):
            m = strata == s
            G = np.array([(C[m] >= k).mean() for k in range(K + 1)])
            if G[hz] < min_G: continue
            ev = m & (E == 1) & (T <= hz); ne = m & ((T > hz) | ((T == hz) & (E == 0)))
            w[ev] = 1 / G[T[ev]]; w[ne] = 1 / G[hz]; use |= m
        y = ((E == 1) & (T <= hz)).astype(float); r = F[:, hz - 1]; nU = use.sum()
        if nU == 0: continue
        out[f"IBS_h{hz}"] = np.sum((w * (r - y) ** 2)[use]) / nU
        out[f"ObsIPCW@{hz}"] = np.sum((w * y)[use]) / nU
        out[f"Pred@{hz}"] = r[use].mean()
        m2 = use & (w > 0)
        out[f"AUCw@{hz}"] = roc_auc_score(y[m2], r[m2], sample_weight=w[m2]) if 0 < y[m2].sum() < m2.sum() else np.nan
        out[f"nEval@{hz}"] = int(nU)
    hs = [h for h in horizons if f"IBS_h{h}" in out]
    out["IBS"] = np.mean([out[f"IBS_h{h}"] for h in hs])
    return out
