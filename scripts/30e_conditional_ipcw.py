"""Revision analysis 30e (Reviewer 1, comment 4): IPCW evaluation with covariate-dependent censoring models.

The main evaluation estimates the monitoring survival function G(k) = P(C >= k) within calendar year. Here the
censoring time C (first unobserved week after the landmark) is modelled conditionally on water-quality status, the
threshold margin and recent breach history, as a discrete-time logistic hazard fitted to the person-period data of
all 2015-2018 test landmarks (rows: landmark i at horizon k with C_i >= k-1; event: week k unobserved):
  CM-A  year x horizon baseline hazards (nesting the main estimator) + covariates
  CM-B  calendar-week fixed effects for the at-risk week t+k (network-wide reporting gaps) + covariates
  covariates: landmark state (breach/compliant), recent-breach state (R), minimum normalized margin and its square,
  current NH4-N, DO and CODMn, 52-week breach share, observation density over the preceding 12 weeks, season.
G_i(k) = prod_{j<=k}(1 - c_ij). Weights are 1/G_i(T_i) for events by horizon h and 1/G_i(h) for event-free landmarks.
Because network-wide gaps leave some windows with G near zero, landmarks with G_i(h) < 0.05 are excluded at that
horizon (as whole years are in the main analysis), and all estimators, including the year-stratified reference,
are normalized by the sum of weights (Hajek form) so that they are directly comparable.
Outputs: results/revision/ipcw_conditional.csv, ipcw_censoring_model.csv"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).parent; sys.path.insert(0, str(HERE))
from survlib import surv_from_h, K
ROOT = HERE.parent; RES = ROOT / "results"; OUT = RES / "revision"
L = pd.read_parquet(RES / "landmarks.parquet").reset_index(drop=True)
H = pd.read_pickle(RES / "core" / "hazards.pkl"); CC = pd.read_pickle(RES / "core" / "baseline_F_test.pkl")
te = (L.date >= "2015-01-01").values
Lt = L[te].reset_index(drop=True)
YEARS = sorted(Lt.date.dt.year.unique())


def covariates(Ls, ii):
    d = pd.DataFrame(index=range(len(ii)))
    d["breach"] = Ls.state.values[ii].astype(float)
    j = Ls.wk_since_ex.values.astype(float)[ii]; d["recent"] = ((Ls.state.values[ii] == 0) & (j >= 1) & (j <= 8)).astype(float)
    mm = np.clip(Ls.margin_min.values[ii], -1, 1); d["margin"] = mm; d["margin2"] = mm ** 2
    for c in ["log_NH4N_0", "DO_0", "log_CODMn_0", "obs_frac12", "woy_sin", "woy_cos"]: d[c] = Ls[c].values[ii]
    d["chronic"] = Ls.ex_rate52.fillna(0).values[ii]
    return d.fillna(d.median())


ALLWEEKS = sorted(set(pd.DatetimeIndex((Lt.date.values[:, None] + (np.arange(1, K + 1) * 7).astype("timedelta64[D]")).ravel())))


def design(Ls, ii, kk, kind):
    d = covariates(Ls, ii)
    if kind == "CM-A":
        yr = Ls.date.dt.year.values[ii]
        for y in YEARS:
            for k in range(1, K + 1):
                if (y, k) != (YEARS[0], 1): d[f"y{y}_k{k}"] = ((yr == y) & (kk == k)).astype(float)
    else:
        wk = pd.DatetimeIndex(Ls.date.values[ii] + (kk * 7).astype("timedelta64[D]"))
        dummies = pd.get_dummies(pd.Categorical(wk, categories=ALLWEEKS), drop_first=True, dtype=float)
        dummies.columns = [f"w{pd.Timestamp(c).date()}" for c in dummies.columns]
        d = pd.concat([d, dummies.reset_index(drop=True)], axis=1)
    return d


def person_period(Ls):
    C = Ls.C.values.astype(int); ii, kk, yy = [], [], []
    for k in range(1, K + 1):
        idx = np.where(C >= k - 1)[0]; ii.append(idx); kk.append(np.full(len(idx), k)); yy.append((C[idx] == k - 1).astype(float))
    return np.concatenate(ii), np.concatenate(kk), np.concatenate(yy)


fits, coef_rows = {}, []
ii, kk, yy = person_period(Lt)
for kind in ("CM-A", "CM-B"):
    Xd = sm.add_constant(design(Lt, ii, kk, kind), has_constant="add")
    # calendar weeks with all or no stations reporting are separated; a negligible ridge keeps the fit finite
    f0 = sm.GLM(yy, Xd, family=sm.families.Binomial()).fit_regularized(alpha=1e-7, L1_wt=0.0, maxiter=500)
    fits[kind] = (f0, list(Xd.columns))
    try:  # covariate effects with station-clustered intervals; separated baseline cells (all or no rows censored) carry
        # no information on the covariates and are dropped for this step only
        base = [c for c in Xd.columns if c.startswith(("y20", "w20"))]
        cell = Xd[base].values.argmax(1) + Xd[base].values.max(1) * 0 if base else np.zeros(len(Xd))
        cell = np.where(Xd[base].values.max(1) > 0, Xd[base].values.argmax(1), -1)
        rate = pd.Series(yy).groupby(cell).transform("mean").values
        keep = (rate > 0) & (rate < 1)
        Xk = Xd[keep]; Xk = Xk.loc[:, (Xk != 0).any(axis=0)]
        f = sm.GLM(yy[keep], Xk, family=sm.families.Binomial()).fit(maxiter=100, cov_type="cluster",
                                                                   cov_kwds={"groups": pd.factorize(Lt.site.values[ii][keep])[0]})
        for t in ["breach", "recent", "margin", "chronic", "log_NH4N_0", "DO_0", "log_CODMn_0"]:
            ci = f.conf_int().loc[t]
            coef_rows.append(dict(model=kind, term=t, odds_ratio=np.exp(f.params[t]), lo=np.exp(ci[0]), hi=np.exp(ci[1]), p=f.pvalues[t]))
    except Exception as ex:
        print(kind, "clustered intervals failed:", repr(ex)[:200])
    p_in = np.asarray(f0.predict(Xd)); print(kind, "columns", Xd.shape[1], "in-sample mean hazard", round(p_in.mean(), 4), "observed", round(yy.mean(), 4), flush=True)
CF = pd.DataFrame(coef_rows); CF.to_csv(OUT / "ipcw_censoring_model.csv", index=False)
pd.set_option("display.width", 220); print(CF.round(3).to_string(), flush=True)


def G_matrix(Ls, kind):
    n = len(Ls); i2 = np.repeat(np.arange(n), K); k2 = np.tile(np.arange(1, K + 1), n)
    f, cols = fits[kind]; Xd = sm.add_constant(design(Ls, i2, k2, kind), has_constant="add").reindex(columns=cols, fill_value=0.0)
    c = np.asarray(f.predict(Xd)).reshape(n, K)
    return np.column_stack([np.ones(n), np.cumprod(1 - c, axis=1)])


def G_strat(Ls):
    yr = Ls.date.dt.year.values; C = Ls.C.values; G = np.zeros((len(Ls), K + 1))
    for y in np.unique(yr):
        m = yr == y; G[m] = np.array([(C[m] >= k).mean() for k in range(K + 1)])
    return G


def evaluate(F, Ls, G, horizons, min_G=0.05):
    T = Ls["T"].values.astype(int); E = Ls["E"].values; r_ = np.arange(len(T)); out = {}
    for hz in horizons:
        ok = G[:, hz] >= min_G
        ev = ok & (E == 1) & (T <= hz); ne = ok & ((T > hz) | ((T == hz) & (E == 0)))
        w = np.zeros(len(T)); w[ev] = 1 / G[r_[ev], T[ev]]; w[ne] = 1 / G[r_[ne], hz]
        y = ev.astype(float); r = F[:, hz - 1]; m = ev | ne; sw = w[m].sum()
        out[f"Brier@{hz}"] = np.sum(w[m] * (r[m] - y[m]) ** 2) / sw
        out[f"Obs@{hz}"] = np.sum(w[m] * y[m]) / sw; out[f"Pred@{hz}"] = r[ok].mean()
        out[f"AUC@{hz}"] = roc_auc_score(y[m], r[m], sample_weight=w[m]) if 0 < y[m].sum() < m.sum() else np.nan
        out[f"n@{hz}"] = int(ok.sum()); out[f"wmean@{hz}"] = w[ok].mean()
    out["MBS"] = np.mean([out[f"Brier@{h}"] for h in horizons])
    return out


MODELS = ["BRR-SL", "Forecast-threshold (max)", "Forecast-threshold (indep.)", "Horizon classifiers", "Station KM", "Population KM"]
HZ = {"2015-2016": (1, 2, 4, 8, 12), "2017-2018": (1, 2, 4)}


def F_of(trans, model):
    if model == "Horizon classifiers": return CC[trans]
    if model == "Forecast-threshold (max)": return CC[trans + "_FTmax"]
    if model == "Forecast-threshold (indep.)": return CC[trans + "_FTind"]
    return 1 - surv_from_h(H[trans][model]["te"])


rows = []
for trans, st in (("onset", 0), ("recovery", 1)):
    Ls = L[te & (L.state.values == st)].reset_index(drop=True); yr = Ls.date.dt.year.values
    Gs = {"year-stratified (Hajek)": G_strat(Ls), "CM-A": G_matrix(Ls, "CM-A"), "CM-B": G_matrix(Ls, "CM-B")}
    for model in MODELS:
        F = F_of(trans, model)
        for per, msk in (("2015-2016", yr <= 2016), ("2017-2018", yr >= 2017)):
            hz = HZ[per]; last = hz[-1]
            for cm, G in Gs.items():
                r = evaluate(F[msk], Ls[msk].reset_index(drop=True), G[msk], hz)
                rows.append(dict(trans=trans, model=model, period=per, censoring=cm, MBS=r["MBS"], AUC1=r["AUC@1"], AUC4=r["AUC@4"],
                                 last_h=last, pred_last=r[f"Pred@{last}"], obs_last=r[f"Obs@{last}"], ratio_last=r[f"Pred@{last}"] / r[f"Obs@{last}"],
                                 n_eval_last=r[f"n@{last}"], mean_weight_last=r[f"wmean@{last}"]))
R = pd.DataFrame(rows)
R["rank_MBS"] = R.groupby(["trans", "period", "censoring"]).MBS.rank()
R.to_csv(OUT / "ipcw_conditional.csv", index=False)
print(R.round(4).to_string())
