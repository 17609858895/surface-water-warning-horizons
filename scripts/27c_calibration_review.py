"""Reviewer checks on calibration, comparators, censoring and surveillance (2015-2016 onset test unless stated).
C1 sensitivity of state-stratified calibration to the chronic cut-off and recent-breach window
C2 station-block bootstrap intervals for group coverage; stations per group
C3 parsimonious discrete-time logistic hazard comparator (state, margin, 52-week breach share, season, horizon)
C4 within-group bound AUC with station-block intervals and paired differences
C5 censoring-aware concordance (Harrell) within groups over all landmarks
C6 censoring diagnostics: network-wide gaps and a model of next-week missingness
C7 completion-weighted test coverage
C8 station-split isotonic recalibration of forecast-to-threshold rules
C9 surveillance capture at the 20% inspection fraction with station-block intervals, by period
C10 indicator and season composition of new onsets"""
import sys, numpy as np, pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).parent; sys.path.insert(0, str(HERE))
src = (HERE / "26_conformal_robustness.py").read_text(encoding="utf-8")
src = src[:src.index("# ---------- A: temporal ----------")]
g = {"__file__": str(HERE / "26_conformal_robustness.py")}; exec(compile(src, "26_defs", "exec"), g)
L, Lva, Lte, Lte0, dense, K = g["L"], g["Lva"], g["Lte"], g["Lte0"], g["dense"], g["K"]
calibrate, lpb, Fcum, truth, ipw, BETAS, H = g["calibrate"], g["lpb"], g["Fcum"], g["truth"], g["ipw"], g["BETAS"], g["H"]
from survlib import surv_from_h, horizon_labels, eval_ipcw_strat, person_period
ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "results" / "review"; OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(2703); B = 500
GROUPS = ["A / non-chronic", "A / chronic", "R / non-chronic", "R / chronic"]
pd.set_option("display.width", 220)


def groups(Lx, cut=0.10, W=8):
    j = Lx.wk_since_ex.values.astype(float); R = (Lx.state.values == 0) & (j >= 1) & (j <= W)
    ch = Lx.ex_rate52.fillna(0).values > cut
    return np.where(R, np.where(ch, "R / chronic", "R / non-chronic"), np.where(ch, "A / chronic", "A / non-chronic")), R, ch


def mondrian(Fv, Ft, gv, gt):
    bm, _ = calibrate(Fv, Lva); sm = lpb(Ft, bm); sg = np.zeros(len(Ft), int); betas = {}
    for x in GROUPS:
        m = gv == x; b, n = calibrate(Fv[m], Lva[m].reset_index(drop=True)) if m.sum() else (np.nan, 0)
        betas[x] = (b if np.isfinite(b) else bm, n); mt = gt == x; sg[mt] = lpb(Ft[mt], betas[x][0])
    return sm, sg, bm, betas


Tte = truth(Lte); keep = Lte.C.values >= K; site = Lte.site.values
F_va, F_te = Fcum("BRR-SL", "va"), Fcum("BRR-SL", "te")

# ---------- C1 ----------
rows = []
for cut, W in [(0.05, 8), (0.10, 8), (0.15, 8), (0.20, 8), (0.10, 4), (0.10, 12)]:
    gv, _, _ = groups(Lva, cut, W); gt, Rt, Ct = groups(Lte, cut, W)
    sm, sg, bm, betas = mondrian(F_va, F_te, gv, gt)
    for meth, sw in [("Marginal", sm), ("Mondrian", sg)]:
        cov = lambda m: float(np.mean(Tte[m & keep] > sw[m & keep]))
        rows.append(dict(cut=cut, window=W, method=meth, overall=cov(np.ones(len(sw), bool)), R_start=cov(Rt), chronic=cov(Ct),
                         worst_group=min(cov(gt == x) for x in GROUPS), mean_bound=float(sw.mean()), R_mean_bound=float(sw[Rt].mean()),
                         betas=";".join(f"{x}:{betas[x][0]:.3f}/{betas[x][1]}" for x in GROUPS)))
C1 = pd.DataFrame(rows); C1.to_csv(OUT / "calibration_sensitivity.csv", index=False); print(C1.round(3).drop(columns="betas").to_string(), flush=True)

# ---------- C2 ----------
gv, _, _ = groups(Lva); gt, Rt, Ct = groups(Lte)
sm, sg, bm, betas = mondrian(F_va, F_te, gv, gt)
sites = np.unique(site); idx_by_site = {s: np.where(site == s)[0] for s in sites}
labels = {"All": np.ones(len(Lte), bool), "R start": Rt, "Chronic": Ct, **{x: gt == x for x in GROUPS}}
boot_idx = [np.concatenate([idx_by_site[s] for s in rng.choice(sites, len(sites))]) for _ in range(B)]
rows = []
for meth, sw in [("Marginal", sm), ("Mondrian", sg)]:
    hit = (Tte > sw)
    for lab, m in labels.items():
        mm = m & keep; est = hit[mm].mean()
        bs = [hit[ix][(m & keep)[ix]].mean() for ix in boot_idx]
        rows.append(dict(method=meth, group=lab, coverage=est, lo=np.nanquantile(bs, 0.025), hi=np.nanquantile(bs, 0.975),
                         n_windows=int(mm.sum()), n_stations=int(len(np.unique(site[mm])))))
C2 = pd.DataFrame(rows); C2.to_csv(OUT / "coverage_station_bootstrap.csv", index=False); print(C2.round(3).to_string(), flush=True)

# ---------- C3 logistic hazard ----------
FEAT = lambda Lx: np.column_stack([((Lx.wk_since_ex.values >= 1) & (Lx.wk_since_ex.values <= 8)).astype(float),
                                   Lx.margin_min.fillna(0).clip(-2, 2).values, Lx.ex_rate52.fillna(0).values,
                                   Lx.woy_sin.values, Lx.woy_cos.values])
tr = (L.date <= "2013-09-30").values & (L.state.values == 0); Ltr = L[tr].reset_index(drop=True)
Xp, kp, yp, _ = person_period(Ltr, FEAT(Ltr))
KD = lambda k: np.eye(K)[k - 1]
lr = LogisticRegression(max_iter=3000, C=10.0).fit(np.column_stack([Xp, KD(kp)]), yp)


def lr_F(Lx):
    X0 = FEAT(Lx); h = np.column_stack([lr.predict_proba(np.column_stack([X0, np.tile(KD(np.array([k])), (len(X0), 1))]))[:, 1] for k in range(1, K + 1)])
    return 1 - np.cumprod(1 - h, axis=1)


F_lr_va, F_lr_te = lr_F(Lva), lr_F(Lte)
yr_te = Lte.date.dt.year.values
perf = []
for name, F in [("BRR-SL", F_te), ("Logistic hazard", F_lr_te), ("Station KM", Fcum("Station KM", "te"))]:
    e = eval_ipcw_strat(F, Lte, yr_te)
    perf.append(dict(model=name, MBS=e["IBS"], AUC4=e["AUCw@4"], ratio12=e["Pred@12"] / e["ObsIPCW@12"]))
print(pd.DataFrame(perf).round(4).to_string(), flush=True)
sm_lr, sg_lr, _, _ = mondrian(F_lr_va, F_lr_te, gv, gt)
_, sg_km, _, _ = mondrian(Fcum("Station KM", "va"), Fcum("Station KM", "te"), gv, gt)

# ---------- C4 within-group AUC ----------
safe = keep & (Tte > K); early = keep & (Tte <= 4)
bounds = {"BRR-SL": sg, "Logistic hazard": sg_lr, "Station KM": sg_km}
rows = []
for x in ["A / chronic", "R / chronic", "R / non-chronic"]:
    m = (gt == x) & (safe | early)
    def auc(ix, b):
        mm = m[ix]; y = safe[ix][mm].astype(int); s_ = b[ix][mm]
        return roc_auc_score(y, s_) if 0 < y.sum() < len(y) else np.nan
    full = np.arange(len(Lte))
    est = {k: auc(full, b) for k, b in bounds.items()}
    bs = {k: np.array([auc(ix, b) for ix in boot_idx[:300]]) for k, b in bounds.items()}
    for k in bounds:
        rows.append(dict(group=x, model=k, auc=est[k], lo=np.nanquantile(bs[k], 0.025), hi=np.nanquantile(bs[k], 0.975),
                         n_safe=int((m & safe).sum()), n_early=int((m & early).sum()), n_stations=int(len(np.unique(site[m])))))
    for k in ["Logistic hazard", "Station KM"]:
        d = bs["BRR-SL"] - bs[k]
        rows.append(dict(group=x, model=f"BRR-SL minus {k}", auc=est["BRR-SL"] - est[k], lo=np.nanquantile(d, 0.025), hi=np.nanquantile(d, 0.975)))
C4 = pd.DataFrame(rows); C4.to_csv(OUT / "bound_auc_bootstrap.csv", index=False); print(C4.round(3).to_string(), flush=True)

# ---------- C5 Harrell C over all landmarks ----------
T = Lte["T"].values.astype(float); E = Lte["E"].values.astype(int)


def harrell(risk, m):
    t, e, r = T[m], E[m], risk[m]; num = den = 0.0
    for i in np.where(e == 1)[0]:
        comp = t > t[i]
        if not comp.any(): continue
        den += comp.sum(); num += (r[i] > r[comp]).sum() + 0.5 * (r[i] == r[comp]).sum()
    return num / den if den else np.nan


rows = []
risks = {"BRR-SL": F_te[:, 3], "Logistic hazard": F_lr_te[:, 3], "Station KM": Fcum("Station KM", "te")[:, 3]}
for x in GROUPS + ["All"]:
    m = np.ones(len(Lte), bool) if x == "All" else gt == x
    for k, r in risks.items():
        est = harrell(r, m)
        bs = [harrell(r[ix], m[ix]) if False else np.nan for ix in []]
        rows.append(dict(group=x, model=k, c_index=est, n=int(m.sum()), events=int((E[m] == 1).sum())))
# station-block intervals for the two chronic groups
for x in ["A / chronic", "R / chronic"]:
    for k, r in risks.items():
        vals = []
        for ix in boot_idx[:200]:
            t0, e0, r0, mm = T[ix], E[ix], r[ix], (gt[ix] == x)
            tt, ee, rr = t0[mm], e0[mm], r0[mm]; num = den = 0.0
            for i in np.where(ee == 1)[0]:
                comp = tt > tt[i]
                if comp.any(): den += comp.sum(); num += (rr[i] > rr[comp]).sum() + 0.5 * (rr[i] == rr[comp]).sum()
            vals.append(num / den if den else np.nan)
        for rr_ in rows:
            if rr_["group"] == x and rr_["model"] == k: rr_.update(lo=np.nanquantile(vals, 0.025), hi=np.nanquantile(vals, 0.975))
C5 = pd.DataFrame(rows); C5.to_csv(OUT / "harrell_c.csv", index=False); print(C5.round(3).to_string(), flush=True)

# ---------- C6 censoring diagnostics ----------
w = pd.read_parquet(ROOT / "results" / "cnemc_weekly.parquet"); w = w[w.date.dt.year >= 2008]
recs = []
for s, gg in w.groupby("site"):
    gg = gg.set_index("date").sort_index(); idx = pd.date_range(gg.index.min(), gg.index.max(), freq="W-MON"); gg = gg.reindex(idx)
    recs.append(pd.DataFrame(dict(site=s, date=idx, miss=~gg[["NH4N", "DO", "CODMn"]].notna().all(axis=1).values)))
P = pd.concat(recs); net = P.groupby("date").miss.mean().rename("net_miss"); P = P.join(net, on="date")
share_in_netgap = P.loc[P.miss, "net_miss"].ge(0.5).mean()
Lc = L[(L.date.dt.year >= 2008)].copy(); Lc["next"] = Lc.date + pd.Timedelta(weeks=1)
Lc = Lc.join(net.rename("net_next"), on="next")
Lc["miss_next"] = (Lc.C.values == 0).astype(int); Lc["breach"] = (Lc.state == 1).astype(int)
Lc["recent"] = ((Lc.state == 0) & (Lc.wk_since_ex >= 1) & (Lc.wk_since_ex <= 8)).astype(int)
Lc["chronic"] = (Lc.ex_rate52.fillna(0) > 0.10).astype(int); Lc["margin"] = Lc.margin_min.fillna(0).clip(-2, 2)
Lc = Lc.dropna(subset=["net_next"])
import statsmodels.api as sm_
Xs = sm_.add_constant(pd.concat([Lc[["breach", "recent", "chronic", "margin", "net_next"]], pd.get_dummies(Lc.date.dt.year, prefix="y", drop_first=True, dtype=float)], axis=1))
fit = sm_.GLM(Lc.miss_next.values, Xs.astype(float), family=sm_.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(Lc.site)[0]})
ci = fit.conf_int()
cens = pd.DataFrame(dict(term=["breach", "recent", "chronic", "margin", "net_next"],
                         odds_ratio=np.exp(fit.params[["breach", "recent", "chronic", "margin", "net_next"]].values),
                         lo=np.exp(ci.loc[["breach", "recent", "chronic", "margin", "net_next"], 0].values),
                         hi=np.exp(ci.loc[["breach", "recent", "chronic", "margin", "net_next"], 1].values)))
cens.loc[len(cens)] = dict(term="share_of_missing_station_weeks_in_weeks_with_>=50%_network_missing", odds_ratio=share_in_netgap, lo=np.nan, hi=np.nan)
cens.loc[len(cens)] = dict(term="overall_next_week_missing_rate", odds_ratio=Lc.miss_next.mean(), lo=np.nan, hi=np.nan)
cens.to_csv(OUT / "censoring_diagnostics.csv", index=False); print(cens.round(3).to_string(), flush=True)

# ---------- C7 completion-weighted coverage ----------
wte = ipw(Lte)
rows = []
for meth, sw in [("Marginal", sm), ("Mondrian", sg)]:
    for lab, m in [("All", np.ones(len(Lte), bool)), ("R start", Rt), ("Chronic", Ct)]:
        mm = m & keep
        rows.append(dict(method=meth, group=lab, coverage_weighted=float(np.sum(wte[mm] * (Tte[mm] > sw[mm])) / wte[mm].sum()),
                         coverage_unweighted=float(np.mean(Tte[mm] > sw[mm]))))
C7 = pd.DataFrame(rows); C7.to_csv(OUT / "coverage_weighted.csv", index=False); print(C7.round(3).to_string(), flush=True)

# ---------- C8 forecast recalibration ----------
BF = pd.read_pickle(ROOT / "results" / "core" / "baseline_F_test.pkl")
Fmax, Find = BF["onset_FTmax"][dense], BF["onset_FTind"][dense]
rows = []
for rep in range(50):
    half = rng.choice(sites, len(sites) // 2, replace=False); A_ = np.isin(site, half); Bm = ~A_
    Lb = Lte[Bm].reset_index(drop=True); yb = Lb.date.dt.year.values
    res = {}
    for name, F in [("Forecast max", Fmax), ("Forecast independent", Find)]:
        Fr = np.zeros_like(F[Bm])
        for k in range(1, K + 1):
            y, kn = horizon_labels(Lte, k); mA = A_ & kn
            iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(F[mA, k - 1], y[mA])
            Fr[:, k - 1] = iso.predict(F[Bm, k - 1])
        Fr = np.maximum.accumulate(Fr, axis=1)
        for tag, FF in [(name, F[Bm]), (name + ", recalibrated", Fr)]:
            e = eval_ipcw_strat(FF, Lb, yb); res[tag] = (e["IBS"], e["Pred@12"] / e["ObsIPCW@12"], e["AUCw@4"])
    e = eval_ipcw_strat(F_te[Bm], Lb, yb); res["BRR-SL"] = (e["IBS"], e["Pred@12"] / e["ObsIPCW@12"], e["AUCw@4"])
    for k_, v in res.items(): rows.append(dict(rep=rep, model=k_, MBS=v[0], ratio12=v[1], AUC4=v[2]))
C8 = pd.DataFrame(rows).groupby("model").agg(MBS=("MBS", "median"), MBS_lo=("MBS", lambda v: v.quantile(0.05)), MBS_hi=("MBS", lambda v: v.quantile(0.95)),
                                            ratio12=("ratio12", "median"), ratio_lo=("ratio12", lambda v: v.quantile(0.05)), ratio_hi=("ratio12", lambda v: v.quantile(0.95)),
                                            AUC4=("AUC4", "median")).reset_index()
C8.to_csv(OUT / "forecast_recalibration.csv", index=False); print(C8.round(4).to_string(), flush=True)

# ---------- C9 capture with station-block intervals ----------
site0 = Lte0.site.values; y4, kn = horizon_labels(Lte0, 4); wk = Lte0.date.values; yr0 = Lte0.date.dt.year.values
scores = {"BRR-SL": (1 - surv_from_h(H["BRR-SL"]["te"]))[:, 3], "Forecast max": BF["onset_FTmax"][:, 3],
          "Station KM": (1 - surv_from_h(H["Station KM"]["te"]))[:, 3],
          "History rule": Lte0.ex_rate52.fillna(0).values + 1e-3 * (1 - Lte0.margin_min.fillna(1).values)}


def capture(ix, sc, period_mask):
    hit = tot = 0
    wv, yv, kv, sv, pm = wk[ix], y4[ix], kn[ix], sc[ix], period_mask[ix]
    for d in np.unique(wv[pm]):
        m = (wv == d) & kv & pm
        if m.sum() < 5: continue
        k_ = max(1, int(round(0.2 * m.sum()))); top = np.argsort(-sv[m], kind="stable")[:k_]
        hit += yv[m][top].sum(); tot += yv[m].sum()
    return hit / max(tot, 1)


sites0 = np.unique(site0); ibs0 = {s: np.where(site0 == s)[0] for s in sites0}
boots0 = [np.concatenate([ibs0[s] for s in rng.choice(sites0, len(sites0))]) for _ in range(300)]
full0 = np.arange(len(Lte0)); rows = []
for per, pm in [("2015-2018", np.ones(len(Lte0), bool)), ("2015-2016", yr0 <= 2016), ("2017-2018", yr0 >= 2017)]:
    bvals = {}
    for nm, sc in scores.items():
        est = capture(full0, sc, pm); bvals[nm] = np.array([capture(ix, sc, pm) for ix in boots0])
        rows.append(dict(period=per, model=nm, capture=est, lo=np.quantile(bvals[nm], 0.025), hi=np.quantile(bvals[nm], 0.975)))
    for nm in ["History rule", "Forecast max"]:
        d = bvals["BRR-SL"] - bvals[nm]
        rows.append(dict(period=per, model=f"BRR-SL minus {nm}", capture=capture(full0, scores["BRR-SL"], pm) - capture(full0, scores[nm], pm),
                         lo=np.quantile(d, 0.025), hi=np.quantile(d, 0.975)))
    print(per, "capture done", flush=True)
C9 = pd.DataFrame(rows); C9.to_csv(OUT / "capture_bootstrap.csv", index=False); print(C9.round(3).to_string(), flush=True)

# ---------- C10 onset composition ----------
rec = []
for s, gg in w.groupby("site"):
    gg = gg.set_index("date").sort_index(); idx = pd.date_range(gg.index.min(), gg.index.max(), freq="W-MON"); gg = gg.reindex(idx)
    ob = gg[["NH4N", "DO", "CODMn"]].notna().all(axis=1).values
    bN, bD, bC = (gg.NH4N > 1).values, (gg.DO < 5).values, (gg.CODMn > 6).values; br = ob & (bN | bD | bC)
    for i in range(1, len(idx)):
        if ob[i] and ob[i - 1] and br[i] and not br[i - 1]:
            rec.append(dict(site=s, quarter=idx[i].quarter, NH4N=bN[i], DO=bD[i], CODMn=bC[i]))
O_ = pd.DataFrame(rec)
comp = dict(n_onsets=len(O_), NH4N=O_.NH4N.mean(), DO=O_.DO.mean(), CODMn=O_.CODMn.mean(),
            multi=(O_[["NH4N", "DO", "CODMn"]].sum(axis=1) >= 2).mean())
for q in range(1, 5):
    m = O_.quarter == q
    comp[f"Q{q}_share"] = m.mean(); comp[f"Q{q}_NH4N"] = O_.NH4N[m].mean(); comp[f"Q{q}_DO"] = O_.DO[m].mean(); comp[f"Q{q}_CODMn"] = O_.CODMn[m].mean()
pd.Series(comp).to_csv(OUT / "onset_composition.csv"); print(pd.Series(comp).round(3).to_string(), flush=True)
pd.DataFrame(perf).to_csv(OUT / "logistic_hazard_performance.csv", index=False)
