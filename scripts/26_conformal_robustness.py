"""Stress tests for the Mondrian conformal safe weeks (innovation point 2).

Lower predictive bound at level beta: LPB_i(beta) = min(#{k: F_ik < beta}, K - 1), F = cumulative onset risk (monotone).
beta is the largest grid value whose IPCW-weighted miscoverage on calibration landmarks with complete 12-week follow-up
is <= alpha(1 + 1/n) - 1/n, separately within each Mondrian group (start state A/R x chronic 52-week share > 0.10).
Coverage is evaluated on test landmarks with complete follow-up (as in Fig. 7).

A  Temporal: calibrate on 2014, evaluate 2015 and 2016 separately; rolling recalibration on 2015, evaluate 2016.
B  Cross-station: within 2015-2016, calibrate on a random half of stations and evaluate on the other half (200 splits).
C  Serial dependence: calibrate on thinned 2014 landmarks (one per station per 8-week block; 200 draws).
D  Partitions not used for calibration: north/south, season, year, monitoring density, limiting indicator.
E  Model ablation: the same Mondrian calibration applied to station KM and population KM hazards."""
import sys, numpy as np, pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, str(Path(__file__).parent))
from survlib import K, surv_from_h
from tristate import start_state

OUT = Path(__file__).resolve().parents[1] / "results"; FD = OUT / "fig7"
L = pd.read_parquet(OUT / "landmarks.parquet").reset_index(drop=True); state = L.state.values.astype(int)
H = pd.read_pickle(OUT / "core" / "hazards.pkl")["onset"]
va = ((L.date >= "2014-01-01") & (L.date <= "2014-09-30")).values & (state == 0)
te = (L.date >= "2015-01-01").values & (state == 0)
Lva = L[va].reset_index(drop=True); Lte0 = L[te].reset_index(drop=True)
dense = Lte0.date.dt.year.values <= 2016; Lte = Lte0[dense].reset_index(drop=True)
ZC = ["ex_rate52", "margin_min", "woy_sin", "woy_cos", "lat", "lon", "obs_frac12"]
BETAS = np.linspace(0.95, 0.001, 950)
ALPHA = 0.10; rng = np.random.default_rng(26)
GROUPS = ["A / non-chronic", "A / chronic", "R / non-chronic", "R / chronic"]


def Fcum(model, part):
    h = H[model][part]; F = 1 - surv_from_h(h)
    return F if part == "va" else F[dense]


def groups(Lx):
    R = start_state(Lx.state.values.astype(int), Lx.wk_since_ex.values) >= 2
    ch = Lx.ex_rate52.fillna(0).values > 0.10
    return np.where(R, np.where(ch, "R / chronic", "R / non-chronic"), np.where(ch, "A / chronic", "A / non-chronic"))


def truth(Lx): return np.where(Lx.E.values == 1, Lx["T"].values, K + 1)
def lpb(F, beta): return np.minimum((F < beta).sum(1), K - 1)


def ipw(Lx):
    y = (Lx.C.values >= K).astype(int)
    if y.min() == y.max() or len(Lx) < 20: return np.ones(len(Lx))
    X = Lx[ZC].fillna(0).values
    return 1 / np.clip(LogisticRegression(max_iter=2000).fit(X, y).predict_proba(X)[:, 1], 0.05, 1)


def calibrate(F, Lc, alpha=ALPHA):
    keep = Lc.C.values >= K; n = int(keep.sum())
    if n < 30: return np.nan, n
    w = ipw(Lc)[keep]; T = truth(Lc)[keep]
    LP = np.minimum((F[keep][:, :, None] < BETAS[None, None, :]).sum(1), K - 1)       # n x len(BETAS)
    miss = (w[:, None] * (T[:, None] <= LP)).sum(0) / w.sum()
    ok = np.where(miss <= alpha * (1 + 1 / n) - 1 / n)[0]
    return (BETAS[ok[0]] if len(ok) else BETAS[-1]), n


def fit_bounds(Fc, Lc, gc, Ft, gt):
    """Returns marginal and Mondrian LPBs for the test set, and the betas."""
    bm, _ = calibrate(Fc, Lc); sw_m = lpb(Ft, bm); sw_g = np.zeros(len(Ft), int); betas = {}
    for g in GROUPS:
        m = gc == g; b, n = calibrate(Fc[m], Lc[m].reset_index(drop=True)) if m.sum() else (np.nan, 0)
        betas[g] = b if np.isfinite(b) else bm
        mt = gt == g; sw_g[mt] = lpb(Ft[mt], betas[g])
    return sw_m, sw_g, bm, betas


def cov_table(sw, Tt, keep, labels, tag):
    out = []
    for lab, m in labels.items():
        mm = m & keep
        if mm.sum() == 0: continue
        out.append(dict(**tag, group=lab, coverage=float(np.mean(Tt[mm] > sw[mm])), n_eval=int(mm.sum()), mean_safe_weeks=float(sw[m].mean())))
    return out


gva = groups(Lva); gte = groups(Lte); Tte = truth(Lte); keep_te = Lte.C.values >= K
F_va, F_te = Fcum("BRR-SL", "va"), Fcum("BRR-SL", "te")
grp_labels = lambda g: {"All": np.ones(len(g), bool), **{x: g == x for x in GROUPS}}

# ---------- A: temporal ----------
rows = []
sw_m, sw_g, bm, betas = fit_bounds(F_va, Lva, gva, F_te, gte)
yr = Lte.date.dt.year.values
for y in (2015, 2016):
    for meth, sw in [("Marginal", sw_m), ("Mondrian", sw_g)]:
        rows += cov_table(sw, Tte, keep_te & (yr == y), grp_labels(gte), dict(test="A", setting=f"calibrate 2014, test {y}", method=meth))
c15 = yr == 2015; t16 = yr == 2016
L15 = Lte[c15].reset_index(drop=True); L16 = Lte[t16].reset_index(drop=True)
sw_m2, sw_g2, _, _ = fit_bounds(F_te[c15], L15, gte[c15], F_te[t16], gte[t16])
for meth, sw in [("Marginal", sw_m2), ("Mondrian", sw_g2)]:
    rows += cov_table(sw, truth(L16), L16.C.values >= K, grp_labels(gte[t16]), dict(test="A", setting="recalibrate 2015, test 2016", method=meth))
print("A done; betas", round(bm, 3), {k: round(v, 3) for k, v in betas.items()}, flush=True)

# ---------- B: cross-station ----------
sites = np.unique(Lte.site.values); brows = []
for b in range(200):
    half = rng.choice(sites, len(sites) // 2, replace=False); cm = np.isin(Lte.site.values, half); tm = ~cm
    Lc, Lt = Lte[cm].reset_index(drop=True), Lte[tm].reset_index(drop=True)
    sm, sg, _, _ = fit_bounds(F_te[cm], Lc, gte[cm], F_te[tm], gte[tm])
    Tt, kt = truth(Lt), Lt.C.values >= K
    for meth, sw in [("Marginal", sm), ("Mondrian", sg)]:
        for r in cov_table(sw, Tt, kt, grp_labels(gte[tm]), dict(test="B", split=b, method=meth)): brows.append(r)
B = pd.DataFrame(brows)
for (meth, g), d in B.groupby(["method", "group"]):
    rows.append(dict(test="B", setting="cross-station halves (200 splits)", method=meth, group=g, coverage=d.coverage.median(),
                     lo=d.coverage.quantile(0.05), hi=d.coverage.quantile(0.95), n_eval=int(d.n_eval.median()), mean_safe_weeks=d.mean_safe_weeks.median()))
worst = B[B.group != "All"].groupby(["method", "split"]).coverage.min().groupby("method")
for meth, v in worst:
    rows.append(dict(test="B", setting="cross-station halves (200 splits)", method=meth, group="Worst group", coverage=v.median(),
                     lo=v.quantile(0.05), hi=v.quantile(0.95)))
B.to_csv(FD / "robust_cross_station_splits.csv", index=False)
print("B done", flush=True)

# ---------- C: serial dependence (thinned calibration) ----------
blk = (Lva.date.dt.isocalendar().week.values.astype(int) - 1) // 8
crow = []
for b in range(200):
    pick = Lva.assign(blk=blk, r=rng.random(len(Lva))).groupby(["site", "blk"]).r.idxmin().values
    Lc = Lva.loc[pick].reset_index(drop=True)
    sm, sg, _, _ = fit_bounds(F_va[pick], Lc, gva[pick], F_te, gte)
    for meth, sw in [("Marginal", sm), ("Mondrian", sg)]:
        crow += cov_table(sw, Tte, keep_te, grp_labels(gte), dict(test="C", split=b, method=meth, n_cal=len(Lc)))
C = pd.DataFrame(crow)
for (meth, g), d in C.groupby(["method", "group"]):
    rows.append(dict(test="C", setting="thinned calibration (1 per station per 8 weeks)", method=meth, group=g, coverage=d.coverage.median(),
                     lo=d.coverage.quantile(0.05), hi=d.coverage.quantile(0.95), n_eval=int(d.n_eval.median()), mean_safe_weeks=d.mean_safe_weeks.median()))
print("C done", flush=True)

# ---------- D: partitions not used for calibration ----------
mcols = Lte[["mNH4N", "mDO", "mCOD"]].values
lim = np.array(["NH4-N", "DO", "CODMn"])[np.nanargmin(np.where(np.isnan(mcols), np.inf, mcols), axis=1)]
q = Lte.date.dt.quarter.values; dens = Lte.obs_frac12.values
parts = {"North (≥33.5°N)": Lte.lat.values >= 33.5, "South (<33.5°N)": Lte.lat.values < 33.5,
         "Jan–Mar": q == 1, "Apr–Jun": q == 2, "Jul–Sep": q == 3, "Oct–Dec": q == 4,
         "2015": yr == 2015, "2016": yr == 2016,
         "Dense monitoring": dens >= np.nanmedian(dens), "Sparse monitoring": dens < np.nanmedian(dens),
         "Nearest limit NH4-N": lim == "NH4-N", "Nearest limit DO": lim == "DO", "Nearest limit CODMn": lim == "CODMn"}
for meth, sw in [("Marginal", sw_m), ("Mondrian", sw_g)]:
    for r in cov_table(sw, Tte, keep_te, parts, dict(test="D", setting="calibrate 2014, partitions of 2015–2016", method=meth)):
        mm = parts[r["group"]] & keep_te; k_ = Tte[mm] > sw[mm]
        bs = [rng.choice(k_, len(k_)).mean() for _ in range(300)]
        r.update(lo=np.quantile(bs, 0.025), hi=np.quantile(bs, 0.975)); rows.append(r)
print("D done", flush=True)

# ---------- E: model ablation (KM hazards under the same Mondrian calibration) ----------
erows = []
for model in ["BRR-SL", "Station KM", "Population KM"]:
    Fv, Ft = Fcum(model, "va"), Fcum(model, "te")
    sm, sg, _, _ = fit_bounds(Fv, Lva, gva, Ft, gte)
    early = keep_te & (Tte <= 4); safe = keep_te & (Tte > K)
    for meth, sw in [("Marginal", sm), ("Mondrian", sg)]:
        for r in cov_table(sw, Tte, keep_te, grp_labels(gte), dict(test="E", setting=model, method=meth)): rows.append(r)
        erows.append(dict(model=model, method=meth, coverage=float(np.mean(Tte[keep_te] > sw[keep_te])), mean_safe_weeks=float(sw.mean()),
                          safe_certified_8wk=float(np.mean(sw[safe] >= 8)), early_breach_certified_4wk=float(np.mean(sw[early] >= 4)),
                          share_zero=float(np.mean(sw == 0)), n_safe=int(safe.sum()), n_early=int(early.sum())))
pd.DataFrame(erows).to_csv(FD / "robust_model_ablation.csv", index=False)
RS = pd.DataFrame(rows); RS.to_csv(FD / "robust_conformal.csv", index=False)
pd.set_option("display.width", 220)
print(RS.round(3).to_string()); print(pd.DataFrame(erows).round(3).to_string())
