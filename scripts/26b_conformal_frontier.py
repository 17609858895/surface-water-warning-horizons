"""(F) Coverage-efficiency frontier: sweep alpha for BRR-SL, station KM and population KM hazards under marginal and Mondrian
calibration (2014 calibration, 2015-2016 evaluation); efficiency = share of truly safe landmarks (no breach in 12 weeks)
certified >= 8 safe weeks, and share of early breaches (<= 4 weeks) wrongly certified >= 4 weeks.
(G) Finer Mondrian groups: start state x chronic x limiting indicator (NH4-N vs DO/CODMn), with fallback to the parent
group's level when a cell has < 30 calibration landmarks."""
import numpy as np, pandas as pd
from pathlib import Path

HERE = Path(__file__).parent
src = (HERE / "26_conformal_robustness.py").read_text(encoding="utf-8")
src = src[:src.index("# ---------- A: temporal ----------")]
ns = {"__file__": str(HERE / "26_conformal_robustness.py")}; exec(compile(src, "26_defs", "exec"), ns)
g = ns; Lva, Lte, gva, gte, Tte, keep_te, K = g["Lva"], g["Lte"], g["gva"], g["gte"], g["Tte"], g["keep_te"], g["K"]
calibrate, lpb, Fcum, GROUPS = g["calibrate"], g["lpb"], g["Fcum"], g["GROUPS"]
FD = g["FD"]
safe = keep_te & (Tte > K); early = keep_te & (Tte <= 4)


def bounds(Fv, Ft, alpha, gc, gt, levels):
    bm, _ = calibrate(Fv, Lva, alpha); out_m = lpb(Ft, bm); out_g = np.zeros(len(Ft), int)
    for lev in levels:
        m = gc == lev; mt = gt == lev
        if mt.sum() == 0: continue
        b, n = calibrate(Fv[m], Lva[m].reset_index(drop=True), alpha) if m.sum() else (np.nan, 0)
        if not np.isfinite(b):
            parent = lev.split(" | ")[0]; pm = np.array([x.split(" | ")[0] == parent for x in gc])
            b, _ = calibrate(Fv[pm], Lva[pm].reset_index(drop=True), alpha)
            if not np.isfinite(b): b = bm
        out_g[mt] = lpb(Ft[mt], b)
    return out_m, out_g


def summary(sw):
    return dict(coverage=float(np.mean(Tte[keep_te] > sw[keep_te])), mean_safe_weeks=float(sw.mean()),
                safe_certified_8wk=float(np.mean(sw[safe] >= 8)), early_certified_4wk=float(np.mean(sw[early] >= 4)),
                worst_group=float(min(np.mean(Tte[keep_te & (gte == x)] > sw[keep_te & (gte == x)]) for x in GROUPS)))


rows = []
for model in ["BRR-SL", "Station KM", "Population KM"]:
    Fv, Ft = Fcum(model, "va"), Fcum(model, "te")
    for alpha in [0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]:
        sm, sg = bounds(Fv, Ft, alpha, gva, gte, GROUPS)
        rows.append(dict(model=model, method="Marginal", alpha=alpha, **summary(sm)))
        rows.append(dict(model=model, method="Mondrian", alpha=alpha, **summary(sg)))
    print(model, "done", flush=True)
FR = pd.DataFrame(rows); FR.to_csv(FD / "robust_frontier.csv", index=False)

# ---------- G: finer Mondrian groups ----------
def limiting(Lx):
    m = Lx[["mNH4N", "mDO", "mCOD"]].values
    return np.where(np.nanargmin(np.where(np.isnan(m), np.inf, m), axis=1) == 0, "NH4-N", "DO/CODMn")


gva8 = np.array([f"{a} | {b}" for a, b in zip(gva, limiting(Lva))]); gte8 = np.array([f"{a} | {b}" for a, b in zip(gte, limiting(Lte))])
Fv, Ft = Fcum("BRR-SL", "va"), Fcum("BRR-SL", "te")
_, sg4 = bounds(Fv, Ft, 0.10, gva, gte, GROUPS); _, sg8 = bounds(Fv, Ft, 0.10, gva8, gte8, np.unique(gte8))
lim_te = limiting(Lte); grow = []
for meth, sw in [("Mondrian (state × chronic)", sg4), ("Mondrian (state × chronic × limiting indicator)", sg8)]:
    for lab, m in [("All", np.ones(len(Lte), bool)), ("Limiting NH4-N", lim_te == "NH4-N"), ("Limiting DO/CODMn", lim_te == "DO/CODMn")] + [(x, gte == x) for x in GROUPS]:
        mm = m & keep_te
        grow.append(dict(method=meth, group=lab, coverage=float(np.mean(Tte[mm] > sw[mm])), n_eval=int(mm.sum()), mean_safe_weeks=float(sw[m].mean())))
cal_n = pd.Series(gva8[Lva.C.values >= K]).value_counts().rename("n_cal")
G8 = pd.DataFrame(grow); G8.to_csv(FD / "robust_mondrian8.csv", index=False)
pd.set_option("display.width", 220)
print(FR.round(3).to_string()); print(G8.round(3).to_string()); print(cal_n.to_string())

# ---------- H: within-group discrimination of the bound (the only place a hazard model can add to the partition) ----------
from sklearn.metrics import roc_auc_score
hrows = []
for model in ["BRR-SL", "Station KM", "Population KM"]:
    Fv, Ft = Fcum(model, "va"), Fcum(model, "te"); _, sg = bounds(Fv, Ft, 0.10, gva, gte, GROUPS)
    for x in GROUPS:
        m = (gte == x) & (safe | early); y = safe[m].astype(int)
        auc = roc_auc_score(y, sg[m] + 1e-6 * Ft[m][:, 3] * -1) if 0 < y.sum() < len(y) else np.nan    # ties broken by 4-week risk
        auc_raw = roc_auc_score(y, sg[m]) if 0 < y.sum() < len(y) else np.nan
        hrows.append(dict(model=model, group=x, auc_bound=auc_raw, auc_bound_tiebreak=auc, n_safe=int(y.sum()), n_early=int((1 - y).sum()),
                          safe_certified_8wk=float(np.mean(sg[(gte == x) & safe] >= 8)), early_certified_4wk=float(np.mean(sg[(gte == x) & early] >= 4))))
HW = pd.DataFrame(hrows); HW.to_csv(FD / "robust_within_group.csv", index=False); print(HW.round(3).to_string())
