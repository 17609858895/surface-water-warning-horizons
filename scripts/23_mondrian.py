"""Group-conditional (Mondrian) conformal safe weeks (Fig. 7 update, Fig. S7).
The marginal weighted bound under-covers recovered (R-start) and chronic stations. Here the quantile level beta is
calibrated separately within groups defined at the landmark (start state A/R x chronic 52-week share > 0.10),
using 2014 validation landmarks with C >= 12 and inverse-probability weights 1/P(C>=12|X) within each group.
Evaluation: 2015-2016 onset landmarks with C >= 12 (truth observable)."""
import sys, numpy as np, pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, str(Path(__file__).parent))
from survlib import *
from tristate import start_state

OUT = Path(__file__).resolve().parents[1] / "results"; FD = OUT / "fig7"
L = pd.read_parquet(OUT / "landmarks.parquet").reset_index(drop=True); state = L.state.values.astype(int)
H = pd.read_pickle(OUT / "core" / "hazards.pkl")["onset"]
va = ((L.date >= "2014-01-01") & (L.date <= "2014-09-30")).values & (state == 0)
te = (L.date >= "2015-01-01").values & (state == 0)
Lva = L[va].reset_index(drop=True); Lte = L[te].reset_index(drop=True)
hva, hte = H["BRR-SL"]["va"], H["BRR-SL"]["te"]
dense = Lte.date.dt.year.values <= 2016; Lte, hte = Lte[dense].reset_index(drop=True), hte[dense]
ZC = ["ex_rate52", "margin_min", "woy_sin", "woy_cos", "lat", "lon", "obs_frac12"]


def groups(Lx):
    s3 = start_state(Lx.state.values.astype(int), Lx.wk_since_ex.values); R = s3 >= 2
    ch = Lx.ex_rate52.fillna(0).values > 0.10
    return np.where(R, np.where(ch, "R / chronic", "R / non-chronic"), np.where(ch, "A / chronic", "A / non-chronic")), R, ch


def qtime(h, beta):
    F = 1 - surv_from_h(h); q = (F >= beta).argmax(1) + 1; q[F[:, -1] < beta] = K + 1; return q


def lpb(h, beta): return np.maximum(np.minimum(qtime(h, beta), K) - 1, 0)
def truth(Lx): return np.where(Lx.E.values == 1, Lx["T"].values, K + 1)


def ipw(Lx):
    y = (Lx.C.values >= K).astype(int)
    if y.min() == y.max(): return np.ones(len(Lx))
    m = LogisticRegression(max_iter=2000).fit(Lx[ZC].fillna(0).values, y)
    return 1 / np.clip(m.predict_proba(Lx[ZC].fillna(0).values)[:, 1], 0.05, 1)


def calibrate(h, Lc, alpha, w):
    keep = Lc.C.values >= K; Tt = truth(Lc)[keep]; ww = w[keep]; n = keep.sum()
    if n < 30: return np.nan, n
    for beta in np.linspace(0.95, 0.001, 950):
        miss = np.sum(ww * (Tt <= lpb(h[keep], beta))) / ww.sum()
        if miss <= alpha * (1 + 1 / n) - 1 / n: return beta, n
    return 0.001, n


gva, _, _ = groups(Lva); gte, Rte, chte = groups(Lte)
wva_all = ipw(Lva); keep_te = Lte.C.values >= K; Tte = truth(Lte)
rows, cond = [], []
for alpha in (0.05, 0.10, 0.15, 0.20):
    b_marg, _ = calibrate(hva, Lva, alpha, wva_all)
    betas = {}
    for g in np.unique(gva):
        m = gva == g; b, n = calibrate(hva[m], Lva[m].reset_index(drop=True), alpha, ipw(Lva[m].reset_index(drop=True)))
        betas[g] = (b if np.isfinite(b) else b_marg, n)
    sw_m = lpb(hte, b_marg)
    sw_g = np.zeros(len(Lte), int)
    for g in np.unique(gte):
        m = gte == g; sw_g[m] = lpb(hte[m], betas.get(g, (b_marg, 0))[0])
    for meth, sw in [("Weighted conformal (marginal)", sw_m), ("Mondrian conformal (state × chronic)", sw_g)]:
        cov = np.mean(Tte[keep_te] > sw[keep_te])
        rows.append(dict(alpha=alpha, nominal=1 - alpha, method=meth, coverage=cov, mean_safe_weeks=sw.mean(), n_eval=int(keep_te.sum())))
        if alpha == 0.10:
            for gname, m in [("All", np.ones(len(Lte), bool)), ("A start", ~Rte), ("R start", Rte), ("Chronic", chte), ("Non-chronic", ~chte),
                             ("A / non-chronic", gte == "A / non-chronic"), ("A / chronic", gte == "A / chronic"),
                             ("R / non-chronic", gte == "R / non-chronic"), ("R / chronic", gte == "R / chronic")]:
                mm = m & keep_te; k_ = Tte[mm] > sw[mm]
                bs = [np.random.default_rng(b).choice(k_, len(k_)).mean() for b in range(400)] if len(k_) > 20 else [np.nan]
                cond.append(dict(method=meth, group=gname, coverage=k_.mean() if len(k_) else np.nan, lo=np.nanquantile(bs, 0.025),
                                 hi=np.nanquantile(bs, 0.975), mean_safe_weeks=sw[m].mean(), n_eval=int(mm.sum()), n_all=int(m.sum())))
    if alpha == 0.10:
        pd.DataFrame([dict(group=g, beta=b, n_cal=n) for g, (b, n) in betas.items()] + [dict(group="marginal", beta=b_marg, n_cal=int((Lva.C >= K).sum()))]).to_csv(FD / "mondrian_betas.csv", index=False)
        np.save(FD / "safe_weeks_mondrian_test.npy", sw_g)
        Lte.assign(safe_mondrian=sw_g, safe_marginal=sw_m, group=gte)[["site", "date", "group", "safe_mondrian", "safe_marginal"]].to_csv(FD / "safe_weeks_by_landmark.csv", index=False)
RS = pd.DataFrame(rows); RS.to_csv(FD / "mondrian_coverage.csv", index=False)
CD = pd.DataFrame(cond); CD.to_csv(FD / "conditional_coverage.csv", index=False)
print(RS.round(3).to_string()); print(CD.round(3).to_string()); print(pd.read_csv(FD / "mondrian_betas.csv").round(3).to_string())
