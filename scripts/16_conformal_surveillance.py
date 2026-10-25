"""Fig. 7 analyses with the final BRR-SL onset hazards: conformal safe weeks (uncalibrated / split / weighted),
safe-week ECDF by start state, dense surveillance capture curves, and two station timelines."""
import sys, numpy as np, pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, str(Path(__file__).parent))
from survlib import *
from tristate import start_state

OUT = Path(__file__).resolve().parents[1] / "results"; CORE = OUT / "core"; FD = OUT / "fig7"; FD.mkdir(exist_ok=True)
L = pd.read_parquet(OUT / "landmarks.parquet").reset_index(drop=True); state = L.state.values.astype(int)
H = pd.read_pickle(CORE / "hazards.pkl")["onset"]; BF = pd.read_pickle(CORE / "baseline_F_test.pkl")
va = ((L.date >= "2014-01-01") & (L.date <= "2014-09-30")).values; te = (L.date >= "2015-01-01").values
Lva = L[va & (state == 0)].reset_index(drop=True); Lte = L[te & (state == 0)].reset_index(drop=True)
hva, hte = H["BRR-SL"]["va"], H["BRR-SL"]["te"]
s3te = start_state(Lte.state.values.astype(int), Lte.wk_since_ex.values)


def qtime(h, beta):
    F = 1 - surv_from_h(h); q = (F >= beta).argmax(1) + 1; q[F[:, -1] < beta] = K + 1; return q


def lpb(h, beta):
    return np.maximum(np.minimum(qtime(h, beta), K) - 1, 0)       # breach-free weeks guaranteed


def truth(Lx):
    T = Lx["T"].values; E = Lx["E"].values; return np.where(E == 1, T, K + 1)   # valid when C >= K


ZC = ["ex_rate52", "margin_min", "woy_sin", "woy_cos", "lat", "lon", "obs_frac12"]
def pC(Lx):
    Z = Lx[ZC].fillna(0).values; y = (Lx.C.values >= K).astype(int)
    m = LogisticRegression(max_iter=2000).fit(Z, y); return np.clip(m.predict_proba(Z)[:, 1], 0.05, 1)


def calibrate(h, Lc, alpha, weights=None):
    keep = Lc.C.values >= K; Tt = truth(Lc)[keep]; wts = np.ones(keep.sum()) if weights is None else weights[keep]
    for beta in np.linspace(0.95, 0.001, 950):
        miss = np.sum(wts * (Tt <= lpb(h[keep], beta))) / wts.sum()
        if miss <= alpha * (1 + 1 / keep.sum()) - 1 / keep.sum(): return beta   # finite-sample correction
    return 0.001


wva = 1 / pC(Lva)
yr = Lte.date.dt.year.values; dense = yr <= 2016
Ld = Lte[dense].reset_index(drop=True); hd = hte[dense]; keep = Ld.C.values >= K
w_te = 1 / pC(Ld)                                                   # IPW to the full 2015-2016 landmark population
res = []
for alpha in (0.05, 0.10, 0.15, 0.20):
    for meth, beta in [("Uncalibrated", alpha), ("Split conformal", calibrate(hva, Lva, alpha)),
                       ("Weighted conformal", calibrate(hva, Lva, alpha, wva))]:
        sw = lpb(hd, beta); cov = np.mean(truth(Ld)[keep] > sw[keep])
        cov_w = np.sum(w_te[keep] * (truth(Ld)[keep] > sw[keep])) / w_te[keep].sum()
        res.append(dict(alpha=alpha, nominal=1 - alpha, method=meth, beta=beta, coverage=cov, coverage_ipw=cov_w,
                        mean_safe_weeks=sw.mean(), mean_safe_weeks_eval=sw[keep].mean(), n_eval=int(keep.sum()), n_all=len(sw)))
CF = pd.DataFrame(res); CF.to_csv(FD / "conformal.csv", index=False); print(CF.round(4).to_string())

beta10 = CF[(CF.alpha == 0.10) & (CF.method == "Weighted conformal")].beta.iloc[0]
sw_all = lpb(hte, beta10)
ec = []
for lab, m in [("A start", s3te == 0), ("R start", s3te >= 2)]:
    v = np.sort(sw_all[m & (yr <= 2016)])
    for x in range(0, K):
        ec.append(dict(group=lab, safe_weeks=x, ecdf=np.mean(v <= x), n=len(v), mean=v.mean()))
pd.DataFrame(ec).to_csv(FD / "safe_weeks_ecdf.csv", index=False)
print(pd.DataFrame(ec).groupby("group")[["mean", "n"]].first())

# surveillance capture (4-week horizon), dense grid
Fst = 1 - surv_from_h(hte); y4, kn = horizon_labels(Lte, 4)
scores = {"BRR-SL": Fst[:, 3], "Forecast-threshold (max)": BF["onset_FTmax"][:, 3],
          "Station KM": (1 - surv_from_h(H["Station KM"]["te"]))[:, 3],
          "Recent-breach rule": Lte.ex_rate52.fillna(0).values + 1e-3 * (1 - Lte.margin_min.fillna(1).values)}
rng = np.random.default_rng(16); scores["Random"] = rng.random(len(Lte))
wk = Lte.date.values; cap = []
for q in np.round(np.arange(0.02, 0.51, 0.02), 2):
    for nm, sc in scores.items():
        hit = tot = 0
        for d in np.unique(wk):
            ix = (wk == d) & kn
            if ix.sum() < 5: continue
            k_ = max(1, int(round(q * ix.sum()))); top = np.argsort(-sc[ix])[:k_]
            hit += y4[ix][top].sum(); tot += y4[ix].sum()
        cap.append(dict(q=q, method=nm, capture=hit / max(tot, 1)))
CP = pd.DataFrame(cap); CP.to_csv(FD / "surveillance_dense.csv", index=False)
print(CP[CP.q.isin([0.1, 0.2, 0.3])].pivot(index="method", columns="q", values="capture").round(3))

# station timelines (2015-2016): weekly state from the observed series, 4-week risk and safe weeks for compliant weeks
O = np.load(OUT / "obs_windows.npy"); Lall = L[te & (L.date.dt.year <= 2016)].copy()
cand = Lte[dense].groupby("site").agg(n=("T", "size"))
cand["brk"] = Lall.groupby("site").state.sum().reindex(cand.index).fillna(0)
cand = cand[cand.n >= 30]; print("timeline candidates", len(cand))
chronic = cand.sort_values(["brk", "n"], ascending=False).index[0]
occ_c = cand[(cand.brk >= 2) & (cand.brk <= 6)]
picks = {"chronic": chronic, "occasional": occ_c.sort_values("n", ascending=False).index[0]}
tl = []
for lab, s in picks.items():
    g = Lall[Lall.site == s].sort_values("date")
    idx = pd.date_range("2015-01-05", "2016-12-26", freq="W-MON")
    on = Lte[(Lte.site == s) & dense]; pos = {d: i for i, d in enumerate(Lte.date[(Lte.site == s)].index)}
    sel = (Lte.site.values == s) & dense
    risk = pd.Series(Fst[sel, 3], index=Lte.date.values[sel]); safe = pd.Series(sw_all[sel], index=Lte.date.values[sel])
    st_ = pd.Series(g.state.values, index=g.date.values)
    s3g = pd.Series(start_state(g.state.values.astype(int), g.wk_since_ex.values), index=g.date.values)
    for d in idx:
        tl.append(dict(case=lab, site=s, date=d, state=st_.get(d, np.nan), s3=s3g.get(d, np.nan), risk4=risk.get(d, np.nan), safe_weeks=safe.get(d, np.nan)))
pd.DataFrame(tl).to_csv(FD / "timelines.csv", index=False); print("timeline sites:", picks)
