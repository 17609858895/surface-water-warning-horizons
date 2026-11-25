"""Reviewer checks on the memoryless nulls (1): time-scale dependence, prospective and baseline-rate nulls, and the
null reference distribution of the within-stratum Mantel-Haenszel rate ratios.

Variants (two-state weekly chains run through every calendar week and observed through the real masks):
  N1      station rates
  N2      station-year rates (retrospective, as in the paper)
  N2h     station x half-year rates
  N2q     station x quarter rates
  Nprosp  prospective rates estimated from the station's preceding 52 calendar weeks only
  Nbase   station-year recovery rates, but onset rates estimated only from compliant weeks > 8 weeks after the last
          observed breach (no post-recovery excess by construction)
For every simulated set the same statistics as the observed data are computed: relapse share, KM re-breach at 4 and 8
weeks, week-1 and week-2 re-breach hazards, and station-year Mantel-Haenszel ratios for weeks 1, 2, 3-4, 5-8 since the
last breach versus > 8 weeks."""
import numpy as np, pandas as pd
from pathlib import Path

HERE = Path(__file__).parent
src = (HERE / "25_relapse_null.py").read_text(encoding="utf-8")
src = src[:src.index('obs = run(1, "observed")')]
ns = {"__file__": str(HERE / "25_relapse_null.py")}; exec(compile(src, "25_defs", "exec"), ns)
series, MEM, KMAX = ns["series"], ns["MEM"], ns["KMAX"]
OUT = Path(__file__).resolve().parents[1] / "results" / "review"; OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(2701); NSIM = 200; NB = 5   # bins: 1, 2, 3-4, 5-8, ref(>8 or none)


def bin_of(gap):
    return np.where(gap == 1, 0, np.where(gap == 2, 1, np.where(gap <= 4, 2, np.where(gap <= MEM, 3, 4))))


def stats_mh(yr, ob, X):
    """Relapse share pieces, KM counts, and per-year MH bin counts for each simulation row of X."""
    n, T = X.shape; years = np.unique(yr); yi = np.searchsorted(years, yr)
    last = np.full(n, -10 ** 6); on = np.zeros(n); rel = np.zeros(n)
    act = np.zeros(n, bool); off = np.zeros(n, int); atr = np.zeros((n, KMAX + 1)); evt = np.zeros((n, KMAX + 1))
    ex = np.zeros((n, len(years), NB)); ev = np.zeros((n, len(years), NB)); r = np.arange(n)
    for i in range(T):
        if i > 0:
            off = np.where(act, off + 1, off)
            if not ob[i]:
                act[:] = False
            else:
                m = act & (off <= KMAX); atr[m, off[m]] += 1
                e = m & (X[:, i] == 1); evt[e, off[e]] += 1
                act = act & (X[:, i] == 0) & (off < KMAX)
        if i > 0 and ob[i] and ob[i - 1]:
            comp = X[:, i - 1] == 0
            gap = (i - 1) - last; b = bin_of(np.where(last < -10 ** 5, 10 ** 6, gap))
            ex[r[comp], yi[i], b[comp]] += 1
            o = comp & (X[:, i] == 1)
            ev[r[o], yi[i], b[o]] += 1
            on += o; rel += o & (gap >= 1) & (gap <= MEM)
            rec = (X[:, i] == 0) & (X[:, i - 1] == 1); act = act | rec; off = np.where(rec, 0, off)
        if ob[i]:
            last = np.where(X[:, i] == 1, i, last)
    return on, rel, atr, evt, ex, ev


def mh(ex, ev):
    """ex, ev: (nsim, strata, NB). Returns (nsim, 4) MH ratios of bins 0-3 versus bin 4."""
    out = np.full((ex.shape[0], 4), np.nan)
    for b in range(4):
        n1, n0 = ex[:, :, b], ex[:, :, 4]; a, c = ev[:, :, b], ev[:, :, 4]; N = n1 + n0
        ok = (n1 > 0) & (n0 > 0)
        num = np.where(ok, a * n0 / np.where(N > 0, N, 1), 0).sum(1); den = np.where(ok, c * n1 / np.where(N > 0, N, 1), 0).sum(1)
        out[:, b] = num / np.where(den > 0, den, np.nan)
    return out


def pairs_counts(ob, x):
    pair = np.zeros(len(x), bool); pair[1:] = ob[1:] & ob[:-1]
    a = np.zeros(len(x)); b = np.zeros(len(x)); a[1:] = x[:-1]; b[1:] = x[1:]
    n0 = pair & (a == 0); n01 = n0 & (b == 1); n1 = pair & (a == 1); n10 = n1 & (b == 0)
    return n0.astype(float), n01.astype(float), n1.astype(float), n10.astype(float)


G = np.zeros(4)
for _, yr, ob, x in series:
    G += np.array([c.sum() for c in pairs_counts(ob, x)])
P0, Q0 = G[1] / G[0], G[3] / G[2]


def rates_by_key(key, c, prior_p, prior_q, k=2.0):
    n0, n01, n1, n10 = c; p = np.empty(len(key)); q = np.empty(len(key))
    for u in np.unique(key):
        m = key == u
        p[m] = (n01[m].sum() + k * prior_p) / (n0[m].sum() + k); q[m] = (n10[m].sum() + k * prior_q) / (n1[m].sum() + k)
    return p, q


def pq(mode, yr, ob, x, dates):
    c = pairs_counts(ob, x); n0, n01, n1, n10 = c
    ps = (n01.sum() + 2 * P0) / (n0.sum() + 2); qs = (n10.sum() + 2 * Q0) / (n1.sum() + 2)
    if mode == "N1": return np.full(len(x), ps), np.full(len(x), qs)
    if mode == "N2": return rates_by_key(yr, c, ps, qs)
    if mode == "N2h": return rates_by_key(yr * 2 + (dates.dayofyear >= 182), c, ps, qs)
    if mode == "N2q": return rates_by_key(yr * 4 + (dates.quarter - 1), c, ps, qs)
    if mode == "Nprosp":
        cs = [np.r_[0, np.cumsum(v)] for v in c]; i = np.arange(len(x)); lo = np.maximum(i - 52, 0)
        w = [cs[j][i] - cs[j][lo] for j in range(4)]           # pairs ending in weeks [i-52, i-1]
        return (w[1] + 2 * ps) / (w[0] + 2), (w[3] + 2 * qs) / (w[2] + 2)
    if mode == "Nbase":
        last = -10 ** 6; refe = np.zeros(len(x)); refo = np.zeros(len(x))
        for i in range(1, len(x)):
            if ob[i] and ob[i - 1] and x[i - 1] == 0 and ((i - 1) - last > MEM):
                refe[i] = 1; refo[i] = x[i]
            if ob[i] and x[i] == 1: last = i
        pref_s = (refo.sum() + 2 * PREF0) / (refe.sum() + 2)
        p, q = rates_by_key(yr, c, ps, qs)
        for u in np.unique(yr):
            m = yr == u; p[m] = (refo[m].sum() + 2 * pref_s) / (refe[m].sum() + 2)
        return p, q
    raise ValueError(mode)


# network reference-bin onset rate
ro = re_ = 0.0
for _, yr, ob, x in series:
    last = -10 ** 6
    for i in range(1, len(x)):
        if ob[i] and ob[i - 1] and x[i - 1] == 0 and ((i - 1) - last > MEM): re_ += 1; ro += x[i]
        if ob[i] and x[i] == 1: last = i
PREF0 = ro / re_

DATES = {}
w = pd.read_parquet(Path(__file__).resolve().parents[1] / "results" / "cnemc_weekly.parquet"); w = w[w.date.dt.year >= 2008]
for s, g in w.groupby("site"):
    DATES[s] = pd.date_range(g.date.min(), g.date.max(), freq="W-MON")


def run(mode, nsim):
    ON = REL = 0; ATR = 0; EVT = 0; EX = []; EV = []
    for s, yr, ob, x in series:
        if mode == "observed":
            X = x[None, :].astype(np.int8)
        else:
            p, q = pq(mode, yr, ob, x, DATES[s]); T = len(x); X = np.zeros((nsim, T), np.int8)
            X[:, 0] = x[0] if ob[0] else (rng.random(nsim) < p[0] / max(p[0] + q[0], 1e-9)); U = rng.random((nsim, T))
            for i in range(1, T): X[:, i] = np.where(X[:, i - 1] == 0, U[:, i] < p[i], U[:, i] >= q[i])
        on, rel, atr, evt, ex, ev = stats_mh(yr, ob, X)
        ON = ON + on; REL = REL + rel; ATR = ATR + atr; EVT = EVT + evt; EX.append(ex); EV.append(ev)
    EX = np.concatenate(EX, axis=1); EV = np.concatenate(EV, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        hz = np.where(ATR[:, 1:] > 0, EVT[:, 1:] / ATR[:, 1:], np.nan)
    F = 1 - np.cumprod(1 - np.nan_to_num(hz), axis=1)
    return dict(share=REL / ON, F4=F[:, 3], F8=F[:, 7], h1=hz[:, 0], h2=hz[:, 1], mh=mh(EX, EV))


res = {"observed": run("observed", 1)}
for mode in ["N1", "N2", "N2h", "N2q", "Nprosp", "Nbase"]:
    res[mode] = run(mode, NSIM); print(mode, "done", flush=True)
rows, mrows = [], []
obs = res["observed"]
for mode, r in res.items():
    for st in ["share", "F4", "F8", "h1", "h2"]:
        v = np.asarray(r[st], float)
        rows.append(dict(model=mode, stat=st, value=np.nanmean(v), lo=np.nanquantile(v, 0.025), hi=np.nanquantile(v, 0.975),
                         p_ge_obs=np.nan if mode == "observed" else float(np.mean(v >= obs[st][0]))))
    for b, lab in enumerate(["1", "2", "3-4", "5-8"]):
        v = r["mh"][:, b]
        mrows.append(dict(model=mode, bin=lab, mh=np.nanmean(v), lo=np.nanquantile(v, 0.025), hi=np.nanquantile(v, 0.975),
                          p_ge_obs=np.nan if mode == "observed" else float(np.mean(v >= obs["mh"][0, b]))))
pd.DataFrame(rows).to_csv(OUT / "null_variants.csv", index=False); pd.DataFrame(mrows).to_csv(OUT / "null_mh_reference.csv", index=False)
pd.set_option("display.width", 200)
print(pd.DataFrame(rows).round(4).to_string()); print(pd.DataFrame(mrows).round(3).to_string())
