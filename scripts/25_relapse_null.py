"""Is relapse more than station chronicity? Memoryless null models for the relapse headline numbers.

Null N1 / N2: each station follows a two-state weekly Markov chain whose onset and recovery probabilities are the
station's own (N1) or the station-year's own (N2) observed rates. The latent chain runs through every calendar week and
is observed with the station's real monitoring mask, so gaps, censoring and station heterogeneity are reproduced, but
the onset probability does not depend on time since the last breach. Statistics are computed exactly as in
24_spatial_aggregates.py / 24b_rebreach_km.py: share of new breaches that are relapses (onset <= 8 weeks after the last
observed breach) and Kaplan-Meier re-breach within 4 / 8 weeks of a recovery (censored at the first gap).

Within-station Mantel-Haenszel rate ratio: onset per at-risk compliant week inside the 8-week window versus outside it
(and versus long-compliant weeks, > 52 weeks), stratified by station or station-year; station bootstrap 95% CI."""
import numpy as np, pandas as pd
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "results"; RD = OUT / "relapse"
w = pd.read_parquet(OUT / "cnemc_weekly.parquet"); w = w[w.date.dt.year >= 2008]
MEM, KMAX, NSIM = 8, 26, 500
rng = np.random.default_rng(2510)

series = []
for s, g in w.groupby("site"):
    g = g.set_index("date").sort_index(); idx = pd.date_range(g.index.min(), g.index.max(), freq="W-MON"); g = g.reindex(idx)
    ob = g[["NH4N", "DO", "CODMn"]].notna().all(axis=1).values
    x = np.where(ob, ((g.NH4N > 1) | (g.DO < 5) | (g.CODMn > 6)).values, False).astype(np.int8)
    series.append((s, idx.year.values, ob, x))


def rates(yr, ob, x, key):
    """Transition counts on consecutive observed pairs, by stratum key (array aligned with week i)."""
    pair = ob[1:] & ob[:-1]; a, b, k = x[:-1][pair], x[1:][pair], key[1:][pair]
    df = pd.DataFrame(dict(k=k, a=a, b=b))
    t = df.groupby("k").apply(lambda d: pd.Series(dict(n0=(d.a == 0).sum(), n01=((d.a == 0) & (d.b == 1)).sum(),
                                                        n1=(d.a == 1).sum(), n10=((d.a == 1) & (d.b == 0)).sum())), include_groups=False)
    return t


GLOBAL = pd.concat([rates(yr, ob, x, np.zeros(len(x), int)) for _, yr, ob, x in series]).sum()
P0, Q0 = GLOBAL.n01 / GLOBAL.n0, GLOBAL.n10 / GLOBAL.n1


def pq_arrays(yr, ob, x, by_year):
    key = yr if by_year else np.zeros(len(x), int)
    t = rates(yr, ob, x, key); st_ = rates(yr, ob, x, np.zeros(len(x), int)).iloc[0]
    ps = (st_.n01 + P0 * 2) / (st_.n0 + 2); qs = (st_.n10 + Q0 * 2) / (st_.n1 + 2)       # light shrinkage to the global rate
    p = np.full(len(x), ps); q = np.full(len(x), qs)
    if by_year:
        for k_, r in t.iterrows():
            m = key == k_
            p[m] = (r.n01 + ps * 2) / (r.n0 + 2) if r.n0 > 0 else ps
            q[m] = (r.n10 + qs * 2) / (r.n1 + 2) if r.n1 > 0 else qs
    return p, q


def stats(yr, ob, X):
    """X: (nsim, T) latent states. Returns per-sim relapse/onset counts by year and KM at-risk/event counts."""
    n, T = X.shape
    last = np.full(n, -999); on = {}; rel = {}
    act = np.zeros(n, bool); off = np.zeros(n, int); atr = np.zeros((n, KMAX + 1)); evt = np.zeros((n, KMAX + 1))
    for i in range(T):
        # continuing post-recovery runs
        if i > 0:
            off = np.where(act, off + 1, off)
            if not ob[i]:
                act[:] = False
            else:
                m = act & (off <= KMAX)
                atr[m, off[m]] += 1
                e = m & (X[:, i] == 1); evt[e, off[e]] += 1
                act = act & (X[:, i] == 0) & (off < KMAX)
        if i > 0 and ob[i] and ob[i - 1]:
            o = (X[:, i] == 1) & (X[:, i - 1] == 0)
            r_ = o & ((i - 1 - last) >= 1) & ((i - 1 - last) <= MEM)
            y = yr[i]; on[y] = on.get(y, 0) + o.astype(int); rel[y] = rel.get(y, 0) + r_.astype(int)
            rec = (X[:, i] == 0) & (X[:, i - 1] == 1)
            act = act | rec; off = np.where(rec, 0, off)
        if ob[i]:
            last = np.where(X[:, i] == 1, i, last)
    return on, rel, atr, evt


def km(atr, evt, k):
    with np.errstate(invalid="ignore", divide="ignore"):
        h = np.where(atr[:, 1:k + 1] > 0, evt[:, 1:k + 1] / atr[:, 1:k + 1], 0)
    return 1 - np.prod(1 - h, axis=1)


def run(nsim, mode):
    ON, REL = {}, {}; ATR = np.zeros((nsim, KMAX + 1)); EVT = np.zeros((nsim, KMAX + 1))
    for s, yr, ob, x in series:
        if mode == "observed":
            X = x[None, :].astype(np.int8)
        else:
            p, q = pq_arrays(yr, ob, x, by_year=(mode == "N2"))
            T = len(x); X = np.zeros((nsim, T), np.int8)
            pi = p[0] / (p[0] + q[0]) if p[0] + q[0] > 0 else 0
            X[:, 0] = x[0] if ob[0] else (rng.random(nsim) < pi)
            U = rng.random((nsim, T))
            for i in range(1, T):
                X[:, i] = np.where(X[:, i - 1] == 0, U[:, i] < p[i], U[:, i] >= q[i])
        on, rel, atr, evt = stats(yr, ob, X)
        for y in on: ON[y] = ON.get(y, 0) + on[y]; REL[y] = REL.get(y, 0) + rel[y]
        ATR += atr; EVT += evt
    yrs = sorted(ON); on_tot = sum(ON[y] for y in yrs); rel_tot = sum(REL[y] for y in yrs)
    return dict(share=rel_tot / on_tot, n_on=on_tot, n_rel=rel_tot, F4=km(ATR, EVT, 4), F8=km(ATR, EVT, 8),
                by_year={y: REL[y] / np.maximum(ON[y], 1) for y in yrs})


obs = run(1, "observed")
res = {"observed": obs}
for mode in ("N1", "N2"):
    res[mode] = run(NSIM, mode); print(mode, "done", flush=True)

rows = []
for mode, r in res.items():
    for stat in ("share", "F4", "F8"):
        v = np.atleast_1d(r[stat]).astype(float)
        rows.append(dict(model=mode, stat=stat, value=v.mean(), lo=np.quantile(v, 0.025), hi=np.quantile(v, 0.975),
                         p_ge_obs=float(np.mean(v >= np.atleast_1d(obs[stat])[0])) if mode != "observed" else np.nan))
    v = np.atleast_1d(r["n_rel"]).astype(float); rows.append(dict(model=mode, stat="n_relapse", value=v.mean(), lo=np.quantile(v, 0.025), hi=np.quantile(v, 0.975)))
    v = np.atleast_1d(r["n_on"]).astype(float); rows.append(dict(model=mode, stat="n_onset", value=v.mean(), lo=np.quantile(v, 0.025), hi=np.quantile(v, 0.975)))
NM = pd.DataFrame(rows); NM.to_csv(RD / "null_model_summary.csv", index=False)
yr_rows = []
for mode, r in res.items():
    for y, v in r["by_year"].items():
        v = np.atleast_1d(v).astype(float); yr_rows.append(dict(model=mode, year=y, share=v.mean(), lo=np.quantile(v, 0.025), hi=np.quantile(v, 0.975)))
pd.DataFrame(yr_rows).to_csv(RD / "null_model_by_year.csv", index=False)
print(NM.round(4).to_string(), flush=True)

# ---------- within-station Mantel-Haenszel rate ratios ----------
risk = []
for s, yr, ob, x in series:
    last = -999
    for i in range(1, len(x)):
        if ob[i] and ob[i - 1] and x[i - 1] == 0:
            gap = (i - 1) - last
            cat = "window" if 1 <= gap <= MEM else ("long" if gap > 52 else "mid")
            risk.append((s, yr[i], cat, int(x[i] == 1)))
        if ob[i] and x[i] == 1: last = i
RK = pd.DataFrame(risk, columns=["site", "year", "cat", "onset"])


def mh(df, strata, ref):
    d = df[df.cat.isin(["window"] + ref)].assign(win=lambda t: (t.cat == "window").astype(int))
    g = d.groupby(strata + ["win"]).onset.agg(["sum", "count"]).unstack("win").dropna()
    a, n1, b, n0 = g[("sum", 1)], g[("count", 1)], g[("sum", 0)], g[("count", 0)]; N = n1 + n0
    return (a * n0 / N).sum() / (b * n1 / N).sum()


sites = RK.site.unique(); grp = {s: RK[RK.site == s] for s in sites}; mh_rows = []
for strata, sname in [(["site"], "station"), (["site", "year"], "station-year")]:
    for ref, rname in [(["mid", "long"], "outside window (> 8 weeks)"), (["long"], "long-compliant (> 52 weeks)")]:
        est = mh(RK, strata, ref)
        bs = []
        for b in range(300):
            pick = rng.choice(sites, len(sites))
            D = pd.concat([grp[s].assign(site=f"{s}_{j}") for j, s in enumerate(pick)])
            bs.append(mh(D, strata, ref))
        crude = RK[RK.cat == "window"].onset.mean() / RK[RK.cat.isin(ref)].onset.mean()
        mh_rows.append(dict(strata=sname, reference=rname, mh_rr=est, lo=np.nanquantile(bs, 0.025), hi=np.nanquantile(bs, 0.975), crude_rr=crude))
        print(sname, rname, round(est, 2), round(crude, 2), flush=True)
pd.DataFrame(mh_rows).to_csv(RD / "mh_rate_ratio.csv", index=False)
