"""Where does post-recovery excess sit, and how concentrated are breach regimes?
(1) Weekly re-breach hazard after recovery: observed versus memoryless nulls N1 (station rates) and N2 (station-year
    rates), weeks 1-26 (reuses the simulation in 25_relapse_null.py).
(2) Within-stratum Mantel-Haenszel onset rate ratios by weeks since the last breach (1, 2, 3-4, 5-8) versus > 8 weeks,
    stratified by station or by station-year; station bootstrap 95% CI.
(3) Concentration of new breaches across station-years (Lorenz curve) and how much of it is visible in advance from the
    previous calendar year's breach share."""
import importlib.util, numpy as np, pandas as pd
from pathlib import Path

HERE = Path(__file__).parent
src = (HERE / "25_relapse_null.py").read_text(encoding="utf-8")
src = src[:src.index("obs = run(1, \"observed\")")]                      # definitions only
ns = {"__file__": str(HERE / "25_relapse_null.py")}; exec(compile(src, "25_relapse_null_defs", "exec"), ns)
series, run_stats, pq_arrays, KMAX, MEM, rng = ns["series"], ns["stats"], ns["pq_arrays"], ns["KMAX"], ns["MEM"], np.random.default_rng(2511)
OUT = Path(__file__).resolve().parents[1] / "results"; RD = OUT / "relapse"; NSIM = 300


def hazards(mode, nsim):
    ATR = np.zeros((nsim, KMAX + 1)); EVT = np.zeros((nsim, KMAX + 1))
    for s, yr, ob, x in series:
        if mode == "observed":
            X = x[None, :].astype(np.int8)
        else:
            p, q = pq_arrays(yr, ob, x, by_year=(mode == "N2")); T = len(x); X = np.zeros((nsim, T), np.int8)
            X[:, 0] = x[0] if ob[0] else (rng.random(nsim) < p[0] / max(p[0] + q[0], 1e-9)); U = rng.random((nsim, T))
            for i in range(1, T): X[:, i] = np.where(X[:, i - 1] == 0, U[:, i] < p[i], U[:, i] >= q[i])
        _, _, atr, evt = run_stats(yr, ob, X); ATR += atr; EVT += evt
    with np.errstate(invalid="ignore", divide="ignore"):
        return EVT[:, 1:] / ATR[:, 1:], ATR[:, 1:]


rows = []
for mode in ("observed", "N1", "N2"):
    h, n = hazards(mode, 1 if mode == "observed" else NSIM)
    for k in range(KMAX):
        v = h[:, k]; rows.append(dict(model=mode, week=k + 1, hazard=np.nanmean(v), lo=np.nanquantile(v, 0.025), hi=np.nanquantile(v, 0.975), at_risk=n[:, k].mean()))
    print(mode, "hazard done", flush=True)
HZ = pd.DataFrame(rows)
# observed binomial CI for the observed hazard
o = HZ.model == "observed"; p_, n_ = HZ.loc[o, "hazard"], HZ.loc[o, "at_risk"]
HZ.loc[o, "lo"] = p_ - 1.96 * np.sqrt(p_ * (1 - p_) / n_); HZ.loc[o, "hi"] = p_ + 1.96 * np.sqrt(p_ * (1 - p_) / n_)
HZ.to_csv(RD / "null_hazard_by_week.csv", index=False)

# ---------- MH by weeks since last breach ----------
risk = []
for s, yr, ob, x in series:
    last = -999
    for i in range(1, len(x)):
        if ob[i] and ob[i - 1] and x[i - 1] == 0:
            gap = (i - 1) - last
            b = "1" if gap == 1 else "2" if gap == 2 else "3–4" if gap <= 4 else "5–8" if gap <= MEM else "ref"
            risk.append((s, yr[i], b, int(x[i] == 1)))
        if ob[i] and x[i] == 1: last = i
RK = pd.DataFrame(risk, columns=["site", "year", "bin", "onset"])


def mh(df, strata, b):
    d = df[df.bin.isin([b, "ref"])].assign(e=lambda t: (t.bin == b).astype(int))
    g = d.groupby(strata + ["e"]).onset.agg(["sum", "count"]).unstack("e").dropna()
    a, n1, c, n0 = g[("sum", 1)], g[("count", 1)], g[("sum", 0)], g[("count", 0)]; N = n1 + n0
    return (a * n0 / N).sum() / (c * n1 / N).sum()


sites = RK.site.unique(); grp = {s: RK[RK.site == s] for s in sites}; mrows = []
boots = [pd.concat([grp[s].assign(site=f"{s}_{j}") for j, s in enumerate(rng.choice(sites, len(sites)))]) for _ in range(200)]
for strata, sname in [(["site"], "station"), (["site", "year"], "station-year")]:
    for b in ["1", "2", "3–4", "5–8"]:
        est = mh(RK, strata, b); bs = [mh(D, strata, b) for D in boots]
        mrows.append(dict(strata=sname, weeks_since_breach=b, mh_rr=est, lo=np.nanquantile(bs, 0.025), hi=np.nanquantile(bs, 0.975),
                          crude_rr=RK[RK.bin == b].onset.mean() / RK[RK.bin == "ref"].onset.mean(), n_exposed=int((RK.bin == b).sum())))
        print(sname, b, round(est, 2), flush=True)
pd.DataFrame(mrows).to_csv(RD / "mh_rate_ratio_by_week.csv", index=False)

# ---------- concentration of new breaches across station-years ----------
on = []
for s, yr, ob, x in series:
    pair = ob[1:] & ob[:-1]; onset = pair & (x[1:] == 1) & (x[:-1] == 0)
    obs_wk = pd.Series(ob, index=yr).groupby(level=0).sum(); brk = pd.Series(np.where(ob, x, 0), index=yr).groupby(level=0).sum()
    ons = pd.Series(onset.astype(int), index=yr[1:]).groupby(level=0).sum()
    for y in obs_wk.index:
        on.append(dict(site=s, year=y, onsets=int(ons.get(y, 0)), obs_weeks=int(obs_wk[y]), breach_weeks=int(brk[y])))
SY = pd.DataFrame(on); SY = SY[SY.obs_weeks >= 10].copy()
SY["share"] = SY.breach_weeks / SY.obs_weeks
SY = SY.sort_values(["site", "year"]); SY["prev_share"] = SY.groupby("site").share.shift(1)
srt = SY.sort_values("onsets", ascending=False); cum = srt.onsets.cumsum() / srt.onsets.sum(); frac = np.arange(1, len(srt) + 1) / len(srt)
pd.DataFrame(dict(frac_station_years=frac, frac_onsets=cum.values)).to_csv(RD / "onset_lorenz.csv", index=False)
top10 = cum.values[int(np.ceil(0.10 * len(srt))) - 1]; top20 = cum.values[int(np.ceil(0.20 * len(srt))) - 1]
xs = np.sort(SY.onsets.values); nn = len(xs); gini = np.sum((2 * np.arange(1, nn + 1) - nn - 1) * xs) / (nn * xs.sum())
P = SY.dropna(subset=["prev_share"]); chronic_prev = P.prev_share > 0.10
pd.Series(dict(n_station_years=len(SY), share_with_onset=(SY.onsets > 0).mean(), top10_share=top10, top20_share=top20, gini=gini,
               prev_chronic_frac_station_years=chronic_prev.mean(), prev_chronic_frac_onsets=P.onsets[chronic_prev].sum() / P.onsets.sum())).to_csv(RD / "onset_concentration.csv")
print(pd.read_csv(RD / "onset_concentration.csv").to_string())
print(HZ[HZ.week <= 8].round(3).to_string())
