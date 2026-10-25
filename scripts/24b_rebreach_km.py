"""Cumulative re-breach after recovery (Kaplan-Meier, censored at monitoring gaps) per indicator and hysteresis band,
and the share of breach onsets that are relapses (<= 8 weeks after a recovery), per indicator, with station bootstrap."""
import numpy as np, pandas as pd
from pathlib import Path
OUT = Path(__file__).resolve().parents[1] / "results"; SP = OUT / "spatial"
w = pd.read_parquet(OUT / "cnemc_weekly.parquet"); w = w[w.date.dt.year >= 2008]
rng = np.random.default_rng(242); MEM = 8
km_rows, share_rows = [], []
for band in (0.0, 0.10):
    recs = {k: [] for k in ("any", "NH4N", "DO", "CODMn")}; ons = {k: [] for k in recs}
    for s, g in w.groupby("site"):
        g = g.set_index("date").sort_index(); idx = pd.date_range(g.index.min(), g.index.max(), freq="W-MON"); g = g.reindex(idx)
        ob = g[["NH4N", "DO", "CODMn"]].notna().all(axis=1).values; N, D, C = g.NH4N.values, g.DO.values, g.CODMn.values
        defs = {"any": ((N > 1) | (D < 5) | (C > 6), (N <= 1 - band) & (D >= 5 * (1 + band)) & (C <= 6 * (1 - band))),
                "NH4N": (N > 1, N <= 1 - band), "DO": (D < 5, D >= 5 * (1 + band)), "CODMn": (C > 6, C <= 6 * (1 - band))}
        for nm, (brk, ok) in defs.items():
            st = np.full(len(idx), np.nan); cur = np.nan
            for i in range(len(idx)):
                if not ob[i]: cur = np.nan; continue
                if brk[i]: cur = 1.0
                elif ok[i] or np.isnan(cur): cur = 0.0
                st[i] = cur
            last = -999
            for i in range(1, len(st)):
                if not (np.isnan(st[i]) or np.isnan(st[i - 1])):
                    if st[i] == 1 and st[i - 1] == 0: ons[nm].append((s, int(1 <= (i - 1) - last <= MEM)))
                    if st[i] == 0 and st[i - 1] == 1:
                        k = 1; ev = 0
                        while i + k < len(st):
                            if np.isnan(st[i + k]): break
                            if st[i + k] == 1: ev = 1; break
                            k += 1
                        recs[nm].append((s, k if ev else k - 1, ev))
                if not np.isnan(st[i]) and st[i] == 1: last = i
    for nm in recs:
        R = pd.DataFrame(recs[nm], columns=["site", "T", "E"]); sites = R.site.unique()
        def km(Rx):
            S = 1.0; out = []
            for t in range(1, 27):
                at = (Rx["T"] >= t).sum(); d = ((Rx["T"] == t) & (Rx.E == 1)).sum(); S *= (1 - d / at) if at else 1; out.append(1 - S)
            return np.array(out)
        est = km(R); grp = {s: R[R.site == s] for s in sites}
        bs = np.array([km(pd.concat([grp[s] for s in rng.choice(sites, len(sites))])) for _ in range(200)])
        for t in range(26):
            km_rows.append(dict(band=band, ind=nm, week=t + 1, F=est[t], lo=np.quantile(bs[:, t], 0.025), hi=np.quantile(bs[:, t], 0.975), n_rec=len(R)))
        O_ = pd.DataFrame(ons[nm], columns=["site", "relapse"]); tab = O_.groupby("site").relapse.agg(["sum", "count"]); us = tab.index.values
        b2 = [tab.loc[rng.choice(us, len(us))].pipe(lambda t: t["sum"].sum() / t["count"].sum()) for _ in range(500)]
        share_rows.append(dict(band=band, ind=nm, share=O_.relapse.mean(), lo=np.quantile(b2, 0.025), hi=np.quantile(b2, 0.975), n_onsets=len(O_)))
        print(band, nm, "4wk", round(est[3], 3), "8wk", round(est[7], 3), "share", round(O_.relapse.mean(), 3), flush=True)
pd.DataFrame(km_rows).to_csv(SP / "rebreach_km.csv", index=False); pd.DataFrame(share_rows).to_csv(SP / "relapse_share_by_indicator.csv", index=False)
