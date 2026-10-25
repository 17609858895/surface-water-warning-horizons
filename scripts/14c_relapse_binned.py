"""Binned relapse-window curves (Fig. 2b,c): weeks since recovery grouped as 1,2,3,4,5-6,7-8,9-12,13-18,19-26;
long-compliant reference = no breach in the previous >52 weeks. Station-bootstrap 95% CI (B=500)."""
import numpy as np, pandas as pd
from pathlib import Path
OUT = Path(__file__).resolve().parents[1] / "results"; RD = OUT / "relapse"
rng = np.random.default_rng(141)
w = pd.read_parquet(OUT / "cnemc_weekly.parquet")
BINS = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 6), (7, 8), (9, 12), (13, 18), (19, 26)]


def series(band):
    rows = []
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
            ss = pd.Series(st)
            last_rec = pd.Series(np.where((ss == 0) & (ss.shift(1) == 1), np.arange(len(st)), np.nan)).ffill().values
            last_brk = pd.Series(np.where(ss == 1, np.arange(len(st)), np.nan)).ffill().values
            rows.append(pd.DataFrame(dict(site=s, ind=nm, cur=st, nxt=ss.shift(-1).values,
                                          since_rec=np.arange(len(st)) - last_rec + 1, since_brk=np.arange(len(st)) - last_brk)))
    return pd.concat(rows, ignore_index=True)


out = []
for band in (0.0, 0.05, 0.10, 0.20):
    S = series(band)
    for ind in (["any", "NH4N", "DO", "CODMn"] if band == 0 else ["any"]):
        d = S[(S.ind == ind) & (S.cur == 0) & S.nxt.notna()].copy()
        d["bin"] = -1
        d.loc[d.since_brk.isna() | (d.since_brk > 52), "bin"] = 0
        for b, (lo, hi) in enumerate(BINS, start=1):
            d.loc[(d.since_brk <= 52) & d.since_rec.between(lo, hi), "bin"] = b
        d = d[d.bin >= 0]
        tab = d.groupby(["site", "bin"]).nxt.agg(["sum", "count"]).unstack(fill_value=0)
        sites = tab.index.values
        def est(t): return (t["sum"].sum(0) / t["count"].sum(0).replace(0, np.nan)).values
        p0 = est(tab); bs = np.array([est(tab.loc[rng.choice(sites, len(sites))]) for _ in range(500)])
        for b, (lo, hi) in enumerate(BINS, start=1):
            out.append(dict(band=band, ind=ind, bin=f"{lo}" if lo == hi else f"{lo}–{hi}", x=(lo + hi) / 2, p=p0[b],
                            lo=np.nanquantile(bs[:, b], 0.025), hi=np.nanquantile(bs[:, b], 0.975), p_long=p0[0],
                            long_lo=np.nanquantile(bs[:, 0], 0.025), long_hi=np.nanquantile(bs[:, 0], 0.975),
                            fold=p0[b] / p0[0], fold_lo=np.nanquantile(bs[:, b] / bs[:, 0], 0.025),
                            fold_hi=np.nanquantile(bs[:, b] / bs[:, 0], 0.975), n=int(tab["count"][b].sum())))
    print("band", band, "done", flush=True)
RB = pd.DataFrame(out); RB.to_csv(RD / "relapse_binned.csv", index=False)
print(RB[RB.bin.isin(["1", "4", "9–12", "19–26"])].round(3).to_string())
