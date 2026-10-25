"""Threshold sensitivity of the relapse phenomenon (Fig. S5): GB 3838-2002 Class II, III and IV limits.
Class II: NH4-N > 0.5, DO < 6, CODMn > 4 mg/L; Class III: 1.0 / 5 / 6; Class IV: 1.5 / 3 / 10.
Outputs pooled weekly onset/recovery/relapse probabilities, breach occupancy and binned relapse folds (station bootstrap)."""
import numpy as np, pandas as pd
from pathlib import Path
OUT = Path(__file__).resolve().parents[1] / "results"; RD = OUT / "relapse"
rng = np.random.default_rng(19); MEM = 8
w = pd.read_parquet(OUT / "cnemc_weekly.parquet"); w = w[w.date.dt.year >= 2008]
CLS = {"Class II": (0.5, 6.0, 4.0), "Class III": (1.0, 5.0, 6.0), "Class IV": (1.5, 3.0, 10.0)}
BINS = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 6), (7, 8), (9, 12), (13, 18), (19, 26)]
trans_rows, curve_rows = [], []
for cname, (tn, td, tc) in CLS.items():
    rows = []
    for s, g in w.groupby("site"):
        g = g.set_index("date").sort_index(); idx = pd.date_range(g.index.min(), g.index.max(), freq="W-MON"); g = g.reindex(idx)
        ob = g[["NH4N", "DO", "CODMn"]].notna().all(axis=1)
        st = ((g.NH4N > tn) | (g.DO < td) | (g.CODMn > tc)).astype(float).where(ob)
        prev = st.shift(1); recent = st.shift(1).rolling(MEM + 1, min_periods=1).max()
        last_rec = pd.Series(np.where((st == 0) & (st.shift(1) == 1), np.arange(len(st)), np.nan), index=idx).ffill()
        last_brk = pd.Series(np.where(st == 1, np.arange(len(st)), np.nan), index=idx).ffill()
        rows.append(pd.DataFrame(dict(site=s, prev=prev.values, cur=st.values, recent=recent.values, nxt=st.shift(-1).values,
                                      since_rec=np.arange(len(st)) - last_rec.values + 1, since_brk=np.arange(len(st)) - last_brk.values)))
    D = pd.concat(rows, ignore_index=True)
    P = D.dropna(subset=["prev", "cur"]).copy()
    P["src"] = np.where(P.prev == 1, "B", np.where(P.recent == 1, "R", "A")); P["y"] = np.where(P.src == "B", 1 - P.cur, P.cur)
    sites = P.site.unique()
    for src, lab in [("A", "onset"), ("B", "recovery"), ("R", "relapse")]:
        d = P[P.src == src]; tab = d.groupby("site").y.agg(["sum", "count"])
        bs = [tab.reindex(rng.choice(sites, len(sites))).fillna(0).pipe(lambda t: t["sum"].sum() / max(t["count"].sum(), 1)) for _ in range(300)]
        trans_rows.append(dict(cls=cname, transition=lab, p=d.y.mean(), lo=np.quantile(bs, 0.025), hi=np.quantile(bs, 0.975), n=len(d)))
    trans_rows.append(dict(cls=cname, transition="occupancy", p=P.cur.mean(), lo=np.nan, hi=np.nan, n=len(P)))
    C = D[(D.cur == 0) & D.nxt.notna()].copy(); C["bin"] = -1
    C.loc[C.since_brk.isna() | (C.since_brk > 52), "bin"] = 0
    for b, (lo, hi) in enumerate(BINS, start=1):
        C.loc[(C.since_brk <= 52) & C.since_rec.between(lo, hi), "bin"] = b
    C = C[C.bin >= 0]; tab = C.groupby(["site", "bin"]).nxt.agg(["sum", "count"]).unstack(fill_value=0); us = tab.index.values
    est = lambda t: (t["sum"].sum(0) / t["count"].sum(0).replace(0, np.nan)).values
    p0 = est(tab); bs = np.array([est(tab.loc[rng.choice(us, len(us))]) for _ in range(400)])
    for b, (lo, hi) in enumerate(BINS, start=1):
        curve_rows.append(dict(cls=cname, bin=f"{lo}" if lo == hi else f"{lo}–{hi}", x=(lo + hi) / 2, p=p0[b], p_long=p0[0],
                               fold=p0[b] / p0[0], fold_lo=np.nanquantile(bs[:, b] / bs[:, 0], 0.025), fold_hi=np.nanquantile(bs[:, b] / bs[:, 0], 0.975)))
    print(cname, "done", flush=True)
TR = pd.DataFrame(trans_rows); TR.to_csv(RD / "threshold_transitions.csv", index=False)
CV = pd.DataFrame(curve_rows); CV.to_csv(RD / "threshold_relapse_curves.csv", index=False)
print(TR.round(4).to_string()); print(CV[CV.bin.isin(["1", "4", "9–12", "19–26"])].round(2).to_string())
