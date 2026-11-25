"""Monthly indicator composition of new onsets (same onset definition as C10 in 27c_calibration_review.py).

An onset is an observed compliant-to-breach transition between consecutive observed weeks from 2008 onward.
For each calendar month: number of onsets and the fraction involving each indicator (fractions can sum above one
because an onset can involve several indicators). Station-bootstrap 95% intervals for each fraction."""
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "results" / "review"; OUT.mkdir(parents=True, exist_ok=True)
w = pd.read_parquet(ROOT / "results" / "cnemc_weekly.parquet"); w = w[w.date.dt.year >= 2008]
IND = ["NH4N", "DO", "CODMn"]
rec = []
for s, gg in w.groupby("site"):
    gg = gg.set_index("date").sort_index(); idx = pd.date_range(gg.index.min(), gg.index.max(), freq="W-MON"); gg = gg.reindex(idx)
    ob = gg[IND].notna().all(axis=1).values
    bN, bD, bC = (gg.NH4N > 1).values, (gg.DO < 5).values, (gg.CODMn > 6).values; br = ob & (bN | bD | bC)
    for i in range(1, len(idx)):
        if ob[i] and ob[i - 1] and br[i] and not br[i - 1]:
            rec.append(dict(site=s, month=idx[i].month, NH4N=bN[i], DO=bD[i], CODMn=bC[i]))
O = pd.DataFrame(rec)
assert len(O) == 1687, len(O)


def table(df):
    g = df.groupby("month")
    t = pd.DataFrame({"n": g.size()}).reindex(range(1, 13), fill_value=0)
    for k in IND:
        t[k] = g[k].mean().reindex(range(1, 13))
    return t


obs = table(O)
rng = np.random.default_rng(2704); sites = O.site.unique(); B = 1000; boot = []
by_site = {s: d for s, d in O.groupby("site")}
for b in range(B):
    pick = rng.choice(sites, len(sites), replace=True)
    boot.append(table(pd.concat([by_site[s] for s in pick], ignore_index=True))[IND].values)
boot = np.array(boot)
rows = []
for mi, m in enumerate(range(1, 13)):
    for ki, k in enumerate(IND):
        v = boot[:, mi, ki]; v = v[~np.isnan(v)]
        rows.append(dict(month=m, indicator=k, n_onsets=int(obs.loc[m, "n"]), fraction=obs.loc[m, k],
                         lo=np.quantile(v, 0.025), hi=np.quantile(v, 0.975)))
res = pd.DataFrame(rows); res.to_csv(OUT / "onset_monthly.csv", index=False)
print(obs.round(3).to_string()); print("stations with onsets:", O.site.nunique())
