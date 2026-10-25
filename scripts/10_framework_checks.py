"""Feasibility numbers for the panel-level figure plan (no model training)."""
import numpy as np, pandas as pd
from pathlib import Path
OUT = Path(__file__).resolve().parents[1] / "results"
L = pd.read_parquet(OUT / "landmarks.parquet")
w = pd.read_parquet(OUT / "cnemc_weekly.parquet")
K = 12
rep = {}

# 1. yearly incomplete 12-week windows and complete-case vs IPCW 12-week onset risk
L["year"] = L.date.dt.year
inc = L.assign(inc=L.C < K).groupby(["year", "trans"]).inc.mean().unstack()
print("incomplete 12-wk window share by year\n", inc.round(3).to_string())
inc.to_csv(OUT / "check_incomplete_by_year.csv")

def cc_vs_ipcw(d):
    T, E, C = d["T"].values.astype(int), d["E"].values, d["C"].values
    pos = (E == 1) & (T <= K); known = pos | (T >= K)
    G = np.array([(C >= k).mean() for k in range(K + 1)])
    ipcw = np.sum(pos / G[np.clip(T, 0, K)]) / len(T)
    return pd.Series(dict(cc=pos[known].mean(), ipcw=ipcw, n=len(T), n_known=known.sum()))
cc = L[L.trans == "onset"].groupby("year").apply(cc_vs_ipcw, include_groups=False)
print("\nonset 12-wk risk: complete-case vs IPCW by year\n", cc.round(3).to_string())
cc.to_csv(OUT / "check_cc_vs_ipcw_by_year.csv")

# 2. episodes per indicator (consecutive observed breach weeks; a gap ends the observed run)
ep_rows = []
for s, g in w.groupby("site"):
    g = g.set_index("date").sort_index(); idx = pd.date_range(g.index.min(), g.index.max(), freq="W-MON"); g = g.reindex(idx)
    ob = g[["NH4N", "DO", "CODMn"]].notna().all(axis=1)
    for nm, ex in [("any", (g.NH4N > 1) | (g.DO < 5) | (g.CODMn > 6)), ("NH4N", g.NH4N > 1), ("DO", g.DO < 5), ("CODMn", g.CODMn > 6)]:
        st = ex.astype(float).where(ob).values
        i = 0
        while i < len(st):
            if st[i] == 1 and (i == 0 or st[i - 1] == 0):
                j = i
                while j + 1 < len(st) and st[j + 1] == 1: j += 1
                ended = (j + 1 < len(st)) and st[j + 1] == 0
                ep_rows.append(dict(site=s, ind=nm, start=idx[i], dur=j - i + 1, observed_end=ended))
                i = j + 1
            else:
                i += 1
EP = pd.DataFrame(ep_rows)
summ = EP.groupby("ind").agg(n_episodes=("dur", "size"), sites=("site", "nunique"), median_dur=("dur", "median"),
                             mean_dur=("dur", "mean"), share_1wk=("dur", lambda d: (d == 1).mean()),
                             share_ended_observed=("observed_end", "mean"))
print("\nbreach episodes (started after an observed compliant week)\n", summ.round(3).to_string())
summ.to_csv(OUT / "check_episodes.csv")

# 3. R-state consistency: onset landmarks already inside the 8-week post-recovery window at landmark time
on = L[L.trans == "onset"]
print("\nonset landmarks with breach in previous 9 weeks (R at landmark):", (on.wk_since_ex <= 9).mean().round(3),
      "| n =", len(on))

# 4. station-level relapse support
pr = pd.read_parquet(OUT / "pairs.parquet")
pr = pr[(pr.ind == "any") & (pr.year >= 2008)]
st_tab = pr.groupby(["site", "src"]).y.agg(["sum", "count"]).unstack()
n_R = st_tab[("count", "R")].fillna(0)
print("\nstations with >=50 R-weeks:", int((n_R >= 50).sum()), "| >=20:", int((n_R >= 20).sum()), "| any:", int((n_R > 0).sum()),
      "| total stations:", len(st_tab))
# empirical-Bayes shrunk station relapse and onset rates (beta-binomial, method of moments prior)
def eb(sum_, cnt):
    p = (sum_ / cnt).replace([np.inf], np.nan)
    m = (sum_.sum() / cnt.sum()); v = np.nanvar(p[cnt >= 20])
    k = max(m * (1 - m) / max(v, 1e-6) - 1, 1)
    return (sum_.fillna(0) + k * m) / (cnt.fillna(0) + k)
rR = eb(st_tab[("sum", "R")], st_tab[("count", "R")]); rA = eb(st_tab[("sum", "A")], st_tab[("count", "A")])
ratio = (rR / rA)
print("station relapse ratio (EB) quantiles:", ratio.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).round(1).to_dict())
pd.DataFrame(dict(relapse=rR, onset=rA, ratio=ratio, n_R=n_R)).to_csv(OUT / "check_station_relapse_eb.csv")

# 5. stations with any breach in 2008-2014
w["year"] = w.date.dt.year
pre = w[(w.year >= 2008) & (w.year <= 2014)]
print("\nstations with >=1 any-breach week 2008-2014:", int(pre.groupby("site").ex_any.any().sum()), "/", pre.site.nunique())
