"""Revision analysis 30d (Reviewer 1, comment 10; Reviewer 2, minor comment 2): weeks excluded by the three-indicator rule.

Counts reported station-weeks with all, two, one or none of NH4-N, DO and CODMn; their distribution by year, river
basin (30a) and station; the missing indicator; and whether the indicators that were reported in partial weeks
exceeded their limits more or less often than in complete weeks (a check of selection on water-quality status).
Also reports the recurrence share and four-week re-breach after recovery when partial weeks with an exceedance are
counted as breach weeks (compliance still requires all three indicators).
Outputs: results/revision/completeness_summary.csv, completeness_by_year.csv, completeness_by_basin.csv"""
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "results" / "revision"
w = pd.read_parquet(ROOT / "results" / "cnemc_weekly.parquet")
bas = pd.read_csv(OUT / "station_basins.csv")[["site", "basin"]]
IND = ["NH4N", "DO", "CODMn"]; LIM = {"NH4N": lambda x: x > 1, "DO": lambda x: x < 5, "CODMn": lambda x: x > 6}
w["n_ind"] = w[IND].notna().sum(axis=1); w = w.merge(bas, on="site")
w["region"] = np.where(w.lat >= 33.5, "north", "south")
tot = len(w); part = w[w.n_ind < 3]
summ = [dict(item="reported station-weeks", value=tot), dict(item="complete (three indicators)", value=int((w.n_ind == 3).sum())),
        dict(item="partial (one or two indicators)", value=int(((w.n_ind > 0) & (w.n_ind < 3)).sum())),
        dict(item="no outcome indicator (pH only)", value=int((w.n_ind == 0).sum())),
        dict(item="excluded share of reported station-weeks", value=len(part) / tot),
        dict(item="stations with any excluded week", value=part.site.nunique()),
        dict(item="max excluded weeks at one station", value=int(part.groupby("site").size().max())),
        dict(item="excluded weeks at the 5 stations with most exclusions", value=int(part.groupby("site").size().nlargest(5).sum()))]
for k in IND: summ.append(dict(item=f"partial weeks missing {k}", value=int(part[k].isna().sum())))
# selection check: exceedance frequency of reported indicators in partial vs complete weeks (same stations and years)
comp = w[w.n_ind == 3]
for k in IND:
    pk = part[part[k].notna()]
    if len(pk) == 0: continue
    # complete weeks restricted to the station-years that contributed partial weeks
    keys = set(zip(pk.site, pk.year)); ck = comp[[(s, y) in keys for s, y in zip(comp.site, comp.year)]]
    summ += [dict(item=f"{k} exceedance, partial weeks (n={len(pk)})", value=LIM[k](pk[k]).mean()),
             dict(item=f"{k} exceedance, complete weeks of the same station-years (n={len(ck)})", value=LIM[k](ck[k]).mean()),
             dict(item=f"{k} exceedance, all complete weeks", value=LIM[k](comp[k]).mean())]
S = pd.DataFrame(summ); S.to_csv(OUT / "completeness_summary.csv", index=False)
by_year = w.groupby("year").agg(reported=("site", "size"), excluded=("n_ind", lambda x: int((x < 3).sum()))).assign(share=lambda d: d.excluded / d.reported).reset_index()
by_basin = w.groupby("basin").agg(reported=("site", "size"), excluded=("n_ind", lambda x: int((x < 3).sum()))).assign(share=lambda d: d.excluded / d.reported).reset_index()
by_region = w.groupby("region").agg(reported=("site", "size"), excluded=("n_ind", lambda x: int((x < 3).sum()))).assign(share=lambda d: d.excluded / d.reported).reset_index()
by_year.to_csv(OUT / "completeness_by_year.csv", index=False); by_basin.to_csv(OUT / "completeness_by_basin.csv", index=False)
by_region.to_csv(OUT / "completeness_by_region.csv", index=False)
pd.set_option("display.width", 200)
print(S.to_string()); print(by_year.round(4).to_string()); print(by_basin.round(4).to_string()); print(by_region.round(4).to_string())


# ---------- recurrence share and re-breach under the partial-week rule ----------
def metrics(partial_rule):
    on = rec_n = 0; rel = 0; rb4 = []
    for s, g in w[w.year >= 2008].groupby("site"):
        g = g.set_index("date").sort_index(); idx = pd.date_range(g.index.min(), g.index.max(), freq="W-MON"); g = g.reindex(idx)
        full = g[IND].notna().all(axis=1).values
        exc = ((g.NH4N > 1) | (g.DO < 5) | (g.CODMn > 6)).fillna(False).values
        if partial_rule:
            br = exc & g[IND].notna().any(axis=1).values; ob = full | br
        else:
            br = exc & full; ob = full
        for i in range(1, len(idx)):
            if ob[i] and ob[i - 1] and br[i] and not br[i - 1]:
                on += 1; rel += int(br[max(0, i - 9):i - 1].any())
            if ob[i] and ob[i - 1] and br[i - 1] and not br[i]:
                rec_n += 1; fut = []
                for j in range(i + 1, min(len(idx), i + 5)):
                    if not ob[j]: break
                    fut.append(br[j])
                if any(fut): rb4.append(1)
                elif len(fut) == 4: rb4.append(0)
    return dict(onsets=on, recurrence_share=rel / on, recoveries=rec_n, rebreach_4wk_complete_case=np.mean(rb4))


M = pd.DataFrame([dict(rule="three indicators required (main)", **metrics(False)),
                  dict(rule="partial weeks with an exceedance counted as breach", **metrics(True))])
M.to_csv(OUT / "completeness_event_metrics.csv", index=False); print(M.round(4).to_string())
