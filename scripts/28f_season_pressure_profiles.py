"""Monthly indicator breach probabilities by station pressure group (descriptive companion to the 28c week models).

NH4-N breach probability by month for stations with no HydroWASTE plant within 25 km and for the lower and upper
halves of positive 25-km treated discharge; DO breach probability by month for tertiles of the maximum temperature
of the warmest month (WorldClim BIO5). Probabilities are pooled observed station-week fractions; 95% intervals from
200 station bootstrap resamples within each group. Output: results/review/season_pressure_profiles.csv"""
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "results" / "review"
S = pd.read_csv(OUT / "station_pressures.csv")
rng = np.random.default_rng(2806); B = 200
pos = S.effluent_m3d_25 > 0; med = S.loc[pos, "effluent_m3d_25"].median()
S["eff_group"] = np.where(~pos, "No plant within 25 km", np.where(S.effluent_m3d_25 <= med, "Lower discharge", "Upper discharge"))
t1, t2 = S.bio5.quantile([1 / 3, 2 / 3])
S["bio5_group"] = np.where(S.bio5 < t1, "Cooler summers", np.where(S.bio5 <= t2, "Intermediate", "Warmer summers"))

w = pd.read_parquet(ROOT / "results" / "cnemc_weekly.parquet"); w = w[w.date.dt.year >= 2008].dropna(subset=["NH4N", "DO", "CODMn"])
w = w.merge(S[["site", "eff_group", "bio5_group"]], on="site"); w["month"] = w.date.dt.month
w["nh4"] = (w.NH4N > 1).astype(float); w["do"] = (w.DO < 5).astype(float)
rows = []
for y, gcol in (("nh4", "eff_group"), ("do", "bio5_group")):
    for g, d in w.groupby(gcol):
        cnt = d.groupby(["site", "month"])[y].agg(["sum", "size"]).unstack(fill_value=0)
        sites = cnt.index.values; num = cnt["sum"].values; den = cnt["size"].values
        est = num.sum(0) / den.sum(0)
        bs = []
        for _ in range(B):
            k = rng.integers(0, len(sites), len(sites)); bs.append(num[k].sum(0) / np.maximum(den[k].sum(0), 1))
        lo, hi = np.quantile(bs, [.025, .975], axis=0)
        for m in range(12):
            rows.append(dict(indicator=y, grouping=gcol, group=g, n_stations=len(sites), month=m + 1, probability=est[m],
                             lo=lo[m], hi=hi[m], station_weeks=int(den[:, m].sum())))
R = pd.DataFrame(rows); R.to_csv(OUT / "season_pressure_profiles.csv", index=False)
print("discharge median among stations with a plant:", round(med), "m3/d; BIO5 tertiles:", round(t1, 1), round(t2, 1))
print(R.pivot_table(index=["indicator", "group"], columns="month", values="probability").round(3).to_string())
