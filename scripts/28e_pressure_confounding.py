"""Confounding checks for the 25-km effluent association with station breach share (28c main model).

The effluent term (log10 treated discharge within 25 km, per SD) is re-estimated under nested adjustment sets:
unadjusted; + urban population; main (+ BIO1, BIO17); main + GDP per capita (PPP, 2008-2015 mean at the station
cell, Kummu et al. gridded product, 5 arc-min); main + latitude and longitude; population served by the same plants
in place of discharge. Quasi-binomial GLMs; intervals from the spatial block bootstrap (2 x 2 degree cells,
1,000 resamples). Output: results/review/pressure_confounding.csv"""
import os
from pathlib import Path
import numpy as np, pandas as pd, h5py
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "results" / "review"
GD = Path(os.environ.get("GLOBAL_DATA_DIR", str(ROOT / "data" / "global")))
S = pd.read_csv(OUT / "station_pressures.csv")
rng = np.random.default_rng(2805); B = 1000

with h5py.File(GD / "gdp" / "GDP_per_capita_PPP_1990_2015_v2.nc", "r") as f:
    lat = f["latitude"][:]; lon = f["longitude"][:]; yrs = f["time"][:]
    k = np.where((yrs >= 2008) & (yrs <= 2015))[0]
    gdp = []
    for la, lo in zip(S.lat, S.lon):
        i = int(np.argmin(np.abs(lat - la))); j = int(np.argmin(np.abs(lon - lo)))
        win = f["GDP_per_capita_PPP"][k, max(0, i - 1):i + 2, max(0, j - 1):j + 2].astype(float)
        win[(win <= 0) | ~np.isfinite(win)] = np.nan
        gdp.append(np.nanmean(win))
S["gdp_pc"] = gdp
S["log_gdp"] = np.log10(S.gdp_pc)
S["log_eff"] = np.log10(S.effluent_m3d_25 + 1); S["log_pop_served"] = np.log10(S.pop_served_25 + 1)
S["log_urban"] = np.log10(S.urban_pop_50 + 1)
S["block"] = (np.floor(S.lat / 2)).astype(int).astype(str) + "_" + (np.floor(S.lon / 2)).astype(int).astype(str)
Z = lambda x: (x - x.mean()) / x.std()

SETS = {"unadjusted": ["log_eff"],
        "+ urban population": ["log_eff", "log_urban"],
        "main (+ BIO1, BIO17)": ["log_eff", "log_urban", "bio1", "bio17"],
        "main + GDP per capita": ["log_eff", "log_urban", "bio1", "bio17", "log_gdp"],
        "main + latitude, longitude": ["log_eff", "log_urban", "bio1", "bio17", "lat", "lon"],
        "population served (main set)": ["log_pop_served", "log_urban", "bio1", "bio17"]}


def fit(df, y, n, preds):
    X = sm.add_constant(df[preds].apply(Z))
    return sm.GLM(df[y], X, family=sm.families.Binomial(), var_weights=df[n]).fit(scale="X2").params


rows = []
for y, n in {"breach_share": "n_weeks", "chronic_year_share": "n_years"}.items():
    d0 = S.dropna(subset=[y, "log_gdp"]); d0 = d0[d0[n] > 0]
    for name, preds in SETS.items():
        term = preds[0]; est = fit(d0, y, n, preds)[term]; bs = []
        for _ in range(B):
            d = pd.concat([d0[d0.block == b] for b in rng.choice(d0.block.unique(), d0.block.nunique())])
            try: bs.append(fit(d, y, n, preds)[term])
            except Exception: pass
        rows.append(dict(outcome=y, adjustment=name, term=term, n_stations=len(d0), odds_ratio_per_sd=np.exp(est),
                         lo=np.exp(np.nanquantile(bs, .025)), hi=np.exp(np.nanquantile(bs, .975))))
R = pd.DataFrame(rows); R.to_csv(OUT / "pressure_confounding.csv", index=False)
print("GDP per capita (PPP): median", round(S.gdp_pc.median()), "range", round(S.gdp_pc.min()), round(S.gdp_pc.max()),
      "| Spearman with log effluent", round(S[["log_gdp", "log_eff"]].corr("spearman").iloc[0, 1], 2))
print(R.round(2).to_string())
S[["site", "gdp_pc"]].to_csv(OUT / "station_gdp.csv", index=False)
