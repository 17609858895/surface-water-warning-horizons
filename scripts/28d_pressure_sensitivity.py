"""Sensitivity of the station pressure associations (28c).

(i) Dilution: among stations with at least one HydroWASTE outfall within 25 km, quasi-binomial GLMs of breach
    outcomes on log10 minimum dilution factor, annual temperature (BIO1) and driest-quarter precipitation (BIO17).
(ii) Radius: the effluent term of the main station GLM re-estimated with discharge summed within 10, 25 and 50 km.
(iii) Leave-one-region-out: the 25-km effluent term with each 5 x 5 degree cell removed in turn.
Intervals: spatial block bootstrap (2 x 2 degree cells, 1,000 resamples). Output: results/review/pressure_sensitivity.csv"""
from pathlib import Path
import numpy as np, pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "results" / "review"
S = pd.read_csv(OUT / "station_pressures.csv")
rng = np.random.default_rng(2804); B = 1000
S["block"] = (np.floor(S.lat / 2)).astype(int).astype(str) + "_" + (np.floor(S.lon / 2)).astype(int).astype(str)
S["region"] = (np.floor(S.lat / 5)).astype(int).astype(str) + "_" + (np.floor(S.lon / 5)).astype(int).astype(str)
S["log_urban"] = np.log10(S.urban_pop_50 + 1); S["log_df"] = np.log10(S.df_min_25)
for r in (10, 25, 50): S[f"log_eff_{r}"] = np.log10(S[f"effluent_m3d_{r}"] + 1)
Z = lambda x: (x - x.mean()) / x.std()
OUTS = {"breach_share": "n_weeks", "chronic_year_share": "n_years", "nh4_share": "n_weeks", "do_share": "n_weeks", "cod_share": "n_weeks"}


def fit(df, y, n, preds):
    X = sm.add_constant(df[preds].apply(Z))
    return sm.GLM(df[y], X, family=sm.families.Binomial(), var_weights=df[n]).fit(scale="X2").params


def boot(df, y, n, preds, term):
    est = fit(df, y, n, preds)[term]; bs = []
    blocks = df.block.unique()
    for _ in range(B):
        d = pd.concat([df[df.block == b] for b in rng.choice(blocks, len(blocks))])
        try: bs.append(fit(d, y, n, preds)[term])
        except Exception: pass
    return np.exp(est), np.exp(np.nanquantile(bs, .025)), np.exp(np.nanquantile(bs, .975))


rows = []
D = S.dropna(subset=["log_df"])
for y, n in OUTS.items():
    d = D.dropna(subset=[y]); d = d[d[n] > 0]
    o, lo, hi = boot(d, y, n, ["log_df", "bio1", "bio17"], "log_df")
    rows.append(dict(analysis="dilution (stations with outfall <=25 km)", outcome=y, term="log_df", n_stations=len(d), odds_ratio_per_sd=o, lo=lo, hi=hi))
for r in (10, 25, 50):
    for y, n in OUTS.items():
        d = S.dropna(subset=[y]); d = d[d[n] > 0]
        o, lo, hi = boot(d, y, n, [f"log_eff_{r}", "log_urban", "bio1", "bio17"], f"log_eff_{r}")
        rows.append(dict(analysis=f"effluent radius {r} km", outcome=y, term=f"log_eff_{r}", n_stations=len(d), odds_ratio_per_sd=o, lo=lo, hi=hi))
d = S[S.n_weeks > 0]
loro = [np.exp(fit(d[d.region != g], "breach_share", "n_weeks", ["log_eff_25", "log_urban", "bio1", "bio17"])["log_eff_25"]) for g in d.region.unique()]
rows.append(dict(analysis="leave-one-region-out (5 x 5 degrees)", outcome="breach_share", term="log_eff_25", n_stations=len(d),
                 odds_ratio_per_sd=np.median(loro), lo=np.min(loro), hi=np.max(loro)))
R = pd.DataFrame(rows); R.to_csv(OUT / "pressure_sensitivity.csv", index=False); print(R.round(2).to_string())
