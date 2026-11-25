"""Associations between station pressures (28a) and breach outcomes.

H1  Persistent breach regimes are more frequent where treated-wastewater discharge and urban population near the
    station are larger.
H2  Winter ammonium breaches increase more with nearby effluent than ammonium breaches in other seasons
    (slower nitrification and lower dilution in the cold dry season).
H3  Summer dissolved-oxygen breaches increase with warm-season temperature and nearby effluent.
H4  First-week re-breach after recovery depends on the recovery margin; effluent pressure is examined after
    adjustment for margin and chronic history.
Station models: quasi-binomial GLMs with standardized predictors; 95% intervals from a spatial block bootstrap
(2 x 2 degree cells, 1,000 resamples). Week models: binomial GLMs with station-clustered standard errors; recovery model: GEE logistic regression
clustered by station (exchangeable working correlation). Outputs: results/review/pressure_*.csv"""
from pathlib import Path
import numpy as np, pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "results" / "review"
S = pd.read_csv(OUT / "station_pressures.csv")
rng = np.random.default_rng(2803); B = 1000
S["log_eff"] = np.log10(S.effluent_m3d_25 + 1); S["log_urban"] = np.log10(S.urban_pop_50 + 1)
S["low_dilution"] = (S.df_min_25 < 10).astype(float)
S["block"] = (np.floor(S.lat / 2)).astype(int).astype(str) + "_" + (np.floor(S.lon / 2)).astype(int).astype(str)
Z = lambda x: (x - x.mean()) / x.std()
PRED = ["log_eff", "log_urban", "bio1", "bio17"]

# ---------- Spearman correlations ----------
outs = ["breach_share", "chronic_year_share", "nh4_share", "nh4_winter", "do_share", "do_summer", "cod_share", "onset_rate", "recurrence_share", "rebreach_w1"]
press = ["log_eff", "log_urban", "df_min_25", "bio1", "bio5", "bio6", "bio12", "bio17"]
blocks = S.block.unique(); idx_by_block = {b: np.where(S.block == b)[0] for b in blocks}
boot_idx = [np.concatenate([idx_by_block[b] for b in rng.choice(blocks, len(blocks))]) for _ in range(B)]
rows = []
for o in outs:
    for p in press:
        ok = S[[o, p]].notna().all(axis=1).values
        rho = spearmanr(S.loc[ok, o], S.loc[ok, p]).correlation
        bs = []
        for ii in boot_idx:
            d = S.iloc[ii][[o, p]].dropna()
            if len(d) > 10: bs.append(spearmanr(d[o], d[p]).correlation)
        rows.append(dict(outcome=o, pressure=p, n=int(ok.sum()), rho=rho, lo=np.nanquantile(bs, .025), hi=np.nanquantile(bs, .975)))
C = pd.DataFrame(rows); C.to_csv(OUT / "pressure_spearman.csv", index=False)
print(C.pivot(index="outcome", columns="pressure", values="rho").round(2).to_string(), flush=True)


# ---------- station quasi-binomial GLMs ----------
def qglm(df, y, n, preds):
    X = sm.add_constant(df[preds].apply(Z)); m = sm.GLM(df[y], X, family=sm.families.Binomial(), var_weights=df[n]).fit(scale="X2")
    return m.params


models = {"breach_share": "n_weeks", "chronic_year_share": "n_years", "nh4_share": "n_weeks", "do_share": "n_weeks",
          "cod_share": "n_weeks", "nh4_winter": "n_winter", "do_summer": "n_summer"}
rows = []
for y, n in models.items():
    df = S[[y, n] + PRED + ["block"]].dropna(); df = df[df[n] > 0]
    est = qglm(df, y, n, PRED); bs = []
    for _ in range(B):
        pick = rng.choice(df.block.unique(), df.block.nunique())
        d = pd.concat([df[df.block == b] for b in pick])
        try: bs.append(qglm(d, y, n, PRED).values)
        except Exception: pass
    bs = np.array(bs)
    for k, p in enumerate(["const"] + PRED):
        rows.append(dict(outcome=y, term=p, n_stations=len(df), coef=est[p], odds_ratio_per_sd=np.exp(est[p]),
                         lo=np.exp(np.nanquantile(bs[:, k], .025)), hi=np.exp(np.nanquantile(bs[:, k], .975))))
G = pd.DataFrame(rows); G.to_csv(OUT / "pressure_station_glm.csv", index=False)
print(G[G.term != "const"][["outcome", "term", "odds_ratio_per_sd", "lo", "hi"]].round(2).to_string(), flush=True)

# ---------- week-level GEE (H2, H3) ----------
w = pd.read_parquet(ROOT / "results" / "cnemc_weekly.parquet"); w = w[w.date.dt.year >= 2008].dropna(subset=["NH4N", "DO", "CODMn"])
w = w.merge(S[["site", "log_eff", "log_urban", "bio1", "bio5", "bio6", "bio17"]], on="site")
w["winter"] = w.date.dt.month.isin([12, 1, 2]).astype(float); w["summer"] = w.date.dt.month.isin([6, 7, 8, 9]).astype(float)
for c in ["log_eff", "log_urban", "bio1", "bio5", "bio6", "bio17"]: w[c + "_z"] = Z(w[c])
w["nh4"] = (w.NH4N > 1).astype(float); w["do"] = (w.DO < 5).astype(float)
specs = {"NH4 breach (H2)": ("nh4", ["winter", "log_eff_z", "winter:log_eff_z", "log_urban_z", "bio6_z", "bio17_z"]),
         "DO breach (H3)": ("do", ["summer", "bio5_z", "summer:bio5_z", "log_eff_z", "summer:log_eff_z", "log_urban_z", "bio17_z"])}
rows = []
for name, (y, terms) in specs.items():
    X = pd.DataFrame({"const": 1.0}, index=w.index)
    for t in terms:
        X[t] = w[t.split(":")[0]] * w[t.split(":")[1]] if ":" in t else w[t]
    # Binomial GLM with station-clustered standard errors (the exchangeable GEE did not converge for DO)
    m = sm.GLM(w[y], X, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(w.site)[0]})
    ci = m.conf_int()
    for t in terms:
        rows.append(dict(model=name, term=t, odds_ratio=np.exp(m.params[t]), lo=np.exp(ci.loc[t, 0]), hi=np.exp(ci.loc[t, 1]), p=m.pvalues[t]))
    print(name, "n weeks", len(w), "stations", w.site.nunique(), flush=True)
W = pd.DataFrame(rows); W.to_csv(OUT / "pressure_week_gee.csv", index=False); print(W.round(3).to_string(), flush=True)

# ---------- recovery-level GEE (H4) ----------
rec = []
for s, g in w.groupby("site"):
    g = g.set_index("date").sort_index(); idx = pd.date_range(g.index.min(), g.index.max(), freq="W-MON"); g = g.reindex(idx)
    ob = g[["NH4N", "DO", "CODMn"]].notna().all(axis=1).values
    br = ob & ((g.NH4N > 1) | (g.DO < 5) | (g.CODMn > 6)).values
    m = np.minimum.reduce([(1 - g.NH4N.values) / 1, (g.DO.values - 5) / 5, (6 - g.CODMn.values) / 6])
    share52 = pd.Series(np.where(ob, br.astype(float), np.nan)).shift(1).rolling(52, min_periods=10).mean().values
    for t in range(1, len(idx) - 1):
        if ob[t] and ob[t - 1] and br[t - 1] and not br[t] and ob[t + 1]:
            rec.append(dict(site=s, y=float(br[t + 1]), margin=m[t], chronic=float(share52[t] > 0.10) if np.isfinite(share52[t]) else 0.0))
R = pd.DataFrame(rec).merge(S[["site", "log_eff", "log_urban", "bio1"]], on="site")
R["margin_c"] = R.margin.clip(upper=1.0)
for c in ["log_eff", "log_urban", "bio1"]: R[c + "_z"] = Z(R[c])
X = sm.add_constant(R[["margin_c", "chronic", "log_eff_z", "log_urban_z", "bio1_z"]])
m = sm.GEE(R.y, X, groups=R.site, family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable()).fit()
ci = m.conf_int()
H4 = pd.DataFrame({"term": X.columns[1:], "odds_ratio": np.exp(m.params[1:]).values, "lo": np.exp(ci.iloc[1:, 0]).values,
                   "hi": np.exp(ci.iloc[1:, 1]).values, "p": m.pvalues[1:].values})
H4["n_recoveries"] = len(R); H4.to_csv(OUT / "pressure_recovery_gee.csv", index=False); print(H4.round(3).to_string())
# observed first-week re-breach by margin band
R["band"] = pd.cut(R.margin, [-np.inf, 0.05, 0.10, 0.20, 0.40, np.inf], labels=["<0.05", "0.05-0.10", "0.10-0.20", "0.20-0.40", ">=0.40"])
print(R.groupby(["band", "chronic"], observed=True).y.agg(["mean", "size"]).round(3).to_string())
R.groupby(["band", "chronic"], observed=True).y.agg(["mean", "size"]).reset_index().to_csv(OUT / "pressure_rebreach_by_margin.csv", index=False)
