"""Robustness of the station pressure associations and of the confirmation-rule rates (response to editorial reading).

A  Exposure form: presence of any HydroWASTE plant within 25 km plus log10 discharge per tenfold among stations with a
   plant (centred; zero for stations without a plant), adjusted for urban population, BIO1 and BIO17.
B  Dilution: log10 minimum dilution factor additionally adjusted for log discharge and urban population; and the
   discharge-weighted geometric-mean dilution factor in place of the minimum.
C  Region: discharge per tenfold of (discharge + 1) within stations north and south of 33.5 N.
D  Period: main model with outcomes recomputed from 2014-2018 weeks only (HydroWASTE is a single recent snapshot).
E  Spatial autocorrelation: Moran's I of Pearson residuals from the main breach-share model (5 nearest neighbours,
   999 permutations).
F  Confirmation rules: Kaplan-Meier probability of a breach within four weeks after declaration, censoring at the first
   unobserved week, compared with the complete-case false rate.
Station-model intervals: spatial block bootstrap (2 x 2 degree cells, 1,000 resamples).
Outputs: results/review/pressure_robustness.csv, confirmation_km.csv"""
from pathlib import Path
import numpy as np, pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "results" / "review"
S = pd.read_csv(OUT / "station_pressures.csv")
rng = np.random.default_rng(2807); B = 1000
S["block"] = (np.floor(S.lat / 2)).astype(int).astype(str) + "_" + (np.floor(S.lon / 2)).astype(int).astype(str)
S["log_eff"] = np.log10(S.effluent_m3d_25 + 1); S["log_urban"] = np.log10(S.urban_pop_50 + 1)
S["present"] = (S.effluent_m3d_25 > 0).astype(float)
pos = S.present == 1; S["tenfold"] = 0.0
S.loc[pos, "tenfold"] = np.log10(S.loc[pos, "effluent_m3d_25"]) - np.log10(S.loc[pos, "effluent_m3d_25"]).mean()
S["log_df"] = np.log10(S.df_min_25); S["log_dfgm"] = np.log10(S.df_gm_25)
Z = lambda x: (x - x.mean()) / x.std()


def fit(df, y, n, std, raw):
    X = pd.concat([df[std].apply(Z), df[raw]], axis=1) if std else df[raw].copy()
    X = sm.add_constant(X)
    return sm.GLM(df[y], X, family=sm.families.Binomial(), var_weights=df[n]).fit(scale="X2")


def boot(df, y, n, std, raw, terms):
    est = fit(df, y, n, std, raw).params[terms]; bs = []
    for _ in range(B):
        d = pd.concat([df[df.block == b] for b in rng.choice(df.block.unique(), df.block.nunique())])
        try: bs.append(fit(d, y, n, std, raw).params[terms].values)
        except Exception: pass
    bs = np.array(bs)
    return [(t, np.exp(est[t]), np.exp(np.nanquantile(bs[:, k], .025)), np.exp(np.nanquantile(bs[:, k], .975))) for k, t in enumerate(terms)]


rows = []
def add(analysis, outcome, res, n):
    for t, o, lo, hi in res: rows.append(dict(analysis=analysis, outcome=outcome, term=t, n_stations=n, odds_ratio=o, lo=lo, hi=hi))


COV = ["log_urban", "bio1", "bio17"]
for y, n in {"breach_share": "n_weeks", "chronic_year_share": "n_years"}.items():
    d = S.dropna(subset=[y]); d = d[d[n] > 0]
    add("A presence + per tenfold", y, boot(d, y, n, COV, ["present", "tenfold"], ["present", "tenfold"]), len(d))
    dd = d.dropna(subset=["log_df"])
    add("B dilution + discharge + urban", y, boot(dd, y, n, ["log_df", "log_eff", "log_urban", "bio1", "bio17"], [], ["log_df"]), len(dd))
    dg = d.dropna(subset=["log_dfgm"])
    add("B geometric-mean dilution", y, boot(dg, y, n, ["log_dfgm", "bio1", "bio17"], [], ["log_dfgm"]), len(dg))
    for name, m in (("C north of 33.5N", d.lat >= 33.5), ("C south of 33.5N", d.lat < 33.5)):
        ds = d[m].copy(); ds["log_eff_raw"] = ds.log_eff
        add(name + " (per tenfold)", y, boot(ds, y, n, COV, ["log_eff_raw"], ["log_eff_raw"]), len(ds))

# D: outcomes from 2014-2018 only
w = pd.read_parquet(ROOT / "results" / "cnemc_weekly.parquet"); w = w[w.date.dt.year >= 2014]
ok = w[["NH4N", "DO", "CODMn"]].notna().all(axis=1)
w = w[ok].assign(br=lambda x: ((x.NH4N > 1) | (x.DO < 5) | (x.CODMn > 6)).astype(float))
late = w.groupby("site").agg(n_weeks_late=("br", "size"), breach_late=("br", "mean")).reset_index()
d = S.merge(late, on="site"); d = d[d.n_weeks_late >= 20]
add("D outcomes 2014-2018", "breach_share", boot(d, "breach_late", "n_weeks_late", ["log_eff"] + COV, [], ["log_eff"]), len(d))

# E: Moran's I of Pearson residuals
d = S[S.n_weeks > 0].copy(); m = fit(d, "breach_share", "n_weeks", ["log_eff"] + COV, [])
r = m.resid_pearson.values; r = r - r.mean()
la, lo = np.radians(d.lat.values), np.radians(d.lon.values)
D = 2 * 6371 * np.arcsin(np.sqrt(np.sin((la[:, None] - la) / 2) ** 2 + np.cos(la[:, None]) * np.cos(la) * np.sin((lo[:, None] - lo) / 2) ** 2))
np.fill_diagonal(D, np.inf); Wm = np.zeros_like(D)
for i in range(len(D)): Wm[i, np.argsort(D[i])[:5]] = 1
Wm = Wm / Wm.sum(1, keepdims=True)
moran = lambda v: len(v) / Wm.sum() * (v @ Wm @ v) / (v @ v)
I = moran(r); perm = np.array([moran(rng.permutation(r)) for _ in range(999)])
rows.append(dict(analysis="E Moran's I, main-model residuals (5 nearest neighbours)", outcome="breach_share", term="Moran's I",
                 n_stations=len(d), odds_ratio=I, lo=np.quantile(perm, .025), hi=np.quantile(perm, .975)))
pval = (1 + (perm >= I).sum()) / 1000
R = pd.DataFrame(rows); R.to_csv(OUT / "pressure_robustness.csv", index=False)
print(R.round(3).to_string()); print("Moran permutation p (one-sided)", round(pval, 3), "median nn distance km", round(np.median(np.sort(D, 1)[:, 0]), 1))

# F: Kaplan-Meier four-week breach after declaration
src = (Path(__file__).parent / "28b_confirmation_rules.py").read_text(encoding="utf-8")
ns = {"__file__": str(Path(__file__).parent / "28b_confirmation_rules.py")}
exec(src.split("\n\nE = episodes()")[0], ns)
E = ns["episodes"](); declare = ns["declare"]
km = []
for grp, G in {"All": E, "Chronic": E[E.chronic], "Non-chronic": E[~E.chronic]}.items():
    for rule in ["single", "consecutive-2", "consecutive-3", "margin-0.2"]:
        times, events = [], []
        for q, mg in zip(G.seq, G.margins):
            d0 = declare(q, mg, rule)
            if d0 in ("g", None): continue
            t, e = 4, 0
            for j in range(1, 5):
                k = d0 + j
                if k >= len(q) or q[k] == "g": t, e = j - 1, 0; break
                if q[k] == "b": t, e = j, 1; break
            times.append(t); events.append(e)
        times, events = np.array(times), np.array(events); surv = 1.0
        for k in range(1, 5):
            at_risk = (times >= k).sum(); ev = ((times == k) & (events == 1)).sum()
            if at_risk: surv *= 1 - ev / at_risk
        km.append(dict(group=grp, rule=rule, n_declared=len(times), km_breach_4wk=1 - surv,
                       complete_case=np.mean(events[(events == 1) | (times == 4)]) if len(times) else np.nan))
K = pd.DataFrame(km); K.to_csv(OUT / "confirmation_km.csv", index=False); print(K.round(3).to_string())
