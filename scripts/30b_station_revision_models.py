"""Revision analysis 30b: station-level pressure models requested by the reviewers.

All models follow the main specification of 28c: quasi-binomial GLM of a station outcome (breach-week share weighted
by observed weeks; chronic station-year share weighted by station-years) on standardized predictors, with log urban
population within 50 km, BIO1 and BIO17 as covariates. Intervals: spatial block bootstrap of 2 x 2 degree cells
(1,000 resamples) unless stated.

A  Routed exposure (R1.1): treated discharge of upstream-connected plants within 25 and 50 river-km (30a); presence
   of such a plant plus discharge per tenfold; connected and unconnected plants within 25 km entered jointly; station
   dilution (natural discharge at the station reach / upstream treated discharge); nearest-reach snapping.
B  Period (R1.2): outcomes recomputed for 2008-2010, 2011-2013, 2014-2016 and 2017-2018.
C  Load (R1.3): indicator outcomes on the upstream effluent fraction (a concentration-equivalent load proxy).
D  Thresholds (R1.9): breach-week share recomputed under Class II and Class IV limits.
E  Completeness (R1.10): weeks with one or two indicators reported classified as breach when an available indicator
   exceeded its limit.
F  Spatial dependence (R2.4): Moran's I correlogram of main-model residuals; block bootstrap with 4, 6 and 8 degree
   cells and with HydroBASINS level-4 basins as clusters.
G  Dilution as modifier (R2.5.1): discharge, minimum dilution factor and their product term among stations with a
   plant within 25 km.
H  Dilution correlates (R2.5.2): Spearman correlations of the minimum dilution factor with discharge and plant counts.
I  North-south contrast of dilution factors (R2.5.3).
Outputs: results/revision/station_models.csv, correlogram.csv, dilution_correlates.csv, station_period_outcomes.csv"""
import os
from pathlib import Path
import numpy as np, pandas as pd, geopandas as gpd
import statsmodels.api as sm
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "results" / "revision"
GD = Path(os.environ.get("GLOBAL_DATA_DIR", str(ROOT / "data" / "global")))
S = pd.read_csv(ROOT / "results" / "review" / "station_pressures.csv")
X = pd.read_csv(OUT / "station_routing.csv").drop(columns=["lon", "lat"])
S = S.merge(X, on="site")
rng = np.random.default_rng(3002); B = 1000
Z = lambda x: (x - x.mean()) / x.std()
S["log_eff"] = np.log10(S.effluent_m3d_25 + 1); S["log_urban"] = np.log10(S.urban_pop_50 + 1)
for r in (25, 50): S[f"log_up_{r}"] = np.log10(S[f"up_dis_{r}"] + 1); S[f"log_up_{r}_ns"] = np.log10(S[f"up_dis_{r}_nearestsnap"] + 1)
S["log_conn"] = np.log10(S.near_conn_dis_25 + 1); S["log_unconn"] = np.log10(S.near_unconn_dis_25 + 1)
S["log_df"] = np.log10(S.df_min_25)
COV = ["log_urban", "bio1", "bio17"]


def blocks(df, deg):
    return (np.floor(df.lat / deg)).astype(int).astype(str) + "_" + (np.floor(df.lon / deg)).astype(int).astype(str)


S["block"] = blocks(S, 2)
# HydroBASINS level 4 membership
hb = gpd.read_file(GD / "hydrobasins_as_lev04" / "hybas_as_lev04_v1c.shp", engine="pyogrio")[["HYBAS_ID", "geometry"]]
pts = gpd.GeoDataFrame(S[["site"]], geometry=gpd.points_from_xy(S.lon, S.lat), crs="EPSG:4326")
j = gpd.sjoin(pts, hb.to_crs("EPSG:4326"), how="left", predicate="within").drop_duplicates("site")
S["hybas4"] = S.site.map(dict(zip(j.site, j.HYBAS_ID))).fillna(-1).astype(np.int64).astype(str)
print("HydroBASINS level-4 basins containing stations:", S.hybas4.nunique(), flush=True)


def _irls(Xm, y, wt, iters=50):
    """Weighted logistic regression with fractional outcomes (quasi-binomial point estimates)."""
    b = np.zeros(Xm.shape[1])
    for _ in range(iters):
        eta = np.clip(Xm @ b, -30, 30); p = 1 / (1 + np.exp(-eta)); W = wt * p * (1 - p)
        step = np.linalg.solve(Xm.T @ (W[:, None] * Xm) + 1e-10 * np.eye(len(b)), Xm.T @ (wt * (y - p)))
        b += step
        if np.max(np.abs(step)) < 1e-8: break
    return b


def _design(df, std, raw):
    cols = []
    for c in std:
        v = df[c].values.astype(float); cols.append((v - v.mean()) / v.std(ddof=1))
    for c in raw: cols.append(df[c].values.astype(float))
    return np.column_stack([np.ones(len(df))] + cols)


def fit(df, y, n, std, raw=()):
    """Point estimates of the main quasi-binomial GLM (statsmodels, for residuals)."""
    Xm = pd.concat([df[list(std)].apply(Z), df[list(raw)]], axis=1) if std else df[list(raw)].copy()
    Xm = sm.add_constant(Xm, has_constant="add")
    return sm.GLM(df[y], Xm, family=sm.families.Binomial(), var_weights=df[n]).fit(scale="X2")


def boot(df, y, n, std, terms, raw=(), cl="block"):
    names = ["const"] + list(std) + list(raw); pos = [names.index(t) for t in terms]
    df = df.reset_index(drop=True)
    est = _irls(_design(df, std, raw), df[y].values.astype(float), df[n].values.astype(float))[pos]
    ub = df[cl].unique(); gi = {b: np.where(df[cl].values == b)[0] for b in ub}; bs = []
    for _ in range(B):
        ii = np.concatenate([gi[b] for b in rng.choice(ub, len(ub))]); d = df.iloc[ii]
        try:
            bs.append(_irls(_design(d, std, raw), d[y].values.astype(float), d[n].values.astype(float))[pos])
        except Exception: pass
    bs = np.array(bs)
    return [(t, np.exp(est[k]), np.exp(np.nanquantile(bs[:, k], .025)), np.exp(np.nanquantile(bs[:, k], .975))) for k, t in enumerate(terms)]


rows = []
def add(analysis, outcome, res, n, note=""):
    for t, o, lo, hi in res:
        rows.append(dict(analysis=analysis, outcome=outcome, term=t, n_stations=n, odds_ratio=o, lo=lo, hi=hi, note=note))
    print(analysis, outcome, [(t, round(o, 2), round(lo, 2), round(hi, 2)) for t, o, lo, hi in res], flush=True)


OUTS = {"breach_share": "n_weeks", "chronic_year_share": "n_years"}
# ---------- reference: main model ----------
for y, n in OUTS.items():
    d = S.dropna(subset=[y]); d = d[d[n] > 0]
    add("Main: discharge within 25 km (great-circle)", y, boot(d, y, n, ["log_eff"] + COV, ["log_eff"]), len(d))

# ---------- A: routed exposures ----------
for y, n in OUTS.items():
    d = S.dropna(subset=[y]); d = d[d[n] > 0]
    for r in (25, 50):
        add(f"A1 upstream-connected discharge within {r} river-km", y, boot(d, y, n, [f"log_up_{r}"] + COV, [f"log_up_{r}"]), len(d))
        add(f"A5 nearest-reach snapping, {r} river-km", y, boot(d, y, n, [f"log_up_{r}_ns"] + COV, [f"log_up_{r}_ns"]), len(d))
    for r in (25, 50):
        dd = d.copy(); dd["present"] = (dd[f"up_dis_{r}"] > 0).astype(float); pos = dd.present == 1; dd["tenfold"] = 0.0
        lv = np.log10(dd.loc[pos, f"up_dis_{r}"]); dd.loc[pos, "tenfold"] = lv - lv.mean()
        add(f"A2 upstream plant within {r} river-km: presence and per tenfold", y,
            boot(dd, y, n, COV, ["present", "tenfold"], raw=["present", "tenfold"]), len(dd), note=f"{int(pos.sum())} exposed stations")
    add("A3 plants within 25 km: connected vs unconnected (joint model)", y, boot(d, y, n, ["log_conn", "log_unconn"] + COV, ["log_conn", "log_unconn"]), len(d))
    dd = d[d.up_dis_50 > 0].copy(); dd["log_dfst"] = np.log10(dd.df_station_50)
    add("A4 station dilution factor (upstream within 50 river-km)", y, boot(dd, y, n, ["log_dfst", "bio1", "bio17"], ["log_dfst"]), len(dd))
    add("A4 station dilution factor + discharge + urban", y, boot(dd, y, n, ["log_dfst", "log_up_50", "log_urban", "bio1", "bio17"], ["log_dfst"]), len(dd))

# medians of breach share by station dilution class
S["dfst_class"] = pd.cut(S.df_station_50, [0, 10, 100, np.inf], labels=["<10", "10-100", ">=100"]).astype(object)
S.loc[S.up_dis_50 == 0, "dfst_class"] = "no upstream plant"
dfcls = S.groupby("dfst_class").agg(n=("site", "size"), median_breach=("breach_share", "median"), mean_breach=("breach_share", "mean"),
                                     median_chronic=("chronic_year_share", "median")).reset_index()
print(dfcls.round(3).to_string(), flush=True)
dfcls.to_csv(OUT / "station_dilution_classes.csv", index=False)

# ---------- B: periods ----------
w = pd.read_parquet(ROOT / "results" / "cnemc_weekly.parquet"); w = w[w.date.dt.year >= 2008]
ok = w[["NH4N", "DO", "CODMn"]].notna().all(axis=1)
wc = w[ok].copy(); wc["br"] = ((wc.NH4N > 1) | (wc.DO < 5) | (wc.CODMn > 6)).astype(float)
PER = {"2008-2010": (2008, 2010), "2011-2013": (2011, 2013), "2014-2016": (2014, 2016), "2017-2018": (2017, 2018)}
per_rows = []
for pname, (a, b) in PER.items():
    q = wc[wc.date.dt.year.between(a, b)].groupby("site").agg(n_p=("br", "size"), y_p=("br", "mean")).reset_index()
    d = S.merge(q, on="site"); d = d[d.n_p >= 20]
    per_rows.append(q.assign(period=pname))
    add(f"B period {pname}: discharge within 25 km", "breach_share", boot(d, "y_p", "n_p", ["log_eff"] + COV, ["log_eff"]), len(d))
    add(f"B period {pname}: upstream discharge within 50 river-km", "breach_share", boot(d, "y_p", "n_p", ["log_up_50"] + COV, ["log_up_50"]), len(d))
pd.concat(per_rows).to_csv(OUT / "station_period_outcomes.csv", index=False)

# ---------- C: indicator outcomes on effluent fraction ----------
S["log_efffrac"] = np.log10(S.eff_frac_50 + 1e-4)
for y in ["nh4_share", "do_share", "cod_share"]:
    d = S[S.n_weeks > 0]
    add("C upstream effluent fraction (50 river-km)", y, boot(d, y, "n_weeks", ["log_efffrac"] + COV, ["log_efffrac"]), len(d))
    add("C population served within 25 km", y, boot(d.assign(log_ps=np.log10(d.pop_served_25 + 1)), y, "n_weeks", ["log_ps"] + COV, ["log_ps"]), len(d))
for y in ["breach_share"]:
    d = S[S.n_weeks > 0]
    add("C upstream effluent fraction (50 river-km)", y, boot(d, y, "n_weeks", ["log_efffrac"] + COV, ["log_efffrac"]), len(d))


# ---------- D/E: threshold classes and partial-week rule ----------
def station_share(rule):
    out = []
    for s, g in w.groupby("site"):
        nh, do, co = g.NH4N, g.DO, g.CODMn
        if rule in ("II", "III", "IV"):
            tn, td, tc = {"II": (0.5, 6, 4), "III": (1, 5, 6), "IV": (1.5, 3, 10)}[rule]
            okk = g[["NH4N", "DO", "CODMn"]].notna().all(axis=1); br = ((nh > tn) | (do < td) | (co > tc)) & okk
            out.append(dict(site=s, n=int(okk.sum()), y=br.sum() / max(1, okk.sum())))
        else:  # partial-week rule: breach if any available indicator exceeds; compliant only when all three observed
            any_obs = g[["NH4N", "DO", "CODMn"]].notna().any(axis=1); okk = g[["NH4N", "DO", "CODMn"]].notna().all(axis=1)
            br = ((nh > 1) | (do < 5) | (co > 6)).fillna(False) & any_obs
            cls = okk | br
            out.append(dict(site=s, n=int(cls.sum()), y=br[cls].sum() / max(1, cls.sum())))
    return pd.DataFrame(out)


for rule, lab in (("II", "D Class II limits"), ("IV", "D Class IV limits"), ("partial", "E partial weeks with an exceedance counted as breach")):
    q = station_share(rule); d = S.merge(q, on="site"); d = d[d.n > 0]
    add(lab + ": discharge within 25 km", "breach_share", boot(d, "y", "n", ["log_eff"] + COV, ["log_eff"]), len(d))
    add(lab + ": upstream discharge within 50 river-km", "breach_share", boot(d, "y", "n", ["log_up_50"] + COV, ["log_up_50"]), len(d))

# ---------- F: spatial dependence ----------
d = S[S.n_weeks > 0].copy()
for deg in (4, 6, 8):
    d["blk"] = blocks(d, deg)
    add(f"F block bootstrap {deg} x {deg} degrees", "breach_share", boot(d, "breach_share", "n_weeks", ["log_eff"] + COV, ["log_eff"], cl="blk"), len(d),
        note=f"{d.blk.nunique()} blocks")
    add(f"F block bootstrap {deg} x {deg} degrees (routed)", "breach_share", boot(d, "breach_share", "n_weeks", ["log_up_50"] + COV, ["log_up_50"], cl="blk"), len(d))
add("F cluster bootstrap by HydroBASINS level-4 basin", "breach_share", boot(d, "breach_share", "n_weeks", ["log_eff"] + COV, ["log_eff"], cl="hybas4"), len(d),
    note=f"{d.hybas4.nunique()} basins")
add("F cluster bootstrap by HydroBASINS level-4 basin (routed)", "breach_share", boot(d, "breach_share", "n_weeks", ["log_up_50"] + COV, ["log_up_50"], cl="hybas4"), len(d))
m = fit(d, "breach_share", "n_weeks", ["log_eff"] + COV); r = m.resid_pearson.values; r = r - r.mean()
la, lo = np.radians(d.lat.values), np.radians(d.lon.values)
D = 2 * 6371 * np.arcsin(np.sqrt(np.sin((la[:, None] - la) / 2) ** 2 + np.cos(la[:, None]) * np.cos(la) * np.sin((lo[:, None] - lo) / 2) ** 2))
cg = []
for a, b in [(0, 100), (100, 200), (200, 300), (300, 500), (500, 800), (800, 1200)]:
    Wm = ((D > a) & (D <= b)).astype(float); np.fill_diagonal(Wm, 0)
    if Wm.sum() == 0: continue
    I = len(r) / Wm.sum() * (r @ Wm @ r) / (r @ r)
    perm = np.array([(lambda v: len(v) / Wm.sum() * (v @ Wm @ v) / (v @ v))(rng.permutation(r)) for _ in range(999)])
    cg.append(dict(band_km=f"{a}-{b}", pairs=int(Wm.sum() / 2), morans_I=I, p_one_sided=(1 + (perm >= I).sum()) / 1000, null_hi=np.quantile(perm, .975)))
CG = pd.DataFrame(cg); CG.to_csv(OUT / "correlogram.csv", index=False); print(CG.round(3).to_string(), flush=True)

# ---------- G: dilution as an independent predictor or modifier ----------
for y, n in OUTS.items():
    dd = S.dropna(subset=["log_df", y]); dd = dd[dd[n] > 0].copy()
    dd["eff_z"] = Z(dd.log_eff); dd["df_z"] = Z(dd.log_df); dd["effxdf"] = dd.eff_z * dd.df_z
    add("G discharge, dilution and product term (stations with a plant <=25 km)", y,
        boot(dd, y, n, COV, ["eff_z", "df_z", "effxdf"], raw=["eff_z", "df_z", "effxdf"]), len(dd))
    add("G discharge and dilution, no product term", y, boot(dd, y, n, COV, ["eff_z", "df_z"], raw=["eff_z", "df_z"]), len(dd))

# ---------- H: correlates of the minimum dilution factor ----------
dd = S.dropna(subset=["log_df"])
hc = []
for v, lab in [("log_eff", "log discharge within 25 km"), ("wwtp_n_25", "plants within 25 km"), ("log_urban", "log urban population within 50 km"),
               ("bio17", "driest-quarter precipitation"), ("lat", "latitude")]:
    rho = spearmanr(dd.log_df, dd[v]).correlation; hc.append(dict(variable=lab, n=len(dd), spearman_rho_with_log_min_DF=rho))
dd = dd.assign(df_class=pd.cut(dd.df_min_25, [0, 10, 100, np.inf], labels=["<10", "10-100", ">=100"]))
grp = dd.groupby("df_class", observed=True).agg(n=("site", "size"), median_discharge_m3d=("effluent_m3d_25", "median"), median_plants=("wwtp_n_25", "median"),
                                                median_breach=("breach_share", "median"), north_share=("lat", lambda x: (x >= 33.5).mean())).reset_index()
HC = pd.DataFrame(hc); HC.to_csv(OUT / "dilution_correlates.csv", index=False); grp.to_csv(OUT / "dilution_classes.csv", index=False)
print(HC.round(3).to_string()); print(grp.round(3).to_string(), flush=True)

# ---------- I: north-south dilution ----------
ns = S.dropna(subset=["df_min_25"]).assign(region=lambda x: np.where(x.lat >= 33.5, "north", "south")).groupby("region").agg(
    n=("site", "size"), median_df=("df_min_25", "median"), share_df_lt10=("df_min_25", lambda x: (x < 10).mean()),
    median_breach=("breach_share", "median")).reset_index()
ns.to_csv(OUT / "dilution_north_south.csv", index=False); print(ns.round(3).to_string())
basin = S.groupby("basin").agg(stations=("site", "size"), weeks=("n_weeks", "sum"), median_breach=("breach_share", "median"),
                               with_plant_25km=("wwtp_n_25", lambda x: int((x > 0).sum())), with_upstream_50rkm=("up_n_50", lambda x: int((x > 0).sum()))).reset_index()
basin.to_csv(OUT / "station_basin_summary.csv", index=False); print(basin.to_string())
pd.DataFrame(rows).to_csv(OUT / "station_models.csv", index=False)
