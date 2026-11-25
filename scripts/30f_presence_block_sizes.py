"""Revision analysis 30f (Reviewer 2, comment 4): intervals of the headline presence model under larger spatial blocks.

Model (28g, analysis A): station breach-week share (and chronic station-year share) on the presence of any HydroWASTE
plant within 25 km plus log10 discharge per tenfold among stations with a plant (centred), adjusted for standardized
log urban population, BIO1 and BIO17 (quasi-binomial). Intervals: block bootstrap with 2, 4, 6 and 8 degree cells and
with HydroBASINS level-4 basins as clusters (1,000 resamples each). Output: results/revision/presence_block_sizes.csv"""
import os
from pathlib import Path
import numpy as np, pandas as pd, geopandas as gpd

ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "results" / "revision"
GD = Path(os.environ.get("GLOBAL_DATA_DIR", str(ROOT / "data" / "global")))
S = pd.read_csv(ROOT / "results" / "review" / "station_pressures.csv")
rng = np.random.default_rng(3006); B = 1000
S["log_urban"] = np.log10(S.urban_pop_50 + 1); S["present"] = (S.effluent_m3d_25 > 0).astype(float)
pos = S.present == 1; S["tenfold"] = 0.0
S.loc[pos, "tenfold"] = np.log10(S.loc[pos, "effluent_m3d_25"]) - np.log10(S.loc[pos, "effluent_m3d_25"]).mean()
hb = gpd.read_file(GD / "hydrobasins_as_lev04" / "hybas_as_lev04_v1c.shp", engine="pyogrio")[["HYBAS_ID", "geometry"]]
pts = gpd.GeoDataFrame(S[["site"]], geometry=gpd.points_from_xy(S.lon, S.lat), crs="EPSG:4326")
j = gpd.sjoin(pts, hb.to_crs("EPSG:4326"), how="left", predicate="within").drop_duplicates("site")
S["hybas4"] = S.site.map(dict(zip(j.site, j.HYBAS_ID))).fillna(-1).astype(np.int64).astype(str)


def irls(X, y, w):
    b = np.zeros(X.shape[1])
    for _ in range(50):
        p = 1 / (1 + np.exp(-np.clip(X @ b, -30, 30))); W = w * p * (1 - p)
        step = np.linalg.solve(X.T @ (W[:, None] * X) + 1e-10 * np.eye(len(b)), X.T @ (w * (y - p))); b += step
        if np.abs(step).max() < 1e-8: break
    return b


def design(d):
    z = [(d[c] - d[c].mean()) / d[c].std() for c in ("log_urban", "bio1", "bio17")]
    return np.column_stack([np.ones(len(d)), d.present, d.tenfold] + z)


rows = []
for y, n in (("breach_share", "n_weeks"), ("chronic_year_share", "n_years")):
    d = S.dropna(subset=[y]); d = d[d[n] > 0].reset_index(drop=True)
    est = irls(design(d), d[y].values, d[n].values)[1:3]
    for lab, cl in [("2 x 2 degrees (main)", 2), ("4 x 4 degrees", 4), ("6 x 6 degrees", 6), ("8 x 8 degrees", 8), ("HydroBASINS level-4 basins", None)]:
        g = d.hybas4.values if cl is None else ((np.floor(d.lat / cl)).astype(int).astype(str) + "_" + (np.floor(d.lon / cl)).astype(int).astype(str)).values
        ub = np.unique(g); gi = {b: np.where(g == b)[0] for b in ub}; bs = []
        for _ in range(B):
            ii = np.concatenate([gi[b] for b in rng.choice(ub, len(ub))]); dd = d.iloc[ii]
            if dd.present.nunique() < 2: continue
            bs.append(irls(design(dd), dd[y].values, dd[n].values)[1:3])
        bs = np.array(bs)
        for k, t in enumerate(("present", "tenfold")):
            rows.append(dict(outcome=y, blocks=lab, n_clusters=len(ub), term=t, odds_ratio=np.exp(est[k]),
                             lo=np.exp(np.quantile(bs[:, k], .025)), hi=np.exp(np.quantile(bs[:, k], .975))))
R = pd.DataFrame(rows); R.to_csv(OUT / "presence_block_sizes.csv", index=False)
pd.set_option("display.width", 200); print(R.round(2).to_string())
