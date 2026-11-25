"""Station-level anthropogenic and climatic pressures and breach outcomes (CNEMC weekly stations, 2008-2018).

Pressures (no river routing; distances are great-circle distances from each station):
  HydroWASTE v1.0 (China): treated wastewater discharge (m3/d), population served and dilution factor (DF) of
    WWTP outfalls within 10, 25 and 50 km; minimum DF and discharge-weighted geometric-mean DF within 25 km.
    Chinese WWTP location accuracy was not assessed by HydroWASTE (QUAL_LOC = 4).
  GHS Urban Centre Database 2015: urban-centre population within 25 and 50 km.
  WorldClim 2.1 (10 arc-min, 1970-2000): BIO1 annual mean temperature, BIO5 maximum temperature of the warmest month,
    BIO6 minimum temperature of the coldest month, BIO12 annual precipitation, BIO15 precipitation seasonality,
    BIO17 precipitation of the driest quarter.
Outcomes from the weekly panel: breach-week share, share of chronic station-years (>10% breach weeks), indicator
breach shares overall and by season, onset rate, recurrence share and first-week re-breach.
Output: results/review/station_pressures.csv"""
import os
from pathlib import Path
import io, zipfile
import numpy as np, pandas as pd, tifffile

ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "results" / "review"; OUT.mkdir(parents=True, exist_ok=True)
GD = Path(os.environ.get("GLOBAL_DATA_DIR", str(ROOT / "data" / "global")))
w = pd.read_parquet(ROOT / "results" / "cnemc_weekly.parquet"); w = w[w.date.dt.year >= 2008]
IND = ["NH4N", "DO", "CODMn"]


def hav(lat1, lon1, lat2, lon2):
    p = np.pi / 180; a = np.sin((lat2 - lat1) * p / 2) ** 2 + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


# ---------- station outcomes ----------
rows = []
for s, g in w.groupby("site"):
    g = g.set_index("date").sort_index(); idx = pd.date_range(g.index.min(), g.index.max(), freq="W-MON"); g = g.reindex(idx)
    ob = g[IND].notna().all(axis=1).values
    bN, bD, bC = (g.NH4N > 1).values & ob, (g.DO < 5).values & ob, (g.CODMn > 6).values & ob
    br = bN | bD | bC; mon = idx.month.values; yr = idx.year.values
    win = np.isin(mon, [12, 1, 2]); summ = np.isin(mon, [6, 7, 8, 9])
    ys = pd.DataFrame({"y": yr, "ob": ob, "br": br}).groupby("y").agg(n=("ob", "sum"), b=("br", "sum"))
    ys = ys[ys.n >= 10]
    on = [i for i in range(1, len(idx)) if ob[i] and ob[i - 1] and br[i] and not br[i - 1]]
    rel = [i for i in on if br[max(0, i - 9):i - 1].any()]
    rec = [i for i in range(1, len(idx) - 1) if ob[i] and ob[i - 1] and br[i - 1] and not br[i] and ob[i + 1]]
    n = ob.sum()
    rows.append(dict(site=s, lon=g.lon.dropna().iloc[0], lat=g.lat.dropna().iloc[0], n_weeks=int(n),
                     breach_share=br.sum() / n if n else np.nan,
                     chronic_year_share=(ys.b / ys.n > 0.10).mean() if len(ys) else np.nan, n_years=len(ys),
                     nh4_share=bN.sum() / n, do_share=bD.sum() / n, cod_share=bC.sum() / n,
                     nh4_winter=bN[win].sum() / max(1, (ob & win).sum()), nh4_summer=bN[summ].sum() / max(1, (ob & summ).sum()),
                     do_winter=bD[win].sum() / max(1, (ob & win).sum()), do_summer=bD[summ].sum() / max(1, (ob & summ).sum()),
                     n_winter=int((ob & win).sum()), n_summer=int((ob & summ).sum()),
                     n_onsets=len(on), onset_rate=len(on) / max(1, (ob & ~br).sum()),
                     recurrence_share=len(rel) / len(on) if on else np.nan,
                     n_recoveries=len(rec), rebreach_w1=np.mean([br[i + 1] for i in rec]) if rec else np.nan))
S = pd.DataFrame(rows)

# ---------- HydroWASTE ----------
z = zipfile.ZipFile(GD / "hydrowaste" / "HydroWASTE_v10.zip")
H = pd.read_csv(io.BytesIO(z.read("HydroWASTE_v10.csv")), encoding="latin1", low_memory=False)
H = H[(H.CNTRY_ISO == "CHN")].copy()
H["lat"] = H.LAT_OUT.fillna(H.LAT_WWTP); H["lon"] = H.LON_OUT.fillna(H.LON_WWTP)
for r in (10, 25, 50):
    agg = {"n": [], "dis": [], "pop": [], "dfmin": [], "dfgm": []}
    for la, lo in zip(S.lat, S.lon):
        d = hav(la, lo, H.lat.values, H.lon.values); m = d <= r; h = H[m]
        agg["n"].append(int(m.sum())); agg["dis"].append(h.WASTE_DIS.sum()); agg["pop"].append(h.POP_SERVED.sum())
        df = h.DF.dropna(); wdis = h.WASTE_DIS[df.index]
        agg["dfmin"].append(df.min() if len(df) else np.nan)
        agg["dfgm"].append(np.exp(np.average(np.log(df), weights=wdis)) if len(df) else np.nan)
    S[f"wwtp_n_{r}"] = agg["n"]; S[f"effluent_m3d_{r}"] = agg["dis"]; S[f"pop_served_{r}"] = agg["pop"]
    S[f"df_min_{r}"] = agg["dfmin"]; S[f"df_gm_{r}"] = agg["dfgm"]
S["dist_wwtp_km"] = [hav(la, lo, H.lat.values, H.lon.values).min() for la, lo in zip(S.lat, S.lon)]

# ---------- urban centres ----------
z = zipfile.ZipFile(GD / "population" / "GHS_STAT_UCDB2015MT_GLOBE_R2019A_V1_2.zip")
U = pd.read_csv(io.BytesIO(z.read("GHS_STAT_UCDB2015MT_GLOBE_R2019A/GHS_STAT_UCDB2015MT_GLOBE_R2019A_V1_2.csv")), encoding="latin1", low_memory=False)
U = U[U.CTR_MN_ISO == "CHN"]
for r in (25, 50):
    S[f"urban_pop_{r}"] = [U.P15[hav(la, lo, U.GCPNT_LAT.values, U.GCPNT_LON.values) <= r].sum() for la, lo in zip(S.lat, S.lon)]

# ---------- WorldClim ----------
zc = zipfile.ZipFile(GD / "worldclim" / "wc2.1_10m_bio.zip")
for b in (1, 5, 6, 12, 15, 17):
    a = tifffile.imread(io.BytesIO(zc.read(f"wc2.1_10m_bio_{b}.tif"))).astype(float); a[a < -1e30] = np.nan
    vals = []
    for la, lo in zip(S.lat, S.lon):
        r0, c0 = int((90 - la) * 6), int((lo + 180) * 6); v = a[r0, c0]
        if not np.isfinite(v):  # coastal or water cell: nearest valid value in a 5 x 5 window
            win = a[max(0, r0 - 2):r0 + 3, max(0, c0 - 2):c0 + 3]; v = np.nanmean(win) if np.isfinite(win).any() else np.nan
        vals.append(v)
    S[f"bio{b}"] = vals
S.to_csv(OUT / "station_pressures.csv", index=False)
print(S.shape); print(S[["wwtp_n_25", "effluent_m3d_25", "df_min_25", "urban_pop_50", "bio1", "bio17"]].describe().round(2).to_string())
print("stations with >=1 WWTP within 10/25/50 km:", [(S[f"wwtp_n_{r}"] > 0).sum() for r in (10, 25, 50)])
