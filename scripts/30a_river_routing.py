"""Revision analysis 30a: hydrological connection between HydroWASTE outfalls and CNEMC stations along HydroRIVERS.

Reviewer 1 (comment 1) and Reviewer 2 (comments 2 and 5): great-circle proximity does not establish that a plant
discharges upstream of a station. Here every station is snapped to the HydroRIVERS v1.0 network and every HydroWASTE
outfall is routed downstream along NEXT_DOWN from its HYRIV_ID reach.

Station snapping: among reaches within 3 km of the station, the reach with the largest upstream area (UPLAND_SKM) is
chosen (national stations sit on main rivers, lakes and reservoirs); if no reach lies within 3 km the nearest reach is
used. A nearest-reach snap is kept as a sensitivity variant.

Routing: from each outfall reach, reaches are followed downstream until the accumulated river distance exceeds 150 km
or the network ends. A plant is upstream-connected to a station when the station reach lies on this path; the river
distance is measured from the outfall to the station (both projected onto their reaches). Plants within 10 km of the
coast without a dilution factor (HydroWASTE assumes ocean discharge) are excluded from routed exposures.

Station exposures (river distance R = 10, 25, 50, 100 km):
  up_n_R, up_dis_R          number and treated discharge (m3/d) of upstream-connected plants
  eff_frac_R                upstream treated discharge / (natural discharge at the station reach + effluent)
  df_station_R              natural discharge at the station reach / upstream treated discharge
  near_unconn_dis_25        discharge of plants within 25 km great-circle distance that are not upstream-connected
                            within 100 river-km (downstream or in another catchment)
Basin assignment (Reviewer 1, comment 8): river system of the station reach (MAIN_RIV outlet and downstream path),
grouped into China's nine first-level river basins (+ south-west and north-west rivers).
Outputs: results/revision/station_routing.csv, routing_pairs.csv, station_basins.csv"""
import os
from pathlib import Path
import io, zipfile
import numpy as np, pandas as pd, geopandas as gpd
from shapely.geometry import Point

ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "results" / "revision"; OUT.mkdir(parents=True, exist_ok=True)
GD = Path(os.environ.get("GLOBAL_DATA_DIR", str(ROOT / "data" / "global")))
SHP = GD / "hydrorivers_as" / "HydroRIVERS_v10_as_shp" / "HydroRIVERS_v10_as.shp"
S = pd.read_csv(ROOT / "results" / "review" / "station_pressures.csv")
CRS_M = "EPSG:3857"  # metric only for snapping within a few km; distances along rivers use LENGTH_KM


def hav(lat1, lon1, lat2, lon2):
    p = np.pi / 180; a = np.sin((lat2 - lat1) * p / 2) ** 2 + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


# ---------- network (China bounding box plus downstream reaches of international rivers) ----------
cols = ["HYRIV_ID", "NEXT_DOWN", "MAIN_RIV", "LENGTH_KM", "DIST_DN_KM", "UPLAND_SKM", "ENDORHEIC", "DIS_AV_CMS", "ORD_STRA"]
R = gpd.read_file(SHP, bbox=(70, 5, 145, 56), columns=cols, engine="pyogrio")
print("reaches read", len(R), flush=True)
R = R.set_index("HYRIV_ID", drop=False)
nxt = R.NEXT_DOWN.to_dict(); length = R.LENGTH_KM.to_dict(); geom = R.geometry.to_dict()

# ---------- HydroWASTE (China) ----------
z = zipfile.ZipFile(GD / "hydrowaste" / "HydroWASTE_v10.zip")
H = pd.read_csv(io.BytesIO(z.read("HydroWASTE_v10.csv")), encoding="latin1", low_memory=False)
H = H[H.CNTRY_ISO == "CHN"].copy().reset_index(drop=True)
H["lat"] = H.LAT_OUT.fillna(H.LAT_WWTP); H["lon"] = H.LON_OUT.fillna(H.LON_WWTP)
H["coastal"] = H.DF.isna()
print("plants", len(H), "coastal (no DF)", int(H.coastal.sum()), "without HYRIV_ID", int(H.HYRIV_ID.isna().sum()), flush=True)

# ---------- snap stations ----------
Rm = R[["HYRIV_ID", "UPLAND_SKM", "geometry"]].to_crs(CRS_M)
sidx = Rm.sindex
st = gpd.GeoDataFrame(S[["site", "lon", "lat"]], geometry=gpd.points_from_xy(S.lon, S.lat), crs="EPSG:4326").to_crs(CRS_M)
snap = []
for site, p, la in zip(st.site, st.geometry, S.lat):
    k = np.cos(np.radians(la))  # Web-Mercator metres x cos(latitude) = ground metres
    cand = list(sidx.query(p.buffer(3000 / k)))
    if cand:
        sub = Rm.iloc[cand]; d = sub.distance(p); keep = (d <= 3000 / k).values
        cand = cand if not keep.any() else [c for c, kk in zip(cand, keep) if kk]
    if cand and keep.any():
        sub = Rm.iloc[cand]; d = sub.distance(p)
        big = sub.loc[sub.UPLAND_SKM.idxmax()]
        near = sub.iloc[int(np.argmin(d.values))]
        snap.append(dict(site=site, reach=int(big.HYRIV_ID), snap_m=float(big.geometry.distance(p)) * k, reach_nearest=int(near.HYRIV_ID),
                         nearest_m=float(d.min()) * k, within3km=True))
    else:
        dd = Rm.distance(p); j = int(np.argmin(dd.values)); near = Rm.iloc[j]
        snap.append(dict(site=site, reach=int(near.HYRIV_ID), snap_m=float(dd.iloc[j]) * k, reach_nearest=int(near.HYRIV_ID),
                         nearest_m=float(dd.iloc[j]) * k, within3km=False))
SN = pd.DataFrame(snap)
SN["upland_skm"] = SN.reach.map(R.UPLAND_SKM); SN["q_nat_cms"] = SN.reach.map(R.DIS_AV_CMS); SN["ord_stra"] = SN.reach.map(R.ORD_STRA)
print("stations snapped within 3 km:", int(SN.within3km.sum()), "median snap distance m", round(SN.snap_m.median()), flush=True)


def pos_along(reach, lon, lat):
    """Fractional position (0 upstream end, 1 downstream end) of a point projected on a reach (lines run downstream)."""
    g = geom[reach]
    return g.project(Point(lon, lat), normalized=True)


def path_from(reach, start_frac, maxkm=150.0):
    """Downstream path from a point on `reach`: list of (reach, river km from the point to the reach's upstream end)."""
    out = []; d = -start_frac * length[reach]; r = reach
    while r in nxt and d <= maxkm:
        out.append((r, d)); d += length[r]; n = nxt[r]
        if n == 0 or n not in nxt: break
        r = n
    return out


# ---------- route every outfall ----------
stations_on = {}
for v, s in (("main", SN.reach), ("nearest", SN.reach_nearest)):
    m = {}
    for site, rr in zip(SN.site, s): m.setdefault(int(rr), []).append(site)
    stations_on[v] = m
spos = {v: {site: pos_along(int(rr), lo, la) for site, rr, lo, la in zip(SN.site, SN[c], S.lon, S.lat)}
        for v, c in (("main", "reach"), ("nearest", "reach_nearest"))}
pairs = []
for i, h in H.iterrows():
    if not np.isfinite(h.HYRIV_ID) or int(h.HYRIV_ID) not in nxt: continue
    r0 = int(h.HYRIV_ID); f0 = pos_along(r0, h.lon, h.lat)
    for v in ("main", "nearest"):
        for r, dkm in path_from(r0, f0):
            for site in stations_on[v].get(r, []):
                d = dkm + spos[v][site] * length[r]
                if d < 0: continue  # same reach, outfall below the station
                pairs.append(dict(variant=v, plant=i, site=site, river_km=d, dis=h.WASTE_DIS, coastal=bool(h.coastal), df_out=h.DF))
P = pd.DataFrame(pairs); P.to_csv(OUT / "routing_pairs.csv", index=False)
print("upstream plant-station pairs (<=150 river-km):", P.groupby("variant").size().to_dict(), flush=True)

# ---------- station exposures ----------
X = SN[["site", "reach", "snap_m", "within3km", "upland_skm", "q_nat_cms", "ord_stra"]].copy()
for v in ("main", "nearest"):
    pv = P[(P.variant == v) & ~P.coastal]
    suf = "" if v == "main" else "_nearestsnap"
    for Rk in (10, 25, 50, 100):
        q = pv[pv.river_km <= Rk].groupby("site").agg(n=("plant", "nunique"), dis=("dis", "sum"))
        X[f"up_n_{Rk}{suf}"] = X.site.map(q.n).fillna(0).astype(int); X[f"up_dis_{Rk}{suf}"] = X.site.map(q.dis).fillna(0.0)
        if v == "main":
            qe = X[f"up_dis_{Rk}"] / 86400.0
            X[f"eff_frac_{Rk}"] = qe / (X.q_nat_cms + qe)
            X[f"df_station_{Rk}"] = np.where(qe > 0, X.q_nat_cms / qe.replace(0, np.nan), np.nan)
            X[f"min_dist_up_{Rk}"] = X.site.map(pv[pv.river_km <= Rk].groupby("site").river_km.min())
# plants within 25 km great-circle distance: connected (upstream within 100 river-km) or not
pc = P[(P.variant == "main") & (P.river_km <= 100) & ~P.coastal]
conn = set(zip(pc.plant, pc.site)); coastal = H.coastal.values
rows = []
for site, la, lo in zip(S.site, S.lat, S.lon):
    d = hav(la, lo, H.lat.values, H.lon.values); idx = np.where(d <= 25)[0]
    c = [i for i in idx if (i, site) in conn]; u = [i for i in idx if (i, site) not in conn and not coastal[i]]
    rows.append(dict(site=site, near_n_25=len(idx), near_conn_n_25=len(c), near_unconn_n_25=len(u),
                     near_conn_dis_25=H.WASTE_DIS.values[c].sum(), near_unconn_dis_25=H.WASTE_DIS.values[u].sum(),
                     near_coastal_n_25=int(H.coastal.values[idx].sum())))
X = X.merge(pd.DataFrame(rows), on="site")

# ---------- river basins ----------
def outlet_xy(reach):
    """Downstream end of the network path from a reach and whether the system is endorheic."""
    r = reach; seen = 0
    while nxt.get(r, 0) not in (0, None) and nxt[r] in nxt and seen < 20000: r = nxt[r]; seen += 1
    x, y = list(geom[r].coords)[-1]
    return x, y, int(R.ENDORHEIC.get(r, 0)), r


def passes(reach, box, maxsteps=20000):
    r = reach
    for _ in range(maxsteps):
        x, y = list(geom[r].coords)[-1]
        if box[0] <= x <= box[1] and box[2] <= y <= box[3]: return True
        n = nxt.get(r, 0)
        if n in (0, None) or n not in nxt: return False
        r = n
    return False


HONGZE = (118.15, 118.95, 32.95, 33.65)  # Hongze Lake: the Huai main stem passes it before reaching the Yangtze
def basin(reach):
    x, y, endo, _ = outlet_xy(reach)
    if endo or (x < 105 and y > 35): return "North-west inland rivers"
    if 119.5 <= x <= 122.6 and 30.9 <= y <= 32.4:
        return "Huai River" if passes(reach, HONGZE) else "Yangtze River"
    if 118.3 <= x <= 119.6 and 37.2 <= y <= 38.25: return "Yellow River"
    if 119.0 <= x <= 121.5 and 32.4 <= y <= 35.6: return "Huai River"
    if 116.5 <= x <= 119.6 and 38.25 <= y <= 40.2: return "Hai River"
    if (120.8 <= x <= 122.8 and 39.5 <= y <= 41.2) or (x > 123 and y > 39.5): return "Songhua and Liao rivers"
    if 105.0 <= x <= 117.7 and 18.0 <= y <= 23.9: return "Pearl River and south coast"
    if 117.3 <= x <= 122.6 and 23.0 <= y <= 30.6: return "South-east rivers"
    if x < 110 and y < 23.6: return "South-west international rivers"
    return "Other"


X["basin"] = [basin(int(r)) for r in X.reach]
X = X.merge(S[["site", "lon", "lat"]], on="site")
X.to_csv(OUT / "station_routing.csv", index=False)
X[["site", "lon", "lat", "basin"]].to_csv(OUT / "station_basins.csv", index=False)
print(X.basin.value_counts().to_string(), flush=True)
print("stations with >=1 great-circle plant <=25 km:", int((X.near_n_25 > 0).sum()),
      "| of these with >=1 upstream-connected (<=100 river-km):", int(((X.near_n_25 > 0) & (X.near_conn_n_25 > 0)).sum()))
for Rk in (10, 25, 50, 100):
    print(f"stations with upstream plant within {Rk} river-km:", int((X[f'up_n_{Rk}'] > 0).sum()),
          "| nearest-snap:", int((X[f'up_n_{Rk}_nearestsnap'] > 0).sum()))
print("share of great-circle <=25 km plant-station pairs that are upstream-connected:",
      round(X.near_conn_n_25.sum() / max(1, X.near_n_25.sum() - X.near_coastal_n_25.sum()), 3))
print("share of great-circle <=25 km discharge that is upstream-connected:",
      round(X.near_conn_dis_25.sum() / X[["near_conn_dis_25", "near_unconn_dis_25"]].sum().sum(), 3))
