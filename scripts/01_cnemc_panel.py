"""Build CNEMC weekly station panel and Class-III exceedance states (GB 3838-2002)."""
import pandas as pd, numpy as np
from pathlib import Path
OUT = Path(__file__).resolve().parents[1] / "results"
df = pd.read_parquet(OUT / "china_full.parquet")
c = df[df.src == "CNEMC"].copy()
c["v"] = pd.to_numeric(c.Value, errors="coerce")
w = c.pivot_table(index=["MonitoringLocationIdentifier", "MonitoringDate"], columns="IndicatorsName", values="v", aggfunc="mean").reset_index()
coords = c.groupby("MonitoringLocationIdentifier")[["LongitudeMeasure_WGS84", "LatitudeMeasure_WGS84"]].first()
w = w.merge(coords, left_on="MonitoringLocationIdentifier", right_index=True)
w.columns.name = None
w = w.rename(columns={"MonitoringLocationIdentifier": "site", "MonitoringDate": "date", "LongitudeMeasure_WGS84": "lon", "LatitudeMeasure_WGS84": "lat"})
print(w.shape); print(w.describe().T.round(3))
# Class III limits
w["ex_NH4N"] = w.NH4N > 1.0
w["ex_DO"] = w.DO < 5.0
w["ex_CODMn"] = w.CODMn > 6.0
w["ex_any"] = w[["ex_NH4N", "ex_DO", "ex_CODMn"]].any(axis=1)
w["year"] = w.date.dt.year
print(w.groupby("year")[["ex_NH4N", "ex_DO", "ex_CODMn", "ex_any"]].mean().round(3).assign(n=w.groupby("year").size(), sites=w.groupby("year").site.nunique()))
# site activity span
span = w.groupby("site").date.agg(["min", "max", "count"])
print(span.describe())
print("sites active both <=2013 and >=2016:", ((span["min"] <= "2013-01-01") & (span["max"] >= "2016-06-01")).sum())
print("sites by last year:", span["max"].dt.year.value_counts().sort_index().to_dict())
print("sites by first year:", span["min"].dt.year.value_counts().sort_index().to_dict())
# gaps in weeks
w = w.sort_values(["site", "date"])
gap = w.groupby("site").date.diff().dt.days / 7
print("gap weeks distribution:", gap.value_counts().head(10).to_dict())
# zeros / suspicious
for v in ["NH4N", "DO", "CODMn", "pH"]:
    print(v, "zeros", (w[v] == 0).sum(), "neg", (w[v] < 0).sum(), "q999", w[v].quantile(0.999))
w.to_parquet(OUT / "cnemc_weekly.parquet")
