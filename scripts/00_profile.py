"""Profile china_full.csv (Lin et al. 2024 ESSD, Figshare v4 file 43613973)."""
import pandas as pd, numpy as np
from pathlib import Path
import os
ROOT = Path(__file__).resolve().parents[1]
(ROOT / "results").mkdir(parents=True, exist_ok=True)
DATA = Path(os.environ.get("WATER_QUALITY_DATA", str(ROOT / "data" / "china_full.csv")))
df = pd.read_csv(DATA, dtype={"Value": str})
df["MonitoringDate"] = pd.to_datetime(df["MonitoringDate"])
df["src"] = df["SourceProvider"].str.extract(r"\((\w+)\)")[0].fillna(df["SourceProvider"])
print(df["src"].value_counts())
for s, g in df.groupby("src"):
    print("\n=====", s, len(g), "sites", g.MonitoringLocationIdentifier.nunique(), g.MonitoringDate.min().date(), g.MonitoringDate.max().date())
    print(g.groupby("IndicatorsName").agg(n=("Value", "size"), sites=("MonitoringLocationIdentifier", "nunique"), dl=("Value", lambda v: (v.str.contains("<")).sum())).sort_values("n", ascending=False))
    per_site = g.groupby("MonitoringLocationIdentifier").MonitoringDate.nunique()
    print("dates per site: ", per_site.describe().round(1).to_dict())
    print("weekday dist:", g.MonitoringDate.dt.dayofweek.value_counts().sort_index().to_dict())
    print("year dist:", g.MonitoringDate.dt.year.value_counts().sort_index().to_dict())
# non-numeric values
bad = df[~df.Value.str.match(r"^-?\d+(\.\d+)?([eE][-+]?\d+)?$")]
print("\nnon-numeric values:", bad.Value.value_counts().head(10).to_dict())
df.to_parquet(Path(__file__).resolve().parents[1] / "results" / "china_full.parquet")
