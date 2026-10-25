"""Attach the observed future status sequence o_0..o_K (with monitoring gaps as NaN) to every landmark."""
import pandas as pd, numpy as np
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "results"
K = 12
L = pd.read_parquet(OUT / "landmarks.parquet")
w = pd.read_parquet(OUT / "cnemc_weekly.parquet")
O = np.full((len(L), K + 1), np.nan)
for s, g in w.groupby("site"):
    g = g.set_index("date").sort_index()
    idx = pd.date_range(g.index.min(), g.index.max(), freq="W-MON")
    g = g.reindex(idx)
    obs = g[["NH4N", "DO", "CODMn"]].notna().all(axis=1)
    ex = ((g.NH4N > 1.0) | (g.DO < 5.0) | (g.CODMn > 6.0)).astype(float).where(obs)
    sv = np.concatenate([ex.values, np.full(K, np.nan)])
    pos = pd.Series(np.arange(len(idx)), index=idx)
    li = np.where(L.site.values == s)[0]
    p = pos.reindex(L.date.values[li]).values.astype(int)
    O[li] = np.stack([sv[p + k] for k in range(K + 1)], 1)
assert np.allclose(O[:, 0], L.state.values)
np.save(OUT / "obs_windows.npy", O)
gap = np.isnan(O[:, 1:]).any(1)
after = (~np.isnan(O[:, 1:])).sum(1) - L.C.values
print("landmarks with >=1 gap week in next 12:", gap.mean().round(3))
print("mean observed weeks AFTER the first gap (discarded by right-censoring):", after[gap].mean().round(2))
print("by period:", pd.Series(after[gap]).groupby((L.date.values[gap] >= np.datetime64("2015-01-01"))).mean().round(2).to_dict())
