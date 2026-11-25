"""Revision analysis 30g (Reviewer 2, comment 5.1): minimum and discharge-weighted geometric-mean dilution factors under
the same adjustment sets, among stations with a plant and dilution estimate within 25 km (quasi-binomial; 2-degree
block bootstrap, 1,000 resamples). Output: results/revision/dilution_forms.csv"""
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "results" / "revision"
S = pd.read_csv(ROOT / "results" / "review" / "station_pressures.csv")
S["log_eff"] = np.log10(S.effluent_m3d_25 + 1); S["log_urban"] = np.log10(S.urban_pop_50 + 1)
S["log_dfmin"] = np.log10(S.df_min_25); S["log_dfgm"] = np.log10(S.df_gm_25)
S["block"] = (np.floor(S.lat / 2)).astype(int).astype(str) + "_" + (np.floor(S.lon / 2)).astype(int).astype(str)
rng = np.random.default_rng(3007)
def irls(X, y, w):
    b = np.zeros(X.shape[1])
    for _ in range(50):
        p = 1 / (1 + np.exp(-np.clip(X @ b, -30, 30))); W = w * p * (1 - p)
        st = np.linalg.solve(X.T @ (W[:, None] * X) + 1e-10 * np.eye(len(b)), X.T @ (w * (y - p))); b += st
        if np.abs(st).max() < 1e-8: break
    return b
def des(d, cols): return np.column_stack([np.ones(len(d))] + [(d[c] - d[c].mean()) / d[c].std() for c in cols])
rows = []
for y, n in (("breach_share", "n_weeks"), ("chronic_year_share", "n_years")):
    d = S.dropna(subset=["log_dfmin", "log_dfgm", y]); d = d[d[n] > 0].reset_index(drop=True)
    for dfv in ("log_dfmin", "log_dfgm"):
        for lab, cols in (("climate", [dfv, "bio1", "bio17"]), ("climate + discharge + urban", [dfv, "log_eff", "log_urban", "bio1", "bio17"])):
            est = irls(des(d, cols), d[y].values, d[n].values)[1]; ub = d.block.unique(); gi = {b: np.where(d.block == b)[0] for b in ub}; bs = []
            for _ in range(1000):
                dd = d.iloc[np.concatenate([gi[b] for b in rng.choice(ub, len(ub))])]
                bs.append(irls(des(dd, cols), dd[y].values, dd[n].values)[1])
            rows.append(dict(outcome=y, dilution=dfv, adjustment=lab, n=len(d), odds_ratio=np.exp(est), lo=np.exp(np.quantile(bs, .025)), hi=np.exp(np.quantile(bs, .975))))
R = pd.DataFrame(rows); R.to_csv(OUT / "dilution_forms.csv", index=False); print(R.round(2).to_string())
print("Spearman min vs gm DF:", round(S[["log_dfmin", "log_dfgm"]].corr("spearman").iloc[0, 1], 2))
