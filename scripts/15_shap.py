"""Fig. 6: horizon-specific TreeSHAP of the 3S-Boost+FT hazard learner (log-odds scale).
Rows are evaluated on 2015-2016 test landmarks: onset (A start), recovery (B start), relapse (R_j start, j advanced by k-1, capped at 8)."""
import sys, numpy as np, pandas as pd, lightgbm as lgb
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from survlib import feat_cols, K
from tristate import start_state

OUT = Path(__file__).resolve().parents[1] / "results"; CORE = OUT / "core"; SD = OUT / "shap"; SD.mkdir(exist_ok=True)
rng = np.random.default_rng(15)
L = pd.read_parquet(OUT / "landmarks.parquet").reset_index(drop=True); FF = np.load(OUT / "forecast_features.npy")
FC = feat_cols(L); state = L.state.values.astype(int)
XH = np.column_stack([L[FC].values.astype(float), state, FF])
FFN = [f"{p}_{v}_k{k}" for v in ["NH4N", "DO", "CODMn"] for k in (1, 2, 4, 8, 12) for p in ("fc", "pexc")] + [f"pbreach_k{k}" for k in (1, 2, 4, 8, 12)]
NAMES = FC + ["state"] + FFN + ["k", "cal_sin", "cal_cos", "type", "j"]
bst = lgb.Booster(model_str=(CORE / "tristate_FT.txt").read_text(encoding="utf-8"))
assert bst.num_feature() == len(NAMES), (bst.num_feature(), len(NAMES))
s3 = start_state(state, L.wk_since_ex.values)
sel = ((L.date >= "2015-01-01") & (L.date <= "2016-12-31")).values
ws, wc = L.woy_sin.values, L.woy_cos.values


def design(ix, k, t, j):
    ang = np.arctan2(ws[ix], wc[ix]) + 2 * np.pi * k / 52.18
    return np.column_stack([XH[ix], np.full(len(ix), k), np.sin(ang), np.cos(ang), np.full(len(ix), t), j])


def group(f):
    if f in ("mNH4N", "mDO", "mCOD", "margin_min") or f.endswith("_0") or f in ("NH4N", "DO", "CODMn"): return "Proximity / current level"
    if f.startswith(("ex_rate", "wk_since")) or f.endswith(("_m52", "_s52")): return "Chronic history"
    if f.startswith(("fc_", "pexc_", "pbreach_")): return "Concentration forecasts"
    if f in ("woy_sin", "woy_cos", "cal_sin", "cal_cos"): return "Season"
    if f in ("lon", "lat"): return "Location"
    if f in ("k", "type", "j", "state", "obs_frac12"): return "Structure"
    return "Short-term dynamics"


TYPES = {"Onset": (0, s3 == 0), "Recovery": (1, s3 == 1), "Relapse": (2, s3 >= 2)}
imp = []; store = {}
for tname, (t, mask) in TYPES.items():
    ix_all = np.where(sel & mask)[0]; ix = rng.choice(ix_all, min(3000, len(ix_all)), replace=False)
    for k in (1, 4, 12):
        j = np.where(t == 2, np.minimum(s3[ix] - 1 + k - 1, 8), 0)
        D = design(ix, k, t, j)
        sv = bst.predict(D, pred_contrib=True)[:, :-1]
        store[(tname, k)] = (ix, D, sv)
        for f, v in zip(NAMES, np.abs(sv).mean(0)):
            imp.append(dict(transition=tname, k=k, feature=f, group=group(f), mean_abs=v))
IMP = pd.DataFrame(imp)
IMP["norm"] = IMP.mean_abs / IMP.groupby(["transition", "k"]).mean_abs.transform("max")
IMP.to_csv(SD / "shap_mean_abs.csv", index=False)
excl = {"type", "j", "k", "state"}
top = (IMP[~IMP.feature.isin(excl)].groupby("feature").norm.mean().sort_values(ascending=False).head(12).index.tolist())
pd.Series(top).to_csv(SD / "top12.csv", index=False)
print("top12:", top)
gshare = IMP[~IMP.feature.isin(excl)].groupby(["transition", "k", "group"]).mean_abs.sum().unstack()
(gshare.div(gshare.sum(1), axis=0)).round(3).to_csv(SD / "group_share.csv"); print(gshare.div(gshare.sum(1), axis=0).round(3))

# beeswarm data: onset k=4
ix, D, sv = store[("Onset", 4)]
col = IMP[(IMP.transition == "Onset") & (IMP.k == 4) & ~IMP.feature.isin(excl)].sort_values("mean_abs", ascending=False).feature.head(10).tolist()
bee = []
for f in col:
    c = NAMES.index(f); x = D[:, c]; r = pd.Series(x).rank(pct=True).values
    for a, b, cc in zip(sv[:, c], x, r):
        bee.append(dict(feature=f, shap=a, value=b, value_pct=cc))
pd.DataFrame(bee).to_csv(SD / "beeswarm_onset_k4.csv", index=False)

# dependence (onset A vs relapse R, k=1) for the leading chronic and forecast features
dep = []
for f in ("ex_rate52", "pbreach_k4", "margin_min"):
    c = NAMES.index(f)
    for tname in ("Onset", "Relapse"):
        ix, D, sv = store[(tname, 1)]
        for a, b in zip(sv[:, c], D[:, c]):
            dep.append(dict(feature=f, transition=tname, value=b, shap=a))
pd.DataFrame(dep).to_csv(SD / "dependence.csv", index=False)

# interaction ex_rate52 x season for relapse and onset (k=1)
import shap
expl = shap.TreeExplainer(bst)
cm = NAMES.index("ex_rate52"); cs = [NAMES.index(f) for f in ("woy_sin", "woy_cos", "cal_sin", "cal_cos")]
rows = []
for tname in ("Onset", "Relapse"):
    ix, D, sv = store[(tname, 1)]; sub = rng.choice(len(ix), min(500, len(ix)), replace=False)
    inter = expl.shap_interaction_values(D[sub])
    iv = inter[:, cm, cs].sum(1) * 2
    rows.append(pd.DataFrame(dict(transition=tname, ex_rate52=D[sub, cm], month=L.date.dt.month.values[ix[sub]], interaction=iv)))
pd.concat(rows).to_csv(SD / "interaction_chronic_season.csv", index=False)
print("interaction done")
