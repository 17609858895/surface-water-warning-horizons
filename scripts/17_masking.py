"""Masking experiment with known truth (Fig. S3).
Training/validation windows that are fully observed for 12 weeks get artificial monitoring gaps drawn from empirical
post-landmark observation patterns (MCAR: 2008-2014 or 2017-2018 patterns) or an informative scenario (breach weeks
more likely to be missing). Each gap-handling rule is trained on the masked labels and evaluated against the complete
truth on fully observed 2015-2016 test windows.
Rules: oracle (no mask), complete-case (drop windows with any gap), right-censoring at the first gap, naive gap filling
(carry the last observed status), three-state pairs without EM, three-state with forward-backward EM."""
import sys, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from survlib import *
from msboost import GAMS

OUT = Path(__file__).resolve().parents[1] / "results"; MD = OUT / "masking"; MD.mkdir(exist_ok=True)
L = pd.read_parquet(OUT / "landmarks.parquet").reset_index(drop=True); O = np.load(OUT / "obs_windows.npy")
FC = feat_cols(L); state = L.state.values.astype(int)
X = np.column_stack([L[FC].values.astype(float), state]); ws, wc = L.woy_sin.values, L.woy_cos.values
full = ~np.isnan(O[:, 1:]).any(1)
d = L.date
SPL = {"tr": (d <= "2013-09-30").values & full, "va": ((d >= "2014-01-01") & (d <= "2014-09-30")).values & full,
       "te": ((d >= "2015-01-01") & (d <= "2016-12-31")).values & full}
print({k: int(v.sum()) for k, v in SPL.items()}, flush=True)
POOL = {"MCAR 2008–2014 pattern": ~np.isnan(O[((d >= "2008-01-01") & (d <= "2014-12-31")).values][:, 1:]),
        "MCAR 2017–2018 pattern": ~np.isnan(O[(d >= "2017-01-01").values][:, 1:])}


def te_from(Om, s0):
    n = len(s0); T = np.full(n, K); E = np.zeros(n, int); C = np.full(n, K)
    for i in range(n):
        for k in range(1, K + 1):
            o = Om[i, k]
            if np.isnan(o): T[i] = k - 1; C[i] = k - 1; break
            if o != s0[i]: T[i] = k; E[i] = 1; break
    return T, E, C


def frame(mask, T, E, C):
    return pd.DataFrame(dict(T=T, E=E, C=C, woy_sin=ws[mask], woy_cos=wc[mask]))


def apply_mask(Ow, rng, pool, informative=False):
    Om = Ow.copy(); obs = pool[rng.integers(0, len(pool), len(Ow))]
    Om[:, 1:][~obs] = np.nan
    if informative:   # breach weeks additionally missing with probability 0.3
        extra = (Ow[:, 1:] == 1) & (rng.random(Om[:, 1:].shape) < 0.3)
        Om[:, 1:][extra] = np.nan
    return Om


te_idx = {t: SPL["te"] & (state == st) for t, st in [("onset", 0), ("recovery", 1)]}
truth = {t: frame(m, *te_from(O[m], state[m])) for t, m in te_idx.items()}


def evaluate(trans, h, rule, scen, seed):
    Lt = truth[trans]; F = 1 - surv_from_h(h)
    T = Lt["T"].values; E = Lt["E"].values
    obs12 = ((E == 1) & (T <= 12)).mean(); r = eval_ipcw(F, Lt)
    return dict(scenario=scen, seed=seed, trans=trans, rule=rule, NLL=nll_rc(h, Lt), IBS=r["IBS"], AUC4=r["AUCw@4"],
                pred12=F[:, -1].mean(), obs12=obs12, calib12=F[:, -1].mean() / obs12)


def fp_models(scen, seed, Otr, Ova, rule):
    """first-passage DH-GBM per transition under a gap rule"""
    res = []
    for trans, st in [("onset", 0), ("recovery", 1)]:
        mtr, mva = SPL["tr"] & (state == st), SPL["va"] & (state == st)
        Ot, Ov = Otr[(state[SPL["tr"]] == st)], Ova[(state[SPL["va"]] == st)]
        s_t, s_v = state[mtr], state[mva]
        if rule == "Naive gap filling":
            def fill(Om, s0):
                F_ = Om.copy()
                for k in range(1, K + 1):
                    nanm = np.isnan(F_[:, k]); F_[nanm, k] = F_[nanm, k - 1]
                return F_
            Ot, Ov = fill(Ot, s_t), fill(Ov, s_v)
        Tt, Et, Ct = te_from(Ot, s_t); Tv, Ev, Cv = te_from(Ov, s_v)
        Xt, Xv = X[mtr], X[mva]; wt, wv = (ws[mtr], wc[mtr]), (ws[mva], wc[mva])
        Lt_, Lv_ = frame(mtr, Tt, Et, Ct), frame(mva, Tv, Ev, Cv)
        if rule == "Complete case":
            kt = ~np.isnan(Ot[:, 1:]).any(1); kv = ~np.isnan(Ov[:, 1:]).any(1)
            if kt.sum() < 200 or kv.sum() < 50:
                res.append(dict(scenario=scen, seed=seed, trans=trans, rule=rule, NLL=np.nan, IBS=np.nan, AUC4=np.nan,
                                pred12=np.nan, obs12=np.nan, calib12=np.nan, note="no complete windows"))
                continue
            Lt_, Lv_ = Lt_[kt].reset_index(drop=True), Lv_[kv].reset_index(drop=True)
            Xt, Xv = Xt[kt], Xv[kv]; wt, wv = (wt[0][kt], wt[1][kt]), (wv[0][kv], wv[1][kv])
        m = LGBHazard().fit(Lt_, Xt, wt, Lv_, Xv, wv)
        mte = te_idx[trans]
        res.append(evaluate(trans, m.hazard(X[mte], ws[mte], wc[mte]), rule, scen, seed))
    return res


prev = pd.read_csv(MD / "masking_results.csv") if (MD / "masking_results.csv").exists() else pd.DataFrame()
rows = prev.to_dict("records") if len(prev) else []
done = set(zip(prev.scenario, prev.seed)) if len(prev) else set()
# oracle (no masking): identical across scenarios
for trans, st in ([] if ("all", 0) in done else [("onset", 0), ("recovery", 1)]):
    mtr, mva = SPL["tr"] & (state == st), SPL["va"] & (state == st)
    Lt_, Lv_ = frame(mtr, *te_from(O[mtr], state[mtr])), frame(mva, *te_from(O[mva], state[mva]))
    m = LGBHazard().fit(Lt_, X[mtr], (ws[mtr], wc[mtr]), Lv_, X[mva], (ws[mva], wc[mva]))
    mte = te_idx[trans]; rows.append(evaluate(trans, m.hazard(X[mte], ws[mte], wc[mte]), "Oracle (no gaps)", "all", 0))
SCEN = [("MCAR 2008–2014 pattern", False), ("MCAR 2017–2018 pattern", False), ("Informative (breach weeks +30% missing)", True)]
for scen, inf in SCEN:
    pool = POOL["MCAR 2008–2014 pattern"] if scen.startswith("Informative") else POOL[scen]
    for seed in range(3):
        if (scen, seed) in done: continue
        rng = np.random.default_rng(100 + seed)
        Otr = apply_mask(O[SPL["tr"]], rng, pool, inf); Ova = apply_mask(O[SPL["va"]], rng, pool, inf)
        miss = np.isnan(Otr[:, 1:]).any(1).mean()
        for rule in ["Complete case", "Right-censor at gap", "Naive gap filling"]:
            rows += fp_models(scen, seed, Otr, Ova, rule)
        for rule, em in [("3-state pairs (no EM)", False), ("3-state EM", True)]:
            g = GAMS(use_em=em, n_em=4).fit(Otr, state[SPL["tr"]], X[SPL["tr"]], (ws[SPL["tr"]], wc[SPL["tr"]]),
                                            Ova, state[SPL["va"]], X[SPL["va"]], (ws[SPL["va"]], wc[SPL["va"]]), verbose=False)
            for trans, st in [("onset", 0), ("recovery", 1)]:
                mte = te_idx[trans]; a, b, c = g.hazards(X[mte], ws[mte], wc[mte])
                rows.append(evaluate(trans, a if st == 0 else b, rule, scen, seed))
        print(scen, seed, f"windows with >=1 gap: {miss:.2f}", flush=True)
        pd.DataFrame(rows).to_csv(MD / "masking_results.csv", index=False)
R = pd.DataFrame(rows); R.to_csv(MD / "masking_results.csv", index=False)
print(R.groupby(["scenario", "trans", "rule"])[["NLL", "IBS", "AUC4", "calib12"]].mean().round(4).to_string())
