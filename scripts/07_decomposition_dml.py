"""Science module.
(1) Station-level weekly transition probabilities (onset A->B, recovery B->R, relapse R->B) by era, and a Shapley
    decomposition of the change in long-run breach occupancy into channels (balanced station panel, station bootstrap).
(2) DML event study on one-step transitions: Y = transition indicator, D_y = year dummies (ref 2014), X = state history,
    season, location; cross-fitting grouped by station; station-clustered SE. Placebo-in-time on pre-2015 years.
"""
import sys, numpy as np, pandas as pd, lightgbm as lgb
from pathlib import Path
from sklearn.model_selection import GroupKFold
sys.path.insert(0, str(Path(__file__).parent))
OUT = Path(__file__).resolve().parents[1] / "results"
rng = np.random.default_rng(7)
MEM = 8  # weeks after a recovery during which a compliant week counts as "R" (relapse-prone)

w = pd.read_parquet(OUT / "cnemc_weekly.parquet")
P = []
for s, g in w.groupby("site"):
    g = g.set_index("date").sort_index(); idx = pd.date_range(g.index.min(), g.index.max(), freq="W-MON"); g = g.reindex(idx)
    ob = g[["NH4N", "DO", "CODMn"]].notna().all(axis=1)
    rec = {}
    for nm, ex in [("any", (g.NH4N > 1) | (g.DO < 5) | (g.CODMn > 6)), ("NH4N", g.NH4N > 1), ("DO", g.DO < 5), ("CODMn", g.CODMn > 6)]:
        st = ex.astype(float).where(ob)
        # A vs R: compliant week is "R" if a breach occurred within the previous 8 observed-or-not weeks
        recent = st.shift(1).rolling(MEM + 1, min_periods=1).max()
        d = pd.DataFrame(dict(site=s, ind=nm, date=idx, prev=st.shift(1).values, cur=st.values, recent=recent.values))
        P.append(d)
D = pd.concat(P).dropna(subset=["prev", "cur"])
D["year"] = D.date.dt.year
D["src"] = np.where(D.prev == 1, "B", np.where(D.recent == 1, "R", "A"))
D["y"] = np.where(D.src == "B", 1 - D.cur, D.cur)          # B: recovery event; A/R: (re)breach event
D.to_parquet(OUT / "pairs.parquet")

# ---------- (1) era decomposition on stations observed in both eras ----------
ERA = {"pre": (2008, 2014), "post": (2015, 2018)}   # Water Pollution Prevention and Control Action Plan issued April 2015
D["era"] = np.where(D.year <= 2014, "pre", "post")
D = D[D.year >= 2008]


def occupancy(a, b, c):
    """Exact long-run breach occupancy of the chain A -> B -> R1..R_MEM -> A (R_j -> B with prob c)."""
    n = 2 + MEM; Pm = np.zeros((n, n))           # 0=A, 1=B, 2..=R1..R_MEM
    Pm[0, 1] = a; Pm[0, 0] = 1 - a
    Pm[1, 2] = b; Pm[1, 1] = 1 - b
    for j in range(MEM):
        r = 2 + j
        Pm[r, 1] = c
        Pm[r, r + 1 if j < MEM - 1 else 0] = 1 - c
    ev, vec = np.linalg.eig(Pm.T)
    pi = np.real(vec[:, np.argmin(np.abs(ev - 1))]); pi = pi / pi.sum()
    return pi[1]


def rates(d):
    g = d.groupby("src").y.agg(["sum", "count"])
    get = lambda s: (g.loc[s, "sum"] + 0.5) / (g.loc[s, "count"] + 1) if s in g.index else np.nan
    return get("A"), get("B"), get("R")


def decompose(dd):
    pre = rates(dd[dd.era == "pre"]); post = rates(dd[dd.era == "post"])
    names = ["onset", "recovery", "relapse"]
    base = occupancy(*pre); full = occupancy(*post)
    # Shapley over 3 channels
    import itertools
    contrib = {n: 0.0 for n in names}
    perms = list(itertools.permutations(range(3)))
    for perm in perms:
        cur = list(pre)
        prev = occupancy(*cur)
        for j in perm:
            cur[j] = post[j]; nv = occupancy(*cur)
            contrib[names[j]] += (nv - prev) / len(perms); prev = nv
    return dict(pre_a=pre[0], pre_b=pre[1], pre_c=pre[2], post_a=post[0], post_b=post[1], post_c=post[2],
                occ_pre=base, occ_post=full, **{f"d_{n}": v for n, v in contrib.items()})


both = D.groupby(["ind", "site"]).era.nunique()
rows = []
for ind in ["any", "NH4N", "DO", "CODMn"]:
    dd = D[(D.ind == ind) & D.site.isin(both.loc[ind][both.loc[ind] == 2].index)]
    est = decompose(dd)
    sites = dd.site.unique(); boots = []
    grp = {s: g for s, g in dd.groupby("site")}
    for b in range(300):
        smp = rng.choice(sites, len(sites), replace=True)
        boots.append(decompose(pd.concat([grp[s] for s in smp])))
    B = pd.DataFrame(boots)
    for k, v in est.items():
        rows.append(dict(ind=ind, quantity=k, est=v, lo=B[k].quantile(0.025), hi=B[k].quantile(0.975), n_sites=len(sites)))
    print(ind, {k: round(v, 4) for k, v in est.items()}, flush=True)
pd.DataFrame(rows).to_csv(OUT / "era_decomposition.csv", index=False)

# empirical occupancy check (observed) by era
obs = D[D.ind == "any"].groupby("era").cur.mean()
print("observed breach occupancy by era (any):", obs.round(4).to_dict())

# ---------- (2) DML event study on one-step transitions ----------
L = pd.read_parquet(OUT / "landmarks.parquet")
FC = [c for c in L.columns if c not in {"site", "date", "state", "T", "E", "C", "trans", "year", "post", "inc"}]
Om = np.load(OUT / "obs_windows.npy")
L["next"] = Om[:, 1]
L = L[~np.isnan(L["next"]) & (L.date.dt.year >= 2008)].reset_index(drop=True)
L["yr"] = L.date.dt.year
X = L[FC].values.astype(float)
groups = L.site.values


def dml_event(mask, ref, years, label):
    """Partially linear DML with multiple binary treatments (year dummies), station-grouped cross-fitting."""
    d = L[mask].reset_index(drop=True); Xm = X[mask]
    Y = d["outcome"].values.astype(float)
    Dm = np.column_stack([(d.yr == y).astype(float) for y in years if y != ref])
    yrs = [y for y in years if y != ref]
    ry = np.zeros_like(Y); rD = np.zeros_like(Dm)
    gkf = GroupKFold(n_splits=5)
    prm = dict(objective="regression", learning_rate=0.05, num_leaves=31, min_data_in_leaf=100, feature_fraction=0.7,
               bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=24, seed=7)
    for fi, (a, b) in enumerate(gkf.split(Xm, Y, d.site.values)):
        m = lgb.train(dict(prm, objective="binary"), lgb.Dataset(Xm[a], Y[a]), 300)
        ry[b] = Y[b] - m.predict(Xm[b])
        for j in range(Dm.shape[1]):
            mj = lgb.train(dict(prm, objective="binary"), lgb.Dataset(Xm[a], Dm[a, j]), 200)
            rD[b, j] = Dm[b, j] - mj.predict(Xm[b])
    theta = np.linalg.lstsq(rD, ry, rcond=None)[0]
    eps = ry - rD @ theta
    # station-clustered sandwich
    Jinv = np.linalg.inv(rD.T @ rD)
    meat = np.zeros((len(theta), len(theta)))
    for s in np.unique(d.site.values):
        ix = d.site.values == s; u = rD[ix].T @ eps[ix]; meat += np.outer(u, u)
    V = Jinv @ meat @ Jinv
    se = np.sqrt(np.diag(V))
    out = pd.DataFrame(dict(transition=label, year=yrs, theta=theta, se=se, lo=theta - 1.96 * se, hi=theta + 1.96 * se,
                            base_rate=Y[(d.yr == ref).values].mean()))
    return out


ES = []
yrs = list(range(2008, 2019))
for label, st in [("onset", 0), ("recovery", 1)]:
    m = (L.state == st).values
    L["outcome"] = (L["next"] != L["state"]).astype(int)
    r = dml_event(m, 2014, yrs, label)
    ES.append(r); print(r.round(4).to_string(), flush=True)
ES = pd.concat(ES); ES.to_csv(OUT / "dml_event_study.csv", index=False)
