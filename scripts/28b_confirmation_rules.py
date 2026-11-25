"""Decision analysis of recovery-confirmation rules on the CNEMC weekly panel (2008-2018).

A recovery starts at an observed breach-to-compliant transition (week t). A rule declares the station recovered at
week d >= t, or withholds the declaration if a breach is observed first. A declaration is a false recovery when a
breach is observed within H weeks after d; it is a sustained recovery when all H following weeks are observed and
compliant. Declarations whose follow-up is interrupted by a gap before any breach are treated as unresolved and
excluded from the rates. Rules:
  single        declare at t (current practice)
  consecutive-k declare after k consecutive observed compliant weeks (k = 2, 3)
  margin-x      declare at the first compliant week of the run whose minimum relative margin (Eq. S1) is >= x,
                or at the fourth consecutive compliant week if the margin is not reached
  combined      declare at t if margin >= 0.2, otherwise after two consecutive compliant weeks
Outputs: results/review/confirmation_rules.csv (overall and by chronic history) and confirmation_rules_by_indicator.csv.
Intervals: station bootstrap (500 resamples)."""
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "results" / "review"; OUT.mkdir(parents=True, exist_ok=True)
w = pd.read_parquet(ROOT / "results" / "cnemc_weekly.parquet"); w = w[w.date.dt.year >= 2008]
IND = ["NH4N", "DO", "CODMn"]; HS = (4, 8); B = 500
RULES = ["single", "consecutive-2", "consecutive-3", "margin-0.1", "margin-0.2", "combined"]
rng = np.random.default_rng(2801)


BG = []


def episodes():
    rec = []
    for s, g in w.groupby("site"):
        g = g.set_index("date").sort_index(); idx = pd.date_range(g.index.min(), g.index.max(), freq="W-MON"); g = g.reindex(idx)
        ob = g[IND].notna().all(axis=1).values
        bN, bD, bC = (g.NH4N > 1).values, (g.DO < 5).values, (g.CODMn > 6).values
        br = ob & (bN | bD | bC)
        m = np.minimum.reduce([(1 - g.NH4N.values) / 1, (g.DO.values - 5) / 5, (6 - g.CODMn.values) / 6])
        # 52-week breach share before t (chronic history), as in the landmark definition
        share52 = pd.Series(np.where(ob, br.astype(float), np.nan)).shift(1).rolling(52, min_periods=10).mean().values
        n = len(idx)
        # background reference: A-state compliant weeks with eight observed breach-free preceding weeks
        for t in range(8, n):
            if ob[t] and not br[t] and not br[t - 8:t].any() and ob[t - 8:t].all():
                seq = "".join("g" if not ob[j] else ("b" if br[j] else "c") for j in range(t, min(n, t + max(HS) + 1)))
                BG.append(dict(site=s, seq=seq, chronic=bool(share52[t] > 0.10) if np.isfinite(share52[t]) else False))
        for t in range(1, n):
            if not (ob[t] and ob[t - 1] and br[t - 1] and not br[t]):
                continue
            lim = [k for k, b in zip(IND, (bN[t - 1], bD[t - 1], bC[t - 1])) if b]
            seq = []  # status of weeks t, t+1, ...: 'c' compliant, 'b' breach, 'g' gap
            for j in range(t, min(n, t + 3 + max(HS) + 1)):
                seq.append("g" if not ob[j] else ("b" if br[j] else "c"))
            rec.append(dict(site=s, t=t, month=idx[t].month, margin=m[t], margins=[m[j] if j < n else np.nan for j in range(t, t + 4)],
                            seq="".join(seq), chronic=bool(share52[t] > 0.10) if np.isfinite(share52[t]) else False,
                            prev_ind="+".join(lim)))
    return pd.DataFrame(rec)


def declare(seq, margins, rule):
    """Return (offset of declaration from t, or None if withheld, or 'g' if unresolved by a gap before decision)."""
    if rule == "single":
        return 0
    if rule.startswith("consecutive"):
        k = int(rule.split("-")[1])
        for j in range(1, k):
            if j >= len(seq) or seq[j] == "g": return "g"
            if seq[j] == "b": return None
        return k - 1
    if rule.startswith("margin"):
        x = float(rule.split("-")[1])
        for j in range(0, 4):
            if j >= len(seq) or seq[j] == "g": return "g"
            if seq[j] == "b": return None
            if np.isfinite(margins[j]) and margins[j] >= x: return j
        return 3  # four consecutive compliant weeks without the margin: declare at the fourth
    if rule == "combined":
        return 0 if np.isfinite(margins[0]) and margins[0] >= 0.2 else declare(seq, margins, "consecutive-2")
    raise ValueError(rule)


def outcome(seq, d, H):
    """'false' if a breach is observed within H weeks after d, 'sustained' if H weeks observed compliant, else 'unresolved'."""
    for j in range(d + 1, d + H + 1):
        if j >= len(seq) or seq[j] == "g": return "unresolved"
        if seq[j] == "b": return "false"
    return "sustained"


def summarize(E, rule, H):
    dec = [declare(q, mg, rule) for q, mg in zip(E.seq, E.margins)]
    rows = []
    for q, d in zip(E.seq, dec):
        if d == "g": rows.append(("unresolved_decision", np.nan)); continue
        if d is None: rows.append(("withheld", np.nan)); continue
        rows.append((outcome(q, d, H), d))
    o = pd.DataFrame(rows, columns=["status", "delay"])
    resolved = o[o.status.isin(["false", "sustained", "withheld"])]
    decl = o[o.status.isin(["false", "sustained"])]
    return dict(rule=rule, H=H, n_recoveries=len(o), n_resolved=len(resolved), n_declared=len(decl),
                false_rate=(decl.status == "false").mean() if len(decl) else np.nan,
                false_per_recovery=(resolved.status == "false").mean() if len(resolved) else np.nan,
                withheld_share=(resolved.status == "withheld").mean() if len(resolved) else np.nan,
                mean_delay_sustained=decl.delay[decl.status == "sustained"].mean() if (decl.status == "sustained").any() else np.nan)


E = episodes(); print("recoveries", len(E), "stations", E.site.nunique(), flush=True)
groups = {"All": E, "Chronic": E[E.chronic], "Non-chronic": E[~E.chronic]}
by_site = {s: d for s, d in E.groupby("site")}; sites = np.array(sorted(by_site))
res = []
for H in HS:
    for gname, G in groups.items():
        for rule in RULES:
            r = summarize(G, rule, H); r["group"] = gname; res.append(r)
    # station bootstrap for the overall group
    boots = {rule: [] for rule in RULES}
    for b in range(B):
        pick = rng.choice(sites, len(sites), replace=True)
        Eb = pd.concat([by_site[s] for s in pick], ignore_index=True)
        for rule in RULES:
            r = summarize(Eb, rule, H); boots[rule].append((r["false_rate"], r["false_per_recovery"], r["mean_delay_sustained"]))
    for r in res:
        if r["H"] == H and r["group"] == "All":
            a = np.array(boots[r["rule"]], float)
            for k, name in enumerate(["false_rate", "false_per_recovery", "mean_delay_sustained"]):
                r[name + "_lo"], r[name + "_hi"] = np.nanquantile(a[:, k], [0.025, 0.975])
    print("H", H, "done", flush=True)
bg = pd.DataFrame(BG)
for H in HS:
    for gname, G in {"All": bg, "Chronic": bg[bg.chronic], "Non-chronic": bg[~bg.chronic]}.items():
        o = pd.Series([outcome(q, 0, H) for q in G.seq]); o = o[o != "unresolved"]
        res.append(dict(rule="A-week background", H=H, group=gname, n_recoveries=len(G), n_resolved=len(o), n_declared=len(o),
                        false_rate=(o == "false").mean()))
R = pd.DataFrame(res); R.to_csv(OUT / "confirmation_rules.csv", index=False)
print(R[R.group == "All"][["H", "rule", "n_declared", "false_rate", "false_rate_lo", "false_rate_hi", "false_per_recovery", "withheld_share", "mean_delay_sustained"]].round(3).to_string())
print(R[R.group != "All"][["H", "group", "rule", "n_declared", "false_rate", "mean_delay_sustained"]].round(3).to_string())
# by indicator that caused the preceding breach (single-indicator breaches only)
rows = []
for ind in IND:
    G = E[E.prev_ind == ind]
    for rule in ["single", "consecutive-2", "margin-0.2"]:
        r = summarize(G, rule, 8); r["indicator"] = ind; rows.append(r)
pd.DataFrame(rows).to_csv(OUT / "confirmation_rules_by_indicator.csv", index=False)
print(pd.DataFrame(rows)[["indicator", "rule", "n_declared", "false_rate", "mean_delay_sustained"]].round(3).to_string())
