"""Revision analysis 30c (Reviewer 1, comment 5): confirmation rules evaluated over windows anchored at the recovery week.

In 28b each rule's false-declaration rate is measured over the H weeks after its own declaration week d, so rules that
declare later are evaluated over later calendar weeks. Here every rule is also evaluated over the same weeks relative
to the observed recovery week t, which begin after the latest possible declaration (d <= t + 3):
  common window  weeks t+4 ... t+3+H (H = 4, 8)
  (i)  breach rate among declared recoveries over the common window (window fully observed or breached);
  (ii) false declarations per resolved recovery with a fixed end: a declaration is false when a breach is observed in
       (d, t+3+H]; withheld recoveries count in the denominator, as in the main analysis.
A reference row gives the common-window breach rate for all recoveries irrespective of any rule.
Intervals: station bootstrap (500 resamples). Output: results/revision/confirmation_common_window.csv"""
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).parent; ROOT = HERE.parent; OUT = ROOT / "results" / "revision"; OUT.mkdir(parents=True, exist_ok=True)
src = (HERE / "28b_confirmation_rules.py").read_text(encoding="utf-8")
ns = {"__file__": str(HERE / "28b_confirmation_rules.py")}
exec(src.split("\n\nE = episodes()")[0], ns)
E = ns["episodes"](); declare = ns["declare"]
RULES = ["single", "consecutive-2", "consecutive-3", "margin-0.1", "margin-0.2", "combined"]
rng = np.random.default_rng(3003); B = 500


def window_status(seq, a, b):
    """'b' if a breach is observed in seq[a..b], 'c' if all observed compliant, else 'u' (gap before any breach)."""
    for j in range(a, b + 1):
        if j >= len(seq) or seq[j] == "g": return "u"
        if seq[j] == "b": return "b"
    return "c"


def summarize(G, rule, H):
    end = 3 + H; common, fixed_end = [], []
    for q, mg in zip(G.seq, G.margins):
        d = declare(q, mg, rule) if rule != "all recoveries" else 0
        if d == "g": continue
        if d is None:
            fixed_end.append(0); continue  # withheld: resolved, not false
        s = window_status(q, 4, end)
        if s != "u": common.append(s == "b")
        f = window_status(q, d + 1, end)
        if f != "u": fixed_end.append(int(f == "b"))
    return (np.mean(common) if common else np.nan, len(common), np.mean(fixed_end) if fixed_end else np.nan, len(fixed_end))


rows = []
by_site = {s: g for s, g in E.groupby("site")}; sites = np.array(sorted(by_site))
for H in (4, 8):
    for gname, G in {"All": E, "Chronic": E[E.chronic], "Non-chronic": E[~E.chronic]}.items():
        for rule in ["all recoveries"] + RULES:
            c, nc, f, nf = summarize(G, rule, H)
            row = dict(H=H, group=gname, rule=rule, common_window=f"t+4..t+{3 + H}", n_common=nc, breach_common=c,
                       n_resolved_fixed_end=nf, false_per_recovery_fixed_end=f)
            if gname == "All":
                bs = []
                for _ in range(B):
                    Eb = pd.concat([by_site[s] for s in rng.choice(sites, len(sites))], ignore_index=True)
                    cb, _, fb, _ = summarize(Eb, rule, H); bs.append((cb, fb))
                bs = np.array(bs, float)
                row.update(breach_common_lo=np.nanquantile(bs[:, 0], .025), breach_common_hi=np.nanquantile(bs[:, 0], .975),
                           fixed_end_lo=np.nanquantile(bs[:, 1], .025), fixed_end_hi=np.nanquantile(bs[:, 1], .975))
            rows.append(row)
    print("H", H, "done", flush=True)
R = pd.DataFrame(rows); R.to_csv(OUT / "confirmation_common_window.csv", index=False)
pd.set_option("display.width", 200)
print(R.round(3).to_string())
