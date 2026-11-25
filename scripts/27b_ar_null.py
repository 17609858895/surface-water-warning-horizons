"""Reviewer checks on the memoryless nulls (2): a continuous concentration null.

Each station's three indicators (log1p NH4-N, DO, log1p CODMn) are represented as station-year levels plus an annual
harmonic and a weekly AR(1) residual with station-specific persistence, station-year residual scale and cross-indicator
innovation correlation. Latent series are simulated on every calendar week, observed through the station's real mask
and classified with the Class III thresholds. Because concentrations persist from week to week, a recovery recorded
just inside a limit can re-cross it soon without any change in the underlying process; this null therefore asks how
much of the first-week re-breach excess is expected from autocorrelated concentrations near the thresholds."""
import numpy as np, pandas as pd
from pathlib import Path
import importlib.util

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("nv", HERE / "27a_null_variants.py")
src = (HERE / "27a_null_variants.py").read_text(encoding="utf-8")
src = src[:src.index("res = {\"observed\": run(\"observed\", 1)}")]
ns = {"__file__": str(HERE / "27a_null_variants.py")}; exec(compile(src, "27a_defs", "exec"), ns)
stats_mh, mh, MEM, KMAX = ns["stats_mh"], ns["mh"], ns["MEM"], ns["KMAX"]
ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "results" / "review"
w = pd.read_parquet(ROOT / "results" / "cnemc_weekly.parquet"); w = w[w.date.dt.year >= 2008]
rng = np.random.default_rng(2702); NSIM = 100


def fit_station(g):
    idx = pd.date_range(g.index.min(), g.index.max(), freq="W-MON"); g = g.reindex(idx)
    yr = idx.year.values; ang = 2 * np.pi * idx.dayofyear.values / 365.25
    Xv = np.column_stack([np.log1p(g.NH4N.values.clip(min=0)), g.DO.values, np.log1p(g.CODMn.values.clip(min=0))])
    ob = ~np.isnan(Xv).any(axis=1)
    years = np.unique(yr); D = np.column_stack([(yr == y).astype(float) for y in years] + [np.sin(ang), np.cos(ang)])
    mean = np.zeros_like(Xv); Z = np.full_like(Xv, np.nan); sd_y = np.ones((len(years), 3)); phi = np.zeros(3)
    for j in range(3):
        o = ~np.isnan(Xv[:, j])
        cols = [c for c in range(D.shape[1]) if D[o, c].any() or c >= len(years)]
        beta = np.linalg.lstsq(D[o][:, cols], Xv[o, j], rcond=None)[0]
        full = np.zeros(D.shape[1]); full[cols] = beta
        yfit = np.array([full[k] for k in range(len(years))])
        missing = ~np.isin(np.arange(len(years)), cols); yfit[missing] = np.nanmean(yfit[~missing])
        mean[:, j] = D[:, :len(years)] @ yfit + D[:, len(years):] @ full[len(years):]
        res = Xv[:, j] - mean[:, j]
        sall = np.nanstd(res) if np.sum(o) > 3 else 1.0
        for k, y in enumerate(years):
            m = (yr == y) & o
            sd_y[k, j] = np.nanstd(res[m]) if m.sum() >= 5 and np.nanstd(res[m]) > 0 else sall
        Z[:, j] = res / sd_y[np.searchsorted(years, yr), j]
        pr = ~np.isnan(Z[1:, j]) & ~np.isnan(Z[:-1, j])
        phi[j] = np.clip(np.corrcoef(Z[1:, j][pr], Z[:-1, j][pr])[0, 1], 0, 0.98) if pr.sum() > 10 else 0.5
    inn = Z[1:] - phi * Z[:-1]; ok = ~np.isnan(inn).any(axis=1)
    R = np.corrcoef(inn[ok].T) if ok.sum() >= 30 else np.eye(3)
    R = 0.8 * R + 0.2 * np.eye(3)
    x_obs = np.where(ob, ((g.NH4N > 1) | (g.DO < 5) | (g.CODMn > 6)).values, False).astype(np.int8)
    return dict(yr=yr, ob=ob, mean=mean, sd=sd_y[np.searchsorted(years, yr)], phi=phi, R=R, x=x_obs)


def simulate(f, nsim):
    T = len(f["yr"]); Lc = np.linalg.cholesky(f["R"]); phi = f["phi"]; sc = np.sqrt(1 - phi ** 2)
    Z = np.empty((nsim, T, 3)); Z[:, 0] = rng.standard_normal((nsim, 3)) @ Lc.T
    E = rng.standard_normal((nsim, T, 3)) @ Lc.T
    for t in range(1, T): Z[:, t] = phi * Z[:, t - 1] + sc * E[:, t]
    Xc = f["mean"][None] + f["sd"][None] * Z
    nh4, do, cod = np.expm1(Xc[..., 0]), Xc[..., 1], np.expm1(Xc[..., 2])
    return ((nh4 > 1) | (do < 5) | (cod > 6)).astype(np.int8)


ON = REL = ATR = EVT = 0; EX = []; EV = []; OBS = dict(ON=0, REL=0, ATR=0, EVT=0, EX=[], EV=[]); PHI = []
for s, g in w.groupby("site"):
    g = g.set_index("date").sort_index()
    f = fit_station(g); PHI.append(f["phi"])
    X = simulate(f, NSIM)
    on, rel, atr, evt, ex, ev = stats_mh(f["yr"], f["ob"], X)
    ON = ON + on; REL = REL + rel; ATR = ATR + atr; EVT = EVT + evt; EX.append(ex); EV.append(ev)
    on, rel, atr, evt, ex, ev = stats_mh(f["yr"], f["ob"], f["x"][None, :])
    OBS["ON"] += on; OBS["REL"] += rel; OBS["ATR"] = OBS["ATR"] + atr; OBS["EVT"] = OBS["EVT"] + evt; OBS["EX"].append(ex); OBS["EV"].append(ev)


def summarise(ON, REL, ATR, EVT, EX, EV):
    EX = np.concatenate(EX, axis=1); EV = np.concatenate(EV, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        hz = np.where(ATR[:, 1:] > 0, EVT[:, 1:] / ATR[:, 1:], np.nan)
    F = 1 - np.cumprod(1 - np.nan_to_num(hz), axis=1)
    return dict(share=REL / ON, F4=F[:, 3], F8=F[:, 7], h1=hz[:, 0], h2=hz[:, 1], h3=hz[:, 2], h4=hz[:, 3], mh=mh(EX, EV),
                breach_week_share=None)


sim = summarise(ON, REL, ATR, EVT, EX, EV); obs = summarise(OBS["ON"], OBS["REL"], OBS["ATR"], OBS["EVT"], OBS["EX"], OBS["EV"])
rows = []
for name, r in [("observed", obs), ("AR null", sim)]:
    for st in ["share", "F4", "F8", "h1", "h2", "h3", "h4"]:
        v = np.asarray(r[st], float)
        rows.append(dict(model=name, stat=st, value=np.nanmean(v), lo=np.nanquantile(v, 0.025), hi=np.nanquantile(v, 0.975),
                         p_ge_obs=np.nan if name == "observed" else float(np.mean(v >= obs[st][0]))))
    for b, lab in enumerate(["1", "2", "3-4", "5-8"]):
        v = r["mh"][:, b]
        rows.append(dict(model=name, stat="mh_" + lab, value=np.nanmean(v), lo=np.nanquantile(v, 0.025), hi=np.nanquantile(v, 0.975),
                         p_ge_obs=np.nan if name == "observed" else float(np.mean(v >= obs["mh"][0, b]))))
P = np.array(PHI)
rows.append(dict(model="AR null", stat="median_phi_NH4N_DO_CODMn", value=np.nan, lo=np.nan, hi=np.nan,
                 p_ge_obs=np.nan, note=";".join(f"{v:.2f}" for v in np.median(P, axis=0))))
pd.DataFrame(rows).to_csv(OUT / "ar_null.csv", index=False)
pd.set_option("display.width", 200); print(pd.DataFrame(rows).round(4).to_string())
