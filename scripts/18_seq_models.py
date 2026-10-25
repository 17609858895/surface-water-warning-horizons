"""Deep sequence baselines (Fig. S4): a GRU encoder over the past 52 weeks of the four indicators (+ masks + state).
(1) GRU-Forecast: direct 1-12-week concentration forecasts (NH4-N, DO, CODMn) -> NSE and forecast->threshold warnings.
(2) GRU-Hazard: discrete-time hazard head (right-censored likelihood), per transition; also tested inside the BRR-SL stack.
Same temporal split as the core analysis; normalisation from training years only; year-stratified IPCW evaluation."""
import sys, numpy as np, pandas as pd, torch, torch.nn as nn
from pathlib import Path
from scipy.stats import norm
sys.path.insert(0, str(Path(__file__).parent))
from survlib import *
from nnhazard import rc_loss

torch.set_num_threads(12); torch.manual_seed(18); np.random.seed(18)
OUT = Path(__file__).resolve().parents[1] / "results"; SD = OUT / "seq"; SD.mkdir(exist_ok=True)
L = pd.read_parquet(OUT / "landmarks.parquet").reset_index(drop=True); w = pd.read_parquet(OUT / "cnemc_weekly.parquet")
state = L.state.values.astype(int); W = 52
SPL = {"tr": (L.date <= "2013-09-30").values, "va": ((L.date >= "2014-01-01") & (L.date <= "2014-09-30")).values,
       "te": (L.date >= "2015-01-01").values}
TF = {"NH4N": np.log1p, "DO": lambda x: x, "CODMn": np.log1p, "pH": lambda x: x}
THR = {"NH4N": (np.log1p(1.0), +1), "DO": (5.0, -1), "CODMn": (np.log1p(6.0), +1)}

# ---------- sequences ----------
SEQ = np.zeros((len(L), W, 10), np.float32); Y = {v: np.full((len(L), K + 1), np.nan, np.float32) for v in THR}
tr_rows = w.date <= "2013-09-30"
mu = {v: np.nanmean(TF[v](w.loc[tr_rows, v].values)) for v in TF}; sd = {v: np.nanstd(TF[v](w.loc[tr_rows, v].values)) for v in TF}
for s, g in w.groupby("site"):
    g = g.set_index("date").sort_index(); idx = pd.date_range(g.index.min(), g.index.max(), freq="W-MON"); g = g.reindex(idx)
    n = len(idx); A = np.zeros((n + W, 10), np.float32)
    for j, v in enumerate(TF):
        x = (TF[v](g[v].values) - mu[v]) / sd[v]; A[W:, j] = np.nan_to_num(x); A[W:, 4 + j] = ~np.isnan(x)
    ob = g[["NH4N", "DO", "CODMn"]].notna().all(axis=1).values
    ex = ((g.NH4N > 1) | (g.DO < 5) | (g.CODMn > 6)).values & ob
    A[W:, 8] = ex; A[W:, 9] = ob
    pos = pd.Series(np.arange(n), index=idx); li = np.where(L.site.values == s)[0]; p = pos.reindex(L.date.values[li]).values.astype(int)
    for a, pi in zip(li, p): SEQ[a] = A[pi + 1: pi + 1 + W]
    for v in THR:
        sv = np.concatenate([TF[v](g[v].values), np.full(K, np.nan)]); Y[v][li] = np.stack([sv[p + k] for k in range(K + 1)], 1)
ST = np.column_stack([(L.lat - L.lat.mean()) / L.lat.std(), (L.lon - L.lon.mean()) / L.lon.std(), L.woy_sin, L.woy_cos, state]).astype(np.float32)
print("sequences built", SEQ.shape, flush=True)


class Net(nn.Module):
    def __init__(self, out, h=64):
        super().__init__()
        self.gru = nn.GRU(10, h, batch_first=True); self.head = nn.Sequential(nn.Linear(h + ST.shape[1], 128), nn.GELU(), nn.Dropout(0.2), nn.Linear(128, out))
    def forward(self, s, z):
        _, hN = self.gru(s); return self.head(torch.cat([hN[-1], z], 1))


def train(idx_tr, idx_va, out, loss_fn, targets, epochs=30, lr=1e-3):
    net = Net(out); opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    S, Z = torch.tensor(SEQ), torch.tensor(ST); best, bad, keep = 1e9, 0, None
    for ep in range(epochs):
        net.train(); perm = np.random.permutation(idx_tr)
        for b in range(0, len(perm), 512):
            ix = perm[b:b + 512]; opt.zero_grad(); loss = loss_fn(net(S[ix], Z[ix]), *[t[ix] for t in targets]); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad(): v = float(loss_fn(net(S[idx_va], Z[idx_va]), *[t[idx_va] for t in targets]))
        if v < best - 1e-4: best, bad, keep = v, 0, {k: x.clone() for k, x in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 4: break
    net.load_state_dict(keep); net.eval(); return net


def predict(net, idx):
    with torch.no_grad(): return net(torch.tensor(SEQ[idx]), torch.tensor(ST[idx])).numpy()


# ---------- (1) GRU forecast ----------
YT = torch.tensor(np.concatenate([np.nan_to_num(Y[v][:, 1:]) for v in THR], 1))
MT = torch.tensor(np.concatenate([~np.isnan(Y[v][:, 1:]) for v in THR], 1).astype(np.float32))
mse = lambda o, y, m: ((o - y) ** 2 * m).sum() / m.sum().clamp(min=1)
itr, iva, ite = [np.where(SPL[k])[0] for k in ("tr", "va", "te")]
fnet = train(itr, iva, 36, mse, [YT, MT])
Pv, Pt = predict(fnet, iva), predict(fnet, ite)
nse, pe = [], {}
for j, v in enumerate(THR):
    for k in range(1, K + 1):
        c = j * K + (k - 1); yv, yt = Y[v][iva, k], Y[v][ite, k]
        sig = np.nanstd(yv - Pv[:, c]); m = ~np.isnan(yt)
        nse.append(dict(ind=v, k=k, NSE_GRU=1 - np.sum((yt[m] - Pt[m, c]) ** 2) / np.sum((yt[m] - yt[m].mean()) ** 2)))
        thr, sgn = THR[v]; z = (thr - Pt[:, c]) / sig; pe[(v, k)] = norm.sf(z) if sgn > 0 else norm.cdf(z)
NSE = pd.DataFrame(nse); NSE.to_csv(SD / "gru_nse.csv", index=False); print(NSE[NSE.k.isin([1, 4, 12])].round(3).to_string(), flush=True)
pweek = 1 - np.prod([1 - np.column_stack([pe[(v, k)] for k in range(1, K + 1)]) for v in THR], axis=0)

# ---------- (2) GRU hazard + stack ----------
H = pd.read_pickle(OUT / "core" / "hazards.pkl"); rows = []
TT = torch.tensor(L["T"].values, dtype=torch.float32); EE = torch.tensor(L["E"].values, dtype=torch.float32)
for trans, st in [("onset", 0), ("recovery", 1)]:
    m = {k: np.where(SPL[k] & (state == st))[0] for k in SPL}
    hnet = train(m["tr"], m["va"], K, rc_loss, [TT, EE])
    hv = 1 / (1 + np.exp(-predict(hnet, m["va"]))); ht = 1 / (1 + np.exp(-predict(hnet, m["te"])))
    Lva = L.iloc[m["va"]].reset_index(drop=True); Lte = L.iloc[m["te"]].reset_index(drop=True); yr = Lte.date.dt.year.values
    mem = ["DH-GBM+FT", "XGB-AFT+FT", "DH-NN+FT", "3S-Boost+FT"]
    wgt = stack_weights([H[trans][b]["va"] for b in mem] + [hv], Lva)
    hst = sum(wi * x for wi, x in zip(wgt, [H[trans][b]["te"] for b in mem] + [ht]))
    pw = pweek[state[SPL["te"]] == st]; pw = pw if st == 0 else 1 - pw
    cand = {"GRU-Hazard": ht, "BRR-SL": H[trans]["BRR-SL"]["te"], "BRR-SL + GRU-Hazard": hst}
    Fs = {k: 1 - surv_from_h(v) for k, v in cand.items()}
    Fs["GRU-Forecast→threshold (max)"] = np.maximum.accumulate(pw, 1); Fs["GRU-Forecast→threshold (indep.)"] = 1 - np.cumprod(1 - pw, 1)
    for nm, F in Fs.items():
        for per, msk in [("2015-2016", yr <= 2016), ("2017-2018", yr >= 2017)]:
            r = eval_ipcw_strat(F[msk], Lte[msk], yr[msk], horizons=(1, 2, 4, 8, 12))
            nll = nll_rc(cand[nm][msk], Lte[msk]) if nm in cand else np.nan
            rows.append(dict(trans=trans, model=nm, period=per, NLL=nll, **r))
    rows.append(dict(trans=trans, model="stack weights", period="", **{f"w_{b}": wi for b, wi in zip(mem + ["GRU-Hazard"], wgt)}))
    print(trans, "stack weights", dict(zip(mem + ["GRU-Hazard"], np.round(wgt, 3))), flush=True)
R = pd.DataFrame(rows); R.to_csv(SD / "gru_eval.csv", index=False)
print(R[R.period == "2015-2016"][["trans", "model", "NLL", "IBS", "AUCw@1", "AUCw@4", "AUCw@12", "Pred@12", "ObsIPCW@12"]].round(4).to_string())
