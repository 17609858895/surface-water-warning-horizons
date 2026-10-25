"""Neural discrete-time hazard (logistic hazard / Nnet-survival style), right- or interval-censored."""
import numpy as np, torch, torch.nn as nn
torch.set_num_threads(16)
K = 12
class Prep:
    def fit(self, X):
        self.med = np.nanmedian(X, 0); Z = np.where(np.isnan(X), self.med, X)
        self.mu = Z.mean(0); self.sd = Z.std(0) + 1e-6; return self
    def __call__(self, X):
        miss = np.isnan(X).astype(np.float32)
        Z = (np.where(np.isnan(X), self.med, X) - self.mu) / self.sd
        return np.hstack([np.clip(Z, -6, 6), miss[:, self.mcols]]).astype(np.float32)
class Net(nn.Module):
    def __init__(self, d, hdim=128, p=0.2):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(d, hdim), nn.GELU(), nn.Dropout(p), nn.Linear(hdim, hdim), nn.GELU(), nn.Dropout(p), nn.Linear(hdim, K))
    def forward(self, x): return self.f(x)
def rc_loss(logit, T, E):
    lh = torch.nn.functional.logsigmoid(logit); l1h = torch.nn.functional.logsigmoid(-logit)
    cs = torch.cat([torch.zeros(len(T), 1), torch.cumsum(l1h, 1)], 1)
    r = torch.arange(len(T)); t = T.long()
    ll = torch.where(E > 0, cs[r, (t - 1).clamp(min=0)] + lh[r, (t - 1).clamp(min=0)], cs[r, t])
    m = (t > 0).float()
    return -(ll * m).sum() / m.sum().clamp(min=1)
class NNHazard:
    name = "DH-NN"
    def __init__(self, epochs=60, lr=1e-3, wd=1e-4, seed=7):
        self.ep = epochs; self.lr = lr; self.wd = wd; self.seed = seed
    def fit(self, Ltr, Xtr, wtr, Lva, Xva, wva):
        torch.manual_seed(self.seed); np.random.seed(self.seed)
        self.pp = Prep().fit(Xtr); self.pp.mcols = np.where(np.isnan(Xtr).mean(0) > 0.01)[0]
        A = torch.tensor(self.pp(Xtr)); B = torch.tensor(self.pp(Xva))
        T = torch.tensor(Ltr["T"].values, dtype=torch.float32); E = torch.tensor(Ltr["E"].values, dtype=torch.float32)
        Tv = torch.tensor(Lva["T"].values, dtype=torch.float32); Ev = torch.tensor(Lva["E"].values, dtype=torch.float32)
        self.net = Net(A.shape[1]); opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.wd)
        best = 1e9; state = None; bad = 0
        for ep in range(self.ep):
            self.net.train(); perm = torch.randperm(len(A))
            for b in range(0, len(A), 512):
                ix = perm[b:b + 512]; opt.zero_grad(); loss = rc_loss(self.net(A[ix]), T[ix], E[ix]); loss.backward(); opt.step()
            self.net.eval()
            with torch.no_grad(): v = rc_loss(self.net(B), Tv, Ev).item()
            if v < best - 1e-4: best = v; state = {k: x.clone() for k, x in self.net.state_dict().items()}; bad = 0
            else:
                bad += 1
                if bad >= 8: break
        self.net.load_state_dict(state); return self
    def hazard(self, X, ws=None, wc=None):
        self.net.eval()
        with torch.no_grad(): return torch.sigmoid(self.net(torch.tensor(self.pp(X)))).numpy()
