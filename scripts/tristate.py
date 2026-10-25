"""3S-Boost: landmark-conditioned three-state (A/B/R) discrete-time hazard boosting.

States over the K weeks after a landmark:
  A  = compliant, no breach in the previous MEM weeks
  B  = in breach
  R_j = j-th compliant week after a recovery (j = 1..MEM); R_MEM without relapse returns to A
Transitions: A->B onset a_k; B->R_1 recovery b_k; R_j->B relapse c_{k,j}.
Training uses observed consecutive pairs before the first monitoring gap (hard labels), with one
LightGBM learner and the transition type (and j for R) as inputs.
With two_state=True, R is merged into A (no relapse memory) for the ablation.
"""
import numpy as np, lightgbm as lgb
from survlib import K, LGB_PARAMS

MEM = 8
NS = 2 + MEM  # 0=A, 1=B, 2..=R_1..R_MEM


def start_state(state, wk_since_ex, two_state=False):
    s = np.where(state == 1, 1, 0)
    if not two_state:
        j = np.asarray(wk_since_ex, float)
        inR = (state == 0) & (j >= 1) & (j <= MEM)
        s = np.where(inR, 1 + j.astype(int).clip(1, MEM), s)
    return s.astype(int)


def next_state(s, o, two_state=False):
    """Deterministic next hidden state given current state s and observed status o (0/1)."""
    if o == 1:
        return 1
    if s == 1:
        return 0 if two_state else 2
    if s == 0:
        return 0
    return 0 if s == 1 + MEM else s + 1


class TriStateBoost:
    def __init__(self, two_state=False, params=None):
        self.two = two_state
        self.p = dict(LGB_PARAMS, **(params or {}))

    @staticmethod
    def _type_j(s):
        if s == 0: return 0, 0
        if s == 1: return 1, 0
        return 2, s - 1

    def _rows(self, O, s0):
        idx, ks, ty, jj, y = [], [], [], [], []
        for i in range(O.shape[0]):
            s = s0[i]
            for k in range(1, K + 1):
                o = O[i, k]
                if np.isnan(o): break
                t, j = self._type_j(s)
                idx.append(i); ks.append(k); ty.append(t); jj.append(j)
                y.append(int((o == 1) if t != 1 else (o == 0)))
                s = next_state(s, int(o), self.two)
        return np.array(idx), np.array(ks), np.array(ty), np.array(jj), np.array(y)

    @staticmethod
    def _design(X, ws, wc, idx, ks, ty, jj):
        ang = np.arctan2(ws[idx], wc[idx]) + 2 * np.pi * ks / 52.18
        return np.column_stack([X[idx], ks, np.sin(ang), np.cos(ang), ty, jj])

    def fit(self, Otr, s0tr, Xtr, wtr, Ova, s0va, Xva, wva):
        a = self._rows(Otr, s0tr); b = self._rows(Ova, s0va)
        A = self._design(Xtr, *wtr, *a[:4]); B = self._design(Xva, *wva, *b[:4])
        self.bst = lgb.train(self.p, lgb.Dataset(A, a[4]), 3000, valid_sets=[lgb.Dataset(B, b[4])],
                             callbacks=[lgb.early_stopping(100, verbose=False)])
        self.n_feat = Xtr.shape[1]
        return self

    def all_hazards(self, X, ws, wc):
        """Hazard for every source state and step: array n x K x NS (prob. of leaving the state)."""
        n = len(X); H = np.zeros((n, K, NS))
        states = [0, 1] + ([] if self.two else list(range(2, NS)))
        base_idx = np.repeat(np.arange(n), K); kk = np.tile(np.arange(1, K + 1), n)
        for s in states:
            t, j = self._type_j(s)
            D = self._design(X, ws, wc, base_idx, kk, np.full(n * K, t), np.full(n * K, j))
            H[:, :, s] = np.clip(self.bst.predict(D, num_iteration=self.bst.best_iteration), 1e-5, 1 - 1e-5).reshape(n, K)
        if self.two:
            H[:, :, 2:] = H[:, :, [0]]
        return H

    def propagate(self, H, s0):
        """Forward recursion. Returns first-passage CDFs and breach occupancy.
        F_onset[i,k]  = P(first breach by k | compliant start), absorbing B
        F_recov[i,k]  = P(first compliant week by k | breach start)
        occ[i,k]      = P(in breach at week k)"""
        n = len(s0)
        Fon = np.zeros((n, K)); Frc = np.zeros((n, K)); occ = np.zeros((n, K))
        # occupancy / multi-state
        p = np.zeros((n, NS)); p[np.arange(n), s0] = 1
        # first passage (compliant start): mass not yet absorbed
        q = p.copy(); q[:, 1] = 0; absorbed = np.zeros(n)
        surv_b = (s0 == 1).astype(float)
        for k in range(K):
            h = H[:, k, :]
            newp = np.zeros_like(p)
            # A
            newp[:, 0] += p[:, 0] * (1 - h[:, 0]); newp[:, 1] += p[:, 0] * h[:, 0]
            # B
            newp[:, 1] += p[:, 1] * (1 - h[:, 1])
            if self.two: newp[:, 0] += p[:, 1] * h[:, 1]
            else: newp[:, 2] += p[:, 1] * h[:, 1]
            # R_j
            for s in range(2, NS):
                newp[:, 1] += p[:, s] * h[:, s]
                nxt = 0 if s == NS - 1 else s + 1
                newp[:, nxt] += p[:, s] * (1 - h[:, s])
            p = newp; occ[:, k] = p[:, 1]
            # first-passage onset
            hit = (q * h).copy(); hit[:, 1] = 0
            absorbed += hit.sum(1)
            nq = np.zeros_like(q)
            nq[:, 0] += q[:, 0] * (1 - h[:, 0])
            for s in range(2, NS):
                nxt = 0 if s == NS - 1 else s + 1
                nq[:, nxt] += q[:, s] * (1 - h[:, s])
            q = nq; Fon[:, k] = absorbed
            surv_b = surv_b * (1 - h[:, 1]); Frc[:, k] = 1 - surv_b
        return Fon, Frc, occ
