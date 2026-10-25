"""Gap-aware three-state hazard boosting (GAMS).

Hidden discrete-time chain over the K weeks after a landmark:
  A = compliant, not yet breached since landmark;  B = in breach;  R = compliant after a within-window recovery
Transitions: A->B (onset a_k), B->R (recovery b_k), R->B (relapse c_k).
Observed weekly status o_k in {0,1,NaN}; 0 -> {A,R}, 1 -> {B}; NaN (monitoring gap) -> any state.
EM: forward-backward gives posterior source-state occupancies and transition probabilities per (landmark,k);
M-step fits one gradient-boosted hazard with transition type as a feature
(cross-entropy on soft labels, weights = posterior source-state occupancy).
"""
import numpy as np, lightgbm as lgb
from survlib import K, LGB_PARAMS


def emissions(O):
    """O: n x (K+1) with 0/1/nan -> n x (K+1) x 3 emission indicators for (A,B,R)."""
    e = np.ones((O.shape[0], O.shape[1], 3))
    e[O == 0, 1] = 0
    e[O == 1, 0] = 0
    e[O == 1, 2] = 0
    return e


def trans_mats(a, b, c):
    n = a.shape[0]
    P = np.zeros((n, K, 3, 3))
    P[:, :, 0, 0] = 1 - a; P[:, :, 0, 1] = a
    P[:, :, 1, 1] = 1 - b; P[:, :, 1, 2] = b
    P[:, :, 2, 2] = 1 - c; P[:, :, 2, 1] = c
    return P


def forward_backward(O, s0, a, b, c):
    """Posterior occupancy gamma (n,K+1,3), pair posteriors xi (n,K,3,3), observed-data log-lik per landmark."""
    n = O.shape[0]
    e = emissions(O)
    P = trans_mats(a, b, c)
    al = np.zeros((n, K + 1, 3)); be = np.ones((n, K + 1, 3)); sc = np.ones((n, K + 1))
    al[:, 0, 0] = (s0 == 0); al[:, 0, 1] = (s0 == 1)
    for k in range(1, K + 1):
        v = np.einsum("ni,nij->nj", al[:, k - 1], P[:, k - 1]) * e[:, k]
        s = v.sum(1); s[s <= 0] = 1e-300
        sc[:, k] = s; al[:, k] = v / s[:, None]
    for k in range(K - 1, -1, -1):
        v = np.einsum("nij,nj->ni", P[:, k], e[:, k + 1] * be[:, k + 1])
        be[:, k] = v / sc[:, k + 1][:, None]
    gam = al * be
    gam /= np.maximum(gam.sum(2, keepdims=True), 1e-300)
    xi = al[:, :-1, :, None] * P * (e[:, 1:] * be[:, 1:])[:, :, None, :] / sc[:, 1:, None, None]
    ll = np.log(sc[:, 1:]).sum(1)
    return gam, xi, ll


def last_obs(O):
    m = ~np.isnan(O)
    return np.where(m, np.arange(O.shape[1])[None, :], 0).max(1)


class GAMS:
    name = "GAMS"

    def __init__(self, n_em=5, rounds=800, params=None, use_em=True, three_state=True):
        self.n_em = n_em; self.rounds = rounds
        self.p = dict(LGB_PARAMS, objective="cross_entropy", **(params or {}))
        self.use_em = use_em; self.three = three_state

    def _design(self, X, ws, wc):
        n = len(X)
        kk = np.tile(np.arange(1, K + 1), n)
        ang = np.arctan2(np.repeat(ws, K), np.repeat(wc, K)) + 2 * np.pi * kk / 52.18
        base = np.column_stack([np.repeat(X, K, 0), kk, np.sin(ang), np.cos(ang)])
        # relapse shares the onset type code when the chain is collapsed to two states
        codes = (0, 1, 2) if self.three else (0, 1, 0)
        return [np.column_stack([base, np.full(n * K, t)]) for t in codes]

    def _predict_h(self, Z, n):
        return [np.clip(self.bst.predict(z), 1e-5, 1 - 1e-5).reshape(n, K) for z in Z]

    def _soft_rows(self, O, s0, a, b, c, kmax):
        gam, xi, ll = forward_backward(O, s0, a, b, c)
        valid = np.arange(1, K + 1)[None, :] <= kmax[:, None]
        occ = gam[:, :-1, :]
        lab = [xi[:, :, 0, 1], xi[:, :, 1, 2], xi[:, :, 2, 1]]
        W, Y = [], []
        for t in range(3):
            w = occ[:, :, t] * valid
            y = np.where(w > 1e-9, lab[t] / np.maximum(w, 1e-12), 0)
            W.append(w.ravel()); Y.append(np.clip(y.ravel(), 0, 1))
        return W, Y

    @staticmethod
    def _hard_rows(O, s0):
        """Observed consecutive pairs whose hidden source state is known (stops at the first gap)."""
        n = O.shape[0]
        W = [np.zeros((n, K)) for _ in range(3)]; Y = [np.zeros((n, K)) for _ in range(3)]
        st = np.where(s0 == 0, 0, 1).astype(float)
        known = np.ones(n, bool)
        for k in range(1, K + 1):
            o = O[:, k]; ok = known & ~np.isnan(o)
            for t in range(3):
                m = ok & (st == t)
                W[t][m, k - 1] = 1
                Y[t][m, k - 1] = (o[m] == (0 if t == 1 else 1))
            new = np.where(o == 1, 1, np.where(st == 1, 2, st))
            st = np.where(ok, new, st); known = ok
        return [w.ravel() for w in W], [y.ravel() for y in Y]

    def _stack_rows(self, Z, W, Y):
        Xs = np.vstack(Z); ys = np.concatenate(Y); ws = np.concatenate(W)
        m = ws > 1e-6
        return Xs[m], ys[m], ws[m]

    def fit(self, Otr, s0tr, Xtr, wtr, Ova, s0va, Xva, wva, verbose=True):
        Ztr = self._design(Xtr, *wtr); Zva = self._design(Xva, *wva)
        ntr, nva = len(Xtr), len(Xva)
        ktr = last_obs(Otr); kva = last_obs(Ova)
        W, Y = self._hard_rows(Otr, s0tr); Wv, Yv = self._hard_rows(Ova, s0va)
        self.trace = []
        for it in range(self.n_em + 1):
            Xa, ya, wa = self._stack_rows(Ztr, W, Y); Xb, yb, wb = self._stack_rows(Zva, Wv, Yv)
            self.bst = lgb.train(self.p, lgb.Dataset(Xa, ya, weight=wa), 3000,
                                 valid_sets=[lgb.Dataset(Xb, yb, weight=wb)],
                                 callbacks=[lgb.early_stopping(100, verbose=False)])
            a, b, c = self._predict_h(Zva, nva)
            llv = forward_backward(Ova, s0va, a, b, c)[2].mean()
            self.trace.append(dict(it=it, val_obs_loglik=llv, trees=self.bst.best_iteration))
            if verbose:
                print("  EM", it, "val observed-data loglik", round(llv, 4), "trees", self.bst.best_iteration, flush=True)
            if not self.use_em or it == self.n_em:
                break
            a, b, c = self._predict_h(Ztr, ntr)
            W, Y = self._soft_rows(Otr, s0tr, a, b, c, ktr)
            Wv, Yv = self._soft_rows(Ova, s0va, *self._predict_h(Zva, nva), kva)
        return self

    def hazards(self, X, ws, wc):
        return self._predict_h(self._design(X, ws, wc), len(X))

    @staticmethod
    def first_passage(h):
        return 1 - np.cumprod(1 - h, axis=1)

    @staticmethod
    def occupancy(s0, a, b, c):
        n = len(s0)
        P = trans_mats(a, b, c)
        p = np.zeros((n, 3)); p[s0 == 0, 0] = 1; p[s0 == 1, 1] = 1
        occ = np.zeros((n, K))
        for k in range(K):
            p = np.einsum("ni,nij->nj", p, P[:, k]); occ[:, k] = p[:, 1]
        return occ
