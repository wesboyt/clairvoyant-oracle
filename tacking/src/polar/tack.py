"""Tacking dynamics: on-policy Q^pi labels (south) with the clairvoyant max-flow relative regret (north) swapped in
at closed doors by the trader's gate rule (PLAN D23a), and the two pure arms. theory.md S4.

Labels per player i and key k (exact expectations; one 'update' = the expected step over on-policy pivots):
    south   q = Q^{pi~}(k, .): the pruned rollout value (D17: at non-pivot nodes keep a iff pi(a) >= p_prune max pi,
            renormalise; the pivot key expands every action)
    north   q = mf(k, .): the clairvoyant max-flow field (oracle.clairvoyant_field) at every key; two players: the
            clairvoyant best response against the CURRENT opponent (poker's node-locked BR, made clairvoyant)
    tackall the refinement D23b: as tack, but every closed north outlier is raised (not only mf's argmax)
    tack    south, except at flagged keys (D23a): a* = argmax mf, pi(a*) <= eps (5%), r = mf(a*) - sum_b pi(b) mf(b),
            z = r / s_mf >= z_min (1.0) -> q(a*) = max_b q(b) + v_eps * s_q (v_eps = 0.25); siblings keep q
    vswap   the pre-D23 rule: at keys with a closed door a* (pi <= eps) that is mf's argmax (within delta),
            q <- mf - mean mf + mean q (every sibling)
Update (the repo's surrogate at ratio 1 = NeuRD, ALGORITHM_SPEC S5 / leduc ALGORITHM S5.1), plain SGD:
    theta(k, a) += lr * w(k) * (q(k, a) - mean_a q(k, .)) / sigma,   w(k) = occ(k) / max occ  (on-policy pivots)
with sigma an EMA of the occupancy-weighted RMS advantage (sigma_ref frozen at step 0). Optional entropy anchor.
"""
import numpy as np
import scipy.sparse as sp

from . import oracle as O


def pruned(pl, pi, p_prune):
    if p_prune <= 0:
        return pi
    mx = np.maximum.reduceat(pi[1:], pl.start[:-1] - 1)[pl.key_of_seq[1:]]
    keep = pi[1:] >= p_prune * mx
    out = pi.copy()
    out[1:] = np.where(keep, pi[1:], 0.0)
    sm = pl.seg_sum(out)
    out[1:] = out[1:] / sm[pl.key_of_seq[1:]]
    return out


class Clairvoyant:
    """North field. One player: fixed (no opponent). Two players: recomputed against the current opponent."""

    def __init__(self, game):
        self.g = game
        self.fixed = None
        if game.players == 1:
            if not hasattr(game, '_north_fixed'):
                game._north_fixed = [O.clairvoyant_field(game, game.uniform(), 0)[0]]
            self.fixed = game._north_fixed
        else:
            n0, n1 = game.plex[0].n, game.plex[1].n
            self.Ad = []
            for p, nodes, ent in zip(game.deal_probs, game.deal_graphs, game.deal_A):
                ij = np.array([(a, b) for a, b, _ in ent], np.int64)
                A = sp.csr_matrix((np.array([u for _, _, u in ent]), (ij[:, 0], ij[:, 1])), shape=(n0, n1))
                rk = [[], []]
                for (pl_i, k, _, sq_opp, ch) in nodes:
                    rk[pl_i].append((k, sq_opp))
                self.Ad.append((p, A, A.T.tocsr(), rk))

    def field(self, pis, xs):
        if self.fixed is not None:
            return self.fixed
        out = []
        for i in range(2):
            pl = self.g.plex[i]
            num = np.zeros(pl.n)
            den = np.zeros(pl.K)
            for p, A, AT, rk in self.Ad:
                gd = A @ xs[1] if i == 0 else -(AT @ xs[0])
                W, _, _ = pl.backup(np.asarray(gd).ravel())
                Rd = np.zeros(pl.K)
                for k, sq in rk[i]:
                    Rd[k] += p * xs[1 - i][sq]
                ks = np.flatnonzero(Rd > 0)
                if len(ks):
                    m = np.isin(pl.key_of_seq, ks)
                    num[m] += W[m]
                    den[ks] += Rd[ks]
            dens = den[pl.key_of_seq[1:]]
            mf = np.full(pl.n, np.nan)
            mf[1:] = np.where(dens > 0, num[1:] / np.where(dens > 0, dens, 1.0), np.nan)
            out.append(mf)
        return out


def seg_std(pl, v):
    n = pl.nact.astype(float)
    m = pl.seg_sum(v) / n
    var = pl.seg_sum(v ** 2) / n - m ** 2
    return np.sqrt(np.maximum(var, 0.0)), m


def gate_d23a(pl, q, mf, pi, eps=0.05, z_min=1.0, v_eps=0.25):
    """-> (new q, flagged key mask, a* sequence index per key or -1, z per key)."""
    K = pl.K
    q = q.copy()
    flag = np.zeros(K, bool)
    astar = np.full(K, -1)
    zs = np.zeros(K)
    s_mf, _ = seg_std(pl, np.nan_to_num(mf))
    s_q, _ = seg_std(pl, q)
    for k in range(K):
        s0, s1 = pl.start[k], pl.start[k + 1]
        if s1 - s0 < 2:
            continue
        m = mf[s0:s1]
        if not np.isfinite(m).all() or s_mf[k] <= 1e-12:
            continue
        j = int(np.argmax(m))
        if pi[s0 + j] > eps:
            continue
        r = m[j] - pi[s0:s1] @ m
        z = r / s_mf[k]
        zs[k] = z
        if z >= z_min:
            flag[k] = True
            astar[k] = s0 + j
            q[s0 + j] = q[s0:s1].max() + v_eps * s_q[k]
    return q, flag, astar, zs


def gate_all(pl, q, mf, pi, eps=0.05, z_min=1.0, v_eps=0.25):
    """Refinement 'D23b': fire on EVERY closed door a (pi(a) <= eps) that is a north outlier,
    z_a = (mf(a) - sum_b pi(b) mf(b)) / s_mf >= z_min, not only on mf's argmax."""
    q0 = q
    q = q.copy()
    flag = np.zeros(pl.K, bool)
    zs = np.zeros(pl.K)
    s_mf, _ = seg_std(pl, np.nan_to_num(mf))
    s_q, _ = seg_std(pl, q0)
    for k in range(pl.K):
        s0, s1 = pl.start[k], pl.start[k + 1]
        m = mf[s0:s1]
        if s1 - s0 < 2 or not np.isfinite(m).all() or s_mf[k] <= 1e-12:
            continue
        base = pi[s0:s1] @ m
        top = q0[s0:s1].max()
        for j in range(s1 - s0):
            if pi[s0 + j] <= eps:
                z = (m[j] - base) / s_mf[k]
                zs[k] = max(zs[k], z)
                if z >= z_min:
                    flag[k] = True
                    q[s0 + j] = top + v_eps * s_q[k]
    return q, flag, zs


def gate_vswap(pl, q, mf, pi, eps=0.05, delta=0.05):
    q = q.copy()
    flag = np.zeros(pl.K, bool)
    for k in range(pl.K):
        s0, s1 = pl.start[k], pl.start[k + 1]
        m = mf[s0:s1]
        if s1 - s0 < 2 or not np.isfinite(m).all():
            continue
        j = int(np.argmax(m))
        if pi[s0 + j] <= eps:
            flag[k] = True
            q[s0:s1] = m - m.mean() + q[s0:s1].mean()
    return q, flag


class Tacker:
    def __init__(self, game, arm, lr=0.5, p_prune=0.01, eps=0.05, z_min=1.0, v_eps=0.25, alpha_ent=0.0,
                 init=None, optim='sgd'):
        self.g, self.arm, self.lr, self.p_prune = game, arm, lr, p_prune
        self.eps, self.z_min, self.v_eps, self.alpha_ent = eps, z_min, v_eps, alpha_ent
        self.optim = optim
        pls = game.plex
        self.theta = [np.zeros(pl.n) for pl in pls] if init is None else \
            [np.log(np.maximum(p, 1e-300)) for p in init]
        self.pis = [pl.softmax(th) for pl, th in zip(pls, self.theta)]
        self.north = Clairvoyant(game) if arm != 'south' else None
        self.north_shared = None
        self.sigma = None
        self.t = 0
        self.xs = game.realise(self.pis)
        self.avg_num = [x.copy() for x in self.xs]
        self.avg_w = 1.0
        self.adam_v = [np.zeros(pl.n) for pl in pls]
        self.last = {}

    def avg(self):
        return [a / self.avg_w for a in self.avg_num]

    def labels(self):
        g = self.g
        pls = g.plex
        xs = self.xs
        pit = [pruned(pl, pi, self.p_prune) for pl, pi in zip(pls, self.pis)]
        grads = g.grads(xs)
        out, flags, zs = [], [], []
        mf = self.north.field(self.pis, xs) if self.north is not None else None
        for i, pl in enumerate(pls):
            W, _, _ = pl.backup(grads[i], pit[i])
            R = g.cf_reach(xs, i)
            Rs = np.concatenate([[1.0], R[pl.key_of_seq[1:]]])
            q = np.where(Rs > 1e-300, W / np.maximum(Rs, 1e-300), 0.0)
            fl = np.zeros(pl.K, bool)
            z = np.zeros(pl.K)
            if self.arm == 'north':
                q = np.nan_to_num(mf[i], nan=0.0)
            elif self.arm == 'tack':
                q, fl, _, z = gate_d23a(pl, q, mf[i], self.pis[i], self.eps, self.z_min, self.v_eps)
            elif self.arm == 'tackall':
                q, fl, z = gate_all(pl, q, mf[i], self.pis[i], self.eps, self.z_min, self.v_eps)
            elif self.arm == 'vswap':
                q, fl = gate_vswap(pl, q, mf[i], self.pis[i], self.eps)
            out.append(q)
            flags.append(fl)
            zs.append(z)
        return out, flags, zs, mf

    def step(self):
        g = self.g
        pls = g.plex
        qs, flags, zs, mf = self.labels()
        xs = self.xs
        advs, ws, msq = [], [], 0.0
        for i, pl in enumerate(pls):
            occ = pl.reach_in(xs[i]) * g.cf_reach(xs, i)
            w = occ / max(occ.max(), 1e-300)
            mean = pl.seg_sum(qs[i]) / pl.nact
            a = np.zeros(pl.n)
            a[1:] = qs[i][1:] - mean[pl.key_of_seq[1:]]
            advs.append(a)
            ws.append(w)
            msq += float((occ / max(occ.sum(), 1e-300) * pl.seg_sum(a ** 2) / pl.nact).sum())
        msq = max(msq / len(pls), 1e-12)
        if self.sigma is None:
            self.sigma = np.sqrt(msq)
            self.sigma_ref = self.sigma
        else:
            self.sigma = max(np.sqrt(0.99 * self.sigma ** 2 + 0.01 * msq), 0.05 * self.sigma_ref)
        for i, pl in enumerate(pls):
            wk = np.concatenate([[0.0], ws[i][pl.key_of_seq[1:]]])
            grad = wk * advs[i] / self.sigma
            if self.alpha_ent > 0:
                pi = self.pis[i]
                lp_ = np.log(np.maximum(pi, 1e-300))
                H = -pl.seg_sum(pi * lp_)
                dH = np.zeros(pl.n)
                dH[1:] = -pi[1:] * (lp_[1:] + H[pl.key_of_seq[1:]])
                grad = grad + self.alpha_ent * wk * dH
            if self.optim == 'adam':
                v = self.adam_v[i] = 0.999 * self.adam_v[i] + 0.001 * grad ** 2
                grad = grad / (np.sqrt(v / (1 - 0.999 ** (self.t + 1))) + 1e-8)
            self.theta[i] = self.theta[i] + self.lr * grad
            self.theta[i][1:] -= np.maximum.reduceat(self.theta[i][1:], pl.start[:-1] - 1)[pl.key_of_seq[1:]]
            self.pis[i] = pl.softmax(self.theta[i])
        self.t += 1
        self.xs = g.realise(self.pis)
        w = float(self.t + 1)
        for i in range(len(pls)):
            self.avg_num[i] += w * self.xs[i]
        self.avg_w += w
        self.last = dict(flags=[int(f.sum()) for f in flags],
                         zmax=[float(z.max()) if len(z) else 0.0 for z in zs])
        return self.pis
