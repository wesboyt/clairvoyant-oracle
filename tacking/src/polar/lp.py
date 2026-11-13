"""Exact solvers: the game value, the optimal (Nash) set X*_i as a polytope, its extreme points ('poles'), its
dimension, and the L1 distance of a strategy to it.

Two players (zero-sum, sequence form; von Stengel 1996):
    X*_0 = {x : E x = e, x >= 0, exists q: F^T q <= A^T x, f^T q >= v}
    X*_1 = {y : F y = f, y >= 0, exists p: E^T p >= A y, e^T p <= v}
One player: X* = {x in X : x(k, a) = 0 whenever W*(k, a) < V*(k)} (the optimal face; theory.md Lemma 1).
"""
import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

TOL = 1e-9


def _lp(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=None):
    r = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    if r.status != 0:
        raise RuntimeError('LP failed: %s' % r.message)
    return r


class OptimalSet:
    """The polytope X*_i of player i's optimal strategies, as LP constraints over z = [x_i, dual]."""

    def __init__(self, game, i, v=None):
        self.game, self.i = game, i
        pl = game.plex[i]
        self.n = pl.n
        if game.players == 1:
            W, V, vopt = pl.backup(game.c)
            self.v = vopt
            Rs = np.concatenate([[1.0], np.ones(pl.n - 1)])
            R = np.asarray(game.B[0] @ np.ones(1)).ravel()
            gap = (V[pl.key_of_seq[1:]] - W[1:]) / R[pl.key_of_seq[1:]]       # normalised Q gap
            self.zero = np.concatenate([[False], gap > TOL * max(1.0, abs(vopt))])
            nz = self.n
            self.A_eq = pl.E
            self.b_eq = pl.e
            self.A_ub, self.b_ub = None, None
            self.bounds = [(0, 0) if z else (0, None) for z in self.zero]
            self.nz = nz
            return
        A = game.A if i == 0 else -game.A.T            # player i maximises x_i^T A_i x_-i
        Ei, ei = pl.E, pl.e
        Fo, fo = game.plex[1 - i].E, game.plex[1 - i].e
        no, ko = Fo.shape[1], Fo.shape[0]
        # max fo^T q  s.t.  Fo^T q - A_i^T x <= 0
        if v is None:
            c = np.concatenate([np.zeros(self.n), -fo])
            A_ub = sp.hstack([-A.T, Fo.T]).tocsr()
            A_eq = sp.hstack([Ei, sp.csr_matrix((Ei.shape[0], ko))]).tocsr()
            bounds = [(0, None)] * self.n + [(None, None)] * ko
            r = _lp(c, A_ub, np.zeros(no), A_eq, ei, bounds)
            v = -r.fun
            self.x_lp = r.x[:self.n]
        self.v = v
        self.A_ub = sp.vstack([sp.hstack([-A.T, Fo.T]),
                               sp.hstack([sp.csr_matrix((1, self.n)), -sp.csr_matrix(fo)])]).tocsr()
        self.b_ub = np.concatenate([np.zeros(no), [-(v - TOL * max(1.0, abs(v)))]])
        self.A_eq = sp.hstack([Ei, sp.csr_matrix((Ei.shape[0], ko))]).tocsr()
        self.b_eq = ei
        self.bounds = [(0, None)] * self.n + [(None, None)] * ko
        self.nz = self.n + ko

    def extreme(self, w):
        """argmax_{x in X*} w . x  (a vertex)."""
        c = np.zeros(self.nz)
        c[:self.n] = -w
        r = _lp(c, self.A_ub, self.b_ub, self.A_eq, self.b_eq, self.bounds)
        return r.x[:self.n]

    def l1_dist(self, x):
        """min_{x' in X*} ||x - x'||_1 (an LP)."""
        n = self.n
        # variables [z (nz), t (n)]:  x' - t <= x ;  -x' - t <= -x
        I = sp.identity(n, format='csr')
        Zx = sp.hstack([I, sp.csr_matrix((n, self.nz - n))])
        rows = [sp.hstack([Zx, -I]), sp.hstack([-Zx, -I])]
        b = [x, -x]
        if self.A_ub is not None:
            rows.append(sp.hstack([self.A_ub, sp.csr_matrix((self.A_ub.shape[0], n))]))
            b.append(self.b_ub)
        A_ub = sp.vstack(rows).tocsr()
        A_eq = sp.hstack([self.A_eq, sp.csr_matrix((self.A_eq.shape[0], n))]).tocsr()
        c = np.concatenate([np.zeros(self.nz), np.ones(n)])
        r = _lp(c, A_ub, np.concatenate(b), A_eq, self.b_eq, list(self.bounds) + [(0, None)] * n)
        return float(r.fun)

    def coordinate_ranges(self, idx=None):
        """For each sequence j: (min, max) of x_j over X*. Sequences with min < max are 'optional doors'."""
        idx = range(1, self.n) if idx is None else idx
        out = {}
        for j in idx:
            w = np.zeros(self.n); w[j] = 1.0
            hi = self.extreme(w)[j]
            lo = self.extreme(-w)[j]
            out[j] = (lo, hi)
        return out

    def vertices(self, n_rand=16, rng=None, coords=True):
        """A set of extreme points: +-e_j for every sequence (coords) and random directions."""
        rng = rng or np.random.default_rng(0)
        V = []
        if coords:
            for j in range(1, self.n):
                for sgn in (1.0, -1.0):
                    w = np.zeros(self.n); w[j] = sgn
                    V.append(self.extreme(w))
        for _ in range(n_rand):
            V.append(self.extreme(rng.standard_normal(self.n)))
        V = np.array(V)
        # dedupe
        keep = []
        for v in V:
            if not any(np.abs(v - u).max() < 1e-7 for u in keep):
                keep.append(v)
        return np.array(keep)


def affine_dim(V, tol=1e-7):
    if len(V) <= 1:
        return 0
    D = V[1:] - V[0]
    s = np.linalg.svd(D, compute_uv=False)
    return int((s > tol * max(1.0, s.max())).sum())


def diameter_pair(V):
    """The two most distant points of a finite set (the 'polar optima')."""
    if len(V) == 1:
        return V[0], V[0]
    G = ((V[:, None, :] - V[None, :, :]) ** 2).sum(-1)
    i, j = np.unravel_index(np.argmax(G), G.shape)
    return V[i], V[j]


def single_agent_optimum(game):
    """(V_opt, pi* (DP argmax, subgame perfect), the optimal-face structure: number of optimal actions per key)."""
    pl = game.plex[0]
    W, V, vopt = pl.backup(game.c)
    pi = pl.br_policy(W)
    R = np.asarray(game.B[0] @ np.ones(1)).ravel()
    gap = (V[pl.key_of_seq[1:]] - W[1:]) / R[pl.key_of_seq[1:]]           # normalised Q gap (cents)
    opt = np.concatenate([[True], gap <= TOL * max(1.0, abs(vopt))])
    nopt = np.add.reduceat(opt[1:].astype(int), pl.start[:-1] - 1)
    return vopt, pi, opt, nopt


def single_agent_poles(game):
    """Two deterministic optimal policies that differ at every key with a tie: first vs last optimal action.
    Both are vertices of X*; also returns the optimal-face dimension = sum over keys reachable by optimal
    play of (#optimal actions - 1)."""
    pl = game.plex[0]
    vopt, pi1, opt, nopt = single_agent_optimum(game)
    pi2 = np.zeros(pl.n); pi2[0] = 1.0
    for k in range(pl.K):
        s0, s1 = pl.start[k], pl.start[k + 1]
        o = np.flatnonzero(opt[s0:s1])
        pi2[s0 + o[-1]] = 1.0
    # keys reachable with some optimal policy: forward pass with uniform over optimal actions
    piu = np.where(opt, 1.0, 0.0); piu[0] = 1.0
    sm = pl.seg_sum(piu)
    piu[1:] = piu[1:] / sm[pl.key_of_seq[1:]]
    xu = pl.realise(piu)
    reach = pl.reach_in(xu) > 0
    dim = int(((nopt - 1) * reach).sum())
    return pl.realise(pi1), pl.realise(pi2), dict(v=vopt, dim=dim, tie_keys=int(((nopt > 1) & reach).sum()),
                                                   reach_keys=int(reach.sum()), pi1=pi1, pi2=pi2, opt=opt,
                                                   uniform_opt=piu)


def nash(game):
    """Exact Nash by LP: (x*, y*, v) with subgame-perfect-ish completion off the equilibrium path (best response
    to the opponent's equilibrium strategy at keys the equilibrium does not reach)."""
    S0, S1 = OptimalSet(game, 0), OptimalSet(game, 1)
    return S0.x_lp, S1.x_lp, S0.v, (S0, S1)
