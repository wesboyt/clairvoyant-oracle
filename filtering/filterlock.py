"""The disagreement filter with a KL lock, on exactly solvable games.

Every quantity is exact except the choice of decisions, which is sampled on-policy as the method prescribes:
    Q^pi(k, a)   on-policy action value (the opponent and the player's own continuation play pi_t)
    Q*(k, a)     the oracle: the player best-responds at every later decision, the opponent is held at pi_t
    V(k)         sum_a pi_t(a|k) Q^pi(k, a)
    d(k)         max_a Q*(k, a) - V(k) >= 0      the decision's disagreement with the oracle
Candidates are drawn with probability proportional to occupancy (own reach x chance x opponent reach).

One update, per player, simultaneously for all players:
    1. sample candidates: n_u from the unfiltered decisions, K * n_f from the filtered ones
    2. bar b = the (1 - f) quantile of the filtered candidates' scores; S = the first n_f candidates above it (filled
       with the highest-scoring remaining ones if short); pivots P = unfiltered candidates + S; lock set
       L = filtered candidates not in P whose score is at most b (they agree with the oracle)
    3. labels on pivots: q = Q^pi, swapped to Q* where some action has pi < eps and Q* within delta of the best
    4. ascent on  (1/|P|) sum_P <pi_theta / pi_t, A>/|A(k)|  -  beta (1/|L|) sum_L KL(pi_T(.|k) || pi_theta(.|k))
       with A = (q - mean_a q) / sigma clipped to +-3, sigma an EMA of the occupancy-weighted RMS advantage over all
       of the player's decisions (so the selection does not change the step size)
    5. every M updates the teacher T <- theta

Policy classes (FeaturePolicy): logit(k, a) = sum over the features f of decision k of E[f, label(a)], a sparse
linear map z = Phi e of one parameter vector e. Three feature maps:
    tabular   one feature per decision: no parameter is shared, nothing spills
    hashed    two hashed feature tables with fewer buckets than decisions: random sharing
    tokens    the tokens of the information set (private card, public card, every action token of the history,
              a bias): the same token at different depths is the same parameter, so a step at one decision moves
              every decision that shares a token with it, related or not
"""
import hashlib

import numpy as np
import scipy.sparse as sp


def _bucket(table, key, buckets):
    h = hashlib.md5(('%d|%r' % (table, key)).encode()).digest()
    return int.from_bytes(h[:8], 'little') % buckets


def poker_tokens(key):
    """Tokens of a Kuhn key (card, history) or a Leduc key (private, public, (round-1 history, round-2 history))."""
    if len(key) == 2:                                          # Kuhn
        card, hist = key
        return [('bias',), ('card', card)] + [('act', a) for a in hist]
    priv, pub, (h1, h2) = key                                  # Leduc
    toks = [('bias',), ('priv', priv)] + [('act', a) for a in h1] + [('act', a) for a in h2]
    if pub is not None:
        toks += [('pub', pub), ('pair', pub == priv)]
    return toks


def features(pl, kind, buckets=None, tokenizer=None):
    """-> list over decisions of feature ids, and the number of features."""
    if kind == 'tabular':
        return [[k] for k in range(pl.K)], pl.K
    if kind == 'hashed':
        return [[t * buckets + _bucket(t, pl.key_list[k], buckets) for t in range(2)] for k in range(pl.K)], 2 * buckets
    if kind == 'tokens':
        vocab = {}
        out = []
        for k in range(pl.K):
            out.append(sorted({vocab.setdefault(tok, len(vocab)) for tok in tokenizer(pl.key_list[k])}))
        return out, len(vocab)
    raise ValueError(kind)


class FeaturePolicy:
    """Softmax over one player's sequences; logits z = Phi e (Phi sparse [n_seq, n_feat * n_labels])."""

    def __init__(self, pl, kind='tabular', buckets=None, tokenizer=None):
        self.pl = pl
        feats, nf = features(pl, kind, buckets, tokenizer)
        labels = sorted({a for acts in pl.key_actions for a in acts}, key=str)
        lab = {a: i for i, a in enumerate(labels)}
        nl = len(labels)
        rows, cols = [], []
        for k in range(pl.K):
            for j, a in enumerate(pl.key_actions[k]):
                s = pl.start[k] + j
                for f in feats[k]:
                    rows.append(s)
                    cols.append(f * nl + lab[a])
        self.Phi = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(pl.n, nf * nl))
        self.e = np.zeros(nf * nl)
        self.kind = kind

    def logits(self, e=None):
        z = self.Phi @ (self.e if e is None else e)
        z[0] = 0.0
        return z

    def pi(self, e=None):
        return self.pl.softmax(self.logits(e))

    def ascend(self, g, lr):
        """e += lr * Phi' g for a per-sequence logit gradient g: the chain rule through the shared features."""
        gg = g.copy()
        gg[0] = 0.0
        self.e += lr * (self.Phi.T @ gg)

    def coupling(self, rows_a, rows_b):
        """Logit-space kernel K = Phi_a Phi_b' between two sets of sequences (the spill a unit logit step on b sends
        to a)."""
        return (self.Phi[rows_a] @ self.Phi[rows_b].T).toarray()


def oracle_fields(game, pis):
    """Per player: dict(Qpi [n], Qbr [n], V [K] normalised, d [K], occ [K], gap = best-response gain)."""
    xs = game.realise(pis)
    out = []
    for i, ((Wq, Q, V, R), pl, g) in enumerate(zip(game.q_values(pis, xs), game.plex, game.grads(xs))):
        Wbr, _, br_total = pl.backup(g)
        Rs = np.concatenate([[1.0], R[pl.key_of_seq[1:]]])
        Qbr = np.where(Rs > 1e-300, Wbr / np.maximum(Rs, 1e-300), 0.0)
        Vn = np.where(R > 1e-300, V / np.maximum(R, 1e-300), 0.0)
        mx = np.maximum.reduceat(Qbr[1:], pl.start[:-1] - 1)
        occ = pl.reach_in(xs[i]) * R
        own = float(g @ xs[i])
        out.append(dict(Qpi=Q, Qbr=Qbr, V=Vn, d=np.maximum(mx - Vn, 0.0), occ=occ, gap=float(br_total) - own))
    return out, xs


def kl_per_key(pl, p, q):
    """KL(p(.|k) || q(.|k)) per decision."""
    with np.errstate(divide='ignore', invalid='ignore'):
        t = np.where(p[1:] > 0, p[1:] * (np.log(p[1:]) - np.log(np.maximum(q[1:], 1e-300))), 0.0)
    return np.add.reduceat(t, pl.start[:-1] - 1)


class FilterLock:
    """The update on a polar.plex.Game. arm: 'onpolicy' (every candidate a pivot, no lock), 'filter' (filter, no
    lock), 'filter_lock' (filter + KL lock to the segment's teacher)."""

    def __init__(self, game, arm='filter_lock', kind='tabular', buckets=None, tokenizer=None, n_u=2, n_f=4, K=5,
                 f=0.25, beta=2.0, eps=0.05, delta=0.02, M=100, lr=0.5, filtered=None, swap=True, seed=0):
        self.game, self.arm = game, arm
        self.pols = [FeaturePolicy(pl, kind, buckets, tokenizer) for pl in game.plex]
        self.n_u, self.n_f, self.K, self.f, self.beta = n_u, n_f, K, f, beta
        self.eps, self.delta, self.M, self.lr, self.swap = eps, delta, M, lr, swap
        self.rng = np.random.default_rng(seed)
        self.filtered = [np.array([bool(filtered(pl.key_list[k])) if filtered else True for k in range(pl.K)])
                         for pl in game.plex]
        self.sigma = [None] * len(self.pols)
        self.t = 0
        self.xbar, self.wsum = None, 0.0
        self.segments = []                          # per finished segment and player: drift record
        self._start_segment()

    # ---------------------------------------------------------------- segments
    def _start_segment(self):
        self.teacher = [p.e.copy() for p in self.pols]
        pis = self.policies()
        flds, _ = oracle_fields(self.game, pis)
        self.seg0 = []
        for i, (pl, fld) in enumerate(zip(self.game.plex, flds)):
            F = self.filtered[i]
            occ = np.where(F, fld['occ'], 0.0)
            # the agreeing region at the segment's start: filtered decisions at or below the occupancy-weighted
            # (1 - f) quantile of the disagreement (the population version of the lock set)
            o = np.argsort(fld['d'])
            cum = np.cumsum(occ[o]) / max(occ.sum(), 1e-300)
            bar = fld['d'][o][min(np.searchsorted(cum, 1 - self.f), len(o) - 1)]
            agree = F & (fld['d'] <= bar) & (occ > 0)
            self.seg0.append(dict(pi=pis[i], agree=agree, w=occ * agree / max((occ * agree).sum(), 1e-300),
                                  gap=fld['gap'], bar=float(bar)))

    def _close_segment(self):
        pis = self.policies()
        for i, pl in enumerate(self.game.plex):
            s0 = self.seg0[i]
            kl = kl_per_key(pl, s0['pi'], pis[i])
            tv = 0.5 * np.add.reduceat(np.abs(s0['pi'][1:] - pis[i][1:]), pl.start[:-1] - 1)
            self.segments.append(dict(t=self.t, player=i, kl=float((s0['w'] * kl).sum()),
                                      tv=float((s0['w'] * tv).sum()), gap=s0['gap'], bar=s0['bar']))

    # ---------------------------------------------------------------- update
    def policies(self):
        return [p.pi() for p in self.pols]

    def _sample(self, occ, mask, n):
        w = np.where(mask, occ, 0.0)
        if n == 0 or w.sum() <= 0:
            return np.zeros(0, np.int64)
        return self.rng.choice(len(w), size=n, p=w / w.sum())

    def select(self, i, fld):
        """-> (pivots, lock set, filtered candidates, bar) for player i."""
        occ, d, F = fld['occ'], fld['d'], self.filtered[i]
        cu = self._sample(occ, ~F, self.n_u)
        if self.arm == 'onpolicy':
            cf = self._sample(occ, F, self.n_f)
            return np.concatenate([cu, cf]), np.zeros(0, np.int64), cf, np.nan
        cf = self._sample(occ, F, self.K * self.n_f)
        if len(cf) == 0:
            return cu, np.zeros(0, np.int64), cf, np.nan
        sc = d[cf]
        b = float(np.quantile(sc, 1.0 - self.f))
        chosen = [j for j in range(len(cf)) if sc[j] > b][:self.n_f]
        if len(chosen) < self.n_f:
            taken = set(chosen)
            chosen += [j for j in np.argsort(-sc, kind='stable') if j not in taken][:self.n_f - len(chosen)]
        taken = set(chosen)
        piv = np.concatenate([cu, cf[chosen]])
        lock = cf[[j for j in range(len(cf)) if j not in taken and sc[j] <= b]]
        if self.arm == 'filter':
            lock = np.zeros(0, np.int64)
        return piv, lock, cf, b

    def step(self):
        pis = self.policies()
        flds, xs = oracle_fields(self.game, pis)
        self._average(xs)
        info = []
        for i, (pol, pl, fld) in enumerate(zip(self.pols, self.game.plex, flds)):
            piv, lock, cand, b = self.select(i, fld)
            pi = pis[i]
            g = np.zeros(pl.n)
            advs = []
            for k in piv:
                s0, s1 = pl.start[k], pl.start[k + 1]
                q = fld['Qpi'][s0:s1].copy()
                if self.swap:
                    qbr = fld['Qbr'][s0:s1]
                    if np.any((pi[s0:s1] < self.eps) & (qbr >= qbr.max() - self.delta)):
                        q = qbr.copy()
                advs.append(q - q.mean())
            if advs:
                Q = fld['Qpi']
                mean_k = np.add.reduceat(Q[1:], pl.start[:-1] - 1) / pl.nact
                A2 = np.add.reduceat((Q[1:] - mean_k[pl.key_of_seq[1:]]) ** 2, pl.start[:-1] - 1) / pl.nact
                w = fld['occ']
                rms = np.sqrt((w * A2).sum() / max(w.sum(), 1e-300)) + 1e-12
                self.sigma[i] = rms if self.sigma[i] is None else 0.9 * self.sigma[i] + 0.1 * rms
                for k, a in zip(piv, advs):
                    s0, s1 = pl.start[k], pl.start[k + 1]
                    g[s0:s1] += np.clip(a / self.sigma[i], -3, 3) / (s1 - s0) / len(piv)
            if len(lock):                    # -beta KL(pi_T || pi_theta): logit gradient beta (pi_T - pi_theta)
                piT = pol.pi(self.teacher[i])
                for k in lock:
                    s0, s1 = pl.start[k], pl.start[k + 1]
                    g[s0:s1] += self.beta * (piT[s0:s1] - pi[s0:s1]) / len(lock)
            pol.ascend(g, self.lr)
            info.append(dict(piv=piv, lock=lock, cand=cand, bar=b))
        self.t += 1
        if self.t % self.M == 0:
            self._close_segment()
            self._start_segment()
        return pis, flds, info

    def _average(self, xs):
        w = float(self.t + 1)                           # linear averaging of realisation plans
        if self.xbar is None:
            self.xbar, self.wsum = [x.copy() for x in xs], w
            return
        self.wsum += w
        self.xbar = [xb + (w / self.wsum) * (x - xb) for xb, x in zip(self.xbar, xs)]

    def nashconv(self):
        return self.game.nashconv(self.game.realise(self.policies()))

    def nashconv_avg(self):
        return self.game.nashconv(self.xbar) if self.xbar is not None else np.nan


def decompose(game, i, pis_before, pis_after, flds, groups):
    """Exact performance difference for player i with the others held at pis_before (one player: the exact change
    of V): V_i(pi') - V_i(pi) = sum_k occ'(k) sum_a (pi'(a|k) - pi(a|k)) Q^pi(k, a), occ' the occupancy under
    (pi'_i, pi_-i). groups: name -> boolean mask over player i's keys. -> dict(total, check, <name>, tv_<name>)."""
    pl = game.plex[i]
    joint_after = [pis_after[j] if j == i else pis_before[j] for j in range(len(pis_before))]
    xs_after = game.realise(joint_after)
    xs_before = game.realise(pis_before)
    R = game.cf_reach(xs_before, i)
    occ_after = pl.reach_in(xs_after[i]) * R
    dpi = pis_after[i] - pis_before[i]
    per_key = np.add.reduceat((dpi * flds[i]['Qpi'])[1:], pl.start[:-1] - 1) * occ_after
    tv = 0.5 * np.add.reduceat(np.abs(dpi[1:]), pl.start[:-1] - 1) * flds[i]['occ']
    if game.players == 2:
        sign = 1.0 if i == 0 else -1.0
        v0, v1 = sign * game.value(xs_before), sign * game.value(xs_after)
    else:
        v0, v1 = game.value(xs_before), game.value(xs_after)
    out = dict(total=v1 - v0, check=float(per_key.sum()))
    for name, m in groups.items():
        out[name] = float(per_key[m].sum())
        out['tv_' + name] = float(tv[m].sum())
    return out
