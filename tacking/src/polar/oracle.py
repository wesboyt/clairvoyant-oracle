"""The two oracle fields, built with ONE definition for every game (theory.md S2):

(a) clairvoyant / hindsight field ('north', max flow): reveal the deal d to player i (all chance: hidden cards,
    the board, the whole future tape; NOT the opponent's unobserved moves) and let i best-respond deal by deal to the
    opponent's pi*: Q*_{k,d}(a) = the value of a at key k when i plays on knowing d. Then average over the key's
    posterior: mf(k, a) = sum_d w(k, d) Q*_{k,d}(a) / sum_d w(k, d),  w(k, d) = p(d) x (the opponent's reach of
    (k, d) under pi*). One player: w = p(d), Q*_{k,d} = the per-tape max over the realised lattice. This is the trader's gate oracle (an average of per-future maxima;
    lob/posterior/gate.py:49-71) with K_mf = all candidates.
(b) exact information-respecting optimum ('south'): Q^{pi*}(k, a), the counterfactual value of a at k when both
    players then follow an exact optimal / Nash pi* (DP for one player, LP Nash for two).
(c) posterior variant of (a): K_mf candidate histories drawn from the key's posterior, the realised one excluded
    (lob/posterior/gate.py posterior_maxflow; scripts/edge_lib.py:409 with uniform weights).
"""
import numpy as np

from . import lp


def optimum(game):
    """Exact optimal / Nash behavioural policies pi* (list per player) and the game value."""
    if game.players == 1:
        v, pi, _, _ = lp.single_agent_optimum(game)
        return [pi], v
    x, y, v, _ = lp.nash(game)
    pis = []
    for i, xi in enumerate((x, y)):
        pl = game.plex[i]
        xs = [x, y]
        W, _, _ = pl.backup(game.grads(xs)[i])
        br = pl.br_policy(W)                        # completion at keys pi* does not reach
        pis.append(pl.behavioural(xi, pi_default=br))
    return pis, v


def exact_field(game, pis):
    """Q^{pi*}(k, a) per player (normalised by the key's chance x opponent reach), plus V(k)."""
    out = []
    for W, Q, V, R in game.q_values(pis):
        out.append(Q)
    return out


def clairvoyant_field(game, pis, i=0, keep_members=False):
    """mf(k, a) for player i (see module doc): the deal (all chance: cards, board, the future tape) is revealed to
    player i, the opponent's unobserved moves are not; i best-responds deal by deal to the opponent's pi*.
    -> (mf over i's sequences, E[V*] at the root, members)
    members[k] = list of (weight, Q*_{k,d} vector) per deal d consistent with k (for the posterior variant)."""
    import scipy.sparse as sp
    pl = game.plex[i]
    xs = game.realise(pis)
    num = np.zeros(pl.n)
    den = np.zeros(pl.K)
    members = [[] for _ in range(pl.K)] if keep_members else None
    ev = 0.0
    if game.players == 1:
        for p, nodes in zip(game.deal_probs, game.deal_graphs):
            memo = {}

            def val(nid):
                if nid in memo:
                    return memo[nid]
                _, k, _, _, ch = nodes[nid]
                vals = np.array([(r + tv) + (val(cid)[0] if cid >= 0 else 0.0) for a, sq, cid, r, tv in ch])
                memo[nid] = (vals.max(), vals)
                return memo[nid]

            ev += p * val(0)[0]
            for nid, (_, k, _, _, ch) in enumerate(nodes):
                vals = val(nid)[1]
                s0 = pl.start[k]
                num[s0:s0 + len(vals)] += p * vals
                den[k] += p
                if keep_members:
                    members[k].append((p, vals))
    else:
        n0, n1 = game.plex[0].n, game.plex[1].n
        opp = 1 - i
        for p, nodes, ent in zip(game.deal_probs, game.deal_graphs, game.deal_A):
            ij = np.array([(a, b) for a, b, _ in ent], np.int64)
            Ad = sp.csr_matrix((np.array([u for _, _, u in ent]), (ij[:, 0], ij[:, 1])), shape=(n0, n1))
            gd = Ad @ xs[1] if i == 0 else -(Ad.T @ xs[0])
            W, V, tot = pl.backup(np.asarray(gd).ravel())
            ev += tot
            Rd = np.zeros(pl.K)
            for (pl_i, k, _, sq_opp, ch) in nodes:
                if pl_i == i:
                    Rd[k] += p * xs[opp][sq_opp]
            for k in np.flatnonzero(Rd > 0):
                s0, s1 = pl.start[k], pl.start[k + 1]
                num[s0:s1] += W[s0:s1]
                den[k] += Rd[k]
                if keep_members:
                    members[k].append((Rd[k], W[s0:s1] / Rd[k]))
    dens = den[pl.key_of_seq[1:]]
    mf = np.zeros(pl.n)
    mf[1:] = np.where(dens > 0, num[1:] / np.where(dens > 0, dens, 1.0), np.nan)
    return mf, ev, members


def posterior_field(game, i, members, K=8, rng=None, exclude=False):
    """Sampled clairvoyant field: per key, K candidate histories drawn iid from the key's posterior w (what the toy
    trainer does: lob/toy/core.py ToyWorld.candidates). exclude=True: first draw a 'realised' member h0 ~ w and
    draw the K candidates among the OTHER members (the twin / real-tape rule 'labelled tape excluded'; harmless
    over a continuum of futures, a bias over a finite posterior)."""
    rng = rng or np.random.default_rng(0)
    pl = game.plex[i]
    mfk = np.full(pl.n, np.nan)
    for k, mem in enumerate(members):
        if not mem:
            continue
        w = np.array([m[0] for m in mem])
        if w.sum() <= 0:
            continue
        w = w / w.sum()
        if exclude and len(mem) > 1:
            h0 = rng.choice(len(mem), p=w)
            w2 = w.copy(); w2[h0] = 0.0
            if w2.sum() > 0:
                w = w2 / w2.sum()
        pick = rng.choice(len(mem), size=K, p=w)
        vals = np.stack([mem[j][1] for j in pick]).mean(0)
        s0 = pl.start[k]
        mfk[s0:s0 + len(vals)] = vals
    return mfk


def greedy(pl, field, tol=1e-9):
    """Deterministic greedy policy of a field (first maximiser; nan treated as -inf)."""
    f = np.nan_to_num(field, nan=-np.inf)
    return pl.br_policy(f)


def field_stats(game, i, pis, Qs, mf, occ_policy=None):
    """Comparisons of the clairvoyant field mf with the exact field Q^{pi*} for player i, weighted by the
    occupancy of pi* (chance x both reaches)."""
    pl = game.plex[i]
    xs = game.realise(pis)
    occ = pl.reach_in(xs[i]) * game.cf_reach(xs, i)
    occ = occ / occ.sum()
    tie_exact = 0.0
    tie_mf = 0.0
    agree = 0.0
    prem = 0.0
    spread_ratio = []
    Q = Qs[i]
    for k in range(pl.K):
        s0, s1 = pl.start[k], pl.start[k + 1]
        q, m = Q[s0:s1], mf[s0:s1]
        if s1 - s0 < 2 or not np.isfinite(m).all():
            continue
        tq = (q >= q.max() - 1e-9).sum() > 1
        tm = (m >= m.max() - 1e-9).sum() > 1
        tie_exact += occ[k] * tq
        tie_mf += occ[k] * tm
        agree += occ[k] * (m[np.argmax(q)] >= m.max() - 1e-9)
        prem += occ[k] * (m.max() - q.max())
        if q.std() > 1e-12:
            spread_ratio.append((occ[k], m.std() / q.std()))
    return dict(tie_share_exact=float(tie_exact), tie_share_mf=float(tie_mf), argmax_agree=float(agree),
                clairvoyance_premium=float(prem),
                spread_ratio_wmean=(float(sum(w * r for w, r in spread_ratio) / max(sum(w for w, _ in spread_ratio),
                                                                                    1e-300)) if spread_ratio else None),
                share_keys_fields_differ=float(sum(w for w, r in spread_ratio if abs(r - 1) > 1e-6)))
