"""Exact game structure: a finite game with all chance at the root, compiled into sequence form.

A *process* describes the game (see games/*.py):
    players          1 (a single-agent decision problem, e.g. the trading ladder) or 2 (zero-sum)
    merge            True: identical (deal, state) nodes are merged (a DAG). Only for single-agent games whose
                     infoset key is (observation history, own in-flight state); see theory.md S1.2
    deals()          [(prob, deal)]  -- every chance outcome, including hidden cards and the whole future tape
    root(d)          the initial state
    terminal(d, s)   (done, value to player 0)
    turn(d, s)       the player to move
    actions(d, s)    legal action labels
    step(d, s, a)    (next state, reward to player 0)
    key(d, s, i)     player i's information-set key at s (everything player i has observed, plus whatever makes the
                     game perfect-recall: own action history in trees, own in-flight state in merged DAGs)

The compiled `Game` holds, per player i, a *plex*: sequences (index 0 = the empty sequence), information sets
(keys), each key's contiguous block of action sequences and its parent sequences. The realisation plan x_i is the
own-reach vector of the sequences (x_i[0] = 1); the polytope X_i = {x >= 0 : E_i x = e_i} (flow conservation). Player
0's payoff is x_0^T A x_1 (two players) or c^T x_0 (one player). Everything below is exact and vectorised by levels.
"""
from collections import defaultdict

import numpy as np
import scipy.sparse as sp


class Plex:
    """One player's sequence-form structure."""

    def __init__(self):
        self.keys = {}            # key -> id
        self.key_list = []
        self.key_actions = []     # id -> tuple of action labels
        self.key_start = []       # id -> first sequence index
        self.parents = []         # id -> set of parent sequence indices
        self.n = 1                # sequences, 0 = empty

    def add(self, key, acts):
        k = self.keys.get(key)
        if k is None:
            k = len(self.key_list)
            self.keys[key] = k
            self.key_list.append(key)
            self.key_actions.append(tuple(acts))
            self.key_start.append(self.n)
            self.parents.append(set())
            self.n += len(acts)
        elif self.key_actions[k] != tuple(acts):
            raise ValueError('inconsistent actions at key %r' % (key,))
        return k

    def seq(self, k, a):
        return self.key_start[k] + self.key_actions[k].index(a)

    def finalize(self):
        K, n = len(self.key_list), self.n
        self.K = K
        self.start = np.array(self.key_start + [n], np.int64)
        self.nact = np.diff(self.start)
        self.key_of_seq = np.full(n, -1, np.int64)
        for k in range(K):
            self.key_of_seq[self.start[k]:self.start[k + 1]] = k
        # topological levels: depth = longest chain of keys from a root
        depth = np.zeros(K, np.int64)
        kids = defaultdict(list)
        indeg = np.zeros(K, np.int64)
        for k in range(K):
            for s in self.parents[k]:
                if s > 0:
                    kids[self.key_of_seq[s]].append(k)
                    indeg[k] += 1
        order, frontier = [], [k for k in range(K) if indeg[k] == 0]
        while frontier:
            nxt = []
            for k in frontier:
                order.append(k)
                for c in kids[k]:
                    depth[c] = max(depth[c], depth[k] + 1)
                    indeg[c] -= 1
                    if indeg[c] == 0:
                        nxt.append(c)
            frontier = nxt
        assert len(order) == K, 'cycle in the key graph'
        self.depth = depth
        self.levels = []
        for L in range(int(depth.max()) + 1 if K else 0):
            ks = np.flatnonzero(depth == L)
            seqs = np.concatenate([np.arange(self.start[k], self.start[k + 1]) for k in ks])
            loc = np.repeat(np.arange(len(ks)), self.nact[ks])
            rows, cols = [], []
            for i, k in enumerate(ks):
                for s in self.parents[k]:
                    rows.append(i)
                    cols.append(s)
            Pm = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(ks), n))
            starts = np.concatenate([[0], np.cumsum(self.nact[ks])[:-1]])
            self.levels.append((ks, seqs, loc, Pm, starts))
        # flow constraints E x = e: row 0: x[0] = 1; row 1 + k: sum_a x(k, a) - sum_parents x = 0
        rows, cols, vals = [0], [0], [1.0]
        for k in range(K):
            for s in range(self.start[k], self.start[k + 1]):
                rows.append(1 + k); cols.append(s); vals.append(1.0)
            for s in self.parents[k]:
                rows.append(1 + k); cols.append(s); vals.append(-1.0)
        self.E = sp.csr_matrix((vals, (rows, cols)), shape=(K + 1, n))
        self.e = np.zeros(K + 1); self.e[0] = 1.0
        self.roots = np.array([k for k in range(K) if self.parents[k] == {0}], np.int64)

    # ---------------------------------------------------------------- behavioural <-> sequence form
    def uniform(self):
        pi = np.ones(self.n)
        pi[0] = 1.0
        for k in range(self.K):
            pi[self.start[k]:self.start[k + 1]] = 1.0 / self.nact[k]
        return pi

    def softmax(self, z):
        """Per-key softmax of a logit vector over sequences (entry 0 ignored)."""
        out = np.ones_like(z)
        if self.K == 0:
            return out
        zz = z[1:]
        st = self.start[:-1] - 1
        mx = np.maximum.reduceat(zz, st)
        ez = np.exp(zz - mx[self.key_of_seq[1:]])
        sm = np.add.reduceat(ez, st)
        out[1:] = ez / sm[self.key_of_seq[1:]]
        return out

    def seg_sum(self, v):
        """Per-key sum of a sequence vector."""
        return np.add.reduceat(v[1:], self.start[:-1] - 1)

    def realise(self, pi):
        """Behavioural pi (per sequence: pi(a|k)) -> realisation plan x (own reach)."""
        x = np.zeros(self.n)
        x[0] = 1.0
        for ks, seqs, loc, Pm, _ in self.levels:
            xin = Pm @ x
            x[seqs] = xin[loc] * pi[seqs]
        return x

    def behavioural(self, x, pi_default=None):
        """Realisation plan -> behavioural policy (uniform / pi_default where the key is unreached)."""
        xin = np.zeros(self.K)
        for ks, seqs, loc, Pm, _ in self.levels:
            xin[ks] = Pm @ x
        pi = self.uniform() if pi_default is None else pi_default.copy()
        tot = self.seg_sum(x)
        ok = tot > 1e-300
        kk = self.key_of_seq[1:]
        pi[1:] = np.where(ok[kk], x[1:] / np.where(ok[kk], tot[kk], 1.0), pi[1:])
        return pi

    def reach_in(self, x):
        """Own reach of every key: x_in(k) = sum of x over parent sequences."""
        xin = np.zeros(self.K)
        for ks, seqs, loc, Pm, _ in self.levels:
            xin[ks] = Pm @ x
        return xin

    def backup(self, g, pi=None):
        """Counterfactual backup. g = the payoff gradient over sequences (chance x opponent weighted).
        pi given: W(sigma) = g(sigma) + sum_{child keys} V(k), V(k) = sum_a pi W(k, a)  (expected).
        pi None:  V(k) = max_a W(k, a)  (best response).  -> (W [n], V [K], total = W[0])"""
        W = g.astype(float).copy()
        V = np.zeros(self.K)
        for ks, seqs, loc, Pm, starts in reversed(self.levels):
            if pi is None:
                Vl = np.maximum.reduceat(W[seqs], starts)
            else:
                Vl = np.bincount(loc, weights=pi[seqs] * W[seqs], minlength=len(ks))
            V[ks] = Vl
            W += Pm.T @ Vl
        return W, V, W[0]

    def br_policy(self, W, tol=1e-12):
        """Deterministic best response from a backup W: argmax per key (first maximiser)."""
        pi = np.zeros(self.n)
        pi[0] = 1.0
        for k in range(self.K):
            s0, s1 = self.start[k], self.start[k + 1]
            pi[s0 + int(np.argmax(W[s0:s1]))] = 1.0
        return pi


class Game:
    def __init__(self, proc, name=None):
        self.proc = proc
        self.name = name or getattr(proc, 'name', type(proc).__name__)
        self.players = proc.players
        self.plex = [Plex() for _ in range(self.players)]
        self._build()

    # ---------------------------------------------------------------- compile
    def _build(self):
        P = self.proc
        two = self.players == 2
        merge = getattr(P, 'merge', False)
        assert not (two and merge)
        A_ent = defaultdict(float)          # (s0, s1) -> sum p * u           (two players)
        c = defaultdict(float)              # s0 -> sum p * (r + terminal)     (one player)
        Bent = [defaultdict(float), defaultdict(float)]     # (key, opp seq) -> sum p
        self.deal_graphs = []               # per deal: nodes for the clairvoyant oracle
        self.deal_probs = []
        self.deal_A = []                    # per deal: [(s0, s1, p * u)] (two players)
        for p, d in P.deals():
            if p <= 0:
                continue
            nodes = []                      # (player, key, own seq in, opp seq, [(a, seq, child, r, tv)])
            A_deal = []
            memo = {}

            def visit(s, seqs, acc):
                if merge and s in memo:
                    nid, i, k = memo[s]
                    self.plex[i].parents[k].add(seqs[i])
                    return nid
                i = P.turn(d, s)
                acts = P.actions(d, s)
                key = P.key(d, s, i)
                pl = self.plex[i]
                k = pl.add(key, acts)
                pl.parents[k].add(seqs[i])
                if two:
                    Bent[i][(k, seqs[1 - i])] += p
                else:
                    Bent[0][(k, 0)] += p
                nid = len(nodes)
                nodes.append(None)
                if merge:
                    memo[s] = (nid, i, k)
                ch = []
                for a in acts:
                    sq = pl.seq(k, a)
                    s2, r = P.step(d, s, a)
                    done, tv = P.terminal(d, s2)
                    nseqs = list(seqs)
                    nseqs[i] = sq
                    if done:
                        if two:
                            A_ent[(nseqs[0], nseqs[1])] += p * (acc + r + tv)
                            A_deal.append((nseqs[0], nseqs[1], p * (acc + r + tv)))
                        else:
                            c[sq] += p * (r + tv)
                        ch.append((a, sq, -1, r, tv))
                    else:
                        if not two:
                            c[sq] += p * r
                        cid = visit(s2, nseqs, acc + r)
                        ch.append((a, sq, cid, r, 0.0))
                nodes[nid] = (i, k, seqs[i], seqs[1 - i] if two else 0, ch)
                return nid

            s0 = P.root(d)
            done, tv = P.terminal(d, s0)
            assert not done
            visit(s0, [0, 0], 0.0)
            self.deal_graphs.append(nodes)
            self.deal_A.append(A_deal)
            self.deal_probs.append(p)
        for pl in self.plex:
            pl.finalize()
        n0 = self.plex[0].n
        if two:
            n1 = self.plex[1].n
            ij = np.array(list(A_ent.keys()), np.int64).reshape(-1, 2)
            self.A = sp.csr_matrix((np.array(list(A_ent.values())), (ij[:, 0], ij[:, 1])), shape=(n0, n1))
            self.c = None
        else:
            self.A = None
            self.c = np.zeros(n0)
            for s, v in c.items():
                self.c[s] = v
        self.B = []
        for i in range(self.players):
            pl = self.plex[i]
            no = self.plex[1 - i].n if two else 1
            kk = np.array([k for k, _ in Bent[i]], np.int64)
            ss = np.array([s for _, s in Bent[i]], np.int64)
            self.B.append(sp.csr_matrix((np.array(list(Bent[i].values())), (kk, ss)), shape=(pl.K, no)))
        self.deal_probs = np.array(self.deal_probs)

    # ---------------------------------------------------------------- values
    def uniform(self):
        return [pl.uniform() for pl in self.plex]

    def realise(self, pis):
        return [pl.realise(pi) for pl, pi in zip(self.plex, pis)]

    def grads(self, xs):
        """Payoff gradients over each player's sequences (each player maximises its own payoff)."""
        if self.players == 1:
            return [self.c]
        return [self.A @ xs[1], -(self.A.T @ xs[0])]

    def value(self, xs):
        """Player 0's expected payoff."""
        if self.players == 1:
            return float(self.c @ xs[0])
        return float(xs[0] @ (self.A @ xs[1]))

    def cf_reach(self, xs, i):
        """R_i(k) = chance x opponent reach of player i's key k."""
        if self.players == 1:
            return np.asarray(self.B[0] @ np.ones(1)).ravel()
        return np.asarray(self.B[i] @ xs[1 - i]).ravel()

    def q_values(self, pis, xs=None):
        """Per player: (W unnormalised cf values, Q = W / R normalised, V per key, R per key)."""
        xs = xs if xs is not None else self.realise(pis)
        out = []
        for i, (pl, g) in enumerate(zip(self.plex, self.grads(xs))):
            W, V, _ = pl.backup(g, pis[i])
            R = self.cf_reach(xs, i)
            Rs = np.concatenate([[1.0], R[pl.key_of_seq[1:]]])
            Q = np.where(Rs > 1e-300, W / np.maximum(Rs, 1e-300), 0.0)
            out.append((W, Q, V, R))
        return out

    def br_values(self, xs):
        """Best-response value of each player against the others (in its own payoff units)."""
        return [pl.backup(g)[2] for pl, g in zip(self.plex, self.grads(xs))]

    def nashconv(self, xs):
        """Two players: sum of best-response gains (zero iff Nash). One player: V_opt - V(x)."""
        br = self.br_values(xs)
        if self.players == 1:
            return float(br[0] - self.c @ xs[0])
        return float(br[0] + br[1])
