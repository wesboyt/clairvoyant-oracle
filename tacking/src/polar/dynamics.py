"""Learning dynamics on exact (full-width) values. Every method updates each player's behavioural policy from the
exact counterfactual values W (unnormalised) or Q = W / R (normalised) of the current joint policy.

    gd       softmax policy gradient on logits (exact), simultaneous: theta += eta * x_in(k) pi(a|k) (W(k,a) - V(k))
    hedge    per-key Hedge / multiplicative weights on counterfactual values (simultaneous): theta += eta * W
    ohedge   optimistic Hedge: theta += eta * (2 W_t - W_{t-1})
    eg       extragradient Hedge: half step, then the step with the half-step values
    mmd      magnetic mirror descent (Sokota et al. 2023), behavioural closed form with normalised Q, magnet
             uniform (fixed: converges to a QRE) or refreshed every K steps (proximal point / R-NaD style)
    recipe   a tabular analogue of the trader repo's RL update: centred advantage / sigma, anchors (eta_reg to the
             magnet, eta_unif to uniform, entropy, forward KL to the magnet), Adam beta1 = 0, magnet refine every
             K updates, a Linear-CFR average reset on refine
    cfr+     regret matching+, alternating updates, linear averaging
    pgd/peg/ogd  Euclidean projected gradient / extragradient / optimistic gradient on the simplex (normal-form games
             only: one key per player) -- the geometry in which theory.md Prop. 1 is exact.
Every learner also keeps a linear (weight t) sequence-form average of its iterates.
"""
import numpy as np


def proj_simplex(v):
    u = np.sort(v)[::-1]
    css = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, len(v) + 1) > (css - 1))[0][-1]
    th = (css[rho] - 1) / (rho + 1.0)
    return np.maximum(v - th, 0)


class Learner:
    def __init__(self, game, method, eta=0.1, alpha=0.0, K=None, seed=0, init=None, **kw):
        self.g, self.method, self.eta, self.alpha, self.K = game, method, eta, alpha, K
        self.kw = kw
        self.t = 0
        pls = game.plex
        rng = np.random.default_rng(seed)
        if init is None:
            self.theta = [np.zeros(pl.n) for pl in pls]
        else:
            self.theta = [np.log(np.maximum(p, 1e-300)) for p in init]
        if kw.get('init_noise', 0) > 0:
            self.theta = [th + kw['init_noise'] * rng.standard_normal(th.shape) for th in self.theta]
        self.pis = [pl.softmax(th) for pl, th in zip(pls, self.theta)]
        self.magnet = [pl.uniform() for pl in pls]
        self.prevW = None
        self.reg = [np.zeros(pl.n) for pl in pls]                 # cfr+
        if method in ('pgd', 'peg', 'ogd'):
            assert all(pl.K == 1 for pl in pls), 'Euclidean methods: normal-form games only'
            self.xe = [p.copy() for p in self.pis]
        if method == 'recipe':
            self.adam_v = [np.zeros(pl.n) for pl in pls]
            self.sigma = None
            self.sigma_ref = None
        self.xs = game.realise(self.pis)
        self.avg_num = [np.zeros(pl.n) for pl in pls]
        self.avg_w = 0.0
        self.avg_t0 = 0
        self._accumulate()

    # ------------------------------------------------------------------ helpers
    def _accumulate(self):
        w = float(self.t - self.avg_t0 + 1)
        for i, x in enumerate(self.xs):
            self.avg_num[i] += w * x
        self.avg_w += w

    def avg(self):
        return [a / self.avg_w for a in self.avg_num]

    def _W(self, pis):
        xs = self.g.realise(pis)
        return [pl.backup(g, pi)[:2] for pl, g, pi in zip(self.g.plex, self.g.grads(xs), pis)], xs

    def _hedge_step(self, theta, pis, gain, eta):
        out = []
        for pl, th, gn in zip(self.g.plex, theta, gain):
            out.append(th + eta * gn)
        return out

    # ------------------------------------------------------------------ one update
    def step(self):
        g, m, eta = self.g, self.method, self.eta
        pls = g.plex
        if m in ('pgd', 'peg', 'ogd'):
            self._euclid_step()
        elif m == 'cfr+':
            for i in range(g.players):
                WV, xs = self._W(self.pis)
                W, V = WV[i]
                pl = pls[i]
                adv = W - np.concatenate([[0.0], V[pl.key_of_seq[1:]]])
                self.reg[i] = np.maximum(self.reg[i] + adv, 0.0)
                self.reg[i][0] = 0.0
                tot = pl.seg_sum(self.reg[i])
                pi = pl.uniform()
                ok = tot[pl.key_of_seq[1:]] > 0
                pi[1:] = np.where(ok, self.reg[i][1:] / np.where(ok, tot[pl.key_of_seq[1:]], 1.0), pi[1:])
                self.pis[i] = pi
        elif m in ('hedge', 'ohedge', 'eg', 'gd'):
            WV, xs = self._W(self.pis)
            Ws = [w for w, _ in WV]
            if m == 'gd':
                gains = []
                for pl, (W, V), x, pi in zip(pls, WV, xs, self.pis):
                    xin = pl.reach_in(x)
                    gn = np.zeros(pl.n)
                    gn[1:] = xin[pl.key_of_seq[1:]] * pi[1:] * (W[1:] - V[pl.key_of_seq[1:]])
                    gains.append(gn)
                self.theta = self._hedge_step(self.theta, self.pis, gains, eta)
            elif m == 'hedge':
                self.theta = self._hedge_step(self.theta, self.pis, Ws, eta)
            elif m == 'ohedge':
                prev = self.prevW if self.prevW is not None else Ws
                self.theta = self._hedge_step(self.theta, self.pis, [2 * a - b for a, b in zip(Ws, prev)], eta)
                self.prevW = Ws
            else:  # eg
                th_half = self._hedge_step(self.theta, self.pis, Ws, eta)
                pis_half = [pl.softmax(th) for pl, th in zip(pls, th_half)]
                WVh, _ = self._W(pis_half)
                self.theta = self._hedge_step(self.theta, self.pis, [w for w, _ in WVh], eta)
            self.theta = [th - np.concatenate([[0.0], np.maximum.reduceat(th[1:], pl.start[:-1] - 1)
                                               [pl.key_of_seq[1:]]]) for pl, th in zip(pls, self.theta)]
            self.pis = [pl.softmax(th) for pl, th in zip(pls, self.theta)]
        elif m == 'mmd':
            qv = g.q_values(self.pis)
            al = self.alpha
            new = []
            for pl, pi, mag, (W, Q, V, R) in zip(pls, self.pis, self.magnet, qv):
                z = (np.log(np.maximum(pi, 1e-300)) + eta * al * np.log(np.maximum(mag, 1e-300)) + eta * Q) / (
                    1 + eta * al)
                z = np.maximum(z, np.log(1e-12) + np.concatenate([[0], np.maximum.reduceat(z[1:], pl.start[:-1] - 1)
                                                                   [pl.key_of_seq[1:]]]))
                new.append(pl.softmax(z))
            self.pis = new
            if self.K and (self.t + 1) % self.K == 0:
                self.magnet = [p.copy() for p in self.pis]
        elif m == 'recipe':
            self._recipe_step()
        else:
            raise ValueError(m)
        self.t += 1
        self.xs = g.realise(self.pis)
        self._accumulate()
        return self.pis

    def _euclid_step(self):
        g, eta = self.g, self.eta

        def grads(xe):
            return [np.asarray(v).ravel() for v in g.grads(xe)]

        if self.method == 'pgd':
            gr = grads(self.xe)
            self.xe = [np.concatenate([[1.0], proj_simplex(x[1:] + eta * gg[1:])]) for x, gg in zip(self.xe, gr)]
        elif self.method == 'peg':
            gr = grads(self.xe)
            half = [np.concatenate([[1.0], proj_simplex(x[1:] + eta * gg[1:])]) for x, gg in zip(self.xe, gr)]
            gr2 = grads(half)
            self.xe = [np.concatenate([[1.0], proj_simplex(x[1:] + eta * gg[1:])]) for x, gg in zip(self.xe, gr2)]
        else:  # ogd (optimistic)
            gr = grads(self.xe)
            prev = self.prevW if self.prevW is not None else gr
            self.xe = [np.concatenate([[1.0], proj_simplex(x[1:] + eta * (2 * a[1:] - b[1:]))])
                       for x, a, b in zip(self.xe, gr, prev)]
            self.prevW = gr
        self.pis = [x.copy() for x in self.xe]

    def _recipe_step(self):
        """Tabular analogue of the trader's update (docs/ALGORITHM_SPEC.md S5)."""
        g, kw = self.g, self.kw
        pls = g.plex
        lr = kw.get('lr', 0.05)
        eta_reg, eta_unif, a_ent, b_fwd = kw.get('eta_reg', 0.2), kw.get('eta_unif', 0.05), \
            kw.get('alpha_ent', 0.04), kw.get('beta_fwd', 0.08)
        qv = g.q_values(self.pis)
        xs = self.xs
        advs, ws = [], []
        msq = 0.0
        for i, (pl, pi, (W, Q, V, R)) in enumerate(zip(pls, self.pis, qv)):
            occ = pl.reach_in(xs[i]) * R
            occ = occ / max(occ.sum(), 1e-300)
            mean = pl.seg_sum(Q) / pl.nact
            a = np.zeros(pl.n)
            a[1:] = Q[1:] - mean[pl.key_of_seq[1:]]
            advs.append(a)
            ws.append(occ)
            msq += float((occ * pl.seg_sum(a ** 2) / pl.nact).sum())
        msq /= len(pls)
        if self.sigma is None:
            self.sigma = max(np.sqrt(msq), 1e-6)
            self.sigma_ref = self.sigma
        else:
            self.sigma = max(np.sqrt(0.99 * self.sigma ** 2 + 0.01 * msq), 0.05 * self.sigma_ref)
        lam = self.sigma_ref / self.sigma
        for i, (pl, pi, mag) in enumerate(zip(pls, self.pis, self.magnet)):
            lp_ = np.log(np.maximum(pi, 1e-300))
            lu = np.concatenate([[0.0], -np.log(pl.nact[pl.key_of_seq[1:]].astype(float))])
            anch = np.clip(lam * (np.clip(eta_reg * (lp_ - np.log(np.maximum(mag, 1e-300))), -1, 1)
                                  + eta_unif * (lp_ - lu)), -1.5, 1.5)
            at = np.clip(advs[i] / self.sigma - anch, -3, 3)
            mean = pl.seg_sum(at) / pl.nact
            at[1:] -= mean[pl.key_of_seq[1:]]
            H = -pl.seg_sum(pi * lp_)
            dH = np.zeros(pl.n)
            dH[1:] = -pi[1:] * (lp_[1:] + H[pl.key_of_seq[1:]])
            grad = at / np.concatenate([[1.0], pl.nact[pl.key_of_seq[1:]]]) + a_ent * min(lam, 2) * dH \
                - b_fwd * (pi - mag)
            wk = np.concatenate([[0.0], ws[i][pl.key_of_seq[1:]]])
            grad = grad * wk
            grad[0] = 0.0
            v = self.adam_v[i] = 0.999 * self.adam_v[i] + 0.001 * grad ** 2
            vhat = v / (1 - 0.999 ** (self.t + 1))
            self.theta[i] = self.theta[i] + lr * grad / (np.sqrt(vhat) + 1e-8)
            self.pis[i] = pl.softmax(self.theta[i])
        K = self.K or 250
        if (self.t + 1) % K == 0:
            self.magnet = [p.copy() for p in self.pis]
            if kw.get('avg_reset', True):
                self.avg_num = [np.zeros(pl.n) for pl in pls]
                self.avg_w = 0.0
                self.avg_t0 = self.t + 1
