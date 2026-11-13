"""The trading-ladder toy games, re-implemented from their written specifications (toy order-
landing ladder L0-L2 and the round-trip rung R2 on T2 fill physics). Prices in cents on a $1 binary contract; we buy
one contract after an upward push of a leading market. Maker fee 0, taker fee ceil(7 P (1 - P)) cents.

L0  'bandit'  one decision; observed push g in {-1, 0, +1}; hidden follow J (settlement 50 + J) and the print depth
              m: post k (price 50 + k) fills iff m <= k; take pays the ask 52 plus the fee.
L1  'plan'    H = 4 frames; hidden card c (follow / none) and, if follow, the lift frame L; prints of volume v_t at
              the bid, observed one frame late; queue ahead DISP = 2 at the bid; a bid that lifts onto our resting
              price puts us at the head of the new bid (the pre-positioning trap).
L2  'chicken' two seats (us, a competing maker with private reservation r), 2 frames x 2 half-steps, both race for
              the inside rungs 51, 52 of a 50/53 book; zero-sum: u = [we fill](F - p) - [it fills](r - p').
R2  'round'   H = 5 frames, entry + exit legs, T2 physics: latency (orders land next frame), prints-only fills with
              FIFO queue, post-only, the cancel race, take-k IOC orders walking the landing book; an open position
              is force-sold into the bid at the horizon.

Single-agent games (L0, L1, R2) are merged DAGs: the information-set key is (everything observed so far, own
in-flight state); because the tape is exogenous, the posterior over tapes given a key does not depend on the policy,
so the flow polytope over (key, action) is the exact strategy space (theory.md S1.2).
"""
import itertools
import math

import numpy as np

BID0 = 50


def fee(p):
    return math.ceil(7 * p * (100 - p) / 1e4 - 1e-9)


def post(k):
    return 'post%+d' % k


class L0:
    name = 'L0'
    players = 1
    merge = True
    KS = (-2, -1, 0, 1)
    PG = {-1: 0.3, 0: 0.4, 1: 0.3}
    PJ = {1: {-1: .10, 0: .15, 2: .25, 4: .50}, 0: {-1: .30, 0: .30, 2: .25, 4: .15},
          -1: {-3: .50, 0: .32, 2: .12, 4: .06}}

    @staticmethod
    def pm(J):
        if J >= 2:
            return {2: .45, 1: .30, 0: .15, -1: .07, -2: .03}
        if J == 0:
            return {2: .25, 1: .25, 0: .25, -1: .15, -2: .10}
        return {2: .10, 1: .20, 0: .30, -1: .25, -2: .15}

    def deals(self):
        return [(pg * pj * pm, (g, J, m)) for g, pg in self.PG.items() for J, pj in self.PJ[g].items()
                for m, pm in self.pm(J).items()]

    def root(self, d):
        return (0, 'flat')

    def terminal(self, d, s):
        if s[1] == 'fill':
            return True, float(BID0 + d[1] - s[2])
        return (s[0] >= 1), 0.0

    def turn(self, d, s):
        return 0

    def actions(self, d, s):
        return ['wait'] + [post(k) for k in self.KS] + ['take']

    def step(self, d, s, a):
        if a == 'wait':
            return (1, 'flat'), 0.0
        if a == 'take':
            return (1, 'fill', BID0 + 2), -float(fee(BID0 + 2))
        k = int(a[4:])
        return ((1, 'fill', BID0 + k) if d[2] <= k else (1, 'flat')), 0.0

    def key(self, d, s, i):
        return ('g', d[0])

    def card(self, d):
        return d[1:]


class L1:
    name = 'L1'
    players = 1
    merge = True
    KS = (-1, 0, 1)
    DISP = 2

    def __init__(self, H=4):
        self.H = H
        self.prm = dict(pg2=0.5, pf={1: 0.35, 2: 0.70}, g1_hi=0.75, g1_lo=0.30, F_follow=54, F_none=49,
                        lift_spread=1, nv={0: .45, 1: .3025, 2: .165, 3: .0825}, lv={0: .10, 1: .30, 2: .30, 3: .30},
                        qv={0: .90, 1: .10})
        self.prm['pL'] = {L: 1.0 / (H - 1) for L in range(1, H)}

    def bids(self, d):
        if d['c'] == 0:
            return [BID0] * self.H
        return [BID0 + min(max(t - d['L'] + 1, 0), 2) for t in range(self.H)]

    def ask(self, d, t):
        b = d['bid'][t]
        return b + (self.prm['lift_spread'] if b > BID0 else 2)

    def deals(self):
        p = self.prm
        out = []
        for g0, pg in ((1, 1 - p['pg2']), (2, p['pg2'])):
            for c in (0, 1):
                pc = p['pf'][g0] if c else 1 - p['pf'][g0]
                for L, pl in (tuple(p['pL'].items()) if c else ((0, 1.0),)):
                    hi = p['g1_hi'] if c else p['g1_lo']
                    for g1, pg1 in ((1, hi), (0, 1 - hi)):
                        dists = [p['nv'] if c == 0 else (p['lv'] if t == L else p['qv']) for t in range(self.H)]
                        for vs in itertools.product(*[list(x.items()) for x in dists]):
                            pv = float(np.prod([x[1] for x in vs]))
                            d = dict(g0=g0, c=c, L=L, g1=g1, v=tuple(x[0] for x in vs))
                            d['bid'] = self.bids(d)
                            out.append((pg * pc * pl * pg1 * pv, _Frozen(d)))
        return out

    def root(self, d):
        return (0, 'flat')

    def terminal(self, d, s):
        if s[1] == 'fill':
            F = self.prm['F_follow'] if d['c'] else self.prm['F_none']
            return True, float(F - s[2])
        return s[0] >= self.H, 0.0

    def turn(self, d, s):
        return 0

    def actions(self, d, s):
        t = s[0]
        bid, ask = d['bid'][t], self.ask(d, t)
        acts = ['wait']
        if s[1] == 'rest':
            acts.append('cancel')
        for k in self.KS:
            p = bid + k
            if p < ask and not (s[1] == 'rest' and s[2] == p):
                acts.append(post(k))
        acts.append('take')
        return acts

    def step(self, d, s, a):
        t = s[0]
        b = d['bid']
        bid, ask = b[t], self.ask(d, t)
        if a == 'take':
            return (t + 1, 'fill', ask), -float(fee(ask))
        st = s[1:]
        if a == 'cancel':
            st = ('flat',)
        elif a != 'wait':
            k = int(a[4:])
            p = bid + k
            st = ('rest', p, 0 if k > 0 else self.DISP)
        if st[0] != 'rest':
            return (t + 1,) + st, 0.0
        _, p, q = st
        v = d['v'][t]
        if p > bid:
            if v >= 1:
                return (t + 1, 'fill', p), 0.0
        elif p == bid:
            if v > q:
                return (t + 1, 'fill', p), 0.0
            q -= v
        elif p == bid - 1:
            reach = max(v - self.DISP, 0)
            if reach > q:
                return (t + 1, 'fill', p), 0.0
            q -= reach
        if t + 1 < self.H:
            nb = b[t + 1]
            if nb > bid and p == nb:
                q = 0
        return (t + 1, 'rest', p, q), 0.0

    def key(self, d, s, i):
        t = s[0]
        b = d['bid']
        obs = [d['g0']]
        for r in range(1, t + 1):
            obs.append((d['g1'] if r == 1 else 'cme', b[r] - BID0, d['v'][r - 1]))
        return (tuple(obs), s)

    def card(self, d):
        return (d['c'], d['L'])


class L2:
    name = 'L2'
    players = 2
    merge = False
    ASK = 53
    PRICES = (51, 52)

    def __init__(self):
        self.prm = dict(pf=0.5, g1_hit=0.75, rho={1: 0.35, 0: 0.6}, F_follow=54, F_none=50, R=(51, 52, 53))

    def deals(self):
        p = self.prm
        out = []
        for c in (0, 1):
            pc = p['pf'] if c else 1 - p['pf']
            for g1 in (0, 1):
                pg = p['g1_hit'] if g1 == c else 1 - p['g1_hit']
                for r in p['R']:
                    for a0, a1 in itertools.product((0, 1), (0, 1)):
                        rho = p['rho'][c]
                        pa = (rho if a0 else 1 - rho) * (rho if a1 else 1 - rho)
                        for z0, z1 in itertools.product((0, 1), (0, 1)):
                            out.append((pc * pg * pa * 0.25 / len(p['R']),
                                        _Frozen(dict(c=c, g1=g1, r=r, a=(a0, a1), z=(z0, z1)))))
        return out

    # state: ('end', who, price) | (t, o1, o2, o1_vis, hist0, hist1); o = None | (price, frame)
    def root(self, d):
        return (0, None, None, None, (), ())

    def terminal(self, d, s):
        if s[0] == 'end':
            _, who, p = s
            if who == 0:
                F = self.prm['F_follow'] if d['c'] else self.prm['F_none']
                return True, float(F - p)
            return True, -float(d['r'] - p)
        return s[0] >= 4, 0.0

    def turn(self, d, s):
        return s[0] % 2

    def actions(self, d, s):
        t, o1, o2 = s[0], s[1], s[2]
        seat = t % 2
        own = o1 if seat == 0 else o2
        acts = ['wait']
        if own is not None:
            acts.append('cancel')
        for p in self.PRICES:
            if not (own is not None and own[0] == p):
                acts.append(post(p - BID0))
        if seat == 0:
            acts.append('take')
        return acts

    def _post(self, o, a, f):
        if a == 'wait':
            return o
        if a == 'cancel':
            return None
        return (BID0 + int(a[4:]), f)

    def step(self, d, s, a):
        t, o1, o2, o1v, h0, h1 = s
        f, seat = divmod(t, 2)
        if seat == 0:
            if a == 'take':
                return ('end', 0, self.ASK), -float(fee(self.ASK))
            n1 = self._post(o1, a, f)
            return (t + 1, n1, o2, o1, h0 + (a,), h1), 0.0
        n2 = self._post(o2, a, f)
        h1 = h1 + (a,)
        if d['a'][f]:
            cands = []
            if o1 is not None:
                cands.append((-o1[0], o1[1], 0, o1[0]))
            if n2 is not None:
                cands.append((-n2[0], n2[1], 1, n2[0]))
            if cands:
                cands.sort(key=lambda x: (x[0], x[1]))
                best = cands[0]
                if len(cands) == 2 and cands[0][:2] == cands[1][:2]:
                    best = [c for c in cands if c[2] == d['z'][cands[0][1]]][0]
                return ('end', best[2], best[3]), 0.0
        return (t + 1, o1, n2, None, h0, h1), 0.0

    @staticmethod
    def _otok(o, f):
        return None if o is None else (o[0] - BID0, f - o[1])

    def key(self, d, s, i):
        t, o1, o2, o1v, h0, h1 = s
        f, seat = divmod(t, 2)
        pre = (2 - f, d['r'] if seat else None, (d['g1'], d['a'][0]) if f >= 1 else None)
        own, other = (o1, o2) if seat == 0 else (o2, o1v)
        return (seat, pre, self._otok(own, f), self._otok(other, f), h0 if seat == 0 else h1)

    def card(self, d):
        return (d['c'], d['r'])


class R2:
    name = 'R2'
    players = 1
    merge = True
    KS = (-1, 0, 1)
    DISP = 2
    AMAX = 2
    TAKES = (0, 1, 2)

    def __init__(self, H=5):
        self.H = H
        self.prm = dict(pg2=0.5, pf={1: 0.5, 2: 0.9}, pL={1: {1: 0.1, 2: 0.9}, 2: {1: 0.4, 2: 0.6}}, fade=3,
                        lift=(3, 5), lag_s={1: 0.3, 2: 0.1},
                        p_none=dict(s=0.15, b=0.02), p_pre=dict(s=0.05, b=0.02), p_lag=dict(s=0.30, b=0.03),
                        p_lift=dict(s=0.10, b=0.05), p_post=dict(s=0.05, b=0.25), p_postr=dict(s=0.20, b=0.15),
                        p_rev=0.5, rev=2, vs_none=3, vs_follow=1, vs_lag=3, vs_lift=3, vs_postr=3,
                        vb_none=1, vb_follow=3)

    def book(self, d):
        p = self.prm
        if d['c'] == 0:
            bid = [BID0 if t < p['fade'] else BID0 - 1 for t in range(self.H + 1)]
            return bid, [b + 2 for b in bid]
        L = d['L']
        cum = [0] + list(np.cumsum(p['lift']))
        bid = [BID0 + int(cum[min(max(t - L + 1, 0), len(p['lift']))]) for t in range(self.H + 1)]
        if d['r']:
            bid = [b - (p['rev'] if t >= L + 3 else 0) for t, b in enumerate(bid)]
        return bid, [b + (1 if t >= L else 2) for t, b in enumerate(bid)]

    def _pdist(self, d, t):
        p = self.prm
        if d['c'] == 0:
            dd, vs, vb = p['p_none'], p['vs_none'], p['vb_none']
        else:
            L = d['L']
            ph = 'pre' if t < L - 1 else ('lag' if t == L - 1 else ('lift' if t == L else 'post'))
            if ph == 'post' and d['r']:
                ph = 'postr'
            dd = dict(p['p_' + ph])
            if ph == 'lag':
                dd['s'] = p['lag_s'][d['g0']]
            vs, vb = p.get('vs_' + ph, p['vs_follow']), p['vb_follow']
        return [((0, 0), 1 - dd['s'] - dd['b']), ((1, vs), dd['s']), ((2, vb), dd['b'])]

    def deals(self):
        p = self.prm
        out = []
        for g0, pg in ((1, 1 - p['pg2']), (2, p['pg2'])):
            for c in (0, 1):
                pc = p['pf'][g0] if c else 1 - p['pf'][g0]
                Ls = tuple(p['pL'][g0].items()) if c else ((0, 1.0),)
                Rs = ((0, 1 - p['p_rev']), (1, p['p_rev'])) if c else ((0, 1.0),)
                for (L, pl), (r, prr) in itertools.product(Ls, Rs):
                    d0 = dict(g0=g0, c=c, L=L, r=r)
                    dists = [self._pdist(d0, t) for t in range(self.H)]
                    for os_ in itertools.product(*dists):
                        pr = pg * pc * pl * prr * float(np.prod([x[1] for x in os_]))
                        if pr > 0:
                            d = dict(d0, o=tuple(x[0] for x in os_))
                            d['bid'], d['ask'] = self.book(d)
                            out.append((pr, _Frozen(d)))
        return out

    # T2 kernels
    def _pbuy(self, p, q, side, vol, b):
        if side != 1:
            return False, q
        if p > b:
            return True, q
        if p == b:
            return (True, q) if vol > q else (False, q - vol)
        if p == b - 1:
            reach = max(vol - self.DISP, 0)
            return (True, q) if reach > q else (False, q - reach)
        return False, q

    def _psell(self, p, q, side, vol, a):
        if side != 2:
            return False, q
        if p < a:
            return True, q
        if p == a:
            return (True, q) if vol > q else (False, q - vol)
        if p == a + 1:
            reach = max(vol - self.DISP, 0)
            return (True, q) if reach > q else (False, q - reach)
        return False, q

    def _land_buy(self, p, nb, na):
        return None if p >= na else (0 if p > nb else self.DISP)

    def _land_sell(self, p, nb, na):
        return None if p <= nb else (0 if p < na else self.DISP)

    # state: (t, 'flat') | (t, 'rb', p, q) | (t, 'long', e, age, o, how) | (t, 'closed')
    def root(self, d):
        return (0, 'flat')

    def terminal(self, d, s):
        if s[1] == 'closed':
            return True, 0.0
        if s[0] >= self.H:
            if s[1] == 'long':
                b = d['bid'][self.H]
                return True, float(b - s[2] - fee(b))
            return True, 0.0
        return False, 0.0

    def turn(self, d, s):
        return 0

    def actions(self, d, s):
        t = s[0]
        b, ka = d['bid'][t], d['ask'][t]
        acts = ['wait']
        if s[1] in ('flat', 'rb'):
            if s[1] == 'rb':
                acts.append('cancel')
            for k in self.KS:
                p = b + k
                if p < ka and not (s[1] == 'rb' and s[2] == p):
                    acts.append('make%+d' % k)
        else:
            o = s[4]
            if o is not None:
                acts.append('cancel')
            for k in self.KS:
                p = ka - k
                if p > b and not (o is not None and o[0] == p):
                    acts.append('make%+d' % k)
        acts += ['take%d' % k for k in self.TAKES]
        return acts

    def step(self, d, s, a):
        t = s[0]
        bid, ask = d['bid'], d['ask']
        b, ka = bid[t], ask[t]
        side, vol = d['o'][t]
        nb, na = bid[t + 1], ask[t + 1]
        tk = int(a[4:]) if a.startswith('take') else None
        if s[1] in ('flat', 'rb'):
            st = s[1:]
            if st[0] == 'rb':
                f, q = self._pbuy(st[1], st[2], side, vol, b)
                if f:
                    return (t + 1, 'long', st[1], 0, None, 'M'), 0.0
                st = ('rb', st[1], q)
            if tk is not None:
                if na <= ka + tk:
                    return (t + 1, 'long', na, 0, None, 'T'), -float(fee(na))
                return (t + 1, 'flat'), 0.0
            if a == 'cancel':
                return (t + 1, 'flat'), 0.0
            if a == 'wait':
                return (t + 1,) + st, 0.0
            p = b + int(a[4:])
            q0 = self._land_buy(p, nb, na)
            return ((t + 1, 'flat') if q0 is None else (t + 1, 'rb', p, q0)), 0.0
        _, _, e, age, o, how = s
        age1 = min(age + 1, self.AMAX)
        if o is not None:
            f, q = self._psell(o[0], o[1], side, vol, ka)
            if f:
                return (t + 1, 'closed'), float(o[0] - e)
            o = (o[0], q)
        if tk is not None:
            if nb >= b - tk:
                return (t + 1, 'closed'), float(nb - e - fee(nb))
            return (t + 1, 'long', e, age1, None, how), 0.0
        if a == 'cancel':
            return (t + 1, 'long', e, age1, None, how), 0.0
        if a == 'wait':
            return (t + 1, 'long', e, age1, o, how), 0.0
        p = ka - int(a[4:])
        q0 = self._land_sell(p, nb, na)
        return (t + 1, 'long', e, age1, None if q0 is None else (p, q0), how), 0.0

    def key(self, d, s, i):
        t = s[0]
        obs = (d['g0'], tuple(d['bid'][:t + 1]), tuple(d['ask'][:t + 1]), tuple(d['o'][:t]))
        return (obs, s)

    def card(self, d):
        return (d['c'], d['L'], d['r'])


class _Frozen(dict):
    """A dict usable as part of a hashable state (deals are read-only)."""

    def __hash__(self):
        return id(self)
