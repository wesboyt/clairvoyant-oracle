"""Kuhn and Leduc poker (re-implemented from the textbook rules; not copied from any other code base).

Kuhn: 3 cards J < Q < K, ante 1, one bet of 1. Player 0 acts first: check / bet; facing a bet: fold / call.
Game value for player 0: -1/18. Player 0's Nash set is the one-parameter family alpha in [0, 1/3] (Kuhn 1950);
player 1's Nash strategy is unique.

Leduc: 6 cards (ranks J, Q, K in 2 suits), ante 1, round 1 raise 2, round 2 raise 4, at most 2 raises per round,
player 0 first in each round, a public board card between the rounds. A private card that pairs the board wins,
otherwise the higher rank; equal ranks split. Information sets: (public betting, own rank, board rank) -> 288 sets.
Game value for player 0 ~ -0.0856.
"""
import itertools


class Kuhn:
    name = 'Kuhn'
    players = 2
    merge = False

    def deals(self):
        return [(1 / 6, (a, b)) for a, b in itertools.permutations(range(3), 2)]

    def root(self, d):
        return ''

    def terminal(self, d, s):
        win = 1.0 if d[0] > d[1] else -1.0
        if s in ('cc',):
            return True, win
        if s in ('bc', 'cbc'):
            return True, 2 * win
        if s == 'bf':
            return True, 1.0
        if s == 'cbf':
            return True, -1.0
        return False, 0.0

    def turn(self, d, s):
        return len(s) % 2

    def actions(self, d, s):
        return ['f', 'c'] if s.endswith('b') else ['c', 'b']

    def step(self, d, s, a):
        return s + a, 0.0

    def key(self, d, s, i):
        return (d[i], s)


class Leduc:
    name = 'Leduc'
    players = 2
    merge = False
    RAISE = (2, 4)

    def deals(self):
        cards = range(6)                              # rank = c // 2
        return [(1 / 120, (a // 2, b // 2, c // 2)) for a, b, c in itertools.permutations(cards, 3)]

    # state: (round, history tuple per round, contributions (c0, c1), folded-by or None)
    def root(self, d):
        return (0, ((), ()), (1, 1), None)

    def _round_over(self, h):
        if len(h) >= 2 and h[-1] == 'c':
            return True
        return False

    def terminal(self, d, s):
        rnd, hist, con, fold = s
        if fold is not None:
            return True, float(con[fold]) * (-1 if fold == 0 else 1)
        if rnd == 2:
            r0, r1, b = d
            if r0 == b and r1 != b:
                w = 1
            elif r1 == b and r0 != b:
                w = -1
            else:
                w = (r0 > r1) - (r0 < r1)
            return True, float(w * con[1 if w > 0 else 0]) if w else 0.0
        return False, 0.0

    def turn(self, d, s):
        rnd, hist, con, fold = s
        return len(hist[rnd]) % 2

    def actions(self, d, s):
        rnd, hist, con, fold = s
        h = hist[rnd]
        facing = con[0] != con[1]
        nr = sum(1 for x in h if x == 'r')
        acts = []
        if facing:
            acts.append('f')
        acts.append('c')
        if nr < 2:
            acts.append('r')
        return acts

    def step(self, d, s, a):
        rnd, hist, con, fold = s
        i = len(hist[rnd]) % 2
        con = list(con)
        if a == 'f':
            return (rnd, hist, tuple(con), i), 0.0
        h = hist[rnd] + (a,)
        if a == 'c':
            con[i] = con[1 - i]
        else:
            con[i] = con[1 - i] + self.RAISE[rnd]
        hist = tuple(h if r == rnd else hist[r] for r in range(2))
        if a == 'c' and len(h) >= 2:
            return (rnd + 1, hist, tuple(con), None), 0.0
        return (rnd, hist, tuple(con), None), 0.0

    def key(self, d, s, i):
        rnd, hist, con, fold = s
        return (d[i], d[2] if rnd >= 1 else None, hist)
