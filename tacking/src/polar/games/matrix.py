"""Zero-sum matrix games as two-player processes (player 0 maximises x^T M y; player 1 does not see player 0's move).

    MP      matching pennies: unique interior Nash; plain gradient play cycles
    RPS     rock-paper-scissors: unique interior Nash
    MPdup   matching pennies with column 0 duplicated: player 1's Nash set is a SEGMENT (y0 + y2 = 1/2)
    MPtri   column 0 triplicated: player 1's Nash set is a TRIANGLE (2-dimensional) -- the adversarial case
    Skew    a 3x3 game with a unique pure-strategy saddle (gradient play converges; control)
"""
import numpy as np

MATS = {
    'MP': [[1, -1], [-1, 1]],
    'RPS': [[0, -1, 1], [1, 0, -1], [-1, 1, 0]],
    'MPdup': [[1, -1, 1], [-1, 1, -1]],
    'MPtri': [[1, -1, 1, 1], [-1, 1, -1, -1]],
    'Skew': [[3, 1, 4], [2, 0, 1], [1, -1, 0]],
}


class Matrix:
    players = 2
    merge = False

    def __init__(self, name):
        self.name = name
        self.M = np.array(MATS[name], float)

    def deals(self):
        return [(1.0, None)]

    def root(self, d):
        return ()

    def terminal(self, d, s):
        if len(s) == 2:
            return True, float(self.M[s[0], s[1]])
        return False, 0.0

    def turn(self, d, s):
        return len(s)

    def actions(self, d, s):
        return list(range(self.M.shape[len(s)]))

    def step(self, d, s, a):
        return s + (a,), 0.0

    def key(self, d, s, i):
        return ('root', i)
