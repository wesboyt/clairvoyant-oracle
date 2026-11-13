"""The candidate pole sets of a game (theory.md S3.1):
    south   two extreme points of each player's optimal set X*_i, as far apart as found (F1 'polar optima')
    hull    up to m extreme points spanning X*_i (for the m-pole certificate, Lemma 3)
    field   the two oracle fields' poles: the clairvoyant greedy policy (north) and the exact pi* (south) (F2)
    centroid  one point: the barycentre of the optimal cloud (one player: uniform over optimal actions at every
            key; two players: the mean of the hull vertices) (F1's 'centroid of the optimal cloud')
"""
import numpy as np

from . import lp, oracle as O


def get_poles(game, coords=True, n_rand=24, m_hull=8, seed=0):
    rng = np.random.default_rng(seed)
    P = game.players
    info = dict(dim=[], n_vertices=[], diam=[], optional_doors=[])
    south, hull, sets = [], [], []
    if P == 1:
        p1, p2, st = lp.single_agent_poles(game)
        south.append([p1, p2])
        pl = game.plex[0]
        sel = [p1, p2]
        opt = st['opt']
        for _ in range(m_hull - 2):            # random deterministic optimal policies (vertices of X*)
            pi = np.zeros(pl.n); pi[0] = 1.0
            for k in range(pl.K):
                s0, s1 = pl.start[k], pl.start[k + 1]
                o = np.flatnonzero(opt[s0:s1])
                pi[s0 + rng.choice(o)] = 1.0
            sel.append(pl.realise(pi))
        hull.append(sel)
        info['dim'].append(st['dim'])
        info['n_vertices'].append(None)
        info['diam'].append(float(np.linalg.norm(p1[1:] - p2[1:])))
        info['optional_doors'].append(st['tie_keys'])
        sets.append(lp.OptimalSet(game, 0))
        v = st['v']
        centroid = [pl.realise(st['uniform_opt'])]
    else:
        for i in range(P):
            S = lp.OptimalSet(game, i)
            sets.append(S)
            V = S.vertices(n_rand=n_rand, rng=rng, coords=coords)
            a, b = lp.diameter_pair(V)
            south.append([a, b])
            # greedy farthest-point selection of up to m_hull vertices
            sel = [a, b]
            while len(sel) < min(m_hull, len(V)):
                dmin = np.min([np.linalg.norm(V - s, axis=1) for s in sel], axis=0)
                j = int(np.argmax(dmin))
                if dmin[j] < 1e-9:
                    break
                sel.append(V[j])
            hull.append(sel)
            info['dim'].append(lp.affine_dim(V))
            info['n_vertices'].append(int(len(V)))
            info['diam'].append(float(np.linalg.norm(a[1:] - b[1:])))
            rng_ = (V.max(0) - V.min(0))[1:]
            info['optional_doors'].append(int((rng_ > 1e-7).sum()))
        v = sets[0].v
        centroid = [np.mean(np.array(h), axis=0) for h in hull]
    pis, _ = O.optimum(game)
    xs = game.realise(pis)
    north = []
    for i in range(P):
        mf, _, _ = O.clairvoyant_field(game, pis, i)
        north.append(game.plex[i].realise(O.greedy(game.plex[i], mf)))
    field = [[north[i], xs[i]] for i in range(P)]
    # pad hull lists to the same length per player (repeat the last pole)
    m = max(len(h) for h in hull)
    hull = [h + [h[-1]] * (m - len(h)) for h in hull]
    cen = [[c] for c in centroid]
    return dict(south=south, hull=hull, field=field, centroid=cen), v, sets, info
