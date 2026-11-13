"""Run a learner and record the polar metric along the trajectory (theory.md S3-S4)."""
import numpy as np

from . import metrics as M


def run(game, learner, T, poles, v, every=1, l1_sets=None, l1_at=()):
    """poles: {name: [list of pole realisations for player 0, ... for player 1]} (each player's list has the same
    length m). Returns a dict of numpy arrays.
      d[name]        [T+1, players, m]   distances of the iterate to the poles
      dA[name]       [T+1, players, m]   the same for the linear average
      cos[name]      [T, players, m]     cos of the angle between the step and (pole - iterate)
      hull[name], hullA[name]  [n_eval, players]  distance to conv(poles) from distances only
      nc, ncA        [n_eval]            NashConv (one player: V_opt - V) of the iterate / average
      gap[name]      [n_eval, m]         restricted gaps of the iterate against the poles
    """
    P = game.players
    D2 = {nm: [M.pole_gram(ps[i]) for i in range(P)] for nm, ps in poles.items()}
    out = dict(t=[], nc=[], ncA=[], val=[], valA=[])
    for nm in poles:
        out['d_' + nm], out['dA_' + nm], out['cos_' + nm] = [], [], []
        out['hull_' + nm], out['hullA_' + nm], out['gap_' + nm] = [], [], []
    l1 = dict(t=[], iter=[], avg=[])

    def record(xs, xa, t):
        for nm, ps in poles.items():
            out['d_' + nm].append([M.dists(xs[i], ps[i]) for i in range(P)])
            out['dA_' + nm].append([M.dists(xa[i], ps[i]) for i in range(P)])
        if t % every == 0 or t == T:
            out['t'].append(t)
            out['nc'].append(game.nashconv(xs))
            out['ncA'].append(game.nashconv(xa))
            out['val'].append(game.value(xs))
            out['valA'].append(game.value(xa))
            for nm, ps in poles.items():
                d, dA = out['d_' + nm][-1], out['dA_' + nm][-1]
                out['hull_' + nm].append([M.hull_dist_from_distances(np.array(d[i]), D2[nm][i])[0] for i in range(P)])
                out['hullA_' + nm].append([M.hull_dist_from_distances(np.array(dA[i]), D2[nm][i])[0]
                                           for i in range(P)])
                out['gap_' + nm].append(M.restricted_gaps(game, xs, ps, v))
        if l1_sets is not None and t in l1_at:
            l1['t'].append(t)
            l1['iter'].append([S.l1_dist(xs[i]) for i, S in enumerate(l1_sets)])
            l1['avg'].append([S.l1_dist(xa[i]) for i, S in enumerate(l1_sets)])

    xs = [x.copy() for x in learner.xs]
    record(xs, learner.avg(), 0)
    for t in range(1, T + 1):
        learner.step()
        xn = learner.xs
        for nm, ps in poles.items():
            out['cos_' + nm].append([[M.cos_angle(xn[i] - xs[i], xs[i], p) for p in ps[i]] for i in range(P)])
        xs = [x.copy() for x in xn]
        record(xs, learner.avg(), t)
    res = {k: np.array(v_) for k, v_ in out.items()}
    if l1_sets is not None:
        res['l1_t'] = np.array(l1['t'])
        res['l1_iter'] = np.array(l1['iter'])
        res['l1_avg'] = np.array(l1['avg'])
    return res


def summarise(res, poles, tail=0.1):
    """Scalars: final NashConv, final hull distances, Fejer violations (share of steps where a pole distance grew
    by more than 1e-12), mean cos over the last `tail` of the run."""
    s = dict(nc_final=float(res['nc'][-1]), ncA_final=float(res['ncA'][-1]), nc_min=float(res['nc'].min()))
    for nm in poles:
        d = res['d_' + nm]
        inc = np.diff(d, axis=0) > 1e-9 * np.maximum(1.0, d[:-1])
        s['fejer_viol_' + nm] = float(inc.mean())
        s['hull_final_' + nm] = res['hull_' + nm][-1].tolist()
        s['hullA_final_' + nm] = res['hullA_' + nm][-1].tolist()
        n = max(1, int(len(res['cos_' + nm]) * tail))
        c = res['cos_' + nm][-n:]
        s['cos_tail_' + nm] = [[float(np.nanmean(c[:, i, j])) if np.isfinite(c[:, i, j]).any() else None
                                for j in range(c.shape[2])] for i in range(c.shape[1])]
        s['d_final_' + nm] = d[-1].tolist()
    return s
