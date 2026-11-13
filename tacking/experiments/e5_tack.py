"""E5: tacking between the south field (Q^pi rollout labels) and the north field (clairvoyant max-flow relative
regret swapped in at closed doors by the argmax gate), against each field alone. -> results/tack/<game>_<arm>.json

Per step: the Lyapunov quantity L (NashConv; one player: V_opt - V) of the iterate and of the linear average,
gate flags, door masses, the distance of the iterate to the hull of up to 8 optimal vertices (polar instrument,
Lemma 3) and its decrease per step ('velocity made good', VMG), and the angle of the logit step to each field
(cos to the centred south advantage A_S and to the centred north advantage A_N, occupancy-weighted).
Tacks = maximal runs of steps with at least one gate flag; progress per tack = the drop of L across the tack and
across the south leg that follows it.

    python experiments/e5_tack.py [--only GAME ...] [--quick]
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'src'))
from polar import metrics as M, tack as TK  # noqa: E402
from polar.games import build  # noqa: E402
from polar.poles import get_poles  # noqa: E402

OUT = os.path.join(HERE, '..', 'results', 'tack')

# game: (T, lr, closed doors {action: prob} at matching keys (key filter), arms)
CFG = {
    'DoorTrap': dict(T=3000, lr=0.3, doors={'door': 0.01, 'good': 0.01}),
    'FalseDoor': dict(T=3000, lr=0.3, doors={'door': 0.01}),
    'Sailboat': dict(T=3000, lr=0.3, doors={'door': 0.01, 'good': 0.01}),
    'WrongDoor': dict(T=3000, lr=0.3, doors={'door': 0.01, 'good': 0.01, 'false': 0.01}),
    'L1': dict(T=1500, lr=0.3, doors={'post+1': 0.01}, keyf='l1_trap'),
    'R2': dict(T=600, lr=0.3, doors={'take0': 0.01, 'take1': 0.01, 'take2': 0.01}, keyf='r2_root'),
    'MP': dict(T=2000, lr=0.05, probs={'0': [0.8, 0.2], '1': [0.3, 0.7]}),
    'Kuhn': dict(T=2000, lr=0.1),
    'L2': dict(T=600, lr=0.1),
    'Leduc': dict(T=200, lr=0.1),
}
ARMS = ('south', 'north', 'tack', 'tackall', 'vswap')


def keyf_ok(name, key):
    if name == 'l1_trap':
        return key[0] == (2,) and key[1] == (0, 'flat')
    if name == 'r2_root':
        return key[1] == (0, 'flat')
    return True


def make_init(g, c):
    pis = g.uniform()
    if 'probs' in c:
        for i, pr in c['probs'].items():
            pis[int(i)] = np.concatenate([[1.0], np.array(pr, float)])
        return pis, []
    doors = []
    for i, pl in enumerate(g.plex):
        for k, key in enumerate(pl.key_list):
            if not keyf_ok(c.get('keyf', ''), key):
                continue
            acts = pl.key_actions[k]
            hit = [a for a in c.get('doors', {}) if a in acts]
            if not hit:
                continue
            s0 = pl.start[k]
            shut = sum(c['doors'][a] for a in hit)
            rest = (1 - shut) / (len(acts) - len(hit))
            for j, a in enumerate(acts):
                pis[i][s0 + j] = c['doors'][a] if a in hit else rest
                if a in hit:
                    doors.append((i, s0 + j))
    return pis, doors


def centred(pl, q, occ):
    mean = pl.seg_sum(q) / pl.nact
    a = np.zeros(pl.n)
    a[1:] = (q[1:] - mean[pl.key_of_seq[1:]]) * occ[pl.key_of_seq[1:]]
    return a


def run_arm(g, arm, c, poles, quick):
    T = c['T'] // (10 if quick else 1)
    init, doors = make_init(g, c)
    L = TK.Tacker(g, arm, lr=c['lr'], init=init)
    hull = poles['hull']
    D2 = [M.pole_gram(hull[i]) for i in range(g.players)]
    north = TK.Clairvoyant(g)
    rec = dict(L=[], LA=[], flags=[], door=[], H=[], cosS=[], cosN=[], zmax=[])

    def hdist(xs):
        return float(np.sqrt(sum(M.hull_dist_from_distances(M.dists(xs[i], hull[i]), D2[i])[0] ** 2
                                 for i in range(g.players))))

    rec['L'].append(g.nashconv(L.xs))
    rec['LA'].append(g.nashconv(L.avg()))
    rec['H'].append(hdist(L.xs))
    rec['door'].append([float(L.pis[i][s]) for i, s in doors])
    for t in range(T):
        th0 = [th.copy() for th in L.theta]
        # the two fields at the current point (for the angle instrument)
        xs = L.xs
        mf = north.field(L.pis, xs)
        cs, cn = [], []
        pit = [TK.pruned(pl, pi, L.p_prune) for pl, pi in zip(g.plex, L.pis)]
        grads = g.grads(xs)
        AS, AN = [], []
        for i, pl in enumerate(g.plex):
            occ = pl.reach_in(xs[i]) * g.cf_reach(xs, i)
            occ = occ / max(occ.max(), 1e-300)
            W, _, _ = pl.backup(grads[i], pit[i])
            R = g.cf_reach(xs, i)
            Rs = np.concatenate([[1.0], R[pl.key_of_seq[1:]]])
            qS = np.where(Rs > 1e-300, W / np.maximum(Rs, 1e-300), 0.0)
            AS.append(centred(pl, qS, occ))
            AN.append(centred(pl, np.nan_to_num(mf[i]), occ))
        L.step()
        dth = np.concatenate([(L.theta[i] - th0[i])[1:] - 0 for i in range(g.players)])
        # remove per-key shifts (logits are defined up to a constant per key)
        parts = []
        for i, pl in enumerate(g.plex):
            d = (L.theta[i] - th0[i])
            m = pl.seg_sum(d) / pl.nact
            dd = d.copy()
            dd[1:] -= m[pl.key_of_seq[1:]]
            parts.append(dd[1:])
        dth = np.concatenate(parts)
        aS = np.concatenate([a[1:] for a in AS])
        aN = np.concatenate([a[1:] for a in AN])

        def cos(u, v):
            nu, nv = np.linalg.norm(u), np.linalg.norm(v)
            return float(u @ v / (nu * nv)) if nu > 1e-15 and nv > 1e-15 else np.nan
        rec['cosS'].append(cos(dth, aS))
        rec['cosN'].append(cos(dth, aN))
        rec['flags'].append(int(sum(L.last['flags'])))
        rec['zmax'].append(float(max(L.last['zmax'])))
        rec['L'].append(g.nashconv(L.xs))
        rec['LA'].append(g.nashconv(L.avg()))
        rec['H'].append(hdist(L.xs))
        rec['door'].append([float(L.pis[i][s]) for i, s in doors])
    Lc = np.array(rec['L'])
    fl = np.array(rec['flags']) > 0
    tacks = []
    t = 0
    while t < len(fl):
        if fl[t]:
            s = t
            while t < len(fl) and fl[t]:
                t += 1
            e = t                                   # steps s..e-1 flagged; L index s -> e
            nxt = e
            while nxt < len(fl) and not fl[nxt]:
                nxt += 1
            tacks.append(dict(start=s, end=e, dL_tack=float(Lc[e] - Lc[s]), dL_south_after=float(Lc[nxt] - Lc[e]),
                              len_after=int(nxt - e)))
        else:
            t += 1
    H = np.array(rec['H'])
    vmg = -np.diff(H)
    summ = dict(arm=arm, T=T, L0=float(Lc[0]), L_final=float(Lc[-1]), LA_final=float(rec['LA'][-1]),
                L_min=float(Lc.min()), flagged_steps=int(fl.sum()), n_tacks=len(tacks),
                H0=float(H[0]), H_final=float(H[-1]),
                vmg_mean_flagged=float(vmg[fl].mean()) if fl.any() else None,
                vmg_mean_unflagged=float(vmg[~fl].mean()) if (~fl).any() else None,
                dL_mean_flagged=float(np.diff(Lc)[fl].mean()) if fl.any() else None,
                dL_mean_unflagged=float(np.diff(Lc)[~fl].mean()) if (~fl).any() else None,
                cosS_mean=float(np.nanmean(rec['cosS'])), cosN_mean=float(np.nanmean(rec['cosN'])),
                cosS_flagged=float(np.nanmean(np.array(rec['cosS'])[fl])) if fl.any() else None,
                cosN_flagged=float(np.nanmean(np.array(rec['cosN'])[fl])) if fl.any() else None,
                door_final=rec['door'][-1], steps_to_L_below_1pct=next(
                    (int(i) for i, v in enumerate(Lc) if v <= 0.01 * max(Lc[0], 1e-12)), None))
    return dict(summary=summ, curves=rec, tacks=tacks[:200])


def door_field_grid(g, n=13):
    """DoorTrap: the expected step of each arm on a grid of (pi(door), pi(good)) (other actions share the rest
    as in the closed-door initialisation)."""
    pl = g.plex[0]
    out = {}
    us = np.linspace(0.01, 0.97, n)
    for arm in ('south', 'north', 'tack'):
        arrows = []
        for u in us:
            for w in us:
                spec = dict(door=u, good=w)
                c = dict(doors=spec)
                init, doors = make_init(g, c)
                L = TK.Tacker(g, arm, lr=0.3, init=init)
                L.step()
                d0 = L.pis[0][doors[0][1]] - u
                d1 = np.mean([L.pis[0][s] for _, s in doors[1:]]) - w
                arrows.append([float(u), float(w), float(d0), float(d1)])
        out[arm] = arrows
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', nargs='*')
    ap.add_argument('--quick', action='store_true')
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    for game, c in CFG.items():
        if a.only and game not in a.only:
            continue
        g = build(game)
        poles, v, sets, info = get_poles(g, coords=(g.players == 2 and game != 'Leduc'), n_rand=24)
        res = dict(game=game, cfg=c, poles_info=info)
        for arm in ARMS:
            t0 = time.time()
            r = run_arm(g, arm, c, poles, a.quick)
            r['sec'] = time.time() - t0
            res[arm] = r
            s = r['summary']
            print('%-9s %-6s T=%4d %5.0fs L %.3g -> %.3g (avg %.3g, min %.3g) flags %d tacks %d H %.3g -> %.3g '
                  'cosS %.2f cosN %.2f door %s' % (game, arm, s['T'], r['sec'], s['L0'], s['L_final'],
                                                    s['LA_final'], s['L_min'], s['flagged_steps'], s['n_tacks'],
                                                    s['H0'], s['H_final'], s['cosS_mean'], s['cosN_mean'],
                                                    np.round(s['door_final'][:4], 3)), flush=True)
        if game == 'DoorTrap':
            res['grid'] = door_field_grid(g)
        with open(os.path.join(OUT, game + '.json'), 'w') as f:
            json.dump(res, f)


if __name__ == '__main__':
    main()
