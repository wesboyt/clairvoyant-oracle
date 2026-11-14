"""Is the spill into known regions correlated with what those regions need? (FILTER_PAPER.md section 3)

For every update and every decision k, the logit change dz_k of the step is compared with the decision's own
improvement direction, its centred on-policy advantage A_k = Q^pi(k, .) - V(k), in the policy's Fisher metric:
    corr_k = <F_k dz_k, A_k> / (|dz_k|_F |A_k|_F),   F_k = diag(pi_k) - pi_k pi_k'
corr = 1: the decision moved exactly the way its own interventional values point; 0: unrelated to them.
Reported separately for pivots (intervened on), known decisions (d <= bar, not pivots) and novel non-pivots, as
the movement-weighted mean of corr and the share of the step's Fisher movement each group receives.

    python filtering/causal.py [--steps 1500] -> filtering/results/causal.json
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, '..', 'tacking', 'src'), os.path.join(HERE, '..')]
from polar.games import build  # noqa: E402
from filtering.filterlock import FilterLock, poker_tokens  # noqa: E402


def measure(game, kind, buckets, filtered, kw, steps, seeds):
    acc = {g: dict(w=0.0, wc=0.0, move=0.0) for g in ('pivots', 'known', 'novel')}
    for seed in range(seeds):
        fl = FilterLock(game, arm='filter', kind=kind, buckets=buckets, tokenizer=poker_tokens, filtered=filtered,
                        seed=seed, **kw)
        for _ in range(steps):
            z0 = [p.logits() for p in fl.pols]
            pis, flds, info = fl.step()
            for i, (pol, pl) in enumerate(zip(fl.pols, game.plex)):
                dz = pol.logits() - z0[i]
                d, bar = flds[i]['d'], info[i]['bar']
                piv = set(info[i]['piv'].tolist())
                for k in range(pl.K):
                    if flds[i]['occ'][k] <= 0:
                        continue
                    s0, s1 = pl.start[k], pl.start[k + 1]
                    p = pis[i][s0:s1]
                    F = np.diag(p) - np.outer(p, p)
                    a = flds[i]['Qpi'][s0:s1] - p @ flds[i]['Qpi'][s0:s1]
                    nz, na = np.sqrt(max(dz[s0:s1] @ F @ dz[s0:s1], 0.0)), np.sqrt(max(a @ F @ a, 0.0))   # PSD up to rounding
                    if nz < 1e-14 or na < 1e-14:
                        continue
                    c = (dz[s0:s1] @ F @ a) / (nz * na)
                    grp = 'pivots' if k in piv else ('known' if fl.filtered[i][k] and d[k] <= bar else 'novel')
                    w = flds[i]['occ'][k] * nz                       # occupancy x Fisher movement
                    acc[grp]['w'] += w
                    acc[grp]['wc'] += w * c
                    acc[grp]['move'] += w
    tot = sum(v['move'] for v in acc.values())
    return {g: dict(corr=v['wc'] / max(v['w'], 1e-300), movement_share=v['move'] / max(tot, 1e-300))
            for g, v in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=1500)
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--games', nargs='*', default=['Kuhn', 'Leduc'])
    a = ap.parse_args()
    path = os.path.join(HERE, 'results', 'causal.json')
    out = json.load(open(path)) if os.path.exists(path) else {}
    for gname, buckets, filt, kw in (('Kuhn', 4, None, dict(n_u=0, n_f=3, lr=0.5)),
                                     ('Leduc', 40, lambda k: k[1] is not None, dict(n_u=8, n_f=16, lr=1.0))):
        if gname not in a.games:
            continue
        g = build(gname)
        for kind in ('tabular', 'hashed', 'tokens'):
            r = measure(g, kind, buckets, filt, kw, a.steps, a.seeds)
            out['%s/%s' % (gname, kind)] = r
            print(gname, kind, {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in r.items()}, flush=True)
    os.makedirs(os.path.join(HERE, 'results'), exist_ok=True)
    with open(path, 'w') as fh:
        json.dump(out, fh, indent=1)


if __name__ == '__main__':
    main()
