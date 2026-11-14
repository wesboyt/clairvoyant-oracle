"""Where the edge is, and where the samples go (FILTER_PAPER.md section 2).

A player's best-response gain splits exactly over its decisions (performance difference with the best response
as the comparison policy):
    G_i = sum_k rho*(k) r(k),   r(k) = Q^pi(k, a*(k)) - V(k),   rho*(k) = best-response own reach x R(k)
where a*(k) is the oracle's action at k. A decision's share of the edge is rho*(k) r(k) / G_i; its share of the
samples is occ(k) / sum occ (on-policy sampling draws decisions in proportion to occupancy).

Leaf decisions: decisions after which the player never acts again. Logged every 50 updates for the on-policy,
filter and filter + lock arms: the leaves' share of samples, of the edge, of the pivots, and NashConv.

    python filtering/regions.py [--steps 3000] -> filtering/results/regions.json
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, '..', 'tacking', 'src'), os.path.join(HERE, '..')]
from polar.games import build  # noqa: E402
from filtering.filterlock import FilterLock, oracle_fields, poker_tokens  # noqa: E402


def leaves(pl):
    """Decisions with no later decision of the same player."""
    has_child = np.zeros(pl.K, bool)
    for k2 in range(pl.K):
        for s in pl.parents[k2]:
            if s > 0:
                has_child[pl.key_of_seq[s]] = True
    return ~has_child


def edge_split(game, pis):
    """Per player: (G, rho* r per decision, occupancy per decision)."""
    flds, xs = oracle_fields(game, pis)
    out = []
    for i, (pl, fld) in enumerate(zip(game.plex, flds)):
        Qbr = fld['Qbr']
        astar = np.array([pl.start[k] + int(np.argmax(Qbr[pl.start[k]:pl.start[k + 1]])) for k in range(pl.K)])
        br = np.zeros(pl.n)
        br[0] = 1.0
        br[astar] = 1.0
        xbr = pl.realise(br)
        R = game.cf_reach(xs, i)
        rho = pl.reach_in(xbr) * R
        r = fld['Qpi'][astar] - fld['V']
        out.append((fld['gap'], rho * r, fld['occ']))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=3000)
    ap.add_argument('--seeds', type=int, default=3)
    a = ap.parse_args()
    res = {}
    for gname, buckets, filt, kw in (('Kuhn', 4, None, dict(n_u=0, n_f=3, lr=0.5)),
                                     ('Leduc', 40, lambda k: k[1] is not None, dict(n_u=8, n_f=16, lr=1.0))):
        g = build(gname)
        leaf = [leaves(pl) for pl in g.plex]
        for kind in ('tabular', 'tokens'):
            for arm in ('onpolicy', 'filter', 'filter_lock'):
                runs = []
                for seed in range(a.seeds):
                    fl = FilterLock(g, arm=arm, kind=kind, buckets=buckets, tokenizer=poker_tokens, filtered=filt,
                                    beta=2.0, seed=seed, **kw)
                    log = []
                    piv_leaf = piv_all = 0
                    for t in range(a.steps + 1):
                        if t % 50 == 0:
                            sp = edge_split(g, fl.policies())
                            G = sum(s[0] for s in sp)
                            edge_leaf = sum(s[1][lf].sum() for s, lf in zip(sp, leaf))
                            occ_leaf = sum(s[2][lf].sum() for s, lf in zip(sp, leaf)) / sum(s[2].sum() for s in sp)
                            check = max(abs(s[0] - s[1].sum()) for s in sp)
                            log.append(dict(t=t, nashconv=fl.nashconv(), G=G, edge_leaf_share=edge_leaf / max(G, 1e-12),
                                            sample_leaf_share=occ_leaf, check=check,
                                            pivot_leaf_share=piv_leaf / max(piv_all, 1)))
                            piv_leaf = piv_all = 0
                        if t < a.steps:
                            _, _, info = fl.step()
                            for i, inf in enumerate(info):
                                piv_leaf += int(leaf[i][inf['piv']].sum())
                                piv_all += len(inf['piv'])
                    runs.append(log)
                key = '%s/%s/%s' % (gname, kind, arm)
                res[key] = runs
                last = [r[-1] for r in runs]
                first = [r[0] for r in runs]
                print('%-26s start: samples %.3f edge %.3f | end: NC %.3f samples %.3f edge %.3f pivots %.3f | check %.0e'
                      % (key, np.mean([f['sample_leaf_share'] for f in first]), np.mean([f['edge_leaf_share'] for f in first]),
                         np.mean([l['nashconv'] for l in last]), np.mean([l['sample_leaf_share'] for l in last]),
                         np.mean([l['edge_leaf_share'] for l in last]), np.mean([l['pivot_leaf_share'] for l in last]),
                         max(l['check'] for r in runs for l in r)), flush=True)
    os.makedirs(os.path.join(HERE, 'results'), exist_ok=True)
    with open(os.path.join(HERE, 'results', 'regions.json'), 'w') as fh:
        json.dump(res, fh)


if __name__ == '__main__':
    main()
