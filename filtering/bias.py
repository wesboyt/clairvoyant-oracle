"""Removing filtered-out samples biases the update toward optimism; locking them removes the bias
(FILTER_PAPER.md section 3.3).

The update believes it gains what its batch says: the first-order gain at the included pivots, measured with the
labels the batch was trained on (on-policy values, swapped to the oracle's at nearly closed doors). What it really
gains is the exact value change of the step at every decision (performance difference). Per update and player:
    predicted   sum over pivots of occ'(k) <dpi(.|k), q_label(k, .)>
    pivots_true sum over pivots of occ'(k) <dpi(.|k), Q^pi(k, .)>      (the same, with true values)
    realized    the exact value change of the step (all decisions)
so that  predicted - realized = (predicted - pivots_true)  +  (pivots_true - realized)
                                 label optimism                 omitted change at the excluded decisions
For two-player games the realized own-step gains are gains against the opponent of the moment; their sum divided by
the NashConv reduction of the run (booked_per_progress) measures how much gain the update books that is not progress
toward equilibrium: optimism about the opponent standing still.
Also logged: label_lift = mean disagreement of the included pivots / occupancy-weighted mean over the filtered
decisions (the truncation of the batch, E[d | included] / E[d]).

    python filtering/bias.py [--steps 3000] [--seeds 4] -> filtering/results/bias.json
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


def run(game, kind, buckets, filt, kw, arm, beta, steps, seed):
    fl = FilterLock(game, arm=arm, kind=kind, buckets=buckets, tokenizer=poker_tokens, filtered=filt, beta=beta,
                    seed=seed, **kw)
    tot = dict(predicted=0.0, pivots_true=0.0, realized=0.0)
    nc0 = fl.nashconv()
    lifts = []
    for _ in range(steps):
        pis, flds, info = fl.step()
        after = fl.policies()
        for i, (pl, fld, inf) in enumerate(zip(game.plex, flds, info)):
            joint = [after[j] if j == i else pis[j] for j in range(len(pis))]
            xs_a, xs_b = game.realise(joint), game.realise(pis)
            occ_a = pl.reach_in(xs_a[i]) * game.cf_reach(xs_b, i)
            dpi = after[i] - pis[i]
            for k in np.unique(inf['piv']):                  # a decision drawn twice is still one decision
                s0, s1 = pl.start[k], pl.start[k + 1]
                q = fld['Qpi'][s0:s1]
                qbr = fld['Qbr'][s0:s1]
                lab = qbr if (fl.swap and np.any((pis[i][s0:s1] < fl.eps) & (qbr >= qbr.max() - fl.delta))) else q
                tot['predicted'] += occ_a[k] * (dpi[s0:s1] @ lab)
                tot['pivots_true'] += occ_a[k] * (dpi[s0:s1] @ q)
            sign = (1.0 if i == 0 else -1.0) if game.players == 2 else 1.0
            tot['realized'] += sign * (game.value(xs_a) - game.value(xs_b))
            F, d, w = fl.filtered[i], fld['d'], np.where(fl.filtered[i], fld['occ'], 0.0)
            piv_f = [k for k in inf['piv'] if F[k]]
            if piv_f and (w * d).sum() > 0:
                lifts.append(d[piv_f].mean() / ((w * d).sum() / w.sum()))
    nc1 = fl.nashconv()
    # booked gain per unit of equilibrium progress: own-step gains against the opponent of the moment, summed over
    # the run, divided by the NashConv reduction they bought
    return dict(tot, label_lift=float(np.mean(lifts)) if lifts else np.nan, nashconv=nc1, nashconv0=nc0,
                booked_per_progress=tot['realized'] / max(nc0 - nc1, 1e-9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=3000)
    ap.add_argument('--seeds', type=int, default=4)
    a = ap.parse_args()
    res = {}
    for gname, buckets, filt, kw in (('Kuhn', 4, None, dict(n_u=0, n_f=3, lr=0.5)),
                                     ('Leduc', 40, lambda k: k[1] is not None, dict(n_u=8, n_f=16, lr=1.0))):
        g = build(gname)
        for kind in ('tabular', 'hashed', 'tokens'):
            for arm, beta in (('onpolicy', 0.0), ('filter', 0.0), ('filter_lock', 2.0)):
                runs = [run(g, kind, buckets, filt, kw, arm, beta, a.steps, s) for s in range(a.seeds)]
                P = np.array([r['predicted'] for r in runs])
                Pt = np.array([r['pivots_true'] for r in runs])
                R = np.array([r['realized'] for r in runs])
                ms = lambda x: [float(np.mean(x)), float(np.std(x, ddof=1) / np.sqrt(len(x)))]  # noqa: E731
                agg = dict(predicted=ms(P), realized=ms(R), optimism_ratio=ms(P / R),
                           label_optimism=ms(P - Pt), omitted=ms(Pt - R),
                           label_lift=ms([r['label_lift'] for r in runs]), nashconv=ms([r['nashconv'] for r in runs]),
                           booked_per_progress=ms([r['booked_per_progress'] for r in runs]))
                key = '%s/%s/%s' % (gname, kind, arm)
                res[key] = dict(agg, runs=runs)
                print('%-24s lift %.2f | booked gain %.2f predicted %.2f | NC %.3f | booked per unit progress %.1f'
                      % (key, agg['label_lift'][0], R.mean(), P.mean(), agg['nashconv'][0],
                         agg['booked_per_progress'][0]), flush=True)
    os.makedirs(os.path.join(HERE, 'results'), exist_ok=True)
    with open(os.path.join(HERE, 'results', 'bias.json'), 'w') as fh:
        json.dump(res, fh)


if __name__ == '__main__':
    main()
