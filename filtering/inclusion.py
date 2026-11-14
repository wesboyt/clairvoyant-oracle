"""How the filter includes and excludes samples (FILTER_PAPER.md section 3).

Every update draws m = K n_f filtered candidates on-policy and splits them by a statistical test: a candidate is
INCLUDED as a pivot if its disagreement exceeds b_hat, the (1 - f) quantile of the m sampled scores (the first n_f
such, in sampling order), and otherwise EXCLUDED from the policy-gradient batch; excluded candidates at or below the
bar become lock rows. The population bar b is the occupancy-weighted (1 - f) quantile of d over the filtered
decisions. Logged per update:
    pivot_novel     share of included pivots that are novel at the population bar (d > b)
    lock_known      share of lock rows that are known at the population bar (d <= b)
    q_band          |F(b_hat) - (1 - f)|: how far the sampled bar sits from the population quantile, in quantile
                    units, compared with the DKW radius sqrt(ln(2 / delta) / (2 m))
    known_grad      share of the batch's policy-gradient rows that come from known decisions (population bar):
                    0 when the test is exact, (1 - f) in expectation under on-policy sampling
    novel_rate      a novel decision's inclusion weight relative to on-policy sampling (pivot share / sample share)

    python filtering/inclusion.py [--steps 1500] -> filtering/results/inclusion.json
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, '..', 'tacking', 'src'), os.path.join(HERE, '..')]
from polar.games import build  # noqa: E402
from filtering.filterlock import FilterLock, oracle_fields  # noqa: E402


def pop_bar(d, occ, F, f):
    w = np.where(F, occ, 0.0)
    o = np.argsort(d)
    cum = np.cumsum(w[o]) / max(w.sum(), 1e-300)
    return float(d[o][min(np.searchsorted(cum, 1 - f), len(o) - 1)]), (lambda x: float(w[d <= x].sum() / w.sum()))


def measure(game, filt, kw, arm, steps, seeds, delta=0.05):
    rows = []
    for seed in range(seeds):
        fl = FilterLock(game, arm=arm, kind='tabular', filtered=filt, seed=seed, **kw)
        for _ in range(steps):
            pis = fl.policies()
            flds, _ = oracle_fields(game, pis)
            _, _, info = fl.step()
            for i, inf in enumerate(info):
                d, occ, F = flds[i]['d'], flds[i]['occ'], fl.filtered[i]
                b, cdf = pop_bar(d, occ, F, fl.f)
                piv_f = [k for k in inf['piv'] if F[k]]
                novel_mass = 1.0 - cdf(b)
                r = dict(m=len(inf['cand']))
                if piv_f:
                    r['pivot_novel'] = float(np.mean(d[piv_f] > b))
                    r['known_grad'] = float(np.mean(d[piv_f] <= b))
                if len(inf['lock']):
                    r['lock_known'] = float(np.mean(d[inf['lock']] <= b))
                if np.isfinite(inf['bar']):
                    r['q_band'] = abs(cdf(inf['bar']) - (1 - fl.f))
                if piv_f and novel_mass > 0:
                    r['novel_rate'] = float(np.mean(d[piv_f] > b)) / novel_mass
                rows.append(r)
    out = {}
    for key in ('pivot_novel', 'lock_known', 'q_band', 'known_grad', 'novel_rate'):
        v = [r[key] for r in rows if key in r]
        out[key] = [float(np.mean(v)), float(np.quantile(v, 0.9))] if v else None
    m = int(np.median([r['m'] for r in rows]))
    out['m'] = m
    out['dkw_radius'] = float(np.sqrt(np.log(2 / delta) / (2 * max(m, 1))))
    if out['q_band']:
        out['q_band_within_dkw'] = float(np.mean([r['q_band'] <= out['dkw_radius'] for r in rows if 'q_band' in r]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=1500)
    ap.add_argument('--seeds', type=int, default=3)
    a = ap.parse_args()
    res = {}
    for gname, filt, kw in (('Kuhn', None, dict(n_u=0, n_f=3, lr=0.5)),
                            ('Leduc', lambda k: k[1] is not None, dict(n_u=8, n_f=16, lr=1.0))):
        g = build(gname)
        for arm in ('onpolicy', 'filter_lock'):
            r = measure(g, filt, kw, arm, a.steps, a.seeds)
            res['%s/%s' % (gname, arm)] = r
            print(gname, arm, json.dumps(r), flush=True)
    os.makedirs(os.path.join(HERE, 'results'), exist_ok=True)
    with open(os.path.join(HERE, 'results', 'inclusion.json'), 'w') as fh:
        json.dump(res, fh, indent=1)


if __name__ == '__main__':
    main()
