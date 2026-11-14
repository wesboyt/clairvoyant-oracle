"""Figures for FILTER_PAPER.md from filtering/results/*.json.

    python filtering/figures.py
fig_<exp>_<class>.svg     NashConv (ladder: V_opt - V) of the iterate per arm, mean over seeds
fig_drift_<exp>.svg       drift of the agreeing region per segment (KL from the teacher) against beta, per class
fig_segment.svg           drift per segment against the segment length M, with and without the lock
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'tacking', 'src'))
from polar.svgplot import plot  # noqa: E402

RES = os.path.join(HERE, 'results')


def load(name):
    p = os.path.join(RES, name + '.json')
    return json.load(open(p)) if os.path.exists(p) else None


def label(arm, beta):
    return {'onpolicy': 'on-policy', 'filter': 'filter'}.get(arm, 'filter + lock, beta %g' % beta)


def main():
    for exp in ('kuhn', 'leduc', 'ladder'):
        d = load(exp)
        if d is None:
            continue
        keys = list(d['runs'])
        for cls in sorted({k.split('/')[0] for k in keys}):
            series = []
            for k in keys:
                c, arm, b, M = k.split('/')
                if c != cls:
                    continue
                ys = np.mean([r['curve'] for r in d['runs'][k]], axis=0)
                series.append((label(arm, float(b[4:])), list(np.arange(len(ys)) * d['every']), list(ys)))
            plot(os.path.join(RES, 'fig_%s_%s.svg' % (exp, cls)), series, logy=True, xlabel='update',
                 ylabel='V_opt - V' if exp == 'ladder' else 'NashConv',
                 title='%s, %s policy: iterate (mean of %d seeds)' % (exp, cls, d['seeds']))
        series = []
        for cls in sorted({k.split('/')[0] for k in keys}):
            xs, ys = [], []
            for k in keys:
                c, arm, b, M = k.split('/')
                if c == cls and arm != 'onpolicy':
                    xs.append(max(float(b[4:]), 0.1))
                    ys.append(np.mean([np.mean(r['seg_kl'][1:]) for r in d['runs'][k]]))
            series.append((cls, xs, ys))
        plot(os.path.join(RES, 'fig_drift_%s.svg' % exp), series, logx=True, logy=True,
             xlabel='lock weight beta (0.1 = no lock)', ylabel='KL drift per segment',
             title='%s: drift of the agreeing region from the teacher' % exp)
    d = load('segment')
    if d is not None:
        series = []
        for cls in ('hashed', 'tokens'):
            for arm, b in (('filter', 'beta0'), ('filter_lock', 'beta2')):
                xs, ys = [], []
                for M in (25, 100, 400):
                    k = '%s/%s/%s/M%d' % (cls, arm, b, M)
                    if k in d['runs']:
                        xs.append(M)
                        ys.append(np.mean([np.mean(r['seg_kl'][1:]) for r in d['runs'][k]]))
                series.append(('%s, %s' % (cls, 'lock' if arm == 'filter_lock' else 'no lock'), xs, ys))
        plot(os.path.join(RES, 'fig_segment.svg'), series, logx=True, logy=True, xlabel='segment length M',
             ylabel='KL drift per segment', title='Leduc: drift against segment length')


if __name__ == '__main__':
    main()
