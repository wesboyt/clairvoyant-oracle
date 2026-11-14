"""Experiments for FILTER_PAPER.md: the disagreement filter and the KL lock on exactly solvable games.

    python filtering/run.py [--only kuhn leduc ladder ladder_allframes segment] [--seeds 6] [--quick]
-> filtering/results/<experiment>.json (every run) and summary_<experiment>.json

experiments
    kuhn, leduc   policy class (tabular / hashed / tokens) x arm (onpolicy, filter, filter + lock at beta 0.5, 2, 8)
    ladder        the one-player ladder L1 (tabular / hashed), the same arms
    segment       Leduc, filter vs filter + lock (beta 2), segment length M in {25, 100, 400}, tokens and hashed

Per run: NashConv (one player: V_opt - V) of the iterate and of the linear average every 10 updates; per segment
and player the drift of the agreeing region from the segment's teacher (occupancy-weighted KL and TV); every 10
updates the exact performance-difference split of the step (pivots / agreeing / other decisions); and the coupling
density of the policy class (share of decision pairs that share a parameter).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, '..', 'tacking', 'src'), os.path.join(HERE, '..')]
from polar.games import build  # noqa: E402
from filtering.filterlock import FilterLock, decompose, poker_tokens  # noqa: E402

ROUND2 = lambda k: k[1] is not None            # noqa: E731  Leduc: decisions after the public card
LAST_FRAME = lambda k: k[1][0] == 3            # noqa: E731  L1: decisions in the last frame
ARMS = [('onpolicy', 0.0), ('filter', 0.0), ('filter_lock', 0.5), ('filter_lock', 2.0), ('filter_lock', 8.0)]

EXPERIMENTS = {
    'kuhn': dict(game='Kuhn', T=3000, kinds=['tabular', 'hashed', 'tokens'], buckets=4, filtered=None,
                 kw=dict(n_u=0, n_f=3, lr=0.5), arms=ARMS, Ms=[100]),
    'leduc': dict(game='Leduc', T=3000, kinds=['tabular', 'hashed', 'tokens'], buckets=40, filtered=ROUND2,
                  kw=dict(n_u=8, n_f=16, lr=1.0), arms=ARMS, Ms=[100]),
    'ladder': dict(game='L1', T=1500, kinds=['tabular', 'hashed'], buckets=600, filtered=LAST_FRAME,
                   kw=dict(n_u=8, n_f=8, lr=1.0), arms=ARMS, Ms=[100]),
    # every ladder decision filtered: an early decision's oracle gap includes its whole subtree (Lemma 2)
    'ladder_allframes': dict(game='L1', T=1500, kinds=['tabular', 'hashed'], buckets=600, filtered=None,
                             kw=dict(n_u=0, n_f=16, lr=1.0), arms=[('onpolicy', 0.0), ('filter', 0.0)], Ms=[100]),
    'segment': dict(game='Leduc', T=2000, kinds=['hashed', 'tokens'], buckets=40, filtered=ROUND2,
                    kw=dict(n_u=8, n_f=16, lr=1.0), arms=[('filter', 0.0), ('filter_lock', 2.0)], Ms=[25, 100, 400],
                    seeds=4),
}
EVERY = 10


def coupling_density(fl):
    """Share of ordered pairs of distinct decisions whose logits share at least one parameter."""
    out = []
    for pol, pl in zip(fl.pols, fl.game.plex):
        S = sp.csr_matrix((np.ones(pl.n - 1), (pl.key_of_seq[1:], np.arange(1, pl.n))), shape=(pl.K, pl.n))
        D = S @ abs(pol.Phi)                                     # decision -> every parameter behind any of its logits
        Kk = (D @ D.T).toarray() != 0
        np.fill_diagonal(Kk, False)
        out.append(float(Kk.mean()))
    return float(np.mean(out))


def run_one(game, cfg, kind, arm, beta, M, seed, T):
    fl = FilterLock(game, arm=arm, kind=kind, buckets=cfg['buckets'], tokenizer=poker_tokens, beta=beta, M=M,
                    filtered=cfg['filtered'], seed=seed, **cfg['kw'])
    curve, curve_avg, split = [], [], []
    for t in range(T):
        pis, flds, info = fl.step()
        if t % EVERY == 0 or t == T - 1:
            curve.append(fl.game.nashconv(fl.game.realise(pis)))
            curve_avg.append(fl.nashconv_avg())
            after = fl.policies()
            for i, inf in enumerate(info):
                K = fl.game.plex[i].K
                P = np.zeros(K, bool)
                P[inf['piv']] = True
                d, F = flds[i]['d'], fl.filtered[i]
                bar = inf['bar'] if np.isfinite(inf['bar']) else (
                    float(np.quantile(d[inf['cand']], 1 - fl.f)) if len(inf['cand']) else 0.0)
                agree = F & ~P & (d <= bar)
                r = decompose(fl.game, i, pis, after, flds, dict(pivots=P, agree=agree, other=~P & ~agree))
                split.append([r[k] for k in ('total', 'check', 'pivots', 'agree', 'other', 'tv_agree', 'tv_pivots')])
    split = np.array(split)
    seg = fl.segments
    return dict(curve=curve, curve_avg=curve_avg,
                seg_kl=[s['kl'] for s in seg], seg_tv=[s['tv'] for s in seg], seg_gap=[s['gap'] for s in seg],
                split_mean=split.mean(axis=0).tolist(), agree_neg=float(np.mean(split[:, 3] < -1e-12)),
                pdl_err=float(np.max(np.abs(split[:, 0] - split[:, 1]))), density=coupling_density(fl))


def ms(x):
    x = np.asarray(x, float)
    return [float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0]


def aggregate(runs):
    tail = lambda c: np.mean(c[-max(1, len(c) // 5):])     # noqa: E731  last fifth of the run
    return dict(
        final=ms([r['curve'][-1] for r in runs]), final_avg=ms([r['curve_avg'][-1] for r in runs]),
        tail=ms([tail(r['curve']) for r in runs]),
        seg_kl=ms([np.mean(r['seg_kl'][1:]) for r in runs]), seg_tv=ms([np.mean(r['seg_tv'][1:]) for r in runs]),
        seg_kl_over_gap=ms([np.mean(np.array(r['seg_kl'][1:]) / np.maximum(np.array(r['seg_gap'][1:]), 1e-12))
                            for r in runs]),
        pivots=ms([r['split_mean'][2] for r in runs]), agree=ms([r['split_mean'][3] for r in runs]),
        tv_agree=ms([r['split_mean'][5] for r in runs]), tv_pivots=ms([r['split_mean'][6] for r in runs]),
        agree_neg=ms([r['agree_neg'] for r in runs]), pdl_err=float(max(r['pdl_err'] for r in runs)),
        density=runs[0]['density'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', nargs='*')
    ap.add_argument('--seeds', type=int, default=6)
    ap.add_argument('--quick', action='store_true')
    a = ap.parse_args()
    out_dir = os.path.join(HERE, 'results')
    os.makedirs(out_dir, exist_ok=True)
    for name, cfg in EXPERIMENTS.items():
        if a.only and name not in a.only:
            continue
        game = build(cfg['game'])
        T = cfg['T'] // (10 if a.quick else 1)
        seeds = cfg.get('seeds', a.seeds)
        res = {}
        for kind in cfg['kinds']:
            for M in cfg['Ms']:
                for arm, beta in cfg['arms']:
                    key = '%s/%s/beta%g/M%d' % (kind, arm, beta, M)
                    runs = []
                    for seed in range(seeds):
                        t0 = time.time()
                        runs.append(run_one(game, cfg, kind, arm, beta, M, seed, T))
                        print('%-8s %-34s seed %d  final %.4f  avg %.4f  segKL %.5f  (%.0fs)' % (
                            name, key, seed, runs[-1]['curve'][-1], runs[-1]['curve_avg'][-1],
                            np.mean(runs[-1]['seg_kl'][1:]) if len(runs[-1]['seg_kl']) > 1 else np.nan,
                            time.time() - t0), flush=True)
                    res[key] = runs
        with open(os.path.join(out_dir, name + '.json'), 'w') as fh:
            json.dump(dict(T=T, seeds=seeds, every=EVERY, runs=res), fh)
        with open(os.path.join(out_dir, 'summary_%s.json' % name), 'w') as fh:
            json.dump({k: aggregate(v) for k, v in res.items()}, fh, indent=1)


if __name__ == '__main__':
    main()
