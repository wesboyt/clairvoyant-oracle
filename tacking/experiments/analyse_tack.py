"""Tables and figures for E5 (results/tack/*.json) -> results/fig_tack_*.svg, results/tack_summary.json."""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'src'))
from polar.svgplot import plot, quiver  # noqa: E402

RES = os.path.join(HERE, '..', 'results')
ARMS = ('south', 'north', 'tack', 'tackall', 'vswap')


def main():
    table = {}
    for f in sorted(glob.glob(os.path.join(RES, 'tack', '*.json'))):
        r = json.load(open(f))
        game = r['game']
        table[game] = {}
        ser = []
        for arm in ARMS:
            if arm not in r:
                continue
            s = r[arm]['summary']
            tk = r[arm]['tacks']
            s['tack_dL_mean'] = float(np.mean([t['dL_tack'] for t in tk])) if tk else None
            s['south_leg_dL_mean'] = float(np.mean([t['dL_south_after'] for t in tk])) if tk else None
            table[game][arm] = s
            Lc = np.maximum(np.array(r[arm]['curves']['L']), 1e-9)
            tt = r[arm]['curves'].get('t_index', list(range(len(Lc))))
            idx = np.unique(np.linspace(0, len(Lc) - 1, 400).astype(int))
            ser.append((arm, [tt[i] for i in idx], Lc[idx].tolist()))
        plot(os.path.join(RES, 'fig_tack_%s.svg' % game), ser,
             '%s: Lyapunov L (NashConv or V_opt - V) of the iterate' % game, 'update t', 'L (floored 1e-9)',
             logy=True)
        if 'grid' in r:
            trajs = []
            for arm in ('south', 'north', 'tack'):
                d = np.array(r[arm]['curves']['door'])
                trajs.append(('traj ' + arm, d[:, 0].tolist(), d[:, 1:].mean(1).tolist()))
            quiver(os.path.join(RES, 'fig_doortrap_fields.svg'),
                   [('south field (the wind)', r['grid']['south']), ('north field', r['grid']['north']),
                    ('tack (switched)', r['grid']['tack'])], trajs,
                   'DoorTrap: the fields on (pi(door), pi(good)); optimum at (1, 1)', 'pi(door) at the root',
                   'pi(good) behind the door')
    json.dump(table, open(os.path.join(RES, 'tack_summary.json'), 'w'), indent=1)
    print('%-9s %-8s %9s %9s %9s %6s %5s %8s %8s %6s %6s %9s %9s' % (
        'game', 'arm', 'L0', 'L_final', 'LA_final', 'flagd', 'tacks', 'H0', 'H_final', 'cosS', 'cosN', 'dL/tack',
        'dL/south'))
    for game, rows in table.items():
        for arm, s in rows.items():
            f = lambda v: ('%9.3g' % v) if v is not None else '%9s' % '-'  # noqa: E731
            print('%-9s %-8s %s %s %s %6d %5d %8.3g %8.3g %6.2f %6.2f %s %s' % (
                game, arm, f(s['L0']), f(s['L_final']), f(s['LA_final']), s['flagged_steps'], s['n_tacks'],
                s['H0'], s['H_final'], s['cosS_mean'], s['cosN_mean'], f(s['tack_dL_mean']),
                f(s['south_leg_dL_mean'])))


if __name__ == '__main__':
    main()
