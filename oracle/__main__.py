"""Command line.

    python -m oracle build  --start 2026-11-12T14:00:00Z --end 2026-11-12T17:00:00Z [--step 1] [--out tape.npz]
    python -m oracle regret --tape tape.npz [--taker-only] [--report results/report.json]
    python -m oracle node   --tape tape.npz --at 2026-11-12T15:00:00Z [--stake 1] [--top 12] [--taker-only]
    python -m oracle demo   --tape tape.npz --at 2026-11-12T15:00:00Z --minutes 5 --out docs/data/w1
"""
import argparse
import json
import os
import sys
import time

import numpy as np

from . import books, lattice as LT, regret as R

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, 'data', 'kalshi')


def _layer(tape, at):
    return int(np.clip((books.iso_us(at) - tape.t0) // tape.step, 0, tape.J - 1))


def cmd_build(a):
    t = time.time()
    tape = books.build(a.data, books.iso_us(a.start), books.iso_us(a.end), int(a.step * 1e6), a.series)
    tape.save(a.out)
    print('%s  (%.1f s) -> %s' % (tape.describe(), time.time() - t, a.out))


def summarize(tape, maker):
    tab, V, L = R.relative_regret_table(tape, maker=maker)
    kinds = tab['kind']
    best = tab['rho'] <= 1e-12
    lay = tab['layer']
    # the clairvoyant's choice per layer (ties: count every tied action)
    per_kind = {LT.KIND_NAMES[k]: dict(actions=int((kinds == k).sum()), optimal=int((best & (kinds == k)).sum()),
                                       median_rho=float(np.median(tab['rho'][kinds == k])) if (kinds == k).any()
                                       else None)
                for k in (LT.WAIT,) + LT.ENTRY_KINDS}
    z = R.zscores(tab)
    entry_best = np.isin(z['kind'], LT.ENTRY_KINDS)
    out = dict(
        tape=tape.describe(), maker_arcs=maker,
        log_value_usd0=float(V[L.usd(0)]),
        decision_nodes=int(len(np.unique(lay))), actions=int(len(kinds)),
        actions_per_node=float(len(kinds) / max(1, len(np.unique(lay)))),
        per_kind=per_kind,
        best_is_entry_share=float(entry_best.mean()) if len(z) else None,
        best_is_outlier_share=float((z['z'] >= 1.0).mean()) if len(z) else None,
        best_is_entry_and_outlier_share=float(((z['z'] >= 1.0) & entry_best).mean()) if len(z) else None,
        rho_quantiles={str(q): float(np.quantile(tab['rho'], q)) for q in (0.01, 0.1, 0.5, 0.9, 0.99)},
    )
    # the bridge: the shared dollar node against each series on its own
    solo = {}
    for s in sorted(set(tape.series.tolist())):
        sub = subset(tape, tape.series == s)
        Vs, _, Ls = R.values(sub, maker=maker)
        solo[s] = float(Vs[Ls.usd(0)])
    out['log_value_usd0_by_series_alone'] = solo
    out['bridge_gain_log'] = out['log_value_usd0'] - max(solo.values()) if solo else None
    return out


def subset(tape, keep_mkt):
    """A tape restricted to the markets in the boolean mask keep_mkt [M]."""
    t = books.Tape()
    t.__dict__.update(tape.__dict__)
    newid = np.cumsum(keep_mkt) - 1
    cm = keep_mkt[tape.mkt]
    old = np.flatnonzero(cm)
    remap = np.full(len(tape.mkt), -1, np.int64)
    remap[old] = np.arange(len(old))
    for k in ('bid_px', 'bid_sz', 'ask_px', 'ask_sz', 'mb_px', 'mb_sz', 'ms_px', 'ms_sz', 'layer'):
        t.__dict__[k] = tape.__dict__[k][old]
    t.mkt = newid[tape.mkt[old]].astype(np.int32)
    nx = tape.nxt[old]
    t.nxt = np.where(nx >= 0, remap[np.maximum(nx, 0)], -1)
    t.ptr = np.searchsorted(t.layer, np.arange(tape.J + 2)).astype(np.int64)
    for k in ('tickers', 'series', 'result', 'settle_us', 'close_us'):
        t.__dict__[k] = tape.__dict__[k][keep_mkt]
    return t


def cmd_regret(a):
    tape = books.Tape.load(a.tape)
    out = {'maker': summarize(tape, True), 'taker_only': summarize(tape, False)}
    if a.taker_only:
        out.pop('maker')
    txt = json.dumps(out, indent=1)
    print(txt)
    if a.report:
        os.makedirs(os.path.dirname(os.path.abspath(a.report)), exist_ok=True)
        with open(a.report, 'w', encoding='utf-8') as fh:
            fh.write(txt + '\n')


def cmd_node(a):
    tape = books.Tape.load(a.tape)
    j = _layer(tape, a.at)
    maker = not a.taker_only
    V, A, L = R.values(tape, maker=maker)
    rows = R.decisions(tape, V, A, L, j, cash=True)
    print('decision node USD_%d (%s), log V = %.6f, %d actions' % (j, a.at, V[L.usd(j)], len(rows)))
    rows = rows[:a.top]
    acts = [('wait',) if r['kind'] == 'wait' else (LT.KIND_NAMES.index(r['kind']), r['cell']) for r in rows]
    nan = np.full(len(rows), np.nan)
    try:
        vE, qE, rhoE = R.capacitated(tape, j, a.stake, acts, maker=maker, horizon=a.horizon)
    except ImportError:
        vE, qE, rhoE = np.nan, nan, nan
    # the same horizon sub-game at a vanishing stake: the exact backward pass
    Vh, Ah, Lh = R.values(tape, maker=maker, trade_end=j + a.horizon)
    qh = {(r['kind'], r['cell']): r['q'] for r in R.decisions(tape, Vh, Ah, Lh, j, cash=True)}
    rho0 = [1.0 - np.exp(qh.get((r['kind'], r['cell']), -np.inf) - Vh[Lh.usd(j)]) for r in rows]
    print('capacitated, trading horizon %d layers: V_E = %.6f per $ at a $%g stake' % (a.horizon, vE, a.stake))
    print('%-12s %-34s %14s %14s %12s %12s' % ('action', 'market', 'rho (window)', 'rho (horizon)', 'Q_E per $',
                                               'rho_E'))
    for r, r0, q, re in zip(rows, rho0, qE, rhoE):
        print('%-12s %-34s %14.6f %14.6f %12.6f %12.6f' % (r['kind'], r['ticker'], r['rho'], r0, q, re))


def cmd_demo(a):
    """Export one window's arcs for the browser demo: a little-endian binary of parallel arrays + JSON metadata."""
    tape = books.Tape.load(a.tape)
    j0 = _layer(tape, a.at)
    j1 = min(tape.J, j0 + int(a.minutes * 60e6 // tape.step))
    A, L = LT.arcs(tape, j0, j1, maker=True)
    used, inv = np.unique(np.concatenate([[L.S, L.T], A['frm'], A['to']]), return_inverse=True)
    n = len(A['frm'])
    frm, to = inv[2:2 + n].astype('<i4'), inv[2 + n:].astype('<i4')
    usd_ids = np.searchsorted(used, L.usd(np.arange(j0, j1 + 1))).astype('<i4')
    cap = np.where(np.isinf(A['cap']), -1.0, A['cap']).astype('<f8')
    gain = A['gain'].astype('<f8')
    kind = A['kind'].astype('<i1')
    cell = A['cell']
    mk = np.where(cell >= 0, tape.mkt[np.maximum(cell, 0)], -1).astype('<i4')
    lay = np.where(cell >= 0, tape.layer[np.maximum(cell, 0)], -1).astype('<i4')
    key = R.node_keys(tape, L, j0, j1)[used] - j0                  # topological key, relative to the window
    key = np.where(np.isinf(key), 1e9, key).astype('<f8')
    o = np.argsort(-key[frm], kind='stable')                      # backward-pass order: tails by descending key
    frm, to, mk, lay, cap, gain, kind = (x[o] for x in (frm, to, mk, lay, cap, gain, kind))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    import gzip
    with gzip.open(a.out + '.bin.gz', 'wb', compresslevel=9) as fh:
        for arr in (frm, to, mk, lay - np.where(lay >= 0, j0, 0), cap, gain, key, kind):
            fh.write(arr.astype(arr.dtype.newbyteorder('<')).tobytes())
    ms = np.unique(mk[mk >= 0])
    V, _, Lw = R.values(tape, j0, j1, maker=True)
    Vt, _, _ = R.values(tape, j0, j1, maker=False)
    meta = dict(
        arcs=int(n), nodes=int(len(used)), source=0, sink=1, layers=int(j1 - j0), step_s=tape.step / 1e6,
        t0=books.datetime.fromtimestamp(tape.times[j0] / 1e6, books.timezone.utc).isoformat(),
        usd_nodes=usd_ids.tolist(), kinds=LT.KIND_NAMES,
        markets={int(m): dict(ticker=str(tape.tickers[m]), series=str(tape.series[m]),
                              result=None if np.isnan(tape.result[m]) else float(tape.result[m])) for m in ms},
        layout='frm i4, to i4, market i4, layer i4 (window-relative), cap f8 (-1 unbounded), gain f8 [arcs]; '
               'key f8 [nodes] (topological, window-relative); kind i1 [arcs]',
        check=dict(log_value_usd=[float(x) for x in V[Lw.usd(np.arange(j0, j1 + 1))]],
                   log_value_usd_taker_only=[float(x) for x in Vt[Lw.usd(np.arange(j0, j1 + 1))]]),
    )
    with open(a.out + '.json', 'w', encoding='utf-8') as fh:
        json.dump(meta, fh, separators=(',', ':'))
    print('wrote %s.bin.gz (%d arcs, %.1f MB) and .json: %d layers from %s, %d markets'
          % (a.out, n, os.path.getsize(a.out + '.bin.gz') / 1e6, j1 - j0, meta['t0'], len(ms)))


def main(argv=None):
    ap = argparse.ArgumentParser(prog='python -m oracle')
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('build', help='replay the bundled klob files into a tape')
    p.add_argument('--data', default=DATA)
    p.add_argument('--start', required=True)
    p.add_argument('--end', required=True)
    p.add_argument('--step', type=float, default=1.0, help='seconds between layers')
    p.add_argument('--series', nargs='*')
    p.add_argument('--out', default='tape.npz')
    p.set_defaults(fn=cmd_build)
    p = sub.add_parser('regret', help='relative regret over every decision node; summary report')
    p.add_argument('--tape', default='tape.npz')
    p.add_argument('--taker-only', action='store_true')
    p.add_argument('--report')
    p.set_defaults(fn=cmd_regret)
    p = sub.add_parser('node', help='one decision node: marginal and capacitated relative regret')
    p.add_argument('--tape', default='tape.npz')
    p.add_argument('--at', required=True)
    p.add_argument('--stake', type=float, default=1.0)
    p.add_argument('--top', type=int, default=12)
    p.add_argument('--horizon', type=int, default=30, help='layers after which the capacitated sub-game stops trading')
    p.add_argument('--taker-only', action='store_true')
    p.set_defaults(fn=cmd_node)
    p = sub.add_parser('demo', help='export a window for the browser demo')
    p.add_argument('--tape', default='tape.npz')
    p.add_argument('--at', required=True)
    p.add_argument('--minutes', type=float, default=5)
    p.add_argument('--out', required=True)
    p.set_defaults(fn=cmd_demo)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == '__main__':
    sys.exit(main())
