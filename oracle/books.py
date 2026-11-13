"""Replay klob1 files of every series into one time grid shared by all markets: the Tape.

Layer j is the instant t_j = t0 + j * step (j = 0..J). For each market alive at t_j, a *cell* (j, m) records

    the book at t_j (after every record with time <= t_j), top K levels per side, in YES terms:
        bid_px/bid_sz   YES bids, best (highest) first
        ask_px/ask_sz   YES offers (= NO bids at 1 - p), best (lowest) first
    the prints in (t_j, t_{j+1}], which a resting maker order placed at t_j could have been:
        mb_px/mb_sz     taker BOUGHT YES at p: a maker could have SOLD YES (or bought NO at 1 - p) there;
                        best (highest p) first
        ms_px/ms_sz     taker SOLD YES at p: a maker could have BOUGHT YES at p (or sold NO at 1 - p) there;
                        best (lowest p) first

Prices are dollars, sizes contracts. A zero size marks an empty level. A market is alive at t_j from the first
instant its book is known until its close time (trading stops at close_time) or the window's end. Cells are stored
flat, sorted by layer: layer j's cells are rows ptr[j]:ptr[j+1], and nxt[c] is the same market's cell at layer j+1
(-1 when the market is no longer alive at j+1: it closed, or the window ended).

Settlement (clairvoyant: the recorded result is known) comes from settlements.jsonl: result 1 (yes) / 0 (no) /
nan (not in the bundle), and the instant the cash is paid (settlement_ts, else close_time).
"""
import glob
import json
import os
from datetime import datetime, timezone

import numpy as np

from . import klob

K = 5
HOUR_US = 3_600_000_000


def iso_us(s):
    if not s:
        return None
    return int(datetime.fromisoformat(s.replace('Z', '+00:00')).timestamp() * 1_000_000)


def load_settlements(root):
    out = {}
    for path in glob.glob(os.path.join(root, '*', 'settlements.jsonl')):
        with open(path, encoding='utf-8') as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                res = {'yes': 1.0, 'no': 0.0}.get(r.get('result'), np.nan)
                out[r['ticker']] = dict(result=res,
                                        settle_us=iso_us(r.get('settlement_ts')) or iso_us(r.get('close_time')),
                                        close_us=iso_us(r.get('close_time')))
    return out


def _levels4(bids, asks):
    """[4, K] float32 rows: bid px, bid sz (highest first), ask px, ask sz (lowest first)."""
    out = np.zeros((4, K), np.float32)
    for i, (p, q) in enumerate(sorted(bids.items(), reverse=True)[:K]):
        out[0, i], out[1, i] = p / klob.PRICE_ONE, q / klob.SIZE_ONE
    for i, (p, q) in enumerate(sorted(asks.items())[:K]):
        out[2, i], out[3, i] = p / klob.PRICE_ONE, q / klob.SIZE_ONE
    return out


_EMPTY4 = np.zeros((4, K), np.float32)


class _Mkt:
    __slots__ = ('bids', 'asks', 'known', 'close_us', 'closed', 'buys', 'sells', 'dirty', 'top', 'last')

    def __init__(self, close_us):
        self.bids, self.asks = {}, {}
        self.known = False
        self.close_us = close_us
        self.closed = False
        self.buys, self.sells = {}, {}
        self.dirty = True
        self.top = None
        self.last = None       # index of this market's latest row, which collects the prints that follow it

    def book(self):
        if self.dirty or self.top is None:
            self.top = _levels4(self.bids, self.asks)
            self.dirty = False
        return self.top


def files_for(d, t0_us, t1_us):
    """klob files of one series whose hour overlaps [t0, t1] (named SERIES_YYYYMMDD-HHMM[_n].klob.gz)."""
    out = []
    for f in glob.glob(os.path.join(d, '**', '*.klob.gz'), recursive=True):
        stamp = os.path.basename(f)[:-len('.klob.gz')].split('_')[1]
        try:
            start = int(datetime.strptime(stamp, '%Y%m%d-%H%M').replace(tzinfo=timezone.utc).timestamp() * 1e6)
        except ValueError:
            continue
        if start <= t1_us and start + HOUR_US > t0_us:
            out.append((start, f))
    return [f for _, f in sorted(out)]


class Tape:
    """The grid. Attributes: t0, step, J, times[J+1] (us), tickers[M], series[M], result[M], settle_us[M],
    close_us[M], and the flat cell arrays layer, mkt, nxt [C], ptr [J+2], bid_px, bid_sz, ask_px, ask_sz, mb_px,
    mb_sz, ms_px, ms_sz [C, K]."""

    def save(self, path):
        np.savez_compressed(path, **self.__dict__)

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        t = cls()
        for k in z.files:
            v = z[k]
            t.__dict__[k] = v.item() if v.ndim == 0 else v
        return t

    def describe(self):
        alive = np.diff(self.ptr[:-1])
        return ('%d layers x %g s, %d markets in %d series, %d cells (%.0f alive per layer on average)'
                % (self.J, self.step / 1e6, len(self.tickers), len(set(self.series.tolist())), len(self.mkt),
                   alive.mean()))


def build(root, t0_us, t1_us, step_us, series=None):
    """Replay root/<series>/**/*.klob.gz over [t0, t1] on a grid of step_us. series: optional list to keep."""
    J = int((t1_us - t0_us) // step_us)
    times = t0_us + step_us * np.arange(J + 1, dtype=np.int64)
    settle = load_settlements(root)
    dirs = sorted(d for d in glob.glob(os.path.join(root, '*')) if os.path.isdir(d))
    if series:
        keep = set(series)
        dirs = [d for d in dirs if os.path.basename(d) in keep]
    rows = []              # [j, ticker, series, book4, prints4]
    for d in dirs:
        name = os.path.basename(d)
        mk = {}
        state = {'j': 0}   # next layer to emit

        def emit_until(t):
            j = state['j']
            while j <= J and times[j] < t:
                tj = times[j]
                for tk, m in mk.items():
                    if m.last is not None and (m.buys or m.sells):
                        rows[m.last][4] = _levels4(m.buys, m.sells)   # prints in (t_{j-1}, t_j] -> row j-1
                    m.buys, m.sells, m.last = {}, {}, None
                    if m.known and not m.closed and (m.close_us is None or tj < m.close_us):
                        m.last = len(rows)
                        rows.append([j, tk, name, m.book(), _EMPTY4])
                j += 1
            state['j'] = j

        for f in files_for(d, t0_us, t1_us):
            _, it = klob.read(f)
            for tk, r in it:
                tag = r[0]
                if tag == klob.MARKETDEF:
                    if tk not in mk:
                        mk[tk] = _Mkt(settle.get(tk, {}).get('close_us') or iso_us(r[1].get('close_time')))
                    continue
                t = r[1]
                if t > times[-1]:
                    break
                if state['j'] <= J and times[state['j']] < t:
                    emit_until(t)
                m = mk.get(tk)
                if m is None:
                    continue
                if tag == klob.BID:
                    if r[3]:
                        m.bids[r[2]] = r[3]
                    else:
                        m.bids.pop(r[2], None)
                    m.dirty = True
                elif tag == klob.ASK:
                    if r[3]:
                        m.asks[r[2]] = r[3]
                    else:
                        m.asks.pop(r[2], None)
                    m.dirty = True
                elif tag == klob.BUY:
                    m.buys[r[2]] = m.buys.get(r[2], 0) + r[3]
                elif tag == klob.SELL:
                    m.sells[r[2]] = m.sells.get(r[2], 0) + r[3]
                elif tag == klob.SNAPSHOT:
                    m.bids = {p: q for p, q in r[3] if q}
                    m.asks = {p: q for p, q in r[4] if q}
                    m.known, m.dirty = True, True
                elif tag == klob.GAP:
                    m.known, m.dirty = False, True
                    m.bids, m.asks = {}, {}
                    if r[2] == klob.GAP_CLOSED:
                        m.closed = True
        emit_until(times[-1] + 1)
        # prints after t_J (still in buys/sells) are outside the window and are dropped
    return _assemble(rows, times, step_us, settle)


def _assemble(rows, times, step_us, settle):
    J = len(times) - 1
    tickers = sorted({r[1] for r in rows})
    tid = {t: i for i, t in enumerate(tickers)}
    ser = {r[1]: r[2] for r in rows}
    rows.sort(key=lambda r: (r[0], tid[r[1]]))
    C = len(rows)
    layer = np.fromiter((r[0] for r in rows), np.int32, C)
    mkt = np.fromiter((tid[r[1]] for r in rows), np.int32, C)
    book = np.stack([r[3] for r in rows]) if C else np.zeros((0, 4, K), np.float32)
    prn = np.stack([r[4] for r in rows]) if C else np.zeros((0, 4, K), np.float32)
    ptr = np.searchsorted(layer, np.arange(J + 2)).astype(np.int64)
    order = np.lexsort((layer, mkt))                    # by market, then layer
    nxt = np.full(C, -1, np.int64)
    same = (mkt[order[1:]] == mkt[order[:-1]]) & (layer[order[1:]] == layer[order[:-1]] + 1)
    nxt[order[:-1][same]] = order[1:][same]
    tape = Tape()
    tape.t0, tape.step, tape.J = int(times[0]), int(step_us), int(J)
    tape.times = times
    tape.tickers = np.array(tickers)
    tape.series = np.array([ser[x] for x in tickers])
    tape.result = np.array([settle.get(x, {}).get('result', np.nan) for x in tickers], np.float64)
    tape.settle_us = np.array([settle.get(x, {}).get('settle_us') or -1 for x in tickers], np.int64)
    tape.close_us = np.array([settle.get(x, {}).get('close_us') or -1 for x in tickers], np.int64)
    tape.layer, tape.mkt, tape.nxt, tape.ptr = layer, mkt, nxt, ptr
    tape.bid_px, tape.bid_sz, tape.ask_px, tape.ask_sz = (np.ascontiguousarray(book[:, i]) for i in range(4))
    tape.mb_px, tape.mb_sz, tape.ms_px, tape.ms_sz = (np.ascontiguousarray(prn[:, i]) for i in range(4))
    return tape
