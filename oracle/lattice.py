"""The cross-market lattice: one dollar node per layer, bridging every Kalshi event market.

Nodes
    S, T                source and sink
    USD_j               dollars at layer j (j = 0..J), shared by every market: the bridge
    YES_c, NO_c         contracts held in cell c = (layer j, market m)
    YESend_m, NOend_m   a position still open when market m stops trading: it settles

Arcs (an action taken at layer j fills at the book of t_j, or against a print in (t_j, t_{j+1}], and the contracts
or dollars it produces are usable from layer j+1 on; every arc points forward in time, so the lattice is a DAG)
    take YES   USD_j -> YES_{c'}   per ask level (a, n):  gain 1 / (a + f(a))      cap n (a + f(a)) dollars
    take NO    USD_j -> NO_{c'}    per bid level (b, n):  gain 1 / (1 - b + f(b))  cap n (1 - b + f(b))
    sell YES   YES_c -> USD_{j+1}  per bid level:         gain b - f(b)            cap n contracts
    sell NO    NO_c  -> USD_{j+1}  per ask level:         gain 1 - a - f(a)        cap n
    make YES   USD_j -> YES_{c'}   per print where a taker SOLD YES at p:   gain 1 / p        cap size * p
    make NO    USD_j -> NO_{c'}    per print where a taker BOUGHT YES at p: gain 1 / (1 - p)  cap size * (1 - p)
    make-sell  YES_c -> USD_{j+1} at a taker-buy print p (gain p); NO_c -> USD_{j+1} at a taker-sell print (1 - p)
    hold       USD_j -> USD_{j+1}; YES_c -> YES_{c'}; NO_c -> NO_{c'}     unbounded, gain 1
    settle     YESend_m -> USD_s (or T) gain 1 when the market settled YES; NOend_m likewise when NO
    sink       USD_J -> T gain 1
c' is market m's cell at layer j+1, or YESend_m / NOend_m when m stops trading after layer j. s is the first layer
at or after the settlement instant (T when that is past the window). A market whose result is not in the bundle
has no settle arc: its contracts are worth only what the book pays for them before it stops.

f(P) = ceil(7 P (1 - P)) cents per contract is Kalshi's quadratic taker fee for these series, charged here as for
one-contract orders (an upper bound on a larger order's rounded fee). Makers pay nothing. A maker arc is
clairvoyant: it assumes we were the resting order the print traded against (queue position ignored).

Every arc carries an action label, so the same arrays feed the exact DP (oracle.regret) and gmaxflow.
"""
import numpy as np

# action kinds (also the arc kinds that are decisions)
WAIT, TAKE_YES, TAKE_NO, MAKE_YES, MAKE_NO, HOLD, SELL_YES, SELL_NO, MSELL_YES, MSELL_NO, SETTLE, SINK, SOURCE = \
    range(13)
KIND_NAMES = ['wait', 'take_yes', 'take_no', 'make_yes', 'make_no', 'hold', 'sell_yes', 'sell_no',
              'make_sell_yes', 'make_sell_no', 'settle', 'sink', 'source']
ENTRY_KINDS = (TAKE_YES, TAKE_NO, MAKE_YES, MAKE_NO)


def taker_fee(p):
    """Kalshi quadratic taker fee in dollars per contract at price p (dollars): ceil(7 p (1 - p)) cents."""
    p = np.asarray(p, np.float64)
    return np.ceil(7.0 * p * (1.0 - p) - 1e-9) / 100.0


class Layout:
    """Node numbering for a tape."""

    def __init__(self, tape):
        self.J, self.C, self.M = tape.J, len(tape.mkt), len(tape.tickers)
        self.S, self.T = 0, 1
        self.usd0 = 2
        self.cell0 = self.usd0 + self.J + 1
        self.end0 = self.cell0 + 2 * self.C
        self.F = self.end0 + 2 * self.M
        self.num_nodes = self.F + 1

    def usd(self, j):
        return self.usd0 + np.asarray(j)

    def yes(self, c):
        return self.cell0 + 2 * np.asarray(c)

    def no(self, c):
        return self.cell0 + 2 * np.asarray(c) + 1

    def yes_end(self, m):
        return self.end0 + 2 * np.asarray(m)

    def no_end(self, m):
        return self.end0 + 2 * np.asarray(m) + 1


def settle_layer(tape):
    """[M] layer at which a market's settlement cash arrives (J + 1 = past the window, i.e. at the sink)."""
    s = np.searchsorted(tape.times, tape.settle_us, side='left')
    s = np.where(tape.settle_us < 0, tape.J + 1, s)
    return s.astype(np.int64)


def arcs(tape, j0=0, j1=None, maker=True, trade_end=None):
    """All arcs of the lattice restricted to layers j0..j1 (default: the whole tape), as a dict of parallel arrays
    frm, to, cap (inf = unbounded), gain, kind, cell (-1 if none), level. Cells at layer j1 do not trade; dollars
    at j1 go to the sink, and positions open at j1 settle (or are worthless if their result is unknown).
    maker=False drops the clairvoyant maker arcs (a taker-only lattice). trade_end: no trading at layers >= it
    (positions still hold to settlement and dollars still wait to the sink): the horizon sub-game."""
    L = Layout(tape)
    j1 = tape.J if j1 is None else j1
    lay, mkt, nxt = tape.layer, tape.mkt, tape.nxt
    live = (lay >= j0) & (lay <= j1)
    c_all = np.nonzero(live)[0]
    trade = c_all[lay[c_all] < (j1 if trade_end is None else min(j1, trade_end))]   # cells that can act
    last = nxt.copy()
    last[(nxt >= 0) & (lay >= j1)] = -1                  # positions at j1 go to their end nodes
    yes_next = np.where(last >= 0, L.yes(np.maximum(last, 0)), L.yes_end(mkt))
    no_next = np.where(last >= 0, L.no(np.maximum(last, 0)), L.no_end(mkt))
    out = {k: [] for k in ('frm', 'to', 'cap', 'gain', 'kind', 'cell', 'level')}

    def add(frm, to, cap, gain, kind, cell, level):
        n = len(frm)
        if n == 0:
            return
        out['frm'].append(np.asarray(frm, np.int64))
        out['to'].append(np.asarray(to, np.int64))
        out['cap'].append(np.broadcast_to(np.asarray(cap, np.float64), (n,)))
        out['gain'].append(np.broadcast_to(np.asarray(gain, np.float64), (n,)))
        out['kind'].append(np.full(n, kind, np.int8))
        out['cell'].append(np.broadcast_to(np.asarray(cell, np.int64), (n,)))
        out['level'].append(np.broadcast_to(np.asarray(level, np.int8), (n,)))

    c = trade
    usd_now, usd_next = L.usd(lay[c]), L.usd(lay[c] + 1)
    K = tape.ask_px.shape[1]
    for k in range(K):
        a, na = tape.ask_px[c, k].astype(np.float64), tape.ask_sz[c, k].astype(np.float64)
        b, nb = tape.bid_px[c, k].astype(np.float64), tape.bid_sz[c, k].astype(np.float64)
        fa, fb = taker_fee(a), taker_fee(b)
        m = (na > 0) & (a > 0) & (a < 1)
        add(usd_now[m], yes_next[c][m], na[m] * (a[m] + fa[m]), 1.0 / (a[m] + fa[m]), TAKE_YES, c[m], k)
        g = 1.0 - a - fa
        m2 = m & (g > 0)
        add(L.no(c)[m2], usd_next[m2], na[m2], g[m2], SELL_NO, c[m2], k)
        m = (nb > 0) & (b > 0) & (b < 1)
        add(usd_now[m], no_next[c][m], nb[m] * (1 - b[m] + fb[m]), 1.0 / (1 - b[m] + fb[m]), TAKE_NO, c[m], k)
        g = b - fb
        m2 = m & (g > 0)
        add(L.yes(c)[m2], usd_next[m2], nb[m2], g[m2], SELL_YES, c[m2], k)
        if not maker:
            continue
        # maker arcs from the prints that follow t_j
        p, n = tape.ms_px[c, k].astype(np.float64), tape.ms_sz[c, k].astype(np.float64)   # taker sold YES at p
        m = (n > 0) & (p > 0) & (p < 1)
        add(usd_now[m], yes_next[c][m], n[m] * p[m], 1.0 / p[m], MAKE_YES, c[m], k)
        add(L.no(c)[m], usd_next[m], n[m], 1.0 - p[m], MSELL_NO, c[m], k)
        p, n = tape.mb_px[c, k].astype(np.float64), tape.mb_sz[c, k].astype(np.float64)   # taker bought YES at p
        m = (n > 0) & (p > 0) & (p < 1)
        add(usd_now[m], no_next[c][m], n[m] * (1 - p[m]), 1.0 / (1 - p[m]), MAKE_NO, c[m], k)
        add(L.yes(c)[m], usd_next[m], n[m], p[m], MSELL_YES, c[m], k)
    # holds
    js = np.arange(j0, j1)
    add(L.usd(js), L.usd(js + 1), np.inf, 1.0, WAIT, -1, -1)
    add(L.yes(c_all), yes_next[c_all], np.inf, 1.0, HOLD, c_all, -1)
    add(L.no(c_all), no_next[c_all], np.inf, 1.0, HOLD, c_all, -1)
    # settlement of markets that have cells in the window
    ms = np.unique(mkt[c_all])
    s = settle_layer(tape)[ms]
    dest = np.where(s > j1, L.T, L.usd(np.minimum(s, j1)))
    res = tape.result[ms]
    m = res == 1.0
    add(L.yes_end(ms[m]), dest[m], np.inf, 1.0, SETTLE, -1, -1)
    m = res == 0.0
    add(L.no_end(ms[m]), dest[m], np.inf, 1.0, SETTLE, -1, -1)
    add([L.usd(j1)], [L.T], np.inf, 1.0, SINK, -1, -1)
    return {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in out.items()}, L


def problem(tape, j0, stake, forced=None, j1=None, maker=True, horizon=None):
    """Arrays for gmaxflow: `stake` dollars enter at USD_{j0}. forced = None (free), ('wait',), or
    (kind, cell) for an entry action at layer j0: the whole stake must leave USD_{j0} through that action's arcs
    (all its levels). horizon: trading stops `horizon` layers after j0 (open positions settle, dollars wait).
    -> (num_nodes, S, T, frm, to, cap, gain) with cap < 0 meaning unbounded."""
    A, L = arcs(tape, j0, j1, maker, None if horizon is None else j0 + horizon)
    frm, to, cap, gain = A['frm'].copy(), A['to'].copy(), A['cap'].copy(), A['gain'].copy()
    src = L.usd(j0)
    if forced is None:
        extra = ([L.S], [src])
    elif forced[0] == 'wait':
        extra = ([L.S], [L.usd(j0 + 1)])
    else:
        kind, cell = forced
        sel = (A['kind'] == kind) & (A['cell'] == cell) & (A['frm'] == src)
        if not sel.any():
            raise ValueError('action not available at this node')
        frm = np.concatenate([frm, np.full(sel.sum(), L.F)])
        to = np.concatenate([to, A['to'][sel]])
        cap = np.concatenate([cap, A['cap'][sel]])
        gain = np.concatenate([gain, A['gain'][sel]])
        extra = ([L.S], [L.F])
    frm = np.concatenate([frm, extra[0]])
    to = np.concatenate([to, extra[1]])
    cap = np.concatenate([cap, [stake]])
    gain = np.concatenate([gain, [1.0]])
    cap = np.where(np.isinf(cap), -1.0, cap)
    # compact the node ids to the ones this window touches (S -> 0, T -> 1)
    used, inv = np.unique(np.concatenate([[L.S, L.T], frm, to]), return_inverse=True)
    inv = inv[2:]
    n = len(frm)
    return len(used), 0, 1, inv[:n].astype(np.int32), inv[n:].astype(np.int32), cap, gain
