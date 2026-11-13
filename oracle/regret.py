"""Exact relative regret at every decision node of the lattice.

Two oracles, both exact for the game oracle.lattice defines:

1. **Marginal (an infinitesimal stake).** With a stake small enough that no capacity binds, generalized max flow
   from a node is the best product of gains over its paths to the sink (the zero-load node dual). The lattice is a
   DAG, so one backward pass gives every node's value, in logs because clairvoyant compounding overflows floats:
       V(v) = max over arcs v->w of  log gain + V(w),     V(T) = 0
   An action a at decision node v is an outgoing arc group (kind, cell); its value is Q(v,a) = log gain_a + V(w_a)
   (best level), and its **relative regret** is
       rho(v,a) = 1 - exp(Q(v,a) - V(v))  in [0, 1]:
   the fraction of the clairvoyant's value at v that a forfeits by acting a now and playing perfectly after.
   rho = 0 for the clairvoyant's choice; it is scale-free, so it compares nodes whose values differ by orders of
   magnitude, and the offset every sibling shares (the value of perfect play later) cancels in it.

2. **Capacitated (a real stake).** gmaxflow on the window's arrays with `stake` dollars entering at USD_j and the
   first move forced (oracle.lattice.problem): V_E = maxflow(free)/E, Q_E(a) = maxflow(forced a)/E,
   rho_E(a) = 1 - Q_E(a) / max_b Q_E(b). This accounts for depth: compounding stops where the book runs out.

As the stake goes to 0 the capacitated oracle converges to the marginal one (tests/test_oracle.py checks it).
"""
import numpy as np

from . import lattice as LT

NEG = -np.inf


def node_keys(tape, L, j0, j1):
    """A topological key per node: every arc goes from a lower key to a strictly higher one."""
    key = np.full(L.num_nodes, np.inf)
    js = np.arange(j0, j1 + 1)
    key[L.usd(js)] = js
    key[L.yes(np.arange(L.C))] = tape.layer
    key[L.no(np.arange(L.C))] = tape.layer
    last = np.full(L.M, -1.0)
    np.maximum.at(last, tape.mkt, np.minimum(tape.layer, j1).astype(float))
    key[L.yes_end(np.arange(L.M))] = last + 0.5
    key[L.no_end(np.arange(L.M))] = last + 0.5
    key[L.T] = np.inf
    return key


def values(tape, j0=0, j1=None, maker=True, trade_end=None):
    """Backward pass. -> (V [nodes] log value, A arcs dict with 'q' = log gain + V(to), L layout)."""
    j1 = tape.J if j1 is None else j1
    A, L = LT.arcs(tape, j0, j1, maker, trade_end)
    key = node_keys(tape, L, j0, j1)
    kf, kt = key[A['frm']], key[A['to']]
    assert np.all(kt > kf), 'the lattice must be a DAG ordered by key'
    lg = np.log(A['gain'])
    V = np.full(L.num_nodes, NEG)
    V[L.T] = 0.0
    order = np.argsort(-kf, kind='stable')
    kf_sorted = kf[order]
    bounds = np.flatnonzero(np.diff(kf_sorted)) + 1
    for g in np.split(order, bounds):
        q = lg[g] + V[A['to'][g]]
        np.maximum.at(V, A['frm'][g], q)
    A['q'] = lg + V[A['to']]
    return V, A, L


def decisions(tape, V, A, L, j, cash=True):
    """The decision table of USD_j (cash=True) or of every position node at layer j (cash=False).
    -> list of rows {node, kind, cell, ticker, level, gain, q, rho}, best level per (node, kind, cell)."""
    if cash:
        sel = A['frm'] == L.usd(j)
    else:
        cells = np.arange(tape.ptr[j], tape.ptr[j + 1])
        nodes = np.concatenate([L.yes(cells), L.no(cells)])
        sel = np.isin(A['frm'], nodes)
    idx = np.flatnonzero(sel)
    best = {}
    for i in idx:
        k = (int(A['frm'][i]), int(A['kind'][i]), int(A['cell'][i]))
        if k not in best or A['q'][i] > A['q'][best[k]]:
            best[k] = i
    rows = []
    for (node, kind, cell), i in best.items():
        q = float(A['q'][i])
        rho = 1.0 - np.exp(q - V[node]) if np.isfinite(V[node]) and np.isfinite(q) else 1.0
        rows.append(dict(node=node, kind=LT.KIND_NAMES[kind], cell=cell,
                         ticker=str(tape.tickers[tape.mkt[cell]]) if cell >= 0 else '',
                         level=int(A['level'][i]), gain=float(A['gain'][i]), q=q, rho=float(max(rho, 0.0))))
    rows.sort(key=lambda r: (r['node'], r['rho']))
    return rows


def relative_regret_table(tape, j0=0, j1=None, maker=True):
    """Relative regret of every action at every cash decision node USD_j, j0 <= j < j1, in one pass.
    -> dict of parallel arrays: layer, kind, cell, q, v (node value), rho."""
    V, A, L = values(tape, j0, j1, maker)
    j1 = tape.J if j1 is None else j1
    usd_frm = (A['frm'] >= L.usd0) & (A['frm'] < L.usd0 + tape.J + 1)
    i = np.flatnonzero(usd_frm & (A['frm'] < L.usd(j1)))
    layer = A['frm'][i] - L.usd0
    kind, cell, q = A['kind'][i], A['cell'][i], A['q'][i]
    # best level per (layer, kind, cell)
    o = np.lexsort((-q, cell, kind, layer))
    layer, kind, cell, q = layer[o], kind[o], cell[o], q[o]
    first = np.ones(len(o), bool)
    first[1:] = (layer[1:] != layer[:-1]) | (kind[1:] != kind[:-1]) | (cell[1:] != cell[:-1])
    layer, kind, cell, q = layer[first], kind[first], cell[first], q[first]
    v = V[L.usd(layer)]
    with np.errstate(invalid='ignore', over='ignore'):
        rho = np.where(np.isfinite(q), 1.0 - np.exp(q - v), 1.0)
    return dict(layer=layer, kind=kind, cell=cell, q=q, v=v, rho=np.clip(rho, 0.0, 1.0)), V, L


def zscores(table):
    """D23 outlier score of each node's best action: z = (Q(a*) - max_{b != a*} Q(b)) / std_b Q(b), in rho units
    (Q expressed as 1 - rho, so the clairvoyant offset cancels). -> per layer arrays (layer, best kind, z, n)."""
    out = []
    lay = table['layer']
    starts = np.flatnonzero(np.r_[True, lay[1:] != lay[:-1]])
    ends = np.r_[starts[1:], len(lay)]
    for a, b in zip(starts, ends):
        u = 1.0 - table['rho'][a:b]
        if b - a < 2:
            continue
        o = np.argsort(-u)
        s = u.std()
        z = (u[o[0]] - u[o[1]]) / s if s > 0 else 0.0
        out.append((int(lay[a]), int(table['kind'][a:b][o[0]]), float(z), b - a))
    return np.array(out, dtype=[('layer', 'i4'), ('kind', 'i1'), ('z', 'f8'), ('n', 'i4')])


def capacitated(tape, j0, stake, actions, j1=None, solve=None, maker=True, horizon=None):
    """Capacitated values with gmaxflow's array interface. actions: list of ('wait',) or (kind, cell).
    horizon: solve the sub-game in which trading stops `horizon` layers after j0 (positions settle, dollars wait):
    the full-window capacitated problem grows expensive fast, because compounding makes every capacity bind.
    -> (V_E, Q_E [len(actions)], rho_E) in dollars returned per dollar staked."""
    if solve is None:
        import gmaxflow
        solve = gmaxflow.max_flow_arrays
    n, s, t, f, to, cap, g = LT.problem(tape, j0, stake, None, j1, maker, horizon)
    v = solve(n, s, t, f, to, cap, g) / stake
    q = np.empty(len(actions))
    for i, a in enumerate(actions):
        n, s, t, f, to, cap, g = LT.problem(tape, j0, stake, a, j1, maker, horizon)
        q[i] = solve(n, s, t, f, to, cap, g) / stake
    best = max(q.max(), 1e-300)
    return v, q, 1.0 - q / best
