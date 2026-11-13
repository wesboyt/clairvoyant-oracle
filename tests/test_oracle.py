"""Exactness tests: python -m pytest -q tests  (gmaxflow optional; its tests are skipped without it)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from oracle import books, lattice as LT, regret as R  # noqa: E402
from oracle.__main__ import subset  # noqa: E402

K = books.K
T0 = 1_800_000_000_000_000
STEP = 1_000_000


def book(bids=(), asks=()):
    out = np.zeros((4, K), np.float32)
    for i, (p, q) in enumerate(bids):
        out[0, i], out[1, i] = p, q
    for i, (p, q) in enumerate(asks):
        out[2, i], out[3, i] = p, q
    return out


def synthetic(seed=0, layers=8, markets=3, prints=True, results=(1.0, 0.0, np.nan)):
    """A random tape: `markets` binary markets over `layers` layers, books two levels deep, optional prints."""
    rnd = np.random.default_rng(seed)
    rows, settle = [], {}
    for m in range(markets):
        tk = 'M%d' % m
        close = T0 + STEP * (layers - 1 - (m % 3))
        settle[tk] = dict(result=results[m % len(results)], settle_us=close + STEP // 2, close_us=close)
        mid = rnd.uniform(0.2, 0.8)
        for j in range(layers):
            t = T0 + STEP * j
            if t >= close:
                break
            mid = float(np.clip(mid + rnd.normal(0, 0.05), 0.05, 0.95))
            b0, a0 = round(mid - 0.01, 2), round(mid + 0.01, 2)
            bk = book([(b0, rnd.uniform(1, 9)), (b0 - 0.01, rnd.uniform(1, 9))],
                      [(a0, rnd.uniform(1, 9)), (a0 + 0.01, rnd.uniform(1, 9))])
            pr = np.zeros((4, K), np.float32)
            if prints and rnd.random() < 0.5:
                pr = book([(a0, rnd.uniform(1, 5))], [(b0, rnd.uniform(1, 5))])
            rows.append([j, tk, 'S%d' % (m % 2), bk, pr])
    times = T0 + STEP * np.arange(layers + 1, dtype=np.int64)
    return books._assemble(rows, times, STEP, settle)


def test_the_taker_fee_matches_kalshi_quadratic_schedule():
    f = LT.taker_fee([0.5, 0.01, 0.99, 0.25, 0.1])
    assert np.allclose(f, [0.02, 0.01, 0.01, 0.02, 0.01])


@pytest.mark.parametrize('maker', [True, False])
def test_the_lattice_is_a_dag_and_every_gain_is_positive(maker):
    tape = synthetic(1)
    A, L = LT.arcs(tape, maker=maker)
    key = R.node_keys(tape, L, 0, tape.J)
    assert np.all(key[A['to']] > key[A['frm']])
    assert np.all(A['gain'] > 0)


def test_relative_regret_is_zero_at_the_clairvoyants_choice_and_in_unit_range():
    tape = synthetic(2)
    tab, V, L = R.relative_regret_table(tape)
    assert np.all((tab['rho'] >= 0) & (tab['rho'] <= 1))
    for j in np.unique(tab['layer']):
        r = tab['rho'][tab['layer'] == j]
        assert r.min() <= 1e-12          # some action attains the node value


def test_waiting_relative_regret_compares_against_the_next_layer():
    tape = synthetic(3)
    V, A, L = R.values(tape)
    for j in range(tape.J - 1):
        rows = R.decisions(tape, V, A, L, j)
        wait = [r for r in rows if r['kind'] == 'wait'][0]
        assert wait['q'] == pytest.approx(V[L.usd(j + 1)])


def test_a_known_winner_is_bought_by_the_oracle():
    # one market settling YES, ask 0.40 at layer 0: the clairvoyant buys it (fee 2c -> 1/0.42 contracts per $)
    rows = [[0, 'W', 'S', book([(0.38, 10)], [(0.40, 10)]), np.zeros((4, K), np.float32)],
            [1, 'W', 'S', book([(0.38, 10)], [(0.40, 10)]), np.zeros((4, K), np.float32)]]
    times = T0 + STEP * np.arange(3, dtype=np.int64)
    settle = {'W': dict(result=1.0, settle_us=T0 + 3 * STEP, close_us=T0 + 2 * STEP)}
    tape = books._assemble(rows, times, STEP, settle)
    V, A, L = R.values(tape)
    assert np.exp(V[L.usd(0)]) == pytest.approx(1 / 0.42)
    tab = R.decisions(tape, V, A, L, 0)
    best = min(tab, key=lambda r: r['rho'])
    assert best['kind'] == 'take_yes'
    wait = [r for r in tab if r['kind'] == 'wait'][0]
    assert wait['rho'] == pytest.approx(1 - np.exp(V[L.usd(1)] - V[L.usd(0)]))


def test_the_shared_dollar_node_is_worth_at_least_any_market_group_alone():
    tape = synthetic(4, layers=10, markets=4)
    V, _, L = R.values(tape)
    for s in set(tape.series.tolist()):
        sub = subset(tape, tape.series == s)
        Vs, _, Ls = R.values(sub)
        assert V[L.usd(0)] >= Vs[Ls.usd(0)] - 1e-12


@pytest.mark.parametrize('seed', range(6))
@pytest.mark.parametrize('maker', [True, False])
def test_marginal_values_equal_generalized_max_flow_at_a_small_stake(seed, maker):
    gmaxflow = pytest.importorskip('gmaxflow')
    if not hasattr(gmaxflow, 'max_flow_arrays'):
        pytest.skip('gmaxflow without the array interface')
    tape = synthetic(seed, results=(1.0, 0.0, 1.0))
    V, A, L = R.values(tape, maker=maker)
    for j0 in (0, 2):
        stake = 1e-6
        n, s, t, f, to, cap, g = LT.problem(tape, j0, stake, None, None, maker)
        mf = gmaxflow.max_flow_arrays(n, s, t, f, to, cap, g)
        assert mf / stake == pytest.approx(np.exp(V[L.usd(j0)]), rel=1e-9)
        # and every forced first move, against the DP's Q
        for r in R.decisions(tape, V, A, L, j0):
            act = ('wait',) if r['kind'] == 'wait' else (LT.KIND_NAMES.index(r['kind']), r['cell'])
            n, s, t, f, to, cap, g = LT.problem(tape, j0, stake, act, None, maker)
            q = gmaxflow.max_flow_arrays(n, s, t, f, to, cap, g) / stake
            assert q == pytest.approx(np.exp(r['q']), rel=1e-9, abs=1e-12)


def test_capacity_lowers_the_value_per_dollar_as_the_stake_grows():
    gmaxflow = pytest.importorskip('gmaxflow')
    if not hasattr(gmaxflow, 'max_flow_arrays'):
        pytest.skip('gmaxflow without the array interface')
    tape = synthetic(5, results=(1.0, 1.0, 0.0))
    vals = []
    for stake in (1e-6, 1.0, 100.0, 10_000.0):
        n, s, t, f, to, cap, g = LT.problem(tape, 0, stake)
        vals.append(gmaxflow.max_flow_arrays(n, s, t, f, to, cap, g) / stake)
    assert all(a >= b - 1e-9 for a, b in zip(vals, vals[1:]))
    assert vals[-1] >= 1.0 - 1e-9      # holding dollars is always available


def test_the_bundled_hours_replay_when_present():
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'kalshi')
    if not os.path.isdir(root) or not os.listdir(root):
        pytest.skip('no bundled data')
    files = [os.path.join(dp, f) for dp, _, fs in os.walk(root) for f in fs if f.endswith('.klob.gz')]
    assert files
    from oracle import klob
    header, it = klob.read(files[0])
    assert header['format'] == 'klob1'
    assert sum(1 for _ in zip(range(1000), it)) > 0


@pytest.mark.parametrize('seed', range(4))
def test_the_horizon_sub_game_matches_generalized_max_flow(seed):
    gmaxflow = pytest.importorskip('gmaxflow')
    if not hasattr(gmaxflow, 'max_flow_arrays'):
        pytest.skip('gmaxflow without the array interface')
    tape = synthetic(seed, layers=10, results=(1.0, 0.0, 1.0))
    for j0, h in ((0, 3), (2, 4)):
        V, A, L = R.values(tape, j0, None, True, j0 + h)
        n, s, t, f, to, cap, g = LT.problem(tape, j0, 1e-6, None, None, True, h)
        assert gmaxflow.max_flow_arrays(n, s, t, f, to, cap, g) / 1e-6 == pytest.approx(np.exp(V[L.usd(j0)]), rel=1e-9)
        Vfull, _, _ = R.values(tape, j0)
        assert V[L.usd(j0)] <= Vfull[L.usd(j0)] + 1e-12     # stopping trading early never helps

