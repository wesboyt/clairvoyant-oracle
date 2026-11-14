"""Exactness of the filter / lock machinery behind FILTER_PAPER.md: python -m pytest -q tests/test_filter.py"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, 'tacking', 'src'), ROOT]
pytest.importorskip('scipy')
from polar.games import build  # noqa: E402
from filtering.filterlock import FilterLock, decompose, kl_per_key, oracle_fields, poker_tokens  # noqa: E402


@pytest.fixture(scope='module')
def kuhn():
    return build('Kuhn')


@pytest.mark.parametrize('kind', ['tabular', 'hashed', 'tokens'])
@pytest.mark.parametrize('arm', ['onpolicy', 'filter', 'filter_lock'])
def test_the_performance_difference_split_is_exact(kuhn, kind, arm):
    fl = FilterLock(kuhn, arm=arm, kind=kind, buckets=4, tokenizer=poker_tokens, n_u=0, n_f=3, lr=0.5, M=10, seed=3)
    for _ in range(25):
        pis, flds, info = fl.step()
        after = fl.policies()
        for i in range(2):
            P = np.zeros(kuhn.plex[i].K, bool)
            P[info[i]['piv']] = True
            r = decompose(kuhn, i, pis, after, flds, dict(p=P, rest=~P))
            assert r['total'] == pytest.approx(r['check'], abs=1e-12)
            assert r['p'] + r['rest'] == pytest.approx(r['check'], abs=1e-12)


def test_disagreement_bounds_the_local_regret_and_equals_it_at_last_decisions(kuhn):
    rng = np.random.default_rng(0)
    pis = [pl.softmax(rng.normal(size=pl.n)) for pl in kuhn.plex]
    flds, _ = oracle_fields(kuhn, pis)
    for i, pl in enumerate(kuhn.plex):
        local = np.maximum.reduceat(flds[i]['Qpi'][1:], pl.start[:-1] - 1) - flds[i]['V']
        assert np.all(flds[i]['d'] >= local - 1e-12)                      # Lemma 1: l <= d
        for k, key in enumerate(pl.key_list):
            if (i == 0 and key[1] == 'cb') or i == 1:                     # no later decision of the player
                assert flds[i]['d'][k] == pytest.approx(local[k], abs=1e-12)


def test_the_gap_is_the_sum_of_first_decisions_disagreement(kuhn):
    """Lemma 1: G_i = sum over the player's first decisions of R(k) d(k)."""
    rng = np.random.default_rng(1)
    pis = [pl.softmax(rng.normal(size=pl.n)) for pl in kuhn.plex]
    flds, xs = oracle_fields(kuhn, pis)
    for i, pl in enumerate(kuhn.plex):
        R = kuhn.cf_reach(xs, i)
        first = [k for k in range(pl.K) if pl.parents[k] == {0}]
        assert flds[i]['gap'] == pytest.approx(sum(R[k] * flds[i]['d'][k] for k in first), abs=1e-12)


def test_the_variance_bound_at_agreeing_decisions(kuhn):
    """Lemma 3: Var_pi Q^pi(k, .) <= 2 span_k l(k), and l(k) <= d(k)."""
    rng = np.random.default_rng(2)
    for _ in range(20):
        pis = [pl.softmax(rng.normal(size=pl.n) * 2) for pl in kuhn.plex]
        flds, _ = oracle_fields(kuhn, pis)
        for i, pl in enumerate(kuhn.plex):
            for k in range(pl.K):
                s0, s1 = pl.start[k], pl.start[k + 1]
                q, p = flds[i]['Qpi'][s0:s1], pis[i][s0:s1]
                v = p @ q
                assert p @ (q - v) ** 2 <= 2 * (q.max() - q.min()) * (q.max() - v) + 1e-12
                assert q.max() - v <= flds[i]['d'][k] + 1e-12


def test_tabular_steps_do_not_spill(kuhn):
    fl = FilterLock(kuhn, arm='filter', kind='tabular', n_u=0, n_f=3, lr=0.5, seed=4)
    for _ in range(10):
        pis, _, info = fl.step()
        after = fl.policies()
        for i, pl in enumerate(kuhn.plex):
            moved = np.add.reduceat(np.abs(after[i] - pis[i])[1:], pl.start[:-1] - 1) > 1e-15
            assert set(np.flatnonzero(moved)) <= set(info[i]['piv'].tolist())


def test_token_steps_spill_and_the_spill_law_is_exact(kuhn):
    """Theorem 1: a step e += lr Phi' g moves the logits at every decision by lr K g, K = Phi Phi'."""
    fl = FilterLock(kuhn, arm='filter', kind='tokens', tokenizer=poker_tokens, n_u=0, n_f=3, lr=0.5, seed=5)
    pol = fl.pols[0]
    z0 = pol.logits()
    g = np.zeros(kuhn.plex[0].n)
    g[1:3] = [0.3, -0.3]                                                   # a step at the first decision only
    pol.ascend(g, 0.5)
    dz = pol.logits() - z0
    K = (pol.Phi @ pol.Phi.T).toarray()
    assert np.allclose(dz[1:], 0.5 * (K @ g)[1:])
    assert np.abs(dz[3:]).max() > 0                                        # other decisions moved: spill


def test_the_lock_has_no_gradient_at_the_teacher_and_the_drift_record_starts_at_zero(kuhn):
    fl = FilterLock(kuhn, arm='filter_lock', kind='tokens', tokenizer=poker_tokens, n_u=0, n_f=3, lr=0.5, M=5,
                    seed=1)
    for i, pol in enumerate(fl.pols):
        assert np.allclose(pol.pi(), pol.pi(fl.teacher[i]))
        assert np.allclose(kl_per_key(kuhn.plex[i], fl.seg0[i]['pi'], pol.pi()), 0.0)
    for _ in range(5):
        fl.step()
    assert len(fl.segments) == 2 and all(s['kl'] >= 0 for s in fl.segments)


def test_a_larger_lock_weight_keeps_the_agreeing_region_closer_to_the_teacher(kuhn):
    """Theorem 2 in its simplest regime (tabular, where the descent condition holds for these steps)."""
    drift = []
    for beta in (0.0, 2.0, 8.0):
        fl = FilterLock(kuhn, arm='filter_lock' if beta else 'filter', kind='tabular', n_u=0, n_f=3, lr=0.5,
                        beta=beta, M=50, seed=6)
        for _ in range(500):
            fl.step()
        drift.append(np.mean([s['kl'] for s in fl.segments[2:]]))
    assert drift[0] > drift[1] > drift[2]
