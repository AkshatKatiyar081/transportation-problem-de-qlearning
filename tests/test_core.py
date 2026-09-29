"""T1 decoder, T2 tier costs."""
import numpy as np
import pytest

from core.decoder import decode
from core.problem import Problem, load_instance


def _rand_problem(m, n, seed):
    rng = np.random.default_rng(seed)
    s = rng.integers(10, 101, m)
    tot = int(s.sum())
    d = 1 + rng.multinomial(tot - n, np.ones(n) / n)
    return Problem("r", rng.integers(1, 101, (m, n)), s, d)


@pytest.mark.parametrize("name", ["A1", "A3", "A5", "L5", "N5", "L10"])
def test_decoder_feasible_integer_basic(name):
    p = load_instance(name).balanced()
    rng = np.random.default_rng(0)
    for _ in range(1000):
        X, unmet = decode(rng.random(p.m * p.n), p)
        assert unmet == 0
        assert (X.sum(1) == p.supply).all() and (X.sum(0) == p.demand).all()
        assert (X >= 0).all() and X.dtype.kind == "i"
        assert (X > 0).sum() <= p.m + p.n - 1


def test_decoder_random_6x8():
    p = _rand_problem(6, 8, 5)
    rng = np.random.default_rng(1)
    for _ in range(1000):
        X, unmet = decode(rng.random(48), p)
        assert unmet == 0 and (X > 0).sum() <= 13


def test_decoder_example_order():
    keys = [0.7, 0.2, 0.9, 0.1, 0.5, 0.3, 0.8, 0.4, 0.6]
    p = load_instance("A1")
    _, _, steps = decode(keys, p, return_steps=True)
    assert [s[0] for s in steps] == [2, 6, 0, 8, 4, 7, 5, 1, 3]


@pytest.mark.parametrize("x,expected", [(0, 0), (10, 10.0), (11, 10.9), (50, 46.0), (51, 46.8),
                                        (60, 54.0), (100, 86.0)])
@pytest.mark.parametrize("c", [1, 7])
def test_incremental_tiers(x, expected, c):
    p = Problem("t", [[c]], [x if x else 1], [x if x else 1],
                discount={"type": "incremental", "tiers": [[10, 1.0], [50, 0.9], [None, 0.8]]})
    assert p.components(np.array([[x]]))["cost"] == pytest.approx(c * expected)


def test_worked_example_1080():
    p = Problem("t", [[20]], [60], [60], discount={"type": "incremental", "tiers": [[10, 1.0], [50, 0.9], [None, 0.8]]})
    assert p.components(np.array([[60]]))["cost"] == pytest.approx(1080)


@pytest.mark.parametrize("x,expected", [(0, 0), (20, 20), (21, 21 * 0.85), (60, 51), (61, 61 * 0.7)])
def test_all_units(x, expected):
    p = Problem("t", [[1]], [max(x, 1)], [max(x, 1)],
                discount={"type": "all_units", "tiers": [[20, 1.0], [60, 0.85], [None, 0.7]]})
    assert p.components(np.array([[x]]))["cost"] == pytest.approx(expected)


def test_fixed_charge_and_constraints():
    p = Problem("t", [[1, 2], [3, 4]], [10, 10], [10, 10], fixed_charge=[[5, 5], [5, 5]],
                capacity=[[8, None], [None, None]], min_ship=[[3, 3], [3, 3]])
    X = np.array([[8, 2], [2, 8]])
    ev = p.evaluate(X)
    assert ev["cost"] == pytest.approx(8 + 4 + 6 + 32 + 20)
    assert ev["violations"]["min_ship"] == 2
    assert not ev["feasible"]


def test_json_roundtrip():
    for name in ["A5", "N5", "X1", "X2", "X3", "X4", "U1"]:
        p = load_instance(name)
        q = Problem.from_json(p.to_json())
        assert np.allclose(p.cost, q.cost) and p.features() == q.features()
