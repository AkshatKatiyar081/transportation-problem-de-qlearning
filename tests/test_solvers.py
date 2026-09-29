"""T3 MODI optimality (duality certificate), T4/T5 DE vs brute force on discount problems, lab optima."""
import itertools

import numpy as np
import pytest

from algorithms.registry import run
from core.decoder import DecoderCache, decode
from core.problem import Problem, load_instance


@pytest.mark.parametrize("name,opt", [("A1", 3850), ("A3", 149), ("A5", 695)])
def test_lab_optima(name, opt):
    assert run("modi", load_instance(name)).cost == pytest.approx(opt)


def test_a5_plan_check():
    p = load_instance("A5")
    r = run("modi", p, {"initial": p.meta["proposed_plan"], "trace": True})
    assert r.trace[0]["cost"] == 795 and r.cost == 695


def _instances():
    rng = np.random.default_rng(1000)
    out = []
    for k in range(100):
        m, n = [(3, 3), (4, 5), (6, 8)][k % 3]
        s = rng.integers(10, 101, m)
        d = 1 + rng.multinomial(int(s.sum()) - n, np.ones(n) / n)
        out.append(Problem(f"b{k}", rng.integers(1, 101, (m, n)), s, d))
    for k in range(20):   # unbalanced
        s = rng.integers(10, 101, 4)
        d = rng.integers(10, 101, 5)
        out.append(Problem(f"u{k}", rng.integers(1, 101, (4, 5)), s, d))
    for k in range(20):   # degenerate: equal partial sums
        m = n = 3 + k % 3
        s = rng.integers(5, 40, m)
        out.append(Problem(f"d{k}", rng.integers(1, 101, (m, n)), s, s[rng.permutation(m)] if k % 2 else s))
    return out


@pytest.mark.parametrize("p", _instances(), ids=lambda p: p.name)
def test_modi_duality_certificate(p):
    """Independent optimality proof: feasible plan + dual-feasible (u, v) + equal objectives."""
    r = run("modi", p)
    assert r.feasible
    b = p.balanced()
    u, v = r.diagnostics["u"], r.diagnostics["v"]
    assert (b.cost - u[:, None] - v[None, :] >= -1e-7).all()          # dual feasibility
    dual = float(b.supply @ u + b.demand @ v)
    assert abs(dual - r.cost) < 1e-6                                     # strong duality


@pytest.mark.parametrize("seed", range(10))
def test_modi_equals_exhaustive_3x3(seed):
    """For a linear 3x3 problem every basic plan is some decoder order: enumerate all 9! orders."""
    rng = np.random.default_rng(seed)
    s = rng.integers(5, 40, 3)
    d = 1 + rng.multinomial(int(s.sum()) - 3, np.ones(3) / 3)
    p = Problem("e", rng.integers(1, 50, (3, 3)), s, d)
    cache = DecoderCache(p)
    best = min(float((p.cost * decode(np.array(perm, float), p, cache)[0]).sum())
               for perm in itertools.permutations(range(9)))
    assert run("modi", p).cost == pytest.approx(best)


def test_discount_bruteforce_2x2():
    p = Problem("t4", [[5, 10], [8, 3]], [60, 80], [70, 70],
                discount={"type": "incremental", "tiers": [[10, 1.0], [50, 0.9], [None, 0.8]]})
    best = min(p.components(np.array([[x, 60 - x], [70 - x, 10 + x]]))["cost"] for x in range(61))
    assert best == pytest.approx(536.0)
    r = run("hybrid", p, {"fe_budget": 2000}, seed=0)
    assert r.cost == pytest.approx(536.0)
    assert r.matrix.tolist() == [[60, 0], [10, 70]]


@pytest.mark.parametrize("kind", ["incremental", "fixed_charge"])
def test_de_vs_exhaustive_3x3(kind):
    rng = np.random.default_rng(7)
    s = rng.integers(20, 81, 3)
    d = 20 + rng.multinomial(int(s.sum()) - 60, np.ones(3) / 3)
    kw = {"discount": {"type": "incremental", "tiers": [[30, 1.0], [80, 0.7], [None, 0.5]]}}
    if kind != "incremental":
        kw = {"fixed_charge": rng.integers(0, 60, (3, 3))}
    p = Problem("t5", rng.integers(5, 50, (3, 3)), s, d, **kw)
    cache = DecoderCache(p)
    best = min(p.components(decode(np.array(perm[::-1], float), p, cache)[0])["cost"]
               for perm in itertools.permutations(range(9)))
    assert run("hybrid", p, {"fe_budget": 5000}, seed=0).cost == pytest.approx(best)


def test_all_units_polish_bruteforce():
    # all-units optimum may be non-basic: the polish step must find it
    p = Problem("au", [[5, 6], [6, 5]], [30, 30], [30, 30],
                discount={"type": "all_units", "tiers": [[20, 1.0], [None, 0.5]]})
    best = min(p.components(np.array([[x, 30 - x], [30 - x, x]]))["cost"] for x in range(31))
    assert run("hybrid", p, {"fe_budget": 2000, "polish": True}, seed=0).cost == pytest.approx(best)


def test_constraints_example_reaches_best_known():
    p = load_instance("X2")
    r = run("hybrid", p, {"fe_budget": 10000, "polish": True}, seed=0)
    assert r.feasible and r.objective == pytest.approx(p.meta["reference_optimum"])


def test_de_reaches_lab_optima():
    for name, opt in [("A1", 3850), ("A3", 149), ("A5", 695)]:
        r = run("hybrid", load_instance(name), {"fe_budget": 5000}, seed=0)
        assert r.cost == pytest.approx(opt)


def test_unbalanced_dummy():
    r = run("modi", load_instance("U1"))
    assert r.feasible and r.diagnostics["dummy_units"] == 30


def test_naive_applicability():
    from algorithms.registry import applicable
    assert applicable("naive", load_instance("N5"))[0]
    assert not applicable("naive", load_instance("X2"))[0]
    assert not applicable("naive", load_instance("A1"))[0]
