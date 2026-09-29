"""One interface for every method:  run(method, problem, params, seed, callback) -> SolverResult."""
from __future__ import annotations

import time
import traceback

import numpy as np

from algorithms import exact, modi
from algorithms.de import run_de
from algorithms.local_search import polish as _polish
from algorithms.result import SolverResult

METHODS = {
    "modi": "MODI (VAM + u-v)",
    "lp": "LP (HiGHS)",
    "milp": "MILP exact (HiGHS)",
    "naive": "Naive plan (ignore nonlinear terms)",
    "de": "Standard DE",
    "de_random": "DE-Random (ablation)",
    "hybrid": "Hybrid DE + Q-Learning",
}
DE_METHODS = ("de", "de_random", "hybrid")

DEFAULTS = {
    "de": {"NP": 50, "F": 0.5, "CR": 0.9, "fe_budget": 25000, "polish": False},
    "de_random": {"NP": 50, "fe_budget": 25000, "polish": False},
    "hybrid": {"NP": 50, "fe_budget": 25000, "alpha": 0.1, "gamma": 0.9, "eps_frac": 0.4,
               "sr_hi": 0.20, "sr_lo": 0.05, "polish": False},
    "milp": {"time_limit": 300.0},
    "modi": {},
    "lp": {},
    "naive": {},
}


def applicable(method, prob):
    """(ok, reason)."""
    if method == "modi":
        if prob.is_standard:
            return True, ""
        return False, "MODI needs a standard problem: linear cost, no route constraints, one objective, no noise"
    if method == "lp":
        if prob.is_linear_program:
            return True, ""
        return False, "LP needs linear costs (no discounts, fixed charges, minimum shipments or risk term)"
    if method == "milp":
        if prob.noise is not None and prob.robust_k > 0:
            return False, "the robust risk term (k * sd) is a square root - not linear; set k = 0 to use MILP"
        return True, ""
    if method == "naive":
        if prob.is_standard:
            return False, "problem is already linear - naive plan = MODI"
        return True, ""
    return True, ""


def _finish(res, prob, bal, Xb, t0):
    X = bal.strip_dummy(Xb)
    ev = prob.evaluate(X)
    res.matrix = X
    res.objective = ev["objective"]
    res.cost = ev["cost"]
    res.components = {k: v for k, v in ev.items() if k not in ("violations", "feasible", "objective")}
    res.feasible = ev["feasible"]
    res.diagnostics["violations"] = ev["violations"]
    if bal.meta.get("dummy"):
        dm = bal.dummy_mask()
        res.diagnostics["dummy_units"] = int(np.asarray(Xb)[dm].sum())
        res.diagnostics["dummy"] = bal.meta["dummy"]
    res.time_s = time.perf_counter() - t0
    return res


def _linear_version(prob):
    """Drop nonlinear terms: discounts, fixed charges, minimum shipments, risk (keeps mean costs)."""
    d = prob.to_dict()
    d.update({"discount": None, "fixed_charge": None, "min_ship": None, "noise": None})
    from core.problem import Problem
    return Problem.from_dict(d)


def run(method, prob, params=None, seed=None, callback=None) -> SolverResult:
    p = dict(DEFAULTS.get(method, {}))
    p.update(params or {})
    ok, why = applicable(method, prob)
    if not ok:
        return SolverResult(method, None, None, None, status="not_applicable", message=why, feasible=False)
    t0 = time.perf_counter()
    try:
        bal = prob.balanced()
        if method == "modi":
            init = p.get("initial")
            if init is not None:
                init = np.asarray(init, dtype=np.int64)
                if bal.meta.get("dummy") == "row":
                    init = np.vstack([init, (bal.demand - init.sum(0))[None, :]])
                elif bal.meta.get("dummy") == "col":
                    init = np.hstack([init, (bal.supply - init.sum(1))[:, None]])
            out = modi.modi(bal.cost, bal.supply, bal.demand, initial=init, trace=p.get("trace", False))
            res = SolverResult(method, None, None, None, status="optimal" if out["optimal"] else "ok")
            res.trace = out["trace"]
            res.diagnostics.update({"u": out["u"], "v": out["v"], "delta": out["delta"], "basis": out["basis"],
                                    "iterations": out["iterations"], "start": out["start"],
                                    "notes": out["notes"], "balanced_shape": (bal.m, bal.n)})
            return _finish(res, prob, bal, out["X"], t0)
        if method == "lp":
            X, st, msg = exact.solve_lp(bal)
            if X is None:
                return SolverResult(method, None, None, None, status=st, message=msg, feasible=False,
                                    time_s=time.perf_counter() - t0)
            return _finish(SolverResult(method, None, None, None, status=st), prob, bal, X, t0)
        if method == "milp":
            X, st, info = exact.solve_milp(bal, time_limit=p["time_limit"])
            if X is None:
                return SolverResult(method, None, None, None, status=st, message=str(info.get("message")),
                                    feasible=False, time_s=time.perf_counter() - t0, diagnostics=info)
            res = SolverResult(method, None, None, None, status=st, diagnostics=dict(info))
            return _finish(res, prob, bal, X, t0)
        if method == "naive":
            lin = _linear_version(prob)
            sub = "modi" if lin.is_standard else "lp"
            r = run(sub, lin)
            if r.matrix is None:
                return SolverResult(method, None, None, None, status=r.status, message=r.message, feasible=False)
            Xb = r.matrix
            if bal.meta.get("dummy"):
                Xb = np.vstack([Xb, bal.demand - Xb.sum(0)]) if bal.meta["dummy"] == "row" else \
                    np.hstack([Xb, (bal.supply - Xb.sum(1))[:, None]])
            res = SolverResult(method, None, None, None, status="ok",
                               message=f"plan from {METHODS[sub]} on the linear version, priced with the real costs")
            res.diagnostics["linear_cost"] = r.cost
            return _finish(res, prob, bal, Xb, t0)
        if method in DE_METHODS:
            rng = np.random.default_rng(seed)
            mode = {"de": "fixed", "de_random": "random", "hybrid": "qlearn"}[method]
            ql = None
            if method == "hybrid":
                ql = {k: p[k] for k in ("alpha", "gamma", "eps_frac", "sr_hi", "sr_lo")}
            out = run_de(bal, NP=p["NP"], fe_budget=p["fe_budget"], rng=rng, mode=mode,
                         F=p.get("F", 0.5), CR=p.get("CR", 0.9), ql=ql, callback=callback,
                         init_keys=p.get("init_keys"))
            res = SolverResult(method, None, None, None, status="ok", fe_used=out["fe"], history=out["history"],
                               actions=out["actions"], states=out["states"], rewards=out["rewards"],
                               sr=out["sr"], q_table=out["Q"])
            res.diagnostics.update({"neutral_frac_mean": float(np.mean(out["neutral"])) if out["neutral"] else 0.0,
                                    "distinct_plans_final": out["distinct_final"], "generations": out["G"],
                                    "unmet": out["unmet"], "keys": out["keys"],
                                    "neutral": out["neutral"]})
            Xb = out["X"]
            if p.get("polish"):
                before = bal.evaluate(Xb)["objective"]
                Xb, nimp, nev = _polish(bal, Xb)
                res.diagnostics.update({"polish_improvements": nimp, "polish_evals": nev,
                                        "objective_before_polish": before})
            return _finish(res, prob, bal, Xb, t0)
        raise ValueError(f"unknown method {method}")
    except Exception as e:  # shown in the dashboard
        return SolverResult(method, None, None, None, status="error", message=f"{e}\n{traceback.format_exc()}",
                            feasible=False, time_s=time.perf_counter() - t0)


def reference_solution(prob, time_limit=300.0):
    """Best exact answer available: LP for linear problems, else MILP.  (value, method, status)."""
    if prob.is_linear_program:
        r = run("lp", prob)
    else:
        ok, _ = applicable("milp", prob)
        if not ok:
            return None, None, "no exact method (robust objective)"
        r = run("milp", prob, {"time_limit": time_limit})
    return r.objective, r.method, r.status


def evaluate_plan(prob, X):
    """Check a user plan: feasibility + objective components."""
    return prob.evaluate(np.asarray(X, dtype=float))
