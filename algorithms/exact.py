"""Exact solvers: LP (scipy linprog / HiGHS) and MILP (scipy milp / HiGHS).

The MILP handles every cost / constraint feature except the robust (square-root) risk term:
incremental and all-units discounts, fixed charges, capacities, forbidden routes, minimum
shipments and a weighted second objective.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.sparse import lil_matrix


def _weights(prob):
    """Coefficients (a_c, a_t) of cost and objective-2 in the scalar objective."""
    if prob.objective2 is None:
        return 1.0, 0.0
    w = prob.objective2["weight"]
    nc, nt = prob.norms()
    return (1 - w) / nc, w / nt


def solve_lp(prob):
    """Plain LP for problems with linear cost (capacity / forbidden / obj2 allowed). prob balanced."""
    m, n = prob.m, prob.n
    a_c, a_t = _weights(prob)
    c = a_c * prob.cost.ravel()
    if prob.objective2 is not None:
        c = c + a_t * prob.objective2["matrix"].ravel()
    A = np.zeros((m + n, m * n))
    for i in range(m):
        A[i, i * n:(i + 1) * n] = 1
    for j in range(n):
        A[m + j, j::n] = 1
    b = np.concatenate([prob.supply, prob.demand]).astype(float)
    U = prob.upper_bounds().ravel()
    res = linprog(c, A_eq=A, b_eq=b, bounds=list(zip(np.zeros(m * n), U)), method="highs")
    if res.status != 0:
        return None, "infeasible" if res.status == 2 else "error", res.message
    X = np.rint(res.x).astype(np.int64).reshape(m, n)
    return X, "optimal", ""


class _Model:
    def __init__(self):
        self.lb, self.ub, self.integ, self.c = [], [], [], []
        self.rows = []  # (dict var->coef, lo, hi)

    def var(self, lb, ub, integer, cost=0.0):
        self.lb.append(lb)
        self.ub.append(ub)
        self.integ.append(1 if integer else 0)
        self.c.append(cost)
        return len(self.c) - 1

    def row(self, coefs, lo, hi):
        self.rows.append((coefs, lo, hi))


def solve_milp(prob, time_limit=300.0):
    """Exact MILP. prob must be balanced and have no robust risk term (k = 0 or no noise).
    Returns (X or None, status, info dict)."""
    m, n = prob.m, prob.n
    a_c, a_t = _weights(prob)
    U = prob.upper_bounds()
    L = prob.min_ship_matrix()
    F = prob.fixed_charge
    disc = prob.discount
    M = _Model()
    xv = {}
    for i in range(m):
        for j in range(n):
            u = float(U[i, j])
            if u <= 0:
                continue
            c = float(prob.cost[i, j])
            t = 0.0 if prob.objective2 is None else float(prob.objective2["matrix"][i, j])
            x = M.var(0, u, True, a_t * t + (a_c * c if disc is None else 0.0))
            xv[i, j] = x
            need_y = (F is not None and F[i, j] > 0) or L[i, j] > 0
            if need_y:
                fc = 0.0 if F is None else float(F[i, j])
                y = M.var(0, 1, True, a_c * fc)
                M.row({x: 1.0, y: -u}, -np.inf, 0.0)          # x <= U y
                if L[i, j] > 0:
                    if L[i, j] > u:                             # route can never be used
                        M.row({y: 1.0}, 0.0, 0.0)
                    else:
                        M.row({x: 1.0, y: -float(L[i, j])}, 0.0, np.inf)  # x >= L y
            if disc is None:
                continue
            tiers = disc["tiers"]
            if disc["type"] == "incremental":
                segs = []
                lo = 0.0
                for up, r in tiers:
                    if lo >= u:
                        break
                    hi = u if up is None else min(float(up), u)
                    full = None if up is None else float(up) - lo
                    segs.append((M.var(0, hi - lo, False, a_c * c * r), hi - lo, full))
                    lo = hi if up is None else float(up)
                M.row({x: 1.0, **{s: -1.0 for s, _, _ in segs}}, 0.0, 0.0)
                bins = [None] + [M.var(0, 1, True) for _ in segs[1:]]
                for k in range(1, len(segs)):
                    s_k, w_k, _ = segs[k]
                    M.row({s_k: 1.0, bins[k]: -w_k}, -np.inf, 0.0)          # s_k <= w_k b_k
                    s_prev, _, full_prev = segs[k - 1]
                    M.row({s_prev: 1.0, bins[k]: -full_prev}, 0.0, np.inf)  # s_{k-1} >= full b_k
                    if k >= 2:
                        M.row({bins[k]: 1.0, bins[k - 1]: -1.0}, -np.inf, 0.0)
            else:  # all-units
                parts = []
                prev_up = None
                for up, r in tiers:
                    lo = 0.0 if prev_up is None else float(prev_up) + 1
                    hi = u if up is None else min(float(up), u)
                    prev_up = up
                    if lo > u:
                        break
                    z = M.var(0, 1, True)
                    w = M.var(0, hi, False, a_c * c * r)
                    M.row({w: 1.0, z: -hi}, -np.inf, 0.0)
                    M.row({w: 1.0, z: -lo}, 0.0, np.inf)
                    parts.append((z, w))
                    if up is None:
                        break
                M.row({z: 1.0 for z, _ in parts}, 1.0, 1.0)
                M.row({x: 1.0, **{w: -1.0 for _, w in parts}}, 0.0, 0.0)
    for i in range(m):
        M.row({xv[i, j]: 1.0 for j in range(n) if (i, j) in xv}, float(prob.supply[i]), float(prob.supply[i]))
    for j in range(n):
        M.row({xv[i, j]: 1.0 for i in range(m) if (i, j) in xv}, float(prob.demand[j]), float(prob.demand[j]))
    A = lil_matrix((len(M.rows), len(M.c)))
    lo, hi = [], []
    for r, (coefs, l, h) in enumerate(M.rows):
        for v, a in coefs.items():
            A[r, v] = a
        lo.append(l)
        hi.append(h)
    res = milp(np.array(M.c), integrality=np.array(M.integ), bounds=Bounds(M.lb, M.ub),
               constraints=LinearConstraint(A.tocsr(), lo, hi),
               options={"time_limit": float(time_limit), "disp": False})
    info = {"message": res.message, "mip_gap": getattr(res, "mip_gap", None),
            "bound": getattr(res, "mip_dual_bound", None), "n_vars": len(M.c), "n_rows": len(M.rows),
            "n_binaries": int(sum(M.integ)) - len(xv)}
    if res.x is None:
        status = "infeasible" if res.status == 2 else ("time_limit" if res.status == 1 else "error")
        return None, status, info
    X = np.zeros((m, n), dtype=np.int64)
    for (i, j), v in xv.items():
        X[i, j] = int(round(res.x[v]))
    status = "optimal" if res.status == 0 else "time_limit"
    return X, status, info
