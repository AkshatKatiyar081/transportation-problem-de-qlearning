"""Priority-based (random-key) decoder.

A key vector (one real number per cell) is turned into a shipment plan by filling
cells in order of decreasing key, each time shipping as much as possible:
    q = min(remaining supply, remaining demand, route capacity)
Forbidden routes are skipped, and a route is skipped when q is below its minimum
shipment.  For a balanced problem without route constraints the result is always a
feasible, integer, basic solution.  With route constraints some supply may be left
unshipped ("unmet"); the fitness adds a penalty for it.
"""
from __future__ import annotations

import numpy as np


class DecoderCache:
    """Pre-computed plain-Python lists for fast decoding of one (balanced) problem."""

    def __init__(self, prob):
        self.m, self.n = prob.m, prob.n
        self.supply = [int(v) for v in prob.supply]
        self.demand = [int(v) for v in prob.demand]
        U = prob.upper_bounds()
        L = prob.min_ship_matrix()
        self.constrained = prob.has_route_constraints
        self.cap = [float(v) for v in U.ravel()]
        self.low = [float(v) for v in L.ravel()]
        self.cell_i = [k // self.n for k in range(self.m * self.n)]
        self.cell_j = [k % self.n for k in range(self.m * self.n)]


def decode(keys, prob, cache: DecoderCache | None = None, return_steps: bool = False):
    """Decode keys into a plan.  Returns (X, unmet) or (X, unmet, steps)."""
    c = cache or DecoderCache(prob)
    order = np.argsort(-np.asarray(keys, dtype=float), kind="stable").tolist()
    s = c.supply[:]
    d = c.demand[:]
    x = [0] * (c.m * c.n)
    steps = [] if return_steps else None
    ci, cj = c.cell_i, c.cell_j
    if not c.constrained:
        for k in order:
            i = ci[k]
            j = cj[k]
            q = s[i] if s[i] < d[j] else d[j]
            if q > 0:
                x[k] = q
                s[i] -= q
                d[j] -= q
            if steps is not None:
                steps.append((k, q, s[:], d[:], ""))
    else:
        cap, low = c.cap, c.low
        for k in order:
            i = ci[k]
            j = cj[k]
            q = s[i] if s[i] < d[j] else d[j]
            note = ""
            if cap[k] < q:
                q = int(cap[k])
                note = "capacity"
            if q > 0 and q < low[k]:
                note = "below minimum shipment - skipped"
                q = 0
            if cap[k] <= 0:
                note = "forbidden"
            if q > 0:
                x[k] = q
                s[i] -= q
                d[j] -= q
            if steps is not None:
                steps.append((k, q, s[:], d[:], note))
    X = np.array(x, dtype=np.int64).reshape(c.m, c.n)
    unmet = sum(s)
    if return_steps:
        return X, unmet, steps
    return X, unmet


def keys_from_plan(X, prob):
    """Keys that decode (for standard problems) to a given basic plan: larger shipments first
    is not always right, so we use a leaf-elimination order on the positive cells."""
    X = np.asarray(X)
    m, n = X.shape
    pos = {(i, j) for i in range(m) for j in range(n) if X[i, j] > 0}
    order = []
    s = prob.supply.astype(int).copy()
    d = prob.demand.astype(int).copy()
    remaining = set(pos)
    while remaining:
        progressed = False
        for (i, j) in sorted(remaining):
            if X[i, j] == min(s[i], d[j]):
                order.append(i * n + j)
                s[i] -= X[i, j]
                d[j] -= X[i, j]
                remaining.discard((i, j))
                progressed = True
                break
        if not progressed:  # not reachable greedily (non-basic plan) - append the rest
            order.extend(i * n + j for (i, j) in sorted(remaining))
            break
    rest = [k for k in range(m * n) if k not in order]
    keys = np.zeros(m * n)
    D = m * n
    for r, k in enumerate(order + rest):
        keys[k] = 1.0 - r / D
    return keys
