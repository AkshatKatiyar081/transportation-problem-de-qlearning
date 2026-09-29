"""Vogel's Approximation Method + MODI (u-v) method for the linear transportation problem.

Works on a balanced problem.  The basis is kept as a spanning tree of m + n - 1 cells
(degenerate basic cells carry 0, the textbook "epsilon").
"""
from __future__ import annotations

from collections import deque

import numpy as np

EPS = 1e-9


# ------------------------------------------------------------------ VAM
def vam(cost, supply, demand):
    """Return (X, basis) with exactly m + n - 1 basic cells."""
    m, n = cost.shape
    s = supply.astype(np.int64).copy()
    d = demand.astype(np.int64).copy()
    rows, cols = set(range(m)), set(range(n))
    X = np.zeros((m, n), dtype=np.int64)
    basis = []

    def penalty(vals):
        v = sorted(vals)
        return v[0] if len(v) == 1 else v[1] - v[0]

    while rows and cols:
        best = None  # (penalty, -mincost, is_row_first, index)
        for i in sorted(rows):
            vals = [cost[i, j] for j in cols]
            key = (penalty(vals), -min(vals), 1, -i)
            if best is None or key > best[0]:
                best = (key, "row", i)
        for j in sorted(cols):
            vals = [cost[i, j] for i in rows]
            key = (penalty(vals), -min(vals), 0, -j)
            if key > best[0]:
                best = (key, "col", j)
        _, kind, idx = best
        if kind == "row":
            i = idx
            j = min(cols, key=lambda c: (cost[i, c], c))
        else:
            j = idx
            i = min(rows, key=lambda r: (cost[r, j], r))
        q = min(s[i], d[j])
        X[i, j] = q
        basis.append((i, j))
        s[i] -= q
        d[j] -= q
        if s[i] == 0 and d[j] == 0:
            # degenerate: cross out only one line so the basis keeps m + n - 1 cells
            if len(rows) > 1:
                rows.discard(i)
            else:
                cols.discard(j)
        elif s[i] == 0:
            rows.discard(i)
        else:
            cols.discard(j)
    return X, basis


# ------------------------------------------------------------ tree tools
def _tree_path(basis, m, start, goal):
    """Path of basic cells between node `start` and node `goal` (rows 0..m-1, cols m..)."""
    adj = {}
    for (i, j) in basis:
        adj.setdefault(i, []).append((m + j, (i, j)))
        adj.setdefault(m + j, []).append((i, (i, j)))
    prev = {start: None}
    dq = deque([start])
    while dq:
        u = dq.popleft()
        if u == goal:
            break
        for v, cell in adj.get(u, []):
            if v not in prev:
                prev[v] = (u, cell)
                dq.append(v)
    if goal not in prev:
        return None
    path = []
    u = goal
    while prev[u] is not None:
        u, cell = prev[u]
        path.append(cell)
    return path  # ordered from goal back to start


def _uf_find(p, a):
    while p[a] != a:
        p[a] = p[p[a]]
        a = p[a]
    return a


def complete_basis(X, cost):
    """Turn any feasible plan into a basic one with a spanning-tree basis.

    1. While the positive cells contain a cycle, shift flow around it (in the
       cost-non-increasing direction) until one cell becomes 0.
    2. Add zero-valued (epsilon) cells, cheapest first, until m + n - 1 cells form a tree.
    Returns (X, basis, notes).
    """
    X = np.array(X, dtype=np.int64)
    m, n = X.shape
    notes = []
    while True:
        cells = [(i, j) for i in range(m) for j in range(n) if X[i, j] > 0]
        parent = list(range(m + n))
        forest = []
        closing = None
        for (i, j) in cells:
            a, b = _uf_find(parent, i), _uf_find(parent, m + j)
            if a == b:
                closing = (i, j)
                break
            parent[a] = b
            forest.append((i, j))
        if closing is None:
            break
        i, j = closing
        path = _tree_path(forest, m, i, m + j)  # cells from col j back to row i
        loop = [(i, j)] + path                 # signs alternate +, -, +, ...
        signs = [1 if k % 2 == 0 else -1 for k in range(len(loop))]
        delta = sum(sg * cost[c] for sg, c in zip(signs, loop))
        if delta > 0:
            signs = [-s for s in signs]
        theta = min(X[c] for sg, c in zip(signs, loop) if sg < 0)
        for sg, c in zip(signs, loop):
            X[c] += sg * theta
        notes.append(f"removed a cycle of {len(loop)} cells (shifted {theta} units)")
    basis = [(i, j) for i in range(m) for j in range(n) if X[i, j] > 0]
    parent = list(range(m + n))
    for (i, j) in basis:
        parent[_uf_find(parent, i)] = _uf_find(parent, m + j)
    added = []
    for (i, j) in sorted(((i, j) for i in range(m) for j in range(n) if X[i, j] == 0),
                         key=lambda c: (cost[c], c)):
        if len(basis) == m + n - 1:
            break
        a, b = _uf_find(parent, i), _uf_find(parent, m + j)
        if a != b:
            parent[a] = b
            basis.append((i, j))
            added.append((i, j))
    if added:
        notes.append(f"degenerate plan: added epsilon cells {added}")
    return X, basis, notes


def _potentials(cost, basis, m, n):
    u = [None] * m
    v = [None] * n
    u[0] = 0.0
    changed = True
    while changed:
        changed = False
        for (i, j) in basis:
            if u[i] is not None and v[j] is None:
                v[j] = cost[i, j] - u[i]
                changed = True
            elif v[j] is not None and u[i] is None:
                u[i] = cost[i, j] - v[j]
                changed = True
    return np.array(u, dtype=float), np.array(v, dtype=float)


# ------------------------------------------------------------------ MODI
def modi(cost, supply, demand, initial=None, max_iter=1000, trace=False):
    """Solve a balanced linear TP.  Returns dict with X, basis, u, v, delta, iterations, trace."""
    m, n = cost.shape
    notes = []
    if initial is None:
        X, basis = vam(cost, supply, demand)
        start = "VAM"
    else:
        X, basis, notes = complete_basis(initial, cost)
        start = "given plan"
    steps = []
    it = 0
    while True:
        u, v = _potentials(cost, basis, m, n)
        delta = cost - u[:, None] - v[None, :]
        bset = set(basis)
        for c in bset:
            delta[c] = 0.0
        e = min(((i, j) for i in range(m) for j in range(n) if (i, j) not in bset),
                key=lambda c: (delta[c], c), default=None)
        optimal = e is None or delta[e] >= -EPS
        step = {"iter": it, "X": X.copy(), "basis": list(basis), "u": u.copy(), "v": v.copy(),
                "delta": delta.copy(), "cost": float((cost * X).sum()), "optimal": optimal}
        if optimal or it >= max_iter:
            steps.append(step)
            break
        i, j = e
        path = _tree_path(basis, m, i, m + j)  # cells from col j back to row i
        loop = [e] + path
        minus = loop[1::2]
        theta = min(X[c] for c in minus)
        leave = next(c for c in minus if X[c] == theta)
        for k, c in enumerate(loop):
            X[c] += theta if k % 2 == 0 else -theta
        basis.remove(leave)
        basis.append(e)
        step.update({"entering": e, "loop": loop, "theta": int(theta), "leaving": leave})
        steps.append(step)
        it += 1
    final = steps[-1]
    return {"X": X, "basis": basis, "u": final["u"], "v": final["v"], "delta": final["delta"],
            "iterations": it, "optimal": final["optimal"], "start": start, "notes": notes,
            "trace": steps if trace else None}
