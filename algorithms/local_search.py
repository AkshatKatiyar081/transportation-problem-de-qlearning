"""Cycle-move local search ("polish") for any cost model.

A move picks two rows i, k and two columns j, l and shifts delta units around the
rectangle: (i,j) +d, (k,l) +d, (i,l) -d, (k,j) -d.  Row and column totals are unchanged,
so feasibility with respect to supply/demand is kept.  Candidate deltas include 1, the
largest possible shift, and shifts that put a cell exactly on a tier breakpoint or on its
minimum shipment - this lets the search reach non-basic optima (all-units discounts,
minimum shipments) that the decoder alone cannot produce.
"""
from __future__ import annotations

import itertools

import numpy as np


def _fitness(prob, X, penalty):
    ev = prob.evaluate(X)
    return ev["objective"] + penalty * sum(ev["violations"].values())


def polish(prob, X, max_passes=10, max_evals=200000):
    """prob: balanced problem.  Returns (X, n_improvements, evals)."""
    X = np.array(X, dtype=np.int64)
    m, n = X.shape
    U = prob.upper_bounds()
    penalty = prob.unit_penalty()
    marks = set()
    if prob.discount is not None:
        for up, _ in prob.discount["tiers"]:
            if up is not None:
                marks.update({up, up + 1})
    Lmat = prob.min_ship_matrix()
    best = _fitness(prob, X, penalty)
    improvements = evals = 0
    for _ in range(max_passes):
        improved_pass = False
        for (i, k) in itertools.combinations(range(m), 2):
            for (j, l) in itertools.combinations(range(n), 2):
                for plus, minus in ((((i, j), (k, l)), ((i, l), (k, j))), (((i, l), (k, j)), ((i, j), (k, l)))):
                    dmax = min(X[minus[0]], X[minus[1]], U[plus[0]] - X[plus[0]], U[plus[1]] - X[plus[1]])
                    if dmax <= 0:
                        continue
                    cands = {1, int(dmax)}
                    for c in plus:
                        for t in marks | {int(Lmat[c])}:
                            if 0 < t - X[c] <= dmax:
                                cands.add(int(t - X[c]))
                    for c in minus:
                        for t in marks | {int(Lmat[c]) - 1}:
                            if 0 < X[c] - t <= dmax:
                                cands.add(int(X[c] - t))
                    for dlt in sorted(cands):
                        for c in plus:
                            X[c] += dlt
                        for c in minus:
                            X[c] -= dlt
                        f = _fitness(prob, X, penalty)
                        evals += 1
                        if f < best - 1e-9:
                            best = f
                            improvements += 1
                            improved_pass = True
                            break
                        for c in plus:
                            X[c] -= dlt
                        for c in minus:
                            X[c] += dlt
                    if evals >= max_evals:
                        return X, improvements, evals
        if not improved_pass:
            break
    return X, improvements, evals
