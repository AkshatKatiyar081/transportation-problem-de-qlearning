"""Paired statistics: Wilcoxon signed-rank (two-sided, zsplit), Holm correction, effect sizes."""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata, wilcoxon


def rank_biserial(a, b):
    """Matched-pairs rank-biserial correlation for d = a - b (negative = a is lower/better)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[np.abs(d) > 1e-9]
    if d.size == 0:
        return 0.0
    r = rankdata(np.abs(d))
    return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())


def a12(a, b):
    """Vargha-Delaney A12 for minimisation: P(a < b) + 0.5 P(a == b)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    lt = (a[:, None] < b[None, :] - 1e-9).sum()
    eq = (np.abs(a[:, None] - b[None, :]) <= 1e-9).sum()
    return float((lt + 0.5 * eq) / (a.size * b.size))


def compare(a, b, max_ties=25):
    """a, b: paired samples (same seeds).  Returns dict with p, r_rb, A12, ties, informative."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ties = int((np.abs(a - b) <= 1e-9).sum())
    out = {"n": int(a.size), "ties": ties, "r_rb": rank_biserial(a, b), "A12": a12(a, b),
           "mean_diff": float((a - b).mean())}
    if ties == a.size:
        out.update(p=1.0, informative=False)
        return out
    try:
        p = wilcoxon(a, b, zero_method="zsplit", alternative="two-sided").pvalue
    except ValueError:
        p = 1.0
    out.update(p=float(p), informative=ties < max_ties)
    return out


def holm(pvals):
    """Holm step-down adjusted p-values (same order as input)."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    k = len(p)
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (k - rank) * p[idx]))
        adj[idx] = running
    return adj.tolist()


def effect_label(r):
    r = abs(r)
    return "negligible" if r < 0.1 else "small" if r < 0.3 else "medium" if r < 0.5 else "large"
