"""Differential Evolution (DE/rand/1/bin) on priority keys, with three parameter controllers:

  fixed   : constant F, CR                                  (standard DE)
  random  : (F, CR) drawn uniformly from the 3x3 grid each generation   (DE-Random ablation)
  qlearn  : tabular Q-learning picks (F, CR) each generation            (Hybrid DE + QL)
"""
from __future__ import annotations

import numpy as np

from core.decoder import DecoderCache, decode

F_VALUES = (0.3, 0.6, 0.9)
CR_VALUES = (0.1, 0.5, 0.9)
ACTIONS = [(f, cr) for f in F_VALUES for cr in CR_VALUES]   # a0..a8, row-major (F, CR)


class Evaluator:
    """fitness(keys) = scalar objective + penalty * unmet units (balanced problem)."""

    def __init__(self, prob):
        self.prob = prob
        self.cache = DecoderCache(prob)
        self.penalty = prob.unit_penalty()
        self.simple = (prob.discount is None and prob.fixed_charge is None and prob.objective2 is None
                       and prob.noise is None)
        self.cost = prob.cost
        self.fe = 0

    def __call__(self, keys):
        X, unmet = decode(keys, self.prob, self.cache)
        self.fe += 1
        if self.simple:
            f = float((self.cost * X).sum())
        else:
            f = self.prob.scalar(self.prob.components(X))
        return f + self.penalty * unmet, X, unmet


class QController:
    def __init__(self, rng, G, alpha=0.1, gamma=0.9, eps_frac=0.4, sr_hi=0.20, sr_lo=0.05, eps_min=0.05):
        self.rng, self.G = rng, G
        self.alpha, self.gamma, self.eps_frac, self.eps_min = alpha, gamma, eps_frac, eps_min
        self.sr_hi, self.sr_lo = sr_hi, sr_lo
        self.Q = np.zeros((3, len(ACTIONS)))
        self.state = 0
        self.last = None

    def bucket(self, sr):
        return 0 if sr > self.sr_hi else (1 if sr > self.sr_lo else 2)

    def eps(self, g):
        return max(self.eps_min, 1.0 - g / max(1.0, self.eps_frac * self.G))

    def choose(self, g):
        if self.rng.random() < self.eps(g):
            a = int(self.rng.integers(len(ACTIONS)))
        else:
            row = self.Q[self.state]
            best = np.flatnonzero(row == row.max())
            a = int(self.rng.choice(best))
        self.last = a
        return a

    def update(self, sr, improved):
        r = sr + (0.5 if improved else 0.0)
        s2 = self.bucket(sr)
        s, a = self.state, self.last
        self.Q[s, a] += self.alpha * (r + self.gamma * self.Q[s2].max() - self.Q[s, a])
        self.state = s2
        return r


def run_de(prob, NP=50, fe_budget=25000, rng=None, mode="fixed", F=0.5, CR=0.9, ql=None,
           callback=None, init_keys=None):
    """Run DE on a balanced problem.  Returns a dict with best keys/plan and logs."""
    rng = rng if rng is not None else np.random.default_rng()
    ev = Evaluator(prob)
    D = prob.m * prob.n
    NP = max(4, int(NP))
    pop = rng.random((NP, D))
    if init_keys is not None:
        pop[0] = init_keys
    fit = np.empty(NP)
    plans = [None] * NP
    for i in range(NP):
        fit[i], X, _ = ev(pop[i])
        plans[i] = X.tobytes()
    G = max(0, (int(fe_budget) - NP) // NP)
    ctrl = QController(rng, G, **(ql or {})) if mode == "qlearn" else None
    b = int(np.argmin(fit))
    best_f, best_keys = float(fit[b]), pop[b].copy()
    history = [(ev.fe, best_f)]
    actions, states, rewards, srs, neutral, distinct = [], [], [], [], [], []
    others = np.array([[k for k in range(NP) if k != i] for i in range(NP)])
    for g in range(G):
        if mode == "fixed":
            f_g, cr_g, a = F, CR, None
        elif mode == "random":
            a = int(rng.integers(len(ACTIONS)))
            f_g, cr_g = ACTIONS[a]
        else:
            states.append(ctrl.state)
            a = ctrl.choose(g)
            f_g, cr_g = ACTIONS[a]
        # mutation (rand/1) for the whole population
        pick = np.array([rng.choice(others[i], 3, replace=False) for i in range(NP)])
        V = pop[pick[:, 0]] + f_g * (pop[pick[:, 1]] - pop[pick[:, 2]])
        low, high = V < 0, V > 1
        V[low] = pop[low] / 2                      # midpoint rule
        V[high] = (pop[high] + 1) / 2
        mask = rng.random((NP, D)) <= cr_g
        mask[np.arange(NP), rng.integers(D, size=NP)] = True
        Utr = np.where(mask, V, pop)
        new_pop, new_fit = pop.copy(), fit.copy()
        better = neut = 0
        for i in range(NP):
            fu, Xu, _ = ev(Utr[i])
            pb = Xu.tobytes()
            if pb == plans[i]:
                neut += 1
            if fu < fit[i] - 1e-12:
                better += 1
            if fu <= fit[i]:
                new_pop[i], new_fit[i], plans[i] = Utr[i], fu, pb
        pop, fit = new_pop, new_fit
        b = int(np.argmin(fit))
        improved = fit[b] < best_f - 1e-12
        if improved:
            best_f, best_keys = float(fit[b]), pop[b].copy()
        sr = better / NP
        srs.append(sr)
        neutral.append(neut / NP)
        if mode != "fixed":
            actions.append(a)
        if ctrl is not None:
            rewards.append(ctrl.update(sr, improved))
        history.append((ev.fe, best_f))
        if callback is not None and (g % max(1, G // 100) == 0 or g == G - 1):
            callback((g + 1) / G, best_f)
    distinct = len(set(plans))
    X, unmet = decode(best_keys, prob, ev.cache)
    return {"keys": best_keys, "X": X, "unmet": unmet, "fitness": best_f, "fe": ev.fe, "history": history,
            "actions": actions or None, "states": states or None, "rewards": rewards or None, "sr": srs,
            "neutral": neutral, "distinct_final": distinct, "G": G,
            "Q": None if ctrl is None else ctrl.Q.copy()}
