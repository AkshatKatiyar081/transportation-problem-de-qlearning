"""Problem model for standard and non-standard transportation problems.

A Problem holds the data of one transportation problem plus optional extras:

  discount      : {"type": "none" | "incremental" | "all_units", "tiers": [[upper|None, mult], ...]}
  fixed_charge  : m x n matrix, paid once if a route is used (x > 0)
  capacity      : m x n matrix, max units on a route (None / inf = unlimited)
  forbidden     : m x n 0/1 matrix, 1 = route not allowed
  min_ship      : m x n matrix, a used route must carry at least this many units (x = 0 or x >= L)
  objective2    : {"name": str, "matrix": m x n, "weight": w in [0, 1]}  (second linear objective)
  noise         : {"sd": m x n, "k": float}  (uncertain unit costs; robust cost = mean + k * sd)

Supplies and demands are integers.  Costs may be real numbers.
"""
from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

INF = float("inf")


def _mat(x, shape, fill, dtype=float):
    """Convert a nested list (possibly with None) to a float matrix, None -> fill."""
    if x is None:
        return None
    a = np.array([[fill if v is None else v for v in row] for row in x], dtype=dtype)
    if a.shape != shape:
        raise ValueError(f"matrix has shape {a.shape}, expected {shape}")
    return a


@dataclass
class Problem:
    name: str
    cost: np.ndarray
    supply: np.ndarray
    demand: np.ndarray
    row_labels: list = None
    col_labels: list = None
    discount: dict | None = None
    fixed_charge: np.ndarray | None = None
    capacity: np.ndarray | None = None
    forbidden: np.ndarray | None = None
    min_ship: np.ndarray | None = None
    objective2: dict | None = None
    noise: dict | None = None
    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ setup
    def __post_init__(self):
        self.cost = np.asarray(self.cost, dtype=float)
        self.supply = np.asarray(self.supply, dtype=np.int64)
        self.demand = np.asarray(self.demand, dtype=np.int64)
        m, n = self.cost.shape
        if self.supply.shape != (m,) or self.demand.shape != (n,):
            raise ValueError("supply/demand length does not match cost matrix")
        if self.row_labels is None:
            self.row_labels = [f"S{i + 1}" for i in range(m)]
        if self.col_labels is None:
            self.col_labels = [f"D{j + 1}" for j in range(n)]
        if self.discount is not None:
            t = self.discount.get("type", "none")
            if t not in ("none", "incremental", "all_units"):
                raise ValueError(f"unknown discount type {t}")
            if t == "none":
                self.discount = None
            else:
                tiers = [[None if (u is None or u == INF) else int(u), float(r)]
                         for u, r in self.discount["tiers"]]
                if tiers[-1][0] is not None:
                    tiers.append([None, tiers[-1][1]])
                ups = [u for u, _ in tiers[:-1]]
                if any(u is None for u in ups) or ups != sorted(ups) or len(set(ups)) != len(ups) or (ups and ups[0] <= 0):
                    raise ValueError("tier upper limits must be positive, strictly increasing, last one open")
                self.discount = {"type": t, "tiers": tiers}
        for attr in ("fixed_charge", "min_ship"):
            v = getattr(self, attr)
            if v is not None:
                v = np.asarray(v, dtype=float)
                if not v.any():
                    v = None
                setattr(self, attr, v)
        if self.capacity is not None:
            c = np.array(self.capacity, dtype=float)
            c[np.isnan(c)] = INF
            self.capacity = None if np.all(np.isinf(c)) else c
        if self.forbidden is not None:
            f = np.asarray(self.forbidden).astype(bool)
            self.forbidden = f if f.any() else None
        if self.objective2 is not None:
            o = dict(self.objective2)
            o["matrix"] = np.asarray(o["matrix"], dtype=float)
            o["weight"] = float(o.get("weight", 0.5))
            o.setdefault("name", "Objective 2")
            self.objective2 = o
        if self.noise is not None:
            nz = dict(self.noise)
            nz["sd"] = np.asarray(nz["sd"], dtype=float)
            nz["k"] = float(nz.get("k", 1.0))
            self.noise = nz
        self._validate()
        self._norms = None

    def _validate(self):
        if (self.supply < 0).any() or (self.demand < 0).any():
            raise ValueError("supplies and demands must be non-negative")
        if self.supply.sum() == 0 or self.demand.sum() == 0:
            raise ValueError("total supply and total demand must be positive")
        if (self.cost < 0).any():
            raise ValueError("costs must be non-negative")
        for name in ("fixed_charge", "min_ship"):
            v = getattr(self, name)
            if v is not None and (v.shape != self.cost.shape or (v < 0).any()):
                raise ValueError(f"{name}: wrong shape or negative values")
        if self.capacity is not None and (self.capacity.shape != self.cost.shape or (self.capacity < 0).any()):
            raise ValueError("capacity: wrong shape or negative values")
        if self.forbidden is not None and self.forbidden.shape != self.cost.shape:
            raise ValueError("forbidden: wrong shape")
        if self.objective2 is not None:
            o = self.objective2
            if o["matrix"].shape != self.cost.shape or not 0 <= o["weight"] <= 1:
                raise ValueError("objective2: wrong shape or weight outside [0, 1]")
        if self.noise is not None:
            if self.noise["sd"].shape != self.cost.shape or (self.noise["sd"] < 0).any():
                raise ValueError("noise sd: wrong shape or negative values")

    # ------------------------------------------------------------ properties
    @property
    def m(self):
        return self.cost.shape[0]

    @property
    def n(self):
        return self.cost.shape[1]

    @property
    def is_balanced(self):
        return int(self.supply.sum()) == int(self.demand.sum())

    @property
    def cost_type(self):
        return "linear" if self.discount is None else self.discount["type"]

    @property
    def has_route_constraints(self):
        return self.capacity is not None or self.forbidden is not None or self.min_ship is not None

    @property
    def is_standard(self):
        """Classic balanced-or-not linear TP: MODI applies."""
        return (self.discount is None and self.fixed_charge is None and not self.has_route_constraints
                and self.objective2 is None and self.noise is None)

    @property
    def is_linear_program(self):
        """Solvable as a plain LP (linear cost, linear constraints only)."""
        return (self.discount is None and self.fixed_charge is None and self.min_ship is None
                and self.noise is None)

    @property
    def robust_k(self):
        return 0.0 if self.noise is None else self.noise["k"]

    def features(self):
        f = []
        if self.discount is not None:
            f.append("incremental discount" if self.discount["type"] == "incremental" else "all-units discount")
        if self.fixed_charge is not None:
            f.append("fixed charge")
        if self.capacity is not None:
            f.append("route capacity")
        if self.forbidden is not None:
            f.append("forbidden routes")
        if self.min_ship is not None:
            f.append("minimum shipment")
        if self.objective2 is not None:
            f.append(f"2nd objective ({self.objective2['name']})")
        if self.noise is not None:
            f.append("uncertain costs")
        if not self.is_balanced:
            f.append("unbalanced")
        return f

    def upper_bounds(self):
        """Max units per route: min(supply, demand, capacity), 0 if forbidden."""
        U = np.minimum.outer(self.supply, self.demand).astype(float)
        if self.capacity is not None:
            U = np.minimum(U, self.capacity)
        if self.forbidden is not None:
            U[self.forbidden] = 0
        return U

    def min_ship_matrix(self):
        return np.zeros(self.cost.shape) if self.min_ship is None else self.min_ship

    # -------------------------------------------------------------- balance
    def balanced(self) -> "Problem":
        """Return a balanced copy (adds a zero-cost dummy row or column)."""
        S, D = int(self.supply.sum()), int(self.demand.sum())
        if S == D:
            p = self.copy()
            p.meta["dummy"] = None
            p.meta["orig_shape"] = (self.m, self.n)
            return p
        add_row = D > S
        gap = abs(S - D)

        def ext(a, fill):
            if a is None:
                return None
            a = np.asarray(a)
            if add_row:
                return np.vstack([a, np.full((1, a.shape[1]), fill, dtype=a.dtype)])
            return np.hstack([a, np.full((a.shape[0], 1), fill, dtype=a.dtype)])

        kw = dict(
            name=self.name,
            cost=ext(self.cost, 0.0),
            supply=np.append(self.supply, gap) if add_row else self.supply.copy(),
            demand=self.demand.copy() if add_row else np.append(self.demand, gap),
            row_labels=self.row_labels + (["Dummy"] if add_row else []),
            col_labels=self.col_labels + ([] if add_row else ["Dummy"]),
            discount=copy.deepcopy(self.discount),
            fixed_charge=ext(self.fixed_charge, 0.0),
            capacity=ext(self.capacity, INF),
            forbidden=ext(self.forbidden, False),
            min_ship=ext(self.min_ship, 0.0),
            objective2=None,
            noise=None,
            meta=copy.deepcopy(self.meta),
        )
        if self.objective2 is not None:
            o = dict(self.objective2)
            o["matrix"] = ext(o["matrix"], 0.0)
            kw["objective2"] = o
        if self.noise is not None:
            nz = dict(self.noise)
            nz["sd"] = ext(nz["sd"], 0.0)
            kw["noise"] = nz
        p = Problem(**kw)
        p.meta["dummy"] = "row" if add_row else "col"
        p.meta["orig_shape"] = (self.m, self.n)
        p.meta["dummy_cells"] = True
        return p

    def strip_dummy(self, X):
        """Remove the dummy row/column from a plan of the balanced problem."""
        X = np.asarray(X)
        d = self.meta.get("dummy")
        if d == "row":
            return X[:-1, :]
        if d == "col":
            return X[:, :-1]
        return X

    def dummy_mask(self):
        mask = np.zeros(self.cost.shape, dtype=bool)
        d = self.meta.get("dummy")
        if d == "row":
            mask[-1, :] = True
        elif d == "col":
            mask[:, -1] = True
        return mask

    # ----------------------------------------------------------- evaluation
    def effective_units(self, X):
        """h(x): quantity after discount, so that route cost = c * h(x)."""
        X = np.asarray(X, dtype=float)
        if self.discount is None:
            return X
        tiers = self.discount["tiers"]
        if self.discount["type"] == "incremental":
            h = np.zeros_like(X)
            lo = 0.0
            for up, r in tiers:
                hi = INF if up is None else float(up)
                h += r * np.clip(X - lo, 0, hi - lo)
                lo = hi
            return h
        # all-units: the whole quantity is charged at the rate of the tier it falls in
        rate = np.full_like(X, tiers[-1][1])
        for up, r in reversed(tiers[:-1]):
            rate = np.where(X <= up, r, rate)
        return rate * X

    def components(self, X):
        """Objective components of a plan X (same shape as this problem)."""
        X = np.asarray(X, dtype=float)
        h = self.effective_units(X)
        cost = float((self.cost * h).sum())
        if self.fixed_charge is not None:
            cost += float(self.fixed_charge[X > 0].sum())
        out = {"cost": cost}
        if self.noise is not None:
            sd = float(np.sqrt(((self.noise["sd"] * h) ** 2).sum()))
            out["cost_sd"] = sd
            out["robust_cost"] = cost + self.noise["k"] * sd
        if self.objective2 is not None:
            out["obj2"] = float((self.objective2["matrix"] * X).sum())
        return out

    def norms(self):
        """Normalisers for the weighted sum: value of a greedy 'cheapest first' plan per objective."""
        if self._norms is None:
            from core.decoder import decode
            b = self if self.is_balanced else self.balanced()
            X1, _ = decode(-b.cost.ravel(), b)
            c1 = b.components(X1)
            nc = c1.get("robust_cost", c1["cost"])
            nt = 1.0
            if b.objective2 is not None:
                X2, _ = decode(-b.objective2["matrix"].ravel(), b)
                nt = b.components(X2)["obj2"]
            self._norms = (max(nc, 1e-9), max(nt, 1e-9))
        return self._norms

    def scalar(self, comps):
        """Single value that all solvers minimise."""
        base = comps.get("robust_cost", comps["cost"])
        if self.objective2 is None:
            return base
        w = self.objective2["weight"]
        nc, nt = self.norms()
        return (1 - w) * base / nc + w * comps["obj2"] / nt

    def unit_penalty(self):
        """Penalty per unit of unmet supply/demand or violation (dominates any real cost)."""
        mults = [1.0] if self.discount is None else [r for _, r in self.discount["tiers"]]
        per = float(self.cost.max()) * max(mults) + (0 if self.fixed_charge is None else float(self.fixed_charge.max()))
        if self.noise is not None:
            per += self.noise["k"] * float(self.noise["sd"].max()) * max(mults)
        if self.objective2 is None:
            return 10.0 * (per + 1.0)
        w = self.objective2["weight"]
        nc, nt = self.norms()
        return 10.0 * ((1 - w) * (per + 1.0) / nc + w * (float(self.objective2["matrix"].max()) + 1.0) / nt)

    def violations(self, X):
        """Return dict of constraint violations (units) for a plan of this problem's shape."""
        X = np.asarray(X, dtype=float)
        rs, cs = X.sum(1), X.sum(0)
        S, D = int(self.supply.sum()), int(self.demand.sum())
        # unbalanced: the larger side is only an upper limit (excess goes to the dummy)
        vs = np.clip(rs - self.supply, 0, None).sum() if S > D else np.abs(rs - self.supply).sum()
        vd = np.clip(cs - self.demand, 0, None).sum() if D > S else np.abs(cs - self.demand).sum()
        v = {
            "supply": float(vs),
            "demand": float(vd),
            "negative": float(-X[X < 0].sum()),
        }
        if self.capacity is not None:
            v["capacity"] = float(np.clip(X - self.capacity, 0, None).sum())
        if self.forbidden is not None:
            v["forbidden"] = float(X[self.forbidden].sum())
        if self.min_ship is not None:
            used = X > 0
            v["min_ship"] = float(np.clip(self.min_ship - X, 0, None)[used].sum())
        return v

    def evaluate(self, X):
        """Full evaluation: components, scalar objective, violations, feasibility."""
        comps = self.components(X)
        viol = self.violations(X)
        total_v = sum(viol.values())
        return {
            **comps,
            "objective": self.scalar(comps),
            "violations": viol,
            "feasible": total_v < 1e-6,
        }

    # ------------------------------------------------------------- utilities
    def copy(self) -> "Problem":
        return Problem.from_dict(self.to_dict())

    def to_dict(self) -> dict:
        def L(a):
            if a is None:
                return None
            a = np.asarray(a)
            if a.dtype == bool:
                return a.astype(int).tolist()
            return [[None if (isinstance(v, float) and math.isinf(v)) else (int(v) if float(v).is_integer() else float(v))
                     for v in row] for row in a.tolist()]

        d = {
            "name": self.name,
            "cost": L(self.cost),
            "supply": [int(v) for v in self.supply],
            "demand": [int(v) for v in self.demand],
            "row_labels": list(self.row_labels),
            "col_labels": list(self.col_labels),
            "discount": copy.deepcopy(self.discount),
            "fixed_charge": L(self.fixed_charge),
            "capacity": L(self.capacity),
            "forbidden": L(self.forbidden),
            "min_ship": L(self.min_ship),
            "objective2": None if self.objective2 is None else {
                "name": self.objective2["name"], "matrix": L(self.objective2["matrix"]),
                "weight": self.objective2["weight"]},
            "noise": None if self.noise is None else {"sd": L(self.noise["sd"]), "k": self.noise["k"]},
        }
        meta = {k: v for k, v in self.meta.items() if k not in ("dummy", "orig_shape", "dummy_cells")}
        d.update(meta)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Problem":
        cost = np.array(d["cost"], dtype=float)
        shape = cost.shape
        known = {"name", "cost", "supply", "demand", "row_labels", "col_labels", "discount", "fixed_charge",
                 "capacity", "forbidden", "min_ship", "objective2", "noise"}
        o2 = d.get("objective2")
        if o2 is not None:
            o2 = dict(o2)
            o2["matrix"] = _mat(o2["matrix"], shape, 0.0)
        nz = d.get("noise")
        if nz is not None:
            nz = dict(nz)
            nz["sd"] = _mat(nz["sd"], shape, 0.0)
        forb = d.get("forbidden")
        return cls(
            name=d.get("name", "Custom"),
            cost=cost,
            supply=d["supply"],
            demand=d["demand"],
            row_labels=d.get("row_labels"),
            col_labels=d.get("col_labels"),
            discount=d.get("discount"),
            fixed_charge=_mat(d.get("fixed_charge"), shape, 0.0),
            capacity=_mat(d.get("capacity"), shape, INF),
            forbidden=None if forb is None else _mat(forb, shape, 0, dtype=float).astype(bool),
            min_ship=_mat(d.get("min_ship"), shape, 0.0),
            objective2=o2,
            noise=nz,
            meta={k: v for k, v in d.items() if k not in known},
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, s: str) -> "Problem":
        return cls.from_dict(json.loads(s))


INSTANCE_DIR = Path(__file__).resolve().parent.parent / "data" / "instances"


def load_instance(name: str) -> Problem:
    return Problem.from_json((INSTANCE_DIR / f"{name}.json").read_text())


def list_instances() -> list[dict]:
    out = []
    for p in sorted(INSTANCE_DIR.glob("*.json")):
        d = json.loads(p.read_text())
        out.append({"name": d["name"], "group": d.get("group", "Other"), "description": d.get("description", ""),
                    "order": d.get("order", 99)})
    return sorted(out, key=lambda r: (r["order"], r["name"]))
