from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SolverResult:
    method: str
    objective: float | None            # value minimised (cost, robust cost, or weighted sum)
    cost: float | None                 # transport cost of the plan (incl. discounts / fixed charges)
    matrix: np.ndarray | None          # plan on the ORIGINAL problem shape (dummy removed)
    time_s: float = 0.0
    status: str = "ok"                 # ok | optimal | time_limit | infeasible | not_applicable | error
    message: str = ""
    feasible: bool = True
    components: dict = field(default_factory=dict)
    fe_used: int | None = None
    history: list | None = None        # [(FE, best objective)] per generation
    actions: list | None = None
    states: list | None = None
    rewards: list | None = None
    sr: list | None = None
    q_table: np.ndarray | None = None
    trace: list | None = None          # MODI iterations
    diagnostics: dict = field(default_factory=dict)
