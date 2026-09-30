"""Shared helpers for experiment scripts."""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402

from algorithms.registry import run  # noqa: E402
from core.problem import load_instance  # noqa: E402

RESULTS = ROOT / "results"
BUDGET = {"A1": 2000, "A3": 2000, "A5": 2000, "L5": 5000, "N5": 5000, "N5-pilot": 5000,
          "L10": 50000, "N10": 50000, "S1": 25000, "S2": 25000, "S3": 20000}
N_CHECKPOINTS = 100
_REF = {}


def reference(name):
    if name not in _REF:
        d = json.loads((ROOT / "data" / "instances" / f"{name}.json").read_text())
        _REF[name] = d.get("reference_optimum")
    return _REF[name]


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "no-git"


def parse_seeds(s):
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def first_fe(history, threshold):
    for fe, best in history:
        if best <= threshold + 1e-9:
            return fe
    return None


def checkpoints(history, budget):
    """Best-so-far at N_CHECKPOINTS evenly spaced FE values."""
    fes = np.array([h[0] for h in history])
    best = np.array([h[1] for h in history])
    pts = np.linspace(fes[0], max(budget, fes[-1]), N_CHECKPOINTS).round().astype(int)
    idx = np.searchsorted(fes, pts, side="right") - 1
    idx = np.clip(idx, 0, len(best) - 1)
    return list(zip(pts.tolist(), best[idx].tolist()))


def run_task(task):
    """task: dict(study, instance, method, seed, params, config_id).  Returns dict of rows."""
    prob = load_instance(task["instance"])
    budget = task["params"].get("fe_budget", BUDGET[task["instance"]])
    params = dict(task["params"], fe_budget=budget)
    t0 = time.perf_counter()
    r = run(task["method"], prob, params, seed=task["seed"])
    ref = reference(task["instance"])
    gap = None if ref is None else 100.0 * (r.objective - ref) / abs(ref)
    row = {
        "study": task["study"], "config_id": task.get("config_id", ""), "instance": task["instance"],
        "method": task["method"], "seed": task["seed"], "params_json": json.dumps(params, sort_keys=True),
        "cost": r.objective, "reference": ref, "gap_pct": gap,
        "success": None if ref is None else abs(r.objective - ref) / abs(ref) < 1e-6,
        "fe_used": r.fe_used,
        "fe_to_target": None if ref is None else first_fe(r.history, ref * 1.005),
        "fe_to_opt": None if ref is None else first_fe(r.history, ref * (1 + 1e-6)),
        "time_s": time.perf_counter() - t0,
        "neutral_frac_mean": r.diagnostics.get("neutral_frac_mean"),
        "distinct_plans_final": r.diagnostics.get("distinct_plans_final"),
        "status": r.status,
    }
    for k in ("F", "CR", "alpha", "gamma", "eps_frac", "sr_hi", "sr_lo"):
        if k in params:
            row[k] = params[k]
    hist = [{"study": task["study"], "config_id": task.get("config_id", ""), "instance": task["instance"],
             "method": task["method"], "seed": task["seed"], "fe": fe, "best_cost": b}
            for fe, b in checkpoints(r.history, budget)]
    log, qrows = [], []
    if r.method == "hybrid" and task["study"] in ("main", "pilot", "showcase"):
        best_by_gen = [h[1] for h in r.history[1:]]
        from algorithms.de import ACTIONS
        for g, (s, a, rw, sr) in enumerate(zip(r.states, r.actions, r.rewards, r.sr)):
            log.append({"study": task["study"], "config_id": task.get("config_id", ""),
                        "instance": task["instance"], "seed": task["seed"], "generation": g, "state": s,
                        "action": a, "F": ACTIONS[a][0], "CR": ACTIONS[a][1], "SR": sr, "reward": rw,
                        "best_cost": best_by_gen[g]})
        for s in range(r.q_table.shape[0]):
            for a in range(r.q_table.shape[1]):
                qrows.append({"study": task["study"], "config_id": task.get("config_id", ""),
                              "instance": task["instance"], "seed": task["seed"], "state": s, "action": a,
                              "q": float(r.q_table[s, a])})
    return {"run": row, "history": hist, "log": log, "q": qrows}


def machine_info():
    info = {"os": platform.platform(), "python": platform.python_version(), "cpu": platform.processor() or
            platform.machine(), "cores": os.cpu_count(), "date": time.strftime("%Y-%m-%d %H:%M")}
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                info["cpu"] = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / 2 ** 30, 1)
    except Exception:
        try:
            info["ram_gb"] = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2 ** 30, 1)
        except Exception:
            info["ram_gb"] = None
    import numpy
    import pandas
    import scipy
    info.update(numpy=numpy.__version__, scipy=scipy.__version__, pandas=pandas.__version__)
    return info
