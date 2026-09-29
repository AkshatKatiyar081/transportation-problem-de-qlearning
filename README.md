# Transportation Problem Lab

Operations Research mini project: solving **standard and non-standard transportation problems** with
classical methods (VAM + MODI, LP, MILP) and metaheuristics (**Differential Evolution** and a
**Hybrid DE + Q-Learning** algorithm), with an interactive Streamlit dashboard.

---

## 1. Quick start (Windows / macOS / Linux)

```bash
cd or_project
python -m venv .venv
.venv\Scripts\activate            # Windows   (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
streamlit run app.py              # opens http://localhost:8501
```

Python 3.10 or newer. `requirements.txt` lists minimum versions. `requirements-lock.txt` lists the
exact versions the study was run with.

Run the tests (185+ checks, about 30 s):

```bash
python -m pytest -q tests
```

---

## 2. What the dashboard does

The **sidebar** picks the problem (pre-loaded example or ✎ custom), the method and its parameters.
**RUN** solves it.

| Tab | Contents |
|-----|----------|
| 📋 Problem | Problem data, detected type (standard / non-standard features), which methods apply and why. For a custom problem, this tab is the **problem builder**. |
| ▶️ Solve | Objective, cost, gap to the exact optimum, time, the shipment plan (downloadable CSV), the convergence curve, and method-specific views: MODI iteration stepper (u, v, Δ, loop, θ), MILP status and bound, Hybrid Q-table and action usage. |
| 📊 Compare | For study instances: the pre-computed 30-run comparison with statistical tests. For any problem: a **quick comparison** that runs all applicable methods now. Also contains the full study report. |
| 🤖 Hybrid insight | What the Q-learning agent learned (Q-table heatmap, actions over time, time spent in each state). |
| 🔢 Decoder | Step-by-step view of how DE's priority keys become a feasible shipment plan. |
| ✅ Check a plan | Type any plan to check feasibility and cost. For standard problems, MODI can start from that plan and step it to optimality (Lab Activity 5: 795 → 695). |
| ⚖️ Pareto & Risk | For problems with a 2nd objective: a weight sweep that draws the Pareto front. For uncertain costs: a Monte-Carlo comparison of the mean-cost plan vs robust plans. |

### Pre-loaded examples

| Name | Type | Notes |
|------|------|-------|
| A1, A3, A5 | Lab Exp. 5 activities | Optima Rs 3,850 (= 385 load-km), **149** (the manual says 150; the LP proves 149), 695 |
| L5, L10 | Generated linear 5×5, 10×10 | Seeds 42, 43 |
| N5, N10 | Same data + incremental quantity discount | First 30 units ×1.0, next 50 ×0.7, rest ×0.5 |
| N5-pilot | Discount 5×5 | Used only to tune the Q-learning thresholds |
| X1 | All-units discount (discontinuous) + fixed charge per route | |
| X2 | Forbidden routes + route capacities + minimum shipment (x = 0 or x ≥ 15) | |
| X3 | Bi-objective: cost vs CO2 | Try the Pareto tab |
| X4 | Uncertain unit costs (mean ± sd), robust objective mean + k·sd | Try the Risk tab |
| U1 | Unbalanced (supply > demand) | A dummy destination is added automatically |

### Giving your own problem (✎ Custom problem)

1. Choose **✎ Custom problem** in the sidebar.
2. Set the number of sources and destinations (2–10 each) and their names.
3. Type or **paste from Excel** the unit costs, supplies and demands. Unbalanced totals are allowed.
4. Optionally switch on non-standard features:

| Feature | How it is entered | Meaning |
|---------|-------------------|---------|
| Incremental discount | Tier table (limit, rate × base cost) | Each tier's units are charged at its rate (like tax slabs) |
| All-units discount | Same table | The whole shipment is charged at the rate of its tier, so the cost jumps (discontinuous) |
| Fixed charge | Matrix | Paid once if a route is used |
| Forbidden routes | Tick-box matrix | Route cannot be used |
| Route capacity | Matrix (empty = unlimited) | Maximum units on a route |
| Minimum shipment | Matrix | A used route must carry x ≥ L. This makes the feasible set discontinuous. |
| Second objective | Matrix + weight w | Minimise (1-w)·cost/Nc + w·obj2/N2 |
| Uncertain costs | sd matrix (or "sd = x% of cost") + k | Minimise expected cost + k · sd(total cost) |

5. Solve with any applicable method. The Problem tab tells you which methods apply.
6. **Download this problem (.json)** saves it, and *Start from… → Import* loads it again. You can also
   load any pre-loaded example into the builder and edit it, or generate a random problem.

### Which method can solve what

| Method | Applies to |
|--------|-----------|
| MODI (VAM + u-v) | Standard problems only (linear cost, no extra constraints, one objective) |
| LP (HiGHS) | Linear costs, plus capacities / forbidden routes / weighted 2nd objective |
| MILP (HiGHS) | Everything except the robust risk term (k > 0): discounts, fixed charges, min shipments, capacities, forbidden routes, 2nd objective |
| Naive plan | Non-standard problems: solves the linear version and prices the plan with the real costs (shows what ignoring the nonlinearity costs) |
| DE / DE-Random / Hybrid DE + QL | Everything. The optional local-search polish helps with all-units discounts and minimum shipments. |

---

## 3. How it works (short)

- **Encoding:** a DE individual holds one priority key per cell. The **decoder** fills cells from the
  highest key down, shipping `min(supply left, demand left, capacity)`. It skips forbidden routes and
  shipments below the minimum. For standard problems the result is always feasible, integer and
  basic (at most m+n-1 routes), and every basic solution can be reached. Supply that cannot be
  shipped because of route constraints gets a penalty.
- **DE:** DE/rand/1/bin with the midpoint bound rule and greedy selection (≤).
- **DE-Random:** (F, CR) drawn uniformly from a 3×3 grid every generation. This is the ablation baseline.
- **Hybrid DE + Q-Learning:**
  - State = success rate of the last generation (productive / slowing / stagnant).
  - 9 actions = (F, CR) ∈ {0.3, 0.6, 0.9} × {0.1, 0.5, 0.9}.
  - Reward = success rate + 0.5 if the best improved.
  - ε-greedy, with ε decaying linearly to 0.05 over the first 40% of the run.
- **Polish (optional):** rectangle cycle moves (±δ around a 2×2 loop), including jumps to tier
  breakpoints. This reaches non-basic optima.
- **MILP:** tier binaries for discounts, a usage binary for fixed charges and minimum shipments.
  Solved with `scipy.optimize.milp` (HiGHS). The exact optima of all examples were checked this way.

---

## 4. Reproducing the study

```bash
python data/generate.py --check                 # instances in data/instances/*.json are frozen
python experiments/pilot.py                     # tunes Q-learning thresholds on N5-pilot (seeds 100-104)
python experiments/run_all.py --estimate        # projected run time
python experiments/run_all.py --study all       # main (30 seeds) + sensitivity studies
python experiments/analyse.py                   # tables, Wilcoxon + Holm, effect sizes, figures
```

Outputs go to `results/`:
- `runs.csv`, `history.csv`, `hybrid_log.csv`, `qtables.csv`, `exact.csv`
- `table1..5*.csv/.md`, `stats.csv`, `summary.md`
- `figures/*.png` (for the report)

The committed results were produced on the machine recorded in `results/machine.json`, in about
15 minutes on 2 cores.

### Main findings (from `results/summary.md`)

- **Linear problems:** MODI and LP are exact and take under 1 ms. All DE variants also reach the
  optimum on A1, A3, A5 and L5 in 100% of runs, but need 50–1,700 fitness evaluations and are
  100–1000× slower.
- **Discounts change the optimal plan:** the naive linear plan is 1.64% (N5) and 0.32% (N10) more
  expensive than the MILP optimum.
- **Hybrid vs baselines:**
  - On the small instances (up to 5×5) there is no significant difference; every method solves them almost immediately.
  - On the 10×10 instances the Hybrid needs **significantly fewer evaluations** to reach the optimum
    than fixed DE (p_Holm < 0.001, large effect) and than DE-Random (p_Holm ≈ 0.02, medium/large effect).
    It succeeds in 26/30 (L10) and 30/30 (N10) runs, vs 13/30 and 24/30 for fixed DE.
- **What the agent learned:** the Q-table prefers **low CR (0.1)** actions. This agrees with the
  sensitivity study, where low-CR settings are best for this encoding.
- **Caveats:**
  - A fixed DE *tuned on the same instance* (F = 0.3, CR = 0.5) needs about 12k FE, which is fewer than
    the Hybrid's ~18k. The Hybrid's advantage is that it gets there **without tuning**.
  - Its own parameters barely matter: 15–21k FE for every α, γ and ε setting tested.
- **MILP vs DE:** at these sizes MILP is both exact and faster (0.8 s on N10 vs about 3 s for DE).
  DE is the tool for objectives MILP cannot express, such as the robust risk term on X4.

### Deviations from the written spec (and why)

| Spec | Implemented | Reason |
|------|-------------|--------|
| Discount tiers 10 / 50 at ×0.9 / ×0.8 | 30 / 80 at ×0.7 / ×0.5 | With the mild tiers the discounted optimum was *identical* to the linear one (naive gap 0%), so the instances could not show anything |
| FE budgets 5k / 25k / 100k | 2k / 5k / 50k | Every method solved the instances within ~1k FE, so the Q-agent spent its whole run in the "stagnant" state. Tighter budgets make the comparison informative. |
| Pilot grid SR_hi {0.15–0.25}, SR_lo {0.03–0.08} | {0.05, 0.10, 0.20} × {0.01, 0.02, 0.05} | With the original grid no pair gave every state ≥ 10% of generations. The chosen pair is (0.10, 0.05). |
| Sensitivity on N5, 30 seeds | On N10, 10 seeds | N5 is solved by every setting, so the heatmap would be flat |
| PuLP + CBC | `scipy.optimize.milp` (HiGHS) | One fewer dependency, and it returns the dual bound and MIP gap |
| Integer cost arithmetic (tenths) | Floats with 1e-6 relative tolerance | Needed once user-defined tier rates and noisy costs are allowed |

---

## 5. Project structure

```
core/         problem.py (Problem model, features, evaluation, balancing, JSON), decoder.py
algorithms/   modi.py (VAM, MODI, basis repair), exact.py (LP, MILP), de.py (DE, DE-Random, Hybrid QL),
              local_search.py (polish), registry.py (one run() interface), result.py
data/         generate.py, instances/*.json (frozen)
experiments/  pilot.py, run_all.py, analyse.py, stats.py, common.py
ui/           builder.py (custom problem builder), tabs.py, common.py
tests/        test_core.py, test_solvers.py, test_app.py
results/      pre-computed study results + figures
app.py        Streamlit dashboard
```

## 6. Known limitations

- **Flat regions:** many key vectors decode to the same plan. The "neutral fraction" diagnostic
  measures how often this happens.
- **Decoder:** it only produces basic plans. That is enough for linear and concave costs; for
  all-units discounts and minimum shipments use the polish option (or MILP).
- **Scope:** the instances are small and synthetic. The conclusions concern these instances.
- **Timing:** measured on one machine. Fitness evaluations are the primary comparison.
