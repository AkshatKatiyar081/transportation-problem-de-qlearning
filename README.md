# Transportation Problem Lab

Operations Research mini project: solving **standard and non-standard transportation problems** with
the classical method (VAM + MODI) and metaheuristics (**Differential Evolution** and a
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

Run the tests (about 200 checks, 1-2 minutes):

```bash
python -m pytest -q tests
```

---

## 2. What the dashboard does

In the **sidebar** you pick a problem (a pre-loaded example or ✎ *My own problem*) and a method,
then press **RUN**. There are three tabs:

| Tab | What you see |
|-----|--------------|
| Problem | The cost table, supplies and demands, and the problem type. For your own problem, this tab is the **problem builder**. |
| Solve | The result of the selected method: cost, gap, time, shipment plan (downloadable) and the convergence curve. Below it, a **comparison table** gets one new row for every RUN on this problem. Depending on the method and problem, collapsible sections add MODI step by step (u, v, Δ, loop, θ), how the Hybrid chose F and CR, the Pareto front (2nd objective), the risk simulation (uncertain costs) and how DE decodes a plan. |
| Algorithm Comparison | Compares every run made on the current problem. **Run several seeds** runs chosen methods with seeds 0, 1, 2, … in one click. Four views: (1) final result of each run and all convergence curves, (2) solution quality vs computation time, (3) repeated-run statistics (mean / best / worst / std, runs within 1%, box plot), (4) side-by-side comparison of any two plans with a cost breakdown and the routes that differ. |

**What is a seed?** DE is random. The seed fixes its random numbers, so the same seed always gives
the same run. Different seeds give slightly different results, so methods should be compared over
several seeds.

**Suggested demo (5 minutes):**
1. **A5, MODI, start from the lab's proposed plan:** the cost goes from 795 to 695 step by step.
2. **S2 → Algorithm Comparison → Run several seeds** (3 DE methods × 5 seeds, about 20 s). The box
   plot and the statistics table show Hybrid < DE-Random < Standard DE.
3. **S3 → Algorithm Comparison → Run several seeds** (naive plan + 3 DE methods × 5 seeds, about
   20 s). The naive MODI plan costs 20-25% more than the DE plans, and Hybrid is the cheapest.
4. **Solve tab:** every run is listed in the comparison table.
5. **X3 → Solve → Trade-off**, and **X4 → Solve → Risk**.
6. **✎ My own problem:** type in any problem the examiner gives you.

### Pre-loaded examples

The **Problem** tab describes each example as a problem statement: a short real-world scenario, a
network diagram of sources and destinations, a checklist of the standard transportation problem's
assumptions (and which ones this problem breaks), the mathematical formulation, and the data. For
non-standard problems it also shows what the naive MODI plan would cost.

| Name | Type | Notes |
|------|------|-------|
| A5 | Lab Exp. 5, Activity 5: is the proposed plan optimal? | MODI improves the lab's plan from 795 to the optimum 695 |
| L10 | Standard 10×10 problem | MODI is exact in about 1 ms; DE variants need 50,000 evaluations and reach the optimum in 43–87% of runs |
| **S1** | **Showcase:** 12×12 bulk discounts + fixed truck cost per route | Hybrid 1.8% < DE-Random 2.9% < DE 3.7% average gap |
| **S2** | **Showcase:** 12×12 fixed-charge transportation problem | Hybrid 1.0% < DE-Random 1.7% < DE 2.7% (30-run study), all differences significant |
| **S3** | **Showcase:** 10×10 high fixed costs (500–2,500 per route) + steep bulk discounts: naive MODI plan vs DE variants | Naive plan Rs 37,635 (+26.6%) ≫ DE 3.7% > DE-Random 2.9% > Hybrid 1.7% average gap (30-run study); Hybrid significantly better than both DE variants |
| X1 | All-units discount (discontinuous) + fixed charge per route | |
| X2 | Forbidden routes + route capacities + minimum shipment (x = 0 or x ≥ 15) | |
| X3 | Bi-objective: cost vs CO2 | See Solve → Pareto front |
| X4 | Uncertain unit costs (mean ± sd), robust objective mean + k·sd | See Solve → Risk |

Other instances are kept in `data/instances/` for the tests and the experiment study but are not
listed in the dashboard: the lab activities A1 and A3 (optima Rs 3,850 = 385 load-km and **149**; the
manual says 150, MODI's u-v test proves 149), U1 (unbalanced; a dummy row or column is added
automatically, also for your own problems), and the research instances L5, N5, N10, N5-pilot.

**Why S1–S3 separate the methods and the small examples do not:** on small problems (up to 10×10
with mild costs) all three variants reach the optimum, so there is nothing to see. S1 and S2 are
larger (144 routes) and rugged: fixed charges create many local optima. With a fixed budget of
25,000 evaluations, how well F and CR are chosen then decides how close each variant gets. S3
(10×10, 20,000 evaluations) makes the fixed charges large compared with the unit costs and the
discounts steep. The best plan puts the flow on a few big routes. MODI only sees the unit costs, so
the naive plan spreads the flow over many cheap-per-unit routes and pays their fixed costs.

### Giving your own problem (✎ My own problem)

1. Choose **✎ My own problem** in the sidebar.
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
| Naive plan | Problems whose only non-standard part is the cost (discounts, fixed charges, min shipments, noise): solves the linear version with MODI and prices the plan with the real costs (shows what ignoring the nonlinearity costs) |
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
- **Reference values (for the gap %):**
  - Standard problems: the MODI optimum. It is proved optimal by the u-v test, and the tests check
    it independently with a duality certificate.
  - Non-standard problems: a **best-known value**, the best of 10 long Hybrid DE + polish runs
    (`python data/generate.py --verify`). A best-known value is not a proof of optimality.

---

## 4. Reproducing the study

```bash
python data/generate.py --check                 # instances in data/instances/*.json are frozen
python experiments/pilot.py                     # tunes Q-learning thresholds on N5-pilot (seeds 100-104)
python experiments/run_all.py --estimate        # projected run time
python experiments/run_all.py --study all       # main + showcase (30 seeds) + sensitivity studies
python experiments/analyse.py                   # tables, Wilcoxon + Holm, effect sizes, figures
```

Outputs go to `results/`:
- `runs.csv`, `history.csv`, `hybrid_log.csv`, `qtables.csv`, `baselines.csv` (MODI / naive plan)
- `table1..6*.csv/.md`, `stats.csv`, `stats_showcase.csv`, `summary.md`
- `figures/*.png` (for the report)

The committed results were produced on the machine recorded in `results/machine.json`, in about
15 minutes on 2 cores.

### Main findings (from `results/summary.md`)

- **Showcase S1/S2 (12×12, fixed budget of 25,000 evaluations) and S3 (10×10, 20,000 evaluations),
  30 paired runs each:**

  | | Standard DE | DE-Random | Hybrid DE + QL |
  |---|---|---|---|
  | S1 average gap | 3.65% | 2.86% | **1.78%** |
  | S2 average gap | 2.68% | 1.74% | **0.96%** |
  | S2 runs within 1% | 6/30 | 5/30 | **19/30** |
  | S3 average gap | 3.74% | 2.87% | **1.73%** |
  | S3 best / worst (Rs) | 29,720 / 31,520 | 29,793 / 31,171 | **29,720 / 31,074** |
  | S3 runs within 1% | 3/30 | 3/30 | **13/30** |

  - The Hybrid is significantly better than Standard DE on all three (S1, S2 p_Holm < 0.001; S3
    p_Holm = 0.0005; large effect).
  - It is significantly better than DE-Random on S2 (p_Holm < 0.001) and S3 (p_Holm = 0.008). On S1 the
    difference points the same way but is not significant (p_Holm ≈ 0.06, medium effect).
  - S3's naive plan (MODI on the linear costs, priced with the real costs) costs Rs 37,635, 26.6% above
    the best-known 29,720. All 30 runs of every DE variant beat it (Hybrid vs naive: one-sample
    Wilcoxon p = 1.6e-06).

- **Linear problems:** MODI is exact and takes under 1 ms. All DE variants also reach the
  optimum on A1, A3, A5 and L5 in 100% of runs, but need 50–1,700 fitness evaluations and are
  100–1000× slower.
- **Discounts change the optimal plan:** the naive linear plan is 1.64% (N5) and 0.32% (N10) more
  expensive than the best-known plan.
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
- **MODI vs DE:** for standard problems MODI is exact and far faster, so use it. DE is the tool
  once the problem stops being standard: discounts, fixed charges, route constraints, a second
  objective, or the robust risk term on X4.

### Deviations from the written spec (and why)

| Spec | Implemented | Reason |
|------|-------------|--------|
| Discount tiers 10 / 50 at ×0.9 / ×0.8 | 30 / 80 at ×0.7 / ×0.5 | With the mild tiers the discounted optimum was *identical* to the linear one (naive gap 0%), so the instances could not show anything |
| FE budgets 5k / 25k / 100k | 2k / 5k / 50k | Every method solved the instances within ~1k FE, so the Q-agent spent its whole run in the "stagnant" state. Tighter budgets make the comparison informative. |
| Pilot grid SR_hi {0.15–0.25}, SR_lo {0.03–0.08} | {0.05, 0.10, 0.20} × {0.01, 0.02, 0.05} | With the original grid no pair gave every state ≥ 10% of generations. The chosen pair is (0.10, 0.05). |
| Sensitivity on N5, 30 seeds | On N10, 10 seeds | N5 is solved by every setting, so the heatmap would be flat |
| Integer cost arithmetic (tenths) | Floats with 1e-6 relative tolerance | Needed once user-defined tier rates and noisy costs are allowed |

---

## 5. Project structure

```
core/         problem.py (Problem model, features, evaluation, balancing, JSON), decoder.py
algorithms/   modi.py (VAM, MODI, basis repair), de.py (DE, DE-Random, Hybrid QL),
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
  all-units discounts and minimum shipments use the polish option.
- **Scope:** the instances are small and synthetic. The conclusions concern these instances.
- **No optimality proof for non-standard problems:** without an exact solver, gaps there are measured
  against best-known values, which could in principle be improved.
- **Timing:** measured on one machine. Fitness evaluations are the primary comparison.
