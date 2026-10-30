# Transportation Problem Lab

**Operations Research project: E272 Akshat Katiyar, SVKM's NMIMS**

Solving **standard and non-standard transportation problems** with the classical method
(VAM + MODI) and with metaheuristics: **Differential Evolution (DE)** and a
**Hybrid DE + Q-Learning** algorithm. Everything runs in an interactive Streamlit dashboard.

---

## 1. The idea in short

- **VAM + MODI is exact and instant**, but only for the textbook problem: constant cost per unit,
  only supply and demand rules, one objective.
- **Real freight costs break those assumptions:** bulk discounts, a fixed cost for opening a route,
  route capacities and closed routes, minimum loads, a second objective (CO2), or uncertain costs.
  MODI's optimality test (Δ = c − u − v ≥ 0) no longer applies.
- **Differential Evolution** handles any of these, because it only needs to *calculate* the cost of
  a plan.
- **Standard DE uses one fixed F and CR** for the whole run. The **Hybrid** lets a Q-learning agent
  choose F and CR before every generation, from what has worked so far, at no extra cost.

### Reference paper

The idea of steering DE with reinforcement learning comes from:

> P. G. Giannopoulos, G. P. Chondrokoukis, T. K. Dasaklis (2024). *Hybrid Differential Evolution
> Algorithm Integrating Operations Research and Reinforcement Learning approaches.* Proceedings of
> the 28th Pan-Hellenic Conference on Progress in Computing and Informatics (PCI 2024), Egaleo,
> Greece, ACM, pp. 250–256. https://doi.org/10.1145/3716554.3716592

**What this project adopts:** Q-learning that adjusts the mutation (F) and crossover (CR) settings
of DE every generation, with the same Q-learning update rule and learning rate α = 0.1, binomial
crossover and differential mutation.

**What is different:** the paper tests continuous benchmark functions (Ackley, Griewank, Sum of
Squares); this project applies the idea to non-standard transportation problems. The paper's full
method also adds PROMETHEE II multi-criteria selection and simulated-annealing elitism; those are
not implemented here (future work). In the paper's terms, this project's Hybrid is a
"QL-enhanced DE".

Other foundations: Storn & Price (1997), the original DE; Watkins & Dayan (1992), Q-learning;
Holm (1979), the multiple-test correction used in the statistics.

---

## 2. Install and run

Requires **Python 3.10–3.13** (the study was run with Python 3.11).

```bash
git clone https://github.com/AkshatKatiyar081/transportation-problem-de-qlearning.git
cd transportation-problem-de-qlearning
python -m venv .venv
.venv\Scripts\activate            # Windows   (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
streamlit run app.py              # opens http://localhost:8501 in the browser
```

`requirements.txt` lists minimum versions. `requirements-lock.txt` lists the exact versions the
study was run with (`pip install -r requirements-lock.txt` to reproduce it exactly).

Run the automated tests (212 checks, about 1–2 minutes):

```bash
python -m pytest -q tests
```

No internet connection is needed once the packages are installed.

---

## 3. Using the dashboard

In the **sidebar** pick a problem and a method, then press **RUN**. There are three tabs:

| Tab | What it shows |
|-----|---------------|
| **Problem** | A problem statement (real-world scenario), a checklist of the standard transportation problem's assumptions and which ones this problem breaks, the mathematical formulation and the data. For *✎ My own problem* this tab is the **problem builder**. |
| **Solve** | The result of the chosen method: cost, gap to the reference value, time, the shipment plan (downloadable as CSV) and the convergence curve. Every run is added to a comparison table. Extra panels: MODI step by step (u, v, Δ), how the Hybrid chose F and CR, the cost vs CO2 trade-off (X3) and the risk simulation (X4). |
| **Algorithm Comparison** | **Run several seeds** runs chosen methods with several random seeds in one click, then shows the final cost of each run, all convergence curves, quality vs time, repeated-run statistics (mean / best / worst / runs within 1%, box plot) and a side-by-side comparison of two plans. |

**What is a seed?** DE is random; the seed fixes its random numbers, so the same seed always gives
the same run. Methods are compared over several seeds.

### Pre-loaded problems

| Name | Type | What it shows |
|------|------|---------------|
| **L10** | Standard 10×10 problem | VAM gives 10,981; MODI improves it in 4 steps to the proven optimum 10,634 in about 1 ms. The classical method is the right choice here. |
| **S1** | 12×12, bulk discounts + fixed truck cost | Hybrid 1.8% < DE-Random 2.9% < Standard DE 3.7% average gap |
| **S2** | 12×12, fixed truck cost | Hybrid 1.0% < DE-Random 1.7% < Standard DE 2.7%; 19/30 Hybrid runs within 1% vs 6/30 |
| **S3** | 10×10, high fixed costs + steep discounts | The VAM + MODI plan costs 26.6% more than the best-known plan; Hybrid 1.7% < DE-Random 2.9% < Standard DE 3.7% |
| **X1** | All-units discount + fixed charge | VAM + MODI plan 8,272 vs DE 8,104 |
| **X2** | Closed routes, route capacities, minimum loads | MODI cannot be applied; DE solves it |
| **X3** | Two objectives: cost vs CO2 | Pareto front of the trade-off (Solve tab) |
| **X4** | Uncertain costs (mean ± sd) | Robust plan; VAM + MODI plan 10,512 vs DE 9,961; risk simulation |

Other instances in `data/instances/` (A1, A3, A5, U1, L5, N5, N10, N5-pilot) are used by the tests
and the experimental study but are not listed in the dashboard.

### Entering your own problem

Choose **✎ My own problem** in the sidebar, set the number of sources and destinations, then type
or paste (from Excel) the costs, supplies and demands. Unbalanced problems get a dummy row or
column automatically. Optional non-standard features: incremental or all-units discounts, fixed
charges, closed routes, route capacities, minimum shipments, a second objective, uncertain costs.
The Problem tab says which methods apply. Problems can be downloaded and re-imported as JSON.

### Which method solves what

| Method | Applies to |
|--------|-----------|
| MODI (VAM + u-v) | Standard problems only: linear cost, no extra route rules, one objective |
| Naive plan | Problems whose only non-standard part is the cost: VAM + MODI on the linear version, then priced with the real costs (shows what ignoring the non-linearity costs) |
| Standard DE / DE-Random / Hybrid DE + QL | Every problem |

---

## 4. How the methods work

- **VAM + MODI:** VAM builds the starting plan; the MODI (u-v) test either proves it optimal
  (all Δ ≥ 0) or improves it step by step along a closed loop.
- **Encoding (decoder):** a DE individual holds one priority number (0–1) per route. The decoder
  fills routes from the highest priority down, shipping `min(supply left, demand left, capacity)`,
  so every individual becomes a plan that meets supply and demand. Closed routes and loads below
  the minimum are skipped; anything that cannot be shipped is penalised.
- **Standard DE:** DE/rand/1/bin. Mutation `v = a + F·(b − c)`, binomial crossover with rate CR
  (at least one gene from the mutant), and selection that keeps the child only if it is not worse
  than its parent. Fixed F = 0.5, CR = 0.9, population 50.
- **DE-Random (ablation):** F and CR drawn at random every generation from
  F ∈ {0.3, 0.6, 0.9} × CR ∈ {0.1, 0.5, 0.9}.
- **Hybrid DE + Q-Learning:** before every generation a Q-learning agent picks one of the same 9
  (F, CR) pairs.
  - State = success rate of the last generation (share of children that beat their parent):
    high > 10%, medium 5–10%, low < 5%.
  - Reward = success rate + 0.5 if the best plan improved.
  - Update: Q(s,a) ← Q(s,a) + α[r + γ·max Q(s′,·) − Q(s,a)], with α = 0.1, γ = 0.9.
  - ε-greedy exploration, ε falling from 1 to 0.05 over the first 40% of the run.
  - It uses exactly the same number of cost evaluations as Standard DE.
- **Polish (optional):** a local search with 2×2 loop moves, which helps with all-units discounts
  and minimum loads.
- **Reference values (for the gap %):** for standard problems the MODI optimum (proved by the u-v
  test); for non-standard problems a best-known value, the best of 10 long Hybrid + polish runs.
  A best-known value is not a proof of optimality.

---

## 5. Experimental study

All methods get the **same evaluation budget** (number of cost calculations), **30 independent
seeds** per method, and are compared with the paired **Wilcoxon signed-rank test** with **Holm**
correction plus effect sizes.

| | Standard DE | DE-Random | Hybrid DE + QL |
|---|---|---|---|
| S1 average gap | 3.65% | 2.86% | **1.78%** |
| S2 average gap | 2.68% | 1.74% | **0.96%** |
| S2 runs within 1% | 6/30 | 5/30 | **19/30** |
| S3 average gap | 3.74% | 2.87% | **1.73%** |
| L10 runs reaching the exact optimum | 13/30 | 19/30 | **26/30** |

- The Hybrid is significantly better than Standard DE on S1, S2, S3 and L10 (p_Holm ≤ 0.0006,
  large effect), and better than DE-Random on S2 and S3 (on S1 the difference is not significant).
- On L10 the Hybrid reaches the optimum with about 20% fewer evaluations than Standard DE.
- The Hybrid keeps the population diverse: about 47 of 50 plans are still different at the end on
  S2, against about 14 for Standard DE (premature convergence).
- For standard problems MODI is exact and far faster, so it should be used there; DE is the tool
  once the problem stops being standard.

### Reproducing the study

```bash
python data/generate.py --check            # checks the frozen problem files
python experiments/pilot.py                # tunes the Q-learning state thresholds on N5-pilot
python experiments/run_all.py --estimate   # projected run time
python experiments/run_all.py --study all  # main + showcase (30 seeds) + sensitivity studies
python experiments/analyse.py              # summary tables, statistics and figures
```

The committed outputs in `results/` (raw runs, convergence histories, Q-tables, statistics and
figures) were produced on the machine recorded in `results/machine.json`. `analyse.py` also writes
the summary tables (`summary.md`, `table1…6.md`), which are not included in this submission.

---

## 6. Folder guide

```
app.py                  Streamlit dashboard (entry point)
core/                   problem.py (problem model, cost evaluation, balancing, JSON), decoder.py
algorithms/             modi.py (VAM, MODI), de.py (Standard DE, DE-Random, Hybrid Q-learning),
                        local_search.py (polish), registry.py (one run() interface), result.py
ui/                     tabs.py, compare.py, builder.py (own-problem builder),
                        statements.py (problem statements), common.py
data/                   generate.py, instances/*.json (frozen problem data)
experiments/            pilot.py, run_all.py, analyse.py, stats.py, common.py
results/                study results: runs.csv, history.csv, hybrid_log.csv, qtables.csv,
                        stats*.csv, table*.csv, pilot*.{csv,json}, machine.json, figures/*.png
tests/                  test_core.py, test_solvers.py, test_app.py, test_statements.py
.streamlit/config.toml  dashboard theme
requirements.txt        minimum package versions
requirements-lock.txt   exact versions used for the study
```

---

## 7. Known limitations

- The test problems are synthetic and mid-sized (up to 12×12). Conclusions concern these problems.
- Gains of the Hybrid over Standard DE are about 1–2% of total cost.
- For non-standard problems there is no exact solver here, so gaps are measured against
  best-known values.
- The Hybrid's state thresholds were tuned on a separate pilot problem (N5-pilot).
- The decoder produces basic plans; for all-units discounts and minimum loads use the polish option.
- Run times depend on the computer; cost evaluations are the fair comparison measure.
