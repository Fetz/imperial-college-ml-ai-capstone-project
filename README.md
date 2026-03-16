# Black-box optimisation challenge

Capstone project for the Professional Certificate in Machine Learning and AI from [Imperial College Executive Education](https://web.archive.org/web/20260201194625/https://www.imperial.ac.uk/business-school/executive-education/technology-analytics-data-science/professional-certificate-machine-learning-and-artificial-intelligence-programme/online/).

## Status - 🟡

This capstone is currently a work in progress.

## Overview

This is my work for the Capstone project on Black-Box Optimisation (BBO) challenge, this challenge is based on similar BBO challenges that happens in Kaggle like NeurIPS 2020 which ran from July–October, 2020.

This challenge involves optimising 8 unknown functions with dimensions from 2 to 8 dimensions with initially a pair 10 to 15 data points and output for each function and every week I will submit one data point for each function and I will receive their correspondent output for 13 weeks.

My personal objective is to deepen my knowledge of ML engineering, experiment with different ML frameworks, and use AI tools that can enhance my Software Engineering career.

## Why BBO is relevant?

BBO is relevant for ML scenarios where the objective/true function is expensive to evaluate or has no closed-form expression (e.g: hyperparameter tuning, drug discovery, materials design, or neural architecture search).

## Project Goal

Find the maximum of each function with as few queries as possible.

## Inputs & Outputs

### 1. Initial data:
Will be provided in binary `.npy` format:
- Structure: 8 pairs of (`function_x/initial_inputs.npy`, `function_x/initial_outputs.npy`).
- Dimensions: Inputs are matrices $X \in \mathbb{R}^{m \times d}$, where $10 \leq m \leq 15$ and $2 \leq d \leq 8$.
- Vectors: $y \in \mathbb{R}^m$, containing signed floating-point values.

### 2. Weekly delta:
Updates are delivered via text `.txt` format. Contrary to initial data, weekly updates will contain exactly 1 output data point per function (8 output data points).

#### A. `input.txt`

A serialized list of NumPy arrays. Each array represents a single coordinate in the domain of one of the eight functions.
- List Index + 1 = Function
- Array Shape: (m, n) where $10 \leq m \leq 15$ and $2 \leq d \leq 8$
- Value Range: [0.000000, 0.999999]
- Value Precision: Maximum 6 decimal places.
- Values Types: Standard Python Float or numpy.float64

#### B. `output.txt`

A serialized list of scalar values corresponding to the function evaluations of the inputs in `input.txt`.
- List Index: Function = Index + 1
- Format: [np.float64(y_1), np.float64(y_2), ...]
- Value Range: $y \in (-\infty, \infty)$ (Supports both positive and negative real numbers).
- Value Type: Standard Python Float or numpy.float64

#### C. Submission

Every week we get to submit 1 data point for each function that we want to know to know its output, the submission needs to be in the following format: $x1-...-$xn, where:
- $x needs to start with `0`
- $x has exactly 6 decimal places
- each $x needs to be separated with `-`

## Objective

**Maximise the output of each unknown function** with limited data points and limited to 13 queries.

Constraints:
- Function: Unknown shape → can't use gradient-based methods, must use surrogate models
- Function observation: limited to 10 to 15 known data points
- Number of queries: Limited to 13 → each query must be carefully chosen to maximise information
- Response: Delayed → can't do real-time adaptive strategies within a week

## Technical approach

- Week 1: Initial exploration using random/grid-based sampling to build a baseline understanding of each function's landscape.
- Week 2: Fitted GP surrogates with per-function kernel choices (RBF for smooth functions, Matern with varying nu for rougher ones). Used EI and UCB acquisition
  functions with ad-hoc kappa/xi values. Identified feature importance via ARD length scales and correlation analysis. Used dense meshgrids for candidate
  generation.
- **Week 3**:
  - Preprocessing: Replaced StandardScaler with QuantileTransformer for functions with extreme outliers (e.g., function 1 spans ~120 orders of magnitude). Added log-space as a second representation.
  - Models: Introduced dual-GP ensembles (one on QT output, one on log-space). Standardised on Matern(nu=2.5) with ARD across all functions.
  - SVMs: Added SVC (RBF kernel, C=10, soft-margin) to classify regions as promising vs not-promising. Used P(promising) as a multiplier on UCB scores to suppress penalty/cliff regions. SVR initially tested as a secondary surrogate alongside the GP.
  - Acquisition: Switched entirely from EI to UCB(kappa=5.0). Replaced meshgrids with Latin Hypercube Sampling, scaling candidates with dimensionality (10K–100K).
  - Feature selection: Used ARD to identify and drop irrelevant dimensions (e.g., function 8 reduced from 8D to 6D after identifying x6 and x8 as noise).
- **Week 4**: Introduced SVR as a second surrogate in the ensemble alongside the GP. Budget-aware kappa formula `5.0 - ((week-1)/13)*3.0` introduced to automatically shift from exploration to exploitation across the 13-week horizon.
- **Week 5**:
  - Fn1: Removed log10 transform (float64 precision collapse — shift constant ~1.004 dwarfs 12 of 14 values, leaving only 3 distinct log values). Simplified to GP+SVR ensemble; kappa raised to 8.0 (exploration override).
  - Fn2: kappa raised 2.0→4.0 to prevent boundary-chasing.
  - Fn4: Switched from StandardScaler to QuantileTransformer — outlier at y=−37.5 was skewing GP covariance.
  - Fn7: x3 identified as noise dimension via ARD (length_scale capped at 100). Dropped to 5D active subspace; x3 fixed at training mean during LHS search.
- **Week 6**:
  - Fn2: Switched from UCB to Expected Improvement (EI) acquisition — UCB was repeatedly recommending x1≈0.9999 (boundary), EI anchors to y_best and avoids low-predicted corners.
  - Fn3: Fixed SVM threshold from hardcoded `yi > −50` (all points labelled promising, SVM constraint inactive) to `yi > np.median(y_log_pos)` for proper 50/50 split.
  - Fn7: Noise detection threshold changed from absolute `ls > 100` to relative `ls > mean + 2×std` with weak `GammaPrior(1, 0.1)` prior; x3 still confirmed as noise dim.
  - Fn8: Re-evaluated active subspace with the same relative threshold — only x8 identified as noise; 7D active space (down from 6D in Week 5 which also dropped x6).
- **Week 7**:
  - **Fn7, Fn8 — Gradient-based acquisition**: Replaced LHS argmax with multi-start gradient ascent through GPyTorch surrogate. x=sigmoid(z) reparameterisation enforces [0,1] bounds; Adam 200 steps from 64 LHS-seeded starts; fast_pred_var() disabled during loop for exact variance gradients.
  - **Fn3 — length_scale_bounds raised 50→1e3**: Freed x1 lengthscale to confirm it as noise (ls hit 1000 ceiling, ConvergenceWarning). x1 confirmed noise → dropped to 2D active subspace (x2, x3); x1 fixed at training mean.
  - **Fn1 — kappa reduced 8.0→5.0**: Three consecutive exploration queries (W5–W7) covered top-right/centre/top-left with no improvement; switching to exploit known signal near [0.483, 0.479]. Warm-start cluster (±0.05, 1K candidates) added.
  - **Fn2 — EI boundary fix**: LHS candidates clipped to [0.05, 0.95] to prevent EI oscillating between opposite corners (x2→1 W5, x2→0 W7 pattern). Exploit cluster ±0.015 around best known added.
  - **Fn4, Fn5 — Warm-start clusters**: Oracle history revealed sharp peaks ([0.433, 0.425, 0.379, 0.388] for Fn4; [0.120, 0.863, 0.880, 0.958] for Fn5). Dense cluster (±0.02, 1K candidates) added around each best known point.
  - **Fn8 exploitation shift**: Not applied — W6 oracle returned 9.889 (improvement confirmed); budget-aware kappa schedule sufficient.
- **Week 8 (planned)**:
  - **Fn1**: If W7 returns no improvement, replace LHS with tight grid (±0.02) around [0.483, 0.479] — surrogate exhausted, probe oracle directly.
  - **Fn2**: If W7 returns ≤ 0.611, accept as likely global max or try final tight probe near [0.703, 0.927]; consider retiring EI.
  - **Fn4**: Force warm-start cluster override — bypass LHS entirely; GP length scales (~1.5) too large to guide away from noise, only oracle-confirmed peak region trustworthy.
  - **Fn5**: If W7 beats 1630, exploit that neighbourhood; otherwise tighten cluster around [0.120, 0.863, 0.880, 0.958].
  - **Fn7, Fn8**: Assess W7 oracle results for boundary behaviour; add proximity constraint if gradient ascent pushed into data-sparse corner with no improvement.
  - **Fn3**: Remove SVM (20/21 support vectors, constraint inactive). Run clean 2D GP + UCB in (x2, x3) active subspace.

### Exploration vs exploitation strategy

My overall strategy prioritises exploration in early weeks and shifts toward exploitation as data accumulates. UCB's kappa controls global exploration; the SVM P(promising) constraint provides exploitation bias by steering candidates away from penalised regions.

# Implementation

## Tools and Libraries

| Tool | Role |
|------|------|
| **NumPy** | Array operations, data loading (`.npy`), candidate manipulation |
| **SciPy** (`stats.qmc.LatinHypercube`) | Candidate generation — space-filling LHS samples (10K–100K per function) |
| **scikit-learn** | `GaussianProcessRegressor` (Matérn ARD kernel, MLE optimisation), `SVC`/`SVR` (RBF, SVM constraint & SVR surrogate), `QuantileTransformer`/`MinMaxScaler` |
| **GPyTorch** | GPU-accelerated GPs with ARD kernel and `GammaPrior` lengthscale priors; used in Fn1, Fn4, Fn7, Fn8 — Fn2, Fn3, Fn5, Fn6 use scikit-learn `GaussianProcessRegressor` |
| **Matplotlib / Seaborn** | Surrogate surfaces, acquisition function contours, SVM boundaries via `utils/plotting_utils.py` |
| **SHAP** | Feature importance analysis inside `plotting_utils.py` |
| **Pandas** | Tabular summaries and data inspection |
| **Jupyter Notebooks** | Per-function weekly experimentation and documentation |

For the per-function surrogate configuration (active dimensions, output transforms, kappa values) see [`notebooks/README.md`](notebooks/README.md).

## How to run

### Setup

```bash
# Install all dependencies (including PyTorch/GPyTorch)
pip install -e '.[torch]'

# Install core dependencies only
pip install -e .

# Launch Jupyter
jupyter notebook notebooks/
```

### Weekly workflow

```bash
# 1. Add new week data (merges delta input/output files into data/week_N/)
python notebooks/.scripts/create_weekly_data.py <week>

# 2. Scaffold notebooks for the new week (copies previous week as template)
python notebooks/.scripts/create_weekly_notebooks.py <week>

# 3. Run each notebook in Jupyter, then extract submission points
python3 notebooks/.scripts/get_submission.py <week>

# 4. Write submission points directly to the week's submission file
python3 notebooks/.scripts/get_submission.py <week> --write

# 5. Generate cross-week results summary (print preview)
python3 notebooks/.scripts/get_submission.py --results

# 5b. Write cross-week results summary to RESULTS.md
python3 notebooks/.scripts/get_submission.py --results --write
```

### Experimental (devcontainer)
```
devpod up .
```
Dependencies are installed automatically via `postCreateCommand` in `.devcontainer/devcontainer.json`.

## References

| Reference | Relevance to this project |
|---|---|
| [Rasmussen & Williams (2006). *Gaussian Processes for Machine Learning*. MIT Press.](http://www.gaussianprocess.org/gpml/) | Foundational GP theory; Matérn-2.5 ARD kernel choice; GP as optimal surrogate for small-N, unknown-smoothness functions |
| [Srinivas et al. (2010). *Gaussian Process Optimization in the Bandit Setting: No Regret and Experimental Design*. ICML.](https://arxiv.org/abs/0912.3995) | Theoretical basis for GP-UCB; justifies the budget-aware kappa schedule `5.0 - ((week-1)/13)*3.0` for exploration–exploitation balance |
| [Jones, Schonlau & Welch (1998). *Efficient Global Optimization of Expensive Black-Box Functions*. Journal of Global Optimization.](https://doi.org/10.1023/A:1008306431147) | Canonical reference for Expected Improvement (EI); directly used in Fn2 (Week 6) after UCB caused repeated boundary recommendations |
| [Wilson et al. (2018). *Maximizing Acquisition Functions for Bayesian Optimization*. NeurIPS.](https://arxiv.org/abs/1805.10196) | Basis for the gradient-based acquisition used in Fn7 and Fn8 (Week 7): multi-start Adam optimisation with sigmoid reparameterisation `x = sigmoid(z)` enforcing [0,1] bounds |
| [Frazier (2018). *A Tutorial on Bayesian Optimization*. arXiv:1807.02811.](https://arxiv.org/abs/1807.02811) | Survey covering EI, UCB, and PI trade-offs; background for the UCB→EI switch for Fn2 and the exploration–exploitation kappa schedule |
| [Letham et al. (2019). *Constrained Bayesian Optimization with Noisy Experiments*. Bayesian Analysis.](https://arxiv.org/abs/1706.07094) | Conceptual basis for using a classifier as an acquisition constraint; note: Letham uses GP-based constraints whereas this project uses an SVM multiplier (`constrained_ucb = ucb_shifted × svm_proba`) |
| [Eriksson et al. (2019). *Scalable Global Optimization via Local Bayesian Optimization* (TuRBO). NeurIPS.](https://arxiv.org/abs/1910.01739) | Motivates local exploitation around the best known point; the warm-start clusters used here (dense uniform jitter ±0.02–0.05 around the best training point for Fn1, Fn2, Fn4, Fn5) are a simplified form of this idea, not a full TuRBO implementation |
| [Cortes & Vapnik (1995). *Support-Vector Networks*. Machine Learning.](https://doi.org/10.1007/BF00994018) | Foundational SVM reference; backs the C parameter choices — C=1 (Fn3, Fn7) to prevent boundary collapse in high-D low-N regimes; C=10 (Fn1, Fn5, Fn6) where a stricter margin is needed. Fn2, Fn4, Fn8 use no SVM |
| [McKay, Beckman & Conover (1979). *A Comparison of Three Methods for Selecting Values of Input Variables*. Technometrics.](https://doi.org/10.1080/00401706.1979.10489755) | Basis for Latin Hypercube Sampling as space-filling candidate generation (10K–100K per function) |
| [Neal (1996). *Bayesian Learning for Neural Networks*. Springer.](https://www.cs.toronto.edu/~radford/ftp/thesis.pdf) | ARD (Automatic Relevance Determination) lengthscale interpretation used for noise-dimension detection in Fn3, Fn7, and Fn8 |
| [Bergstra & Bengio (2012). *Random Search for Hyper-Parameter Optimization*. JMLR.](https://jmlr.org/papers/v13/bergstra12a.html) | Motivates ignoring low-importance dimensions: shows that effective dimensionality is often much lower than nominal dimensionality, supporting the decision to fix noise dims at their training mean rather than searching over them. The ARD detection method itself is grounded in Neal (1996) |

## Additional Sources for Ongoing Refinement

### Contingency research — reach for these before adding more ad-hoc fixes

| Paper | Trigger condition |
|---|---|
| [Hennig & Schuler (2012). *Entropy Search for Information-Efficient Global Optimization*. JMLR.](https://jmlr.org/papers/v13/hennig12a.html) | **Fn1 / Fn2 still stuck after Week 8** — information-theoretic acquisition that targets the location of the maximum directly, not just high-σ regions; more appropriate than UCB/EI when the landscape is flat or the surrogate has uniformly low confidence |
| [Eriksson & Jankowiak (2021). *High-Dimensional Bayesian Optimization with Sparse Axis-Aligned Subspaces* (SAASBO). UAI.](https://arxiv.org/abs/2103.00349) | **Fn4 / Fn5 warm-start clusters don't improve best y** — sparse ARD prior that aggressively shrinks the effective search space; GP length scales ~1.5 on Fn4 are too large to guide acquisition toward narrow peaks, SAASBO's prior would force sparser solutions |
| [Gardner et al. (2014). *Bayesian Optimization with Inequality Constraints*. ICML.](http://proceedings.mlr.press/v32/gardner14.html) | **SVM constraint becomes inactive or collapses** (20+ support vectors, <5% promising coverage) — replaces the SVM P(promising) multiplier with a GP-modelled constraint probability; smoother, uncertainty-aware, and differentiable so it can be included in the gradient ascent objective rather than applied post-hoc |

### Planned software migration
| Library | Status | Why it's relevant |
|---|---|---|
| [BoTorch](https://botorch.org/) | Planned — not yet used | Built on GPyTorch (already used from Week 7); provides `optimize_acqf` with `LogEI` and `qLogNEI` natively, replacing the manually implemented Adam + sigmoid acquisition loop in Fn7/Fn8. Direct migration path: swap the gradient ascent loop for `optimize_acqf` |

## Structure

```
.
├── .devcontainer/          # Devpod/VSCode dev container config (Ubuntu + Nix + Neovim)
│   ├── devcontainer.json
│   ├── setup.sh            # Installs Nix packages, Neovim plugins, project dependencies
│   └── nvim/               # Neovim configuration
├── notebooks/              # Weekly per-function notebooks
│   ├── week_{N}_function_{M}.ipynb   # N=1..7, M=1..8
│   ├── utils/
│   │   └── plotting_utils.py         # Shared visualisation helpers
│   ├── data/
│   │   ├── initial_data/   # Original 10–15 point datasets (excluded from repo)
│   │   ├── updates/        # Raw weekly delta files (input.txt / output.txt)
│   │   └── week_{N}/       # Merged datasets per week, generated by .scripts/
│   └── .scripts/
│       ├── create_weekly_data.py       # Merges delta updates into week_N/ data folders
│       ├── create_weekly_notebooks.py  # Scaffolds new week notebooks from the previous week
│       └── get_submission.py           # Extracts X_next from executed notebooks → submission format
├── submissions/            # Weekly submission records (Week_1.md … Week_N.md)
├── pyproject.toml          # Project dependencies and package config
├── RESULTS.md              # Auto-generated: best y per function per week + submission audit trail
└── README.md
```