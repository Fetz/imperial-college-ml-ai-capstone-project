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
- **Week 7 (planned)**:
  - **Gradient-based acquisition optimisation (Fn7, Fn8)**: Replace LHS argmax with multi-start gradient ascent directly through the GPyTorch surrogate. With 50K candidates in 5D/7D, average LHS spacing is ~17–27% of the range per dimension — a smooth Matern 2.5 GP may have peaks between samples. Implementation: reparameterise input as `x = sigmoid(z)` to enforce `[0,1]` bounds without killing gradients, run Adam for 200 steps from 64 LHS-seeded starts, take the global argmax. Note: disable `fast_pred_var()` during the optimisation loop (it breaks autograd); re-enable for final batch predictions. Applies only to Fn7 and Fn8 which already use GPyTorch; sklearn-based functions (Fn2, Fn3, Fn5, Fn6) are not differentiable and keep LHS argmax.
  - **Fn3**: Raise `length_scale_bounds` upper limit from 50 → 1e3 in `gp_qt` kernel — x1 is hitting the wall (ls=50, ConvergenceWarning). x1 appears to be a noise dimension; loosening the bound lets the optimiser confirm this cleanly. Consider dropping x1 and running LHS in 2D active subspace (x2, x3).
  - **Fn8 exploitation shift**: If week 6 query returns poor y (well below current best 9.889), reduce kappa toward 2.0 and add a proximity weight (penalise candidates far from current best) to exploit the known-good region.

### Exploration vs exploitation strategy

My overall strategy prioritises exploration in early weeks and shifts toward exploitation as data accumulates. UCB's kappa controls global exploration; the SVM P(promising) constraint provides exploitation bias by steering candidates away from penalised regions.

# Implementation

## Tools and Libraries

| Tool | Role |
|------|------|
| **NumPy** | Array operations, data loading (`.npy`), candidate manipulation |
| **SciPy** (`stats.qmc.LatinHypercube`) | Candidate generation — space-filling LHS samples (10K–100K per function) |
| **scikit-learn** | `GaussianProcessRegressor` (Matérn ARD kernel, MLE optimisation), `SVC`/`SVR` (RBF, SVM constraint & SVR surrogate), `QuantileTransformer`/`MinMaxScaler` |
| **GPyTorch** | GPU-accelerated GPs with ARD kernel and `GammaPrior` lengthscale priors; used for Fn4 and Fn8 where prior-guided noise-dim detection is needed |
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

## Structure

```
.
├── .devcontainer/          # Devpod/VSCode dev container config (Ubuntu + Nix + Neovim)
│   ├── devcontainer.json
│   ├── setup.sh            # Installs Nix packages, Neovim plugins, project dependencies
│   └── nvim/               # Neovim configuration
├── notebooks/              # Weekly per-function notebooks
│   ├── week_{N}_function_{M}.ipynb   # N=1..6, M=1..8
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