# Notebooks

## Context

8 unknown black-box functions, each with a different input dimensionality (2D–8D). The goal is to find the input that maximises each function's output using at most 13 weekly queries.

Each function:
- Returns a single scalar output
- Is treated as a maximisation problem
- Has inputs bounded in [0, 1]^d

## Current data (Week 6)

| Fn | Dims | Points | y range |
|----|------|--------|---------|
| 1  | 2D   | 15     | [−3.61e−3, 5.62e−6] |
| 2  | 2D   | 15     | well-behaved positive |
| 3  | 3D   | 20     | [−0.399, −0.033] all negative |
| 4  | 4D   | 35     | [−37.5, 0.40] outlier at min |
| 5  | 4D   | 25     | [0.113, 1630] all positive, wide range |
| 6  | 5D   | 25     | [−2.57, −0.50] all negative |
| 7  | 6D   | 35     | [0.003, 2.53] all positive |
| 8  | 8D   | 45     | well-behaved positive |

## Per-function strategy (Week 6)

| Fn | Dims | Best y (W5) | Surrogate | Active dims | Output transform | SVM? | kappa |
|----|------|-------------|-----------|-------------|------------------|------|-------|
| 1  | 2D   | 5.62e−6     | GP + SVR ensemble | all 2D | QuantileTransformer | Yes (C=10) | 8.0 (override) |
| 2  | 2D   | 6.11e−1     | GP | all 2D | QuantileTransformer | No | EI |
| 3  | 3D   | −3.26e−2    | 2-GP + 2-SVR ensemble | all 3D | QT + log10 | Yes (C=1) | 3.846 |
| 4  | 4D   | 4.01e−1     | GP | all 4D | QuantileTransformer | No | 2.0 (override) |
| 5  | 4D   | 1.630e+3    | 2-GP + 2-SVR ensemble | all 4D | QT + log10 | Yes (C=10) | 4.31 |
| 6  | 5D   | −5.02e−1    | 2-GP + 2-SVR ensemble | all 5D | QT + log10 | Yes (C=10) | 4.31 |
| 7  | 6D   | 1.614e+0    | 2-GP ensemble | 5D (x3 dropped) | QT + log10 | Yes (C=1) | 3.846 |
| 8  | 8D   | 9.852e+0    | GP | 7D (x8 dropped) | QuantileTransformer | No | 3.846 |

**Notes:**
- *Active dims*: dimensions used for LHS candidate search. Noise dimensions are fixed at their training mean before submitting to the oracle.
- *SVM*: SVC (RBF) classifies promising vs unpromising regions. P(promising) multiplies the UCB score. Fail-safe activates if <5% of candidates are classified as promising.
- *kappa*: budget-aware formula gives `5.0 - ((week-1)/13)*3.0`; Fn1 uses 8.0 override for broader exploration. Fn2 uses Expected Improvement (EI) instead of UCB — kappa does not apply.
- *2-GP ensemble*: one GP on QuantileTransformer output, one on log10-shifted output, averaged after UCB.

## Notebook structure

Each `week_{N}_function_{M}.ipynb` follows the same sections:

1. **Setup** — imports
2. **Plan** — week-specific notes and changes
3. **Load data** — from `./data/week_{N}/function_{M}/`
4. **Preprocessing** — output transform (QT and/or log10)
5. **Probabilistic models** — GP fitting with convergence guard
6. **SVM analysis** — SVC classifier + SVR surrogate (where applicable)
7. **Acquisition function** — SVM-constrained UCB, ensemble argmax over LHS candidates

The final cell prints `SUBMISSION: x1-x2-...-xn` — used by `.scripts/get_submission.py` to auto-populate `submissions/Week_N.md`.

## Utility scripts (`.scripts/`)

| Script | Usage |
|--------|-------|
| `create_weekly_data.py` | Merges weekly delta files into `data/week_{N}/` |
| `create_weekly_notebooks.py` | Scaffolds next week's notebooks from the current week |
| `get_submission.py` | Extracts recommended points from executed notebooks |

```bash
# Extract week 6 submissions (print only)
python3 notebooks/.scripts/get_submission.py 6

# Write directly to submissions/Week_6.md
python3 notebooks/.scripts/get_submission.py 6 --write

# Single function only
python3 notebooks/.scripts/get_submission.py 6 --fn 3

# Print cross-week results summary
python3 notebooks/.scripts/get_submission.py --results

# Write cross-week results summary to RESULTS.md
python3 notebooks/.scripts/get_submission.py --results --write
```
