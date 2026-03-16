# Notebooks

## Context

8 unknown black-box functions, each with a different input dimensionality (2D–8D). The goal is to find the input that maximises each function's output using at most 13 weekly queries.

Each function:
- Returns a single scalar output
- Is treated as a maximisation problem
- Has inputs bounded in [0, 1]^d

## Current data (Week 7)

| Fn | Dims | Points | y range |
|----|------|--------|---------|
| 1  | 2D   | 16     | [−3.61e−3, 5.62e−6] |
| 2  | 2D   | 16     | well-behaved positive |
| 3  | 3D   | 21     | [−0.399, −0.022] all negative |
| 4  | 4D   | 36     | [−37.5, 0.40] outlier at min |
| 5  | 4D   | 26     | [0.113, 1630] all positive, wide range |
| 6  | 5D   | 26     | [−2.57, −0.335] all negative |
| 7  | 6D   | 36     | [0.003, 2.857] all positive |
| 8  | 8D   | 46     | well-behaved positive |

## Per-function strategy (Week 7)

| Fn | Dims | Best y (W6) | Surrogate | Active dims | Output transform | SVM? | kappa |
|----|------|-------------|-----------|-------------|------------------|------|-------|
| 1  | 2D   | 5.62e−6     | GP + SVR ensemble | all 2D | QuantileTransformer | Yes (C=10) | 5.0 (reduced from 8.0; warm-start ±0.05 around best) |
| 2  | 2D   | 6.11e−1     | GP | all 2D | QuantileTransformer | No | EI (clip [0.05, 0.95]; exploit cluster ±0.015 around best) |
| 3  | 3D   | −2.24e−2    | 2-GP ensemble | **2D active (x2, x3; x1 noise, dropped W7)** | QT + log10 | Yes (C=1) | 3.615 |
| 4  | 4D   | 4.01e−1     | GP | all 4D | QuantileTransformer | No | 2.0 (override; warm-start ±0.02 around best) |
| 5  | 4D   | 1.630e+3    | 2-GP ensemble | all 4D | QT + log10 | Yes (C=10) | 3.615 (warm-start ±0.02 around best) |
| 6  | 5D   | −3.35e−1    | 2-GP ensemble | all 5D | QT + log10 | Yes (C=10) | 3.615 |
| 7  | 6D   | 2.857e+0    | 2-GP ensemble | 5D (x3 dropped) | QT + log10 | Yes (C=1) | 3.615 (**gradient-based UCB**: 64 starts × 200 Adam steps) |
| 8  | 8D   | 9.889e+0    | GP | 7D (x8 dropped) | QuantileTransformer | No | 3.615 (**gradient-based UCB**: 64 starts × 200 Adam steps) |

**Notes:**
- *Active dims*: dimensions used for LHS candidate search. Noise dimensions are fixed at their training mean before submitting to the oracle.
- *SVM*: SVC (RBF) classifies promising vs unpromising regions. P(promising) multiplies the UCB score. Fail-safe activates if <5% of candidates are classified as promising.
- *kappa*: budget-aware formula gives `5.0 - ((week-1)/13)*3.0`; Fn2 uses Expected Improvement (EI) instead of UCB — kappa does not apply.
- *2-GP ensemble*: one GP on QuantileTransformer output, one on log10-shifted output, averaged after UCB.
- *Gradient UCB (Fn7, Fn8)*: acquisition maximised via Adam (64 LHS-seeded starts × 200 steps, lr=0.05) with sigmoid reparameterisation x=sigmoid(z) for [0,1] bounds. fast_pred_var disabled during loop for exact gradients; re-enabled for final evaluation.
- *Warm-start cluster*: Fn1/2/4/5 append a dense cluster of candidates (±0.01–0.05) around the best known training point so exploitation options always compete against uncertain LHS regions.

## Per-function design decisions and alternatives considered

### Function 1 (2D)
**Output transform**: QuantileTransformer only. Log10 was removed in Week 5 — the shift constant `abs(y.min()) + 1.0 ≈ 1.004` is ~10¹⁸× larger than 12 of the 14 y values (spanning 1e-124 to 1e-15), so at float64 precision all 12 near-zero values collapse to the same log value (0.001563). `gp_log` was learning nothing that `gp_qt` doesn't already capture. QT gives 16 distinct rank-based gradient signals.

**Surrogate**: GP + SVR ensemble (2 surrogates). A 4-surrogate ensemble (Week 4) was reduced because `gp_log` and `svr_log` both relied on the collapsed log representation and added noise rather than signal. SVR(QT) acts as a regularising cross-check on the GP prediction.

**Acquisition kappa**: Reduced 8.0 → 5.0 in Week 7 after three consecutive exploration queries (W5–W7) covering top-right/centre/top-left with no improvement. The budget-aware formula (4.08 at week 7) was overridden upward to 8.0 in weeks 5–6 because spatial coverage was poor; switched back to exploitation once coverage was sufficient.

**SVM**: C=10, gamma='auto' (≈0.5 for 2D). C=1 was too soft — all 16 balanced points fell inside the wide margin, making every point a support vector and the boundary uninformative. gamma='scale' (≈6) was too local and caused boundary collapse.

---

### Function 2 (2D)
**Acquisition function**: Switched UCB → Expected Improvement (EI) in Week 6. UCB with kappa > 2 was repeatedly recommending x1≈0.9999 (boundary), because the uncertainty surface peaks at unvisited boundaries faster than EI. EI anchors to `y_best` and naturally penalises regions where the predicted mean is well below the current best — preventing the boundary-chasing pattern.

**Candidate clipping**: LHS candidates clipped to [0.05, 0.95] in Week 7. Even after switching to EI, the acquisition oscillated between opposite corners (x2→1 at W5, x2→0 at W7). Clipping removes the boundary singularity from the search space entirely without needing a proximity constraint.

**No SVM**: Function 2 outputs are well-behaved with no penalty zone; an SVM constraint would suppress valid high-uncertainty regions for no benefit.

---

### Function 3 (3D → 2D active)
**Noise dimension**: x1 confirmed as noise via ARD in Week 7. The Week 3–6 approach used absolute threshold (ls > 100) which required raising `length_scale_bounds` to 50–1e3 to let the optimizer find the noisy dimension. The final confirmation: x1 ls hit the 1e3 ceiling with a `ConvergenceWarning`, confirming it carries no useful signal. x1 fixed at training mean; LHS search in 2D active subspace (x2, x3).

**Noise detection caveat**: The x1 noise decision is based on gp_qt alone (ls = 1000, ceiling hit, ConvergenceWarning). gp_log gives x1 ls = 16.88 — elevated but not at the ceiling, so it would not independently flag x1 as noise. The decision is therefore supported by one of the two GPs; the ConvergenceWarning is treated as the deciding evidence.

**SVM**: To be removed in Week 8. With only 21 points in 3D and 20/21 as support vectors, the decision boundary wraps around every training point — the constraint is inactive (P(promising) ≈ uniform). The median threshold gives a 50/50 label split by design, but with so few points the SVM cannot learn a meaningful boundary.

**Output transform**: All 21 y values are negative. Shift by `abs(min) + 1.0` to bring all values into positive territory for the log transform. QT + log10 ensemble handles the compressed range.

---

### Function 4 (4D)
**Output transform**: Switched StandardScaler → QuantileTransformer in Week 5. A single outlier at y = −37.5 (vs. remaining values in [−4, 0.4]) dominates the StandardScaler variance, giving the GP a distorted covariance structure that over-weights the penalty region. QT eliminates this by using ranks.

**No SVM**: Outputs span both positive and negative values without a clear spatial penalty zone; SVM constraint was tested but did not improve acquisition focus.

**Warm-start cluster**: Added in Week 7 around the oracle-confirmed peak `[0.433, 0.425, 0.379, 0.388]` (y = 0.40). GP length scales (~1.5) are too large to guide toward narrow peaks from global LHS alone — the dense cluster ensures exploitation candidates always appear in the acquisition race.

---

### Function 5 (4D)
**Output transform**: QT + log10. y ∈ [0.113, 1630] — all positive, 4 orders of magnitude. Log10 compresses the spike; QT handles rank ordering. Both GPs in the ensemble are meaningful here (unlike Fn1 where log collapsed).

**SVM**: C=10, soft margin with median threshold. Used to suppress the flat low-value plateau (most y < 200) and focus acquisition on the high-value region near y ≈ 1630.

**Warm-start cluster**: Oracle history shows a sharp peak at `[0.120, 0.863, 0.880, 0.958]`. Dense cluster (±0.02, 1K candidates) added in Week 7 to prevent global LHS from ignoring a narrow but confirmed high-value region.

---

### Function 6 (5D)
**Output transform**: All 26 y values negative, range [−2.57, −0.335]. Shift + log10 + QT. The shift ensures log10 receives positive inputs; QT then handles the compressed range.

**SVM**: C=10, median threshold. The negative output domain means the SVM distinguishes "less negative" (promising) from "more negative" (unpromising) — a meaningful boundary in this function's landscape.

**No noise-dim dropping**: ARD length scales across all 5 dims are within the same order of magnitude; no dimension shows the extreme ls divergence seen in Fn7/Fn8. All 5 dims retained.

---

### Function 7 (6D → 5D active)
**Noise dimension**: x3 dropped via ARD. Week 5 sklearn ARD found ls = 438,000 for x3 (vs. ≤ 1.6 for all others) — unambiguous noise signal. GPyTorch with a standard `GammaPrior(3, 6)` (mode ≈ 0.33) suppresses ls below ~2.0, so a weak `GammaPrior(1, 0.1)` (mean = 10, heavy tail) is used for detection to allow divergence. Relative threshold (ls > mean + 2×std) applied; x3 confirmed as noise, fixed at training mean.

**Gradient-based acquisition (Week 7)**: Replaced LHS argmax with multi-start gradient ascent (Wilson et al., 2018). 64 LHS-seeded starts × 200 Adam steps with sigmoid reparameterisation `x = sigmoid(z)` enforcing [0, 1] bounds. `fast_pred_var()` disabled during the loop so exact variance gradients flow through the GP posterior. Alternative considered: increasing LHS from 50K to 500K — rejected because the computational cost scales badly in 5D and still cannot resolve narrow peaks that gradient ascent finds directly.

**SVM**: C=1 (soft margin). At 36 points in 6D, a tighter margin (C=10) risks decision boundary collapse — C=1 prevents the boundary from wrapping tightly around individual training points.

---

### Function 8 (8D → 7D active)
**Noise dimension**: x8 dropped. Week 5 identified both x6 and x8 as noise (6D active subspace). Week 6 re-evaluation with the relative ARD threshold found only x8 as noise; x6 restored to active subspace (7D). The relative threshold is more robust than an absolute cutoff because GPyTorch's default prior compresses all length scales — an absolute threshold of 100 would never fire.

**Gradient-based acquisition (Week 7)**: Same rationale as Fn7. With 46 points in 7D, the acquisition surface has many local maxima; gradient ascent from 64 diverse seeds is more reliable at finding the global maximum than a fixed 100K LHS grid, which at 7D gives sparse coverage (~3.7 points per dimension per unit volume on average).

**No SVM**: Well-behaved positive outputs with no observed penalty zone. SVM constraint would reduce coverage without adding signal.

**No output outliers**: StandardScaler sufficient in early weeks; switched to QT in later weeks for consistency with the ensemble framework.

---

## Notebook structure

Each `week_{N}_function_{M}.ipynb` follows the same sections:

1. **Setup** — imports
2. **Plan** — week-specific notes and changes
3. **Load data** — from `./data/week_{N}/function_{M}/`
4. **Preprocessing** — output transform (QT and/or log10)
5. **Probabilistic models** — GP fitting with convergence guard
6. **SVM analysis** — SVC classifier + SVR surrogate (where applicable)
7. **Acquisition function** — SVM-constrained UCB or EI, ensemble argmax over LHS candidates + warm-start cluster. Fn7/Fn8 use gradient-based UCB (Adam + sigmoid) instead of LHS.

The final cell prints `SUBMISSION: x1-x2-...-xn` — used by `.scripts/get_submission.py` to auto-populate `submissions/Week_N.md`.

## Utility scripts (`.scripts/`)

| Script | Usage |
|--------|-------|
| `create_weekly_data.py` | Merges weekly delta files into `data/week_{N}/` |
| `create_weekly_notebooks.py` | Scaffolds next week's notebooks from the current week |
| `get_submission.py` | Extracts recommended points from executed notebooks |

```bash
# Extract week 7 submissions (print only)
python3 notebooks/.scripts/get_submission.py 7

# Write directly to submissions/Week_7.md
python3 notebooks/.scripts/get_submission.py 7 --write

# Single function only
python3 notebooks/.scripts/get_submission.py 7 --fn 3

# Print cross-week results summary
python3 notebooks/.scripts/get_submission.py --results

# Write cross-week results summary to RESULTS.md
python3 notebooks/.scripts/get_submission.py --results --write
```
