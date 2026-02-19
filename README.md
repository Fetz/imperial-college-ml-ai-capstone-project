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
- Week 3:
  - Preprocessing: Replaced StandardScaler with QuantileTransformer for functions with extreme outliers (e.g., function 1 spans ~120 orders of magnitude). Added
  log-space as a second representation
  - Models: Introduced dual-GP ensembles (one on QT output, one on log-space). Standardised on Matern(nu=2.5) with ARD across all functions
  - SVMs: Added SVC (RBF kernel, C=10, soft-margin) to classify regions as promising vs not-promising. Used P(promising) as a multiplier on UCB scores to
  suppress penalty/cliff regions. SVR was tested but excluded from the acquisition ensemble due to lacking native uncertainty estimates
  - Acquisition: Switched entirely from EI to UCB(kappa=5.0). Replaced meshgrids with Latin Hypercube Sampling, scaling candidates with dimensionality (10K–100K)
  - Feature selection: Used ARD to identify and drop irrelevant dimensions (e.g., function 8 reduced from 8D to 6D after identifying x6 and x8 as noise)

### Exploration vs exploitation strategy

My overall strategy prioritises exploration in early weeks and shifts toward exploitation as data accumulates to balance both continuously, I started to introduce UUCB's kappa to control global exploration and the SVM P(promising) constraint to provide exploitation bias.

# Implementation

## Tools and Libraries used

- (WIP) [Devpod](https://devpod.sh/) - to setup dev enviroment
- (WIP) [Orbstack](https://orbstack.dev/) - docker container alternative in MacOS
- (WIP) [Ubuntu + Nix package manager](https://nixos.org/) - Linux Distro used in the container
- (WIP) [Neovim](https://neovim.io/) - Vim based terminal text editor
- [Jupyter Notebooks](https://jupyter.org/) - Interactive coding documentation
- [Pandas](https://pandas.pydata.org/)
- [NumPy](https://numpy.org/)
- [Scikit‑learn](https://scikit-learn.org/)

## How to run

### Experimental
```
devpod up .
```

## Structure

```
.
├── .devcontainer           # [devpod](https://devpod.sh/) configuration
├── notebooks               # weekly notebooks
├   ├── .scripts            # scripts to automate weekly data
├   ├── data      
├       ├── intial_data     # data (excluded from the repo) - download link will later be provided
├       ├── week_*          # data (excluded from the repo) - these folders are generated using the .scripts and update so we can compare data from different weeks
├       ├── update          # delta data that we get from each week
├── submissions             # weekly submissions

```
