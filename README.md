# Black-box optimisation challenge

Capstone project for the Professional Certificate in Machine Learning and AI from [Imperial College Executive Education](https://web.archive.org/web/20260201194625/https://www.imperial.ac.uk/business-school/executive-education/technology-analytics-data-science/professional-certificate-machine-learning-and-artificial-intelligence-programme/online/).

## Status - 🟡

This capstone is currently a work in progress.

## Overview

This is my work for the Capstone project on Black-Box Optimisation (BBO) challenge, this challenge is based on similar BBO challenges that happens in Kaggle like NeurIPS 2020 which ran from July–October, 2020.

## 🎯 Goal

This challenge involves optimising 8 unknown functions with dimensions from 2 to 8 dimensions with initially a pair 10 to 15 data points and output for each function and every week I will submit one data point for each function and I will receive their correspondent output for 13 weeks.

## Inputs & Outputs

### 1. Initial data:
Will be provided in binary `.npy` format:
- Structure: 8 pairs of (`function_x/initial_inputs.npy`, `function_x/initial_outputs.npy`).
- Dimensions: Inputs are matrices $X \in \mathbb{R}^{m \times d}$, where $10 \leq m \leq 15$ and $2 \leq d \leq 8$.
- Vectors: $y \in \mathbb{R}^m$, containing signed floating-point values.

### 2. Weekly delta:
Updates are delivered via text `.txt` format. Contrary to initial data, weekly updates will contain exactly 1 data point per function (m = 8).

#### A. `input.txt`

A serialized list of NumPy arrays. Each array represents a single coordinate in the domain of one of the eight functions.
- List Index + 1 = Function
- Array Shape: (m, n) where $10 \leq m \leq 15$ and $2 \leq d \leq 8$
- Value Range: [0.000000, 0.999999]
- Value Precision: Maximun 6 decimal places.
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

**Maximise the output of each unknow function** with limited data points and limited to 13 queries.

Constraints:
- Function: Unknown shape and structure
- Function observation: limited to 10 to 15 know data points
- Number of queries: Limited to 13
- Response: Delayed

## Technical approach

### Early strategy (week 1-...)

My early strategy is to focused on exploration over exploitation to increase the model confidence on unexplored areas and then move to exploitation around week 7.

#### Methods used
- Gaussian Process (GP).
- Kernel: RBF & Matérn kernels.
- Feature importance: ARD (Matérn), Random Forest (purely for visualization).
- Acquisition Functions: Expected Improvement (EI) & Upper Confidence Bound (UCB).
- Normalized input & output.

#### Future extensions
TODO

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

### ⚠️ Experimental
```
devpod up .
```

## 🗂️ Structure

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
