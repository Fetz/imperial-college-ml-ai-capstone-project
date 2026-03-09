# Notebooks

## Context

We have 8 unknown functions with the following shape:
1. Function 1: ((10, 2)) - 10DP & 2D
2. Function 2: ((10, 2)) - 10DP & 2D
3. Function 3: ((15, 3)) - 15DP & 3D
4. Function 4: ((30, 4)) - 30DP & 4D
5. Function 5: ((20, 4)) - 20DP & 4D
6. Function 6: ((20, 5)) - 20DP & 5D
7. Function 7: ((30, 6)) - 30DP & 6D
8. Function 8: ((40, 8)) - 40DP & 8D

## Goal

Use Bayesian optimisation to optimise eight synthetic black-box functions.

Each function:
- Has a different input dimensionality (2D to 8D)
- Returns a single output value
- Is framed as a maximisation problem (even if the real-world analogy is minimisation, as it's transformed so that higher is better)

Objective is to find the input combination that maximises the output, using limited
queries and the initial data provided in `.npy` files.
