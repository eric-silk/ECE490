# ECE490
Repo to collaborate within for UIUC's ECE490, Spring 2023

## Installation
### Installation from PyPI
This should be as simple as:
```bash
python -m pip install uiuc-490-sp23
```
### Installation from Source
This project uses [Poetry](https://python-poetry.org/) Install it via
[their instructions](https://python-poetry.org/docs/#installing-with-the-official-installer)
and run:
```bash
poetry install
```
from this directory (preferably from within a `venv`).

## Executing
Once installed, individual assignments can be ran by:
```bash
python -m uiuc_490_sp23.assignment<n>
```
where `<n>` is the assignment number. These will have a CLI implemented using argparse; if you're not sure,
just run the above with `--help` to get a full list of commands.


## Assignments
### Assignment 1: Gradient Descent
This solves a simple, randomly generated quadratic minimization problem using Gradient Descent. It does so using
both fixed step size, as well as an adaptive backtracking line search (Armijo's rule).

### Assignment 2: Newton's Method and Projected Gradient Descent
This assignment does two things:
- Implements Newton's method to solve a simple 2D cubic polynomial and plots a fractal displaying the resulting solution based on the starting point
- Implements Projected gradient descent and uses it to solve a simple box constrained optimization problem

### Assignment 3: Augmented Lagrangian
This assignment implements the augmented Lagrangian to solve an equality constrained quadratic problem.