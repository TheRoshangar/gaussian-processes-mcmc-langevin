# Gaussian Processes, MCMC & Langevin Dynamics

A from-scratch (NumPy/SciPy only, no GP libraries) implementation of Gaussian Process priors,
GP regression, Bayesian hyperparameter inference via Metropolis–Hastings, and gradient-based
sampling via Langevin dynamics — built as the final project for a Stochastic Processes course.

The project's throughline is **Brownian motion showing up in two different roles**: first as a
GP covariance kernel (Task 1), and later as the literal noise source driving the Langevin SDE
(Task 4).

## Contents

| Task | Topic | Key Idea |
|------|-------|----------|
| [1](#task-1--gp-priors-rbf--brownian-motion-kernels) | GP Priors | Sampling functions from RBF and Brownian-motion covariance kernels |
| [2](#task-2--gp-regression) | GP Regression | Closed-form Bayesian posterior over functions given noisy observations |
| [3](#task-3--mcmc-for-the-length-scale-hyperparameter) | MCMC | Metropolis–Hastings to infer a kernel hyperparameter instead of fixing it |
| [4](#task-4--langevin-dynamics) | Langevin Dynamics | Sampling a target density by simulating an SDE, no accept/reject step |

## Task 1 — GP Priors: RBF & Brownian Motion Kernels

Implements two covariance kernels and draws sample paths from each:

- **RBF (squared-exponential):** `k(x, x') = σ² exp(-(x-x')² / 2ℓ²)`, vectorized over an entire
  grid at once using the identity `‖xᵢ-xⱼ‖² = ‖xᵢ‖² + ‖xⱼ‖² - 2xᵢxⱼ` instead of a pairwise Python loop.
- **Brownian motion (Wiener process):** `k(s, t) = min(s, t)`, derived directly from independent
  increments and `W(0) = 0`.

RBF samples are smooth (infinitely differentiable kernel); Brownian samples are jagged
(continuous everywhere, differentiable nowhere) — both properties are visible directly in the
plotted sample paths.

## Task 2 — GP Regression

Given 9 noisy observations of `y = sin(x) + ε`, computes the closed-form GP posterior:

```
μ* = K*ₙ(Kₙₙ + σₙ²I)⁻¹y      Σ* = K** - K*ₙ(Kₙₙ + σₙ²I)⁻¹Kₙ*
```

Rather than forming `Kₙₙ⁻¹` explicitly, this is computed via a **Cholesky factorization**
(`scipy.linalg.cho_factor` / `cho_solve` + `solve_triangular`), following Rasmussen & Williams'
GPML Algorithm 2.1. Forming an explicit inverse is both an unnecessary `O(n³)` operation and
numerically riskier (its condition number is the square of the original matrix's), so the
factor is computed once and reused for both the posterior mean and covariance.

The resulting 95% confidence band is narrow at observed points and widens between them, as
expected of a well-calibrated posterior.

## Task 3 — MCMC for the Length-Scale Hyperparameter

Instead of fixing the RBF length-scale `ℓ`, this task infers it from data with a **random-walk
Metropolis–Hastings sampler**:

- `Gamma(2, 1)` prior on `ℓ`, GP marginal likelihood as the likelihood term.
- Symmetric proposal `ℓ' = ℓ + N(0, 0.15²)` → no Hastings correction term needed.
- Acceptance computed entirely in **log-space** (`log(U) < log α`) to avoid underflow from
  near-zero likelihood ratios.
- 15,000 iterations, 20% burn-in.
- Numerically guarded: `slogdet` instead of `log(det(·))` (avoids under/overflow), a
  `sign ≤ 0` check for non-positive-definite proposals, and a `try/except LinAlgError` around
  the linear algebra so a single bad proposal 10,000 iterations in doesn't crash the whole chain.

## Task 4 — Langevin Dynamics

Samples a symmetric two-mode Gaussian mixture by simulating the **overdamped Langevin SDE**:

```
dXₜ = ½∇log π(Xₜ) dt + dWₜ
```

discretized with **Euler–Maruyama**:

```
X_{t+ε} = Xₜ + (ε/2)∇log π(Xₜ) + √ε·Z,    Z ~ N(0,1)
```

The score `∇log π(x)` is approximated with a central finite difference (`O(ε²)` error, robust to
changing the target density without re-deriving a gradient by hand). The empirical distribution
of 20,000 steps (after 5,000-step burn-in) recovers both modes of the target density with no
accept/reject step at all — a direct contrast with Task 3's MCMC approach to sampling.

## Why NumPy / SciPy, no loops

Every object in this project — a GP sample, a kernel matrix, an MCMC covariance rebuild — is a
vector or matrix operation, not a scalar loop. All kernel computations are vectorized via array
broadcasting rather than pairwise Python loops (see `rbf_kernel`, Task 1), and every numerically
sensitive operation (matrix inverse, determinant, log, sqrt) is either avoided by solving a
linear system instead of inverting, guarded with an early-exit check, wrapped in
`try/except`, or stabilized with a small jitter term. None of this is required in exact
arithmetic — it exists because floating-point computation isn't exact, and the project treats
that as a first-class design constraint rather than an afterthought.

## Repository Structure

```
.
├── SPFinalProject.ipynb   # full implementation, all 4 tasks
├── report_I_results.pdf   # what we built and what we observed
├── report_II_code.pdf     # function-by-function walkthrough of implementation decisions
└── README.md
```

## Tech

- Python, NumPy, SciPy (`scipy.linalg` for Cholesky-based solves), Matplotlib
- No GP/probabilistic-programming libraries (no GPy, GPyTorch, PyMC, etc.) — every kernel,
  posterior, and sampler is implemented directly.

## Reports

Two companion reports are included alongside the notebook:
- **Report I** walks through each task's goal, math, and results (with figures).
- **Report II** goes function-by-function through the code, explaining specific implementation
  choices — e.g. why noise variance only belongs in `K_xx` and not `K_xs`/`K_ss`, why the
  Metropolis acceptance ratio is computed in log-space, and why `√ε` (not `ε`) scales the
  Langevin noise term.

## Team

This was a team project for the Stochastic Processes course (Dr. Peyvandi), built jointly by:

- **Mobin Yousefi**
- **Mehrshad Haghighat**

---
*Final project for Stochastic Processes (Dr. Peyvandi).*
