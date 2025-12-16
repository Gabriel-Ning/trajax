# Trajax - Trajectory Optimization in JAX

A modular library for trajectory optimization and optimal control in JAX, featuring multiple solvers, QP backends, and a unified MPC interface.

## Overview

This is a refactored fork of [google/trajax](https://github.com/google/trajax) with enhanced modularity, additional solver backends, and improved interfaces for model predictive control.

### Key Features

- **Multiple Optimization Algorithms**: iLQR, Constrained iLQR, SQP, CEM, Random Shooting
- **Multiple QP Backends**: OSQP, Clarabel, CVXPY, ProxQP, and augmented Lagrangian iLQR
- **Unified MPC Interface**: Easy-to-use controller with warm starting and periodic relinearization
- **LQR Solvers**: Time-varying LQR, Riccati equation solvers (DARE/CARE)
- **JIT-Friendly**: "Build once, pass parameters" pattern for efficient compilation
- **Backward Compatible**: Legacy API maintained for existing code

## Installation (uv)

This repo is configured for [uv](https://docs.astral.sh/uv/) with reproducible dependency locking.

### Quick Setup

```bash
# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh  # or: pip install uv

# Configure environment (handles large wheel downloads and CUDA torch)
export UV_CACHE_DIR=.uv-cache
export UV_HTTP_TIMEOUT=120

# Install base dependencies
uv sync

# Also install dev dependencies (pytest, jupyter, etc.)
uv sync --group dev
```

### What's Installed

- **JAX 0.6.2** with CUDA 12 support (with jax-cuda12-pjrt and jax[cuda12])
- **PyTorch 2.9.1+** from PyTorch CUDA 12.8 index
- **Trajectory Optimization**: iLQR, Constrained iLQR, SQP, CEM solvers
- **QP Backends**: OSQP, Clarabel, CVXPY, ProxQP
- **Testing**: pytest, pytest-xdist, frozendict

### Using the Environment

Activate the virtual environment or use `uv run`:

```bash
# Activate virtual environment
source .venv/bin/activate

# Or prefix commands with uv run
uv run python -c "import trajax; print(trajax.__version__)"
```

### Common Workflows

```bash
# Run the full test suite
uv run pytest

# Run tests in parallel
uv run pytest -n auto

# Start Jupyter notebook
uv run jupyter notebook

# Add a new dependency
uv add <package>
uv lock --update

# Update all dependencies
uv lock --upgrade
```

### Troubleshooting

If `casadi` or other large wheels fail to download:
```bash
# Increase timeout and retry
UV_HTTP_TIMEOUT=300 uv sync
```

If you prefer pip, `pip install -e .` also works, but uv provides reproducible, locked dependencies.

## Quick Start

### Basic Trajectory Optimization

```python
import jax.numpy as jnp
from trajax.core import TrajectoryProblem
from trajax.solvers import ILQROptimizer

# Define dynamics and cost
def dynamics(x, u, t, params):
    return x + u * dt

def cost(x, u, t, params):
    return jnp.sum(x**2) + jnp.sum(u**2)

# Create problem
problem = TrajectoryProblem(
    state_dim=4,
    control_dim=2,
    horizon=50,
    dynamics=dynamics,
    cost=cost,
)

# Solve
optimizer = ILQROptimizer(maxiter=100)
result = optimizer.solve(problem, x0, U0)

print(f"Objective: {result.obj}")
print(f"Status: {result.status}")
```

### Model Predictive Control

```python
from trajax.mpc import MPCProblem, MPCConfig, MPCController

# Create MPC problem with reference tracking
problem = MPCProblem.tracking_problem(
    dynamics=dynamics,
    x_ref=reference_trajectory,
    u_ref=nominal_controls,
    Q=jnp.diag([10.0, 10.0, 1.0, 1.0]),
    R=jnp.diag([0.1, 0.1]),
)

# Configure controller
config = MPCConfig(
    horizon=20,
    dt=0.01,
    warm_start=True,
    relinearize_every=5,
)

# Build controller (JIT compile once)
controller = MPCController(problem, config)
controller.build()

# Control loop
for t in range(num_steps):
    u, info = controller.step(x_current)
    x_current = simulate(x_current, u)
    print(f"Step {t}: objective={info['objective']:.3f}, "
          f"solve_time={info['solve_time_ms']:.2f}ms")
```

### Using Different Solvers

```python
from trajax.solvers import get_solver

# Create solver by name
optimizer = get_solver('constrained_ilqr', maxiter=50)
result = optimizer.solve(problem, x0, U0)

# Or use specific optimizer class
from trajax.solvers import ConstrainedILQROptimizer

optimizer = ConstrainedILQROptimizer(
    maxiter=50,
    maxiter_al=10,
    constraints_threshold=1e-3,
)
```

## Package Structure

```
trajax/
├── core/          # Core abstractions (TrajectoryProblem, Trajectory, types)
├── solvers/       # Trajectory optimizers (iLQR, CEM, SQP, etc.)
├── mpc/           # MPC controller interface
├── lqr/           # LQR solvers (TVLQR, Riccati equations)
├── qp/            # QP solver backends
├── utils/         # Utilities (linearize, rollout, adjoint, PSD projection)
└── legacy/        # Backward-compatible API
```

## Modules

### trajax.core
Core abstractions for trajectory optimization:
- `TrajectoryProblem`: Problem specification with dynamics, costs, and constraints
- `Trajectory`: Solution container with states, controls, and solver info
- Type definitions and protocols

### trajax.solvers
Trajectory optimization algorithms:
- `ILQROptimizer`: Iterative Linear Quadratic Regulator
- `ConstrainedILQROptimizer`: Constrained iLQR with augmented Lagrangian
- `CEMOptimizer`: Cross-Entropy Method
- `RandomShootingOptimizer`: Random shooting baseline

### trajax.mpc
Model Predictive Control interface:
- `MPCProblem`: MPC-specific problem formulation with reference tracking
- `MPCConfig`: Configuration for horizon, timestep, relinearization
- `MPCController`: Receding horizon controller with warm starting

### trajax.lqr
Linear Quadratic Regulator solvers:
- `tvlqr`: Time-varying LQR solver
- `ctvlqr`: Constrained TVLQR with ADMM
- `dare_solve`: Discrete-time algebraic Riccati equation
- `care_scipy`: Continuous-time algebraic Riccati equation

### trajax.qp
QP solver backends (optional dependencies):
- `OSQPBackend`: OSQP solver
- `ClarabelBackend`: Clarabel solver
- `CVXPYBackend`: CVXPY wrapper
- `ProxQPBackend`: ProxQP solver
- `ALiLQRBackend`: Augmented Lagrangian iLQR as QP solver

### trajax.utils
Utility functions:
- `linearize`: Linearize dynamics via autodiff
- `quadratize`: Quadratize cost functions
- `rollout`: Forward simulation
- `adjoint`: Adjoint equation solver
- `project_psd_cone`: PSD projection for numerical stability

## Backward Compatibility

The legacy API is maintained for existing code:

```python
# Old API still works
from trajax.legacy import ilqr

X, U, obj, grad, adjoints, lqr_state, iters = ilqr(
    cost, dynamics, x0, U, maxiter=100
)

# Or use legacy imports
import trajax
result = trajax.optimizers.ilqr(cost, dynamics, x0, U)
```

## License

This project maintains the Apache License 2.0 from the original trajax library.
See LICENSE and NOTICE files for details.

## Citation

If you use this library, please cite the original trajax:

```bibtex
@software{trajax2021,
  author = {Google LLC},
  title = {Trajax: Trajectory Optimization in JAX},
  url = {https://github.com/google/trajax},
  year = {2021},
}
```

## Acknowledgments

This is a derivative work based on [google/trajax](https://github.com/google/trajax).
All core algorithms and many implementations originate from the original project.
This fork adds modularity, additional backends, and enhanced interfaces.
