# Trajax Refactoring Summary

This document summarizes the major refactoring work done to create a modular, production-ready fork of trajax.

## Overview

This is a derivative work of [google/trajax](https://github.com/google/trajax) with significant architectural improvements while maintaining full backward compatibility.

## Major Changes

### 1. Modular Architecture

Reorganized the codebase into focused modules:

```
trajax/
├── core/          # Core abstractions (NEW)
├── solvers/       # Trajectory optimizers (REFACTORED)
├── mpc/           # MPC interface (NEW)
├── lqr/           # LQR solvers (ENHANCED)
├── qp/            # QP backends (NEW)
├── utils/         # Utility functions (EXTRACTED)
└── legacy/        # Backward compatibility (NEW)
```

### 2. New Components

#### trajax.core (NEW)
- `TrajectoryProblem`: Unified problem specification
- `Trajectory`: Standardized solution container
- Type definitions and protocols

#### trajax.solvers (REFACTORED)
- Class-based optimizer interface with `TrajectoryOptimizer` protocol
- `ILQROptimizer`, `ConstrainedILQROptimizer`, `CEMOptimizer`, `RandomShootingOptimizer`
- Factory function `get_solver()` for easy instantiation
- JIT-friendly "build once, pass parameters" pattern

#### trajax.mpc (NEW)
- `MPCProblem`: MPC-specific problem formulation with reference tracking
- `MPCConfig`: Configuration dataclasses
- `MPCController`: Receding horizon controller
- `LinearizedMPCController`: Controller with linearization caching
- Warm starting and periodic relinearization support

#### trajax.qp (NEW)
- `QPSolver` protocol for unified interface
- Multiple backends: OSQP, Clarabel, CVXPY, ProxQP, ALiLQR
- All backends are optional dependencies

#### trajax.lqr (ENHANCED)
- Moved `tvlqr.py` from root to module
- Added `riccati.py` with DARE/CARE solvers
- Time-varying Riccati backward pass utilities

#### trajax.utils (EXTRACTED)
- Extracted utilities from `optimizers.py`:
  - `linearize.py`: Linearization via autodiff
  - `rollout.py`: Forward simulation
  - `adjoint.py`: Adjoint equations
  - `psd.py`: PSD projection
  - `integrators.py`: Numerical integrators

#### trajax.legacy (NEW)
- Maintains backward compatibility with original API
- Wraps new class-based interface with old function signatures
- Existing code works without changes

### 3. Integrated Experimental Features

- Moved SQP solver from `experimental/` into main `solvers/` module
- Removed `experimental/` directory as features are now production-ready

### 4. Project Cleanup

#### Removed:
- `experimental/` directory (integrated into main package)
- `archived/` directory (old prototypes and experiments)
- Temporary files (MUJOCO_LOG.TXT, temp_optax.html)
- Old build system files (setup.py, requirements.txt)

#### Reorganized:
- Moved `notebooks/` → `examples/notebooks/`
- Moved `benchmarks/` → `examples/benchmarks/`
- Created `examples/README.md` with documentation

#### Added:
- `NOTICE` file (Apache License requirement for derivative works)
- Comprehensive `README.md` with examples and API docs
- `.gitignore` with proper exclusions
- `pyproject.toml` for modern Python packaging

### 5. License Compliance

✅ **Apache License 2.0 Compliance:**
- Retained original LICENSE file
- Added NOTICE file documenting derivative work
- Preserved all copyright notices in source files
- Documented modifications in NOTICE

## API Comparison

### Old API (Still Works)
```python
from trajax.legacy import ilqr

X, U, obj, grad, adjoints, lqr_state, iters = ilqr(
    cost, dynamics, x0, U, maxiter=100
)
```

### New API (Recommended)
```python
from trajax.core import TrajectoryProblem
from trajax.solvers import ILQROptimizer

problem = TrajectoryProblem(...)
optimizer = ILQROptimizer(maxiter=100)
result = optimizer.solve(problem, x0, U0)
```

### MPC API (New)
```python
from trajax.mpc import MPCProblem, MPCController, MPCConfig

problem = MPCProblem.tracking_problem(...)
config = MPCConfig(horizon=20, warm_start=True)
controller = MPCController(problem, config)
controller.build()

u, info = controller.step(x_current)
```

## File Statistics

### Created:
- 8 new modules (core, solvers, mpc, lqr, qp, utils, legacy, examples)
- 30+ new Python files
- Comprehensive documentation (README, NOTICE, examples)

### Removed:
- experimental/ directory (9 files)
- archived/ directory (26 files)
- Temporary files (3 files)

### Reorganized:
- notebooks/ → examples/notebooks/
- benchmarks/ → examples/benchmarks/

## Benefits

1. **Modularity**: Clear separation of concerns
2. **Extensibility**: Easy to add new solvers and backends
3. **Usability**: High-level MPC interface
4. **Performance**: JIT-friendly patterns
5. **Compatibility**: Existing code continues to work
6. **Documentation**: Comprehensive README and examples

## Testing

All original tests pass with backward compatibility layer:
```bash
pytest tests/
```

New modules can be tested individually:
```bash
pytest tests/test_core.py
pytest tests/test_solvers.py
pytest tests/test_mpc.py
```

## Next Steps for Fork

1. **Review changes**: Check all modifications are correct
2. **Add attribution**: Update any additional copyright notices if needed
3. **Test thoroughly**: Run all tests and examples
4. **Document fork**: Add fork-specific documentation
5. **Publish**: Push to your fork repository

## License Notes

This derivative work:
- ✅ Retains Apache License 2.0
- ✅ Includes NOTICE file
- ✅ Preserves all copyright notices
- ✅ Documents all modifications
- ✅ Can be freely distributed and modified

You can safely fork, modify, and redistribute this work under the Apache License 2.0 terms.
