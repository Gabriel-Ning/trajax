# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Base classes and protocols for trajectory optimizers.

This module defines the standard interface for trajectory optimization
algorithms. All optimizers implement the TrajectoryOptimizer protocol.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Protocol, runtime_checkable

from jax import Array

from trajax.core.types import PyTree
from trajax.core.problem import TrajectoryProblem
from trajax.core.trajectory import Trajectory


@runtime_checkable
class TrajectoryOptimizer(Protocol):
    """Protocol for trajectory optimization algorithms.

    All trajectory optimizers must implement this interface.
    The key methods are:
    - solve(): Standard solve interface
    - build_solver(): Creates a JIT-compiled solver for repeated calls

    Attributes:
        name: Human-readable name of the optimizer.
        supports_constraints: Whether optimizer handles constraints natively.
        is_jittable: Whether solve() can be JIT-compiled.
    """

    name: str
    supports_constraints: bool
    is_jittable: bool

    def solve(
        self,
        problem: TrajectoryProblem,
        x0: Array,
        U0: Array,
        params: PyTree = (),
        options: Optional[Dict[str, Any]] = None,
    ) -> Trajectory:
        """Solve trajectory optimization problem.

        Args:
            problem: TrajectoryProblem specification.
            x0: Initial state of shape (n,).
            U0: Initial control guess of shape (T, m).
            params: Parameters passed to cost/dynamics functions.
            options: Solver-specific options (overrides defaults).

        Returns:
            Trajectory containing optimal X, U, and solver info.
        """
        ...

    def build_solver(
        self,
        problem: TrajectoryProblem,
        options: Optional[Dict[str, Any]] = None,
    ) -> Callable[[Array, Array, PyTree], Trajectory]:
        """Build a JIT-compiled solver function.

        This method implements the "build once, pass parameters" pattern.
        The returned function captures the problem structure at compile time,
        allowing repeated solves without recompilation.

        Args:
            problem: TrajectoryProblem specification.
            options: Solver options to bake into the compiled function.

        Returns:
            A function (x0, U0, params) -> Trajectory that can be called
            repeatedly without recompilation.

        Example:
            >>> optimizer = ILQROptimizer(maxiter=50)
            >>> solve_fn = optimizer.build_solver(problem)
            >>> # First call compiles
            >>> result1 = solve_fn(x0, U0, params1)
            >>> # Subsequent calls reuse compilation
            >>> result2 = solve_fn(x0, U0, params2)
        """
        ...


class TrajectoryOptimizerBase(ABC):
    """Abstract base class for trajectory optimizers.

    Provides common functionality and enforces the interface.
    Subclasses should implement _solve_impl() and optionally override
    build_solver() for custom JIT compilation.
    """

    name: str = "base"
    supports_constraints: bool = False
    is_jittable: bool = True

    def __init__(self, **options):
        """Initialize optimizer with default options.

        Args:
            **options: Solver-specific default options.
        """
        self.default_options = options

    def solve(
        self,
        problem: TrajectoryProblem,
        x0: Array,
        U0: Array,
        params: PyTree = (),
        options: Optional[Dict[str, Any]] = None,
    ) -> Trajectory:
        """Solve trajectory optimization problem.

        Merges options with defaults and calls _solve_impl().
        """
        merged_options = {**self.default_options}
        if options:
            merged_options.update(options)

        return self._solve_impl(problem, x0, U0, params, merged_options)

    @abstractmethod
    def _solve_impl(
        self,
        problem: TrajectoryProblem,
        x0: Array,
        U0: Array,
        params: PyTree,
        options: Dict[str, Any],
    ) -> Trajectory:
        """Internal solve implementation.

        Subclasses must implement this method.
        """
        ...

    def build_solver(
        self,
        problem: TrajectoryProblem,
        options: Optional[Dict[str, Any]] = None,
    ) -> Callable[[Array, Array, PyTree], Trajectory]:
        """Build JIT-compiled solver.

        Default implementation wraps solve() with JAX JIT.
        Subclasses may override for more efficient compilation.
        """
        import jax

        merged_options = {**self.default_options}
        if options:
            merged_options.update(options)

        @jax.jit
        def solve_fn(x0: Array, U0: Array, params: PyTree) -> Trajectory:
            return self._solve_impl(problem, x0, U0, params, merged_options)

        return solve_fn


def get_solver(name: str, **kwargs) -> TrajectoryOptimizerBase:
    """Factory function to create solver by name.

    Args:
        name: Solver name ('ilqr', 'constrained_ilqr', 'sqp', 'cem', etc.)
        **kwargs: Solver-specific options.

    Returns:
        TrajectoryOptimizer instance.

    Raises:
        ValueError: If solver name is not recognized.
    """
    from trajax.solvers.ilqr import ILQROptimizer
    from trajax.solvers.constrained_ilqr import ConstrainedILQROptimizer
    from trajax.solvers.cem import CEMOptimizer
    from trajax.solvers.sampling import RandomShootingOptimizer

    _SOLVERS = {
        'ilqr': ILQROptimizer,
        'constrained_ilqr': ConstrainedILQROptimizer,
        'cem': CEMOptimizer,
        'random_shooting': RandomShootingOptimizer,
    }

    # Try to import SQP (may have additional dependencies)
    try:
        from trajax.solvers.sqp import SQPOptimizer
        _SOLVERS['sqp'] = SQPOptimizer
    except ImportError:
        pass

    name_lower = name.lower()
    if name_lower not in _SOLVERS:
        available = list(_SOLVERS.keys())
        raise ValueError(
            f"Unknown solver: {name}. Available: {available}"
        )

    return _SOLVERS[name_lower](**kwargs)


# Alias for backward compatibility
get_optimizer = get_solver
