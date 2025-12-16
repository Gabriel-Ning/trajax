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

"""Type definitions for trajectory optimization."""

from __future__ import annotations

from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Callable, Protocol, Tuple

from jax import Array

if TYPE_CHECKING:
    from trajax.core.trajectory import Trajectory


# Type aliases for common shapes
# State: (n,) array
# Control: (m,) array
# StateTrajectory: (T+1, n) array
# ControlTrajectory: (T, m) array

PyTree = Any


class SolverStatus(Enum):
    """Status codes for trajectory optimization solvers."""
    SOLVED = auto()           # Converged to solution
    MAX_ITERATIONS = auto()   # Reached maximum iterations
    INFEASIBLE = auto()       # Problem is infeasible
    UNBOUNDED = auto()        # Problem is unbounded
    STALLED = auto()          # Solver stalled (no progress)
    LINE_SEARCH_FAILED = auto()  # Line search failed
    UNKNOWN = auto()          # Unknown status


# Function type protocols

class DynamicsFn(Protocol):
    """Protocol for dynamics functions.

    Signature: dynamics(x, u, t, params) -> x_next

    Args:
        x: State vector (n,)
        u: Control vector (m,)
        t: Time index (scalar int)
        params: Optional parameters (PyTree)

    Returns:
        x_next: Next state vector (n,)
    """
    def __call__(
        self,
        x: Array,
        u: Array,
        t: int,
        params: PyTree = ()
    ) -> Array:
        ...


class CostFn(Protocol):
    """Protocol for cost functions.

    Signature: cost(x, u, t, params) -> scalar

    Args:
        x: State vector (n,)
        u: Control vector (m,)
        t: Time index (scalar int)
        params: Optional parameters (PyTree)

    Returns:
        cost: Scalar cost value
    """
    def __call__(
        self,
        x: Array,
        u: Array,
        t: int,
        params: PyTree = ()
    ) -> float:
        ...


class TerminalCostFn(Protocol):
    """Protocol for terminal cost functions.

    Signature: terminal_cost(x, params) -> scalar

    Args:
        x: Terminal state vector (n,)
        params: Optional parameters (PyTree)

    Returns:
        cost: Scalar terminal cost value
    """
    def __call__(
        self,
        x: Array,
        params: PyTree = ()
    ) -> float:
        ...


class ConstraintFn(Protocol):
    """Protocol for constraint functions.

    Signature: constraint(x, u, t, params) -> constraint_values

    For inequality constraints: constraint(x, u, t) <= 0
    For equality constraints: constraint(x, u, t) == 0

    Args:
        x: State vector (n,)
        u: Control vector (m,)
        t: Time index (scalar int)
        params: Optional parameters (PyTree)

    Returns:
        c: Constraint values (n_constraints,)
    """
    def __call__(
        self,
        x: Array,
        u: Array,
        t: int,
        params: PyTree = ()
    ) -> Array:
        ...


# Solver function type after compilation
SolverFn = Callable[[Array, Array, PyTree], 'Trajectory']


# LQR parameter types
LQRParams = Tuple[
    Array,  # Q: (T+1, n, n) state cost Hessians
    Array,  # q: (T+1, n) state cost gradients
    Array,  # R: (T, m, m) control cost Hessians
    Array,  # r: (T, m) control cost gradients
    Array,  # M: (T, n, m) cross-term Hessians
    Array,  # A: (T, n, n) dynamics Jacobians wrt state
    Array,  # B: (T, n, m) dynamics Jacobians wrt control
]


# Bounds type
Bounds = Tuple[Array, Array]  # (lower, upper)
