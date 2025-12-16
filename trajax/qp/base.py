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

"""Base classes and protocols for QP solver backends.

This module defines the standard interface for QP solvers used in
trajectory optimization algorithms like SQP. Different QP solver
backends (OSQP, Clarabel, CVXPY, etc.) implement this interface.

The QP formulation follows the TrajQP structure:
    min sum_k 0.5 * dz_k' Z_k dz_k + q_k' dx_k + r_k' du_k
    s.t. dx_{k+1} = A_k dx_k + B_k du_k
         Cu_k + Ju_k du_k >= 0
         Cx_k + Jx_k dx_k >= 0
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, Optional, Protocol, Tuple, runtime_checkable

import jax.numpy as jnp
from jax import Array


class QPStatus(Enum):
    """Status codes for QP solver results."""
    SOLVED = auto()           # Optimal solution found
    SOLVED_INACCURATE = auto()  # Solution found but may be inaccurate
    MAX_ITERATIONS = auto()   # Reached iteration limit
    INFEASIBLE = auto()       # Problem is primal infeasible
    DUAL_INFEASIBLE = auto()  # Problem is dual infeasible (unbounded)
    NUMERICAL_ERROR = auto()  # Numerical issues encountered
    UNKNOWN = auto()          # Unknown status


@dataclass
class QPFormulation:
    """QP sub-problem formulation for trajectory optimization.

    Represents the quadratic programming sub-problem:

        min sum_{k=0}^{T-1} 0.5 * dz_k' Z_k dz_k + q_k' dx_k + r_k' du_k
            + 0.5 * dx_T' Q_T dx_T + q_T' dx_T

        s.t. dx_{k+1} = A_k dx_k + B_k du_k,  k = 0, ..., T-1
             Cu_k + Ju_k du_k >= 0,           k = 0, ..., T-1
             Cx_k + Jx_k dx_k >= 0,           k = 1, ..., T

    where dz_k = (dx_k, du_k) are perturbations from a nominal trajectory.

    Attributes:
        state_dim: State dimension (n).
        control_dim: Control dimension (m).
        horizon: Time horizon (T).
        n_control_constraints: Number of control constraints per timestep (n_cu).
        n_state_constraints: Number of state constraints per timestep (n_cx).

        Z: Hessian matrices, shape (T, n+m, n+m).
            Z[k] = [[Q_k, M_k], [M_k', R_k]]
        Q_T: Terminal state Hessian, shape (n, n).
        q: State cost gradients, shape (T+1, n).
        r: Control cost gradients, shape (T, m).

        A: Dynamics Jacobians wrt state, shape (T, n, n).
        B: Dynamics Jacobians wrt control, shape (T, n, m).

        Cu: Control constraint values, shape (T, n_cu).
        Ju: Control constraint Jacobians, shape (T, n_cu, m).
        Cx: State constraint values, shape (T+1, n_cx).
        Jx: State constraint Jacobians, shape (T+1, n_cx, n).
    """

    # Dimensions
    state_dim: int
    control_dim: int
    horizon: int
    n_control_constraints: int = 0
    n_state_constraints: int = 0

    # Cost Hessians
    Z: Optional[Array] = None      # (T, n+m, n+m)
    Q_T: Optional[Array] = None    # (n, n) terminal
    Q: Optional[Array] = None      # (T+1, n, n) alternative form

    # Cost gradients
    q: Optional[Array] = None      # (T+1, n)
    r: Optional[Array] = None      # (T, m)

    # Dynamics
    A: Optional[Array] = None      # (T, n, n)
    B: Optional[Array] = None      # (T, n, m)

    # Constraints (inequality: c + J @ d >= 0)
    Cu: Optional[Array] = None     # (T, n_cu)
    Ju: Optional[Array] = None     # (T, n_cu, m)
    Cx: Optional[Array] = None     # (T+1, n_cx)
    Jx: Optional[Array] = None     # (T+1, n_cx, n)

    def __post_init__(self):
        """Validate formulation."""
        n, m, T = self.state_dim, self.control_dim, self.horizon

        # Initialize default zero arrays if not provided
        if self.Z is None:
            self.Z = jnp.zeros((T, n + m, n + m))
        if self.Q_T is None:
            self.Q_T = jnp.zeros((n, n))
        if self.q is None:
            self.q = jnp.zeros((T + 1, n))
        if self.r is None:
            self.r = jnp.zeros((T, m))
        if self.A is None:
            self.A = jnp.zeros((T, n, n))
        if self.B is None:
            self.B = jnp.zeros((T, n, m))

        n_cu = self.n_control_constraints
        n_cx = self.n_state_constraints

        if n_cu > 0:
            if self.Cu is None:
                self.Cu = jnp.zeros((T, n_cu))
            if self.Ju is None:
                self.Ju = jnp.zeros((T, n_cu, m))
        if n_cx > 0:
            if self.Cx is None:
                self.Cx = jnp.zeros((T + 1, n_cx))
            if self.Jx is None:
                self.Jx = jnp.zeros((T + 1, n_cx, n))

    @property
    def has_constraints(self) -> bool:
        """Return True if problem has constraints."""
        return self.n_control_constraints > 0 or self.n_state_constraints > 0

    def get_block_hessian(self, k: int) -> Tuple[Array, Array, Array]:
        """Extract Q, R, M blocks from Z[k].

        Args:
            k: Timestep index.

        Returns:
            Q_k: State Hessian (n, n).
            R_k: Control Hessian (m, m).
            M_k: Cross-term Hessian (n, m).
        """
        n, m = self.state_dim, self.control_dim
        Z_k = self.Z[k]
        Q_k = Z_k[:n, :n]
        R_k = Z_k[n:, n:]
        M_k = Z_k[:n, n:]
        return Q_k, R_k, M_k

    def update(self, **kwargs) -> 'QPFormulation':
        """Return a new QPFormulation with updated fields.

        Args:
            **kwargs: Fields to update.

        Returns:
            New QPFormulation with updated values.
        """
        from dataclasses import replace
        return replace(self, **kwargs)


@dataclass
class QPSolution:
    """Solution from a QP solver.

    Attributes:
        status: Solver status.
        dU: Control perturbations, shape (T, m).
        dX: State perturbations, shape (T+1, n).
        Yu: Control constraint duals, shape (T, n_cu).
        Yx: State constraint duals, shape (T+1, n_cx).
        obj: Objective value at solution.
        iterations: Number of solver iterations.
        info: Solver-specific information (for warm-starting, etc.).
    """

    status: QPStatus
    dU: Optional[Array] = None
    dX: Optional[Array] = None
    Yu: Optional[Array] = None
    Yx: Optional[Array] = None
    obj: float = float('inf')
    iterations: int = 0
    info: Dict[str, Any] = field(default_factory=dict)

    @property
    def solved(self) -> bool:
        """Return True if QP was solved successfully."""
        return self.status in (QPStatus.SOLVED, QPStatus.SOLVED_INACCURATE)


@runtime_checkable
class QPSolver(Protocol):
    """Protocol for QP solver backends.

    All QP solver backends must implement this interface to be used
    with the SQP trajectory optimizer.

    Attributes:
        name: Human-readable name of the solver.
        supports_warm_start: Whether solver supports warm-starting.
        is_jittable: Whether solve() can be JIT-compiled.
    """

    name: str
    supports_warm_start: bool
    is_jittable: bool

    def setup(self, formulation: QPFormulation) -> None:
        """One-time setup for the parametric QP problem.

        This method is called once to create the solver structure.
        After setup, solve() can be called repeatedly with different
        parameter values without reconstruction.

        Args:
            formulation: QPFormulation defining the problem structure.
        """
        ...

    def solve(
        self,
        formulation: QPFormulation,
        warm_start: Optional[QPSolution] = None,
    ) -> QPSolution:
        """Solve the QP with updated parameters.

        Args:
            formulation: QPFormulation with updated parameter values.
            warm_start: Previous solution for warm-starting (optional).

        Returns:
            QPSolution containing primal/dual solutions and status.
        """
        ...


class QPSolverBase(ABC):
    """Abstract base class for QP solver backends.

    Provides common functionality and enforces the interface.
    """

    name: str = "base"
    supports_warm_start: bool = False
    is_jittable: bool = False

    def __init__(self, **options):
        """Initialize solver with options.

        Args:
            **options: Solver-specific options.
        """
        self.options = options
        self._is_setup = False
        self._formulation: Optional[QPFormulation] = None

    @abstractmethod
    def setup(self, formulation: QPFormulation) -> None:
        """Set up the solver for the given problem structure."""
        self._formulation = formulation
        self._is_setup = True

    @abstractmethod
    def solve(
        self,
        formulation: QPFormulation,
        warm_start: Optional[QPSolution] = None,
    ) -> QPSolution:
        """Solve the QP."""
        ...

    def reset(self) -> None:
        """Reset solver state."""
        self._is_setup = False
        self._formulation = None


def create_qp_formulation_from_trajectory(
    X: Array,
    U: Array,
    Q: Array,
    q: Array,
    R: Array,
    r: Array,
    M: Array,
    A: Array,
    B: Array,
    Cu: Optional[Array] = None,
    Ju: Optional[Array] = None,
    Cx: Optional[Array] = None,
    Jx: Optional[Array] = None,
) -> QPFormulation:
    """Create QPFormulation from trajectory optimization data.

    Convenience function to construct a QPFormulation from the typical
    outputs of linearization and quadratization.

    Args:
        X: State trajectory (T+1, n).
        U: Control trajectory (T, m).
        Q: State cost Hessians (T+1, n, n).
        q: State cost gradients (T+1, n).
        R: Control cost Hessians (T, m, m) or (T+1, m, m).
        r: Control cost gradients (T, m) or (T+1, m).
        M: Cross-term Hessians (T, n, m) or (T+1, n, m).
        A: Dynamics Jacobians wrt state (T, n, n).
        B: Dynamics Jacobians wrt control (T, n, m).
        Cu: Control constraint values (T, n_cu), optional.
        Ju: Control constraint Jacobians (T, n_cu, m), optional.
        Cx: State constraint values (T+1, n_cx), optional.
        Jx: State constraint Jacobians (T+1, n_cx, n), optional.

    Returns:
        QPFormulation ready for solving.
    """
    T = U.shape[0]
    n = X.shape[1]
    m = U.shape[1]

    # Construct Z matrices from Q, R, M
    # Ensure we only use T timesteps for Z
    Q_stage = Q[:T]
    R_stage = R[:T] if R.shape[0] > T else R
    M_stage = M[:T] if M.shape[0] > T else M

    # Build block Hessians: Z[k] = [[Q[k], M[k]], [M[k]', R[k]]]
    Z = jnp.zeros((T, n + m, n + m))
    for k in range(T):
        Z = Z.at[k, :n, :n].set(Q_stage[k])
        Z = Z.at[k, n:, n:].set(R_stage[k])
        Z = Z.at[k, :n, n:].set(M_stage[k])
        Z = Z.at[k, n:, :n].set(M_stage[k].T)

    # Terminal cost
    Q_T = Q[-1]

    # Constraint dimensions
    n_cu = Cu.shape[1] if Cu is not None else 0
    n_cx = Cx.shape[1] if Cx is not None else 0

    return QPFormulation(
        state_dim=n,
        control_dim=m,
        horizon=T,
        n_control_constraints=n_cu,
        n_state_constraints=n_cx,
        Z=Z,
        Q_T=Q_T,
        Q=Q,
        q=q,
        r=r[:T] if r.shape[0] > T else r,
        A=A,
        B=B,
        Cu=Cu,
        Ju=Ju,
        Cx=Cx,
        Jx=Jx,
    )
