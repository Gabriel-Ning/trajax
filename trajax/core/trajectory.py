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

"""Trajectory data structures for trajectory optimization."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import jax.numpy as jnp
from jax import Array

from trajax.core.types import SolverStatus, LQRParams


@dataclass
class Trajectory:
    """Container for trajectory optimization results.

    This dataclass holds the state and control trajectories along with
    optimization metadata. It is the standard return type for all solvers.

    Attributes:
        X: State trajectory of shape (T+1, n), where T is the horizon
            and n is the state dimension. X[t] is the state at time t.
        U: Control trajectory of shape (T, m), where m is the control
            dimension. U[t] is the control applied at time t.
        obj: Total objective value (scalar).
        gradient: Gradient of objective w.r.t. controls, shape (T, m).
            None if not computed.
        status: Solver status indicating convergence or failure mode.
        info: Dictionary containing solver-specific information such as:
            - 'iterations': Number of iterations performed
            - 'adjoints': Adjoint variables (costate trajectory)
            - 'lqr': Final LQR parameters (Q, q, R, r, M, A, B)
            - 'constraint_violation': Maximum constraint violation
            - 'duals': Dual variables for constraints

    Example:
        >>> result = ilqr.solve(problem, x0, U0, params)
        >>> print(f"Converged in {result.info['iterations']} iterations")
        >>> print(f"Final cost: {result.obj}")
        >>> u_optimal = result.U[0]  # First optimal control
    """

    X: Array
    U: Array
    obj: float
    gradient: Optional[Array] = None
    status: SolverStatus = SolverStatus.UNKNOWN
    info: Dict[str, Any] = field(default_factory=dict)

    @property
    def horizon(self) -> int:
        """Return the time horizon T."""
        return self.U.shape[0]

    @property
    def state_dim(self) -> int:
        """Return the state dimension n."""
        return self.X.shape[1]

    @property
    def control_dim(self) -> int:
        """Return the control dimension m."""
        return self.U.shape[1]

    @property
    def converged(self) -> bool:
        """Return True if solver converged successfully."""
        return self.status == SolverStatus.SOLVED

    def shift(self, fill_control: Optional[Array] = None) -> 'Trajectory':
        """Shift trajectory for MPC warm-starting.

        Drops the first state/control and duplicates the last control.
        This is useful for warm-starting the next MPC solve.

        Args:
            fill_control: Control to use for the last timestep.
                If None, duplicates the last control U[-1].

        Returns:
            New Trajectory with shifted X and U.

        Example:
            >>> # After executing first control, warm-start next solve
            >>> warm_start = result.shift()
            >>> next_result = solver.solve(problem, x_new, warm_start.U, params)
        """
        if fill_control is None:
            fill_control = self.U[-1]

        # Shift states: drop first, duplicate last
        X_shifted = jnp.vstack([self.X[1:], self.X[-1:]])

        # Shift controls: drop first, append fill_control
        U_shifted = jnp.vstack([self.U[1:], fill_control[None, :]])

        return Trajectory(
            X=X_shifted,
            U=U_shifted,
            obj=self.obj,
            gradient=None,  # Gradient no longer valid
            status=SolverStatus.UNKNOWN,  # Status unknown for shifted traj
            info={},
        )

    def truncate(self, new_horizon: int) -> 'Trajectory':
        """Truncate trajectory to a shorter horizon.

        Args:
            new_horizon: New horizon length (must be <= current horizon).

        Returns:
            New Trajectory with truncated X and U.
        """
        if new_horizon > self.horizon:
            raise ValueError(
                f"new_horizon ({new_horizon}) must be <= current horizon ({self.horizon})"
            )

        return Trajectory(
            X=self.X[:new_horizon + 1],
            U=self.U[:new_horizon],
            obj=self.obj,  # Note: obj is not recomputed
            gradient=self.gradient[:new_horizon] if self.gradient is not None else None,
            status=self.status,
            info=self.info,
        )

    def get_lqr_params(self) -> Optional[LQRParams]:
        """Get LQR parameters from info dict if available.

        Returns:
            Tuple (Q, q, R, r, M, A, B) or None if not available.
        """
        return self.info.get('lqr', None)

    def get_adjoints(self) -> Optional[Array]:
        """Get adjoint variables (costate) from info dict if available.

        Returns:
            Adjoint trajectory of shape (T+1, n) or None if not available.
        """
        return self.info.get('adjoints', None)


def trajectory_from_controls(
    dynamics_fn,
    U: Array,
    x0: Array,
    params=(),
) -> Trajectory:
    """Create a Trajectory by rolling out controls through dynamics.

    Args:
        dynamics_fn: Dynamics function with signature (x, u, t, params) -> x_next.
        U: Control sequence of shape (T, m).
        x0: Initial state of shape (n,).
        params: Parameters to pass to dynamics.

    Returns:
        Trajectory with X computed by rollout, U as given, and obj=inf.
    """
    from trajax.core.problem import _rollout  # Avoid circular import

    X = _rollout(dynamics_fn, U, x0, params)

    return Trajectory(
        X=X,
        U=U,
        obj=float('inf'),  # Not computed
        gradient=None,
        status=SolverStatus.UNKNOWN,
        info={},
    )
