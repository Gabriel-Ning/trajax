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

"""Trajectory optimization problem specification."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import Array, lax

from trajax.core.types import (
    PyTree,
    DynamicsFn,
    CostFn,
    TerminalCostFn,
    ConstraintFn,
    Bounds,
)


def _rollout(dynamics: Callable, U: Array, x0: Array, params: PyTree = ()) -> Array:
    """Roll out dynamics: x[t+1] = dynamics(x[t], U[t], t, params).

    Args:
        dynamics: Dynamics function with signature (x, u, t, params) -> x_next.
        U: Control sequence of shape (T, m).
        x0: Initial state of shape (n,).
        params: Parameters to pass to dynamics.

    Returns:
        X: State trajectory of shape (T+1, n).
    """
    def dynamics_for_scan(x, ut):
        u, t = ut
        x_next = dynamics(x, u, t, params)
        return x_next, x_next

    _, X_rest = lax.scan(dynamics_for_scan, x0, (U, jnp.arange(U.shape[0])))
    return jnp.vstack((x0, X_rest))


@dataclass
class TrajectoryProblem:
    """Unified trajectory optimization problem specification.

    This class encapsulates all the components needed to define a trajectory
    optimization problem:

        min  sum_{t=0}^{T-1} cost(x_t, u_t, t, params) + terminal_cost(x_T, params)
        s.t. x_{t+1} = dynamics(x_t, u_t, t, params)
             inequality_constraint(x_t, u_t, t, params) <= 0
             equality_constraint(x_t, u_t, t, params) = 0
             u_min <= u_t <= u_max
             x_min <= x_t <= x_max

    The problem supports parameterization via the `params` argument, enabling
    the "build once, pass parameters" pattern for efficient JIT compilation.

    Attributes:
        state_dim: Dimension of the state vector (n).
        control_dim: Dimension of the control vector (m).
        horizon: Time horizon (T). The trajectory has T+1 states and T controls.
        dynamics: Dynamics function (x, u, t, params) -> x_next.
        cost: Stage cost function (x, u, t, params) -> scalar.
        terminal_cost: Terminal cost function (x, params) -> scalar.
            If None, defaults to cost(x, zeros(m), T, params).
        inequality_constraint: Inequality constraint function returning g(x,u,t) <= 0.
        equality_constraint: Equality constraint function returning h(x,u,t) = 0.
        control_bounds: Tuple (u_min, u_max) for box constraints on controls.
        state_bounds: Tuple (x_min, x_max) for box constraints on states.

    Example:
        >>> def pendulum_dynamics(x, u, t, params):
        ...     theta, omega = x
        ...     dt = params['dt']
        ...     return jnp.array([
        ...         theta + dt * omega,
        ...         omega + dt * (-jnp.sin(theta) + u[0])
        ...     ])
        ...
        >>> def cost(x, u, t, params):
        ...     return 0.5 * (x @ params['Q'] @ x + u @ params['R'] @ u)
        ...
        >>> problem = TrajectoryProblem(
        ...     state_dim=2,
        ...     control_dim=1,
        ...     horizon=100,
        ...     dynamics=pendulum_dynamics,
        ...     cost=cost,
        ...     control_bounds=(jnp.array([-1.0]), jnp.array([1.0])),
        ... )
    """

    # Required fields
    state_dim: int
    control_dim: int
    horizon: int
    dynamics: DynamicsFn
    cost: CostFn

    # Optional fields
    terminal_cost: Optional[TerminalCostFn] = None
    inequality_constraint: Optional[ConstraintFn] = None
    equality_constraint: Optional[ConstraintFn] = None
    control_bounds: Optional[Bounds] = None
    state_bounds: Optional[Bounds] = None

    # Constraint dimensions (computed or specified)
    n_inequality: int = 0
    n_equality: int = 0

    def __post_init__(self):
        """Validate problem specification."""
        if self.state_dim < 1:
            raise ValueError(f"state_dim must be >= 1, got {self.state_dim}")
        if self.control_dim < 1:
            raise ValueError(f"control_dim must be >= 1, got {self.control_dim}")
        if self.horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {self.horizon}")

    @property
    def has_constraints(self) -> bool:
        """Return True if problem has any constraints."""
        return (
            self.inequality_constraint is not None
            or self.equality_constraint is not None
            or self.control_bounds is not None
            or self.state_bounds is not None
        )

    @property
    def has_box_constraints(self) -> bool:
        """Return True if problem has box constraints."""
        return self.control_bounds is not None or self.state_bounds is not None

    def rollout(self, U: Array, x0: Array, params: PyTree = ()) -> Array:
        """Roll out dynamics given control sequence.

        Args:
            U: Control sequence of shape (T, m).
            x0: Initial state of shape (n,).
            params: Parameters to pass to dynamics.

        Returns:
            X: State trajectory of shape (T+1, n).
        """
        return _rollout(self.dynamics, U, x0, params)

    def evaluate_cost(
        self,
        X: Array,
        U: Array,
        params: PyTree = (),
    ) -> float:
        """Evaluate total cost along trajectory.

        Args:
            X: State trajectory of shape (T+1, n).
            U: Control trajectory of shape (T, m).
            params: Parameters to pass to cost functions.

        Returns:
            Total cost (scalar).
        """
        T = U.shape[0]
        timesteps = jnp.arange(T + 1)
        U_padded = jnp.vstack([U, jnp.zeros((1, self.control_dim))])

        # Stage costs
        def stage_cost(x, u, t):
            return self.cost(x, u, t, params)

        costs = jax.vmap(stage_cost)(X[:-1], U, timesteps[:-1])

        # Terminal cost
        if self.terminal_cost is not None:
            terminal = self.terminal_cost(X[-1], params)
        else:
            terminal = self.cost(X[-1], jnp.zeros(self.control_dim), T, params)

        return jnp.sum(costs) + terminal

    def objective(self, U: Array, x0: Array, params: PyTree = ()) -> float:
        """Compute objective value for control sequence.

        This is a convenience method that rolls out dynamics and evaluates cost.

        Args:
            U: Control sequence of shape (T, m).
            x0: Initial state of shape (n,).
            params: Parameters to pass to dynamics and cost.

        Returns:
            Total objective value (scalar).
        """
        X = self.rollout(U, x0, params)
        return self.evaluate_cost(X, U, params)

    def evaluate_constraints(
        self,
        X: Array,
        U: Array,
        params: PyTree = (),
    ) -> Tuple[Optional[Array], Optional[Array]]:
        """Evaluate constraints along trajectory.

        Args:
            X: State trajectory of shape (T+1, n).
            U: Control trajectory of shape (T, m).
            params: Parameters to pass to constraint functions.

        Returns:
            Tuple (inequality_values, equality_values) where each is an array
            of shape (T+1, n_constraint) or None if no such constraint exists.
        """
        T = U.shape[0]
        timesteps = jnp.arange(T + 1)
        U_padded = jnp.vstack([U, jnp.zeros((1, self.control_dim))])

        inequality_values = None
        equality_values = None

        if self.inequality_constraint is not None:
            def ineq_fn(x, u, t):
                return self.inequality_constraint(x, u, t, params)
            inequality_values = jax.vmap(ineq_fn)(X, U_padded, timesteps)

        if self.equality_constraint is not None:
            def eq_fn(x, u, t):
                return self.equality_constraint(x, u, t, params)
            equality_values = jax.vmap(eq_fn)(X, U_padded, timesteps)

        return inequality_values, equality_values

    def max_constraint_violation(
        self,
        X: Array,
        U: Array,
        params: PyTree = (),
    ) -> float:
        """Compute maximum constraint violation.

        Args:
            X: State trajectory of shape (T+1, n).
            U: Control trajectory of shape (T, m).
            params: Parameters to pass to constraint functions.

        Returns:
            Maximum constraint violation (scalar). Returns 0.0 if no constraints.
        """
        ineq, eq = self.evaluate_constraints(X, U, params)

        violations = []
        if ineq is not None:
            # Inequality: g(x,u,t) <= 0, violation is max(0, g)
            violations.append(jnp.max(jnp.maximum(0, ineq)))
        if eq is not None:
            # Equality: h(x,u,t) = 0, violation is |h|
            violations.append(jnp.max(jnp.abs(eq)))

        # Box constraints on controls
        if self.control_bounds is not None:
            u_min, u_max = self.control_bounds
            violations.append(jnp.max(jnp.maximum(0, u_min - U)))
            violations.append(jnp.max(jnp.maximum(0, U - u_max)))

        # Box constraints on states
        if self.state_bounds is not None:
            x_min, x_max = self.state_bounds
            violations.append(jnp.max(jnp.maximum(0, x_min - X)))
            violations.append(jnp.max(jnp.maximum(0, X - x_max)))

        if not violations:
            return 0.0

        return jnp.max(jnp.array(violations))

    @classmethod
    def from_functions(
        cls,
        dynamics: Callable,
        cost: Callable,
        state_dim: int,
        control_dim: int,
        horizon: int,
        **kwargs,
    ) -> 'TrajectoryProblem':
        """Create problem from plain functions.

        This is a convenience constructor for creating problems from
        functions that don't take a params argument.

        Args:
            dynamics: Dynamics function (x, u, t) -> x_next.
            cost: Cost function (x, u, t) -> scalar.
            state_dim: State dimension.
            control_dim: Control dimension.
            horizon: Time horizon.
            **kwargs: Additional arguments passed to TrajectoryProblem.

        Returns:
            TrajectoryProblem instance.
        """
        # Wrap functions to accept params argument
        def wrapped_dynamics(x, u, t, params=()):
            return dynamics(x, u, t)

        def wrapped_cost(x, u, t, params=()):
            return cost(x, u, t)

        return cls(
            state_dim=state_dim,
            control_dim=control_dim,
            horizon=horizon,
            dynamics=wrapped_dynamics,
            cost=wrapped_cost,
            **kwargs,
        )


def create_tracking_problem(
    dynamics: DynamicsFn,
    Q: Array,
    R: Array,
    Q_terminal: Optional[Array] = None,
    x_ref: Optional[Array] = None,
    u_ref: Optional[Array] = None,
    state_dim: Optional[int] = None,
    control_dim: Optional[int] = None,
    horizon: int = 10,
    **kwargs,
) -> TrajectoryProblem:
    """Create a quadratic tracking problem.

    Creates a problem with quadratic cost:
        cost = 0.5 * (x - x_ref)' Q (x - x_ref) + 0.5 * (u - u_ref)' R (u - u_ref)

    Args:
        dynamics: Dynamics function.
        Q: State cost matrix (n, n).
        R: Control cost matrix (m, m).
        Q_terminal: Terminal state cost matrix. If None, uses Q.
        x_ref: Reference state trajectory (T+1, n) or (n,) for constant.
            If None, uses zeros.
        u_ref: Reference control trajectory (T, m) or (m,) for constant.
            If None, uses zeros.
        state_dim: State dimension. Inferred from Q if not provided.
        control_dim: Control dimension. Inferred from R if not provided.
        horizon: Time horizon.
        **kwargs: Additional arguments passed to TrajectoryProblem.

    Returns:
        TrajectoryProblem with quadratic tracking cost.
    """
    n = state_dim or Q.shape[0]
    m = control_dim or R.shape[0]

    if Q_terminal is None:
        Q_terminal = Q

    def cost(x, u, t, params):
        Q_t, R_t, x_ref_t, u_ref_t = params['Q'], params['R'], params['x_ref'], params['u_ref']

        # Handle reference indexing
        if x_ref_t.ndim == 1:
            x_r = x_ref_t
        else:
            x_r = x_ref_t[t]

        if u_ref_t.ndim == 1:
            u_r = u_ref_t
        else:
            u_r = jnp.where(t < horizon, u_ref_t[t], jnp.zeros(m))

        dx = x - x_r
        du = u - u_r
        return 0.5 * (dx @ Q_t @ dx + du @ R_t @ du)

    def terminal_cost(x, params):
        Q_T = params['Q_terminal']
        x_ref_t = params['x_ref']

        if x_ref_t.ndim == 1:
            x_r = x_ref_t
        else:
            x_r = x_ref_t[-1]

        dx = x - x_r
        return 0.5 * dx @ Q_T @ dx

    # Default references
    if x_ref is None:
        x_ref = jnp.zeros(n)
    if u_ref is None:
        u_ref = jnp.zeros(m)

    return TrajectoryProblem(
        state_dim=n,
        control_dim=m,
        horizon=horizon,
        dynamics=dynamics,
        cost=cost,
        terminal_cost=terminal_cost,
        **kwargs,
    )
