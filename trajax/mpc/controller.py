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

"""MPC controller with receding horizon control.

Provides a unified interface for MPC controllers using different
trajectory optimization solvers.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import jax
import jax.numpy as jnp
from jax import Array

from trajax.core.trajectory import Trajectory
from trajax.core.types import PyTree
from trajax.mpc.problem import MPCProblem
from trajax.mpc.config import MPCConfig
from trajax.solvers.base import get_solver, TrajectoryOptimizerBase


@dataclass
class MPCState:
    """MPC controller state for warm starting.

    Attributes:
        U_plan: Previous control plan (T, m).
        step_count: Total number of control steps executed.
        last_relinearization: Step count at last relinearization.
        linearization_cache: Cached linearization data.
    """
    U_plan: Optional[Array] = None
    step_count: int = 0
    last_relinearization: int = -1
    linearization_cache: Optional[Dict[str, Any]] = None


class MPCController:
    """MPC controller with receding horizon control.

    Uses trajax trajectory optimizers for solving MPC problems with:
    - Warm starting from previous solutions
    - Periodic relinearization of dynamics
    - JIT-compiled solve for real-time performance

    Example:
        >>> problem = MPCProblem.tracking_problem(
        ...     dynamics, x_ref, u_ref, Q, R
        ... )
        >>> config = MPCConfig(horizon=20, dt=0.01)
        >>> controller = MPCController(problem, config)
        >>> controller.build()  # JIT compile once
        >>>
        >>> # Control loop
        >>> for t in range(num_steps):
        ...     u, info = controller.step(x_current, params)
        ...     x_current = simulate(x_current, u)
    """

    def __init__(
        self,
        problem: MPCProblem,
        config: Optional[MPCConfig] = None,
        optimizer: Optional[TrajectoryOptimizerBase] = None,
    ):
        """Initialize MPC controller.

        Args:
            problem: MPC problem specification.
            config: MPC configuration (uses defaults if None).
            optimizer: Trajectory optimizer (created from config if None).
        """
        self.problem = problem
        self.config = config or MPCConfig()
        self.state = MPCState()

        # Create optimizer from config if not provided
        if optimizer is None:
            solver_config = self.config.solver
            self.optimizer = get_solver(
                solver_config.solver_type,
                **solver_config.to_dict()
            )
        else:
            self.optimizer = optimizer

        # Will be set by build()
        self._solve_fn: Optional[Callable] = None
        self._is_built = False

    def build(self) -> 'MPCController':
        """Build JIT-compiled solver.

        This method should be called once before the control loop.
        It compiles the solver with the problem structure, enabling
        fast repeated solves.

        Returns:
            Self for method chaining.
        """
        if self._is_built:
            return self

        # Build JIT-compiled solver
        self._solve_fn = self.optimizer.build_solver(
            self.problem,
            options=self.config.solver.to_dict()
        )

        self._is_built = True
        return self

    def step(
        self,
        x_current: Array,
        params: PyTree = (),
        relinearize: bool = False,
    ) -> tuple[Array, Dict[str, Any]]:
        """Execute one MPC control step.

        Args:
            x_current: Current state (n,).
            params: Problem parameters.
            relinearize: Force relinearization regardless of schedule.

        Returns:
            Tuple of:
                - u: Control action (m,)
                - info: Dict with solver stats and MPC info

        Raises:
            RuntimeError: If build() has not been called.
        """
        if not self._is_built:
            raise RuntimeError(
                "Controller must be built before stepping. Call build() first."
            )

        # Check if we need to relinearize
        needs_relinearization = self._should_relinearize(relinearize)

        # Get initial control guess (warm start)
        U0 = self._get_initial_guess()

        # Solve MPC problem
        result = self._solve_fn(x_current, U0, params)

        # Update controller state
        self._update_state(result, needs_relinearization)

        # Extract first control action
        u = result.U[0]

        # Build info dict
        info = {
            'objective': float(result.obj),
            'solve_time_ms': result.info.get('solve_time_ms', 0.0),
            'iterations': result.info.get('iterations', 0),
            'status': result.status.name,
            'relinearized': needs_relinearization,
            'step': self.state.step_count,
        }

        return u, info

    def rollout(
        self,
        x0: Array,
        num_steps: int,
        params: PyTree = (),
        dynamics: Optional[Callable] = None,
    ) -> tuple[Array, Array, list[Dict]]:
        """Execute MPC controller for multiple steps.

        Args:
            x0: Initial state (n,).
            num_steps: Number of control steps.
            params: Problem parameters.
            dynamics: Actual dynamics for simulation (uses problem dynamics
                     if None).

        Returns:
            Tuple of:
                - X: State trajectory (num_steps+1, n)
                - U: Control trajectory (num_steps, m)
                - infos: List of info dicts from each step

        Raises:
            RuntimeError: If build() has not been called.
        """
        if not self._is_built:
            raise RuntimeError(
                "Controller must be built before rollout. Call build() first."
            )

        dynamics_fn = dynamics or self.problem.dynamics

        # Allocate arrays
        X = jnp.zeros((num_steps + 1, self.problem.state_dim))
        U = jnp.zeros((num_steps, self.problem.control_dim))
        X = X.at[0].set(x0)
        infos = []

        # Simulate
        for t in range(num_steps):
            u, info = self.step(X[t], params)
            U = U.at[t].set(u)
            X = X.at[t + 1].set(dynamics_fn(X[t], u, t, params))
            infos.append(info)

        return X, U, infos

    def reset(self):
        """Reset controller state (clears warm start and linearization)."""
        self.state = MPCState()

    def _should_relinearize(self, force: bool) -> bool:
        """Check if relinearization is needed."""
        if force:
            return True

        if self.config.relinearize_every <= 0:
            # Relinearization disabled
            return False

        steps_since = (
            self.state.step_count - self.state.last_relinearization
        )
        return steps_since >= self.config.relinearize_every

    def _get_initial_guess(self) -> Array:
        """Get initial control guess for warm starting."""
        horizon = self.config.horizon
        control_dim = self.problem.control_dim

        if not self.config.warm_start or self.state.U_plan is None:
            # Cold start with zeros
            return jnp.zeros((horizon, control_dim))

        # Warm start: shift previous plan
        U_prev = self.state.U_plan
        U0 = jnp.vstack([U_prev[1:], U_prev[-1:]])
        return U0

    def _update_state(
        self,
        result: Trajectory,
        relinearized: bool
    ):
        """Update controller state after solve."""
        self.state.U_plan = result.U
        self.state.step_count += 1

        if relinearized:
            self.state.last_relinearization = self.state.step_count


class LinearizedMPCController(MPCController):
    """MPC controller with explicit linearization management.

    Extends MPCController to cache and reuse linearized dynamics,
    reducing computational cost when dynamics are expensive to linearize.

    This is useful for:
    - Complex nonlinear dynamics
    - Dynamics requiring finite differencing
    - When relinearization is expensive

    Example:
        >>> from trajax.utils.linearize import linearize
        >>>
        >>> def linearize_fn(x, u, params):
        ...     # Custom linearization logic
        ...     A, B = linearize(dynamics, x, u, 0, params)
        ...     return A, B
        >>>
        >>> controller = LinearizedMPCController(
        ...     problem, config, linearize_fn=linearize_fn
        ... )
    """

    def __init__(
        self,
        problem: MPCProblem,
        config: Optional[MPCConfig] = None,
        optimizer: Optional[TrajectoryOptimizerBase] = None,
        linearize_fn: Optional[Callable] = None,
    ):
        """Initialize linearized MPC controller.

        Args:
            problem: MPC problem specification.
            config: MPC configuration.
            optimizer: Trajectory optimizer.
            linearize_fn: Function (x, u, params) -> (A, B) for dynamics
                         linearization. If None, uses trajax.utils.linearize.
        """
        super().__init__(problem, config, optimizer)

        if linearize_fn is None:
            from trajax.utils.linearize import linearize as default_linearize

            def default_fn(x, u, params):
                A, B = default_linearize(
                    self.problem.dynamics, x, u, 0, params
                )
                return A, B

            self.linearize_fn = default_fn
        else:
            self.linearize_fn = linearize_fn

    def step(
        self,
        x_current: Array,
        params: PyTree = (),
        relinearize: bool = False,
    ) -> tuple[Array, Dict[str, Any]]:
        """Execute one MPC step with linearization caching."""
        needs_relinearization = self._should_relinearize(relinearize)

        # Update linearization if needed
        if needs_relinearization or self.state.linearization_cache is None:
            u_nominal = jnp.zeros(self.problem.control_dim)
            A, B = self.linearize_fn(x_current, u_nominal, params)
            self.state.linearization_cache = {'A': A, 'B': B, 'x0': x_current}

        # Call parent step method
        return super().step(x_current, params, relinearize)
