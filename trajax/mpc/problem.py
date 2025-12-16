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

"""MPC problem specification extending TrajectoryProblem.

Provides MPC-specific problem formulation with reference tracking,
linearization management, and receding horizon control.
"""

from dataclasses import dataclass, field
from typing import Callable, Optional

import jax.numpy as jnp
from jax import Array

from trajax.core.problem import TrajectoryProblem
from trajax.core.types import PyTree


@dataclass
class MPCProblem(TrajectoryProblem):
    """MPC problem specification with tracking references.

    Extends TrajectoryProblem with MPC-specific features:
    - Reference trajectory tracking
    - Periodic linearization
    - State/control bounds
    - Warm starting

    Attributes:
        x_ref: Reference state trajectory (T+1, n) or function.
        u_ref: Reference control trajectory (T, m) or function.
        linearization_point: Cached linearization for relinearization.
        relinearize_every: Relinearize dynamics every N steps (0=never).
    """
    x_ref: Optional[Array | Callable] = None
    u_ref: Optional[Array | Callable] = None
    linearization_point: Optional[tuple[Array, Array]] = None
    relinearize_every: int = 1

    def get_reference(
        self,
        t: int,
        horizon: int,
        params: PyTree = ()
    ) -> tuple[Array, Array]:
        """Get reference trajectory for MPC problem.

        Args:
            t: Current timestep.
            horizon: Prediction horizon.
            params: Optional parameters.

        Returns:
            Tuple of (x_ref, u_ref) for the prediction horizon.
        """
        # Handle callable references
        if callable(self.x_ref):
            x_ref = jnp.stack([
                self.x_ref(t + k, params) for k in range(horizon + 1)
            ])
        elif self.x_ref is not None:
            # Extract window from trajectory
            x_ref = self.x_ref[t:t + horizon + 1]
        else:
            # Zero reference
            x_ref = jnp.zeros((horizon + 1, self.state_dim))

        if callable(self.u_ref):
            u_ref = jnp.stack([
                self.u_ref(t + k, params) for k in range(horizon)
            ])
        elif self.u_ref is not None:
            u_ref = self.u_ref[t:t + horizon]
        else:
            u_ref = jnp.zeros((horizon, self.control_dim))

        return x_ref, u_ref

    @classmethod
    def tracking_problem(
        cls,
        dynamics: Callable,
        x_ref: Array | Callable,
        u_ref: Array | Callable,
        Q: Array,
        R: Array,
        Q_terminal: Optional[Array] = None,
        **kwargs
    ) -> 'MPCProblem':
        """Create tracking MPC problem with quadratic cost.

        Constructs cost function:
            (x - x_ref)' Q (x - x_ref) + (u - u_ref)' R (u - u_ref)

        Args:
            dynamics: Dynamics function (x, u, t, params) -> x_next.
            x_ref: Reference state trajectory or function.
            u_ref: Reference control trajectory or function.
            Q: State cost weight (n, n) or scalar.
            R: Control cost weight (m, m) or scalar.
            Q_terminal: Terminal cost weight (defaults to Q).
            **kwargs: Additional TrajectoryProblem arguments.

        Returns:
            MPCProblem with quadratic tracking cost.
        """
        if Q_terminal is None:
            Q_terminal = Q

        # Ensure Q, R are arrays
        Q = jnp.atleast_1d(jnp.asarray(Q))
        R = jnp.atleast_1d(jnp.asarray(R))
        Q_terminal = jnp.atleast_1d(jnp.asarray(Q_terminal))

        def cost_fn(x, u, t, params):
            """Quadratic tracking cost."""
            # Get reference at time t
            if callable(x_ref):
                x_t = x_ref(t, params)
            else:
                x_t = x_ref[t] if x_ref.ndim > 1 else x_ref

            if callable(u_ref):
                u_t = u_ref(t, params)
            else:
                u_t = u_ref[t] if u_ref.ndim > 1 else u_ref

            # Tracking error
            dx = x - x_t
            du = u - u_t

            # Quadratic cost
            if Q.ndim == 1:  # Diagonal Q
                state_cost = jnp.sum(Q * dx * dx)
            else:
                state_cost = dx @ Q @ dx

            if R.ndim == 1:  # Diagonal R
                control_cost = jnp.sum(R * du * du)
            else:
                control_cost = du @ R @ du

            return state_cost + control_cost

        def terminal_cost_fn(x, t, params):
            """Quadratic terminal cost."""
            if callable(x_ref):
                x_t = x_ref(t, params)
            else:
                x_t = x_ref[t] if x_ref.ndim > 1 else x_ref

            dx = x - x_t

            if Q_terminal.ndim == 1:
                return jnp.sum(Q_terminal * dx * dx)
            else:
                return dx @ Q_terminal @ dx

        # Infer dimensions if not provided
        if 'state_dim' not in kwargs:
            if callable(x_ref):
                # Cannot infer from callable, must be provided
                raise ValueError(
                    "state_dim must be provided when x_ref is callable"
                )
            kwargs['state_dim'] = x_ref.shape[-1]

        if 'control_dim' not in kwargs:
            if callable(u_ref):
                raise ValueError(
                    "control_dim must be provided when u_ref is callable"
                )
            kwargs['control_dim'] = u_ref.shape[-1]

        return cls(
            dynamics=dynamics,
            cost=cost_fn,
            terminal_cost=terminal_cost_fn,
            x_ref=x_ref,
            u_ref=u_ref,
            **kwargs
        )
