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

"""Model Predictive Control (MPC) interface.

This module provides a unified interface for MPC controllers using
trajectory optimization solvers from trajax.solvers.

Key components:
- MPCProblem: MPC problem specification with reference tracking
- MPCConfig: Configuration for MPC controller and solver
- MPCController: Receding horizon controller with warm starting
- LinearizedMPCController: Controller with explicit linearization caching

Example:
    >>> from trajax.mpc import MPCProblem, MPCConfig, MPCController
    >>>
    >>> # Define problem
    >>> problem = MPCProblem.tracking_problem(
    ...     dynamics=dynamics_fn,
    ...     x_ref=reference_trajectory,
    ...     u_ref=nominal_controls,
    ...     Q=jnp.diag([10.0, 10.0, 1.0, 1.0]),
    ...     R=jnp.diag([0.1, 0.1]),
    ... )
    >>>
    >>> # Configure controller
    >>> config = MPCConfig(
    ...     horizon=20,
    ...     dt=0.01,
    ...     warm_start=True,
    ...     relinearize_every=5,
    ...     solver=SolverConfig(solver_type='ilqr', maxiter=50)
    ... )
    >>>
    >>> # Create and build controller
    >>> controller = MPCController(problem, config)
    >>> controller.build()  # JIT compile once
    >>>
    >>> # Control loop
    >>> for t in range(num_steps):
    ...     u, info = controller.step(x_current)
    ...     x_current = simulate(x_current, u)
"""

from trajax.mpc.problem import MPCProblem
from trajax.mpc.config import (
    MPCConfig,
    SolverConfig,
    CostConfig,
    ControllerConfig,
)
from trajax.mpc.controller import (
    MPCController,
    LinearizedMPCController,
    MPCState,
)

__all__ = [
    # Problem
    'MPCProblem',
    # Configuration
    'MPCConfig',
    'SolverConfig',
    'CostConfig',
    'ControllerConfig',
    # Controllers
    'MPCController',
    'LinearizedMPCController',
    'MPCState',
]
