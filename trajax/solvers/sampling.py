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

"""Sampling-based optimizers for trajectory optimization.

Simple baseline methods using random sampling of control sequences.
"""

from typing import Any, Callable, Dict, Optional

import jax
import jax.numpy as jnp
from jax import Array, random, vmap

from trajax.core.types import PyTree, SolverStatus
from trajax.core.problem import TrajectoryProblem
from trajax.core.trajectory import Trajectory
from trajax.solvers.base import TrajectoryOptimizerBase
from trajax.utils.rollout import rollout, objective


class RandomShootingOptimizer(TrajectoryOptimizerBase):
    """Random shooting optimizer.

    A simple baseline that samples random control sequences and returns
    the best one. No iterative refinement is performed.

    Useful as a baseline or for initializing other optimizers.

    Attributes:
        name: "random_shooting"
        supports_constraints: False
        is_jittable: True
    """

    name = "random_shooting"
    supports_constraints = False
    is_jittable = True

    def __init__(
        self,
        num_samples: int = 400,
        sampling_smoothing: float = 0.0,
        seed: int = 0,
    ):
        """Initialize random shooting optimizer.

        Args:
            num_samples: Number of control sequences to sample.
            sampling_smoothing: Temporal smoothing for sampled controls.
            seed: Random seed.
        """
        super().__init__(
            num_samples=num_samples,
            sampling_smoothing=sampling_smoothing,
            seed=seed,
        )

    def _solve_impl(
        self,
        problem: TrajectoryProblem,
        x0: Array,
        U0: Array,
        params: PyTree,
        options: Dict[str, Any],
    ) -> Trajectory:
        """Internal random shooting solve implementation."""
        num_samples = options.get('num_samples', 400)
        sampling_smoothing = options.get('sampling_smoothing', 0.0)
        seed = options.get('seed', 0)

        # Get control bounds
        if problem.control_bounds is not None:
            u_min, u_max = problem.control_bounds
        else:
            u_min = jnp.full(problem.control_dim, -1.0)
            u_max = jnp.full(problem.control_dim, 1.0)

        # Create wrapped functions
        def cost(x, u, t):
            return problem.cost(x, u, t, params)

        def dynamics(x, u, t):
            return problem.dynamics(x, u, t, params)

        # Sample controls
        key = random.PRNGKey(seed)
        mean = U0
        stdev = (u_max - u_min) / 2.0

        T, m = U0.shape
        noises = random.normal(key, shape=(num_samples, T, m))

        # Temporal smoothing
        if sampling_smoothing > 0:
            def smooth_body(t, noises):
                return noises.at[:, t].set(
                    sampling_smoothing * noises[:, t - 1]
                    + jnp.sqrt(1 - sampling_smoothing**2) * noises[:, t]
                )
            noises = jax.lax.fori_loop(1, T, smooth_body, noises)

        samples = mean + noises * stdev
        samples = jnp.clip(samples, u_min, u_max)

        # Evaluate all samples
        def obj_fn(U):
            return objective(cost, dynamics, U, x0)

        costs = vmap(obj_fn)(samples)

        # Select best
        best_idx = jnp.argmin(costs)
        U_best = samples[best_idx]

        # Rollout
        X = rollout(dynamics, U_best, x0)
        obj = costs[best_idx]

        return Trajectory(
            X=X,
            U=U_best,
            obj=float(obj),
            gradient=None,
            status=SolverStatus.SOLVED,
            info={'num_samples': num_samples},
        )

    def build_solver(
        self,
        problem: TrajectoryProblem,
        options: Optional[Dict[str, Any]] = None,
    ) -> Callable[[Array, Array, PyTree], Trajectory]:
        """Build JIT-compiled random shooting solver."""
        merged_options = {**self.default_options}
        if options:
            merged_options.update(options)

        @jax.jit
        def solve_fn(x0: Array, U0: Array, params: PyTree) -> Trajectory:
            return self._solve_impl(problem, x0, U0, params, merged_options)

        return solve_fn
