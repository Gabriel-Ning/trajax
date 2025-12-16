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

"""Cross-Entropy Method (CEM) optimizer.

CEM is a sampling-based zeroth-order optimizer that iteratively refines
a distribution over control sequences by keeping elite samples.
"""

from typing import Any, Callable, Dict, Optional

import jax
import jax.numpy as jnp
from jax import Array, lax, random, vmap

from trajax.core.types import PyTree, SolverStatus
from trajax.core.problem import TrajectoryProblem
from trajax.core.trajectory import Trajectory
from trajax.solvers.base import TrajectoryOptimizerBase
from trajax.utils.rollout import rollout, objective


class CEMOptimizer(TrajectoryOptimizerBase):
    """Cross-Entropy Method optimizer.

    CEM is a derivative-free optimizer that works by:
    1. Sampling control sequences from a Gaussian distribution
    2. Evaluating all samples
    3. Keeping the elite (best) fraction
    4. Updating the distribution to match the elites
    5. Repeating until convergence

    Useful when gradients are unavailable or the cost landscape is non-smooth.

    Attributes:
        name: "cem"
        supports_constraints: False (handles bounds via clipping)
        is_jittable: True
    """

    name = "cem"
    supports_constraints = False
    is_jittable = True

    def __init__(
        self,
        num_samples: int = 400,
        elite_portion: float = 0.1,
        max_iter: int = 10,
        sampling_smoothing: float = 0.0,
        evolution_smoothing: float = 0.1,
        seed: int = 0,
    ):
        """Initialize CEM optimizer.

        Args:
            num_samples: Number of control sequences to sample per iteration.
            elite_portion: Fraction of samples to keep as elites.
            max_iter: Maximum CEM iterations.
            sampling_smoothing: Temporal smoothing for sampled controls.
            evolution_smoothing: Smoothing for distribution updates.
            seed: Random seed.
        """
        super().__init__(
            num_samples=num_samples,
            elite_portion=elite_portion,
            max_iter=max_iter,
            sampling_smoothing=sampling_smoothing,
            evolution_smoothing=evolution_smoothing,
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
        """Internal CEM solve implementation."""
        num_samples = options.get('num_samples', 400)
        elite_portion = options.get('elite_portion', 0.1)
        max_iter = options.get('max_iter', 10)
        sampling_smoothing = options.get('sampling_smoothing', 0.0)
        evolution_smoothing = options.get('evolution_smoothing', 0.1)
        seed = options.get('seed', 0)

        # Get control bounds
        if problem.control_bounds is not None:
            u_min, u_max = problem.control_bounds
        else:
            u_min = jnp.full(problem.control_dim, -jnp.inf)
            u_max = jnp.full(problem.control_dim, jnp.inf)

        # Create wrapped functions
        def cost(x, u, t):
            return problem.cost(x, u, t, params)

        def dynamics(x, u, t):
            return problem.dynamics(x, u, t, params)

        # Initialize distribution
        mean = U0
        stdev = jnp.array([(u_max - u_min) / 2.0] * problem.horizon)
        stdev = jnp.clip(stdev, 0.1, 100.0)  # Ensure reasonable stdev

        key = random.PRNGKey(seed)
        num_elites = int(num_samples * elite_portion)

        def obj_fn(U):
            return objective(cost, dynamics, U, x0)

        def loop_body(_, args):
            mean, stdev, key = args
            key, subkey = random.split(key)

            # Sample controls
            samples = self._gaussian_samples(
                subkey, mean, stdev, u_min, u_max,
                num_samples, sampling_smoothing,
            )

            # Evaluate all samples
            costs = vmap(obj_fn)(samples)

            # Select elites
            elite_idx = jnp.argsort(costs)[:num_elites]
            elites = samples[elite_idx]

            # Update distribution
            new_mean = jnp.mean(elites, axis=0)
            new_stdev = jnp.std(elites, axis=0) + 1e-6

            mean = evolution_smoothing * mean + (1 - evolution_smoothing) * new_mean
            stdev = evolution_smoothing * stdev + (1 - evolution_smoothing) * new_stdev

            return mean, stdev, key

        mean, stdev, key = lax.fori_loop(0, max_iter, loop_body, (mean, stdev, key))

        # Final rollout with optimized mean
        X = rollout(dynamics, mean, x0)
        obj = objective(cost, dynamics, mean, x0)

        return Trajectory(
            X=X,
            U=mean,
            obj=float(obj),
            gradient=None,
            status=SolverStatus.SOLVED,
            info={'final_stdev': stdev},
        )

    @staticmethod
    def _gaussian_samples(key, mean, stdev, u_min, u_max, num_samples, smoothing):
        """Sample control sequences from Gaussian distribution."""
        T, m = mean.shape
        noises = random.normal(key, shape=(num_samples, T, m))

        # Temporal smoothing
        if smoothing > 0:
            def smooth_body(t, noises):
                return noises.at[:, t].set(
                    smoothing * noises[:, t - 1]
                    + jnp.sqrt(1 - smoothing**2) * noises[:, t]
                )
            noises = lax.fori_loop(1, T, smooth_body, noises)

        samples = mean + noises * stdev
        samples = jnp.clip(samples, u_min, u_max)
        return samples

    def build_solver(
        self,
        problem: TrajectoryProblem,
        options: Optional[Dict[str, Any]] = None,
    ) -> Callable[[Array, Array, PyTree], Trajectory]:
        """Build JIT-compiled CEM solver."""
        merged_options = {**self.default_options}
        if options:
            merged_options.update(options)

        @jax.jit
        def solve_fn(x0: Array, U0: Array, params: PyTree) -> Trajectory:
            return self._solve_impl(problem, x0, U0, params, merged_options)

        return solve_fn
