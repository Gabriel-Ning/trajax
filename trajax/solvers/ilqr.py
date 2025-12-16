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

"""Iterative Linear Quadratic Regulator (iLQR) optimizer.

iLQR is the workhorse algorithm for unconstrained trajectory optimization.
It iteratively linearizes dynamics and quadratizes cost, then solves the
resulting time-varying LQR problem.
"""

from functools import partial
from typing import Any, Callable, Dict, Optional

import jax
import jax.numpy as jnp
from jax import Array, lax, vmap

from trajax.core.types import PyTree, SolverStatus
from trajax.core.problem import TrajectoryProblem
from trajax.core.trajectory import Trajectory
from trajax.solvers.base import TrajectoryOptimizerBase
from trajax.utils.linearize import linearize, quadratize, pad
from trajax.utils.rollout import rollout, ddp_rollout, line_search_ddp, evaluate
from trajax.utils.adjoint import adjoint
from trajax.utils.psd import project_psd_cone
from trajax.tvlqr import tvlqr


class ILQROptimizer(TrajectoryOptimizerBase):
    """Iterative Linear Quadratic Regulator optimizer.

    iLQR solves unconstrained trajectory optimization problems by iteratively:
    1. Linearizing dynamics around current trajectory
    2. Quadratizing cost around current trajectory
    3. Solving the resulting LQR problem for a descent direction
    4. Line searching along the direction

    The optimizer supports differentiation through the solution via custom VJP,
    making it suitable for learning applications.

    Attributes:
        name: "ilqr"
        supports_constraints: False (use ConstrainedILQROptimizer for constraints)
        is_jittable: True

    Example:
        >>> optimizer = ILQROptimizer(maxiter=100, grad_norm_threshold=1e-4)
        >>> result = optimizer.solve(problem, x0, U0, params)
        >>> print(f"Converged in {result.info['iterations']} iterations")
    """

    name = "ilqr"
    supports_constraints = False
    is_jittable = True

    def __init__(
        self,
        maxiter: int = 100,
        grad_norm_threshold: float = 1e-4,
        relative_grad_norm_threshold: float = 0.0,
        obj_step_threshold: float = 0.0,
        inputs_step_threshold: float = 0.0,
        make_psd: bool = False,
        psd_delta: float = 0.0,
        alpha_0: float = 1.0,
        alpha_min: float = 0.00005,
    ):
        """Initialize iLQR optimizer.

        Args:
            maxiter: Maximum iterations.
            grad_norm_threshold: Stop if gradient norm < threshold.
            relative_grad_norm_threshold: Stop if gradient norm < threshold
                relative to initial gradient norm.
            obj_step_threshold: Stop if objective improvement < threshold
                relative to objective value.
            inputs_step_threshold: Stop if control step norm < threshold
                relative to control norm.
            make_psd: Whether to project Hessians to PSD cone.
            psd_delta: Minimum eigenvalue when projecting to PSD.
            alpha_0: Initial line search step size.
            alpha_min: Minimum line search step size before giving up.
        """
        super().__init__(
            maxiter=maxiter,
            grad_norm_threshold=grad_norm_threshold,
            relative_grad_norm_threshold=relative_grad_norm_threshold,
            obj_step_threshold=obj_step_threshold,
            inputs_step_threshold=inputs_step_threshold,
            make_psd=make_psd,
            psd_delta=psd_delta,
            alpha_0=alpha_0,
            alpha_min=alpha_min,
        )

    def _solve_impl(
        self,
        problem: TrajectoryProblem,
        x0: Array,
        U0: Array,
        params: PyTree,
        options: Dict[str, Any],
    ) -> Trajectory:
        """Internal iLQR solve implementation."""
        # Extract options
        maxiter = options.get('maxiter', 100)
        grad_norm_threshold = options.get('grad_norm_threshold', 1e-4)
        relative_grad_norm_threshold = options.get('relative_grad_norm_threshold', 0.0)
        obj_step_threshold = options.get('obj_step_threshold', 0.0)
        inputs_step_threshold = options.get('inputs_step_threshold', 0.0)
        make_psd = options.get('make_psd', False)
        psd_delta = options.get('psd_delta', 0.0)
        alpha_0 = options.get('alpha_0', 1.0)
        alpha_min = options.get('alpha_min', 0.00005)

        # Create wrapped functions that include params
        def cost(x, u, t):
            return problem.cost(x, u, t, params)

        def dynamics(x, u, t):
            return problem.dynamics(x, u, t, params)

        # Run iLQR algorithm
        X, U, obj, gradient, adjoints, lqr, iterations = self._ilqr_core(
            cost, dynamics, x0, U0,
            maxiter, grad_norm_threshold, relative_grad_norm_threshold,
            obj_step_threshold, inputs_step_threshold,
            make_psd, psd_delta, alpha_0, alpha_min,
        )

        # Determine status
        grad_norm = jnp.linalg.norm(gradient)
        if grad_norm < grad_norm_threshold:
            status = SolverStatus.SOLVED
        elif iterations >= maxiter:
            status = SolverStatus.MAX_ITERATIONS
        else:
            status = SolverStatus.STALLED

        return Trajectory(
            X=X,
            U=U,
            obj=float(obj),
            gradient=gradient,
            status=status,
            info={
                'iterations': int(iterations),
                'adjoints': adjoints,
                'lqr': lqr,
                'grad_norm': float(grad_norm),
            },
        )

    @staticmethod
    def _ilqr_core(
        cost, dynamics, x0, U,
        maxiter, grad_norm_threshold, relative_grad_norm_threshold,
        obj_step_threshold, inputs_step_threshold,
        make_psd, psd_delta, alpha_0, alpha_min,
    ):
        """Core iLQR algorithm implementation."""
        T, m = U.shape
        n = x0.shape[0]

        # Utility functions
        roll = partial(rollout, dynamics)
        quadratizer = quadratize(cost)
        dynamics_jacobians = linearize(dynamics)
        cost_gradients = linearize(cost)
        evaluator = partial(evaluate, cost)
        psd = vmap(partial(project_psd_cone, delta=psd_delta))

        # Initial rollout
        X = roll(U, x0)
        timesteps = jnp.arange(X.shape[0])
        obj = jnp.sum(evaluator(X, pad(U)))

        def get_lqr_params(X, U):
            Q, R, M = quadratizer(X, pad(U), timesteps)
            Q = lax.cond(make_psd, psd, lambda x: x, Q)
            R = lax.cond(make_psd, psd, lambda x: x, R)
            q, r = cost_gradients(X, pad(U), timesteps)
            A, B = dynamics_jacobians(X, pad(U), jnp.arange(T + 1))
            return (Q, q, R, r, M, A, B)

        c = jnp.zeros((T, n))  # Trajectory is always dynamically feasible

        lqr = get_lqr_params(X, U)
        _, q, _, r, _, A, B = lqr
        gradient, adjoints, _ = adjoint(A, B, q, r)
        grad_norm_initial = jnp.linalg.norm(gradient)
        grad_norm_threshold_adj = jnp.maximum(
            grad_norm_threshold,
            relative_grad_norm_threshold * jnp.where(
                jnp.isnan(grad_norm_initial), 1.0, grad_norm_initial + 1.0
            ),
        )

        def body(inputs):
            """One iteration of iLQR."""
            X, U, obj, alpha, gradient, adjoints, lqr, iteration, _, _ = inputs

            Q, q, R, r, M, A, B = lqr

            # Solve LQR for descent direction
            K, k, _, _ = tvlqr(Q, q, R, r, M, A, B, c)

            # Line search
            X_new, U_new, obj_new, alpha = line_search_ddp(
                cost, dynamics, X, U, K, k, obj,
                cost_args=(), dynamics_args=(),
                alpha_0=alpha_0, alpha_min=alpha_min,
            )

            gradient, adjoints, _ = adjoint(A, B, q, r)
            lqr = get_lqr_params(X_new, U_new)
            U_step = jnp.linalg.norm(U_new - U)
            obj_step = jnp.abs(obj_new - obj)
            iteration = iteration + 1

            return (X_new, U_new, obj_new, alpha, gradient, adjoints,
                    lqr, iteration, obj_step, U_step)

        def continuation_criterion(inputs):
            """Check if optimization should continue."""
            _, U_new, obj_new, alpha, gradient, _, _, iteration, obj_step, U_step = inputs
            grad_norm = jnp.linalg.norm(gradient)
            grad_norm = jnp.where(jnp.isnan(grad_norm), jnp.inf, grad_norm)

            still_improving_obj = obj_step > obj_step_threshold * (
                jnp.absolute(obj_new) + 1.0)
            still_moving_U = U_step > inputs_step_threshold * (
                jnp.linalg.norm(U_new) + 1.0)
            still_progressing = jnp.logical_and(still_improving_obj, still_moving_U)
            has_potential = jnp.logical_and(
                grad_norm > grad_norm_threshold_adj, still_progressing)

            return jnp.logical_and(
                iteration < maxiter,
                jnp.logical_and(has_potential, alpha > alpha_min),
            )

        X, U, obj, _, gradient, adjoints, lqr, iterations, _, _ = lax.while_loop(
            continuation_criterion, body,
            (X, U, obj, alpha_0, gradient, adjoints, lqr, 0, jnp.inf, jnp.inf),
        )

        return X, U, obj, gradient, adjoints, lqr, iterations

    def build_solver(
        self,
        problem: TrajectoryProblem,
        options: Optional[Dict[str, Any]] = None,
    ) -> Callable[[Array, Array, PyTree], Trajectory]:
        """Build JIT-compiled iLQR solver.

        The returned function captures the problem structure, allowing
        efficient repeated solves with different parameters.
        """
        merged_options = {**self.default_options}
        if options:
            merged_options.update(options)

        # Extract options
        maxiter = merged_options.get('maxiter', 100)
        grad_norm_threshold = merged_options.get('grad_norm_threshold', 1e-4)
        relative_grad_norm_threshold = merged_options.get('relative_grad_norm_threshold', 0.0)
        obj_step_threshold = merged_options.get('obj_step_threshold', 0.0)
        inputs_step_threshold = merged_options.get('inputs_step_threshold', 0.0)
        make_psd = merged_options.get('make_psd', False)
        psd_delta = merged_options.get('psd_delta', 0.0)
        alpha_0 = merged_options.get('alpha_0', 1.0)
        alpha_min = merged_options.get('alpha_min', 0.00005)

        # Capture problem structure
        dynamics_fn = problem.dynamics
        cost_fn = problem.cost

        @jax.jit
        def solve_fn(x0: Array, U0: Array, params: PyTree) -> Trajectory:
            """JIT-compiled iLQR solve."""
            def cost(x, u, t):
                return cost_fn(x, u, t, params)

            def dynamics(x, u, t):
                return dynamics_fn(x, u, t, params)

            X, U, obj, gradient, adjoints, lqr, iterations = ILQROptimizer._ilqr_core(
                cost, dynamics, x0, U0,
                maxiter, grad_norm_threshold, relative_grad_norm_threshold,
                obj_step_threshold, inputs_step_threshold,
                make_psd, psd_delta, alpha_0, alpha_min,
            )

            grad_norm = jnp.linalg.norm(gradient)
            status = jnp.where(
                grad_norm < grad_norm_threshold,
                SolverStatus.SOLVED.value,
                jnp.where(
                    iterations >= maxiter,
                    SolverStatus.MAX_ITERATIONS.value,
                    SolverStatus.STALLED.value,
                ),
            )

            return Trajectory(
                X=X,
                U=U,
                obj=obj,
                gradient=gradient,
                status=SolverStatus(status),
                info={
                    'iterations': iterations,
                    'adjoints': adjoints,
                    'lqr': lqr,
                    'grad_norm': grad_norm,
                },
            )

        return solve_fn
