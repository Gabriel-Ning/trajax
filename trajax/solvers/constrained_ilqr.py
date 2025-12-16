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

"""Constrained iLQR optimizer using Augmented Lagrangian method.

This optimizer handles trajectory optimization with equality and inequality
constraints using the Augmented Lagrangian approach with iLQR as inner solver.
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
from trajax.utils.linearize import vectorize, pad


class ConstrainedILQROptimizer(TrajectoryOptimizerBase):
    """Constrained iLQR using Augmented Lagrangian method.

    Handles trajectory optimization with constraints:
        min  sum_t cost(x_t, u_t, t)
        s.t. x_{t+1} = dynamics(x_t, u_t, t)
             equality_constraint(x_t, u_t, t) = 0
             inequality_constraint(x_t, u_t, t) <= 0

    The algorithm alternates between:
    1. Solving unconstrained iLQR on augmented Lagrangian cost
    2. Updating dual variables and penalty parameters

    Attributes:
        name: "constrained_ilqr"
        supports_constraints: True
        is_jittable: True
    """

    name = "constrained_ilqr"
    supports_constraints = True
    is_jittable = True

    def __init__(
        self,
        maxiter_al: int = 5,
        maxiter_ilqr: int = 100,
        constraints_threshold: float = 1e-2,
        penalty_init: float = 1.0,
        penalty_update_rate: float = 10.0,
        grad_norm_threshold: float = 1e-4,
        relative_grad_norm_threshold: float = 0.0,
        obj_step_threshold: float = 0.0,
        inputs_step_threshold: float = 0.0,
        make_psd: bool = True,
        psd_delta: float = 0.0,
        alpha_0: float = 1.0,
        alpha_min: float = 0.00005,
    ):
        """Initialize constrained iLQR optimizer.

        Args:
            maxiter_al: Maximum augmented Lagrangian outer iterations.
            maxiter_ilqr: Maximum iLQR iterations per AL iteration.
            constraints_threshold: Tolerance for constraint violations.
            penalty_init: Initial penalty parameter.
            penalty_update_rate: Multiplier for penalty updates.
            grad_norm_threshold: Gradient norm threshold for iLQR.
            relative_grad_norm_threshold: Relative gradient threshold.
            obj_step_threshold: Objective step threshold.
            inputs_step_threshold: Control step threshold.
            make_psd: Whether to project Hessians to PSD cone.
            psd_delta: Minimum eigenvalue for PSD projection.
            alpha_0: Initial line search step size.
            alpha_min: Minimum line search step size.
        """
        super().__init__(
            maxiter_al=maxiter_al,
            maxiter_ilqr=maxiter_ilqr,
            constraints_threshold=constraints_threshold,
            penalty_init=penalty_init,
            penalty_update_rate=penalty_update_rate,
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
        """Internal constrained iLQR solve implementation."""
        # Use the legacy implementation for now
        from trajax import optimizers as legacy_opt

        # Create wrapped functions
        def cost(x, u, t):
            return problem.cost(x, u, t, params)

        def dynamics(x, u, t):
            return problem.dynamics(x, u, t, params)

        # Handle constraints
        if problem.inequality_constraint is not None:
            def ineq_constraint(x, u, t):
                return problem.inequality_constraint(x, u, t, params)
        else:
            ineq_constraint = lambda x, u, t: jnp.empty(0)

        if problem.equality_constraint is not None:
            def eq_constraint(x, u, t):
                return problem.equality_constraint(x, u, t, params)
        else:
            eq_constraint = lambda x, u, t: jnp.empty(0)

        # Call legacy constrained_ilqr
        results = legacy_opt.constrained_ilqr(
            cost=cost,
            dynamics=dynamics,
            x0=x0,
            U=U0,
            equality_constraint=eq_constraint,
            inequality_constraint=ineq_constraint,
            maxiter_al=options.get('maxiter_al', 5),
            maxiter_ilqr=options.get('maxiter_ilqr', 100),
            grad_norm_threshold=options.get('grad_norm_threshold', 1e-4),
            relative_grad_norm_threshold=options.get('relative_grad_norm_threshold', 0.0),
            obj_step_threshold=options.get('obj_step_threshold', 0.0),
            inputs_step_threshold=options.get('inputs_step_threshold', 0.0),
            constraints_threshold=options.get('constraints_threshold', 1e-2),
            penalty_init=options.get('penalty_init', 1.0),
            penalty_update_rate=options.get('penalty_update_rate', 10.0),
            make_psd=options.get('make_psd', True),
            psd_delta=options.get('psd_delta', 0.0),
            alpha_0=options.get('alpha_0', 1.0),
            alpha_min=options.get('alpha_min', 0.00005),
        )

        (X, U, dual_eq, dual_ineq, penalty, eq_cons, ineq_cons,
         max_violation, obj, gradient, iter_ilqr, iter_al) = results

        # Determine status
        if max_violation < options.get('constraints_threshold', 1e-2):
            status = SolverStatus.SOLVED
        elif iter_al >= options.get('maxiter_al', 5):
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
                'iterations_ilqr': int(iter_ilqr),
                'iterations_al': int(iter_al),
                'dual_equality': dual_eq,
                'dual_inequality': dual_ineq,
                'penalty': float(penalty),
                'equality_constraints': eq_cons,
                'inequality_constraints': ineq_cons,
                'max_constraint_violation': float(max_violation),
            },
        )

    def build_solver(
        self,
        problem: TrajectoryProblem,
        options: Optional[Dict[str, Any]] = None,
    ) -> Callable[[Array, Array, PyTree], Trajectory]:
        """Build JIT-compiled constrained iLQR solver."""
        from trajax import optimizers as legacy_opt

        merged_options = {**self.default_options}
        if options:
            merged_options.update(options)

        dynamics_fn = problem.dynamics
        cost_fn = problem.cost
        ineq_fn = problem.inequality_constraint
        eq_fn = problem.equality_constraint

        maxiter_al = merged_options.get('maxiter_al', 5)
        maxiter_ilqr = merged_options.get('maxiter_ilqr', 100)
        constraints_threshold = merged_options.get('constraints_threshold', 1e-2)
        penalty_init = merged_options.get('penalty_init', 1.0)
        penalty_update_rate = merged_options.get('penalty_update_rate', 10.0)
        grad_norm_threshold = merged_options.get('grad_norm_threshold', 1e-4)
        make_psd = merged_options.get('make_psd', True)
        alpha_0 = merged_options.get('alpha_0', 1.0)
        alpha_min = merged_options.get('alpha_min', 0.00005)

        @jax.jit
        def solve_fn(x0: Array, U0: Array, params: PyTree) -> Trajectory:
            def cost(x, u, t):
                return cost_fn(x, u, t, params)

            def dynamics(x, u, t):
                return dynamics_fn(x, u, t, params)

            if ineq_fn is not None:
                def ineq_constraint(x, u, t):
                    return ineq_fn(x, u, t, params)
            else:
                ineq_constraint = lambda x, u, t: jnp.empty(0)

            if eq_fn is not None:
                def eq_constraint(x, u, t):
                    return eq_fn(x, u, t, params)
            else:
                eq_constraint = lambda x, u, t: jnp.empty(0)

            results = legacy_opt.constrained_ilqr(
                cost, dynamics, x0, U0,
                equality_constraint=eq_constraint,
                inequality_constraint=ineq_constraint,
                maxiter_al=maxiter_al,
                maxiter_ilqr=maxiter_ilqr,
                constraints_threshold=constraints_threshold,
                penalty_init=penalty_init,
                penalty_update_rate=penalty_update_rate,
                grad_norm_threshold=grad_norm_threshold,
                make_psd=make_psd,
                alpha_0=alpha_0,
                alpha_min=alpha_min,
            )

            X, U, dual_eq, dual_ineq, penalty, eq_cons, ineq_cons, \
                max_violation, obj, gradient, iter_ilqr, iter_al = results

            status = jnp.where(
                max_violation < constraints_threshold,
                SolverStatus.SOLVED.value,
                SolverStatus.MAX_ITERATIONS.value,
            )

            return Trajectory(
                X=X,
                U=U,
                obj=obj,
                gradient=gradient,
                status=SolverStatus(status),
                info={
                    'iterations_ilqr': iter_ilqr,
                    'iterations_al': iter_al,
                    'max_constraint_violation': max_violation,
                    'penalty': penalty,
                },
            )

        return solve_fn
