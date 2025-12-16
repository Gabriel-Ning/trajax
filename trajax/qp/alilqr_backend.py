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

"""Augmented Lagrangian iLQR backend for QP sub-problems.

This backend uses the constrained_ilqr from trajax to solve QP sub-problems.
It is fully JIT-compilable and doesn't require external dependencies.
"""

import functools
from typing import Optional

import jax
import jax.numpy as jnp
from jax import Array

from trajax.qp.base import (
    QPFormulation,
    QPSolution,
    QPSolverBase,
    QPStatus,
)


class ALiLQRBackend(QPSolverBase):
    """Augmented Lagrangian iLQR backend for QP sub-problems.

    Uses trajax's constrained_ilqr to solve the QP sub-problem as a
    linear-quadratic optimal control problem with constraints.

    This backend is:
    - Fully JIT-compilable (JAX native)
    - Always available (no external dependencies)
    - Good for problems where constraints are soft or mild

    Attributes:
        name: "alilqr"
        supports_warm_start: True
        is_jittable: True
    """

    name = "alilqr"
    supports_warm_start = True
    is_jittable = True

    def __init__(
        self,
        constraints_threshold: float = 1e-3,
        penalty_update_rate: float = 5.0,
        maxiter_al: int = 10,
        maxiter_ilqr: int = 100,
        grad_norm_threshold: float = 1e-4,
        make_psd: bool = True,
        **kwargs,
    ):
        """Initialize ALiLQR backend.

        Args:
            constraints_threshold: Tolerance for constraint violation.
            penalty_update_rate: Multiplier for penalty updates.
            maxiter_al: Maximum augmented Lagrangian iterations.
            maxiter_ilqr: Maximum iLQR iterations per AL iteration.
            grad_norm_threshold: Gradient norm threshold for iLQR.
            make_psd: Whether to project Hessians to PSD cone.
            **kwargs: Additional options.
        """
        super().__init__(**kwargs)
        self.constraints_threshold = constraints_threshold
        self.penalty_update_rate = penalty_update_rate
        self.maxiter_al = maxiter_al
        self.maxiter_ilqr = maxiter_ilqr
        self.grad_norm_threshold = grad_norm_threshold
        self.make_psd = make_psd

    def setup(self, formulation: QPFormulation) -> None:
        """Set up solver for the given problem structure."""
        super().setup(formulation)
        self._n = formulation.state_dim
        self._m = formulation.control_dim
        self._T = formulation.horizon
        self._n_cu = formulation.n_control_constraints
        self._n_cx = formulation.n_state_constraints

    def _qp_cost(
        self,
        dx: Array,
        du: Array,
        k: int,
        Z: Array,
        Q_T: Array,
        q: Array,
        r: Array,
    ) -> float:
        """QP cost function for one timestep."""
        dz = jnp.concatenate((dx, du))
        stage_cost = (
            0.5 * jnp.vdot(dz, Z[k] @ dz)
            + jnp.vdot(q[k], dx)
            + jnp.vdot(r[k], du)
        )
        term_cost = 0.5 * jnp.vdot(dx, Q_T @ dx) + jnp.vdot(q[-1], dx)
        return jnp.where(k == self._T, term_cost, stage_cost)

    def _qp_cons(
        self,
        dx: Array,
        du: Array,
        k: int,
        Cu: Array,
        Ju: Array,
        Cx: Array,
        Jx: Array,
    ) -> Array:
        """QP constraint function (inequality: return <= 0)."""
        # State constraints (not applied at k=0)
        state_cons = jnp.where(
            k == 0,
            -jnp.ones(self._n_cx),
            -(Cx[k] + Jx[k] @ dx),
        )
        # Control constraints (not applied at k=T)
        control_cons = jnp.where(
            k == self._T,
            -jnp.ones(self._n_cu),
            -(Cu[k] + Ju[k] @ du),
        )
        return jnp.concatenate((state_cons, control_cons))

    def _lin_dyn(
        self,
        dx: Array,
        du: Array,
        k: int,
        A: Array,
        B: Array,
    ) -> Array:
        """Linear dynamics for QP sub-problem."""
        return A[k] @ dx + B[k] @ du

    @functools.partial(jax.jit, static_argnums=(0,))
    def solve(
        self,
        formulation: QPFormulation,
        warm_start: Optional[QPSolution] = None,
    ) -> QPSolution:
        """Solve QP using augmented Lagrangian iLQR."""
        # Import here to avoid circular imports
        from trajax import optimizers

        # Extract parameters
        Z = formulation.Z
        Q_T = formulation.Q_T
        q = formulation.q
        r = formulation.r
        A = formulation.A
        B = formulation.B

        # Pad arrays for indexing consistency
        Z_pad = jnp.concatenate([Z, jnp.zeros((1,) + Z.shape[1:])], axis=0)
        r_pad = jnp.concatenate([r, jnp.zeros((1, self._m))], axis=0)

        # Define cost with captured parameters
        def cost(dx, du, k):
            return self._qp_cost(dx, du, k, Z_pad, Q_T, q, r_pad)

        # Define dynamics
        def dynamics(dx, du, k):
            return self._lin_dyn(dx, du, k, A, B)

        # Initial state perturbation is zero
        dx_0 = jnp.zeros(self._n)

        # Initial control perturbation (warm start or zeros)
        if warm_start is not None and warm_start.dU is not None:
            dU_0 = warm_start.dU
        else:
            dU_0 = jnp.zeros((self._T, self._m))

        # Check if we have constraints
        if formulation.has_constraints:
            Cu = formulation.Cu
            Ju = formulation.Ju
            Cx = formulation.Cx
            Jx = formulation.Jx

            # Pad constraint arrays
            Cu_pad = jnp.concatenate(
                [Cu, jnp.zeros((1, self._n_cu))], axis=0
            )
            Ju_pad = jnp.concatenate(
                [Ju, jnp.zeros((1, self._n_cu, self._m))], axis=0
            )

            def ineq_cons(dx, du, k):
                return self._qp_cons(dx, du, k, Cu_pad, Ju_pad, Cx, Jx)

            results = optimizers.constrained_ilqr(
                cost,
                dynamics,
                dx_0,
                dU_0,
                inequality_constraint=ineq_cons,
                constraints_threshold=self.constraints_threshold,
                penalty_update_rate=self.penalty_update_rate,
                maxiter_al=self.maxiter_al,
                maxiter_ilqr=self.maxiter_ilqr,
                grad_norm_threshold=self.grad_norm_threshold,
                make_psd=self.make_psd,
            )

            dX, dU, _, Y_qp, penalty, eq_cons, ineq_cons_val, max_viol, \
                obj, grad, iter_ilqr, iter_al = results

            # Split duals
            Yx, Yu = jnp.split(Y_qp, [self._n_cx], axis=1)
            Yu = Yu[:-1]  # Remove terminal

            status = jnp.where(
                max_viol < self.constraints_threshold,
                QPStatus.SOLVED.value,
                QPStatus.MAX_ITERATIONS.value,
            )

            return QPSolution(
                status=QPStatus(status),
                dU=dU,
                dX=dX,
                Yu=Yu,
                Yx=Yx,
                obj=obj,
                iterations=iter_ilqr,
                info={'penalty': penalty, 'max_violation': max_viol},
            )
        else:
            # Unconstrained case - use regular iLQR
            results = optimizers.ilqr(
                cost,
                dynamics,
                dx_0,
                dU_0,
                maxiter=self.maxiter_ilqr,
                grad_norm_threshold=self.grad_norm_threshold,
                make_psd=self.make_psd,
            )

            dX, dU, obj, grad, adjoints, lqr, iterations = results

            return QPSolution(
                status=QPStatus.SOLVED,
                dU=dU,
                dX=dX,
                Yu=None,
                Yx=None,
                obj=obj,
                iterations=iterations,
                info={'gradient': grad},
            )
