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

"""CVXPY backend for QP sub-problems.

This backend uses CVXPY with ECOS or CVXOPT solvers for the QP sub-problem.
It provides a reliable fallback when JAX-native solvers have issues.
"""

from typing import Optional

import numpy as np
import jax.numpy as jnp
from jax import device_put

from trajax.qp.base import (
    QPFormulation,
    QPSolution,
    QPSolverBase,
    QPStatus,
)

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False
    cp = None


CVXPY_SOLVERS = {
    "ecos": cp.ECOS if cp else None,
    "cvxopt": cp.CVXOPT if cp else None,
}


class CVXPYBackend(QPSolverBase):
    """CVXPY backend for QP sub-problems.

    Uses CVXPY's parametric QP interface with ECOS or CVXOPT solvers.

    Attributes:
        name: "cvxpy"
        supports_warm_start: False (CVXPY doesn't expose warm-starting)
        is_jittable: False (uses external solver)
    """

    name = "cvxpy"
    supports_warm_start = False
    is_jittable = False

    def __init__(self, solver: str = "ecos", verbose: bool = False, **kwargs):
        """Initialize CVXPY backend.

        Args:
            solver: CVXPY solver to use ("ecos" or "cvxopt").
            verbose: Whether to print solver output.
            **kwargs: Additional CVXPY solver options.
        """
        if not CVXPY_AVAILABLE:
            raise ImportError(
                "cvxpy is required for CVXPYBackend. "
                "Install with: pip install cvxpy"
            )

        super().__init__(**kwargs)
        self.solver_name = solver.lower()
        if self.solver_name not in CVXPY_SOLVERS:
            raise ValueError(
                f"Unknown solver: {solver}. "
                f"Available: {list(CVXPY_SOLVERS.keys())}"
            )
        self._solver = CVXPY_SOLVERS[self.solver_name]
        self.verbose = verbose

        # Problem components (set during setup)
        self._prob = None
        self._dX = None
        self._dU = None
        self._params = {}
        self._constraints = {}

    def setup(self, formulation: QPFormulation) -> None:
        """Set up the parametric CVXPY problem."""
        super().setup(formulation)

        n = formulation.state_dim
        m = formulation.control_dim
        T = formulation.horizon
        n_cu = formulation.n_control_constraints
        n_cx = formulation.n_state_constraints

        # Initialize parameters
        self._params['q'] = cp.Parameter((T + 1, n))
        self._params['r'] = cp.Parameter((T, m))
        self._params['Cu'] = cp.Parameter((T, n_cu)) if n_cu > 0 else None
        self._params['Cx'] = cp.Parameter((T + 1, n_cx)) if n_cx > 0 else None
        self._params['Q_T'] = cp.Parameter((n, n), PSD=True)

        Z, A, B, Ju, Jx = {}, {}, {}, {}, {}
        for j in range(T):
            Z[j] = cp.Parameter((n + m, n + m), PSD=True)
            A[j] = cp.Parameter((n, n))
            B[j] = cp.Parameter((n, m))
            if n_cu > 0:
                Ju[j] = cp.Parameter((n_cu, m))
            if n_cx > 0:
                Jx[j] = cp.Parameter((n_cx, n))
        if n_cx > 0:
            Jx[T] = cp.Parameter((n_cx, n))

        self._params['Z'] = Z
        self._params['A'] = A
        self._params['B'] = B
        self._params['Ju'] = Ju
        self._params['Jx'] = Jx

        # Variables
        self._dX = cp.Variable((T, n))  # k=1,...,T
        self._dU = cp.Variable((T, m))  # k=0,...,T-1

        # Build cost expression
        cost = 0.0
        self._constraints['dyn'] = []
        self._constraints['ineq_u'] = []
        self._constraints['ineq_x'] = []

        for j in range(T):
            if j == 0:
                dx = np.zeros((n,))
            else:
                dx = self._dX[j - 1]

            dz = cp.hstack([dx, self._dU[j]])
            cost += (
                0.5 * cp.quad_form(dz, Z[j])
                + self._params['q'][j] @ dx
                + self._params['r'][j] @ self._dU[j]
            )

            # Control constraints
            if n_cu > 0:
                self._constraints['ineq_u'].append(
                    self._params['Cu'][j] + Ju[j] @ self._dU[j] >= 0.0
                )

            # State constraints (not at k=0)
            if n_cx > 0 and j > 0:
                self._constraints['ineq_x'].append(
                    self._params['Cx'][j] + Jx[j] @ dx >= 0.0
                )

            # Dynamics constraints
            self._constraints['dyn'].append(
                self._dX[j] == A[j] @ dx + B[j] @ self._dU[j]
            )

        # Terminal cost
        cost += (
            0.5 * cp.quad_form(self._dX[-1], self._params['Q_T'])
            + self._params['q'][-1] @ self._dX[-1]
        )

        # Terminal state constraint
        if n_cx > 0:
            self._constraints['ineq_x'].append(
                self._params['Cx'][-1] + Jx[T] @ self._dX[-1] >= 0.0
            )

        # Build problem
        all_constraints = (
            self._constraints['dyn']
            + self._constraints['ineq_u']
            + self._constraints['ineq_x']
        )
        self._prob = cp.Problem(cp.Minimize(cost), all_constraints)

    def _update_params(self, formulation: QPFormulation) -> None:
        """Update CVXPY parameters from formulation."""
        Z = np.array(formulation.Z)
        Q = np.array(formulation.Q) if formulation.Q is not None else None
        q = np.array(formulation.q)
        r = np.array(formulation.r)
        A = np.array(formulation.A)
        B = np.array(formulation.B)

        T = formulation.horizon

        self._params['q'].value = q
        self._params['r'].value = r
        self._params['Q_T'].value = (
            Q[-1] if Q is not None else formulation.Q_T
        )

        for j in range(T):
            self._params['Z'][j].value = Z[j]
            self._params['A'][j].value = A[j]
            self._params['B'][j].value = B[j]

        if formulation.n_control_constraints > 0:
            Cu = np.array(formulation.Cu)
            Ju = np.array(formulation.Ju)
            self._params['Cu'].value = Cu
            for j in range(T):
                self._params['Ju'][j].value = Ju[j]

        if formulation.n_state_constraints > 0:
            Cx = np.array(formulation.Cx)
            Jx = np.array(formulation.Jx)
            self._params['Cx'].value = Cx
            for j in range(T + 1):
                self._params['Jx'][j].value = Jx[j]

    def solve(
        self,
        formulation: QPFormulation,
        warm_start: Optional[QPSolution] = None,
    ) -> QPSolution:
        """Solve QP using CVXPY."""
        if not self._is_setup:
            self.setup(formulation)

        self._update_params(formulation)

        # Solve
        try:
            self._prob.solve(self._solver, verbose=self.verbose, **self.options)
        except Exception as e:
            return QPSolution(
                status=QPStatus.NUMERICAL_ERROR,
                info={'error': str(e)},
            )

        # Parse results
        if self._prob.status in ("infeasible", "infeasible_inaccurate"):
            return QPSolution(status=QPStatus.INFEASIBLE)

        if self._prob.status in ("unbounded", "unbounded_inaccurate"):
            return QPSolution(status=QPStatus.DUAL_INFEASIBLE)

        if self._prob.status not in ("optimal", "optimal_inaccurate"):
            return QPSolution(
                status=QPStatus.UNKNOWN,
                info={'cvxpy_status': self._prob.status},
            )

        n = formulation.state_dim
        T = formulation.horizon

        # Extract solution
        dU = device_put(self._dU.value)
        dX = jnp.vstack([jnp.zeros((n,)), device_put(self._dX.value)])

        # Extract duals
        Yu = None
        Yx = None

        if formulation.n_control_constraints > 0:
            Yu = jnp.vstack([
                device_put(c.dual_value)
                for c in self._constraints['ineq_u']
            ])

        if formulation.n_state_constraints > 0:
            Yx_list = [
                device_put(c.dual_value)
                for c in self._constraints['ineq_x']
            ]
            # Pad with zeros for k=0
            Yx = jnp.vstack([
                jnp.zeros_like(Yx_list[0]),
                jnp.vstack(Yx_list),
            ])

        status = (
            QPStatus.SOLVED if self._prob.status == "optimal"
            else QPStatus.SOLVED_INACCURATE
        )

        return QPSolution(
            status=status,
            dU=dU,
            dX=dX,
            Yu=Yu,
            Yx=Yx,
            obj=self._prob.value,
            info={'cvxpy_status': self._prob.status},
        )
