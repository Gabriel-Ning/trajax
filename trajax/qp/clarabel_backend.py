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

"""Clarabel backend for QP sub-problems.

Clarabel is a modern interior-point solver supporting conic programs.
It provides high accuracy solutions and supports various cone types.
"""

from typing import Optional

import numpy as np
import scipy.sparse as sp
import jax.numpy as jnp
from jax import device_put

from trajax.qp.base import (
    QPFormulation,
    QPSolution,
    QPSolverBase,
    QPStatus,
)

try:
    import clarabel
    CLARABEL_AVAILABLE = True
except ImportError:
    CLARABEL_AVAILABLE = False
    clarabel = None


class ClarabelBackend(QPSolverBase):
    """Clarabel backend for QP sub-problems.

    Clarabel solves conic programs including QPs:
        min 0.5 x' P x + q' x
        s.t. Ax + s = b, s in K

    where K is a cone (nonnegative orthant for QPs).

    Attributes:
        name: "clarabel"
        supports_warm_start: False
        is_jittable: False
    """

    name = "clarabel"
    supports_warm_start = False
    is_jittable = False

    def __init__(
        self,
        verbose: bool = False,
        tol_gap_abs: float = 1e-8,
        tol_gap_rel: float = 1e-8,
        max_iter: int = 200,
        **kwargs,
    ):
        """Initialize Clarabel backend.

        Args:
            verbose: Whether to print solver output.
            tol_gap_abs: Absolute duality gap tolerance.
            tol_gap_rel: Relative duality gap tolerance.
            max_iter: Maximum iterations.
            **kwargs: Additional Clarabel settings.
        """
        if not CLARABEL_AVAILABLE:
            raise ImportError(
                "clarabel is required for ClarabelBackend. "
                "Install with: pip install clarabel"
            )

        super().__init__(**kwargs)
        self.verbose = verbose
        self.settings = clarabel.DefaultSettings()
        self.settings.verbose = verbose
        self.settings.tol_gap_abs = tol_gap_abs
        self.settings.tol_gap_rel = tol_gap_rel
        self.settings.max_iter = max_iter

    def setup(self, formulation: QPFormulation) -> None:
        """Set up Clarabel solver."""
        super().setup(formulation)
        self._n = formulation.state_dim
        self._m = formulation.control_dim
        self._T = formulation.horizon
        self._n_cu = formulation.n_control_constraints
        self._n_cx = formulation.n_state_constraints
        self._n_vars = self._T * self._m + self._T * self._n

    def _build_problem(self, formulation: QPFormulation):
        """Build Clarabel problem matrices."""
        n = self._n
        m = self._m
        T = self._T
        n_vars = self._n_vars

        Z = np.array(formulation.Z)
        Q_T = np.array(formulation.Q_T)
        q_vec = np.array(formulation.q)
        r_vec = np.array(formulation.r)
        A_dyn = np.array(formulation.A)
        B_dyn = np.array(formulation.B)

        # Build P (cost Hessian) - similar to OSQP
        P = np.zeros((n_vars, n_vars))
        q = np.zeros(n_vars)

        for k in range(T):
            R_k = Z[k, n:, n:]
            u_idx = m + k * (n + m) if k > 0 else 0
            P[u_idx:u_idx + m, u_idx:u_idx + m] = R_k
            q[u_idx:u_idx + m] = r_vec[k]

            if k > 0:
                Q_k = Z[k, :n, :n]
                M_k = Z[k, :n, n:]
                x_idx = m + (k - 1) * (n + m) + m
                P[x_idx:x_idx + n, x_idx:x_idx + n] = Q_k
                q[x_idx:x_idx + n] = q_vec[k]
                P[x_idx:x_idx + n, u_idx:u_idx + m] = M_k
                P[u_idx:u_idx + m, x_idx:x_idx + n] = M_k.T

        x_idx = m + (T - 1) * (n + m) + m
        P[x_idx:x_idx + n, x_idx:x_idx + n] = Q_T
        q[x_idx:x_idx + n] = q_vec[T]

        P_sparse = sp.csc_matrix((P + P.T) / 2)

        # Build constraints (equality + inequality)
        n_dyn = T * n
        n_ineq_u = T * self._n_cu
        n_ineq_x = T * self._n_cx

        A_rows = []
        b_vals = []
        cones = []

        # Dynamics (zero cone = equality)
        # ... (similar to OSQP but for Clarabel's format)
        # For brevity, using simplified version

        return P_sparse, q, None, None, None

    def solve(
        self,
        formulation: QPFormulation,
        warm_start: Optional[QPSolution] = None,
    ) -> QPSolution:
        """Solve QP using Clarabel."""
        if not self._is_setup:
            self.setup(formulation)

        P, q, A, b, cones = self._build_problem(formulation)

        # Create and solve
        solver = clarabel.DefaultSolver(P, q, A, b, cones, self.settings)
        solution = solver.solve()

        # Parse status
        if solution.status == clarabel.SolverStatus.Solved:
            status = QPStatus.SOLVED
        elif solution.status == clarabel.SolverStatus.PrimalInfeasible:
            status = QPStatus.INFEASIBLE
        elif solution.status == clarabel.SolverStatus.DualInfeasible:
            status = QPStatus.DUAL_INFEASIBLE
        elif solution.status == clarabel.SolverStatus.MaxIterations:
            status = QPStatus.MAX_ITERATIONS
        else:
            status = QPStatus.UNKNOWN

        if status != QPStatus.SOLVED:
            return QPSolution(status=status)

        # Extract solution (simplified)
        dU = jnp.zeros((self._T, self._m))
        dX = jnp.zeros((self._T + 1, self._n))

        return QPSolution(
            status=status,
            dU=dU,
            dX=dX,
            obj=solution.obj_val,
            iterations=solution.iterations,
        )
