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

"""ProxQP backend for QP sub-problems.

ProxQP is a proximal augmented Lagrangian solver from the proxsuite library.
It is efficient for sparse QPs and supports warm-starting.
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
    import proxsuite
    PROXQP_AVAILABLE = True
except ImportError:
    PROXQP_AVAILABLE = False
    proxsuite = None


class ProxQPBackend(QPSolverBase):
    """ProxQP backend for QP sub-problems.

    ProxQP solves QPs in the form:
        min 0.5 x' H x + g' x
        s.t. Ax = b
             l <= Cx <= u

    Attributes:
        name: "proxqp"
        supports_warm_start: True
        is_jittable: False
    """

    name = "proxqp"
    supports_warm_start = True
    is_jittable = False

    def __init__(
        self,
        verbose: bool = False,
        eps_abs: float = 1e-6,
        max_iter: int = 1000,
        **kwargs,
    ):
        """Initialize ProxQP backend.

        Args:
            verbose: Whether to print solver output.
            eps_abs: Absolute tolerance.
            max_iter: Maximum iterations.
            **kwargs: Additional ProxQP settings.
        """
        if not PROXQP_AVAILABLE:
            raise ImportError(
                "proxsuite is required for ProxQPBackend. "
                "Install with: pip install proxsuite"
            )

        super().__init__(**kwargs)
        self.verbose = verbose
        self.eps_abs = eps_abs
        self.max_iter = max_iter
        self._qp = None

    def setup(self, formulation: QPFormulation) -> None:
        """Set up ProxQP solver."""
        super().setup(formulation)
        self._n = formulation.state_dim
        self._m = formulation.control_dim
        self._T = formulation.horizon
        self._n_cu = formulation.n_control_constraints
        self._n_cx = formulation.n_state_constraints
        self._n_vars = self._T * self._m + self._T * self._n

        # Build initial problem
        H, g, A_eq, b_eq, C, l, u = self._build_problem(formulation)

        n_eq = A_eq.shape[0] if A_eq is not None else 0
        n_ineq = C.shape[0] if C is not None else 0

        self._qp = proxsuite.proxqp.sparse.QP(
            self._n_vars, n_eq, n_ineq
        )
        self._qp.settings.eps_abs = self.eps_abs
        self._qp.settings.max_iter = self.max_iter
        self._qp.settings.verbose = self.verbose

        self._qp.init(H, g, A_eq, b_eq, C, l, u)

    def _build_problem(self, formulation: QPFormulation):
        """Build ProxQP problem matrices."""
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

        # Build H (cost Hessian)
        H = np.zeros((n_vars, n_vars))
        g = np.zeros(n_vars)

        for k in range(T):
            R_k = Z[k, n:, n:]
            u_idx = m + k * (n + m) if k > 0 else 0
            H[u_idx:u_idx + m, u_idx:u_idx + m] = R_k
            g[u_idx:u_idx + m] = r_vec[k]

            if k > 0:
                Q_k = Z[k, :n, :n]
                M_k = Z[k, :n, n:]
                x_idx = m + (k - 1) * (n + m) + m
                H[x_idx:x_idx + n, x_idx:x_idx + n] = Q_k
                g[x_idx:x_idx + n] = q_vec[k]
                H[x_idx:x_idx + n, u_idx:u_idx + m] = M_k
                H[u_idx:u_idx + m, x_idx:x_idx + n] = M_k.T

        x_idx = m + (T - 1) * (n + m) + m
        H[x_idx:x_idx + n, x_idx:x_idx + n] = Q_T
        g[x_idx:x_idx + n] = q_vec[T]

        H_sparse = sp.csc_matrix((H + H.T) / 2)

        # Build equality constraints (dynamics)
        n_eq = T * n
        A_eq = np.zeros((n_eq, n_vars))
        b_eq = np.zeros(n_eq)

        row = 0
        for k in range(T):
            if k == 0:
                u_idx = 0
                x_idx = None
            else:
                u_idx = m + (k - 1) * (n + m) + m + n
                x_idx = m + (k - 1) * (n + m) + m

            x_next_idx = m + k * (n + m) + m

            for i in range(n):
                A_eq[row + i, x_next_idx + i] = 1.0
                for j in range(m):
                    A_eq[row + i, u_idx + j] = -B_dyn[k, i, j]
                if x_idx is not None:
                    for j in range(n):
                        A_eq[row + i, x_idx + j] = -A_dyn[k, i, j]
            row += n

        A_eq_sparse = sp.csc_matrix(A_eq)

        # Build inequality constraints
        n_ineq_u = T * self._n_cu
        n_ineq_x = T * self._n_cx
        n_ineq = n_ineq_u + n_ineq_x

        if n_ineq > 0:
            C = np.zeros((n_ineq, n_vars))
            l = np.full(n_ineq, -np.inf)
            u = np.zeros(n_ineq)
            # ... fill in constraint matrices
            C_sparse = sp.csc_matrix(C)
        else:
            C_sparse = None
            l = None
            u = None

        return H_sparse, g, A_eq_sparse, b_eq, C_sparse, l, u

    def solve(
        self,
        formulation: QPFormulation,
        warm_start: Optional[QPSolution] = None,
    ) -> QPSolution:
        """Solve QP using ProxQP."""
        if not self._is_setup:
            self.setup(formulation)

        H, g, A_eq, b_eq, C, l, u = self._build_problem(formulation)
        self._qp.update(H, g, A_eq, b_eq, C, l, u)

        if warm_start is not None and warm_start.dU is not None:
            x_warm = self._flatten_solution(warm_start.dU, warm_start.dX)
            self._qp.warm_start(x_warm)

        self._qp.solve()

        result = self._qp.results

        if result.info.status == proxsuite.proxqp.QPSolverOutput.PROXQP_SOLVED:
            status = QPStatus.SOLVED
        elif result.info.status == proxsuite.proxqp.QPSolverOutput.PROXQP_MAX_ITER_REACHED:
            status = QPStatus.MAX_ITERATIONS
        else:
            status = QPStatus.UNKNOWN

        if status != QPStatus.SOLVED:
            return QPSolution(status=status, iterations=result.info.iter)

        dU, dX = self._unflatten_solution(result.x)

        return QPSolution(
            status=status,
            dU=device_put(dU),
            dX=device_put(dX),
            obj=result.info.objValue,
            iterations=result.info.iter,
        )

    def _flatten_solution(self, dU, dX):
        """Flatten (dU, dX) to variable vector."""
        n, m, T = self._n, self._m, self._T
        x = np.zeros(self._n_vars)
        x[:m] = np.array(dU[0])
        for k in range(1, T):
            idx = m + (k - 1) * (n + m)
            x[idx + m:idx + m + n] = np.array(dX[k])
            x[idx + m + n:idx + m + n + m] = np.array(dU[k])
        idx = m + (T - 1) * (n + m) + m
        x[idx:idx + n] = np.array(dX[T])
        return x

    def _unflatten_solution(self, x):
        """Unflatten solution to (dU, dX)."""
        n, m, T = self._n, self._m, self._T
        dU = np.zeros((T, m))
        dX = np.zeros((T + 1, n))
        dX[0] = 0.0
        dU[0] = x[:m]
        for k in range(1, T):
            idx = m + (k - 1) * (n + m) + m
            dX[k] = x[idx:idx + n]
            dU[k] = x[idx + n:idx + n + m]
        idx = m + (T - 1) * (n + m) + m
        dX[T] = x[idx:idx + n]
        return dU, dX
