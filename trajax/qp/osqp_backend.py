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

"""OSQP backend for QP sub-problems.

OSQP is a fast, robust QP solver widely used for MPC applications.
This backend provides efficient sparse matrix construction for the
trajectory QP structure.
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
    import osqp
    OSQP_AVAILABLE = True
except ImportError:
    OSQP_AVAILABLE = False
    osqp = None


class OSQPBackend(QPSolverBase):
    """OSQP backend for QP sub-problems.

    OSQP solves QPs in the form:
        min 0.5 x' P x + q' x
        s.t. l <= A x <= u

    This backend converts the trajectory QP formulation to OSQP's form
    using sparse matrices for efficiency.

    Attributes:
        name: "osqp"
        supports_warm_start: True
        is_jittable: False (uses external solver)
    """

    name = "osqp"
    supports_warm_start = True
    is_jittable = False

    def __init__(
        self,
        verbose: bool = False,
        eps_abs: float = 1e-4,
        eps_rel: float = 1e-4,
        max_iter: int = 4000,
        polish: bool = True,
        **kwargs,
    ):
        """Initialize OSQP backend.

        Args:
            verbose: Whether to print solver output.
            eps_abs: Absolute tolerance.
            eps_rel: Relative tolerance.
            max_iter: Maximum iterations.
            polish: Whether to polish the solution.
            **kwargs: Additional OSQP settings.
        """
        if not OSQP_AVAILABLE:
            raise ImportError(
                "osqp is required for OSQPBackend. "
                "Install with: pip install osqp"
            )

        super().__init__(**kwargs)
        self.verbose = verbose
        self.settings = {
            'verbose': verbose,
            'eps_abs': eps_abs,
            'eps_rel': eps_rel,
            'max_iter': max_iter,
            'polish': polish,
            **kwargs,
        }
        self._solver = None

    def setup(self, formulation: QPFormulation) -> None:
        """Set up OSQP solver with problem structure."""
        super().setup(formulation)

        n = formulation.state_dim
        m = formulation.control_dim
        T = formulation.horizon
        n_cu = formulation.n_control_constraints
        n_cx = formulation.n_state_constraints

        # Decision variables: [dU_0, dX_1, dU_1, dX_2, ..., dU_{T-1}, dX_T]
        # Total: T*m + T*n variables
        n_vars = T * m + T * n

        # Store dimensions for later
        self._n = n
        self._m = m
        self._T = T
        self._n_cu = n_cu
        self._n_cx = n_cx
        self._n_vars = n_vars

        # Build sparse constraint matrix structure
        # Constraints:
        # 1. Dynamics: dX_{k+1} = A_k dX_k + B_k dU_k (T*n equality)
        # 2. Control constraints: Cu + Ju @ dU >= 0 (T*n_cu inequality)
        # 3. State constraints: Cx + Jx @ dX >= 0 ((T)*n_cx inequality)

        n_dyn = T * n
        n_cons_u = T * n_cu
        n_cons_x = T * n_cx  # k=1,...,T
        n_constraints = n_dyn + n_cons_u + n_cons_x

        self._n_constraints = n_constraints

        # Create solver instance
        self._solver = osqp.OSQP()

        # Build initial problem with zeros (will be updated in solve)
        P, q = self._build_cost_matrices(formulation)
        A, l, u = self._build_constraint_matrices(formulation)

        self._solver.setup(P, q, A, l, u, **self.settings)

    def _get_var_indices(self, k: int, var_type: str):
        """Get variable indices for timestep k."""
        # Layout: [dU_0, dX_1, dU_1, dX_2, ..., dU_{T-1}, dX_T]
        n, m = self._n, self._m
        if var_type == 'u':
            if k == 0:
                return slice(0, m)
            else:
                return slice(m + (k - 1) * (n + m), m + (k - 1) * (n + m) + m)
        else:  # 'x'
            if k == 0:
                return None  # dX_0 = 0 (not a variable)
            else:
                return slice(m + (k - 1) * (n + m) + m - n, m + (k - 1) * (n + m) + m)

    def _build_cost_matrices(self, formulation: QPFormulation):
        """Build P and q matrices for OSQP."""
        n = self._n
        m = self._m
        T = self._T
        n_vars = self._n_vars

        Z = np.array(formulation.Z)
        Q_T = np.array(formulation.Q_T)
        q_vec = np.array(formulation.q)
        r_vec = np.array(formulation.r)

        # Build dense P first, then convert to sparse
        P = np.zeros((n_vars, n_vars))
        q = np.zeros(n_vars)

        for k in range(T):
            # Extract blocks from Z[k]
            Q_k = Z[k, :n, :n]
            R_k = Z[k, n:, n:]
            M_k = Z[k, :n, n:]

            # Control cost
            u_idx = m + k * (n + m) if k > 0 else 0
            u_end = u_idx + m
            P[u_idx:u_end, u_idx:u_end] = R_k
            q[u_idx:u_end] = r_vec[k]

            # State cost (k > 0)
            if k > 0:
                x_idx = m + (k - 1) * (n + m) + m
                x_end = x_idx + n
                P[x_idx:x_end, x_idx:x_end] = Q_k
                q[x_idx:x_end] = q_vec[k]

                # Cross-term
                P[x_idx:x_end, u_idx:u_end] = M_k
                P[u_idx:u_end, x_idx:x_end] = M_k.T

        # Terminal cost
        x_idx = m + (T - 1) * (n + m) + m
        x_end = x_idx + n
        P[x_idx:x_end, x_idx:x_end] = Q_T
        q[x_idx:x_end] = q_vec[T]

        # Convert to sparse CSC
        P_sparse = sp.csc_matrix((P + P.T) / 2)  # Ensure symmetric

        return P_sparse, q

    def _build_constraint_matrices(self, formulation: QPFormulation):
        """Build A, l, u matrices for OSQP constraints."""
        n = self._n
        m = self._m
        T = self._T
        n_cu = self._n_cu
        n_cx = self._n_cx
        n_vars = self._n_vars

        A_dyn = np.array(formulation.A)
        B_dyn = np.array(formulation.B)

        n_dyn = T * n
        n_cons_u = T * n_cu
        n_cons_x = T * n_cx
        n_constraints = n_dyn + n_cons_u + n_cons_x

        # Build constraint matrix
        A = np.zeros((n_constraints, n_vars))
        l = np.zeros(n_constraints)
        u = np.zeros(n_constraints)

        row = 0

        # Dynamics constraints: dX_{k+1} = A_k dX_k + B_k dU_k
        for k in range(T):
            # dX_{k+1} - A_k dX_k - B_k dU_k = 0

            # dX_{k+1} coefficient (+I)
            x_next_idx = m + k * (n + m) + m if k < T - 1 else m + (T - 1) * (n + m) + m
            # Actually, let me recalculate indices more carefully
            # dU_k index
            if k == 0:
                u_idx = 0
            else:
                u_idx = m + (k - 1) * (n + m) + m + n

            # dX_k index (k > 0)
            if k > 0:
                x_idx = m + (k - 1) * (n + m) + m
            else:
                x_idx = None

            # dX_{k+1} index
            x_next_idx = m + k * (n + m) + m

            for i in range(n):
                # +1 for dX_{k+1}
                A[row + i, x_next_idx + i] = 1.0

                # -B_k for dU_k
                for j in range(m):
                    A[row + i, u_idx + j] = -B_dyn[k, i, j]

                # -A_k for dX_k (k > 0)
                if x_idx is not None:
                    for j in range(n):
                        A[row + i, x_idx + j] = -A_dyn[k, i, j]

            # Equality: l = u = 0
            l[row:row + n] = 0.0
            u[row:row + n] = 0.0
            row += n

        # Control constraints: Cu + Ju @ dU >= 0 => Ju @ dU >= -Cu
        if n_cu > 0:
            Cu = np.array(formulation.Cu)
            Ju = np.array(formulation.Ju)

            for k in range(T):
                if k == 0:
                    u_idx = 0
                else:
                    u_idx = m + (k - 1) * (n + m) + m + n

                for i in range(n_cu):
                    for j in range(m):
                        A[row + i, u_idx + j] = Ju[k, i, j]

                l[row:row + n_cu] = -Cu[k]
                u[row:row + n_cu] = np.inf
                row += n_cu

        # State constraints: Cx + Jx @ dX >= 0 => Jx @ dX >= -Cx
        if n_cx > 0:
            Cx = np.array(formulation.Cx)
            Jx = np.array(formulation.Jx)

            for k in range(1, T + 1):  # k=1,...,T
                x_idx = m + (k - 1) * (n + m) + m

                for i in range(n_cx):
                    for j in range(n):
                        A[row + i, x_idx + j] = Jx[k, i, j]

                l[row:row + n_cx] = -Cx[k]
                u[row:row + n_cx] = np.inf
                row += n_cx

        A_sparse = sp.csc_matrix(A)
        return A_sparse, l, u

    def solve(
        self,
        formulation: QPFormulation,
        warm_start: Optional[QPSolution] = None,
    ) -> QPSolution:
        """Solve QP using OSQP."""
        if not self._is_setup:
            self.setup(formulation)

        # Update problem data
        P, q = self._build_cost_matrices(formulation)
        A, l, u = self._build_constraint_matrices(formulation)

        self._solver.update(Px=P.data, q=q, Ax=A.data, l=l, u=u)

        # Warm start if available
        if warm_start is not None and warm_start.dU is not None:
            x_warm = self._flatten_solution(warm_start.dU, warm_start.dX)
            self._solver.warm_start(x=x_warm)

        # Solve
        result = self._solver.solve()

        # Parse status
        if result.info.status == 'solved':
            status = QPStatus.SOLVED
        elif result.info.status == 'solved_inaccurate':
            status = QPStatus.SOLVED_INACCURATE
        elif result.info.status == 'primal_infeasible':
            status = QPStatus.INFEASIBLE
        elif result.info.status == 'dual_infeasible':
            status = QPStatus.DUAL_INFEASIBLE
        elif result.info.status == 'max_iter_reached':
            status = QPStatus.MAX_ITERATIONS
        else:
            status = QPStatus.UNKNOWN

        if status not in (QPStatus.SOLVED, QPStatus.SOLVED_INACCURATE):
            return QPSolution(
                status=status,
                iterations=result.info.iter,
                info={'osqp_status': result.info.status},
            )

        # Extract solution
        dU, dX = self._unflatten_solution(result.x)

        return QPSolution(
            status=status,
            dU=device_put(dU),
            dX=device_put(dX),
            obj=result.info.obj_val,
            iterations=result.info.iter,
            info={'osqp_status': result.info.status},
        )

    def _flatten_solution(self, dU, dX):
        """Flatten (dU, dX) to OSQP variable vector."""
        n, m, T = self._n, self._m, self._T
        x = np.zeros(self._n_vars)

        # dU_0
        x[:m] = np.array(dU[0])

        # [dX_k, dU_k] for k=1,...,T-1, then dX_T
        for k in range(1, T):
            idx = m + (k - 1) * (n + m)
            x[idx + m:idx + m + n] = np.array(dX[k])
            x[idx + m + n:idx + m + n + m] = np.array(dU[k])

        # dX_T
        idx = m + (T - 1) * (n + m) + m
        x[idx:idx + n] = np.array(dX[T])

        return x

    def _unflatten_solution(self, x):
        """Unflatten OSQP solution to (dU, dX)."""
        n, m, T = self._n, self._m, self._T

        dU = np.zeros((T, m))
        dX = np.zeros((T + 1, n))

        # dX_0 = 0 (always)
        dX[0] = 0.0

        # dU_0
        dU[0] = x[:m]

        # [dX_k, dU_k] for k=1,...,T-1
        for k in range(1, T):
            idx = m + (k - 1) * (n + m) + m
            dX[k] = x[idx:idx + n]
            dU[k] = x[idx + n:idx + n + m]

        # dX_T
        idx = m + (T - 1) * (n + m) + m
        dX[T] = x[idx:idx + n]

        return dU, dX
