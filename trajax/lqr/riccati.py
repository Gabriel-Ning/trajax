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

"""Riccati equation solvers for LQR problems.

Provides utilities for solving discrete-time and continuous-time
algebraic Riccati equations (DARE/CARE).
"""

from typing import Optional

import jax.numpy as jnp
import jax.scipy as jsp
from jax import Array, jit


@jit
def dare_step(
    P: Array,
    Q: Array,
    R: Array,
    A: Array,
    B: Array,
    M: Optional[Array] = None,
    reg: float = 1e-8,
) -> tuple[Array, Array, Array]:
    """Single backward Riccati recursion step (DARE).

    Computes one step of the discrete-time Riccati equation:
        P_prev = Q + A^T P A - (A^T P B + M)^T (R + B^T P B)^{-1}
                 (A^T P B + M)

    Also returns the optimal gain K and value improvement.

    Args:
        P: Current value matrix (n, n).
        Q: State cost matrix (n, n).
        R: Control cost matrix (m, m).
        A: Dynamics matrix (n, n).
        B: Control matrix (n, m).
        M: Cross term matrix (n, m), defaults to zeros.
        reg: Regularization for positive definiteness.

    Returns:
        Tuple of:
            - P_prev: Updated value matrix (n, n)
            - K: Optimal gain (m, n)
            - dV: Value improvement (scalar)
    """
    symmetrize = lambda x: (x + x.T) / 2

    if M is None:
        M = jnp.zeros((A.shape[0], B.shape[1]))

    # Compute intermediate products
    AtP = A.T @ P
    AtPA = symmetrize(AtP @ A)
    BtP = B.T @ P
    BtPB = symmetrize(BtP @ B)

    # Gain computation
    H = BtP @ A + M.T  # (m, n)
    G = R + BtPB  # (m, m)

    # Solve for gain with regularization
    K = -jsp.linalg.solve(G + reg * jnp.eye(G.shape[0]), H, assume_a='pos')

    # Update value matrix
    P_prev = Q + AtPA + K.T @ H + H.T @ K + K.T @ G @ K

    # Value improvement (for convergence checking)
    dV = jnp.trace(P - P_prev)

    return P_prev, K, dV


@jit
def dare_step_affine(
    P: Array,
    p: Array,
    Q: Array,
    q: Array,
    R: Array,
    r: Array,
    A: Array,
    B: Array,
    c: Array,
    M: Optional[Array] = None,
    reg: float = 1e-8,
) -> tuple[Array, Array, Array, Array]:
    """Single affine Riccati recursion step.

    Extends dare_step to handle affine costs and dynamics with
    linear terms (q, r, c).

    Value function: V(x) = x^T P x + 2 p^T x + v0
    Optimal control: u = K x + k

    Args:
        P: Current value matrix (n, n).
        p: Current value vector (n,).
        Q: State cost matrix (n, n).
        q: State cost vector (n,).
        R: Control cost matrix (m, m).
        r: Control cost vector (m,).
        A: Dynamics matrix (n, n).
        B: Control matrix (n, m).
        c: Dynamics offset (n,).
        M: Cross term matrix (n, m).
        reg: Regularization parameter.

    Returns:
        Tuple of:
            - P_prev: Updated value matrix (n, n)
            - p_prev: Updated value vector (n,)
            - K: Optimal gain matrix (m, n)
            - k: Optimal affine term (m,)
    """
    symmetrize = lambda x: (x + x.T) / 2

    if M is None:
        M = jnp.zeros((A.shape[0], B.shape[1]))

    # Compute intermediate products
    AtP = A.T @ P
    AtPA = symmetrize(AtP @ A)
    BtP = B.T @ P
    BtPB = symmetrize(BtP @ B)

    # Gain computation (matrix and affine)
    H = BtP @ A + M.T  # (m, n)
    h = B.T @ p + BtP @ c + r  # (m,)
    G = R + BtPB  # (m, m)

    # Solve for gain [K, k]
    K_k = jsp.linalg.solve(
        G + reg * jnp.eye(G.shape[0]),
        -jnp.column_stack([H, h[:, None]]),
        assume_a='pos'
    )
    K = K_k[:, :-1]
    k = K_k[:, -1]

    # Update value function
    H_GK = H + G @ K
    P_prev = symmetrize(
        Q + AtPA + H_GK.T @ K + K.T @ H
    )
    p_prev = q + A.T @ p + AtP @ c + H_GK.T @ k + K.T @ h

    return P_prev, p_prev, K, k


def dare_solve(
    Q: Array,
    R: Array,
    A: Array,
    B: Array,
    M: Optional[Array] = None,
    P0: Optional[Array] = None,
    maxiter: int = 100,
    tol: float = 1e-6,
    reg: float = 1e-8,
) -> tuple[Array, Array, int]:
    """Solve discrete-time algebraic Riccati equation (DARE).

    Iterates the Riccati recursion until convergence to find the
    steady-state solution of the infinite-horizon LQR problem.

    Args:
        Q: State cost matrix (n, n).
        R: Control cost matrix (m, m).
        A: Dynamics matrix (n, n).
        B: Control matrix (n, m).
        M: Cross term matrix (n, m).
        P0: Initial value matrix (defaults to Q).
        maxiter: Maximum iterations.
        tol: Convergence tolerance on ||P_new - P||.
        reg: Regularization parameter.

    Returns:
        Tuple of:
            - P: Solution to DARE (n, n)
            - K: Optimal infinite-horizon gain (m, n)
            - iters: Number of iterations to convergence
    """
    n = A.shape[0]
    P = P0 if P0 is not None else Q

    for i in range(maxiter):
        P_new, K, _ = dare_step(P, Q, R, A, B, M, reg)

        # Check convergence
        err = jnp.linalg.norm(P_new - P)
        P = P_new

        if err < tol:
            return P, K, i + 1

    # Did not converge
    return P, K, maxiter


def care_scipy(
    Q: Array,
    R: Array,
    A: Array,
    B: Array,
    M: Optional[Array] = None,
) -> tuple[Array, Array]:
    """Solve continuous-time algebraic Riccati equation (CARE).

    Uses SciPy's implementation wrapped for JAX. Note: This is NOT
    JIT-compatible and should only be used for initialization.

    Solves: A^T P + P A - P B R^{-1} B^T P + Q = 0

    Args:
        Q: State cost matrix (n, n).
        R: Control cost matrix (m, m).
        A: Dynamics matrix (n, n).
        B: Control matrix (n, m).
        M: Cross term matrix (n, m).

    Returns:
        Tuple of:
            - P: Solution to CARE (n, n)
            - K: Optimal continuous-time gain (m, n)
    """
    import scipy.linalg

    # Convert to numpy for scipy
    Q_np = jnp.asarray(Q)
    R_np = jnp.asarray(R)
    A_np = jnp.asarray(A)
    B_np = jnp.asarray(B)

    if M is not None:
        S_np = jnp.asarray(M)
        P_np = scipy.linalg.solve_continuous_are(A_np, B_np, Q_np, R_np, s=S_np)
    else:
        P_np = scipy.linalg.solve_continuous_are(A_np, B_np, Q_np, R_np)

    # Compute gain
    P = jnp.array(P_np)
    K = -jsp.linalg.solve(R, B.T @ P, assume_a='pos')

    return P, K


def tvriccati_backward(
    Q: Array,
    R: Array,
    A: Array,
    B: Array,
    M: Optional[Array] = None,
    Q_T: Optional[Array] = None,
    reg: float = 1e-8,
) -> tuple[Array, Array]:
    """Time-varying Riccati backward pass.

    Computes the sequence of value matrices and gains by running
    the Riccati recursion backward in time.

    Args:
        Q: State cost matrices (T+1, n, n).
        R: Control cost matrices (T, m, m).
        A: Dynamics matrices (T, n, n).
        B: Control matrices (T, n, m).
        M: Cross term matrices (T, n, m).
        Q_T: Terminal cost (n, n), defaults to Q[-1].
        reg: Regularization parameter.

    Returns:
        Tuple of:
            - P: Value matrices (T+1, n, n)
            - K: Gain matrices (T, m, n)
    """
    from jax import lax

    T = A.shape[0]
    n = A.shape[1]
    m = B.shape[2]

    # Initialize
    P = jnp.zeros((T + 1, n, n))
    K = jnp.zeros((T, m, n))

    P = P.at[-1].set(Q_T if Q_T is not None else Q[-1])

    def body(tt, carry):
        P, K = carry
        t = T - 1 - tt

        M_t = None if M is None else M[t]
        P_t, K_t, _ = dare_step(P[t + 1], Q[t], R[t], A[t], B[t], M_t, reg)

        P = P.at[t].set(P_t)
        K = K.at[t].set(K_t)
        return P, K

    P, K = lax.fori_loop(0, T, body, (P, K))
    return P, K


__all__ = [
    'dare_step',
    'dare_step_affine',
    'dare_solve',
    'care_scipy',
    'tvriccati_backward',
]
