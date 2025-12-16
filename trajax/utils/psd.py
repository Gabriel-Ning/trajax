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

"""Positive semi-definite matrix utilities.

This module provides functions for projecting matrices to the PSD cone
and related operations used in trajectory optimization algorithms.
"""

import jax.numpy as jnp
from jax import Array, jit, vmap


@jit
def project_psd_cone(Q: Array, delta: float = 0.0) -> Array:
    """Project a symmetric matrix to the PSD cone.

    Computes the nearest positive semi-definite matrix by setting
    negative eigenvalues to zero (or to delta).

    This is useful for ensuring that cost Hessians are positive definite,
    which is required for the LQR sub-problem in iLQR to have a unique solution.

    Args:
        Q: Symmetric matrix of shape (n, n).
        delta: Minimum eigenvalue of the projection. Use delta > 0
            to ensure positive definiteness instead of semi-definiteness.

    Returns:
        Q_psd: Projected matrix of shape (n, n), guaranteed to be symmetric
            with all eigenvalues >= delta.

    Example:
        >>> Q = jnp.array([[1, 2], [2, 1]])  # Not PSD (has negative eigenvalue)
        >>> Q_psd = project_psd_cone(Q)
        >>> eigvals = jnp.linalg.eigvalsh(Q_psd)
        >>> assert jnp.all(eigvals >= 0)
    """
    # Eigendecomposition
    S, V = jnp.linalg.eigh(Q)
    # Clip eigenvalues
    S = jnp.maximum(S, delta)
    # Reconstruct
    Q_plus = jnp.matmul(V, jnp.matmul(jnp.diag(S), V.T))
    # Ensure symmetry
    return 0.5 * (Q_plus + Q_plus.T)


def project_psd_batch(Q_batch: Array, delta: float = 0.0) -> Array:
    """Project a batch of matrices to the PSD cone.

    Args:
        Q_batch: Batch of symmetric matrices of shape (T, n, n).
        delta: Minimum eigenvalue of the projections.

    Returns:
        Q_psd_batch: Projected matrices of shape (T, n, n).
    """
    return vmap(lambda Q: project_psd_cone(Q, delta))(Q_batch)


def make_psd_for_lqr(Q: Array, R: Array, delta: float = 0.0):
    """Ensure Q and R matrices are PSD for LQR.

    Projects both state cost Hessian Q and control cost Hessian R
    to the PSD cone along the trajectory.

    Args:
        Q: State cost Hessians of shape (T+1, n, n).
        R: Control cost Hessians of shape (T+1, m, m).
        delta: Minimum eigenvalue.

    Returns:
        Q_psd: Projected Q of shape (T+1, n, n).
        R_psd: Projected R of shape (T+1, m, m).
    """
    return project_psd_batch(Q, delta), project_psd_batch(R, delta)


def regularize_hessian(H: Array, reg: float = 1e-6) -> Array:
    """Add regularization to a Hessian matrix.

    Adds reg * I to the matrix to improve numerical conditioning.

    Args:
        H: Hessian matrix of shape (n, n).
        reg: Regularization coefficient.

    Returns:
        H_reg: Regularized Hessian of shape (n, n).
    """
    n = H.shape[0]
    return H + reg * jnp.eye(n)


def is_psd(Q: Array, tol: float = 1e-8) -> bool:
    """Check if a matrix is positive semi-definite.

    Args:
        Q: Matrix to check, shape (n, n).
        tol: Tolerance for eigenvalue comparison.

    Returns:
        True if all eigenvalues are >= -tol.
    """
    eigvals = jnp.linalg.eigvalsh(Q)
    return jnp.all(eigvals >= -tol)


def symmetrize(Q: Array) -> Array:
    """Symmetrize a matrix.

    Args:
        Q: Matrix of shape (n, n).

    Returns:
        Symmetric matrix (Q + Q') / 2.
    """
    return 0.5 * (Q + Q.T)


def block_diag(*arrays) -> Array:
    """Create a block diagonal matrix from input arrays.

    Args:
        *arrays: Square matrices to place on the diagonal.

    Returns:
        Block diagonal matrix.

    Example:
        >>> A = jnp.eye(2)
        >>> B = jnp.ones((3, 3))
        >>> C = block_diag(A, B)  # Shape (5, 5)
    """
    if len(arrays) == 0:
        return jnp.array([[]])

    if len(arrays) == 1:
        return arrays[0]

    # Compute total size
    sizes = [arr.shape[0] for arr in arrays]
    total = sum(sizes)

    # Build block diagonal
    result = jnp.zeros((total, total))
    idx = 0
    for arr, size in zip(arrays, sizes):
        result = result.at[idx:idx+size, idx:idx+size].set(arr)
        idx += size

    return result
