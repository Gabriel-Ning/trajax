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

"""Linearization and quadratization utilities for trajectory optimization.

This module provides functions for computing Jacobians and Hessians of
dynamics and cost functions along trajectories.
"""

from typing import Callable, Tuple

import jax
from jax import jacobian, hessian, vmap
from jax import Array
import jax.numpy as jnp


def vectorize(fun: Callable, argnums: int = 3) -> Callable:
    """Returns a jitted and vectorized version of the input function.

    Vectorizes the first `argnums` arguments of the function using vmap,
    allowing efficient batch evaluation along a trajectory.

    Args:
        fun: A function f(*args) to be mapped over.
        argnums: Number of leading arguments of fun to vectorize.

    Returns:
        Vectorized/Batched function with arguments corresponding to fun, but
        extra batch dimension in axis 0 for first argnums arguments
        (x, u, t typically). Remaining arguments are not batched.

    Example:
        >>> def cost(x, u, t):
        ...     return x @ x + u @ u
        ...
        >>> vcost = vectorize(cost, argnums=3)
        >>> # Evaluate cost at all timesteps
        >>> costs = vcost(X, U, timesteps)  # Shape: (T+1,)
    """
    def vfun(*args):
        _fun = lambda tup, *margs: fun(*(margs + tup))
        return vmap(
            _fun, in_axes=(None,) + (0,) * argnums
        )(args[argnums:], *args[:argnums])

    return vfun


def linearize(fun: Callable, argnums: int = 3) -> Callable:
    """Vectorized gradient or jacobian operator.

    Returns a function that computes Jacobians of fun with respect to
    state (x) and control (u) along a trajectory.

    Args:
        fun: Function with signature fun(x, u, t, *args).
            Can be scalar (cost) or vector (dynamics) valued.
        argnums: Number of leading arguments of fun to vectorize.

    Returns:
        A function that evaluates Jacobians along a trajectory:

        For dynamics:
            A, B = linearize(dynamics)(X, pad(U), timesteps)
            where A = df/dx, B = df/du

        For cost:
            q, r = linearize(cost)(X, pad(U), timesteps)
            where q = dc/dx, r = dc/du

    Example:
        >>> dynamics_jacobians = linearize(dynamics)
        >>> cost_gradients = linearize(cost)
        >>> A, B = dynamics_jacobians(X, pad(U), timesteps)  # Shape: (T+1, n, n), (T+1, n, m)
        >>> q, r = cost_gradients(X, pad(U), timesteps)       # Shape: (T+1, n), (T+1, m)
    """
    jacobian_x = jacobian(fun)
    jacobian_u = jacobian(fun, argnums=1)

    def linearizer(*args):
        return jacobian_x(*args), jacobian_u(*args)

    return vectorize(linearizer, argnums)


def quadratize(fun: Callable, argnums: int = 3) -> Callable:
    """Vectorized Hessian operator for a scalar function.

    Returns a function that computes Hessians of fun with respect to
    state and control along a trajectory.

    Args:
        fun: Scalar function with signature fun(x, u, t, *args).
        argnums: Number of leading arguments of fun to vectorize.

    Returns:
        A function that evaluates Hessians along a trajectory:
            Q, R, M = quadratize(cost)(X, pad(U), timesteps)

        where:
            Q = d²c/dx² of shape (T+1, n, n)
            R = d²c/du² of shape (T+1, m, m)
            M = d²c/dxdu of shape (T+1, n, m)

    Example:
        >>> hessians = quadratize(cost)
        >>> Q, R, M = hessians(X, pad(U), timesteps)
    """
    hessian_x = hessian(fun)
    hessian_u = hessian(fun, argnums=1)
    hessian_x_u = jacobian(jax.grad(fun), argnums=1)

    def quadratizer(*args):
        return hessian_x(*args), hessian_u(*args), hessian_x_u(*args)

    return vectorize(quadratizer, argnums)


def linearize_dynamics(
    dynamics: Callable,
    X: Array,
    U: Array,
    params=(),
) -> Tuple[Array, Array]:
    """Linearize dynamics along a trajectory.

    Computes the Jacobians A = df/dx and B = df/du at each point
    along the trajectory.

    Args:
        dynamics: Dynamics function (x, u, t, params) -> x_next.
        X: State trajectory of shape (T+1, n).
        U: Control trajectory of shape (T, m).
        params: Parameters to pass to dynamics.

    Returns:
        A: Dynamics Jacobian wrt state, shape (T, n, n).
        B: Dynamics Jacobian wrt control, shape (T, n, m).
    """
    T = U.shape[0]
    m = U.shape[1]

    # Pad U for consistent indexing
    U_padded = jnp.vstack([U, jnp.zeros((1, m))])
    timesteps = jnp.arange(T + 1)

    def dynamics_wrapper(x, u, t):
        return dynamics(x, u, t, params)

    jacobians = linearize(dynamics_wrapper, argnums=3)
    A_full, B_full = jacobians(X, U_padded, timesteps)

    # Return only T timesteps (exclude terminal)
    return A_full[:T], B_full[:T]


def linearize_cost(
    cost: Callable,
    X: Array,
    U: Array,
    params=(),
) -> Tuple[Array, Array]:
    """Compute cost gradients along a trajectory.

    Args:
        cost: Cost function (x, u, t, params) -> scalar.
        X: State trajectory of shape (T+1, n).
        U: Control trajectory of shape (T, m).
        params: Parameters to pass to cost.

    Returns:
        q: Cost gradient wrt state, shape (T+1, n).
        r: Cost gradient wrt control, shape (T+1, m).
    """
    T = U.shape[0]
    m = U.shape[1]

    U_padded = jnp.vstack([U, jnp.zeros((1, m))])
    timesteps = jnp.arange(T + 1)

    def cost_wrapper(x, u, t):
        return cost(x, u, t, params)

    gradients = linearize(cost_wrapper, argnums=3)
    return gradients(X, U_padded, timesteps)


def quadratize_cost(
    cost: Callable,
    X: Array,
    U: Array,
    params=(),
) -> Tuple[Array, Array, Array]:
    """Compute cost Hessians along a trajectory.

    Args:
        cost: Cost function (x, u, t, params) -> scalar.
        X: State trajectory of shape (T+1, n).
        U: Control trajectory of shape (T, m).
        params: Parameters to pass to cost.

    Returns:
        Q: Hessian wrt state, shape (T+1, n, n).
        R: Hessian wrt control, shape (T+1, m, m).
        M: Mixed Hessian, shape (T+1, n, m).
    """
    T = U.shape[0]
    m = U.shape[1]

    U_padded = jnp.vstack([U, jnp.zeros((1, m))])
    timesteps = jnp.arange(T + 1)

    def cost_wrapper(x, u, t):
        return cost(x, u, t, params)

    hessians = quadratize(cost_wrapper, argnums=3)
    return hessians(X, U_padded, timesteps)


def hamiltonian(cost: Callable, dynamics: Callable) -> Callable:
    """Returns function to evaluate associated Hamiltonian.

    The Hamiltonian is: H(x, u, t, p) = c(x, u, t) + p' f(x, u, t)

    where c is the cost, f is the dynamics, and p is the costate (adjoint).

    Args:
        cost: Cost function (x, u, t, *args) -> scalar.
        dynamics: Dynamics function (x, u, t, *args) -> x_next.

    Returns:
        Function (x, u, t, p, cost_args, dynamics_args) -> scalar.
    """
    def fun(x, u, t, p, cost_args=(), dynamics_args=()):
        return cost(x, u, t, *cost_args) + jnp.dot(
            p, dynamics(x, u, t, *dynamics_args)
        )

    return fun


# Convenience function to pad arrays
def pad(A: Array) -> Array:
    """Pad array with a row of zeros at the end.

    This is useful for aligning control arrays with state arrays:
    X has shape (T+1, n), U has shape (T, m), pad(U) has shape (T+1, m).

    Args:
        A: Array of shape (T, ...).

    Returns:
        Padded array of shape (T+1, ...).
    """
    return jnp.vstack((A, jnp.zeros((1,) + A.shape[1:])))
