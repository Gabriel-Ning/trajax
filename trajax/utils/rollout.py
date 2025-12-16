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

"""Rollout utilities for trajectory optimization.

This module provides functions for simulating trajectories through dynamics.
"""

from functools import partial
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from jax import Array, lax, jit

from trajax.utils.linearize import pad, vectorize


@jit
def safe_cubic_opt(
    x1: float,
    x2: float,
    vg1: Tuple[float, float],
    vg2: Tuple[float, float],
) -> float:
    """Safe cubic optimization for line search.

    Finds the argmin of a cubic interpolant between two points using function
    values and gradients. This is more efficient than bisection for line search
    as it requires fewer function evaluations.

    The cubic interpolant is fitted through (x1, v1) and (x2, v2) with gradients
    g1 and g2. This function returns the minimizer of the cubic, clipped to
    [x1, x2] for safety.

    Args:
        x1: First step size point.
        x2: Second step size point.
        vg1: Tuple of (value, gradient) at x1, i.e., (f(x1), f'(x1)).
        vg2: Tuple of (value, gradient) at x2, i.e., (f(x2), f'(x2)).

    Returns:
        x: Minimizer of cubic interpolant, clipped to [x1, x2].

    References:
        - Nocedal & Wright, "Numerical Optimization" (2006)
        - Original implementation in Shooting-SQP (google/trajax)

    Example:
        >>> x1, x2 = 0.0, 1.0
        >>> vg1 = (10.0, 2.0)  # value=10, gradient=2 at x1
        >>> vg2 = (8.0, 0.5)   # value=8, gradient=0.5 at x2
        >>> x_opt = safe_cubic_opt(x1, x2, vg1, vg2)
        >>> print(x_opt)  # Optimal step size between x1 and x2
    """
    v1, g1 = vg1
    v2, g2 = vg2

    # Compute cubic interpolant coefficients
    # Fit cubic c(x) = a*x^3 + b*x^2 + c*x + d through (x1, v1), (x2, v2)
    # with derivatives matching g1, g2
    b1 = g1 + g2 - 3.0 * ((v1 - v2) / (x1 - x2))
    b2 = jnp.sqrt(jnp.maximum(b1 * b1 - g1 * g2, 0.0))

    # Compute critical point of cubic
    # Formula ensures numerical stability by avoiding division by small numbers
    denom = g2 - g1 + 2.0 * b2
    x = jnp.where(
        jnp.abs(denom) > 1e-14,
        x2 - (x2 - x1) * ((g2 + b2 - b1) / denom),
        0.5 * (x1 + x2),  # Fallback to midpoint if denominator is near zero
    )

    # Clip result to safe interval [x1, x2]
    return jnp.maximum(x1, jnp.minimum(x, x2))


def rollout(
    dynamics: Callable,
    U: Array,
    x0: Array,
    *args,
) -> Array:
    """Roll out dynamics: x[t+1] = dynamics(x[t], U[t], t, *args).

    Args:
        dynamics: Dynamics function (x, u, t, *args) -> x_next.
        U: Control sequence of shape (T, m).
        x0: Initial state of shape (n,).
        *args: Additional arguments passed to dynamics.

    Returns:
        X: State trajectory of shape (T+1, n).

    Example:
        >>> X = rollout(dynamics, U, x0)
        >>> assert X.shape == (U.shape[0] + 1, x0.shape[0])
    """
    def dynamics_for_scan(x, ut):
        u, t = ut
        x_next = dynamics(x, u, t, *args)
        return x_next, x_next

    _, X_rest = lax.scan(dynamics_for_scan, x0, (U, jnp.arange(U.shape[0])))
    return jnp.vstack((x0, X_rest))


@partial(jit, static_argnums=(0,))
def ddp_rollout(
    dynamics: Callable,
    X: Array,
    U: Array,
    K: Array,
    k: Array,
    alpha: float,
    *args,
) -> Tuple[Array, Array]:
    """Rollouts used in Differential Dynamic Programming.

    Performs a closed-loop rollout using feedback gains K and feedforward
    terms k, scaled by line search parameter alpha:

        u_new[t] = U[t] + alpha * k[t] + K[t] @ (x_new[t] - X[t])
        x_new[t+1] = dynamics(x_new[t], u_new[t], t)

    This is the standard DDP/iLQR rollout used during line search.

    Args:
        dynamics: Dynamics function (x, u, t, *args) -> x_next.
        X: Current state trajectory of shape (T+1, n).
        U: Current control sequence of shape (T, m).
        K: Feedback gains of shape (T, m, n).
        k: Feedforward terms of shape (T, m).
        alpha: Line search parameter in (0, 1].
        *args: Additional arguments passed to dynamics.

    Returns:
        X_new: Updated state trajectory of shape (T+1, n).
        U_new: Updated control sequence of shape (T, m).

    Example:
        >>> # During line search in iLQR
        >>> X_new, U_new = ddp_rollout(dynamics, X, U, K, k, alpha=0.5)
    """
    n = X.shape[1]
    T, m = U.shape

    X_new = jnp.zeros((T + 1, n))
    U_new = jnp.zeros((T, m))
    X_new = X_new.at[0].set(X[0])

    def body(t, inputs):
        X_new, U_new = inputs
        # Feedback control: u = U + alpha*k + K*(x_new - x_ref)
        del_u = alpha * k[t] + jnp.matmul(K[t], X_new[t] - X[t])
        u = U[t] + del_u
        # Step dynamics
        x = dynamics(X_new[t], u, t, *args)
        U_new = U_new.at[t].set(u)
        X_new = X_new.at[t + 1].set(x)
        return X_new, U_new

    return lax.fori_loop(0, T, body, (X_new, U_new))


@partial(jit, static_argnums=(0, 1))
def line_search_ddp(
    cost: Callable,
    dynamics: Callable,
    X: Array,
    U: Array,
    K: Array,
    k: Array,
    obj: float,
    cost_args: Tuple = (),
    dynamics_args: Tuple = (),
    alpha_0: float = 1.0,
    alpha_min: float = 0.00005,
) -> Tuple[Array, Array, float, float]:
    """Perform line search along DDP direction.

    Searches for a step size alpha that achieves cost reduction using
    the feedback policy defined by (K, k).

    Args:
        cost: Cost function (x, u, t, *args) -> scalar.
        dynamics: Dynamics function (x, u, t, *args) -> x_next.
        X: Current state trajectory of shape (T+1, n).
        U: Current control sequence of shape (T, m).
        K: Feedback gains of shape (T, m, n).
        k: Feedforward terms of shape (T, m).
        obj: Current objective value.
        cost_args: Arguments passed to cost function.
        dynamics_args: Arguments passed to dynamics function.
        alpha_0: Initial step size (default 1.0).
        alpha_min: Minimum step size before giving up (default 5e-5).

    Returns:
        X_new: Best state trajectory found.
        U_new: Best control sequence found.
        obj_new: Objective value of best trajectory.
        alpha: Step size that achieved best trajectory.
    """
    obj = jnp.where(jnp.isnan(obj), jnp.inf, obj)
    costs = partial(evaluate, cost)
    total_cost = lambda X, U, *margs: jnp.sum(costs(X, pad(U), *margs))

    def line_search(inputs):
        _, _, _, alpha = inputs
        X_new, U_new = ddp_rollout(dynamics, X, U, K, k, alpha, *dynamics_args)
        obj_new = total_cost(X_new, U_new, *cost_args)
        alpha = 0.5 * alpha
        obj_new = jnp.where(jnp.isnan(obj_new), obj, obj_new)

        # Only return new trajs if leads to a strict cost decrease
        X_return = jnp.where(obj_new < obj, X_new, X)
        U_return = jnp.where(obj_new < obj, U_new, U)

        return X_return, U_return, jnp.minimum(obj_new, obj), alpha

    return lax.while_loop(
        lambda inputs: jnp.logical_and(inputs[2] >= obj, inputs[3] > alpha_min),
        line_search,
        (X, U, obj, alpha_0),
    )


def evaluate(
    cost: Callable,
    X: Array,
    U: Array,
    *args,
) -> Array:
    """Evaluate cost at each timestep along a trajectory.

    Args:
        cost: Cost function (x, u, t, *args) -> scalar.
        X: State trajectory of shape (T, n) or (T+1, n).
        U: Control sequence of shape (T, m).
        *args: Additional arguments passed to cost.

    Returns:
        costs: Array of costs at each timestep, shape (T,) or (T+1,).
    """
    timesteps = jnp.arange(X.shape[0])
    return vectorize(cost)(X, U, timesteps, *args)


def objective(
    cost: Callable,
    dynamics: Callable,
    U: Array,
    x0: Array,
) -> float:
    """Evaluate total cost for a control sequence.

    Rolls out dynamics and sums costs along the trajectory.

    Args:
        cost: Cost function (x, u, t) -> scalar.
        dynamics: Dynamics function (x, u, t) -> x_next.
        U: Control sequence of shape (T, m).
        x0: Initial state of shape (n,).

    Returns:
        Total objective value (scalar).
    """
    X = rollout(dynamics, U, x0)
    return jnp.sum(evaluate(cost, X, pad(U)))


def closed_loop_rollout(
    dynamics: Callable,
    X_ref: Array,
    U_ref: Array,
    K: Array,
    x0: Array,
    u_min: Array = None,
    u_max: Array = None,
    *args,
) -> Tuple[Array, Array]:
    """Closed-loop rollout with feedback control.

    Simulates the system using feedback control:
        u[t] = clip(U_ref[t] + K[t] @ (x[t] - X_ref[t]), u_min, u_max)
        x[t+1] = dynamics(x[t], u[t], t)

    Args:
        dynamics: Dynamics function (x, u, t, *args) -> x_next.
        X_ref: Reference state trajectory of shape (T+1, n).
        U_ref: Reference control sequence of shape (T, m).
        K: Feedback gains of shape (T, m, n).
        x0: Initial state of shape (n,).
        u_min: Lower control bounds of shape (m,). Optional.
        u_max: Upper control bounds of shape (m,). Optional.
        *args: Additional arguments passed to dynamics.

    Returns:
        X: Actual state trajectory of shape (T+1, n).
        U: Actual control sequence of shape (T, m).
    """
    T, m = U_ref.shape
    n = x0.shape[0]

    X = jnp.zeros((T + 1, n))
    U = jnp.zeros((T, m))
    X = X.at[0].set(x0)

    def body(t, inputs):
        X, U = inputs
        # Feedback control
        u = U_ref[t] + jnp.matmul(K[t], X[t] - X_ref[t])
        # Clip if bounds provided
        if u_min is not None and u_max is not None:
            u = jnp.clip(u, u_min, u_max)
        # Step dynamics
        x_next = dynamics(X[t], u, t, *args)
        U = U.at[t].set(u)
        X = X.at[t + 1].set(x_next)
        return X, U

    return lax.fori_loop(0, T, body, (X, U))
