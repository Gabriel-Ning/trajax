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

"""Adjoint (costate) equation utilities for trajectory optimization.

The adjoint equations are used to compute gradients of the objective
with respect to controls efficiently via backpropagation through time.
"""

from typing import Tuple

import jax.numpy as jnp
from jax import Array, lax


def adjoint(
    A: Array,
    B: Array,
    q: Array,
    r: Array,
) -> Tuple[Array, Array, Array]:
    """Solve adjoint equations to compute gradient.

    The adjoint equations propagate costates backward through time:
        p[t] = A[t]' @ p[t+1] + q[t]
        g[t] = r[t] + B[t]' @ p[t+1]

    where:
        - p is the costate (adjoint) trajectory
        - g is the gradient of the objective w.r.t. controls
        - A, B are dynamics Jacobians
        - q, r are cost gradients w.r.t. state and control

    Args:
        A: Dynamics Jacobians wrt state, shape (T+1, n, n) or (T, n, n).
        B: Dynamics Jacobians wrt control, shape (T+1, n, m) or (T, n, m).
        q: Cost gradients wrt state, shape (T+1, n).
        r: Cost gradients wrt control, shape (T+1, m) or (T, m).

    Returns:
        gradient: Gradient of objective w.r.t. controls, shape (T, m).
        adjoints: Costate trajectory, shape (T, n).
        p_initial: Initial costate (at t=0), shape (n,).

    Example:
        >>> # After linearizing dynamics and cost
        >>> A, B = linearize_dynamics(dynamics, X, U, params)
        >>> q, r = linearize_cost(cost, X, U, params)
        >>> gradient, adjoints, _ = adjoint(A, B, q, r)
    """
    n = q.shape[1]
    T = q.shape[0] - 1
    m = r.shape[1]

    # Initialize storage
    P = jnp.zeros((T, n))
    g = jnp.zeros((T, m))

    def body(p, t):
        """Backward recursion of adjoint equations."""
        # Gradient w.r.t. control at time t
        g_t = r[t] + jnp.matmul(B[t].T, p)
        # Costate update (backward)
        p_new = jnp.matmul(A[t].T, p) + q[t]
        return p_new, (p_new, g_t)

    # Start from terminal costate (q[T])
    p_final, (P, g) = lax.scan(body, q[T], jnp.arange(T - 1, -1, -1))

    # Flip to get forward-time ordering
    return jnp.flipud(g), jnp.vstack((jnp.flipud(P[:T - 1]), q[T])), p_final


def grad_wrt_controls(
    cost,
    dynamics,
    U: Array,
    x0: Array,
    cost_args: Tuple = (),
    dynamics_args: Tuple = (),
) -> Array:
    """Compute gradient of objective w.r.t. controls using adjoint method.

    This is the efficient way to compute gradients for trajectory optimization:
    O(T * (n + m)^2) instead of O(T^2 * m * n) for naive differentiation.

    Args:
        cost: Cost function (x, u, t, *args) -> scalar.
        dynamics: Dynamics function (x, u, t, *args) -> x_next.
        U: Control sequence of shape (T, m).
        x0: Initial state of shape (n,).
        cost_args: Arguments passed to cost function.
        dynamics_args: Arguments passed to dynamics function.

    Returns:
        gradient: Gradient of total cost w.r.t. controls, shape (T, m).
    """
    from trajax.utils.linearize import linearize, pad
    from trajax.utils.rollout import rollout

    jacobians = linearize(dynamics)
    grad_cost = linearize(cost)

    X = rollout(dynamics, U, x0, *dynamics_args)
    timesteps = jnp.arange(X.shape[0])
    A, B = jacobians(X, pad(U), timesteps, *dynamics_args)
    q, r = grad_cost(X, pad(U), timesteps, *cost_args)
    gradient, _, _ = adjoint(A, B, q, r)
    return gradient


def hvp(
    cost,
    dynamics,
    U: Array,
    x0: Array,
    V: Array,
    cost_args: Tuple = (),
    dynamics_args: Tuple = (),
) -> Tuple[Array, Array]:
    """Compute Hessian-vector product.

    Efficiently computes H @ V where H is the Hessian of the objective
    w.r.t. controls, without materializing the full Hessian.

    Args:
        cost: Cost function (x, u, t, *args) -> scalar.
        dynamics: Dynamics function (x, u, t, *args) -> x_next.
        U: Control sequence of shape (T, m).
        x0: Initial state of shape (n,).
        V: Vector of shape (T, m).
        cost_args: Arguments passed to cost function.
        dynamics_args: Arguments passed to dynamics function.

    Returns:
        gradient: Gradient at U, shape (T, m).
        hvp: Hessian-vector product H @ V, shape (T, m).
    """
    import jax
    from functools import partial

    grad_fn = partial(grad_wrt_controls, cost, dynamics)
    return jax.jvp(
        lambda U1: grad_fn(U1, x0, cost_args, dynamics_args),
        (U,),
        (V,),
    )


def vhp_params(cost):
    """Returns a function that evaluates vector-Hessian products w.r.t. params.

    This is used in the custom VJP for iLQR to differentiate through
    the solver with respect to cost function parameters.

    Args:
        cost: Cost function (x, u, t, *args) -> scalar.

    Returns:
        Function (vector, X, U, A, B, *args) -> (P, gradient) where:
            P: Sensitivity matrix
            gradient: Gradient w.r.t. cost parameters
    """
    import jax
    from jax import jacobian, tree_util

    hessian_u_params = jacobian(jax.grad(cost, argnums=1), argnums=3)
    hessian_x_params = jacobian(jax.grad(cost, argnums=0), argnums=3)

    def vhp(vector, X, U, A, B, *args):
        """Evaluate vector-Hessian product.

        Args:
            vector: Input vector of shape (T+1, m).
            X: State trajectory of shape (T+1, n).
            U: Control trajectory of shape (T+1, m).
            A: Dynamics Jacobians wrt state.
            B: Dynamics Jacobians wrt control.
            *args: Arguments passed to cost.

        Returns:
            Tuple (P, gradient) for parameter differentiation.
        """
        T = X.shape[0] - 1
        params = args[0]
        gradient = tree_util.tree_map(jnp.zeros_like, params)
        Cx = hessian_x_params(X[T], U[T], T, *args)
        contract = lambda x, y: jnp.tensordot(x, y, (-1, 0))

        def body(tt, inputs):
            """Accumulate vector-Hessian product over all time steps."""
            P, g = inputs
            t = T - 1 - tt
            Cx_t = hessian_x_params(X[t], U[t], t, *args)
            Cu_t = hessian_u_params(X[t], U[t], t, *args)
            w = jnp.matmul(B[t], vector[t])
            g = tree_util.tree_map(
                lambda P_, g_, Cu_: g_ + contract(vector[t], Cu_) + contract(w, P_),
                P, g, Cu_t,
            )
            P = tree_util.tree_map(
                lambda P_, Cx_: contract(A[t].T, P_) + Cx_,
                P, Cx_t,
            )
            return P, g

        return lax.fori_loop(0, T, body, (Cx, gradient))

    return vhp
