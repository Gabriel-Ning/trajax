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

"""Manifold utilities for handling circular state components.

This module provides utilities for working with periodic state dimensions,
particularly S¹ (circle) manifolds where angles wrap around at ±π.

Common use cases:
- Handling rotational angles in dynamics
- Ensuring angles stay in canonical [-π, π] range
- Proper state space handling for systems with circular coordinates
"""

from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import Array


def _wrap_to_pi(x: Array) -> Array:
    """Wraps angles to lie within [-π, π] range.

    This is essential for maintaining numerical stability and canonical
    angle representations in trajectory optimization. It uses the modulo
    operation to map any angle to the principal range.

    Args:
        x: Angle(s) in radians, can be scalar or array.

    Returns:
        Wrapped angle(s) in [-π, π] range.

    Example:
        >>> angle = jnp.array(3.5 * jnp.pi)  # 3.5π radians
        >>> wrapped = _wrap_to_pi(angle)
        >>> print(wrapped)  # ≈ -0.5π
    """
    return (x + jnp.pi) % (2 * jnp.pi) - jnp.pi


def get_s1_wrapper(
    s1_ind: Optional[Tuple[int, ...]] = None
) -> Callable[[Array], Array]:
    """Returns a JIT-compiled function that wraps S¹ state components to [-π, π].

    This utility creates a state wrapper function for systems with circular
    (periodic) state dimensions. It wraps specified indices of the state
    vector to the canonical angle range while leaving other dimensions unchanged.

    Args:
        s1_ind: Tuple of state indices corresponding to S¹ (angle) dimensions.
               If None or empty, returns identity function (no wrapping).
               Example: (0, 2) wraps state[0] and state[2] to [-π, π].

    Returns:
        A JIT-compiled function (state) -> wrapped_state that applies angle
        wrapping to specified indices.

    Example:
        >>> s1_ind = (0, 3)  # Wrap state dimensions 0 and 3
        >>> wrapper = get_s1_wrapper(s1_ind)
        >>> x = jnp.array([4.0, 1.0, 2.0, 3.5])
        >>> x_wrapped = wrapper(x)
        >>> print(x_wrapped[0])  # Angle wrapped to [-π, π]
        >>> print(x_wrapped[1])  # Unchanged (1.0)
    """
    identity_fn = lambda x: x

    # Handle None or empty tuple
    if s1_ind is None or len(s1_ind) == 0:
        # Return identity function if no S¹ indices specified
        return jax.jit(identity_fn)

    # Convert to integer array for advanced indexing
    idxs = jnp.array(s1_ind, dtype=jnp.int32)

    def state_wrapper(x: Array) -> Array:
        """Wraps specified state indices to S¹ manifold."""
        x = x.at[idxs].set(_wrap_to_pi(x[idxs]))
        return x

    return jax.jit(state_wrapper)
