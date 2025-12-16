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

"""Utility functions for trajectory optimization.

This module provides the computational building blocks used by trajectory
optimization algorithms:

- Linearization and quadratization of dynamics and costs
- Rollout and evaluation utilities
- Adjoint (costate) equations for gradient computation
- PSD projection and regularization
- Numerical integrators for continuous-time dynamics
- Manifold utilities for circular/periodic state dimensions
"""

# Linearization utilities
from trajax.utils.linearize import (
    vectorize,
    linearize,
    quadratize,
    linearize_dynamics,
    linearize_cost,
    quadratize_cost,
    hamiltonian,
    pad,
)

# Rollout utilities
from trajax.utils.rollout import (
    rollout,
    ddp_rollout,
    line_search_ddp,
    evaluate,
    objective,
    closed_loop_rollout,
    safe_cubic_opt,
)

# Adjoint utilities
from trajax.utils.adjoint import (
    adjoint,
    grad_wrt_controls,
    hvp,
    vhp_params,
)

# PSD utilities
from trajax.utils.psd import (
    project_psd_cone,
    project_psd_batch,
    make_psd_for_lqr,
    regularize_hessian,
    is_psd,
    symmetrize,
    block_diag,
)

# Integrators
from trajax.utils.integrators import (
    euler,
    rk4,
    midpoint,
    heun,
)

# Manifold utilities
from trajax.utils.manifold import (
    _wrap_to_pi,
    get_s1_wrapper,
)

__all__ = [
    # Linearization
    'vectorize',
    'linearize',
    'quadratize',
    'linearize_dynamics',
    'linearize_cost',
    'quadratize_cost',
    'hamiltonian',
    'pad',
    # Rollout
    'rollout',
    'ddp_rollout',
    'line_search_ddp',
    'evaluate',
    'objective',
    'closed_loop_rollout',
    'safe_cubic_opt',
    # Adjoint
    'adjoint',
    'grad_wrt_controls',
    'hvp',
    'vhp_params',
    # PSD
    'project_psd_cone',
    'project_psd_batch',
    'make_psd_for_lqr',
    'regularize_hessian',
    'is_psd',
    'symmetrize',
    'block_diag',
    # Integrators
    'euler',
    'rk4',
    'midpoint',
    'heun',
    # Manifold
    '_wrap_to_pi',
    'get_s1_wrapper',
]
