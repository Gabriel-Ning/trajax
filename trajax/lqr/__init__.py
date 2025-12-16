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

"""Linear Quadratic Regulator (LQR) solvers.

This module provides solvers for time-varying and infinite-horizon
LQR problems, including:
- Time-varying LQR (TVLQR) for finite-horizon problems
- Constrained TVLQR using ADMM
- Riccati equation solvers (DARE/CARE)
- Backward Riccati recursion utilities

Example:
    >>> from trajax.lqr import tvlqr, rollout
    >>>
    >>> # Solve time-varying LQR
    >>> K, k, P, p = tvlqr(Q, q, R, r, M, A, B, c)
    >>>
    >>> # Rollout optimal policy
    >>> X, U = rollout(K, k, x0, A, B, c)
"""

# Time-varying LQR
from trajax.lqr.tvlqr import (
    tvlqr,
    ctvlqr,
    lqr_step,
    rollout,
)

# Riccati solvers
from trajax.lqr.riccati import (
    dare_step,
    dare_step_affine,
    dare_solve,
    care_scipy,
    tvriccati_backward,
)

__all__ = [
    # Time-varying LQR
    'tvlqr',
    'ctvlqr',
    'lqr_step',
    'rollout',
    # Riccati solvers
    'dare_step',
    'dare_step_affine',
    'dare_solve',
    'care_scipy',
    'tvriccati_backward',
]
