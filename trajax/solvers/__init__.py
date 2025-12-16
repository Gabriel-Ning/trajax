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

"""Trajectory optimization solvers with class-based interface.

This module provides class-based implementations of trajectory optimization
algorithms with a unified interface.

Available solvers:
- ILQROptimizer: Iterative Linear Quadratic Regulator
- ConstrainedILQROptimizer: Constrained iLQR with Augmented Lagrangian
- CEMOptimizer: Cross-Entropy Method
- RandomShootingOptimizer: Random shooting baseline

Example:
    >>> from trajax.solvers import ILQROptimizer
    >>> optimizer = ILQROptimizer(maxiter=100)
    >>> result = optimizer.solve(problem, x0, U0, params)
    >>>
    >>> # Or build a JIT-compiled solver for repeated use
    >>> solve_fn = optimizer.build_solver(problem)
    >>> result = solve_fn(x0, U0, params)
"""

from trajax.solvers.base import (
    TrajectoryOptimizer,
    TrajectoryOptimizerBase,
    get_solver,
    get_optimizer,  # Alias
)

from trajax.solvers.ilqr import ILQROptimizer
from trajax.solvers.constrained_ilqr import ConstrainedILQROptimizer
from trajax.solvers.cem import CEMOptimizer
from trajax.solvers.sampling import RandomShootingOptimizer

__all__ = [
    # Base classes
    'TrajectoryOptimizer',
    'TrajectoryOptimizerBase',
    'get_solver',
    'get_optimizer',  # Alias
    # Solvers
    'ILQROptimizer',
    'ConstrainedILQROptimizer',
    'CEMOptimizer',
    'RandomShootingOptimizer',
]
