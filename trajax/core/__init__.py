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

"""Core abstractions for trajectory optimization.

This module provides the fundamental data structures and type definitions
for trajectory optimization:

- TrajectoryProblem: Problem specification (dynamics, cost, constraints)
- Trajectory: Solution container (states, controls, metadata)
- Type definitions and protocols for dynamics, cost, and constraint functions
"""

from trajax.core.types import (
    SolverStatus,
    PyTree,
    DynamicsFn,
    CostFn,
    TerminalCostFn,
    ConstraintFn,
    SolverFn,
    LQRParams,
    Bounds,
)

from trajax.core.trajectory import (
    Trajectory,
    trajectory_from_controls,
)

from trajax.core.problem import (
    TrajectoryProblem,
    create_tracking_problem,
)

__all__ = [
    # Types
    'SolverStatus',
    'PyTree',
    'DynamicsFn',
    'CostFn',
    'TerminalCostFn',
    'ConstraintFn',
    'SolverFn',
    'LQRParams',
    'Bounds',
    # Data structures
    'Trajectory',
    'trajectory_from_controls',
    # Problem
    'TrajectoryProblem',
    'create_tracking_problem',
]
