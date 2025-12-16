"""Trajax: Trajectory optimization in JAX.

A modular library for trajectory optimization and optimal control,
featuring multiple solvers, QP backends, and a unified MPC interface.

Main modules:
- trajax.core: Core abstractions (TrajectoryProblem, Trajectory)
- trajax.solvers: Trajectory optimization algorithms (iLQR, CEM, etc.)
- trajax.mpc: Model Predictive Control interface
- trajax.lqr: Linear Quadratic Regulator solvers
- trajax.qp: QP solver backends (OSQP, Clarabel, etc.)
- trajax.utils: Utility functions (linearize, rollout, etc.)
- trajax.legacy: Backward-compatible API
"""

# Copyright 2023 Google LLC
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

# Legacy modules (backward compatibility)
from . import integrators
from . import optimizers

# New modular interface
from . import core
from . import solvers
from . import mpc
from . import lqr
from . import qp
from . import utils
from . import legacy
