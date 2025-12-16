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

"""Legacy API compatibility layer.

This module provides backward-compatible wrappers for the original trajax API.
New code should use the new class-based interface in trajax.optimizers.

The original function-based API:
    X, U, obj, grad, adjoints, lqr, iter = ilqr(cost, dynamics, x0, U)

The new class-based API:
    optimizer = ILQROptimizer()
    result = optimizer.solve(problem, x0, U0, params)
    X, U, obj = result.X, result.U, result.obj
"""

# Import from the original optimizers.py file (not the new module)
# These are the legacy function-based APIs
import trajax.optimizers as _old_optimizers

# Re-export original functions
ilqr = _old_optimizers.ilqr
constrained_ilqr = _old_optimizers.constrained_ilqr
cem = _old_optimizers.cem
random_shooting = _old_optimizers.random_shooting
scipy_minimize = _old_optimizers.scipy_minimize
rollout = _old_optimizers.rollout
objective = _old_optimizers.objective
linearize = _old_optimizers.linearize
quadratize = _old_optimizers.quadratize
adjoint = _old_optimizers.adjoint
vectorize = _old_optimizers.vectorize
evaluate = _old_optimizers.evaluate
ddp_rollout = _old_optimizers.ddp_rollout
line_search_ddp = _old_optimizers.line_search_ddp
project_psd_cone = _old_optimizers.project_psd_cone
pad = _old_optimizers.pad
grad_wrt_controls = _old_optimizers.grad_wrt_controls
hvp = _old_optimizers.hvp
hamiltonian = _old_optimizers.hamiltonian

__all__ = [
    'ilqr',
    'constrained_ilqr',
    'cem',
    'random_shooting',
    'scipy_minimize',
    'rollout',
    'objective',
    'linearize',
    'quadratize',
    'adjoint',
    'vectorize',
    'evaluate',
    'ddp_rollout',
    'line_search_ddp',
    'project_psd_cone',
    'pad',
    'grad_wrt_controls',
    'hvp',
    'hamiltonian',
]
