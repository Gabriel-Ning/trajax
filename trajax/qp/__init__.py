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

"""QP solver backends for trajectory optimization.

This module provides multiple QP solver backends that implement the
QPSolver protocol. Solvers are loaded conditionally based on available
dependencies.

Available backends:
- alilqr: Augmented Lagrangian iLQR (always available, JAX-native)
- cvxpy: CVXPY with ECOS/CVXOPT (requires cvxpy)
- osqp: OSQP solver (requires osqp)
- clarabel: Clarabel solver (requires clarabel)
- proxqp: ProxQP solver (requires proxsuite)
"""

from trajax.qp.base import (
    QPStatus,
    QPFormulation,
    QPSolution,
    QPSolver,
    QPSolverBase,
    create_qp_formulation_from_trajectory,
)

# Registry of available backends
_AVAILABLE_BACKENDS = {}

# Always available: AL-iLQR backend (JAX native)
try:
    from trajax.qp.alilqr_backend import ALiLQRBackend
    _AVAILABLE_BACKENDS['alilqr'] = ALiLQRBackend
except ImportError:
    pass

# Optional: CVXPY backend
try:
    from trajax.qp.cvxpy_backend import CVXPYBackend
    _AVAILABLE_BACKENDS['cvxpy'] = CVXPYBackend
    _AVAILABLE_BACKENDS['ecos'] = CVXPYBackend
    _AVAILABLE_BACKENDS['cvxopt'] = CVXPYBackend
except ImportError:
    pass

# Optional: OSQP backend
try:
    from trajax.qp.osqp_backend import OSQPBackend
    _AVAILABLE_BACKENDS['osqp'] = OSQPBackend
except ImportError:
    pass

# Optional: Clarabel backend
try:
    from trajax.qp.clarabel_backend import ClarabelBackend
    _AVAILABLE_BACKENDS['clarabel'] = ClarabelBackend
except ImportError:
    pass

# Optional: ProxQP backend
try:
    from trajax.qp.proxqp_backend import ProxQPBackend
    _AVAILABLE_BACKENDS['proxqp'] = ProxQPBackend
except ImportError:
    pass


def get_available_backends():
    """Return list of available QP solver backend names."""
    return list(_AVAILABLE_BACKENDS.keys())


def get_qp_solver(name: str, **kwargs) -> QPSolverBase:
    """Factory function to create QP solver by name.

    Args:
        name: Backend name ('osqp', 'clarabel', 'cvxpy', 'alilqr', etc.)
        **kwargs: Backend-specific options.

    Returns:
        QPSolver instance.

    Raises:
        ValueError: If backend is not available.
    """
    name_lower = name.lower()
    if name_lower not in _AVAILABLE_BACKENDS:
        available = get_available_backends()
        raise ValueError(
            f"QP solver '{name}' not available. "
            f"Available backends: {available}. "
            f"Install the required package to enable more backends."
        )
    return _AVAILABLE_BACKENDS[name_lower](**kwargs)


__all__ = [
    # Base classes
    'QPStatus',
    'QPFormulation',
    'QPSolution',
    'QPSolver',
    'QPSolverBase',
    'create_qp_formulation_from_trajectory',
    # Factory
    'get_qp_solver',
    'get_available_backends',
]
