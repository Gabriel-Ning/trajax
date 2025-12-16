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

"""Configuration classes for MPC controllers.

Provides nested dataclass configuration for MPC problems and solvers.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional


@dataclass
class SolverConfig:
    """Configuration for trajectory optimization solver.

    Attributes:
        solver_type: Type of solver ('ilqr', 'constrained_ilqr', 'sqp', 'cem').
        maxiter: Maximum solver iterations.
        grad_norm_threshold: Gradient norm convergence threshold.
        make_psd: Whether to project Hessians to PSD cone.
        alpha_0: Initial line search step size.
        alpha_min: Minimum line search step size.
        verbose: Whether to print solver progress.
    """
    solver_type: Literal['ilqr', 'constrained_ilqr', 'sqp', 'cem'] = 'ilqr'
    maxiter: int = 100
    grad_norm_threshold: float = 1e-4
    make_psd: bool = True
    psd_delta: float = 0.0
    alpha_0: float = 1.0
    alpha_min: float = 0.00005
    verbose: bool = False

    # Constrained iLQR specific
    maxiter_al: int = 5
    constraints_threshold: float = 1e-2
    penalty_init: float = 1.0
    penalty_update_rate: float = 10.0

    # Additional solver kwargs
    extra_options: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for solver initialization."""
        base = {
            'maxiter': self.maxiter,
            'grad_norm_threshold': self.grad_norm_threshold,
            'make_psd': self.make_psd,
            'psd_delta': self.psd_delta,
            'alpha_0': self.alpha_0,
            'alpha_min': self.alpha_min,
        }
        if self.solver_type == 'constrained_ilqr':
            base.update({
                'maxiter_al': self.maxiter_al,
                'maxiter_ilqr': self.maxiter,
                'constraints_threshold': self.constraints_threshold,
                'penalty_init': self.penalty_init,
                'penalty_update_rate': self.penalty_update_rate,
            })
        base.update(self.extra_options)
        return base


@dataclass
class MPCConfig:
    """Configuration for MPC controller.

    Attributes:
        horizon: MPC prediction horizon (number of timesteps).
        dt: Control timestep.
        relinearize_every: How often to relinearize dynamics (in timesteps).
        warm_start: Whether to warm-start from previous solution.
        solver: Solver configuration.
    """
    horizon: int = 20
    dt: float = 0.01
    relinearize_every: int = 1
    warm_start: bool = True
    solver: SolverConfig = field(default_factory=SolverConfig)

    def __post_init__(self):
        """Convert solver dict to SolverConfig if needed."""
        if isinstance(self.solver, dict):
            self.solver = SolverConfig(**self.solver)


@dataclass
class CostConfig:
    """Configuration for quadratic tracking cost.

    Cost: (x - x_ref)' Q (x - x_ref) + (u - u_ref)' R (u - u_ref)

    Attributes:
        Q: State cost weight (scalar or diagonal).
        R: Control cost weight (scalar or diagonal).
        Q_terminal: Terminal state cost weight (scalar or diagonal).
        position_weight: Weight for position tracking (if applicable).
        velocity_weight: Weight for velocity tracking (if applicable).
        control_weight: Weight for control effort.
    """
    Q: float = 1.0
    R: float = 0.1
    Q_terminal: Optional[float] = None

    # For structured costs
    position_weight: float = 10.0
    velocity_weight: float = 1.0
    control_weight: float = 0.1

    def __post_init__(self):
        if self.Q_terminal is None:
            self.Q_terminal = self.Q * 10.0


@dataclass
class ControllerConfig:
    """Complete MPC controller configuration.

    Combines MPC settings and cost settings.

    Attributes:
        mpc: MPC configuration.
        cost: Cost function configuration.
        state_dim: State dimension (if known).
        control_dim: Control dimension (if known).
    """
    mpc: MPCConfig = field(default_factory=MPCConfig)
    cost: CostConfig = field(default_factory=CostConfig)
    state_dim: Optional[int] = None
    control_dim: Optional[int] = None

    def __post_init__(self):
        if isinstance(self.mpc, dict):
            self.mpc = MPCConfig(**self.mpc)
        if isinstance(self.cost, dict):
            self.cost = CostConfig(**self.cost)
