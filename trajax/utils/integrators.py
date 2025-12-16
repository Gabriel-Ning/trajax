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

"""Numerical integration utilities for continuous-time dynamics.

This module provides integrators for converting continuous-time dynamics
(ODEs) to discrete-time dynamics for use with trajectory optimization.

The trajectory optimization algorithms in trajax work with discrete-time
dynamics x[t+1] = f(x[t], u[t], t). If you have continuous-time dynamics
dx/dt = f_cont(x, u, t), use these integrators to discretize them.
"""

from typing import Callable


def euler(dynamics_continuous: Callable, dt: float) -> Callable:
    """Create a discrete-time dynamics function using Euler integration.

    The Euler method is the simplest integration scheme:
        x[t+1] = x[t] + dt * f_cont(x[t], u[t], t)

    It is first-order accurate and may require small dt for accuracy.

    Args:
        dynamics_continuous: Continuous-time dynamics function
            (x, u, t, *args) -> dx/dt.
        dt: Time step for integration.

    Returns:
        Discrete-time dynamics function (x, u, t, *args) -> x_next.

    Example:
        >>> def pendulum_cont(x, u, t):
        ...     theta, omega = x
        ...     return jnp.array([omega, -jnp.sin(theta) + u[0]])
        ...
        >>> pendulum_discrete = euler(pendulum_cont, dt=0.01)
        >>> x_next = pendulum_discrete(x, u, t)
    """
    def dynamics_discrete(x, u, t, *args):
        return x + dt * dynamics_continuous(x, u, t, *args)

    return dynamics_discrete


def rk4(dynamics_continuous: Callable, dt: float) -> Callable:
    """Create a discrete-time dynamics function using RK4 integration.

    The 4th-order Runge-Kutta method:
        k1 = f(x, u, t)
        k2 = f(x + dt/2 * k1, u, t + dt/2)
        k3 = f(x + dt/2 * k2, u, t + dt/2)
        k4 = f(x + dt * k3, u, t + dt)
        x[t+1] = x[t] + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

    This is fourth-order accurate and the standard choice for most applications.

    Args:
        dynamics_continuous: Continuous-time dynamics function
            (x, u, t, *args) -> dx/dt.
        dt: Time step for integration.

    Returns:
        Discrete-time dynamics function (x, u, t, *args) -> x_next.

    Example:
        >>> def pendulum_cont(x, u, t):
        ...     theta, omega = x
        ...     return jnp.array([omega, -jnp.sin(theta) + u[0]])
        ...
        >>> pendulum_discrete = rk4(pendulum_cont, dt=0.01)
        >>> x_next = pendulum_discrete(x, u, t)
    """
    def dynamics_discrete(x, u, t, *args):
        k1 = dynamics_continuous(x, u, t, *args)
        k2 = dynamics_continuous(x + 0.5 * dt * k1, u, t, *args)
        k3 = dynamics_continuous(x + 0.5 * dt * k2, u, t, *args)
        k4 = dynamics_continuous(x + dt * k3, u, t, *args)
        return x + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    return dynamics_discrete


def midpoint(dynamics_continuous: Callable, dt: float) -> Callable:
    """Create a discrete-time dynamics function using midpoint integration.

    The midpoint method (2nd-order Runge-Kutta):
        k1 = f(x, u, t)
        k2 = f(x + dt/2 * k1, u, t + dt/2)
        x[t+1] = x[t] + dt * k2

    Second-order accurate, a good balance between Euler and RK4.

    Args:
        dynamics_continuous: Continuous-time dynamics function
            (x, u, t, *args) -> dx/dt.
        dt: Time step for integration.

    Returns:
        Discrete-time dynamics function (x, u, t, *args) -> x_next.
    """
    def dynamics_discrete(x, u, t, *args):
        k1 = dynamics_continuous(x, u, t, *args)
        k2 = dynamics_continuous(x + 0.5 * dt * k1, u, t, *args)
        return x + dt * k2

    return dynamics_discrete


def heun(dynamics_continuous: Callable, dt: float) -> Callable:
    """Create a discrete-time dynamics function using Heun's method.

    Heun's method (improved Euler / explicit trapezoidal):
        k1 = f(x, u, t)
        k2 = f(x + dt * k1, u, t + dt)
        x[t+1] = x[t] + dt/2 * (k1 + k2)

    Second-order accurate.

    Args:
        dynamics_continuous: Continuous-time dynamics function
            (x, u, t, *args) -> dx/dt.
        dt: Time step for integration.

    Returns:
        Discrete-time dynamics function (x, u, t, *args) -> x_next.
    """
    def dynamics_discrete(x, u, t, *args):
        k1 = dynamics_continuous(x, u, t, *args)
        k2 = dynamics_continuous(x + dt * k1, u, t, *args)
        return x + 0.5 * dt * (k1 + k2)

    return dynamics_discrete
