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

"""Tests for advanced line search utilities."""

from absl.testing import absltest
from absl.testing import parameterized

import jax
import jax.numpy as jnp
from jax import config
import numpy as np

from trajax.utils import safe_cubic_opt

config.update('jax_enable_x64', True)


class SafeCubicOptTest(parameterized.TestCase):
    """Tests for safe_cubic_opt cubic interpolation line search."""

    def test_result_within_bounds(self):
        """Result should always be clipped to [x1, x2]."""
        x1, x2 = 0.0, 1.0
        vg1 = (10.0, 2.0)   # value, gradient at x1
        vg2 = (8.0, 0.5)    # value, gradient at x2

        result = safe_cubic_opt(x1, x2, vg1, vg2)
        self.assertGreaterEqual(float(result), x1)
        self.assertLessEqual(float(result), x2)

    def test_increasing_function(self):
        """For monotonically increasing function, cubic minimizer may be at endpoints."""
        x1, x2 = 0.0, 1.0
        # Both gradients positive → function increasing
        vg1 = (0.0, 5.0)    # value=0, positive gradient
        vg2 = (2.0, 4.0)    # value=2, still positive

        result = safe_cubic_opt(x1, x2, vg1, vg2)
        # Result should be within bounds
        self.assertGreaterEqual(float(result), x1)
        self.assertLessEqual(float(result), x2)
        # For increasing function, cubic may prefer x1 (lower value) or x2 depending on gradients
        # The key property is that it's a valid interpolation result

    def test_decreasing_function(self):
        """For monotonically decreasing function, should return x2."""
        x1, x2 = 0.0, 1.0
        # Both gradients negative → function decreasing
        vg1 = (10.0, -5.0)   # value=10, negative gradient
        vg2 = (2.0, -1.0)    # value=2, still negative

        result = safe_cubic_opt(x1, x2, vg1, vg2)
        # Should be close to x2 (better value there)
        self.assertGreater(float(result), 0.5)

    def test_optimal_midpoint(self):
        """For symmetric function, optimum should be near middle."""
        x1, x2 = 0.0, 1.0
        # Symmetric quadratic: f(x) = (x - 0.5)^2
        # f(0) = 0.25, f'(0) = -1
        # f(1) = 0.25, f'(1) = 1
        vg1 = (0.25, -1.0)
        vg2 = (0.25, 1.0)

        result = safe_cubic_opt(x1, x2, vg1, vg2)
        # Optimum should be near 0.5
        self.assertAlmostEqual(float(result), 0.5, places=2)

    def test_jittable(self):
        """Function should be JIT-compilable."""
        jitted_opt = jax.jit(safe_cubic_opt)
        x1, x2 = 0.0, 1.0
        vg1 = (10.0, 2.0)
        vg2 = (8.0, 0.5)

        result = jitted_opt(x1, x2, vg1, vg2)
        self.assertGreaterEqual(float(result), x1)
        self.assertLessEqual(float(result), x2)

    def test_various_step_sizes(self):
        """Test with different step size intervals."""
        vg1 = (10.0, 2.0)
        vg2 = (8.0, 0.5)

        for x1, x2 in [(0.0, 1.0), (0.1, 0.9), (0.01, 0.1), (10.0, 20.0)]:
            result = safe_cubic_opt(x1, x2, vg1, vg2)
            self.assertGreaterEqual(float(result), x1)
            self.assertLessEqual(float(result), x2)

    def test_zero_gradient_endpoints(self):
        """Test with zero gradients (critical points)."""
        x1, x2 = 0.0, 1.0
        # Both gradients are zero at endpoints
        vg1 = (5.0, 0.0)
        vg2 = (3.0, 0.0)

        result = safe_cubic_opt(x1, x2, vg1, vg2)
        self.assertGreaterEqual(float(result), x1)
        self.assertLessEqual(float(result), x2)

    def test_negative_step_sizes(self):
        """Test with negative step sizes (backwards line search)."""
        x1, x2 = -1.0, 0.0
        vg1 = (10.0, 2.0)
        vg2 = (8.0, 0.5)

        result = safe_cubic_opt(x1, x2, vg1, vg2)
        self.assertGreaterEqual(float(result), x1)
        self.assertLessEqual(float(result), x2)

    def test_consistency_with_original(self):
        """Test against known behavior of original implementation."""
        # Example from numerical optimization literature
        x1, x2 = 0.0, 1.0
        # f(x) = (x - 0.3)^2 * 10
        # f(0) = 0.9, f'(0) = -6
        # f(1) = 4.9, f'(1) = 14
        vg1 = (0.9, -6.0)
        vg2 = (4.9, 14.0)

        result = safe_cubic_opt(x1, x2, vg1, vg2)
        # Optimum of cubic interpolant should be between x1 and x2
        self.assertGreaterEqual(float(result), 0.0)
        self.assertLessEqual(float(result), 1.0)
        # Should be closer to the actual minimum at x ≈ 0.3
        self.assertLess(float(result), 0.7)

    def test_very_small_interval(self):
        """Test with very small step size interval."""
        x1, x2 = 0.0, 1e-6
        vg1 = (10.0, 2.0)
        vg2 = (8.0, 0.5)

        result = safe_cubic_opt(x1, x2, vg1, vg2)
        self.assertGreaterEqual(float(result), x1)
        self.assertLessEqual(float(result), x2)

    @parameterized.parameters(
        ((0.0, 1.0), (10.0, 2.0), (8.0, 0.5)),
        ((0.1, 0.9), (5.0, -1.0), (3.0, 1.0)),
        ((0.5, 1.5), (20.0, 0.0), (10.0, 5.0)),
        ((-1.0, 0.0), (15.0, -5.0), (12.0, -2.0)),
    )
    def test_various_configurations(self, x_interval, vg1, vg2):
        """Test various configurations of step sizes and gradients."""
        x1, x2 = x_interval
        result = safe_cubic_opt(x1, x2, vg1, vg2)
        self.assertGreaterEqual(float(result), min(x1, x2))
        self.assertLessEqual(float(result), max(x1, x2))


if __name__ == '__main__':
    absltest.main()
