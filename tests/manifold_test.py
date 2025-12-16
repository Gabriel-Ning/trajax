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

"""Tests for manifold utilities (angle wrapping for S¹ manifolds)."""

from absl.testing import absltest
from absl.testing import parameterized

import jax
import jax.numpy as jnp
from jax import config
import numpy as np

from trajax.utils import _wrap_to_pi, get_s1_wrapper

config.update('jax_enable_x64', True)


class WrapToPiTest(parameterized.TestCase):
    """Tests for _wrap_to_pi angle wrapping function."""

    def test_wrap_zero(self):
        """Zero should remain zero."""
        result = _wrap_to_pi(jnp.array(0.0))
        self.assertAlmostEqual(float(result), 0.0, places=6)

    def test_wrap_pi(self):
        """π should wrap to -π."""
        result = _wrap_to_pi(jnp.array(jnp.pi))
        self.assertAlmostEqual(float(result), -jnp.pi, places=6)

    def test_wrap_neg_pi(self):
        """−π should remain −π."""
        result = _wrap_to_pi(jnp.array(-jnp.pi))
        self.assertAlmostEqual(float(result), -jnp.pi, places=6)

    @parameterized.parameters(
        (0.0,),
        (jnp.pi / 2,),
        (-jnp.pi / 2,),
        (jnp.pi / 4,),
        (-3 * jnp.pi / 4,),
    )
    def test_wrap_in_range(self, angle):
        """Angles already in [-π, π] should remain unchanged."""
        result = _wrap_to_pi(jnp.array(angle))
        self.assertTrue(-jnp.pi <= result <= jnp.pi)
        if angle > -jnp.pi and angle < jnp.pi:
            self.assertAlmostEqual(float(result), angle, places=6)

    @parameterized.parameters(
        (3.5 * jnp.pi,),
        (2.5 * jnp.pi,),
        (-2.5 * jnp.pi,),
        (10 * jnp.pi,),
        (-7 * jnp.pi,),
    )
    def test_wrap_out_of_range(self, angle):
        """Angles outside [-π, π] should be wrapped."""
        result = _wrap_to_pi(jnp.array(angle))
        self.assertTrue(-jnp.pi <= result <= jnp.pi,
                       f"Wrapped angle {result} not in [-π, π]")

    def test_wrap_array(self):
        """Should work on arrays of angles."""
        angles = jnp.array([0.0, jnp.pi, 3.5 * jnp.pi, -2.5 * jnp.pi])
        result = _wrap_to_pi(angles)
        self.assertEqual(result.shape, angles.shape)
        self.assertTrue(jnp.all(result >= -jnp.pi) and jnp.all(result <= jnp.pi))

    def test_wrap_jittable(self):
        """Wrapping should be JIT-compilable."""
        jitted_wrap = jax.jit(_wrap_to_pi)
        angle = jnp.array(3.5 * jnp.pi)
        result = jitted_wrap(angle)
        self.assertTrue(-jnp.pi <= result <= jnp.pi)


class GetS1WrapperTest(parameterized.TestCase):
    """Tests for get_s1_wrapper function."""

    def test_identity_wrapper_none(self):
        """get_s1_wrapper(None) should return identity function."""
        wrapper = get_s1_wrapper(None)
        x = jnp.array([1.0, 2.0, 3.0])
        result = wrapper(x)
        np.testing.assert_array_almost_equal(result, x)

    def test_single_index_wrapping(self):
        """Should wrap specified index to [-π, π]."""
        wrapper = get_s1_wrapper((0,))
        x = jnp.array([3.5 * jnp.pi, 2.0, 3.0])
        result = wrapper(x)

        # First element should be wrapped
        self.assertTrue(-jnp.pi <= result[0] <= jnp.pi)
        # Other elements unchanged
        self.assertAlmostEqual(float(result[1]), 2.0, places=6)
        self.assertAlmostEqual(float(result[2]), 3.0, places=6)

    def test_multiple_index_wrapping(self):
        """Should wrap multiple specified indices."""
        wrapper = get_s1_wrapper((0, 2))
        x = jnp.array([3.5 * jnp.pi, 2.0, 2.5 * jnp.pi, 4.0])
        result = wrapper(x)

        # Wrapped indices
        self.assertTrue(-jnp.pi <= result[0] <= jnp.pi)
        self.assertTrue(-jnp.pi <= result[2] <= jnp.pi)
        # Unchanged indices
        self.assertAlmostEqual(float(result[1]), 2.0, places=6)
        self.assertAlmostEqual(float(result[3]), 4.0, places=6)

    def test_wrapper_jittable(self):
        """Wrapper function should be JIT-compilable."""
        wrapper = get_s1_wrapper((0, 2))
        x = jnp.array([3.5 * jnp.pi, 2.0, 2.5 * jnp.pi])
        result = wrapper(x)
        self.assertEqual(result.shape, x.shape)

    def test_notebook_usage_pattern(self):
        """Test usage pattern from example notebooks."""
        # Setup like Car2D notebook
        s1_indices = (2,)
        state_wrap = get_s1_wrapper(s1_indices)

        x = jnp.array([1.75, 1.0, 3.5 * jnp.pi, 0.0])
        goal = jnp.array([3., 3., jnp.pi/2, 0.])

        # This is what the notebook does
        delta = state_wrap(x - goal)

        # Angle component should be in [-π, π]
        self.assertTrue(-jnp.pi <= delta[2] <= jnp.pi)
        # Other components should match unwrapped difference
        np.testing.assert_array_almost_equal(delta[:2], (x - goal)[:2])
        np.testing.assert_array_almost_equal(delta[3:], (x - goal)[3:])

    @parameterized.parameters(
        ((0,), jnp.array([3.5 * jnp.pi, 1.0, 2.0])),
        ((1,), jnp.array([1.0, 2.5 * jnp.pi, 3.0])),
        ((0, 1), jnp.array([3.5 * jnp.pi, 2.5 * jnp.pi, 3.0])),
        ((0, 2), jnp.array([3.5 * jnp.pi, 1.0, 2.5 * jnp.pi])),
    )
    def test_various_index_combinations(self, indices, x):
        """Test various combinations of indices."""
        wrapper = get_s1_wrapper(indices)
        result = wrapper(x)

        # All indices in indices should be wrapped
        for i in indices:
            self.assertTrue(-jnp.pi <= result[i] <= jnp.pi,
                          f"Index {i} not wrapped: {result[i]}")

    def test_empty_index_tuple(self):
        """Empty tuple should work like None (identity)."""
        wrapper_empty = get_s1_wrapper(())
        wrapper_none = get_s1_wrapper(None)

        x = jnp.array([1.0, 2.0, 3.0])
        result_empty = wrapper_empty(x)
        result_none = wrapper_none(x)

        np.testing.assert_array_almost_equal(result_empty, x)
        np.testing.assert_array_almost_equal(result_none, x)


if __name__ == '__main__':
    absltest.main()
