import unittest
import numpy as np
from pilot_runtime import components, corr, ranks


class Metrics(unittest.TestCase):
    def setUp(self):
        self.base = np.array([[2., 4.], [-2., -3.], [0., 2.]])

    def test_uniform_shift_does_not_restore_profile(self):
        m = components(self.base + np.array([7., -4.]), self.base)
        self.assertAlmostEqual(m['pattern_strength'], 1)
        self.assertAlmostEqual(m['residual_rms'], 0)

    def test_correlation_misses_scale(self):
        m = components(self.base * .01, self.base)
        self.assertAlmostEqual(m['pattern_strength'], .01)
        self.assertAlmostEqual(m['cosine'], 1)
        self.assertAlmostEqual(m['occupation_rank_corr'], 1)

    def test_zero_profile_has_no_direction(self):
        m = components(np.ones_like(self.base), self.base)
        self.assertEqual(m['pattern_strength'], 0)
        self.assertIsNone(m['cosine'])
        self.assertIsNone(m['occupation_rank_corr'])

    def test_tied_ranks(self):
        np.testing.assert_array_equal(ranks(np.array([4, 1, 1, 9])), [2, .5, .5, 3])

    def test_reference_degeneracy_fails(self):
        with self.assertRaises(ValueError):
            components(self.base, np.ones_like(self.base))


if __name__ == '__main__':
    unittest.main()
