import math
import unittest

from exprft.weighting import effective_sample_size, exprft_weights, standardize_per_context


class TestStandardize(unittest.TestCase):
    def test_constant_context_maps_to_zero(self):
        self.assertEqual(standardize_per_context([0.8, 0.8, 0.8], ["x"] * 3), [0.0, 0.0, 0.0])

    def test_population_std_with_floor(self):
        r = [0.0, 1.0]  # mu=0.5, pstd=0.5 > floor
        self.assertEqual(standardize_per_context(r, ["x", "x"]), [-1.0, 1.0])
        r = [0.50, 0.52]  # pstd=0.01 < floor 0.1 -> divide by 0.1
        z = standardize_per_context(r, ["x", "x"])
        self.assertAlmostEqual(z[0], -0.1)
        self.assertAlmostEqual(z[1], 0.1)

    def test_contexts_are_independent(self):
        z = standardize_per_context([0.0, 1.0, 0.9], ["a", "a", "b"])
        self.assertEqual(z[:2], [-1.0, 1.0])
        self.assertEqual(z[2], 0.0)


class TestExpRFT(unittest.TestCase):
    def test_weights_positive_and_ordered(self):
        w = exprft_weights([0.2, 0.9, 0.5], ["x"] * 3, beta=0.5)
        self.assertTrue(all(v > 0 for v in w))
        self.assertLess(w[0], w[2])
        self.assertLess(w[2], w[1])

    def test_large_beta_is_uniform(self):
        w = exprft_weights([0.2, 0.9, 0.5], ["x"] * 3, beta=1e9)
        for v in w:
            self.assertAlmostEqual(v, 1.0, places=6)

    def test_exact_values(self):
        w = exprft_weights([0.0, 1.0], ["x", "x"], beta=1.0)
        self.assertAlmostEqual(w[0], math.exp(-1.0))
        self.assertAlmostEqual(w[1], math.exp(1.0))

    def test_beta_must_be_positive(self):
        with self.assertRaises(ValueError):
            exprft_weights([0.1], ["x"], beta=0.0)

    def test_ess(self):
        self.assertAlmostEqual(effective_sample_size([1.0, 1.0, 1.0]), 3.0)
        self.assertAlmostEqual(effective_sample_size([1.0, 0.0, 0.0]), 1.0)


if __name__ == "__main__":
    unittest.main()
