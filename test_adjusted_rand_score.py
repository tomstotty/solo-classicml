"""Regression tests for classicml.adjusted_rand_score.

Discoverable with ``python -m unittest discover``. Standard library only.
"""

import copy
import math
import unittest
from fractions import Fraction

import classicml
from classicml import adjusted_rand_score


class AdjustedRandScoreTests(unittest.TestCase):
    def test_exported_in_all(self):
        self.assertIn("adjusted_rand_score", classicml.__all__)
        self.assertIs(classicml.adjusted_rand_score, adjusted_rand_score)

    def test_returns_float(self):
        result = adjusted_rand_score([0, 0, 1, 1], [0, 0, 1, 1])
        self.assertIs(type(result), float)

    def test_identical_partition_is_one(self):
        labels_true = [0, 0, 1, 1, 2, 2]
        labels_pred = [0, 0, 1, 1, 2, 2]
        self.assertEqual(adjusted_rand_score(labels_true, labels_pred), 1.0)

    def test_label_renaming_same_partition_is_one(self):
        # Same partitions, arbitrary (including negative) renamings.
        labels_true = [0, 1, 0, 1, 1, 2]
        labels_pred = [9, 4, 9, 4, 4, -7]
        self.assertEqual(adjusted_rand_score(labels_true, labels_pred), 1.0)
        # Rows and columns sorted independently: permuting the predicted
        # label names must not change the result.
        labels_permuted = [-7, 9, -7, 9, 9, 4]
        self.assertEqual(
            adjusted_rand_score(labels_true, labels_permuted), 1.0
        )

    def test_ordinary_partition(self):
        # Contingency [[2,1,0],[0,1,2]] with n=6:
        # I=2, A=6, B=3, N=15, E=6/5, M=9/2, ARI=(4/5)/(33/10)=8/33.
        labels_true = [0, 0, 0, 1, 1, 1]
        labels_pred = [0, 0, 1, 1, 2, 2]
        expected = float(Fraction(8, 33))
        result = adjusted_rand_score(labels_true, labels_pred)
        self.assertTrue(math.isfinite(result))
        self.assertEqual(result, expected)
        self.assertAlmostEqual(result, 0.24242424242424246)

    def test_disagreement_is_negative_half(self):
        # Two clusters of two, perfectly crossed: [[1,1],[1,1]]:
        # I=0, A=B=2, N=6, E=2/3, M=2, ARI=-1/2.
        result = adjusted_rand_score([0, 0, 1, 1], [0, 1, 0, 1])
        self.assertEqual(result, float(Fraction(-1, 2)))
        self.assertEqual(result, -0.5)

    def test_single_cluster_both_sides_is_one(self):
        self.assertEqual(
            adjusted_rand_score([0, 0, 0, 0], [5, 5, 5, 5]), 1.0
        )

    def test_one_sample_is_one(self):
        self.assertEqual(adjusted_rand_score([0], [7]), 1.0)

    def test_single_cluster_vs_split_is_zero(self):
        # [[2,2]]: I=2, A=6, B=2, N=6, E=2, M=4 -> ARI=0 (exact -> 0.0).
        result = adjusted_rand_score([0, 0, 0, 0], [0, 1, 0, 1])
        self.assertIs(type(result), float)
        self.assertEqual(result, 0.0)
        # Symmetric: one cluster on the other side.
        result_sym = adjusted_rand_score([0, 1, 0, 1], [0, 0, 0, 0])
        self.assertEqual(result_sym, 0.0)

    def test_all_singletons_matching_is_one(self):
        labels_true = [0, 1, 2, 3]
        labels_pred = [0, 1, 2, 3]
        self.assertEqual(adjusted_rand_score(labels_true, labels_pred), 1.0)

    def test_all_singletons_renamed_is_one(self):
        # Same partition into singletons under a permutation of names.
        labels_true = [0, 1, 2, 3]
        labels_pred = [30, 10, 40, 20]
        self.assertEqual(adjusted_rand_score(labels_true, labels_pred), 1.0)

    def test_exact_zero_normalized_to_float_zero(self):
        result = adjusted_rand_score([0, 0, 0, 0], [1, 2, 1, 2])
        self.assertEqual(result, 0.0)
        self.assertEqual(math.copysign(1.0, result), 1.0)

    def test_deterministic_same_inputs_same_result(self):
        labels_true = [0, 0, 0, 1, 1, 1, 2, 2]
        labels_pred = [0, 1, 0, 1, 2, 2, 2, 0]
        first = adjusted_rand_score(labels_true, labels_pred)
        second = adjusted_rand_score(copy.deepcopy(labels_true),
                                     copy.deepcopy(labels_pred))
        self.assertEqual(first, second)

    def test_inputs_not_modified(self):
        labels_true = [2, 0, 2, 1, 0, 1]
        labels_pred = [-1, -1, 3, 3, 0, 0]
        true_snapshot = copy.deepcopy(labels_true)
        pred_snapshot = copy.deepcopy(labels_pred)
        adjusted_rand_score(labels_true, labels_pred)
        self.assertEqual(labels_true, true_snapshot)
        self.assertEqual(labels_pred, pred_snapshot)

    def test_invalid_inputs(self):
        valid = [0, 1, 1]

        def assert_value_error(true, pred):
            with self.assertRaises(ValueError):
                adjusted_rand_score(true, pred)

        # Not lists.
        assert_value_error((0, 1, 1), [0, 1, 1])
        assert_value_error([0, 1, 1], (0, 1, 1))
        assert_value_error("011", [0, 1, 1])
        # Empty lists.
        assert_value_error([], [0])
        assert_value_error([0], [])
        assert_value_error([], [])
        # Unequal lengths.
        assert_value_error([0, 1], [0, 1, 0])
        # Elements whose type is not exactly int.
        assert_value_error([0, 1.0, 1], [0, 1, 1])
        assert_value_error([0, 1, 1], [0, True, 1])
        assert_value_error([False, 1, 1], [0, 1, 1])
        assert_value_error([0, "1", 1], [0, 1, 1])
        assert_value_error([0, None, 1], [0, 1, 1])
        assert_value_error([0, 1, 1], [0, 1, 1.0])
        # Valid control case does not raise.
        adjusted_rand_score(valid, [0, 0, 1])


if __name__ == "__main__":
    unittest.main()
