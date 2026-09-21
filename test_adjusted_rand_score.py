"""Regression tests for classicml.adjusted_rand_score.

Discoverable with ``python -m unittest discover``. Standard library only.
"""

import copy
import itertools
import math
import os
import shutil
import sys
import unittest
from fractions import Fraction

# ``python -m unittest discover`` writes bytecode caches by default; this
# project must leave no ``__pycache__``/``.pyc`` artifacts behind. The flag
# covers modules imported below (it takes effect before ``import
# classicml``), while this module's own cache is written by the loader
# before this body runs, so remove it explicitly.
sys.dont_write_bytecode = True
_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "__pycache__")
shutil.rmtree(_CACHE_DIR, ignore_errors=True)

import classicml
from classicml import adjusted_rand_score


def _reference_ari(labels_true, labels_pred):
    """Independent Fraction oracle built by enumerating sample pairs."""
    n = len(labels_true)

    def comb2(x):
        return x * (x - 1) // 2

    same_true = 0
    same_pred = 0
    same_both = 0
    for i, j in itertools.combinations(range(n), 2):
        in_true = labels_true[i] == labels_true[j]
        in_pred = labels_pred[i] == labels_pred[j]
        same_true += int(in_true)
        same_pred += int(in_pred)
        same_both += int(in_true and in_pred)

    total = comb2(n)
    if total == 0:
        return 1.0
    expected = Fraction(same_true * same_pred, total)
    mean = Fraction(same_true + same_pred, 2)
    denominator = mean - expected
    if denominator == 0:
        return 1.0
    return float((Fraction(same_both) - expected) / denominator)


class AdjustedRandScoreTest(unittest.TestCase):
    def test_exported_in_all(self):
        self.assertIn("adjusted_rand_score", classicml.__all__)
        self.assertIs(classicml.adjusted_rand_score, adjusted_rand_score)

    def test_label_renaming_same_partition(self):
        # Same partition, different label names and order of appearance.
        self.assertEqual(
            adjusted_rand_score([0, 0, 1, 1], [5, 5, 7, 7]), 1.0
        )
        true = [0, 1, 0, 1, 2, 2, 1]
        pred = [9, 4, 9, 4, -2, -2, 4]
        self.assertEqual(adjusted_rand_score(true, pred), 1.0)
        # Negative, zero, and unordered label values must not matter.
        self.assertEqual(
            adjusted_rand_score([-3, -3, 10, 10], [2, 2, 0, 0]), 1.0
        )

    def test_ordinary_partitions(self):
        # Worked example: 8/33 exactly.
        result = adjusted_rand_score(
            [0, 0, 0, 1, 1, 1], [0, 0, 1, 1, 2, 2]
        )
        self.assertEqual(result, 8.0 / 33.0)
        self.assertAlmostEqual(result, 0.24242424242424246)

        # Balanced 2x2 checkerboard: agreement exactly at chance -> -1/2.
        self.assertEqual(
            adjusted_rand_score([0, 0, 1, 1], [0, 1, 0, 1]), -0.5
        )

    def test_single_cluster(self):
        # Both partitions put every sample in one cluster: indeterminate
        # denominator (M == E) -> 1.0.
        self.assertEqual(adjusted_rand_score([0, 0, 0, 0], [7, 7, 7, 7]), 1.0)
        # A single sample has no pairs (N == 0) -> 1.0.
        self.assertEqual(adjusted_rand_score([0], [5]), 1.0)
        # One side constant, the other split: the observed index equals
        # its expectation, so the score is an exact positive-sign 0.0.
        result = adjusted_rand_score([0, 0, 0, 0], [0, 0, 1, 1])
        self.assertEqual(result, 0.0)
        self.assertEqual(math.copysign(1.0, result), 1.0)

    def test_all_singletons(self):
        # Every point its own cluster in both partitions -> 1.0, and
        # renaming the singleton labels changes nothing.
        self.assertEqual(
            adjusted_rand_score([0, 1, 2, 3, 4], [9, 8, 7, 6, 5]), 1.0
        )
        # Singletons on one side versus a real split on the other gives
        # an exact zero.
        result = adjusted_rand_score([0, 1, 2, 3], [0, 0, 1, 1])
        self.assertEqual(result, 0.0)
        self.assertEqual(math.copysign(1.0, result), 1.0)

    def test_invalid_inputs_raise_value_error(self):
        invalid_pairs = [
            ([], []),  # empty
            ([0, 1], [0]),  # unequal length
            ([0], [0, 1]),
            ((0, 1), (0, 1)),  # tuples, not lists
            ("ab", "ab"),  # strings are not lists of ints
            ([0, 1], None),
            (None, [0, 1]),
            ([True, 0], [0, 0]),  # bool is not a legal element
            ([0, 1], [False, True]),
            ([0, 1.0], [0, 0]),  # float element
            ([0, 1], [0, "1"]),  # str element
            ([0, None], [0, 0]),
        ]
        for true, pred in invalid_pairs:
            with self.subTest(true=true, pred=pred):
                with self.assertRaises(ValueError):
                    adjusted_rand_score(true, pred)

        # Subclasses of int fail the exact-type requirement as well.
        class MyInt(int):
            pass

        with self.assertRaises(ValueError):
            adjusted_rand_score([MyInt(0), 1], [0, 1])
        with self.assertRaises(ValueError):
            adjusted_rand_score([0, 1], [0, MyInt(1)])

    def test_inputs_not_modified(self):
        true = [3, -1, 3, 0, -1, 0]
        pred = [2, 2, 9, 2, 9, 9]
        true_before = copy.deepcopy(true)
        pred_before = copy.deepcopy(pred)
        adjusted_rand_score(true, pred)
        self.assertEqual(true, true_before)
        self.assertEqual(pred, pred_before)

        # Invalid input must not mutate the lists either.
        messy_true = [0, 1, 0]
        messy_pred = [True, 0, 1]
        snapshots = (copy.deepcopy(messy_true), copy.deepcopy(messy_pred))
        with self.assertRaises(ValueError):
            adjusted_rand_score(messy_true, messy_pred)
        self.assertEqual(messy_true, snapshots[0])
        self.assertEqual(messy_pred, snapshots[1])

    def test_returns_float_and_deterministic(self):
        true = [0, 0, 1, 1, 2, 2]
        pred = [0, 1, 0, 1, 0, 2]
        first = adjusted_rand_score(true, pred)
        self.assertIsInstance(first, float)
        self.assertEqual(first, adjusted_rand_score(true, pred))
        self.assertEqual(first, _reference_ari(true, pred))

    def test_matches_pair_enumeration_oracle(self):
        cases = [
            ([0, 0, 1, 1], [0, 0, 1, 1]),
            ([0, 0, 0, 1, 1, 1], [0, 0, 1, 1, 2, 2]),
            ([0, 1, 2, 3, 4, 5], [0, 0, 0, 1, 1, 1]),
            ([0, 0, 0, 0], [0, 1, 2, 3]),
            ([0, 1, 0, 1, 2, 2, 2], [1, 1, 0, 2, 0, 2, 0]),
            ([-1, -1, 2, 2, 2, 0], [3, 3, 3, 1, 1, 1]),
            ([5], [5]),
        ]
        for true, pred in cases:
            with self.subTest(true=true, pred=pred):
                self.assertEqual(
                    adjusted_rand_score(true, pred),
                    _reference_ari(true, pred),
                )


if __name__ == "__main__":
    unittest.main()
