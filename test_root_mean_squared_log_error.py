"""Regression tests for classicml.root_mean_squared_log_error.

Discoverable with ``python -m unittest discover``. Standard library only.
"""

import copy
import math
import os
import shutil
import sys
import unittest

# ``python -m unittest discover`` writes bytecode caches by default; this
# project must leave no ``__pycache__``/``.pyc`` artifacts behind. The flag
# covers modules imported below (it takes effect before ``import
# classicml``), while this module's own cache is written by the loader
# before this body runs, so remove it explicitly.
sys.dont_write_bytecode = True
_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "__pycache__")
shutil.rmtree(_CACHE_DIR, ignore_errors=True)

import classicml
from classicml import mean_squared_log_error, root_mean_squared_log_error


def _reference_rmsle(y_true, y_pred, sample_weight=None):
    """Independent oracle: sqrt of a from-scratch weighted MSLE."""
    n = len(y_true)
    if sample_weight is None:
        weights = [1.0] * n
    else:
        weights = [float(v) for v in sample_weight]
    total_weight = math.fsum(weights)
    terms = []
    for i in range(n):
        e = math.log1p(float(y_pred[i])) - math.log1p(float(y_true[i]))
        terms.append(weights[i] * e ** 2)
    return math.sqrt(math.fsum(terms) / total_weight)


class RootMeanSquaredLogErrorTest(unittest.TestCase):
    def test_exported_in_all(self):
        self.assertIn("root_mean_squared_log_error", classicml.__all__)
        self.assertIs(
            classicml.root_mean_squared_log_error, root_mean_squared_log_error
        )

    def test_unweighted_matches_sqrt_of_msle(self):
        y_true = [1, 2.5, 3, 10.0, 0]
        y_pred = [1.5, 2, 3.5, 9, 0.25]
        expected = math.sqrt(mean_squared_log_error(y_true, y_pred))
        result = root_mean_squared_log_error(y_true, y_pred)
        self.assertIs(type(result), float)
        self.assertEqual(result, expected)
        self.assertAlmostEqual(result, _reference_rmsle(y_true, y_pred))

    def test_weighted_matches_sqrt_of_msle(self):
        y_true = [1, 2.5, 3, 10.0, 0]
        y_pred = [1.5, 2, 3.5, 9, 0.25]
        weights = [0.5, 2, 1.5, 3.0, 1]
        expected = math.sqrt(mean_squared_log_error(y_true, y_pred, weights))
        result = root_mean_squared_log_error(y_true, y_pred, weights)
        self.assertIs(type(result), float)
        self.assertEqual(result, expected)
        self.assertAlmostEqual(
            result, _reference_rmsle(y_true, y_pred, weights)
        )

    def test_known_value(self):
        # log1p(3) - log1p(0) = log(4); RMSLE of a single sample is
        # |log(4)| regardless of the (positive) weight.
        self.assertEqual(
            root_mean_squared_log_error([0], [3]), math.sqrt(math.log(4) ** 2)
        )
        self.assertEqual(
            root_mean_squared_log_error([0], [3], [7.5]),
            math.sqrt(math.log(4) ** 2),
        )

    def test_zero_error_returns_positive_zero(self):
        for args in (
            ([1, 2.5, 3], [1, 2.5, 3]),
            ([1, 2.5, 3], [1, 2.5, 3], [0.5, 2, 1.5]),
            ([0.0], [0.0]),
        ):
            result = root_mean_squared_log_error(*args)
            self.assertIs(type(result), float)
            self.assertEqual(result, 0.0)
            self.assertEqual(math.copysign(1.0, result), 1.0)

    def test_invalid_inputs_raise_value_error(self):
        bad_calls = [
            # containers
            (({"a": 1}, [1, 2]),),
            ((None, [1, 2]),),
            (([1, 2], None),),
            (((1, 2), [1, 2]),),
            (([1, 2], (1, 2)),),
            (("12", [1, 2]),),
            # empty / unequal length
            (([], []),),
            (([1, 2], [1]),),
            (([1], [1, 2]),),
            # booleans
            (([True, 2], [1, 2]),),
            (([1, 2], [1, False]),),
            # wrong element types
            (([1, "2"], [1, 2]),),
            (([1, 2], [1, None]),),
            # negative / non-finite values
            (([-1, 2], [1, 2]),),
            (([1, 2], [1, -0.5]),),
            (([1, float("nan")], [1, 2]),),
            (([1, 2], [1, float("inf")]),),
            (([1, float("-inf")], [1, 2]),),
            # int too large to convert to float
            (([10 ** 1000], [1]),),
            # sample_weight container / length
            (([1, 2], [1, 2], (1, 1)),),
            (([1, 2], [1, 2], [1]),),
            (([1, 2], [1, 2], [1, 1, 1]),),
            # sample_weight element rules
            (([1, 2], [1, 2], [True, 1]),),
            (([1, 2], [1, 2], [1, "1"]),),
            (([1, 2], [1, 2], [1, -1]),),
            (([1, 2], [1, 2], [1, float("nan")]),),
            (([1, 2], [1, 2], [1, float("inf")]),),
            # total weight must be greater than zero
            (([1, 2], [1, 2], [0, 0]),),
            (([1, 2], [1, 2], [0.0, 0.0]),),
        ]
        for (args,) in bad_calls:
            with self.subTest(args=args):
                with self.assertRaises(ValueError):
                    root_mean_squared_log_error(*args)

    def test_underlying_value_error_propagates_unchanged(self):
        # The exact exception object raised by mean_squared_log_error
        # must surface from root_mean_squared_log_error untouched.
        try:
            mean_squared_log_error([1, 2], [1])
        except ValueError as exc:
            expected = exc
        try:
            root_mean_squared_log_error([1, 2], [1])
        except ValueError as exc:
            self.assertEqual(str(exc), str(expected))
            self.assertIs(type(exc), type(expected))
        else:
            self.fail("expected ValueError")

    def test_arithmetic_overflow_raises_floating_point_error(self):
        # w_i * e_i ** 2 overflows to infinity during the MSLE
        # arithmetic; the FloatingPointError propagates unchanged.
        with self.assertRaises(FloatingPointError):
            root_mean_squared_log_error([0.0], [1e308], [1e308])
        with self.assertRaises(FloatingPointError):
            root_mean_squared_log_error(
                [0.0, 0.0], [1e308, 1e308], [1e308, 1.0]
            )

    def test_inputs_are_not_modified(self):
        y_true = [1, 2.5, 3, 10.0, 0]
        y_pred = [1.5, 2, 3.5, 9, 0.25]
        weights = [0.5, 2, 1.5, 3.0, 1]
        snapshot = copy.deepcopy((y_true, y_pred, weights))
        root_mean_squared_log_error(y_true, y_pred, weights)
        self.assertEqual((y_true, y_pred, weights), snapshot)
        # Also on the failure paths.
        bad_true = [1, -2]
        bad_pred = [1, 2]
        bad_weights = [0.0, 0.0]
        with self.assertRaises(ValueError):
            root_mean_squared_log_error(bad_true, bad_pred, [1, 1])
        with self.assertRaises(ValueError):
            root_mean_squared_log_error([1, 2], [1, 2], bad_weights)
        with self.assertRaises(FloatingPointError):
            root_mean_squared_log_error([0.0], [1e308], [1e308])
        self.assertEqual(bad_true, [1, -2])
        self.assertEqual(bad_pred, [1, 2])
        self.assertEqual(bad_weights, [0.0, 0.0])

    def test_deterministic(self):
        y_true = [3, 0.5, 7, 2.25]
        y_pred = [2.5, 1, 6.75, 2]
        weights = [1, 0.25, 2.5, 3]
        first = root_mean_squared_log_error(y_true, y_pred, weights)
        for _ in range(10):
            self.assertEqual(
                root_mean_squared_log_error(y_true, y_pred, weights), first
            )
            self.assertEqual(
                root_mean_squared_log_error(y_true, y_pred),
                root_mean_squared_log_error(y_true, y_pred),
            )


if __name__ == "__main__":
    unittest.main()
