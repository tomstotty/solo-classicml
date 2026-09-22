"""Regression tests for classicml.root_mean_squared_log_error.

Discoverable with ``python -m unittest discover``. Standard library only.
"""

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


class RootMeanSquaredLogErrorTest(unittest.TestCase):
    def test_exported_in_all(self):
        self.assertIn("root_mean_squared_log_error", classicml.__all__)
        self.assertIs(
            classicml.root_mean_squared_log_error,
            root_mean_squared_log_error,
        )

    def test_unweighted_matches_sqrt_of_msle(self):
        y_true = [1.0, 2.5, 3, 0.0]
        y_pred = [1.1, 2.0, 3, 0.5]
        expected = math.sqrt(mean_squared_log_error(y_true, y_pred))
        result = root_mean_squared_log_error(y_true, y_pred)
        self.assertIs(type(result), float)
        self.assertEqual(result, expected)

    def test_unweighted_manual_value(self):
        # e = log1p(3) - log1p(0) = log(4); RMSLE of a single sample is
        # sqrt(e ** 2) == abs(e).
        result = root_mean_squared_log_error([0.0], [3.0])
        self.assertEqual(result, math.log(4.0))

    def test_weighted_matches_sqrt_of_msle(self):
        y_true = [1, 2, 3, 4]
        y_pred = [2, 2, 5, 3]
        weights = [0.5, 1, 2, 0.25]
        expected = math.sqrt(
            mean_squared_log_error(y_true, y_pred, weights)
        )
        self.assertEqual(
            root_mean_squared_log_error(y_true, y_pred, weights), expected
        )

    def test_weighted_manual_value(self):
        # Two samples, weights 1 and 3: m = (e1**2 + 3*e2**2) / 4.
        y_true = [0.0, 1.0]
        y_pred = [3.0, 7.0]
        weights = [1, 3]
        e1 = math.log1p(3.0) - math.log1p(0.0)
        e2 = math.log1p(7.0) - math.log1p(1.0)
        expected = math.sqrt((e1 ** 2 + 3.0 * e2 ** 2) / 4.0)
        self.assertEqual(
            root_mean_squared_log_error(y_true, y_pred, weights), expected
        )

    def test_zero_error_returns_positive_zero(self):
        for args in (
            ([1.0, 2.0], [1.0, 2.0], None),
            ([0, 5], [0, 5], [2, 3]),
            ([0.0], [0.0], [1.0]),
        ):
            result = root_mean_squared_log_error(*args)
            self.assertIs(type(result), float)
            self.assertEqual(result, 0.0)
            self.assertEqual(math.copysign(1.0, result), 1.0)

    def test_invalid_containers_raise_value_error(self):
        bad_calls = [
            ((1.0, 2.0), [1.0, 2.0], None),  # tuple, not list
            ("12", [1.0, 2.0], None),
            (None, [1.0], None),
            ([], [], None),  # empty
            ([1.0], [1.0, 2.0], None),  # length mismatch
            ([1.0], [1.0], "not a list"),
            ([1.0], [1.0], [1.0, 2.0]),  # weight length mismatch
        ]
        for y_true, y_pred, weights in bad_calls:
            with self.assertRaises(ValueError):
                root_mean_squared_log_error(y_true, y_pred, weights)

    def test_invalid_element_types_raise_value_error(self):
        bad_calls = [
            ([True, 1.0], [1.0, 1.0], None),  # bool rejected
            ([1.0, 1.0], [1.0, False], None),
            (["1", 1.0], [1.0, 1.0], None),
            ([1.0, 1.0], [1.0, None], None),
            ([1.0, 1.0], [1.0, 1.0], [True, 1.0]),
            ([1.0, 1.0], [1.0, 1.0], [1.0, "x"]),
        ]
        for y_true, y_pred, weights in bad_calls:
            with self.assertRaises(ValueError):
                root_mean_squared_log_error(y_true, y_pred, weights)

    def test_negative_and_non_finite_raise_value_error(self):
        bad_calls = [
            ([-1.0, 1.0], [1.0, 1.0], None),
            ([1.0, 1.0], [-1, 1.0], None),
            ([float("nan"), 1.0], [1.0, 1.0], None),
            ([1.0, 1.0], [float("inf"), 1.0], None),
            ([float("-inf"), 1.0], [1.0, 1.0], None),
            ([10 ** 400, 1.0], [1.0, 1.0], None),  # int too large: finite check overflows
            ([1.0, 1.0], [1.0, 1.0], [1.0, -0.5]),
            ([1.0, 1.0], [1.0, 1.0], [float("nan"), 1.0]),
            ([1.0, 1.0], [1.0, 1.0], [float("inf"), 1.0]),
        ]
        for y_true, y_pred, weights in bad_calls:
            with self.assertRaises(ValueError):
                root_mean_squared_log_error(y_true, y_pred, weights)

    def test_non_positive_total_weight_raises_value_error(self):
        with self.assertRaises(ValueError):
            root_mean_squared_log_error([1.0, 2.0], [1.0, 2.0], [0.0, 0.0])
        with self.assertRaises(ValueError):
            root_mean_squared_log_error([1.0], [1.0], [0.0])

    def test_arithmetic_overflow_raises_floating_point_error(self):
        # log1p(1e308) is finite, but weight 1e308 times the squared
        # log-error overflows to a non-finite term inside
        # mean_squared_log_error; the FloatingPointError propagates.
        with self.assertRaises(FloatingPointError):
            root_mean_squared_log_error([0.0], [1e308], [1e308])

    def test_underlying_value_error_propagates_unchanged(self):
        # Same validation path as mean_squared_log_error: identical
        # exception type and message.
        with self.assertRaises(ValueError) as ctx_rmsle:
            root_mean_squared_log_error([1.0], [-2.0])
        with self.assertRaises(ValueError) as ctx_msle:
            mean_squared_log_error([1.0], [-2.0])
        self.assertEqual(str(ctx_rmsle.exception), str(ctx_msle.exception))

    def test_inputs_are_not_modified(self):
        y_true = [1, 2.5, 3.0]
        y_pred = [1.5, 2.0, 3]
        weights = [0.5, 1, 2.0]
        snapshot = (list(y_true), list(y_pred), list(weights))
        root_mean_squared_log_error(y_true, y_pred, weights)
        self.assertEqual([y_true, y_pred, weights], list(snapshot))
        # Also on the failure paths.
        bad_true = [1.0, -1.0]
        bad_pred = [1.0, 2.0]
        bad_weights = [0.0, 0.0]
        for args in (
            (bad_true, bad_pred, None),
            ([1.0, 2.0], [1.0, 2.0], bad_weights),
        ):
            with self.assertRaises(ValueError):
                root_mean_squared_log_error(*args)
        self.assertEqual(bad_true, [1.0, -1.0])
        self.assertEqual(bad_pred, [1.0, 2.0])
        self.assertEqual(bad_weights, [0.0, 0.0])

    def test_deterministic(self):
        y_true = [3, 0.5, 10.0, 2]
        y_pred = [2.5, 1, 9.0, 2]
        weights = [1, 0.25, 2, 3.0]
        first = root_mean_squared_log_error(y_true, y_pred, weights)
        for _ in range(5):
            self.assertEqual(
                root_mean_squared_log_error(y_true, y_pred, weights), first
            )
        self.assertEqual(
            root_mean_squared_log_error(y_true, y_pred),
            root_mean_squared_log_error(y_true, y_pred),
        )


if __name__ == "__main__":
    unittest.main()
