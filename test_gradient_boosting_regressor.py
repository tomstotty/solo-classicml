"""Regression tests for classicml.GradientBoostingRegressor.

Discoverable with ``python -m unittest discover``. Standard library only.
"""

import copy
import contextlib
import io
import math
import os
import random
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
from classicml import GradientBoostingRegressor


def _reference_fit(X, y, n_estimators, learning_rate, tol):
    """Independent oracle implementing the specification literally."""
    n = len(X)
    width = len(X[0])
    constant = math.fsum(y) / n
    values = [constant for _ in range(n)]
    saved = []
    for _ in range(n_estimators):
        residuals = [y[i] - values[i] for i in range(n)]
        best = None  # ((error, feature, threshold), left_mean, right_mean)
        for j in range(width):
            thresholds = sorted(set(row[j] for row in X))[:-1]
            for t in thresholds:
                membership = [X[i][j] <= t for i in range(n)]
                left = [residuals[i] for i in range(n) if membership[i]]
                right = [residuals[i] for i in range(n) if not membership[i]]
                left_mean = math.fsum(left) / len(left)
                right_mean = math.fsum(right) / len(right)
                error = math.fsum(
                    (
                        residuals[i]
                        - (left_mean if membership[i] else right_mean)
                    )
                    ** 2
                    for i in range(n)
                )
                key = (error, j, t)
                if best is None or key < best[0]:
                    best = (key, left_mean, right_mean)
        if best is None:
            break
        (_, feature, threshold), left_mean, right_mean = best
        increment_left = learning_rate * left_mean
        increment_right = learning_rate * right_mean
        saved.append((feature, threshold, increment_left, increment_right))
        for i in range(n):
            values[i] += (
                increment_left
                if X[i][feature] <= threshold
                else increment_right
            )
        if max(abs(increment_left), abs(increment_right)) <= tol:
            break
    return constant, saved


def _reference_predict(X, constant, saved):
    results = []
    for row in X:
        prediction = constant
        for feature, threshold, increment_left, increment_right in saved:
            prediction += (
                increment_left if row[feature] <= threshold else increment_right
            )
        results.append(prediction)
    return results


class GradientBoostingRegressorTest(unittest.TestCase):
    def test_exported_in_all(self):
        self.assertIn("GradientBoostingRegressor", classicml.__all__)
        self.assertIs(
            classicml.GradientBoostingRegressor, GradientBoostingRegressor
        )

    def test_default_parameters(self):
        model = GradientBoostingRegressor()
        self.assertEqual(model.n_estimators, 100)
        self.assertEqual(model.learning_rate, 0.1)
        self.assertEqual(model.tol, 1e-8)

    def test_fit_returns_self_and_predict_is_list_of_floats(self):
        X = [[0.0, 1.0], [1.0, 0.0], [2.0, 1.0], [3.0, 0.0]]
        y = [1.0, 2.0, 3.0, 4.0]
        model = GradientBoostingRegressor()
        self.assertIs(model.fit(X, y), model)
        predictions = model.predict(X)
        self.assertIsInstance(predictions, list)
        self.assertEqual(len(predictions), len(X))
        self.assertTrue(all(type(value) is float for value in predictions))

    def test_single_stump_known_values(self):
        # Two well-separated samples, one round with learning rate 1:
        # c = 1, residuals [-1, 1], threshold 0 gives means -1 and 1.
        X = [[0], [1]]
        y = [0.0, 2.0]
        model = GradientBoostingRegressor(
            n_estimators=1, learning_rate=1, tol=1e-12
        ).fit(X, y)
        self.assertEqual(model.predict([[0], [1]]), [0.0, 2.0])
        # Values equal to the threshold belong to the left side.
        self.assertEqual(model.predict([[0], [1], [2], [-3]]),
                         [0.0, 2.0, 2.0, 0.0])

    def test_constant_columns_retain_constant_model(self):
        X = [[7, 1], [7, 1], [7, 1]]
        y = [2.0, 4.0, 6.0]
        model = GradientBoostingRegressor().fit(X, y)
        # Every prediction is the mean c = 4.0 regardless of the query.
        self.assertEqual(model.predict([[7, 1], [0, 0], [9, 9]]),
                         [4.0, 4.0, 4.0])
        self.assertEqual(model.predict(X), [4.0, 4.0, 4.0])

    def test_single_sample_constant_prediction(self):
        X = [[3.0, -2.0]]
        y = [5.5]
        model = GradientBoostingRegressor().fit(X, y)
        self.assertEqual(model.predict(X), [5.5])
        self.assertEqual(model.predict([[100.0, 100.0]]), [5.5])

    def test_within_feature_threshold_tie_prefers_lower_threshold(self):
        # Sorted column values -5, -4, 2, 4 against targets 1, 3, 2, 4:
        # the stump SSE is 2.0 both at threshold -5 and at threshold 2.
        X = [[4], [2], [-5], [-4]]
        y = [4, 2, 1, 3]
        model = GradientBoostingRegressor(
            n_estimators=1, learning_rate=1, tol=1e-12
        ).fit(X, y)
        self.assertEqual(model.predict(X), [3.0, 3.0, 1.0, 3.0])
        self.assertEqual(model.predict([[-5], [-4], [2], [4]]),
                         [1.0, 3.0, 3.0, 3.0])

    def test_cross_feature_tie_prefers_lower_feature_index(self):
        # Both features attain the same minimum SSE 2.0; feature 0 wins.
        X = [[2, 0], [5, 2], [3, 4], [1, 2]]
        y = [0, -1, -2, -3]
        model = GradientBoostingRegressor(
            n_estimators=1, learning_rate=1, tol=1e-12
        ).fit(X, y)
        self.assertEqual(model.predict(X), [-1.0, -1.0, -1.0, -3.0])

    def test_stops_when_both_increments_within_tol(self):
        X = [[0.0], [1.0]]
        y = [0.0, 2.0]
        # With learning rate 0.1 the first increments are +/-0.1, already
        # within tol, so exactly one stump is saved and training stops.
        model = GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.1, tol=0.5
        ).fit(X, y)
        self.assertEqual(model.predict(X), [0.9, 1.1])

    def test_at_most_n_estimators_rounds(self):
        X = [[0.0], [1.0]]
        y = [0.0, 2.0]
        model = GradientBoostingRegressor(
            n_estimators=3, learning_rate=0.1, tol=1e-12
        ).fit(X, y)
        # Each round moves each prediction 10% toward its target, so the
        # residual shrinks by a factor 0.9: after three rounds the gap is
        # 0.9 ** 3 = 0.729 around the constant 1.0.
        gap = 1.0 - 0.9 ** 3
        predictions = model.predict(X)
        self.assertAlmostEqual(predictions[0], 1.0 - gap, places=12)
        self.assertAlmostEqual(predictions[1], 1.0 + gap, places=12)

    def test_matches_independent_oracle_randomized(self):
        rng = random.Random(20260923)
        pool = [0, 1, 2, 3, -1, -2, 0.5, 1.5, -0.25, 4, -3]
        for _ in range(120):
            n = rng.randint(1, 12)
            width = rng.randint(1, 4)
            X = [[rng.choice(pool) for _ in range(width)] for _ in range(n)]
            y = [rng.uniform(-5.0, 5.0) for _ in range(n)]
            n_estimators = rng.randint(1, 12)
            learning_rate = rng.choice((0.05, 0.1, 0.5, 1.0))
            tol = rng.choice((1e-12, 1e-8, 1e-3, 10.0))
            model = GradientBoostingRegressor(
                n_estimators=n_estimators, learning_rate=learning_rate, tol=tol
            ).fit(X, y)
            constant, saved = _reference_fit(
                X, y, n_estimators, learning_rate, tol
            )
            self.assertEqual(model.predict(X),
                             _reference_predict(X, constant, saved))
            query = [[rng.choice(pool) for _ in range(width)]
                     for _ in range(rng.randint(1, 5))]
            self.assertEqual(model.predict(query),
                             _reference_predict(query, constant, saved))

    def test_integer_inputs_produce_float_outputs(self):
        X = [[2, 0], [5, 2], [3, 4], [1, 2]]
        y = [0, -1, -2, -3]
        model = GradientBoostingRegressor().fit(X, y)
        predictions = model.predict(X)
        self.assertTrue(all(type(value) is float for value in predictions))

    def test_exact_zero_is_positive_zero(self):
        model = GradientBoostingRegressor().fit([[0], [1]], [0.0, -0.0])
        self.assertEqual(math.copysign(1.0, model.predict([[0]])[0]), 1.0)
        # A zero constant is likewise positive 0.0.
        self.assertEqual(math.copysign(1.0, model.predict([[1]])[0]), 1.0)

    def test_invalid_n_estimators_raises_value_error(self):
        for bad in (0, -1, 1.5, 1.0, True, False, "10", None, [10]):
            with self.assertRaises(ValueError):
                GradientBoostingRegressor(n_estimators=bad)

    def test_invalid_learning_rate_raises_value_error(self):
        for bad in (0, -0.1, float("inf"), float("-inf"), float("nan"),
                    True, False, "0.1", 1j, None):
            with self.assertRaises(ValueError):
                GradientBoostingRegressor(learning_rate=bad)

    def test_invalid_tol_raises_value_error(self):
        for bad in (0, -1e-8, float("inf"), float("-inf"), float("nan"),
                    True, "x", 1j, None):
            with self.assertRaises(ValueError):
                GradientBoostingRegressor(tol=bad)

    def test_invalid_fit_x_raises_value_error(self):
        y = [1.0, 2.0]
        for bad in ([], [[]], [[1.0], [2.0, 3.0]],
                    ([1.0], [2.0]), [(1.0,), (2.0,)],
                    [[1.0], ["x"]], [[1.0], [True]],
                    [[1.0], [float("nan")]], [[1.0], [float("inf")]],
                    None, "ab"):
            with self.assertRaises(ValueError):
                GradientBoostingRegressor().fit(bad, y)

    def test_invalid_fit_y_raises_value_error(self):
        X = [[1], [2]]
        for bad in ([1.0], [1.0, 2.0, 3.0], (1.0, 2.0),
                    [True, False], [1.0, float("nan")], [1.0, float("inf")],
                    ["a", "b"], None):
            with self.assertRaises(ValueError):
                GradientBoostingRegressor().fit(X, bad)

    def test_predict_before_fit_raises_value_error(self):
        with self.assertRaises(ValueError):
            GradientBoostingRegressor().predict([[0.0]])

    def test_invalid_predict_x_raises_value_error(self):
        model = GradientBoostingRegressor().fit(
            [[0.0, 1.0], [1.0, 0.0]], [1.0, 2.0]
        )
        for bad in ([], [[1.0]], [[1.0, 2.0, 3.0]],
                    [[1.0, float("nan")]], [[True, 1.0]], "xy"):
            with self.assertRaises(ValueError):
                model.predict(bad)

    def test_failed_fit_leaves_model_unfitted(self):
        X = [[0.0, 1.0], [1.0, 0.0]]
        y = [1.0, 2.0]
        model = GradientBoostingRegressor().fit(X, y)
        with self.assertRaises(ValueError):
            model.fit([[1.0], [float("inf")]], [1.0, 2.0])
        with self.assertRaises(ValueError):
            model.predict(X)
        model = GradientBoostingRegressor().fit(X, y)
        with self.assertRaises(FloatingPointError):
            model.fit([[1], [2]], [10 ** 308, 10 ** 308])
        with self.assertRaises(ValueError):
            model.predict(X)

    def test_oversized_integer_finiteness_check_raises_value_error(self):
        # An int too large to convert to float overflows the finiteness
        # check, which is still a rejected input (ValueError), both in the
        # constructor and in the data validators.
        with self.assertRaises(ValueError):
            GradientBoostingRegressor(learning_rate=10 ** 400)
        with self.assertRaises(ValueError):
            GradientBoostingRegressor(tol=10 ** 400)
        with self.assertRaises(ValueError):
            GradientBoostingRegressor().fit(
                [[10 ** 400], [1]], [1.0, 2.0]
            )
        with self.assertRaises(ValueError):
            GradientBoostingRegressor().fit(
                [[1], [2]], [10 ** 400, -(10 ** 400)]
            )
        fitted = GradientBoostingRegressor().fit(
            [[0.0], [1.0]], [1.0, 2.0]
        )
        with self.assertRaises(ValueError):
            fitted.predict([[10 ** 400]])

    def test_arithmetic_overflow_raises_floating_point_error(self):
        # Finite integer inputs whose sum cannot be represented: the
        # finiteness check passes, but the constant fsum overflows.
        with self.assertRaises(FloatingPointError):
            GradientBoostingRegressor().fit(
                [[1], [2]], [10 ** 308, 10 ** 308]
            )
        # learning_rate 2 applied to a maximal residual yields an infinite
        # first increment.
        with self.assertRaises(FloatingPointError):
            GradientBoostingRegressor(
                n_estimators=10, learning_rate=2.0, tol=1e-12
            ).fit([[0.0], [1.0]], [1e308, -1e308])
        # learning_rate 3 drives the predictions apart until they overflow.
        with self.assertRaises(FloatingPointError):
            GradientBoostingRegressor(
                n_estimators=2000, learning_rate=3.0, tol=1e-12
            ).fit([[0.0], [1.0]], [1e200, -1e200])

    def test_inputs_are_not_modified(self):
        X = [[0.0, 1.0], [1.0, 0.0], [2.0, 1.0]]
        y = [1.0, 2.0, 3.0]
        X_copy = copy.deepcopy(X)
        y_copy = copy.deepcopy(y)
        model = GradientBoostingRegressor(n_estimators=10).fit(X, y)
        model.predict(X)
        self.assertEqual(X, X_copy)
        self.assertEqual(y, y_copy)

    def test_deterministic_and_predict_pure(self):
        X = [[0.0, 1.0], [1.0, 0.0], [2.0, 2.0], [3.0, 0.0]]
        y = [1.0, 2.0, 3.0, 4.0]
        first = GradientBoostingRegressor(n_estimators=7).fit(X, y).predict(X)
        second = GradientBoostingRegressor(n_estimators=7).fit(X, y).predict(X)
        self.assertEqual(first, second)
        model = GradientBoostingRegressor(n_estimators=7).fit(X, y)
        self.assertEqual(model.predict(X), model.predict(X))

    def test_does_not_register_cli_entry_point(self):
        # The three CLI subcommands remain unchanged; the new model exposes
        # no new entry point. An unknown subcommand is rejected with code 2.
        import classicml as _module

        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(
                _module.main(["classicml.py", "gradient-boosting"]), 2
            )


if __name__ == "__main__":
    unittest.main()
