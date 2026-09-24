"""Regression tests for classicml.LassoRegression.

Discoverable with ``python -m unittest discover``. Standard library only.
"""

import copy
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
from classicml import LassoRegression


def _reference_fit(X, y, alpha, max_iter, tol):
    """Independent oracle implementing the specification literally."""
    n = len(X)
    width = len(X[0])
    w = [0.0] * width
    b = math.fsum(y) / n
    rounds = 0
    for _ in range(max_iter):
        rounds += 1
        previous_b = b
        b = math.fsum(
            y[i] - math.fsum(w[j] * X[i][j] for j in range(width))
            for i in range(n)
        ) / n
        max_change = abs(b - previous_b)
        for j in range(width):
            previous_w = w[j]
            r = math.fsum(
                X[i][j]
                * (
                    y[i]
                    - b
                    - math.fsum(
                        w[k] * X[i][k] for k in range(width) if k != j
                    )
                )
                for i in range(n)
            ) / n
            z = math.fsum(X[i][j] ** 2 for i in range(n)) / n
            if z == 0:
                updated = 0.0
            else:
                if r > alpha:
                    s = r - alpha
                elif r < -alpha:
                    s = r + alpha
                else:
                    s = 0.0
                updated = s / z
            w[j] = updated
            change = abs(updated - previous_w)
            if change > max_change:
                max_change = change
        if max_change <= tol:
            break
    return w, b, rounds


def _reference_predict(row, w, b):
    value = math.fsum(w[j] * row[j] for j in range(len(w))) + b
    return 0.0 if value == 0 else value


class LassoRegressionTest(unittest.TestCase):
    def test_exported_in_all(self):
        self.assertIn("LassoRegression", classicml.__all__)
        self.assertIs(classicml.LassoRegression, LassoRegression)

    def test_default_parameters(self):
        model = LassoRegression()
        self.assertEqual(model.alpha, 1.0)
        self.assertEqual(model.max_iter, 1000)
        self.assertEqual(model.tol, 1e-8)
        self.assertIsNone(model.w)
        self.assertIsNone(model.b)

    def test_fit_returns_self_and_predict_is_list_of_floats(self):
        X = [[0.0, 1.0], [1.0, 0.0], [2.0, 1.0], [3.0, 0.0]]
        y = [1.0, 2.0, 3.0, 4.0]
        model = LassoRegression()
        self.assertIs(model.fit(X, y), model)
        predictions = model.predict(X)
        self.assertIsInstance(predictions, list)
        self.assertEqual(len(predictions), len(X))
        self.assertTrue(all(type(value) is float for value in predictions))

    def test_perfect_univariate_line_with_tiny_penalty(self):
        X = [[0.0], [1.0], [2.0], [3.0]]
        y = [1.0, 3.0, 5.0, 7.0]
        model = LassoRegression(alpha=1e-10, max_iter=5000, tol=1e-12)
        model.fit(X, y)
        self.assertAlmostEqual(model.w[0], 2.0, places=9)
        self.assertAlmostEqual(model.b, 1.0, places=9)
        for predicted, expected in zip(model.predict(X), y):
            self.assertAlmostEqual(predicted, expected, places=9)

    def test_zero_variance_column_gets_zero_weight(self):
        # The constant column has z == 0, so its weight must stay exactly
        # 0.0; the other feature and the intercept absorb the signal.
        X = [[5.0, 0.0], [5.0, 1.0], [5.0, 2.0]]
        y = [2.0, 4.0, 6.0]
        model = LassoRegression(alpha=1e-12, max_iter=5000, tol=1e-13).fit(X, y)
        self.assertEqual(model.w[0], 0.0)
        self.assertAlmostEqual(model.w[1], 2.0, places=9)
        self.assertAlmostEqual(model.b, 2.0, places=9)

    def test_penalty_shrinks_weight_toward_zero(self):
        X = [[0.0], [1.0], [2.0], [3.0], [4.0]]
        y = [0.0, 2.0, 4.0, 6.0, 8.0]
        weak = LassoRegression(alpha=0.1, max_iter=5000, tol=1e-12).fit(X, y)
        strong = LassoRegression(alpha=5.0, max_iter=5000, tol=1e-12).fit(X, y)
        self.assertGreater(weak.w[0], strong.w[0])
        self.assertGreaterEqual(strong.w[0], 0.0)
        # A penalty above the unpenalized correlation zeroes the feature.
        killed = LassoRegression(alpha=100.0).fit(X, y)
        self.assertEqual(killed.w[0], 0.0)
        # With a null weight every prediction is the intercept.
        self.assertEqual(killed.predict(X), [4.0] * len(X))

    def test_matches_reference_oracle_on_random_problems(self):
        rng = random.Random(4242)
        for trial in range(20):
            n = rng.randint(2, 12)
            width = rng.randint(1, 5)
            X = [
                [rng.uniform(-3.0, 3.0) for _ in range(width)]
                for _ in range(n)
            ]
            true_w = [rng.uniform(-2.0, 2.0) for _ in range(width)]
            true_b = rng.uniform(-1.0, 1.0)
            y = [
                math.fsum(true_w[j] * X[i][j] for j in range(width)) + true_b
                for i in range(n)
            ]
            alpha = rng.choice([1e-9, 0.01, 0.5, 2.0])
            max_iter = rng.randint(1, 200)
            tol = 10.0 ** rng.randint(-12, -2)
            model = LassoRegression(
                alpha=alpha, max_iter=max_iter, tol=tol
            ).fit(X, y)
            ref_w, ref_b, _ = _reference_fit(X, y, alpha, max_iter, tol)
            self.assertEqual(model.w, ref_w)
            self.assertEqual(model.b, ref_b)
            queries = [
                [rng.uniform(-3.0, 3.0) for _ in range(width)]
                for _ in range(3)
            ]
            for row, predicted in zip(queries, model.predict(queries)):
                self.assertEqual(predicted, _reference_predict(row, ref_w, ref_b))

    def test_tol_stops_within_tolerance(self):
        X = [[0.0], [1.0], [2.0], [3.0]]
        y = [1.0, 2.0, 4.0, 3.0]
        model = LassoRegression(alpha=0.1, max_iter=10000, tol=0.25).fit(X, y)
        ref_w, ref_b, rounds = _reference_fit(X, y, 0.1, 10000, 0.25)
        self.assertEqual(model.w, ref_w)
        self.assertEqual(model.b, ref_b)
        self.assertLess(rounds, 10000)

    def test_at_most_max_iter_rounds(self):
        X = [[float(i)] for i in range(10)]
        y = [math.sin(i) for i in range(10)]
        model = LassoRegression(alpha=0.001, max_iter=3, tol=1e-300).fit(X, y)
        ref_w, ref_b, rounds = _reference_fit(X, y, 0.001, 3, 1e-300)
        self.assertEqual(rounds, 3)
        self.assertEqual(model.w, ref_w)
        self.assertEqual(model.b, ref_b)

    def test_integer_inputs_accepted(self):
        X = [[0, 0], [1, 2], [2, 3], [3, 6]]
        y = [1, 2, 3, 4]
        model = LassoRegression(alpha=1).fit(X, y)
        ref_w, ref_b, _ = _reference_fit(X, y, 1, 1000, 1e-8)
        self.assertEqual(model.w, ref_w)
        self.assertEqual(model.b, ref_b)
        self.assertTrue(all(type(v) is float for v in model.predict(X)))

    def test_does_not_mutate_inputs(self):
        X = [[0.0, 1.0], [1.0, 0.0], [2.0, 2.0]]
        y = [1.0, 2.0, 0.0]
        X_copy = copy.deepcopy(X)
        y_copy = copy.deepcopy(y)
        LassoRegression().fit(X, y)
        self.assertEqual(X, X_copy)
        self.assertEqual(y, y_copy)
        model = LassoRegression().fit(X, y)
        model.predict(X)
        self.assertEqual(X, X_copy)

    def test_deterministic(self):
        X = [[0.0, 1.0], [1.0, 0.0], [2.0, 1.0], [3.0, 2.0]]
        y = [1.0, 2.0, 3.0, 5.0]
        first = LassoRegression(alpha=0.3).fit(X, y).predict(X)
        second = LassoRegression(alpha=0.3).fit(X, y).predict(X)
        self.assertEqual(first, second)

    def test_prediction_zero_is_positive_zero(self):
        X = [[1.0], [-1.0]]
        y = [1.0, -1.0]
        model = LassoRegression(alpha=100.0).fit(X, y)
        self.assertEqual(model.w, [0.0])
        self.assertEqual(model.b, 0.0)
        predictions = model.predict([[4.0], [-9.0]])
        self.assertEqual(predictions, [0.0, 0.0])
        self.assertTrue(all(math.copysign(1.0, p) == 1.0 for p in predictions))

    def test_predict_before_fit_raises_value_error(self):
        with self.assertRaises(ValueError):
            LassoRegression().predict([[1.0]])

    def test_failed_fit_leaves_model_unfitted(self):
        model = LassoRegression()
        good_X = [[0.0], [1.0], [2.0]]
        model.fit(good_X, [0.0, 1.0, 2.0])
        with self.assertRaises(ValueError):
            model.fit([[0.0], [1.0]], [0.0])
        self.assertIsNone(model.w)
        self.assertIsNone(model.b)
        with self.assertRaises(ValueError):
            model.predict(good_X)

    def test_invalid_fit_matrices_raise_value_error(self):
        model = LassoRegression()
        bad_matrices = [
            [],
            [[]],
            [[1.0], [1.0, 2.0]],
            [[1.0, 2.0], [3.0]],
            "not a list",
            [["a"]],
            [[1.0], [float("nan")]],
            [[1.0], [float("inf")]],
            [[True], [False]],
        ]
        for bad in bad_matrices:
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    model.fit(bad, [0.0, 0.0])
        # An arbitrarily large exact int is structurally legal input;
        # its arithmetic overflow is a FloatingPointError instead.
        with self.assertRaises(FloatingPointError):
            model.fit([[1], [10 ** 400]], [0.0, 0.0])

    def test_invalid_targets_raise_value_error(self):
        X = [[0.0], [1.0]]
        with self.assertRaises(ValueError):
            LassoRegression().fit(X, [0.0])
        with self.assertRaises(ValueError):
            LassoRegression().fit(X, "01")
        with self.assertRaises(ValueError):
            LassoRegression().fit(X, [0, True])
        with self.assertRaises(ValueError):
            LassoRegression().fit(X, [0.0, float("nan")])
        # An arbitrarily large exact int is legal target input; only the
        # arithmetic overflow rejects the fit, as FloatingPointError.
        with self.assertRaises(FloatingPointError):
            LassoRegression().fit([[0], [1]], [0, 10 ** 400])

    def test_invalid_predict_matrix_raises_value_error(self):
        model = LassoRegression().fit([[0.0, 1.0], [1.0, 0.0]], [0.0, 1.0])
        with self.assertRaises(ValueError):
            model.predict([])
        with self.assertRaises(ValueError):
            model.predict([[0.0]])
        with self.assertRaises(ValueError):
            model.predict([[0.0, 1.0, 2.0]])
        with self.assertRaises(ValueError):
            model.predict([["x", 1.0]])
        with self.assertRaises(ValueError):
            model.predict([[1.0, True]])

    def test_invalid_alpha_raises_value_error(self):
        for alpha in (0, 0.0, -1, -0.5, True, False, "1.0", None,
                      float("inf"), float("-inf"), float("nan"), 1 + 0j):
            with self.subTest(alpha=alpha):
                with self.assertRaises(ValueError):
                    LassoRegression(alpha=alpha)

    def test_invalid_max_iter_raises_value_error(self):
        for max_iter in (0, -1, 1.0, True, False, "1000", None):
            with self.subTest(max_iter=max_iter):
                with self.assertRaises(ValueError):
                    LassoRegression(max_iter=max_iter)

    def test_invalid_tol_raises_value_error(self):
        for tol in (0, 0.0, -1e-8, True, False, "1e-8", None,
                    float("inf"), float("nan")):
            with self.subTest(tol=tol):
                with self.assertRaises(ValueError):
                    LassoRegression(tol=tol)

    def test_overflow_during_fit_raises_floating_point_error(self):
        X = [[1e308], [1e308]]
        y = [1.0, -1.0]
        with self.assertRaises(FloatingPointError):
            LassoRegression(alpha=1.0).fit(X, y)

    def test_non_finite_intercept_mean_raises_floating_point_error(self):
        # Huge finite inputs whose products overflow during the first
        # intercept recomputation.
        X = [[1e308], [-1e308]]
        y = [1e308, -1e308]
        with self.assertRaises(FloatingPointError):
            LassoRegression(alpha=1.0, max_iter=1).fit(X, y)

    def test_overflow_during_predict_raises_floating_point_error(self):
        X = [[0.0], [1.0], [2.0], [3.0]]
        y = [0.0, 3.0, 6.0, 9.0]
        model = LassoRegression(alpha=0.001, max_iter=5000, tol=1e-12).fit(X, y)
        self.assertGreater(model.w[0], 1.8)
        with self.assertRaises(FloatingPointError):
            model.predict([[1e308]])


if __name__ == "__main__":
    unittest.main()
