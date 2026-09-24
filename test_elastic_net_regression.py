"""Regression tests for classicml.ElasticNetRegression.

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
from classicml import ElasticNetRegression, LassoRegression


def _reference_fit(X, y, alpha, l1_ratio, max_iter, tol):
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
            z = math.fsum(X[i][j] ** 2 for i in range(n)) / n + alpha * (
                1 - l1_ratio
            )
            q = alpha * l1_ratio
            if z == 0:
                updated = 0.0
            elif r > q:
                updated = (r - q) / z
            elif r < -q:
                updated = (r + q) / z
            else:
                updated = 0.0
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


class ElasticNetRegressionTest(unittest.TestCase):
    def test_exported_in_all(self):
        self.assertIn("ElasticNetRegression", classicml.__all__)
        self.assertIs(
            classicml.ElasticNetRegression, ElasticNetRegression
        )

    def test_default_parameters(self):
        model = ElasticNetRegression()
        self.assertEqual(model.alpha, 1.0)
        self.assertEqual(model.l1_ratio, 0.5)
        self.assertEqual(model.max_iter, 1000)
        self.assertEqual(model.tol, 1e-8)
        self.assertIsNone(model.w)
        self.assertIsNone(model.b)

    def test_fit_returns_self_and_predict_is_list_of_floats(self):
        X = [[0.0, 1.0], [1.0, 0.0], [2.0, 1.0], [3.0, 0.0]]
        y = [1.0, 2.0, 3.0, 4.0]
        model = ElasticNetRegression()
        self.assertIs(model.fit(X, y), model)
        predictions = model.predict(X)
        self.assertIsInstance(predictions, list)
        self.assertEqual(len(predictions), len(X))
        self.assertTrue(all(type(value) is float for value in predictions))

    def test_l1_ratio_one_reproduces_lasso(self):
        # At l1_ratio=1 the elastic net update is exactly coordinate
        # descent lasso: z = fsum(x_ij**2)/n and q = alpha.
        rng = random.Random(99)
        for _ in range(10):
            n = rng.randint(2, 10)
            width = rng.randint(1, 4)
            X = [
                [rng.uniform(-3.0, 3.0) for _ in range(width)]
                for _ in range(n)
            ]
            y = [rng.uniform(-3.0, 3.0) for _ in range(n)]
            alpha = rng.choice([0.01, 0.5, 2.0])
            max_iter = rng.randint(1, 100)
            tol = 10.0 ** rng.randint(-10, -3)
            en = ElasticNetRegression(
                alpha=alpha, l1_ratio=1.0, max_iter=max_iter, tol=tol
            ).fit(X, y)
            lasso = LassoRegression(
                alpha=alpha, max_iter=max_iter, tol=tol
            ).fit(X, y)
            self.assertEqual(en.w, lasso.w)
            self.assertEqual(en.b, lasso.b)

    def test_matches_reference_oracle_on_random_problems(self):
        rng = random.Random(424242)
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
            l1_ratio = rng.choice([0.0, 0.1, 0.5, 0.9, 1.0])
            max_iter = rng.randint(1, 200)
            tol = 10.0 ** rng.randint(-12, -2)
            model = ElasticNetRegression(
                alpha=alpha,
                l1_ratio=l1_ratio,
                max_iter=max_iter,
                tol=tol,
            ).fit(X, y)
            ref_w, ref_b, _ = _reference_fit(
                X, y, alpha, l1_ratio, max_iter, tol
            )
            self.assertEqual(model.w, ref_w)
            self.assertEqual(model.b, ref_b)
            queries = [
                [rng.uniform(-3.0, 3.0) for _ in range(width)]
                for _ in range(3)
            ]
            for row, predicted in zip(queries, model.predict(queries)):
                self.assertEqual(
                    predicted, _reference_predict(row, ref_w, ref_b)
                )

    def test_at_most_max_iter_rounds(self):
        X = [[float(i)] for i in range(10)]
        y = [math.sin(i) for i in range(10)]
        model = ElasticNetRegression(
            alpha=0.001, l1_ratio=0.5, max_iter=3, tol=1e-300
        ).fit(X, y)
        ref_w, ref_b, rounds = _reference_fit(
            X, y, 0.001, 0.5, 3, 1e-300
        )
        self.assertEqual(rounds, 3)
        self.assertEqual(model.w, ref_w)
        self.assertEqual(model.b, ref_b)

    def test_integer_inputs_accepted(self):
        X = [[0, 0], [1, 2], [2, 3], [3, 6]]
        y = [1, 2, 3, 4]
        model = ElasticNetRegression(alpha=1, l1_ratio=0).fit(X, y)
        ref_w, ref_b, _ = _reference_fit(X, y, 1, 0, 1000, 1e-8)
        self.assertEqual(model.w, ref_w)
        self.assertEqual(model.b, ref_b)
        self.assertTrue(all(type(v) is float for v in model.predict(X)))

    def test_does_not_mutate_inputs(self):
        X = [[0.0, 1.0], [1.0, 0.0], [2.0, 2.0]]
        y = [1.0, 2.0, 0.0]
        X_copy = copy.deepcopy(X)
        y_copy = copy.deepcopy(y)
        ElasticNetRegression().fit(X, y)
        self.assertEqual(X, X_copy)
        self.assertEqual(y, y_copy)
        model = ElasticNetRegression().fit(X, y)
        model.predict(X)
        self.assertEqual(X, X_copy)

    def test_deterministic(self):
        X = [[0.0, 1.0], [1.0, 0.0], [2.0, 1.0], [3.0, 2.0]]
        y = [1.0, 2.0, 3.0, 5.0]
        first = ElasticNetRegression(alpha=0.3, l1_ratio=0.7).fit(X, y).predict(X)
        second = ElasticNetRegression(alpha=0.3, l1_ratio=0.7).fit(X, y).predict(X)
        self.assertEqual(first, second)

    def test_prediction_zero_is_positive_zero(self):
        X = [[1.0], [-1.0]]
        y = [1.0, -1.0]
        model = ElasticNetRegression(alpha=100.0).fit(X, y)
        self.assertEqual(model.w, [0.0])
        self.assertEqual(model.b, 0.0)
        predictions = model.predict([[4.0], [-9.0]])
        self.assertEqual(predictions, [0.0, 0.0])
        self.assertTrue(
            all(math.copysign(1.0, p) == 1.0 for p in predictions)
        )

    def test_predict_before_fit_raises_value_error(self):
        with self.assertRaises(ValueError):
            ElasticNetRegression().predict([[1.0]])

    def test_failed_fit_leaves_model_unfitted(self):
        model = ElasticNetRegression()
        good_X = [[0.0], [1.0], [2.0]]
        model.fit(good_X, [0.0, 1.0, 2.0])
        with self.assertRaises(ValueError):
            model.fit([[0.0], [1.0]], [0.0])
        self.assertIsNone(model.w)
        self.assertIsNone(model.b)
        with self.assertRaises(ValueError):
            model.predict(good_X)

    def test_invalid_fit_matrices_raise_value_error(self):
        model = ElasticNetRegression()
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
        with self.assertRaises(FloatingPointError):
            model.fit([[1], [10 ** 400]], [0.0, 0.0])

    def test_invalid_targets_raise_value_error(self):
        X = [[0.0], [1.0]]
        with self.assertRaises(ValueError):
            ElasticNetRegression().fit(X, [0.0])
        with self.assertRaises(ValueError):
            ElasticNetRegression().fit(X, "01")
        with self.assertRaises(ValueError):
            ElasticNetRegression().fit(X, [0, True])
        with self.assertRaises(ValueError):
            ElasticNetRegression().fit(X, [0.0, float("nan")])
        with self.assertRaises(FloatingPointError):
            ElasticNetRegression().fit([[0], [1]], [0, 10 ** 400])

    def test_invalid_alpha_max_iter_tol_match_lasso(self):
        for alpha in (0, 0.0, -1, -0.5, True, False, "1.0", None,
                      float("inf"), float("-inf"), float("nan"), 1 + 0j):
            with self.subTest(alpha=alpha):
                with self.assertRaises(ValueError):
                    ElasticNetRegression(alpha=alpha)
        for max_iter in (0, -1, 1.0, True, False, "1000", None):
            with self.subTest(max_iter=max_iter):
                with self.assertRaises(ValueError):
                    ElasticNetRegression(max_iter=max_iter)
        for tol in (0, 0.0, -1e-8, True, False, "1e-8", None,
                    float("inf"), float("nan")):
            with self.subTest(tol=tol):
                with self.assertRaises(ValueError):
                    ElasticNetRegression(tol=tol)

    def test_invalid_l1_ratio_raises_value_error(self):
        for l1_ratio in (True, False, "0.5", None, 1 + 0j, -0.01,
                         1.0000001, -1, 2, float("nan"),
                         float("inf"), float("-inf")):
            with self.subTest(l1_ratio=l1_ratio):
                with self.assertRaises(ValueError):
                    ElasticNetRegression(l1_ratio=l1_ratio)
        # The closed-interval endpoints and exact ints are accepted.
        for l1_ratio in (0, 1, 0.0, 1.0, 0.25):
            ElasticNetRegression(l1_ratio=l1_ratio)

    def test_overflow_during_fit_raises_floating_point_error(self):
        X = [[1e308], [1e308]]
        y = [1.0, -1.0]
        with self.assertRaises(FloatingPointError):
            ElasticNetRegression(alpha=1.0).fit(X, y)

    def test_not_serializable(self):
        # ElasticNetRegression is exported but deliberately has no
        # persistence format: dumps rejects it.
        model = ElasticNetRegression().fit([[0.0], [1.0]], [0.0, 1.0])
        with self.assertRaises(ValueError):
            classicml.dumps(model)
        with self.assertRaises(ValueError):
            classicml.loads(
                '{"class":"ElasticNetRegression","alpha":1.0}'
            )


if __name__ == "__main__":
    unittest.main()
