"""Minimal classic machine-learning library (Python standard library only).

Exports:
    LinearRegression -- deterministic ordinary least squares regression with
        an optional L2 (ridge) penalty, trained by full-batch gradient descent.
    LogisticRegression -- deterministic binary logistic regression with an
        optional L2 penalty, trained by full-batch gradient descent.
    KNeighborsClassifier -- deterministic k-nearest-neighbors classifier
        using squared Euclidean distances.
    DecisionTreeClassifier -- deterministic binary decision tree classifier
        using Gini impurity splits.
    RandomForestClassifier -- deterministic bagged forest of Gini decision
        trees with per-node random feature subsampling.
    StandardScaler -- deterministic standardization by column mean and
        population standard deviation.
    KMeans -- deterministic k-means clustering with random-seeded initial
        centroids and squared Euclidean distances.
    accuracy_score -- fraction of positions where two integer label
        vectors agree.
    mean_squared_error -- mean of squared element-wise differences of two
        finite real vectors.

CLI:
    python classicml.py train-linear
    python classicml.py train-logistic
        Reads a UTF-8 JSON object from stdin with the keys
        ``X, y, lr, l2, max_iter, tol`` (in that order) and writes a compact
        JSON object with the keys
        ``class, lr, l2, max_iter, tol, w, b`` (in that order).
    python classicml.py predict-knn
        Reads a UTF-8 JSON object from stdin with the keys
        ``X, y, n_neighbors, Q`` (in that order) and writes a compact
        JSON object with the keys ``class, predictions`` (in that order).
"""

from __future__ import annotations

import json
import math
import random
import sys
from decimal import Decimal, ROUND_HALF_UP

__all__ = [
    "LinearRegression",
    "LogisticRegression",
    "KNeighborsClassifier",
    "DecisionTreeClassifier",
    "RandomForestClassifier",
    "StandardScaler",
    "KMeans",
    "accuracy_score",
    "mean_squared_error",
]

_QUANTUM = Decimal("1E-10")
_TRAIN_KEYS = ("X", "y", "lr", "l2", "max_iter", "tol")
_KNN_KEYS = ("X", "y", "n_neighbors", "Q")


def _is_finite_number(value):
    """Return True for real, finite int/float values (booleans excluded)."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _require_finite_number(value, name):
    if not _is_finite_number(value):
        raise ValueError("%s must be a finite non-boolean real number" % name)
    return value


def _check_matrix(X):
    """Validate a non-empty rectangular matrix of finite non-boolean numbers."""
    if not isinstance(X, list) or len(X) == 0:
        raise ValueError("X must be a non-empty list of rows")
    width = None
    for row in X:
        if not isinstance(row, list) or len(row) == 0:
            raise ValueError("X rows must be non-empty lists")
        if width is None:
            width = len(row)
        elif len(row) != width:
            raise ValueError("X must be rectangular")
        for value in row:
            if not _is_finite_number(value):
                raise ValueError("X must contain only finite non-boolean numbers")
    return width


def _check_matrix_exact(X):
    """Validate a non-empty rectangular matrix of finite numbers whose type
    is exactly int or float (booleans and subclasses are rejected)."""
    if not isinstance(X, list) or len(X) == 0:
        raise ValueError("X must be a non-empty list of rows")
    width = None
    for row in X:
        if not isinstance(row, list) or len(row) == 0:
            raise ValueError("X rows must be non-empty lists")
        if width is None:
            width = len(row)
        elif len(row) != width:
            raise ValueError("X must be rectangular")
        for value in row:
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(
                    "X must contain only finite non-boolean numbers"
                )
    return width


def _check_vector(y, n):
    """Validate a target vector with the same length as X."""
    if not isinstance(y, list) or len(y) != n:
        raise ValueError("y must be a list with the same length as X")
    for value in y:
        if not _is_finite_number(value):
            raise ValueError("y must contain only finite non-boolean numbers")


def _check_binary_vector(y, n):
    """Validate a binary (0/1) target vector with the same length as X."""
    if not isinstance(y, list) or len(y) != n:
        raise ValueError("y must be a list with the same length as X")
    for value in y:
        if type(value) is not int or value not in (0, 1):
            raise ValueError("y must contain only the integers 0 and 1")


def _sigmoid(z):
    """Numerically stable logistic sigmoid."""
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


class LinearRegression:
    """Linear regression fitted with full-batch gradient descent.

    The objective is the mean squared error plus ``l2 * sum(w**2)``; the
    intercept is not penalized. Training uses no randomness.
    """

    def __init__(self, lr=0.01, l2=0.0, max_iter=1000, tol=1e-8):
        _require_finite_number(lr, "lr")
        if lr <= 0:
            raise ValueError("lr must be greater than 0")
        _require_finite_number(l2, "l2")
        if l2 < 0:
            raise ValueError("l2 must be non-negative")
        if isinstance(max_iter, bool) or not isinstance(max_iter, int):
            raise ValueError("max_iter must be an integer")
        if max_iter < 1:
            raise ValueError("max_iter must be at least 1")
        _require_finite_number(tol, "tol")
        if tol <= 0:
            raise ValueError("tol must be greater than 0")

        self.lr = lr
        self.l2 = l2
        self.max_iter = max_iter
        self.tol = tol
        self.w = None
        self.b = None

    def fit(self, X, y):
        width = _check_matrix(X)
        _check_vector(y, len(X))

        n = len(X)
        w = [0.0] * width
        b = 0.0
        lr = self.lr
        l2 = self.l2

        for _ in range(self.max_iter):
            predictions = []
            for row in X:
                p = math.fsum(w[j] * row[j] for j in range(width)) + b
                if not math.isfinite(p):
                    raise FloatingPointError(
                        "non-finite prediction encountered during fit"
                    )
                predictions.append(p)

            new_w = [0.0] * width
            for j in range(width):
                grad = (
                    2.0
                    * (
                        math.fsum(
                            (predictions[i] - y[i]) * X[i][j] for i in range(n)
                        )
                        / n
                    )
                    + 2.0 * l2 * w[j]
                )
                if not math.isfinite(grad):
                    raise FloatingPointError(
                        "non-finite weight gradient encountered during fit"
                    )
                updated = w[j] - lr * grad
                if not math.isfinite(updated):
                    raise FloatingPointError(
                        "non-finite weight value encountered during fit"
                    )
                new_w[j] = updated

            grad_b = 2.0 * (
                math.fsum(predictions[i] - y[i] for i in range(n)) / n
            )
            if not math.isfinite(grad_b):
                raise FloatingPointError(
                    "non-finite bias gradient encountered during fit"
                )
            new_b = b - lr * grad_b
            if not math.isfinite(new_b):
                raise FloatingPointError(
                    "non-finite bias value encountered during fit"
                )

            max_step = abs(new_b - b)
            for j in range(width):
                step = abs(new_w[j] - w[j])
                if step > max_step:
                    max_step = step

            w = new_w
            b = new_b

            if max_step <= self.tol:
                break

        self.w = w
        self.b = b
        return self

    def predict(self, X):
        if self.w is None or self.b is None:
            raise ValueError("model must be fitted before predict is called")
        width = _check_matrix(X)
        if width != len(self.w):
            raise ValueError(
                "X must have the same number of features as the training data"
            )

        results = []
        for row in X:
            p = math.fsum(self.w[j] * row[j] for j in range(width)) + self.b
            if not math.isfinite(p):
                raise FloatingPointError("non-finite prediction encountered")
            quantized = Decimal(str(p)).quantize(
                _QUANTUM, rounding=ROUND_HALF_UP
            )
            if quantized == 0:
                results.append(0.0)
            else:
                results.append(float(quantized))
        return results


class LogisticRegression:
    """Binary logistic regression fitted with full-batch gradient descent.

    The objective is the mean negative log-likelihood plus
    ``l2 * sum(w**2)``; the intercept is not penalized. Training uses no
    randomness.
    """

    def __init__(self, lr=0.01, l2=0.0, max_iter=1000, tol=1e-8):
        _require_finite_number(lr, "lr")
        if lr <= 0:
            raise ValueError("lr must be greater than 0")
        _require_finite_number(l2, "l2")
        if l2 < 0:
            raise ValueError("l2 must be non-negative")
        if isinstance(max_iter, bool) or not isinstance(max_iter, int):
            raise ValueError("max_iter must be an integer")
        if max_iter < 1:
            raise ValueError("max_iter must be at least 1")
        _require_finite_number(tol, "tol")
        if tol <= 0:
            raise ValueError("tol must be greater than 0")

        self.lr = lr
        self.l2 = l2
        self.max_iter = max_iter
        self.tol = tol
        self.w = None
        self.b = None

    def fit(self, X, y):
        width = _check_matrix(X)
        _check_binary_vector(y, len(X))

        n = len(X)
        w = [0.0] * width
        b = 0.0
        lr = self.lr
        l2 = self.l2

        for _ in range(self.max_iter):
            predictions = []
            for row in X:
                z = math.fsum(w[j] * row[j] for j in range(width)) + b
                if not math.isfinite(z):
                    raise FloatingPointError(
                        "non-finite prediction encountered during fit"
                    )
                predictions.append(_sigmoid(z))

            new_w = [0.0] * width
            for j in range(width):
                grad = (
                    math.fsum(
                        (predictions[i] - y[i]) * X[i][j] for i in range(n)
                    )
                    / n
                    + 2.0 * l2 * w[j]
                )
                if not math.isfinite(grad):
                    raise FloatingPointError(
                        "non-finite weight gradient encountered during fit"
                    )
                updated = w[j] - lr * grad
                if not math.isfinite(updated):
                    raise FloatingPointError(
                        "non-finite weight value encountered during fit"
                    )
                new_w[j] = updated

            grad_b = math.fsum(predictions[i] - y[i] for i in range(n)) / n
            if not math.isfinite(grad_b):
                raise FloatingPointError(
                    "non-finite bias gradient encountered during fit"
                )
            new_b = b - lr * grad_b
            if not math.isfinite(new_b):
                raise FloatingPointError(
                    "non-finite bias value encountered during fit"
                )

            max_step = abs(new_b - b)
            for j in range(width):
                step = abs(new_w[j] - w[j])
                if step > max_step:
                    max_step = step

            w = new_w
            b = new_b

            if max_step <= self.tol:
                break

        self.w = w
        self.b = b
        return self

    def predict(self, X):
        if self.w is None or self.b is None:
            raise ValueError("model must be fitted before predict is called")
        width = _check_matrix(X)
        if width != len(self.w):
            raise ValueError(
                "X must have the same number of features as the training data"
            )

        results = []
        for row in X:
            try:
                z = math.fsum(self.w[j] * row[j] for j in range(width))
            except (OverflowError, ValueError) as exc:
                raise FloatingPointError(
                    "non-finite prediction encountered"
                ) from exc
            z += self.b
            if not math.isfinite(z):
                raise FloatingPointError("non-finite prediction encountered")
            results.append(1 if z >= 0 else 0)
        return results


def _check_label_vector(y, n):
    """Validate an integer class-label vector with the same length as X."""
    if not isinstance(y, list) or len(y) != n:
        raise ValueError("y must be a list with the same length as X")
    for value in y:
        if type(value) is not int:
            raise ValueError("y must contain only integers")


def _squared_distance(query, row):
    """Squared Euclidean distance between two equal-length rows.

    Every subtraction, squaring, and summation step is checked; overflow,
    invalid operations, and non-finite intermediate values raise
    FloatingPointError.
    """
    terms = []
    for j in range(len(query)):
        try:
            diff = query[j] - row[j]
        except (OverflowError, ValueError) as exc:
            raise FloatingPointError(
                "non-finite value encountered during distance computation"
            ) from exc
        if isinstance(diff, float) and not math.isfinite(diff):
            raise FloatingPointError(
                "non-finite value encountered during distance computation"
            )
        try:
            square = diff ** 2
        except (OverflowError, ValueError) as exc:
            raise FloatingPointError(
                "non-finite value encountered during distance computation"
            ) from exc
        if isinstance(square, float) and not math.isfinite(square):
            raise FloatingPointError(
                "non-finite value encountered during distance computation"
            )
        terms.append(square)
    try:
        total = math.fsum(terms)
    except (OverflowError, ValueError) as exc:
        raise FloatingPointError(
            "non-finite value encountered during distance computation"
        ) from exc
    if not math.isfinite(total):
        raise FloatingPointError(
            "non-finite value encountered during distance computation"
        )
    return total


class KNeighborsClassifier:
    """Deterministic k-nearest-neighbors classifier.

    Distances are squared Euclidean distances accumulated with
    ``math.fsum`` in feature order. Neighbors are selected by ascending
    ``(distance, training row index)``; ties in the majority vote are
    broken in favor of the smallest label. No randomness is used.
    """

    def __init__(self, n_neighbors=5):
        if isinstance(n_neighbors, bool) or not isinstance(n_neighbors, int):
            raise ValueError("n_neighbors must be an integer")
        if n_neighbors < 1:
            raise ValueError("n_neighbors must be at least 1")
        self.n_neighbors = n_neighbors
        self._X = None
        self._y = None
        self._width = None

    def fit(self, X, y):
        width = _check_matrix(X)
        _check_label_vector(y, len(X))
        self._X = X
        self._y = y
        self._width = width
        return self

    def predict(self, X):
        if self._X is None:
            raise ValueError("model must be fitted before predict is called")
        width = _check_matrix(X)
        if width != self._width:
            raise ValueError(
                "X must have the same number of features as the training data"
            )
        if self.n_neighbors > len(self._X):
            raise ValueError(
                "n_neighbors must not exceed the number of training samples"
            )

        results = []
        for query in X:
            distances = [
                _squared_distance(query, train_row) for train_row in self._X
            ]
            order = sorted(
                range(len(self._X)), key=lambda i: (distances[i], i)
            )
            counts = {}
            for i in order[: self.n_neighbors]:
                label = self._y[i]
                counts[label] = counts.get(label, 0) + 1
            best_label = None
            best_count = -1
            for label in sorted(counts):
                if counts[label] > best_count:
                    best_count = counts[label]
                    best_label = label
            results.append(best_label)
        return results


def _check_tree_matrix(X):
    """Validate a non-empty rectangular matrix whose elements have type
    exactly ``int`` or are finite values of type exactly ``float``
    (booleans and subclasses are rejected)."""
    if not isinstance(X, list) or len(X) == 0:
        raise ValueError("X must be a non-empty list of rows")
    width = None
    for row in X:
        if not isinstance(row, list) or len(row) == 0:
            raise ValueError("X rows must be non-empty lists")
        if width is None:
            width = len(row)
        elif len(row) != width:
            raise ValueError("X must be rectangular")
        for value in row:
            if type(value) is int:
                continue
            if type(value) is float and math.isfinite(value):
                continue
            raise ValueError(
                "X must contain only finite non-boolean numbers"
            )
    return width


def _majority_label(labels):
    """Return the most frequent label; ties go to the smallest label."""
    counts = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    best_label = None
    best_count = -1
    for label in sorted(counts):
        if counts[label] > best_count:
            best_count = counts[label]
            best_label = label
    return best_label


def _gini_impurity(labels):
    """Gini impurity ``1 - fsum((c / n) ** 2)`` of a non-empty label list.

    Counts are accumulated in ascending label order. Any overflow, invalid
    operation, or non-finite intermediate value in the division, power, or
    ``math.fsum`` steps raises FloatingPointError.
    """
    counts = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    n = len(labels)
    terms = []
    for label in sorted(counts):
        try:
            ratio = counts[label] / n
            square = ratio ** 2
        except (OverflowError, ValueError) as exc:
            raise FloatingPointError(
                "non-finite value encountered during gini computation"
            ) from exc
        if not math.isfinite(ratio) or not math.isfinite(square):
            raise FloatingPointError(
                "non-finite value encountered during gini computation"
            )
        terms.append(square)
    try:
        total = math.fsum(terms)
    except (OverflowError, ValueError) as exc:
        raise FloatingPointError(
            "non-finite value encountered during gini computation"
        ) from exc
    if not math.isfinite(total):
        raise FloatingPointError(
            "non-finite value encountered during gini computation"
        )
    return 1.0 - total


class _DecisionTreeNode:
    """Node of a DecisionTreeClassifier tree (leaf when feature is None)."""

    __slots__ = ("label", "feature", "threshold", "left", "right")

    def __init__(self, label):
        self.label = label
        self.feature = None
        self.threshold = None
        self.left = None
        self.right = None


class DecisionTreeClassifier:
    """Deterministic binary decision tree classifier with Gini splits.

    Each node is labeled with the majority label of its samples (ties go to
    the smallest label). A node becomes a leaf when its labels are pure,
    when it reaches ``max_depth`` (the root is at depth 0), or when no
    candidate split strictly improves on the node's Gini impurity. For every
    feature, the distinct values in ascending order -- except the maximum --
    serve as thresholds ``t`` with ``x <= t`` going to the left child. A
    candidate's score is ``(|L| * gini(L) + |R| * gini(R)) / |S|``; only
    scores strictly below the parent Gini impurity are accepted, and the
    best candidate is the first in ascending ``(score, feature index, t)``
    order. Children are built left first, then right. No randomness is used.
    """

    def __init__(self, max_depth=None):
        if max_depth is not None:
            if type(max_depth) is not int or max_depth < 1:
                raise ValueError(
                    "max_depth must be None or a positive integer"
                )
        self.max_depth = max_depth
        self._root = None
        self._n_features = None

    def fit(self, X, y):
        self._root = None
        self._n_features = None
        width = _check_tree_matrix(X)
        _check_label_vector(y, len(X))
        root = self._build(X, y, list(range(len(X))), 0, width)
        self._root = root
        self._n_features = width
        return self

    def _build(self, X, y, indices, depth, width):
        labels = [y[i] for i in indices]
        node = _DecisionTreeNode(_majority_label(labels))

        pure = True
        first = labels[0]
        for label in labels:
            if label != first:
                pure = False
                break
        if pure:
            return node
        if self.max_depth is not None and depth >= self.max_depth:
            return node

        parent_gini = _gini_impurity(labels)
        n = len(indices)
        best = None  # (score, feature index, threshold)
        for j in range(width):
            values = sorted(set(X[i][j] for i in indices))
            for t in values[:-1]:
                left_labels = []
                right_labels = []
                for i in indices:
                    if X[i][j] <= t:
                        left_labels.append(y[i])
                    else:
                        right_labels.append(y[i])
                score = (
                    len(left_labels) * _gini_impurity(left_labels)
                    + len(right_labels) * _gini_impurity(right_labels)
                ) / n
                if score < parent_gini and (
                    best is None or (score, j, t) < best
                ):
                    best = (score, j, t)
        if best is None:
            return node

        _, feature, threshold = best
        left_indices = [i for i in indices if X[i][feature] <= threshold]
        right_indices = [i for i in indices if X[i][feature] > threshold]
        node.feature = feature
        node.threshold = threshold
        node.left = self._build(X, y, left_indices, depth + 1, width)
        node.right = self._build(X, y, right_indices, depth + 1, width)
        return node

    def predict(self, X):
        if self._root is None:
            raise ValueError("model must be fitted before predict is called")
        width = _check_tree_matrix(X)
        if width != self._n_features:
            raise ValueError(
                "X must have the same number of features as the training data"
            )
        results = []
        for row in X:
            node = self._root
            while node.feature is not None:
                if row[node.feature] <= node.threshold:
                    node = node.left
                else:
                    node = node.right
            results.append(node.label)
        return results


class RandomForestClassifier:
    """Deterministic bagged forest of Gini decision-tree classifiers.

    The forest holds ``n_estimators`` unconstrained decision trees, each
    grown exactly as ``DecisionTreeClassifier(max_depth=None)``: node labels
    are the majority label (ties go to the smallest label), thresholds are
    the distinct feature values in ascending order except the maximum with
    ``x <= t`` going left, a candidate's score is
    ``(|L| * gini(L) + |R| * gini(R)) / |S|``, only scores strictly below
    the parent Gini are accepted, the best candidate is the first in
    ascending ``(score, feature index, t)`` order, and a node becomes a leaf
    when it is pure or no candidate split improves the Gini.

    Randomness comes from a single ``random.Random(seed)`` stream consumed in
    tree order. For each tree, ``n`` (the training set size) calls to
    ``rng.randrange(n)`` draw a bootstrap sample with replacement; repeated
    rows are kept in draw order. Then, at every non-pure node in left-before
    right recursion order, ``sorted(rng.sample(range(p), max_features))`` --
    where ``p`` is the number of training features -- selects the features
    searched at that node, and the split is found using only those features
    under the rules above. Pure nodes consume no sample. Prediction votes
    over the trees in order for each input row; ties go to the smallest
    label. The same parameters and inputs always give the same result.
    """

    def __init__(self, n_estimators=10, max_features=1, seed=0):
        if type(n_estimators) is not int:
            raise ValueError("n_estimators must be an integer")
        if n_estimators <= 0:
            raise ValueError("n_estimators must be greater than 0")
        if type(max_features) is not int:
            raise ValueError("max_features must be an integer")
        if max_features <= 0:
            raise ValueError("max_features must be greater than 0")
        if type(seed) is not int:
            raise ValueError("seed must be an integer")

        self.n_estimators = n_estimators
        self.max_features = max_features
        self.seed = seed
        self._trees = None
        self._n_features = None

    def fit(self, X, y):
        self._trees = None
        self._n_features = None
        width = _check_tree_matrix(X)
        _check_label_vector(y, len(X))
        if self.max_features > width:
            raise ValueError(
                "max_features must not exceed the number of training features"
            )

        n = len(X)
        rng = random.Random(self.seed)
        trees = []
        for _ in range(self.n_estimators):
            indices = [rng.randrange(n) for _ in range(n)]
            trees.append(self._build(X, y, indices, width, rng))

        self._trees = trees
        self._n_features = width
        return self

    def _build(self, X, y, indices, width, rng):
        labels = [y[i] for i in indices]
        node = _DecisionTreeNode(_majority_label(labels))

        pure = True
        first = labels[0]
        for label in labels:
            if label != first:
                pure = False
                break
        if pure:
            return node

        features = sorted(rng.sample(range(width), self.max_features))

        parent_gini = _gini_impurity(labels)
        n = len(indices)
        best = None  # (score, feature index, threshold)
        for j in features:
            values = sorted(set(X[i][j] for i in indices))
            for t in values[:-1]:
                left_labels = []
                right_labels = []
                for i in indices:
                    if X[i][j] <= t:
                        left_labels.append(y[i])
                    else:
                        right_labels.append(y[i])
                score = (
                    len(left_labels) * _gini_impurity(left_labels)
                    + len(right_labels) * _gini_impurity(right_labels)
                ) / n
                if score < parent_gini and (
                    best is None or (score, j, t) < best
                ):
                    best = (score, j, t)
        if best is None:
            return node

        _, feature, threshold = best
        left_indices = [i for i in indices if X[i][feature] <= threshold]
        right_indices = [i for i in indices if X[i][feature] > threshold]
        node.feature = feature
        node.threshold = threshold
        node.left = self._build(X, y, left_indices, width, rng)
        node.right = self._build(X, y, right_indices, width, rng)
        return node

    def predict(self, X):
        if self._trees is None:
            raise ValueError("model must be fitted before predict is called")
        width = _check_tree_matrix(X)
        if width != self._n_features:
            raise ValueError(
                "X must have the same number of features as the training data"
            )

        results = []
        for row in X:
            counts = {}
            for tree in self._trees:
                node = tree
                while node.feature is not None:
                    if row[node.feature] <= node.threshold:
                        node = node.left
                    else:
                        node = node.right
                counts[node.label] = counts.get(node.label, 0) + 1
            best_label = None
            best_count = -1
            for label in sorted(counts):
                if counts[label] > best_count:
                    best_count = counts[label]
                    best_label = label
            results.append(best_label)
        return results


class StandardScaler:
    """Standardize columns by their mean and population standard deviation.

    For each column the mean is ``m = fsum(column) / n`` and the population
    variance is ``v = fsum((x - m) ** 2 for x in column) / n``; the scale is
    ``sqrt(v)`` unless ``v == 0``, in which case it is ``1.0``. Statistics
    are accumulated column by column, in input row order, using
    ``math.fsum``. No randomness is used.
    """

    @staticmethod
    def _validate(X):
        # Elements are accepted only when type(value) is exactly int or
        # float and math.isfinite(value) is true; booleans are rejected.
        # math.isfinite raises OverflowError for ints too large to convert
        # to float; such values fail the finite-number requirement.
        try:
            return _check_matrix_exact(X)
        except OverflowError as exc:
            raise ValueError(
                "X must contain only finite non-boolean numbers"
            ) from exc

    def __init__(self):
        self.mean_ = None
        self.scale_ = None
        self.n_features_in_ = None

    def fit(self, X):
        self.mean_ = None
        self.scale_ = None
        self.n_features_in_ = None

        width = self._validate(X)
        n = len(X)

        means = []
        scales = []
        for j in range(width):
            try:
                column_sum = math.fsum(X[i][j] for i in range(n))
            except (OverflowError, ValueError) as exc:
                raise FloatingPointError(
                    "non-finite value encountered during fit"
                ) from exc
            if not math.isfinite(column_sum):
                raise FloatingPointError(
                    "non-finite value encountered during fit"
                )
            try:
                m = column_sum / n
            except (OverflowError, ValueError) as exc:
                raise FloatingPointError(
                    "non-finite value encountered during fit"
                ) from exc
            if not math.isfinite(m):
                raise FloatingPointError(
                    "non-finite value encountered during fit"
                )
            if m == 0:
                m = 0.0

            terms = []
            for i in range(n):
                try:
                    diff = X[i][j] - m
                    square = diff ** 2
                except (OverflowError, ValueError) as exc:
                    raise FloatingPointError(
                        "non-finite value encountered during fit"
                    ) from exc
                if not math.isfinite(diff) or not math.isfinite(square):
                    raise FloatingPointError(
                        "non-finite value encountered during fit"
                    )
                terms.append(square)

            try:
                v = math.fsum(terms) / n
            except (OverflowError, ValueError) as exc:
                raise FloatingPointError(
                    "non-finite value encountered during fit"
                ) from exc
            if not math.isfinite(v):
                raise FloatingPointError(
                    "non-finite value encountered during fit"
                )

            if v == 0:
                s = 1.0
            else:
                try:
                    s = math.sqrt(v)
                except (OverflowError, ValueError) as exc:
                    raise FloatingPointError(
                        "non-finite value encountered during fit"
                    ) from exc
                if not math.isfinite(s):
                    raise FloatingPointError(
                        "non-finite value encountered during fit"
                    )

            means.append(m)
            scales.append(s)

        self.mean_ = means
        self.scale_ = scales
        self.n_features_in_ = width
        return self

    def transform(self, X):
        if (
            self.mean_ is None
            or self.scale_ is None
            or self.n_features_in_ is None
        ):
            raise ValueError(
                "scaler must be fitted before transform is called"
            )
        width = self._validate(X)
        if width != self.n_features_in_:
            raise ValueError(
                "X must have the same number of features as the training data"
            )

        results = []
        for row in X:
            scaled_row = []
            for j in range(width):
                try:
                    diff = row[j] - self.mean_[j]
                    value = diff / self.scale_[j]
                except (OverflowError, ValueError) as exc:
                    raise FloatingPointError(
                        "non-finite value encountered during transform"
                    ) from exc
                if not math.isfinite(value):
                    raise FloatingPointError(
                        "non-finite value encountered during transform"
                    )
                if value == 0:
                    value = 0.0
                scaled_row.append(value)
            results.append(scaled_row)
        return results


class KMeans:
    """Deterministic k-means clustering with random-seeded initial centroids.

    The initial centroids are copies of ``n_clusters`` training rows, taken
    in the order returned by ``random.Random(seed).sample(range(n),
    n_clusters)``. Each iteration assigns every row to the centroid with
    the smallest squared Euclidean distance, accumulated per row with
    ``math.fsum`` over ``(x_j - c_j) ** 2`` in column order; exact ties go
    to the smallest centroid index. Each non-empty cluster centroid is then
    updated synchronously, column by column, to ``fsum(member values) /
    member count``; empty clusters keep their previous centroid. Iteration
    stops when the largest absolute coordinate change is at most ``tol``,
    or after ``max_iter`` iterations. The final centroids are retained and
    prediction reuses the assignment rule. The same parameters and inputs
    always give the same result.
    """

    def __init__(self, n_clusters=8, max_iter=300, tol=1e-4, seed=0):
        if type(n_clusters) is not int:
            raise ValueError("n_clusters must be an integer")
        if n_clusters <= 0:
            raise ValueError("n_clusters must be greater than 0")
        if type(max_iter) is not int:
            raise ValueError("max_iter must be an integer")
        if max_iter <= 0:
            raise ValueError("max_iter must be greater than 0")
        if not _is_finite_number(tol) or tol <= 0:
            raise ValueError("tol must be a finite non-boolean number > 0")
        if type(seed) is not int:
            raise ValueError("seed must be an integer")

        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed
        self._centroids = None
        self._n_features = None

    @staticmethod
    def _validate(X):
        # Elements are accepted only when type(value) is exactly int or
        # float and math.isfinite(value) is true; booleans are rejected.
        # math.isfinite raises OverflowError for ints too large to convert
        # to float; such values fail the finite-number requirement.
        try:
            return _check_matrix_exact(X)
        except OverflowError as exc:
            raise ValueError(
                "X must contain only finite non-boolean numbers"
            ) from exc

    def fit(self, X):
        self._centroids = None
        self._n_features = None

        width = self._validate(X)
        n = len(X)
        if self.n_clusters > n:
            raise ValueError(
                "n_clusters must not exceed the number of training samples"
            )
        k = self.n_clusters

        rng = random.Random(self.seed)
        centroids = [list(X[i]) for i in rng.sample(range(n), k)]

        for _ in range(self.max_iter):
            assignments = self._assign(X, width, centroids)

            new_centroids = [list(centroid) for centroid in centroids]
            for c in range(k):
                members = [i for i in range(n) if assignments[i] == c]
                if not members:
                    continue
                for j in range(width):
                    try:
                        value = math.fsum(
                            X[i][j] for i in members
                        ) / len(members)
                    except (OverflowError, ValueError) as exc:
                        raise FloatingPointError(
                            "non-finite value encountered during fit"
                        ) from exc
                    if isinstance(value, float) and not math.isfinite(value):
                        raise FloatingPointError(
                            "non-finite value encountered during fit"
                        )
                    new_centroids[c][j] = value

            max_change = 0.0
            for c in range(k):
                for j in range(width):
                    try:
                        change = abs(new_centroids[c][j] - centroids[c][j])
                    except (OverflowError, ValueError) as exc:
                        raise FloatingPointError(
                            "non-finite value encountered during fit"
                        ) from exc
                    if isinstance(change, float) and not math.isfinite(change):
                        raise FloatingPointError(
                            "non-finite value encountered during fit"
                        )
                    if change > max_change:
                        max_change = change

            centroids = new_centroids
            if max_change <= self.tol:
                break

        self._centroids = centroids
        self._n_features = width
        return self

    @staticmethod
    def _assign(X, width, centroids):
        assignments = []
        for row in X:
            best_cluster = 0
            best_distance = None
            for c in range(len(centroids)):
                centroid = centroids[c]
                terms = []
                for j in range(width):
                    try:
                        diff = row[j] - centroid[j]
                        square = diff ** 2
                    except (OverflowError, ValueError) as exc:
                        raise FloatingPointError(
                            "non-finite value encountered during distance "
                            "computation"
                        ) from exc
                    if isinstance(square, float) and not math.isfinite(square):
                        raise FloatingPointError(
                            "non-finite value encountered during distance "
                            "computation"
                        )
                    terms.append(square)
                try:
                    distance = math.fsum(terms)
                except (OverflowError, ValueError) as exc:
                    raise FloatingPointError(
                        "non-finite value encountered during distance "
                        "computation"
                    ) from exc
                if isinstance(distance, float) and not math.isfinite(distance):
                    raise FloatingPointError(
                        "non-finite value encountered during distance "
                        "computation"
                    )
                if best_distance is None or distance < best_distance:
                    best_distance = distance
                    best_cluster = c
            assignments.append(best_cluster)
        return assignments

    def predict(self, X):
        if self._centroids is None:
            raise ValueError("model must be fitted before predict is called")
        width = self._validate(X)
        if width != self._n_features:
            raise ValueError(
                "X must have the same number of features as the training data"
            )
        return self._assign(X, width, self._centroids)


def _check_metric_vectors(y_true, y_pred):
    """Validate that both metric inputs are non-empty lists of equal length."""
    if not isinstance(y_true, list) or not isinstance(y_pred, list):
        raise ValueError("y_true and y_pred must be lists")
    if len(y_true) == 0 or len(y_pred) == 0:
        raise ValueError("y_true and y_pred must be non-empty lists")
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    return len(y_true)


def accuracy_score(y_true, y_pred):
    """Return the fraction of positions where two label vectors agree.

    Both arguments must be non-empty lists of equal length whose elements
    are exactly ``int`` (booleans are rejected). The inputs are not
    modified. Deterministic: same inputs, same result.
    """
    n = _check_metric_vectors(y_true, y_pred)
    for value in y_true:
        if type(value) is not int:
            raise ValueError("y_true must contain only integers")
    for value in y_pred:
        if type(value) is not int:
            raise ValueError("y_pred must contain only integers")

    correct = 0
    for i in range(n):
        if y_true[i] == y_pred[i]:
            correct += 1
    return correct / n


def mean_squared_error(y_true, y_pred):
    """Return the mean of the squared element-wise differences.

    Both arguments must be non-empty lists of equal length whose elements
    are finite values of type exactly ``int`` or ``float`` (booleans are
    rejected). The inputs are not modified. Differences, squares, the
    ``math.fsum`` total, and the final division are all checked; overflow,
    invalid operations, and non-finite intermediate values raise
    FloatingPointError. An exact zero result is normalized to ``0.0``.
    Deterministic: same inputs, same result.
    """
    n = _check_metric_vectors(y_true, y_pred)
    for name, values in (("y_true", y_true), ("y_pred", y_pred)):
        for value in values:
            if type(value) not in (int, float):
                raise ValueError(
                    "%s must contain only finite non-boolean numbers" % name
                )
            # math.isfinite raises OverflowError for ints too large to
            # convert to float; such values fail the finite requirement.
            try:
                finite = math.isfinite(value)
            except OverflowError as exc:
                raise ValueError(
                    "%s must contain only finite non-boolean numbers" % name
                ) from exc
            if not finite:
                raise ValueError(
                    "%s must contain only finite non-boolean numbers" % name
                )

    terms = []
    for i in range(n):
        try:
            d = y_true[i] - y_pred[i]
        except (OverflowError, ValueError) as exc:
            raise FloatingPointError(
                "non-finite value encountered during mean squared error"
            ) from exc
        if isinstance(d, float) and not math.isfinite(d):
            raise FloatingPointError(
                "non-finite value encountered during mean squared error"
            )
        try:
            square = d ** 2
        except (OverflowError, ValueError) as exc:
            raise FloatingPointError(
                "non-finite value encountered during mean squared error"
            ) from exc
        if isinstance(square, float) and not math.isfinite(square):
            raise FloatingPointError(
                "non-finite value encountered during mean squared error"
            )
        terms.append(square)
    try:
        total = math.fsum(terms)
    except (OverflowError, ValueError) as exc:
        raise FloatingPointError(
            "non-finite value encountered during mean squared error"
        ) from exc
    if not math.isfinite(total):
        raise FloatingPointError(
            "non-finite value encountered during mean squared error"
        )
    try:
        result = total / n
    except (OverflowError, ValueError) as exc:
        raise FloatingPointError(
            "non-finite value encountered during mean squared error"
        ) from exc
    if not math.isfinite(result):
        raise FloatingPointError(
            "non-finite value encountered during mean squared error"
        )
    if result == 0:
        result = 0.0
    return result


def _format_fixed(value):
    """Format a real number with exactly 10 digits after the decimal point."""
    return format(value + 0.0, ".10f")


def _object_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key: %r" % key)
        result[key] = value
    return result


def _train(raw_bytes, model_class, class_name):
    text = raw_bytes.decode("utf-8")
    data = json.loads(text, object_pairs_hook=_object_pairs)
    if not isinstance(data, dict) or tuple(data.keys()) != _TRAIN_KEYS:
        raise ValueError(
            "input JSON must be an object with exactly the keys "
            "X, y, lr, l2, max_iter, tol in that order"
        )

    model = model_class(
        lr=data["lr"],
        l2=data["l2"],
        max_iter=data["max_iter"],
        tol=data["tol"],
    )
    model.fit(data["X"], data["y"])

    weights = ",".join(_format_fixed(value) for value in model.w)
    return (
        '{"class":"'
        + class_name
        + '",'
        + '"lr":'
        + _format_fixed(model.lr)
        + ',"l2":'
        + _format_fixed(model.l2)
        + ',"max_iter":'
        + str(model.max_iter)
        + ',"tol":'
        + _format_fixed(model.tol)
        + ',"w":['
        + weights
        + "],"
        + '"b":'
        + _format_fixed(model.b)
        + "}"
    )


def _predict_knn(raw_bytes):
    text = raw_bytes.decode("utf-8")
    data = json.loads(text, object_pairs_hook=_object_pairs)
    if not isinstance(data, dict) or tuple(data.keys()) != _KNN_KEYS:
        raise ValueError(
            "input JSON must be an object with exactly the keys "
            "X, y, n_neighbors, Q in that order"
        )

    model = KNeighborsClassifier(n_neighbors=data["n_neighbors"])
    model.fit(data["X"], data["y"])
    predictions = model.predict(data["Q"])

    return (
        '{"class":"KNeighborsClassifier","predictions":['
        + ",".join(str(value) for value in predictions)
        + "]}"
    )


def main(argv):
    if len(argv) != 2 or argv[1] not in (
        "train-linear",
        "train-logistic",
        "predict-knn",
    ):
        sys.stderr.write("ValueError: unknown or missing subcommand\n")
        return 2
    try:
        if argv[1] == "train-linear":
            payload = _train(
                sys.stdin.buffer.read(), LinearRegression, "LinearRegression"
            )
        elif argv[1] == "train-logistic":
            payload = _train(
                sys.stdin.buffer.read(), LogisticRegression, "LogisticRegression"
            )
        else:
            payload = _predict_knn(sys.stdin.buffer.read())
    except Exception as exc:
        sys.stderr.write("ValueError: %s\n" % exc)
        return 2
    sys.stdout.write(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
