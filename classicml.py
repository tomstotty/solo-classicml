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
    KMeans -- deterministic k-means clustering with Lloyd's iterations and
        seeded centroid initialization.
    PCA -- deterministic first-principal-component projection for
        two-dimensional data.
    accuracy_score -- fraction of positions where two integer label
        vectors agree.
    mean_squared_error -- mean of squared element-wise differences of two
        finite real vectors.
    precision_recall_fscore_support -- per-class or averaged precision,
        recall, F1, and support for two integer label vectors.
    confusion_matrix -- confusion matrix for two integer label vectors with
        optional sample weights and row/column/total normalization.
    precision_score -- precision component of precision_recall_fscore_support.
    recall_score -- recall component of precision_recall_fscore_support.
    f1_score -- F1 component of precision_recall_fscore_support.
    dumps -- serialize a fitted KMeans/PCA model to whitespace-free JSON
        text (quantized to 10 decimal places with ROUND_HALF_UP).
    loads -- reconstruct an independent fitted KMeans/PCA model from text
        produced by dumps; anything outside that byte format raises
        ValueError.

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
import re
import sys
from decimal import Decimal, ROUND_HALF_UP, localcontext

__all__ = [
    "LinearRegression",
    "LogisticRegression",
    "KNeighborsClassifier",
    "DecisionTreeClassifier",
    "RandomForestClassifier",
    "StandardScaler",
    "KMeans",
    "PCA",
    "accuracy_score",
    "mean_squared_error",
    "precision_recall_fscore_support",
    "confusion_matrix",
    "precision_score",
    "recall_score",
    "f1_score",
    "dumps",
    "loads",
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


def _check_exact_matrix(X):
    """Validate a non-empty rectangular matrix whose elements have type
    exactly ``int`` (any size) or are finite values of type exactly
    ``float`` (booleans and subclasses are rejected)."""
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


class KMeans:
    """Deterministic k-means clustering with Lloyd's iterations.

    The initial centroids are copies of the training rows selected by
    ``random.Random(seed).sample(range(n), n_clusters)``, in that sample
    order. Each round assigns every row, in row order, to the centroid
    minimizing ``math.fsum((x_j - c_j) ** 2)`` accumulated in column order
    (ties go to the smallest centroid index), then synchronously replaces
    each centroid with the column-wise ``math.fsum`` mean of its members;
    empty clusters keep their previous centroid. Rounds stop when the
    largest absolute centroid coordinate change is at most ``tol``, after
    at most ``max_iter`` rounds, and the final centroids are kept. Neither
    ``fit`` nor ``predict`` modifies its input. The same parameters and
    inputs always give the same result.
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
        if (
            isinstance(tol, bool)
            or not isinstance(tol, (int, float))
            or (isinstance(tol, float) and not math.isfinite(tol))
        ):
            raise ValueError("tol must be a finite non-boolean real number")
        if tol <= 0:
            raise ValueError("tol must be greater than 0")
        if type(seed) is not int:
            raise ValueError("seed must be an integer")

        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed
        self.cluster_centers_ = None
        self._n_features = None

    @staticmethod
    def _assign(row, centroids):
        """Return the index of the centroid closest to ``row``; ties go to
        the smallest centroid index."""
        best_index = 0
        best_distance = None
        for k in range(len(centroids)):
            distance = _squared_distance(row, centroids[k])
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_index = k
        return best_index

    def fit(self, X):
        self.cluster_centers_ = None
        self._n_features = None

        width = _check_exact_matrix(X)
        n = len(X)
        if self.n_clusters > n:
            raise ValueError(
                "n_clusters must not exceed the number of samples"
            )

        rng = random.Random(self.seed)
        indices = rng.sample(range(n), self.n_clusters)
        centroids = [list(X[i]) for i in indices]

        for _ in range(self.max_iter):
            clusters = [[] for _ in range(self.n_clusters)]
            for row in X:
                clusters[self._assign(row, centroids)].append(row)

            new_centroids = []
            for k in range(self.n_clusters):
                members = clusters[k]
                if not members:
                    new_centroids.append(list(centroids[k]))
                    continue
                count = len(members)
                centroid = []
                for j in range(width):
                    try:
                        total = math.fsum(row[j] for row in members)
                    except (OverflowError, ValueError) as exc:
                        raise FloatingPointError(
                            "non-finite value encountered during fit"
                        ) from exc
                    if not math.isfinite(total):
                        raise FloatingPointError(
                            "non-finite value encountered during fit"
                        )
                    try:
                        mean = total / count
                    except (OverflowError, ValueError) as exc:
                        raise FloatingPointError(
                            "non-finite value encountered during fit"
                        ) from exc
                    if not math.isfinite(mean):
                        raise FloatingPointError(
                            "non-finite value encountered during fit"
                        )
                    centroid.append(mean)
                new_centroids.append(centroid)

            max_change = 0.0
            for k in range(self.n_clusters):
                for j in range(width):
                    try:
                        change = abs(new_centroids[k][j] - centroids[k][j])
                    except (OverflowError, ValueError) as exc:
                        raise FloatingPointError(
                            "non-finite value encountered during fit"
                        ) from exc
                    if not math.isfinite(change):
                        raise FloatingPointError(
                            "non-finite value encountered during fit"
                        )
                    if change > max_change:
                        max_change = change

            centroids = new_centroids
            if max_change <= self.tol:
                break

        self.cluster_centers_ = centroids
        self._n_features = width
        return self

    def predict(self, X):
        if self.cluster_centers_ is None:
            raise ValueError("model must be fitted before predict is called")
        width = _check_exact_matrix(X)
        if width != self._n_features:
            raise ValueError(
                "X must have the same number of features as the training data"
            )

        return [self._assign(row, self.cluster_centers_) for row in X]


class PCA:
    """Deterministic first-principal-component projection for 2-D data.

    ``fit`` computes, with ``F = math.fsum`` and terms accumulated in
    ascending index order, the column means ``mu_j = F(x_ij) / n`` and the
    population covariance entries
    ``C_jk = F((x_ij - mu_j) * (x_ik - mu_k)) / n``; the principal
    direction is ``v = [cos(theta), sin(theta)]`` where
    ``theta = atan2(2 * C_01, C_00 - C_11) / 2``. The direction is
    sign-flipped so that its largest-magnitude coordinate (coordinate 0 on
    ties) is non-negative. After a successful fit ``mean_`` is
    ``[mu_0, mu_1]`` and ``components_`` is ``[v]``; both are ``None``
    initially and after a failed fit. ``transform`` returns, per row,
    ``[F((x_j - mu_j) * v_j)]`` with an exact zero normalized to ``0.0``.
    Neither method modifies its input. No randomness is used.
    """

    def __init__(self):
        self.mean_ = None
        self.components_ = None

    @staticmethod
    def _validate(X):
        # Non-empty rectangular list[list] with exactly two columns whose
        # elements have type exactly int (any size) or finite float;
        # booleans and subclasses are rejected.
        if not isinstance(X, list) or len(X) == 0:
            raise ValueError("X must be a non-empty list of rows")
        for row in X:
            if not isinstance(row, list):
                raise ValueError("X rows must be lists")
            if len(row) != 2:
                raise ValueError("X must have exactly two columns")
            for value in row:
                if type(value) is int:
                    continue
                if type(value) is float and math.isfinite(value):
                    continue
                raise ValueError(
                    "X must contain only finite non-boolean numbers"
                )

    def fit(self, X):
        self.mean_ = None
        self.components_ = None

        self._validate(X)
        n = len(X)

        try:
            mu0 = math.fsum(X[i][0] for i in range(n)) / n
            mu1 = math.fsum(X[i][1] for i in range(n)) / n

            c00 = math.fsum(
                (X[i][0] - mu0) * (X[i][0] - mu0) for i in range(n)
            ) / n
            c01 = math.fsum(
                (X[i][0] - mu0) * (X[i][1] - mu1) for i in range(n)
            ) / n
            c11 = math.fsum(
                (X[i][1] - mu1) * (X[i][1] - mu1) for i in range(n)
            ) / n

            theta = math.atan2(2 * c01, c00 - c11) / 2
            v0 = math.cos(theta)
            v1 = math.sin(theta)
        except (OverflowError, ValueError) as exc:
            raise FloatingPointError(
                "non-finite value encountered during fit"
            ) from exc

        for value in (mu0, mu1, c00, c01, c11, theta, v0, v1):
            if not math.isfinite(value):
                raise FloatingPointError(
                    "non-finite value encountered during fit"
                )

        # Flip the direction when its largest-absolute-value coordinate
        # (coordinate 0 on ties) is negative.
        if abs(v1) > abs(v0):
            if v1 < 0.0:
                v0, v1 = -v0, -v1
        elif v0 < 0.0:
            v0, v1 = -v0, -v1

        self.mean_ = [mu0, mu1]
        self.components_ = [[v0, v1]]
        return self

    def transform(self, X):
        if self.mean_ is None or self.components_ is None:
            raise ValueError("PCA must be fitted before transform is called")
        self._validate(X)

        mean = self.mean_
        v = self.components_[0]

        results = []
        for row in X:
            try:
                value = math.fsum(
                    (row[j] - mean[j]) * v[j] for j in range(2)
                )
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
            results.append([value])
        return results

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


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


_PRF_AVERAGES = (None, "binary", "micro", "macro", "weighted")


def precision_recall_fscore_support(
    y_true, y_pred, average=None, pos_label=1, zero_division=0
):
    """Compute precision, recall, F1, and support for integer labels.

    ``y_true`` and ``y_pred`` must be non-empty lists of equal length whose
    elements have type exactly ``int`` (booleans are rejected). The label
    set is the sorted union of the labels appearing in either vector.

    Counts are taken with the true labels as rows and the predicted labels
    as columns. For each class ``c``: ``tp`` counts true ``c`` predicted
    ``c``, ``fp`` counts true non-``c`` predicted ``c``, ``fn`` counts
    true ``c`` predicted non-``c``, and ``support`` counts true ``c``;
    precision is ``tp / (tp + fp)``, recall is ``tp / support``, and
    F1 is ``2 * P * R / (P + R)``. Any zero denominator yields
    ``float(zero_division)``.

    ``average`` must be one of ``None``, ``"binary"``, ``"micro"``,
    ``"macro"``, or ``"weighted"``. ``pos_label`` must be an exact ``int``
    and ``zero_division`` must be exactly ``0`` or ``1``. With ``None`` the
    return is ``(P, R, F, support)`` -- four lists aligned to the sorted
    labels, with ``support`` entries as ``int``. ``"binary"`` requires
    exactly two labels with ``pos_label`` among them and reports that
    positive class; the other modes require ``pos_label == 1``. ``"micro"``
    pools the tp/fp/fn counts first, ``"macro"`` averages per-class values
    with ``math.fsum`` in sorted-label order, and ``"weighted"`` weights
    per-class values by their support and divides by the sample count.
    Every averaging mode returns ``(P, R, F, None)`` with float values.

    The inputs are not modified. Deterministic: same inputs, same result.
    """
    n = _check_metric_vectors(y_true, y_pred)
    for value in y_true:
        if type(value) is not int:
            raise ValueError("y_true must contain only integers")
    for value in y_pred:
        if type(value) is not int:
            raise ValueError("y_pred must contain only integers")

    if average not in _PRF_AVERAGES:
        raise ValueError(
            "average must be one of None, 'binary', 'micro', 'macro', "
            "'weighted'"
        )
    if type(pos_label) is not int:
        raise ValueError("pos_label must be an integer")
    if type(zero_division) is not int or zero_division not in (0, 1):
        raise ValueError("zero_division must be the integer 0 or 1")

    labels = sorted(set(y_true) | set(y_pred))

    if average == "binary":
        if len(labels) != 2 or pos_label not in labels:
            raise ValueError(
                "binary average requires exactly two labels with pos_label "
                "among them"
            )
    elif pos_label != 1:
        raise ValueError(
            "pos_label must be 1 unless average is 'binary'"
        )

    # Confusion counts with true labels as rows and predicted as columns.
    tp = {label: 0 for label in labels}
    fp = {label: 0 for label in labels}
    fn = {label: 0 for label in labels}
    support = {label: 0 for label in labels}
    for i in range(n):
        true_label = y_true[i]
        pred_label = y_pred[i]
        support[true_label] += 1
        if true_label == pred_label:
            tp[true_label] += 1
        else:
            fn[true_label] += 1
            fp[pred_label] += 1

    fill = float(zero_division)

    def prf_for(tp_count, fp_count, fn_count, support_count):
        """Precision, recall, F1 from one class's (or pooled) counts."""
        precision_denominator = tp_count + fp_count
        if precision_denominator == 0:
            p = fill
        else:
            p = tp_count / precision_denominator
        if support_count == 0:
            r = fill
        else:
            r = tp_count / support_count
        f_denominator = p + r
        if f_denominator == 0:
            f = fill
        else:
            f = 2.0 * p * r / f_denominator
        return p, r, f

    if average is None:
        precisions = []
        recalls = []
        fscores = []
        supports = []
        for label in labels:
            p, r, f = prf_for(
                tp[label], fp[label], fn[label], support[label]
            )
            precisions.append(p)
            recalls.append(r)
            fscores.append(f)
            supports.append(support[label])
        return precisions, recalls, fscores, supports

    if average == "binary":
        p, r, f = prf_for(
            tp[pos_label], fp[pos_label], fn[pos_label], support[pos_label]
        )
        return p, r, f, None

    if average == "micro":
        p, r, f = prf_for(
            sum(tp.values()),
            sum(fp.values()),
            sum(fn.values()),
            n,
        )
        return p, r, f, None

    per_class = [
        prf_for(tp[label], fp[label], fn[label], support[label])
        for label in labels
    ]
    if average == "macro":
        p = math.fsum(item[0] for item in per_class) / len(labels)
        r = math.fsum(item[1] for item in per_class) / len(labels)
        f = math.fsum(item[2] for item in per_class) / len(labels)
        return p, r, f, None

    # weighted
    p = math.fsum(
        per_class[k][0] * support[labels[k]] for k in range(len(labels))
    ) / n
    r = math.fsum(
        per_class[k][1] * support[labels[k]] for k in range(len(labels))
    ) / n
    f = math.fsum(
        per_class[k][2] * support[labels[k]] for k in range(len(labels))
    ) / n
    return p, r, f, None


_CONFUSION_NORMALIZES = (None, "true", "pred", "all")


def _confusion_non_finite(exc=None):
    """Build the FloatingPointError raised for non-finite arithmetic."""
    error = FloatingPointError(
        "non-finite value encountered during confusion matrix"
    )
    if exc is not None:
        raise error from exc
    raise error


def confusion_matrix(y_true, y_pred, sample_weight=None, normalize=None):
    """Compute the confusion matrix for two integer label vectors.

    ``y_true`` and ``y_pred`` must be non-empty lists of equal length whose
    elements have type exactly ``int`` (booleans are rejected). The label
    set is the sorted union of the labels appearing in either vector; true
    labels index the rows and predicted labels index the columns.

    ``sample_weight`` must be ``None`` or a list of the same length whose
    elements have type exactly ``int`` or ``float`` (booleans rejected)
    and are finite and non-negative. Without weights each sample counts as
    one and the counts are integers; with weights each cell totals its
    sample weights with ``math.fsum`` in sample order and the entries are
    floats.

    ``normalize`` must be one of ``None``, ``"true"``, ``"pred"``, or
    ``"all"``: no normalization, division by row sums, division by column
    sums, or division by the grand total. Normalized entries whose
    denominator is zero are ``0.0``, and exact zero results are normalized
    to ``0.0``. Without ``normalize`` the unweighted matrix holds ``int``
    entries and the weighted matrix holds ``float`` entries; with
    ``normalize`` the entries are floats.

    Overflow or non-finite intermediate values raise FloatingPointError.
    The inputs are not modified. Deterministic: same inputs, same result.
    """
    n = _check_metric_vectors(y_true, y_pred)
    for value in y_true:
        if type(value) is not int:
            raise ValueError("y_true must contain only integers")
    for value in y_pred:
        if type(value) is not int:
            raise ValueError("y_pred must contain only integers")

    if sample_weight is not None:
        if not isinstance(sample_weight, list) or len(sample_weight) != n:
            raise ValueError(
                "sample_weight must be None or a list with the same length "
                "as y_true"
            )
        for value in sample_weight:
            if type(value) not in (int, float):
                raise ValueError(
                    "sample_weight must contain only finite non-negative "
                    "non-boolean numbers"
                )
            # math.isfinite raises OverflowError for ints too large to
            # convert to float; such values fail the finite requirement.
            try:
                finite = math.isfinite(value)
            except OverflowError as exc:
                raise ValueError(
                    "sample_weight must contain only finite non-negative "
                    "non-boolean numbers"
                ) from exc
            if not finite or value < 0:
                raise ValueError(
                    "sample_weight must contain only finite non-negative "
                    "non-boolean numbers"
                )

    if normalize not in _CONFUSION_NORMALIZES:
        raise ValueError(
            "normalize must be one of None, 'true', 'pred', 'all'"
        )

    labels = sorted(set(y_true) | set(y_pred))
    index = {label: k for k, label in enumerate(labels)}
    size = len(labels)

    if sample_weight is None:
        matrix = [[0] * size for _ in range(size)]
        for i in range(n):
            matrix[index[y_true[i]]][index[y_pred[i]]] += 1
    else:
        cell_weights = [[[] for _ in range(size)] for _ in range(size)]
        for i in range(n):
            cell_weights[index[y_true[i]]][index[y_pred[i]]].append(
                sample_weight[i]
            )
        matrix = []
        for r in range(size):
            row = []
            for c in range(size):
                try:
                    total = math.fsum(cell_weights[r][c])
                except (OverflowError, ValueError) as exc:
                    _confusion_non_finite(exc)
                if not math.isfinite(total):
                    _confusion_non_finite()
                row.append(total)
            matrix.append(row)

    if normalize is None:
        return matrix

    if normalize == "true":
        denominators = []
        for r in range(size):
            try:
                denominator = math.fsum(matrix[r])
            except (OverflowError, ValueError) as exc:
                _confusion_non_finite(exc)
            if not math.isfinite(denominator):
                _confusion_non_finite()
            denominators.append(denominator)
        denominator_at = lambda r, c: denominators[r]
    elif normalize == "pred":
        denominators = []
        for c in range(size):
            try:
                denominator = math.fsum(matrix[r][c] for r in range(size))
            except (OverflowError, ValueError) as exc:
                _confusion_non_finite(exc)
            if not math.isfinite(denominator):
                _confusion_non_finite()
            denominators.append(denominator)
        denominator_at = lambda r, c: denominators[c]
    else:
        try:
            denominator = math.fsum(
                matrix[r][c] for r in range(size) for c in range(size)
            )
        except (OverflowError, ValueError) as exc:
            _confusion_non_finite(exc)
        if not math.isfinite(denominator):
            _confusion_non_finite()
        denominator_at = lambda r, c: denominator

    result = []
    for r in range(size):
        row = []
        for c in range(size):
            denominator = denominator_at(r, c)
            if denominator == 0:
                row.append(0.0)
                continue
            try:
                value = matrix[r][c] / denominator
            except (OverflowError, ZeroDivisionError) as exc:
                _confusion_non_finite(exc)
            if not math.isfinite(value):
                _confusion_non_finite()
            if value == 0:
                value = 0.0
            row.append(value)
        result.append(row)
    return result


def precision_score(y_true, y_pred, average="binary", pos_label=1,
                    zero_division=0):
    """Return the precision reported by precision_recall_fscore_support.

    All arguments are passed through unchanged and item 0 of the resulting
    tuple is returned; validation, return types, zero-division handling,
    and exceptions are exactly those of precision_recall_fscore_support.
    The inputs are not modified. Deterministic: same inputs, same result.
    """
    return precision_recall_fscore_support(
        y_true, y_pred, average=average, pos_label=pos_label,
        zero_division=zero_division,
    )[0]


def recall_score(y_true, y_pred, average="binary", pos_label=1,
                 zero_division=0):
    """Return the recall reported by precision_recall_fscore_support.

    All arguments are passed through unchanged and item 1 of the resulting
    tuple is returned; validation, return types, zero-division handling,
    and exceptions are exactly those of precision_recall_fscore_support.
    The inputs are not modified. Deterministic: same inputs, same result.
    """
    return precision_recall_fscore_support(
        y_true, y_pred, average=average, pos_label=pos_label,
        zero_division=zero_division,
    )[1]


def f1_score(y_true, y_pred, average="binary", pos_label=1,
             zero_division=0):
    """Return the F1 score reported by precision_recall_fscore_support.

    All arguments are passed through unchanged and item 2 of the resulting
    tuple is returned; validation, return types, zero-division handling,
    and exceptions are exactly those of precision_recall_fscore_support.
    The inputs are not modified. Deterministic: same inputs, same result.
    """
    return precision_recall_fscore_support(
        y_true, y_pred, average=average, pos_label=pos_label,
        zero_division=zero_division,
    )[2]


_SERIAL_KEYS_KMEANS = (
    "class",
    "n_clusters",
    "max_iter",
    "tol",
    "seed",
    "cluster_centers",
)
_SERIAL_KEYS_PCA = ("class", "mean", "components")


def _quantize_fixed(value):
    """Quantize a finite non-boolean real to 10 decimal places (HALF_UP)
    and return its canonical fixed-point JSON lexical form.

    ``Decimal(str(float(value)))`` performs the quantization; negative zero
    normalizes to ``0.0000000000``. The decimal context is given enough
    precision for any magnitude a binary float could carry. Every rejection
    (booleans, non-numbers, non-finite or unconvertibly large values) is a
    ValueError.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("serialized values must be finite real numbers")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "serialized values must be finite real numbers"
        ) from exc
    if not math.isfinite(number):
        raise ValueError("serialized values must be finite real numbers")
    with localcontext() as ctx:
        ctx.prec = 400
        decimal_value = Decimal(str(number)).quantize(
            _QUANTUM, rounding=ROUND_HALF_UP
        )
    if decimal_value == 0:
        return "0.0000000000"
    return format(decimal_value, "f")


def _encode_matrix(matrix):
    if not isinstance(matrix, list) or len(matrix) == 0:
        raise ValueError("cluster_centers_ must be a non-empty list of rows")
    width = None
    encoded_rows = []
    for row in matrix:
        if not isinstance(row, list) or len(row) == 0:
            raise ValueError("centroid rows must be non-empty lists")
        if width is None:
            width = len(row)
        elif len(row) != width:
            raise ValueError("cluster_centers_ must be rectangular")
        encoded_rows.append(
            "[" + ",".join(_quantize_fixed(v) for v in row) + "]"
        )
    return "[" + ",".join(encoded_rows) + "]", width


def _encode_vector(vector, length):
    if not isinstance(vector, list) or len(vector) != length:
        raise ValueError("state vector has the wrong type or shape")
    return "[" + ",".join(_quantize_fixed(v) for v in vector) + "]"


def dumps(model):
    """Serialize a fitted KMeans or PCA model to compact JSON text.

    The result contains no whitespace and no trailing newline. Integers
    (``n_clusters``, ``max_iter``, ``seed``) are emitted as JSON integers;
    ``tol`` and every array coordinate is quantized to 10 decimal places
    with ROUND_HALF_UP (negative zero becomes ``0.0000000000``). A positive
    ``tol`` that quantizes to zero is rejected, as are any non-fitted
    models, non-KMeans/PCA objects, invalid construction parameters, and
    malformed or non-finite state. The argument is not modified.
    """
    if isinstance(model, KMeans):
        try:
            return _dumps_kmeans(model)
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError("invalid KMeans state") from exc

    if isinstance(model, PCA):
        try:
            return _dumps_pca(model)
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError("invalid PCA state") from exc

    raise ValueError("dumps only supports fitted KMeans and PCA models")


def _dumps_kmeans(model):
    centers = model.cluster_centers_
    if centers is None:
        raise ValueError("KMeans must be fitted before dumps is called")
    n_clusters = model.n_clusters
    max_iter = model.max_iter
    tol = model.tol
    seed = model.seed
    # Re-validate the construction parameters exactly as __init__ does.
    if (
        type(n_clusters) is not int
        or n_clusters <= 0
        or type(max_iter) is not int
        or max_iter <= 0
        or not _is_valid_tol(tol)
        or type(seed) is not int
    ):
        raise ValueError("KMeans has invalid construction parameters")
    centers_text, width = _encode_matrix(centers)
    if len(centers) != n_clusters:
        raise ValueError("number of centroids must equal n_clusters")
    if type(model._n_features) is not int or model._n_features != width:
        raise ValueError("KMeans feature count does not match centroids")
    tol_text = _quantize_fixed(tol)
    if Decimal(tol_text) == 0:
        raise ValueError(
            "tol must remain positive after quantization to 10 decimals"
        )
    return (
        '{"class":"KMeans","n_clusters":'
        + str(n_clusters)
        + ',"max_iter":'
        + str(max_iter)
        + ',"tol":'
        + tol_text
        + ',"seed":'
        + str(seed)
        + ',"cluster_centers":'
        + centers_text
        + "}"
    )


def _is_valid_tol(tol):
    """Mirror KMeans.__init__'s tol check without raising on ints too
    large for ``math.isfinite`` to convert."""
    if isinstance(tol, bool) or not isinstance(tol, (int, float)):
        return False
    try:
        return math.isfinite(tol) and tol > 0
    except OverflowError:
        return False


def _dumps_pca(model):
    if model.mean_ is None or model.components_ is None:
        raise ValueError("PCA must be fitted before dumps is called")
    mean_text = _encode_vector(model.mean_, 2)
    components = model.components_
    if (
        not isinstance(components, list)
        or len(components) != 1
        or not isinstance(components[0], list)
        or len(components[0]) != 2
    ):
        raise ValueError("components_ must have shape [1][2]")
    components_text = "[" + _encode_vector(components[0], 2) + "]"
    return (
        '{"class":"PCA","mean":'
        + mean_text
        + ',"components":'
        + components_text
        + "}"
    )


_JSON_INT_RE = re.compile(r"^(0|-?[1-9][0-9]*)$")
_JSON_FLOAT_RE = re.compile(r"^-?(0|[1-9][0-9]*)\.[0-9]{10}$")
# The only strings in the format are fixed keys and class names: plain
# printable ASCII without quotes, backslashes, or control characters.
_JSON_STRING_RE = re.compile(r'"[^"\\\x00-\x1f]*"')
_JSON_NUMBER_RE = re.compile(
    r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?"
)
_JSON_WS_RE = re.compile(r"[ \t\n\r]")
_JSON_LITERAL_RE = re.compile(r"true|false|null")


class _Object(list):
    """Marker list holding an object's (key, value) pairs, distinct from a
    plain JSON array (also represented by a list)."""


def _decode_json(text):
    """Strict recursive-descent JSON parser.

    Accepts the JSON grammar while rejecting whitespace anywhere, trailing
    characters, duplicate object keys, and numeric lexical forms outside
    the serialized byte format (exponent notation, leading-plus fractions,
    non-10-digit decimals). Booleans/null are parsed so that their presence
    as values can be rejected by the structural validators.
    """
    return _JsonParser(text).parse()


class _JsonParser:
    def __init__(self, text):
        self.text = text
        self.n = len(text)
        self.pos = 0

    def error(self):
        raise ValueError("malformed serialized model")

    def parse(self):
        value = self.parse_value()
        if self.pos != self.n:
            self.error()
        return value

    def parse_value(self):
        if self.pos >= self.n:
            self.error()
        char = self.text[self.pos]
        if char == "{":
            return self.parse_object()
        if char == "[":
            return self.parse_array()
        if char == '"':
            return self.parse_string_token()
        if char == "-" or "0" <= char <= "9":
            return self.parse_number_token()
        match = _JSON_LITERAL_RE.match(self.text, self.pos)
        if match is not None:
            literal = match.group(0)
            self.pos = match.end()
            if literal == "true":
                return True
            if literal == "false":
                return False
            return None
        self.error()

    def parse_object(self):
        pairs = _Object()
        self.pos += 1
        if self.pos < self.n and self.text[self.pos] == "}":
            self.pos += 1
            return pairs
        while True:
            if self.pos >= self.n or self.text[self.pos] != '"':
                self.error()
            key = self.parse_string_token()
            if self.pos >= self.n or self.text[self.pos] != ":":
                self.error()
            self.pos += 1
            value = self.parse_value()
            pairs.append((key, value))
            if self.pos >= self.n:
                self.error()
            if self.text[self.pos] == ",":
                self.pos += 1
                continue
            if self.text[self.pos] == "}":
                self.pos += 1
                break
            self.error()
        return pairs

    def parse_array(self):
        items = []
        self.pos += 1
        if self.pos < self.n and self.text[self.pos] == "]":
            self.pos += 1
            return items
        while True:
            items.append(self.parse_value())
            if self.pos >= self.n:
                self.error()
            if self.text[self.pos] == ",":
                self.pos += 1
                continue
            if self.text[self.pos] == "]":
                self.pos += 1
                break
            self.error()
        return items

    def parse_string_token(self):
        match = _JSON_STRING_RE.match(self.text, self.pos)
        if match is None:
            self.error()
        self.pos = match.end()
        return match.group(0)[1:-1]

    def parse_number_token(self):
        match = _JSON_NUMBER_RE.match(self.text, self.pos)
        if match is None:
            self.error()
        token = match.group(0)
        self.pos = match.end()
        if _JSON_INT_RE.match(token):
            return ("int", token)
        if _JSON_FLOAT_RE.match(token):
            return ("fixed", token)
        # Any other valid JSON numeric form (e.g. exponents) is not part of
        # the serialized byte format.
        self.error()


def _convert(node):
    """Convert parser output into nested dicts/lists.

    Number leaves keep their ``("int"|"fixed", token)`` tuples so callers
    can enforce lexical/type rules; booleans and null are rejected.
    """
    if isinstance(node, _Object):
        result = {}
        for key, value in node:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = _convert(value)
        return result
    if isinstance(node, list):
        return [_convert(item) for item in node]
    if isinstance(node, tuple) and node[0] in ("int", "fixed"):
        return node
    if isinstance(node, str):
        return node
    # bool and None reached a value position.
    raise ValueError("unexpected boolean or null in serialized model")


def _expect_int(entry, name):
    if not (
        isinstance(entry, tuple)
        and entry[0] == "int"
        and _JSON_INT_RE.match(entry[1])
    ):
        raise ValueError("%s must be a JSON integer" % name)
    return int(entry[1])


def _expect_fixed(entry, name):
    if not (
        isinstance(entry, tuple)
        and entry[0] == "fixed"
        and _JSON_FLOAT_RE.match(entry[1])
    ):
        raise ValueError("%s must be a fixed 10-decimal JSON number" % name)
    token = entry[1]
    value = float(token)
    if not math.isfinite(value):
        raise ValueError("%s must be finite" % name)
    # dumps normalizes negative zero to "0.0000000000".
    if value == 0.0 and token[0] == "-":
        raise ValueError("%s must not be negative zero" % name)
    return value


def _expect_fixed_vector(node, length, name):
    if not isinstance(node, list) or len(node) != length:
        raise ValueError("%s must have length %d" % (name, length))
    return [_expect_fixed(item, name + " element") for item in node]


def _load_kmeans(pairs):
    keys = tuple(key for key, _ in pairs)
    if keys != _SERIAL_KEYS_KMEANS:
        raise ValueError("KMeans JSON must have exactly the serialized keys "
                         "in the serialized order")
    data = _convert(pairs)

    class_name = data["class"]
    if not isinstance(class_name, str) or class_name != "KMeans":
        raise ValueError('class must be "KMeans"')

    n_clusters = _expect_int(data["n_clusters"], "n_clusters")
    max_iter = _expect_int(data["max_iter"], "max_iter")
    seed = _expect_int(data["seed"], "seed")
    if n_clusters <= 0:
        raise ValueError("n_clusters must be greater than 0")
    if max_iter <= 0:
        raise ValueError("max_iter must be greater than 0")

    tol = _expect_fixed(data["tol"], "tol")
    if tol <= 0.0:
        raise ValueError("tol must be greater than 0")

    centers_node = data["cluster_centers"]
    if not isinstance(centers_node, list) or len(centers_node) != n_clusters:
        raise ValueError("number of centroid rows must equal n_clusters")
    centers = []
    width = None
    for row in centers_node:
        if not isinstance(row, list) or len(row) == 0:
            raise ValueError("centroid rows must be non-empty lists")
        if width is None:
            width = len(row)
        elif len(row) != width:
            raise ValueError("centroids must be rectangular")
        centers.append(
            [_expect_fixed(v, "centroid coordinate") for v in row]
        )

    model = KMeans(
        n_clusters=n_clusters, max_iter=max_iter, tol=tol, seed=seed
    )
    model.cluster_centers_ = [list(row) for row in centers]
    model._n_features = width
    return model


def _load_pca(pairs):
    keys = tuple(key for key, _ in pairs)
    if keys != _SERIAL_KEYS_PCA:
        raise ValueError("PCA JSON must have exactly the serialized keys "
                         "in the serialized order")
    data = _convert(pairs)

    class_name = data["class"]
    if not isinstance(class_name, str) or class_name != "PCA":
        raise ValueError('class must be "PCA"')

    mean = _expect_fixed_vector(data["mean"], 2, "mean")

    components_node = data["components"]
    if not isinstance(components_node, list) or len(components_node) != 1:
        raise ValueError("components must have shape [[2]]")
    components = [
        _expect_fixed_vector(components_node[0], 2, "components row")
    ]

    model = PCA()
    model.mean_ = list(mean)
    model.components_ = [list(components[0])]
    return model


def loads(text):
    """Reconstruct a fitted KMeans or PCA from text produced by dumps.

    Only the exact byte format emitted by :func:`dumps` is accepted: a
    ``str`` holding compact JSON with no whitespace, no duplicate keys,
    the exact key sets in order, JSON integers for integer parameters,
    10-decimal fixed-point numbers for floats, and consistent array
    shapes. Anything else -- including non-str input, empty strings,
    parse failures, booleans, exponent notation, non-finite values, or
    illegal parameters -- raises ValueError. The returned model is
    independent of the input and fitted; KMeans recovers its column count
    from centroid width. The argument is not modified.
    """
    if not isinstance(text, str) or len(text) == 0:
        raise ValueError("loads requires a non-empty str")
    if _JSON_WS_RE.search(text):
        raise ValueError("serialized model must contain no whitespace")

    try:
        pairs = _decode_json(text)
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("malformed serialized model") from exc

    if not isinstance(pairs, _Object) or len(pairs) == 0:
        raise ValueError("top-level JSON value must be an object")

    class_entry = None
    for key, value in pairs:
        if key == "class":
            class_entry = value
            break
    if not isinstance(class_entry, str):
        raise ValueError('object must contain a string "class" key')

    if class_entry == "KMeans":
        try:
            return _load_kmeans(pairs)
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError("malformed serialized model") from exc
    if class_entry == "PCA":
        try:
            return _load_pca(pairs)
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError("malformed serialized model") from exc
    raise ValueError("unknown class in serialized model")


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
