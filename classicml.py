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
    r2_score -- coefficient of determination of two finite real vectors,
        optionally weighted.
    precision_recall_fscore_support -- per-class or averaged precision,
        recall, F1, and support for two integer label vectors.
    confusion_matrix -- unweighted or weighted, optionally normalized
        confusion counts with true labels as rows and predicted labels as
        columns.
    precision_score -- precision entry of precision_recall_fscore_support.
    recall_score -- recall entry of precision_recall_fscore_support.
    f1_score -- F1 entry of precision_recall_fscore_support.
    roc_curve -- false/true positive rates and thresholds for a binary
        score ranking.
    roc_auc_score -- trapezoidal area under the roc_curve.
    precision_recall_curve -- precision/recall pairs at descending score
        thresholds for a binary score ranking.
    average_precision_score -- stepwise area under the
        precision_recall_curve.
    calibration_curve -- per-bin positive-class weight fractions and mean
        predicted probabilities for a binary probability vector.
    brier_score_loss -- weighted mean squared probability error for a
        binary probability vector.
    log_loss -- weighted logistic (cross-entropy) loss for a binary
        probability vector with clamped probabilities.
    multiclass_log_loss -- weighted multiclass logistic (cross-entropy)
        loss over a probability matrix with one column per class, using
        the probability of the true class clamped away from zero.
    silhouette_score -- mean silhouette coefficient of a clustering over
        a finite real matrix and an integer label vector.
    davies_bouldin_score -- Davies-Bouldin index of a clustering: the mean,
        over clusters in ascending label order, of the largest ratio of
        summed within-cluster mean distances to centroid separation.
    calinski_harabasz_score -- Calinski-Harabasz index of a clustering:
        the ratio of between-cluster dispersion to within-cluster
        dispersion, each divided by its degrees of freedom.
    adjusted_rand_score -- exact adjusted Rand index of two integer
        partitions, computed with fractions.Fraction and returned as float.
    normalized_mutual_info_score -- mutual information of two integer
        partitions normalized by the geometric mean of their entropies.
    matthews_corrcoef -- Matthews correlation coefficient of two integer
        label vectors, computed from an exact integer contingency table.
    balanced_accuracy_score -- weighted mean per-class recall of two
        integer label vectors, optionally adjusted for chance.
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
from fractions import Fraction

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
    "r2_score",
    "mean_absolute_percentage_error",
    "precision_recall_fscore_support",
    "confusion_matrix",
    "precision_score",
    "recall_score",
    "f1_score",
    "roc_curve",
    "roc_auc_score",
    "precision_recall_curve",
    "average_precision_score",
    "calibration_curve",
    "brier_score_loss",
    "log_loss",
    "multiclass_log_loss",
    "silhouette_score",
    "davies_bouldin_score",
    "calinski_harabasz_score",
    "adjusted_rand_score",
    "normalized_mutual_info_score",
    "matthews_corrcoef",
    "cohen_kappa_score",
    "balanced_accuracy_score",
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


def _r2_float(value):
    """Convert a validated number to float; overflow, invalid operations,
    and non-finite results raise FloatingPointError."""
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise FloatingPointError(
            "non-finite value encountered during r2-score computation"
        ) from exc
    if not math.isfinite(result):
        raise FloatingPointError(
            "non-finite value encountered during r2-score computation"
        )
    return result


def r2_score(y_true, y_pred, sample_weight=None, force_finite=True):
    """Return the coefficient of determination (R^2).

    ``y_true`` and ``y_pred`` must be non-empty lists of equal length whose
    elements are finite values of type exactly ``int`` or ``float``
    (booleans are rejected). ``sample_weight`` must be ``None`` -- every
    sample then weighs ``1.0`` -- or a list of the same length whose
    elements are finite non-negative values of type exactly ``int`` or
    ``float`` (booleans are rejected). ``force_finite`` must be exactly
    ``True`` or ``False``. The total weight must be greater than zero. Any
    violation (including overflow during the finiteness checks) raises
    ValueError.

    After validation, the values and weights are converted to ``float``;
    in input order, ``math.fsum`` computes the total weight
    ``W = sum(w_i)``, the weighted mean ``m = sum(w_i * y_true_i) / W``,
    the residual sum of squares
    ``SSE = sum(w_i * (y_true_i - y_pred_i) ** 2)``, and the total sum of
    squares ``SST = sum(w_i * (y_true_i - m) ** 2)``. When ``SST`` is
    non-zero the result is ``1.0 - SSE / SST``. When ``SST`` and ``SSE``
    are both zero the result is ``1.0``. When only ``SST`` is zero the
    result is ``0.0`` if ``force_finite`` is true and ``float("-inf")``
    otherwise. Overflow, invalid operations during the post-validation
    conversion or arithmetic, and non-finite intermediate values or
    results raise FloatingPointError (the negative-infinity result above
    excepted). An exact zero result is normalized to ``0.0``.

    The return value is a float. The inputs are not modified.
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

    if sample_weight is None:
        weights = [1.0] * n
    else:
        if not isinstance(sample_weight, list) or len(sample_weight) != n:
            raise ValueError(
                "sample_weight must be a list with the same length as "
                "y_true"
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
        weights = [_r2_float(value) for value in sample_weight]

    if type(force_finite) is not bool:
        raise ValueError("force_finite must be a boolean")

    try:
        total_weight = math.fsum(weights)
    except (OverflowError, ValueError) as exc:
        raise FloatingPointError(
            "non-finite value encountered during r2-score computation"
        ) from exc
    if not math.isfinite(total_weight):
        raise FloatingPointError(
            "non-finite value encountered during r2-score computation"
        )
    if total_weight <= 0.0:
        raise ValueError("the total weight must be greater than 0")

    yt = [_r2_float(value) for value in y_true]
    yp = [_r2_float(value) for value in y_pred]

    mean_terms = []
    for i in range(n):
        try:
            term = weights[i] * yt[i]
        except (OverflowError, ValueError) as exc:
            raise FloatingPointError(
                "non-finite value encountered during r2-score computation"
            ) from exc
        if not math.isfinite(term):
            raise FloatingPointError(
                "non-finite value encountered during r2-score computation"
            )
        mean_terms.append(term)
    try:
        mean = math.fsum(mean_terms) / total_weight
    except (OverflowError, ValueError, ZeroDivisionError) as exc:
        raise FloatingPointError(
            "non-finite value encountered during r2-score computation"
        ) from exc
    if not math.isfinite(mean):
        raise FloatingPointError(
            "non-finite value encountered during r2-score computation"
        )

    sse_terms = []
    sst_terms = []
    for i in range(n):
        try:
            sse_term = weights[i] * (yt[i] - yp[i]) ** 2
            sst_term = weights[i] * (yt[i] - mean) ** 2
        except (OverflowError, ValueError) as exc:
            raise FloatingPointError(
                "non-finite value encountered during r2-score computation"
            ) from exc
        if not math.isfinite(sse_term) or not math.isfinite(sst_term):
            raise FloatingPointError(
                "non-finite value encountered during r2-score computation"
            )
        sse_terms.append(sse_term)
        sst_terms.append(sst_term)
    try:
        sse = math.fsum(sse_terms)
        sst = math.fsum(sst_terms)
    except (OverflowError, ValueError) as exc:
        raise FloatingPointError(
            "non-finite value encountered during r2-score computation"
        ) from exc
    if not math.isfinite(sse) or not math.isfinite(sst):
        raise FloatingPointError(
            "non-finite value encountered during r2-score computation"
        )

    if sst != 0.0:
        try:
            result = 1.0 - sse / sst
        except (OverflowError, ValueError, ZeroDivisionError) as exc:
            raise FloatingPointError(
                "non-finite value encountered during r2-score computation"
            ) from exc
        if not math.isfinite(result):
            raise FloatingPointError(
                "non-finite value encountered during r2-score computation"
            )
        if result == 0:
            result = 0.0
        return result
    if sse == 0.0:
        return 1.0
    if force_finite:
        return 0.0
    return float("-inf")


def mean_absolute_percentage_error(y_true, y_pred, sample_weight=None) -> float:
    """Return the weighted mean absolute percentage error (MAPE).

    ``y_true`` and ``y_pred`` must be non-empty lists of equal length
    whose elements are finite values of type exactly ``int`` or
    ``float`` (booleans are rejected). ``sample_weight`` must be
    ``None`` -- every sample then weighs ``1.0`` -- or a list of the
    same length whose elements are finite non-negative values of type
    exactly ``int`` or ``float`` (booleans are rejected). Any
    container, length, type, range, or finiteness violation (including
    ``OverflowError`` raised by ``math.isfinite``) raises ValueError.

    After validation, the values and weights are converted to
    ``float`` in input order. For each index, with ``t`` the true
    value, ``p`` the prediction, and ``w`` the weight, the absolute
    percentage error is
    ``e = abs(t - p) / max(abs(t), sys.float_info.epsilon)``.
    ``math.fsum`` computes ``W = sum(w)`` and ``L = sum(w * e)``. When
    ``W`` is finite and less than or equal to zero a ValueError is
    raised; otherwise the result is ``L / W``. Overflow, invalid
    operations during the post-validation conversion or arithmetic,
    and non-finite intermediate values or results raise
    FloatingPointError. An exact zero result is normalized to ``0.0``.

    The return value is a float. The inputs are not modified.
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

    if sample_weight is not None:
        if not isinstance(sample_weight, list) or len(sample_weight) != n:
            raise ValueError(
                "sample_weight must be a list with the same length as "
                "y_true"
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

    try:
        t = [float(value) for value in y_true]
        p = [float(value) for value in y_pred]
        if sample_weight is None:
            w = [1.0] * n
        else:
            w = [float(value) for value in sample_weight]
    except (OverflowError, ValueError) as exc:
        raise FloatingPointError(
            "non-finite value encountered during mean absolute "
            "percentage error"
        ) from exc
    for values in (t, p, w):
        for value in values:
            if not math.isfinite(value):
                raise FloatingPointError(
                    "non-finite value encountered during mean absolute "
                    "percentage error"
                )

    weight_terms = []
    loss_terms = []
    for i in range(n):
        try:
            abs_true = abs(t[i])
            abs_diff = abs(t[i] - p[i])
        except (OverflowError, ValueError) as exc:
            raise FloatingPointError(
                "non-finite value encountered during mean absolute "
                "percentage error"
            ) from exc
        if not math.isfinite(abs_true) or not math.isfinite(abs_diff):
            raise FloatingPointError(
                "non-finite value encountered during mean absolute "
                "percentage error"
            )
        denom = max(abs_true, sys.float_info.epsilon)
        try:
            error = abs_diff / denom
            term = w[i] * error
        except (OverflowError, ValueError, ZeroDivisionError) as exc:
            raise FloatingPointError(
                "non-finite value encountered during mean absolute "
                "percentage error"
            ) from exc
        if not math.isfinite(error) or not math.isfinite(term):
            raise FloatingPointError(
                "non-finite value encountered during mean absolute "
                "percentage error"
            )
        weight_terms.append(w[i])
        loss_terms.append(term)

    try:
        total_weight = math.fsum(weight_terms)
        total_loss = math.fsum(loss_terms)
    except (OverflowError, ValueError) as exc:
        raise FloatingPointError(
            "non-finite value encountered during mean absolute "
            "percentage error"
        ) from exc
    if not math.isfinite(total_weight) or not math.isfinite(total_loss):
        raise FloatingPointError(
            "non-finite value encountered during mean absolute "
            "percentage error"
        )
    if total_weight <= 0.0:
        raise ValueError("the total weight must be greater than 0")
    try:
        result = total_loss / total_weight
    except (OverflowError, ValueError, ZeroDivisionError) as exc:
        raise FloatingPointError(
            "non-finite value encountered during mean absolute "
            "percentage error"
        ) from exc
    if not math.isfinite(result):
        raise FloatingPointError(
            "non-finite value encountered during mean absolute "
            "percentage error"
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


_CM_NORMALIZES = (None, "true", "pred", "all")


def confusion_matrix(y_true, y_pred, sample_weight=None, normalize=None):
    """Build a confusion matrix with true labels as rows and predictions as
    columns.

    ``y_true`` and ``y_pred`` must be non-empty lists of equal length whose
    elements have type exactly ``int`` (booleans are rejected). The labels
    are the sorted union of the labels appearing in either vector; rows and
    columns share that order. ``sample_weight`` must be ``None`` or a list
    of the same length whose elements are finite non-negative values of
    type exactly ``int`` or ``float`` (booleans are rejected). ``normalize``
    must be one of ``None``, ``"true"``, ``"pred"``, or ``"all"``. Any
    violation raises ValueError.

    Without weights, each sample contributes one exact ``int`` count, so
    the matrix is an ``int`` matrix. With weights, each cell sums the
    weights of its samples with ``math.fsum`` in sample order, giving a
    ``float`` matrix. Normalization divides each row by its total
    (``"true"``), each column by its total (``"pred"``), or every cell by
    the grand total (``"all"``); a zero denominator, or a cell that is an
    exact zero, becomes ``0.0``. Overflow or any non-finite intermediate
    value raises FloatingPointError.

    The inputs are not modified. Deterministic: same inputs, same result.
    """
    n = _check_metric_vectors(y_true, y_pred)
    for value in y_true:
        if type(value) is not int:
            raise ValueError("y_true must contain only integers")
    for value in y_pred:
        if type(value) is not int:
            raise ValueError("y_pred must contain only integers")
    if normalize not in _CM_NORMALIZES:
        raise ValueError(
            "normalize must be one of None, 'true', 'pred', 'all'"
        )

    weighted = sample_weight is not None
    if weighted:
        if not isinstance(sample_weight, list) or len(sample_weight) != n:
            raise ValueError(
                "sample_weight must be a list with the same length as the "
                "label vectors"
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

    labels = sorted(set(y_true) | set(y_pred))
    index = {label: k for k, label in enumerate(labels)}
    size = len(labels)

    if weighted:
        # Per-cell weight lists, accumulated in sample order, so each cell
        # is reduced with math.fsum in that exact order.
        terms = [[[] for _ in range(size)] for _ in range(size)]
        for i in range(n):
            terms[index[y_true[i]]][index[y_pred[i]]].append(
                float(sample_weight[i])
            )
        matrix = [[0.0] * size for _ in range(size)]
        for r in range(size):
            for c in range(size):
                cell_terms = terms[r][c]
                if not cell_terms:
                    continue
                try:
                    total = math.fsum(cell_terms)
                except (OverflowError, ValueError) as exc:
                    raise FloatingPointError(
                        "non-finite value encountered during confusion matrix"
                    ) from exc
                if not math.isfinite(total):
                    raise FloatingPointError(
                        "non-finite value encountered during confusion matrix"
                    )
                matrix[r][c] = total
    else:
        matrix = [[0 for _ in range(size)] for _ in range(size)]
        for i in range(n):
            matrix[index[y_true[i]]][index[y_pred[i]]] += 1

    if normalize is None:
        return matrix

    def checked_quotient(numerator, denominator):
        """``numerator / denominator`` with an exact-zero or zero
        denominator normalized to ``0.0``; non-finite results raise."""
        if numerator == 0 or denominator == 0:
            return 0.0
        try:
            result = numerator / denominator
        except (OverflowError, ValueError) as exc:
            raise FloatingPointError(
                "non-finite value encountered during confusion matrix "
                "normalization"
            ) from exc
        if not math.isfinite(result):
            raise FloatingPointError(
                "non-finite value encountered during confusion matrix "
                "normalization"
            )
        return result

    if normalize == "true":
        normalized = [[0.0] * size for _ in range(size)]
        for r in range(size):
            try:
                row_total = math.fsum(matrix[r])
            except (OverflowError, ValueError) as exc:
                raise FloatingPointError(
                    "non-finite value encountered during confusion matrix "
                    "normalization"
                ) from exc
            if not math.isfinite(row_total):
                raise FloatingPointError(
                    "non-finite value encountered during confusion matrix "
                    "normalization"
                )
            for c in range(size):
                normalized[r][c] = checked_quotient(
                    matrix[r][c], row_total
                )
        return normalized

    if normalize == "pred":
        normalized = [[0.0] * size for _ in range(size)]
        column_totals = []
        for c in range(size):
            try:
                column_total = math.fsum(matrix[r][c] for r in range(size))
            except (OverflowError, ValueError) as exc:
                raise FloatingPointError(
                    "non-finite value encountered during confusion matrix "
                    "normalization"
                ) from exc
            if not math.isfinite(column_total):
                raise FloatingPointError(
                    "non-finite value encountered during confusion matrix "
                    "normalization"
                )
            column_totals.append(column_total)
        for r in range(size):
            for c in range(size):
                normalized[r][c] = checked_quotient(
                    matrix[r][c], column_totals[c]
                )
        return normalized

    # normalize == "all"
    try:
        grand_total = math.fsum(
            matrix[r][c] for r in range(size) for c in range(size)
        )
    except (OverflowError, ValueError) as exc:
        raise FloatingPointError(
            "non-finite value encountered during confusion matrix "
            "normalization"
        ) from exc
    if not math.isfinite(grand_total):
        raise FloatingPointError(
            "non-finite value encountered during confusion matrix "
            "normalization"
        )
    normalized = [[0.0] * size for _ in range(size)]
    for r in range(size):
        for c in range(size):
            normalized[r][c] = checked_quotient(
                matrix[r][c], grand_total
            )
    return normalized


def precision_score(
    y_true, y_pred, average="binary", pos_label=1, zero_division=0
):
    """Return the precision entry of
    :func:`precision_recall_fscore_support` for the same arguments.

    Return types, validation, zero-denominator handling, and exceptions are
    exactly those of the underlying function: a float for the averaging
    modes and a list of floats when ``average`` is ``None``.
    """
    return precision_recall_fscore_support(
        y_true, y_pred, average=average, pos_label=pos_label,
        zero_division=zero_division
    )[0]


def recall_score(
    y_true, y_pred, average="binary", pos_label=1, zero_division=0
):
    """Return the recall entry of
    :func:`precision_recall_fscore_support` for the same arguments.

    Return types, validation, zero-denominator handling, and exceptions are
    exactly those of the underlying function: a float for the averaging
    modes and a list of floats when ``average`` is ``None``.
    """
    return precision_recall_fscore_support(
        y_true, y_pred, average=average, pos_label=pos_label,
        zero_division=zero_division
    )[1]


def f1_score(
    y_true, y_pred, average="binary", pos_label=1, zero_division=0
):
    """Return the F1 entry of
    :func:`precision_recall_fscore_support` for the same arguments.

    Return types, validation, zero-denominator handling, and exceptions are
    exactly those of the underlying function: a float for the averaging
    modes and a list of floats when ``average`` is ``None``.
    """
    return precision_recall_fscore_support(
        y_true, y_pred, average=average, pos_label=pos_label,
        zero_division=zero_division
    )[2]


def _check_roc_vectors(y_true, y_score):
    """Validate that both roc inputs are non-empty lists of equal length."""
    if not isinstance(y_true, list) or not isinstance(y_score, list):
        raise ValueError("y_true and y_score must be lists")
    if len(y_true) == 0 or len(y_score) == 0:
        raise ValueError("y_true and y_score must be non-empty lists")
    if len(y_true) != len(y_score):
        raise ValueError("y_true and y_score must have the same length")
    return len(y_true)


def _check_finite_score(value, name):
    """Validate one finite score/weight of type exactly int or float."""
    if type(value) not in (int, float):
        raise ValueError(
            "%s must contain only finite non-boolean numbers" % name
        )
    # math.isfinite raises OverflowError for ints too large to convert
    # to float; such values fail the finite requirement.
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


def _roc_float(value):
    """Convert a validated score/weight to float; overflow, invalid
    operations, and non-finite results raise FloatingPointError."""
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise FloatingPointError(
            "non-finite value encountered during roc computation"
        ) from exc
    if not math.isfinite(result):
        raise FloatingPointError(
            "non-finite value encountered during roc computation"
        )
    return result


def roc_curve(y_true, y_score, pos_label=1, sample_weight=None):
    """Compute false/true positive rates at descending score thresholds.

    ``y_true`` and ``y_score`` must be non-empty lists of equal length.
    ``y_true`` must contain values of type exactly ``int`` (booleans are
    rejected) drawn from exactly two distinct labels, and ``pos_label``
    must be an exact ``int`` equal to one of them; the other label is the
    negative class. ``y_score`` must contain finite values of type exactly
    ``int`` or ``float`` (booleans are rejected). ``sample_weight`` must be
    ``None`` -- every sample then weighs ``1.0`` -- or a list of the same
    length whose elements are finite non-negative values of type exactly
    ``int`` or ``float``. The total weight of each class must be greater
    than zero. Any violation raises ValueError.

    The thresholds are ``[math.inf]`` followed by the distinct score
    values in descending order, each converted to ``float``. ``fpr`` and
    ``tpr`` start at ``0.0``; each subsequent entry divides, by the
    corresponding class total, the ``math.fsum`` of the weights -- taken
    in input order -- of the negative-class (``fpr``) or positive-class
    (``tpr``) samples whose score is greater than or equal to the
    threshold. An exact zero rate is normalized to ``0.0``. Overflow,
    invalid operations, or non-finite intermediate values after
    validation raise FloatingPointError.

    The return value is ``(fpr, tpr, thresholds)``, three lists of
    floats. The inputs are not modified. Deterministic: same inputs,
    same result.
    """
    n = _check_roc_vectors(y_true, y_score)
    for value in y_true:
        if type(value) is not int:
            raise ValueError("y_true must contain only integers")
    if type(pos_label) is not int:
        raise ValueError("pos_label must be an integer")
    labels = set(y_true)
    if len(labels) != 2 or pos_label not in labels:
        raise ValueError(
            "y_true must contain exactly two distinct labels with "
            "pos_label among them"
        )
    for value in y_score:
        _check_finite_score(value, "y_score")

    if sample_weight is None:
        weights = [1.0] * n
    else:
        if not isinstance(sample_weight, list) or len(sample_weight) != n:
            raise ValueError(
                "sample_weight must be a list with the same length as "
                "y_true"
            )
        for value in sample_weight:
            _check_finite_score(value, "sample_weight")
            if value < 0:
                raise ValueError(
                    "sample_weight must contain only finite non-negative "
                    "non-boolean numbers"
                )
        weights = [_roc_float(value) for value in sample_weight]

    pos_terms_total = []
    neg_terms_total = []
    for i in range(n):
        if y_true[i] == pos_label:
            pos_terms_total.append(weights[i])
        else:
            neg_terms_total.append(weights[i])
    try:
        pos_total = math.fsum(pos_terms_total)
        neg_total = math.fsum(neg_terms_total)
    except (OverflowError, ValueError) as exc:
        raise FloatingPointError(
            "non-finite value encountered during roc computation"
        ) from exc
    if not math.isfinite(pos_total) or not math.isfinite(neg_total):
        raise FloatingPointError(
            "non-finite value encountered during roc computation"
        )
    if pos_total <= 0.0 or neg_total <= 0.0:
        raise ValueError(
            "the total weight of each class must be greater than 0"
        )

    thresholds = [math.inf]
    for value in sorted(set(y_score), reverse=True):
        thresholds.append(_roc_float(value))

    fpr = [0.0]
    tpr = [0.0]
    for threshold in thresholds[1:]:
        neg_terms = []
        pos_terms = []
        for i in range(n):
            if y_score[i] >= threshold:
                if y_true[i] == pos_label:
                    pos_terms.append(weights[i])
                else:
                    neg_terms.append(weights[i])
        try:
            fp = math.fsum(neg_terms)
            tp = math.fsum(pos_terms)
            fpr_value = fp / neg_total
            tpr_value = tp / pos_total
        except (OverflowError, ValueError) as exc:
            raise FloatingPointError(
                "non-finite value encountered during roc computation"
            ) from exc
        if not math.isfinite(fpr_value) or not math.isfinite(tpr_value):
            raise FloatingPointError(
                "non-finite value encountered during roc computation"
            )
        if fpr_value == 0:
            fpr_value = 0.0
        if tpr_value == 0:
            tpr_value = 0.0
        fpr.append(fpr_value)
        tpr.append(tpr_value)
    return fpr, tpr, thresholds


def roc_auc_score(y_true, y_score, pos_label=1, sample_weight=None):
    """Return the area under :func:`roc_curve` for the same arguments.

    Validation and exceptions are exactly those of :func:`roc_curve`.
    The area is the ``math.fsum`` -- in curve order -- of the trapezoid
    terms ``(fpr[k + 1] - fpr[k]) * (tpr[k] + tpr[k + 1]) / 2`` over
    adjacent curve points; an exact zero result is normalized to ``0.0``.
    Overflow, invalid operations, or non-finite intermediate values
    raise FloatingPointError.

    The return value is a float. The inputs are not modified.
    Deterministic: same inputs, same result.
    """
    fpr, tpr, _ = roc_curve(
        y_true, y_score, pos_label=pos_label, sample_weight=sample_weight
    )
    terms = []
    for k in range(len(fpr) - 1):
        try:
            term = (fpr[k + 1] - fpr[k]) * (tpr[k] + tpr[k + 1]) / 2
        except (OverflowError, ValueError) as exc:
            raise FloatingPointError(
                "non-finite value encountered during roc auc computation"
            ) from exc
        if not math.isfinite(term):
            raise FloatingPointError(
                "non-finite value encountered during roc auc computation"
            )
        terms.append(term)
    try:
        auc = math.fsum(terms)
    except (OverflowError, ValueError) as exc:
        raise FloatingPointError(
            "non-finite value encountered during roc auc computation"
        ) from exc
    if not math.isfinite(auc):
        raise FloatingPointError(
            "non-finite value encountered during roc auc computation"
        )
    if auc == 0:
        auc = 0.0
    return auc


def _pr_float(value):
    """Convert a validated score/weight to float; overflow, invalid
    operations, and non-finite results raise FloatingPointError."""
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise FloatingPointError(
            "non-finite value encountered during precision-recall "
            "computation"
        ) from exc
    if not math.isfinite(result):
        raise FloatingPointError(
            "non-finite value encountered during precision-recall "
            "computation"
        )
    return result


def precision_recall_curve(y_true, y_score, pos_label=1, sample_weight=None):
    """Compute precision/recall pairs at descending score thresholds.

    ``y_true`` and ``y_score`` must be non-empty lists of equal length.
    ``y_true`` must contain values of type exactly ``int`` (booleans are
    rejected) drawn from exactly two distinct labels, and ``pos_label``
    must be an exact ``int`` equal to one of them; the other label is the
    negative class. ``y_score`` must contain finite values of type exactly
    ``int`` or ``float`` (booleans are rejected). ``sample_weight`` must be
    ``None`` -- every sample then weighs ``1.0`` -- or a list of the same
    length whose elements are finite non-negative values of type exactly
    ``int`` or ``float``. The total weight of the positive class must be
    greater than zero. Any violation raises ValueError.

    The thresholds are the distinct score values in descending order,
    each converted to ``float``. ``precision`` and ``recall`` start at
    ``1.0`` and ``0.0``; each subsequent entry is computed from the
    ``math.fsum`` of the weights -- taken in input order -- of the samples
    whose score is greater than or equal to the threshold: precision is
    ``tp / (tp + fp)`` (``1.0`` when the denominator is zero) and recall
    is ``tp`` divided by the total positive-class weight. An exact zero
    value is normalized to ``0.0``. Overflow, invalid operations, or
    non-finite intermediate values after validation raise
    FloatingPointError.

    The return value is ``(precision, recall, thresholds)``: the first two
    are lists of floats one element longer than the thresholds list. The
    inputs are not modified. Deterministic: same inputs, same result.
    """
    n = _check_roc_vectors(y_true, y_score)
    for value in y_true:
        if type(value) is not int:
            raise ValueError("y_true must contain only integers")
    if type(pos_label) is not int:
        raise ValueError("pos_label must be an integer")
    labels = set(y_true)
    if len(labels) != 2 or pos_label not in labels:
        raise ValueError(
            "y_true must contain exactly two distinct labels with "
            "pos_label among them"
        )
    for value in y_score:
        _check_finite_score(value, "y_score")

    if sample_weight is None:
        weights = [1.0] * n
    else:
        if not isinstance(sample_weight, list) or len(sample_weight) != n:
            raise ValueError(
                "sample_weight must be a list with the same length as "
                "y_true"
            )
        for value in sample_weight:
            _check_finite_score(value, "sample_weight")
            if value < 0:
                raise ValueError(
                    "sample_weight must contain only finite non-negative "
                    "non-boolean numbers"
                )
        weights = [_pr_float(value) for value in sample_weight]

    pos_terms_total = []
    for i in range(n):
        if y_true[i] == pos_label:
            pos_terms_total.append(weights[i])
    try:
        pos_total = math.fsum(pos_terms_total)
    except (OverflowError, ValueError) as exc:
        raise FloatingPointError(
            "non-finite value encountered during precision-recall "
            "computation"
        ) from exc
    if not math.isfinite(pos_total):
        raise FloatingPointError(
            "non-finite value encountered during precision-recall "
            "computation"
        )
    if pos_total <= 0.0:
        raise ValueError(
            "the total weight of the positive class must be greater than 0"
        )

    thresholds = []
    for value in sorted(set(y_score), reverse=True):
        thresholds.append(_pr_float(value))

    precisions = [1.0]
    recalls = [0.0]
    for threshold in thresholds:
        fp_terms = []
        tp_terms = []
        for i in range(n):
            if y_score[i] >= threshold:
                if y_true[i] == pos_label:
                    tp_terms.append(weights[i])
                else:
                    fp_terms.append(weights[i])
        try:
            fp = math.fsum(fp_terms)
            tp = math.fsum(tp_terms)
            denominator = tp + fp
            if not math.isfinite(denominator):
                raise FloatingPointError(
                    "non-finite value encountered during precision-recall "
                    "computation"
                )
            if denominator == 0.0:
                precision = 1.0
            else:
                precision = tp / denominator
            recall = tp / pos_total
        except (OverflowError, ValueError) as exc:
            raise FloatingPointError(
                "non-finite value encountered during precision-recall "
                "computation"
            ) from exc
        if not math.isfinite(precision) or not math.isfinite(recall):
            raise FloatingPointError(
                "non-finite value encountered during precision-recall "
                "computation"
            )
        if precision == 0:
            precision = 0.0
        if recall == 0:
            recall = 0.0
        precisions.append(precision)
        recalls.append(recall)
    return precisions, recalls, thresholds


def average_precision_score(y_true, y_score, pos_label=1, sample_weight=None):
    """Return the stepwise area under :func:`precision_recall_curve` for
    the same arguments.

    Validation and exceptions are exactly those of
    :func:`precision_recall_curve`. The score is the ``math.fsum`` -- over
    threshold indices ``i`` in ascending order -- of the terms
    ``(recall[i + 1] - recall[i]) * precision[i + 1]``; an exact zero
    result is normalized to ``0.0``. Overflow, invalid operations, or
    non-finite intermediate values raise FloatingPointError.

    The return value is a float. The inputs are not modified.
    Deterministic: same inputs, same result.
    """
    precisions, recalls, thresholds = precision_recall_curve(
        y_true, y_score, pos_label=pos_label, sample_weight=sample_weight
    )
    terms = []
    for i in range(len(thresholds)):
        try:
            term = (recalls[i + 1] - recalls[i]) * precisions[i + 1]
        except (OverflowError, ValueError) as exc:
            raise FloatingPointError(
                "non-finite value encountered during average-precision "
                "computation"
            ) from exc
        if not math.isfinite(term):
            raise FloatingPointError(
                "non-finite value encountered during average-precision "
                "computation"
            )
        terms.append(term)
    try:
        ap = math.fsum(terms)
    except (OverflowError, ValueError) as exc:
        raise FloatingPointError(
            "non-finite value encountered during average-precision "
            "computation"
        ) from exc
    if not math.isfinite(ap):
        raise FloatingPointError(
            "non-finite value encountered during average-precision "
            "computation"
        )
    if ap == 0:
        ap = 0.0
    return ap


def _cal_float(value):
    """Convert a validated calibration probability/weight to float;
    overflow, invalid operations, and non-finite results raise
    FloatingPointError."""
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise FloatingPointError(
            "non-finite value encountered during calibration computation"
        ) from exc
    if not math.isfinite(result):
        raise FloatingPointError(
            "non-finite value encountered during calibration computation"
        )
    return result


def calibration_curve(
    y_true, y_prob, n_bins=5, pos_label=1, sample_weight=None
):
    """Compute per-bin positive-class fractions and mean probabilities.

    ``y_true`` and ``y_prob`` must be non-empty lists of equal length.
    ``y_true`` must contain values of type exactly ``int`` (booleans are
    rejected) drawn from exactly two distinct labels, and ``pos_label``
    must be an exact ``int`` equal to one of them; the other label is the
    negative class. ``y_prob`` must contain finite values in the closed
    interval ``[0, 1]`` whose type is exactly ``int`` or ``float``
    (booleans are rejected; the only accepted ``int`` values are ``0``
    and ``1``). ``n_bins`` must be a positive exact ``int``.
    ``sample_weight`` must be ``None`` -- every sample then weighs
    ``1.0`` -- or a list of the same length whose elements are finite
    non-negative values of type exactly ``int`` or ``float`` (booleans
    are rejected). The total weight must be greater than zero. Any
    violation (including overflow during the finiteness checks) raises
    ValueError.

    Equal-width bins are indexed ``min(int(p * n_bins), n_bins - 1)``.
    Bins are visited in ascending bin index; within a bin, samples are
    accumulated in input order with ``math.fsum`` to obtain the total
    weight ``W``, the positive-class weight ``T``, and the weighted
    probability sum ``P`` (the ``math.fsum`` of the ``w * p`` products).
    Bins with ``W == 0`` are omitted; each retained bin appends ``T / W``
    to the first returned list and ``P / W`` to the second, with an exact
    zero normalized to ``0.0``. Overflow, invalid operations during
    post-validation conversion or arithmetic, and non-finite results
    raise FloatingPointError.

    The return value is ``(fraction_of_positives, mean_predicted_value)``,
    two lists of floats. The inputs are not modified. Deterministic: same
    inputs, same result.
    """
    if not isinstance(y_true, list) or not isinstance(y_prob, list):
        raise ValueError("y_true and y_prob must be lists")
    if len(y_true) == 0 or len(y_prob) == 0:
        raise ValueError("y_true and y_prob must be non-empty lists")
    if len(y_true) != len(y_prob):
        raise ValueError("y_true and y_prob must have the same length")
    n = len(y_true)

    for value in y_true:
        if type(value) is not int:
            raise ValueError("y_true must contain only integers")
    if type(pos_label) is not int:
        raise ValueError("pos_label must be an integer")
    labels = set(y_true)
    if len(labels) != 2 or pos_label not in labels:
        raise ValueError(
            "y_true must contain exactly two distinct labels with "
            "pos_label among them"
        )

    for value in y_prob:
        if type(value) not in (int, float):
            raise ValueError(
                "y_prob must contain only finite numbers in [0, 1]"
            )
        # math.isfinite raises OverflowError for ints too large to
        # convert to float; such values fail the finite requirement.
        try:
            finite = math.isfinite(value)
        except OverflowError as exc:
            raise ValueError(
                "y_prob must contain only finite numbers in [0, 1]"
            ) from exc
        if not finite or value < 0 or value > 1:
            raise ValueError(
                "y_prob must contain only finite numbers in [0, 1]"
            )

    if type(n_bins) is not int or n_bins < 1:
        raise ValueError("n_bins must be a positive integer")

    if sample_weight is None:
        weights = [1.0] * n
    else:
        if not isinstance(sample_weight, list) or len(sample_weight) != n:
            raise ValueError(
                "sample_weight must be a list with the same length as "
                "y_true"
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
        weights = [_cal_float(value) for value in sample_weight]

    try:
        total_weight = math.fsum(weights)
    except (OverflowError, ValueError) as exc:
        raise FloatingPointError(
            "non-finite value encountered during calibration computation"
        ) from exc
    if not math.isfinite(total_weight):
        raise FloatingPointError(
            "non-finite value encountered during calibration computation"
        )
    if total_weight <= 0.0:
        raise ValueError("the total weight must be greater than 0")

    # Each occupied bin maps to its W, T, and w*p term lists, all kept in
    # input order. A dict avoids allocating n_bins lists (n_bins is
    # unbounded) while still letting bins be emitted in index order.
    bins = {}
    last_bin = n_bins - 1
    for i in range(n):
        p = y_prob[i]
        w = weights[i]
        try:
            scaled = p * n_bins
            if isinstance(scaled, float) and not math.isfinite(scaled):
                raise FloatingPointError(
                    "non-finite value encountered during calibration "
                    "computation"
                )
            bin_index = min(int(scaled), last_bin)
            wp = w * p
        except (OverflowError, ValueError) as exc:
            raise FloatingPointError(
                "non-finite value encountered during calibration "
                "computation"
            ) from exc
        if isinstance(wp, float) and not math.isfinite(wp):
            raise FloatingPointError(
                "non-finite value encountered during calibration "
                "computation"
            )

        entry = bins.get(bin_index)
        if entry is None:
            entry = ([], [], [])
            bins[bin_index] = entry
        entry[0].append(w)
        if y_true[i] == pos_label:
            entry[1].append(w)
        entry[2].append(wp)

    fractions = []
    mean_probs = []
    for bin_index in sorted(bins):
        w_terms, pos_terms, wp_terms = bins[bin_index]
        try:
            bin_weight = math.fsum(w_terms)
            pos_weight = math.fsum(pos_terms)
            prob_weight = math.fsum(wp_terms)
        except (OverflowError, ValueError) as exc:
            raise FloatingPointError(
                "non-finite value encountered during calibration "
                "computation"
            ) from exc
        if (
            not math.isfinite(bin_weight)
            or not math.isfinite(pos_weight)
            or not math.isfinite(prob_weight)
        ):
            raise FloatingPointError(
                "non-finite value encountered during calibration "
                "computation"
            )
        if bin_weight == 0.0:
            continue
        try:
            fraction = pos_weight / bin_weight
            mean_prob = prob_weight / bin_weight
        except (OverflowError, ValueError) as exc:
            raise FloatingPointError(
                "non-finite value encountered during calibration "
                "computation"
            ) from exc
        if not math.isfinite(fraction) or not math.isfinite(mean_prob):
            raise FloatingPointError(
                "non-finite value encountered during calibration "
                "computation"
            )
        if fraction == 0:
            fraction = 0.0
        if mean_prob == 0:
            mean_prob = 0.0
        fractions.append(fraction)
        mean_probs.append(mean_prob)
    return fractions, mean_probs


def _brier_float(value):
    """Convert a validated probability/weight to float; overflow, invalid
    operations, and non-finite results raise FloatingPointError."""
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise FloatingPointError(
            "non-finite value encountered during brier-score computation"
        ) from exc
    if not math.isfinite(result):
        raise FloatingPointError(
            "non-finite value encountered during brier-score computation"
        )
    return result


def brier_score_loss(y_true, y_prob, pos_label=1, sample_weight=None):
    """Compute the weighted mean squared probability error.

    ``y_true`` and ``y_prob`` must be non-empty lists of equal length.
    ``y_true`` must contain values of type exactly ``int`` (booleans are
    rejected) drawn from exactly two distinct labels, and ``pos_label``
    must be an exact ``int`` equal to one of them. ``y_prob`` must contain
    finite values in the closed interval ``[0, 1]`` whose type is exactly
    ``int`` or ``float`` (booleans are rejected). ``sample_weight`` must be
    ``None`` -- every sample then weighs ``1.0`` -- or a list of the same
    length whose elements are finite non-negative values of type exactly
    ``int`` or ``float`` (booleans are rejected). The total weight must be
    greater than zero. Any violation (including overflow during the
    finiteness checks) raises ValueError.

    After validation, the probabilities and weights are converted to
    ``float``; in input order, ``t_i`` is ``1.0`` when the label equals
    ``pos_label`` and ``0.0`` otherwise. The result is the quotient of two
    ``math.fsum`` sums -- one of ``w_i * (p_i - t_i) ** 2``, the other of
    the ``w_i`` -- with an exact zero normalized to ``0.0``. Overflow,
    invalid operations during the post-validation conversion or
    arithmetic, and non-finite intermediate values or results raise
    FloatingPointError.

    The return value is a float. The inputs are not modified.
    Deterministic: same inputs, same result.
    """
    if not isinstance(y_true, list) or not isinstance(y_prob, list):
        raise ValueError("y_true and y_prob must be lists")
    if len(y_true) == 0 or len(y_prob) == 0:
        raise ValueError("y_true and y_prob must be non-empty lists")
    if len(y_true) != len(y_prob):
        raise ValueError("y_true and y_prob must have the same length")
    n = len(y_true)

    for value in y_true:
        if type(value) is not int:
            raise ValueError("y_true must contain only integers")
    if type(pos_label) is not int:
        raise ValueError("pos_label must be an integer")
    labels = set(y_true)
    if len(labels) != 2 or pos_label not in labels:
        raise ValueError(
            "y_true must contain exactly two distinct labels with "
            "pos_label among them"
        )

    for value in y_prob:
        if type(value) not in (int, float):
            raise ValueError(
                "y_prob must contain only finite numbers in [0, 1]"
            )
        # math.isfinite raises OverflowError for ints too large to
        # convert to float; such values fail the finite requirement.
        try:
            finite = math.isfinite(value)
        except OverflowError as exc:
            raise ValueError(
                "y_prob must contain only finite numbers in [0, 1]"
            ) from exc
        if not finite or value < 0 or value > 1:
            raise ValueError(
                "y_prob must contain only finite numbers in [0, 1]"
            )

    if sample_weight is None:
        weights = [1.0] * n
    else:
        if not isinstance(sample_weight, list) or len(sample_weight) != n:
            raise ValueError(
                "sample_weight must be a list with the same length as "
                "y_true"
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
        weights = [_brier_float(value) for value in sample_weight]

    try:
        total_weight = math.fsum(weights)
    except (OverflowError, ValueError) as exc:
        raise FloatingPointError(
            "non-finite value encountered during brier-score computation"
        ) from exc
    if not math.isfinite(total_weight):
        raise FloatingPointError(
            "non-finite value encountered during brier-score computation"
        )
    if total_weight <= 0.0:
        raise ValueError("the total weight must be greater than 0")

    error_terms = []
    for i in range(n):
        p = _brier_float(y_prob[i])
        t = 1.0 if y_true[i] == pos_label else 0.0
        try:
            residual = p - t
            term = weights[i] * residual ** 2
        except (OverflowError, ValueError) as exc:
            raise FloatingPointError(
                "non-finite value encountered during brier-score computation"
            ) from exc
        if not math.isfinite(term):
            raise FloatingPointError(
                "non-finite value encountered during brier-score computation"
            )
        error_terms.append(term)

    try:
        total_error = math.fsum(error_terms)
        loss = total_error / total_weight
    except (OverflowError, ValueError) as exc:
        raise FloatingPointError(
            "non-finite value encountered during brier-score computation"
        ) from exc
    if not math.isfinite(total_error) or not math.isfinite(loss):
        raise FloatingPointError(
            "non-finite value encountered during brier-score computation"
        )
    if loss == 0:
        loss = 0.0
    return loss


def _log_loss_float(value):
    """Convert a validated probability/weight to float; overflow, invalid
    operations, and non-finite results raise FloatingPointError."""
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise FloatingPointError(
            "non-finite value encountered during log-loss computation"
        ) from exc
    if not math.isfinite(result):
        raise FloatingPointError(
            "non-finite value encountered during log-loss computation"
        )
    return result


def log_loss(y_true, y_prob, pos_label=1, sample_weight=None):
    """Compute the weighted logistic (cross-entropy) loss.

    ``y_true`` and ``y_prob`` must be non-empty lists of equal length.
    ``y_true`` must contain values of type exactly ``int`` (booleans are
    rejected) drawn from exactly two distinct labels, and ``pos_label``
    must be an exact ``int`` equal to one of them. ``y_prob`` must contain
    finite values in the closed interval ``[0, 1]`` whose type is exactly
    ``int`` or ``float`` (booleans are rejected). ``sample_weight`` must be
    ``None`` -- every sample then weighs ``1.0`` -- or a list of the same
    length whose elements are finite non-negative values of type exactly
    ``int`` or ``float`` (booleans are rejected). The total weight must be
    greater than zero. Any violation (including overflow during the
    finiteness checks) raises ValueError.

    After validation, the probabilities and weights are converted to
    ``float`` and each probability is clamped to
    ``[1e-15, 1 - 1e-15]``; in input order, ``t_i`` is true when the label
    equals ``pos_label``. The per-sample loss is ``-math.log(p_i)`` when
    ``t_i`` is true and ``-math.log1p(-p_i)`` otherwise. The result is the
    quotient of two ``math.fsum`` sums -- one of ``w_i * loss_i``, the
    other of the ``w_i`` -- with an exact zero normalized to ``0.0``.
    Overflow, invalid operations during the post-validation conversion or
    arithmetic, and non-finite intermediate values or results raise
    FloatingPointError.

    The return value is a float. The inputs are not modified.
    Deterministic: same inputs, same result.
    """
    if not isinstance(y_true, list) or not isinstance(y_prob, list):
        raise ValueError("y_true and y_prob must be lists")
    if len(y_true) == 0 or len(y_prob) == 0:
        raise ValueError("y_true and y_prob must be non-empty lists")
    if len(y_true) != len(y_prob):
        raise ValueError("y_true and y_prob must have the same length")
    n = len(y_true)

    for value in y_true:
        if type(value) is not int:
            raise ValueError("y_true must contain only integers")
    if type(pos_label) is not int:
        raise ValueError("pos_label must be an integer")
    labels = set(y_true)
    if len(labels) != 2 or pos_label not in labels:
        raise ValueError(
            "y_true must contain exactly two distinct labels with "
            "pos_label among them"
        )

    for value in y_prob:
        if type(value) not in (int, float):
            raise ValueError(
                "y_prob must contain only finite numbers in [0, 1]"
            )
        # math.isfinite raises OverflowError for ints too large to
        # convert to float; such values fail the finite requirement.
        try:
            finite = math.isfinite(value)
        except OverflowError as exc:
            raise ValueError(
                "y_prob must contain only finite numbers in [0, 1]"
            ) from exc
        if not finite or value < 0 or value > 1:
            raise ValueError(
                "y_prob must contain only finite numbers in [0, 1]"
            )

    if sample_weight is None:
        weights = [1.0] * n
    else:
        if not isinstance(sample_weight, list) or len(sample_weight) != n:
            raise ValueError(
                "sample_weight must be a list with the same length as "
                "y_true"
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
        weights = [_log_loss_float(value) for value in sample_weight]

    try:
        total_weight = math.fsum(weights)
    except (OverflowError, ValueError) as exc:
        raise FloatingPointError(
            "non-finite value encountered during log-loss computation"
        ) from exc
    if not math.isfinite(total_weight):
        raise FloatingPointError(
            "non-finite value encountered during log-loss computation"
        )
    if total_weight <= 0.0:
        raise ValueError("the total weight must be greater than 0")

    loss_terms = []
    for i in range(n):
        p = _log_loss_float(y_prob[i])
        p = min(max(p, 1e-15), 1.0 - 1e-15)
        t = y_true[i] == pos_label
        try:
            if t:
                sample_loss = -math.log(p)
            else:
                sample_loss = -math.log1p(-p)
            term = weights[i] * sample_loss
        except (OverflowError, ValueError) as exc:
            raise FloatingPointError(
                "non-finite value encountered during log-loss computation"
            ) from exc
        if not math.isfinite(term):
            raise FloatingPointError(
                "non-finite value encountered during log-loss computation"
            )
        loss_terms.append(term)

    try:
        total_loss = math.fsum(loss_terms)
        loss = total_loss / total_weight
    except (OverflowError, ValueError) as exc:
        raise FloatingPointError(
            "non-finite value encountered during log-loss computation"
        ) from exc
    if not math.isfinite(total_loss) or not math.isfinite(loss):
        raise FloatingPointError(
            "non-finite value encountered during log-loss computation"
        )
    if loss == 0:
        loss = 0.0
    return loss


def _multiclass_log_loss_float(value):
    """Convert a validated probability/weight to float; overflow, invalid
    operations, and non-finite results raise FloatingPointError."""
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise FloatingPointError(
            "non-finite value encountered during multiclass-log-loss "
            "computation"
        ) from exc
    if not math.isfinite(result):
        raise FloatingPointError(
            "non-finite value encountered during multiclass-log-loss "
            "computation"
        )
    return result


def multiclass_log_loss(y_true, y_prob, sample_weight=None) -> float:
    """Compute the weighted multiclass logistic (cross-entropy) loss.

    ``y_true`` and ``y_prob`` must be non-empty lists of equal length.
    ``y_true`` must contain values of type exactly ``int`` (booleans are
    rejected) drawn from at least two distinct labels; the classes are the
    sorted distinct labels, in ascending order, and each column of
    ``y_prob`` corresponds to one class in that order. Every ``y_prob``
    row must be a list whose length equals the number of classes and whose
    elements are finite values in the closed interval ``[0, 1]`` with type
    exactly ``int`` or ``float`` (booleans are rejected); the elements of
    each row, in column order, must sum to exactly ``1.0`` under
    ``math.fsum``. ``sample_weight`` must be ``None`` -- every sample then
    weighs ``1.0`` -- or a list of the same length whose elements are
    finite non-negative values of type exactly ``int`` or ``float``
    (booleans are rejected). The total weight must be greater than zero.
    Any violation -- container, length, class, shape, type, range,
    row-sum, total-weight, or finiteness checks, including overflow during
    those checks -- raises ValueError.

    After validation, the probabilities and weights are converted to
    ``float`` and the converted weights are summed with ``math.fsum``;
    overflow or an invalid operation during that summation, a non-finite
    total, or a total that is not greater than zero raises ValueError.
    For each sample, in input order, the true class is located by
    ascending class order, its probability ``p`` is clamped to
    ``[1e-15, 1 - 1e-15]``, and the per-sample loss is ``-math.log(p)``.
    The result is the quotient of two ``math.fsum`` sums, both accumulated
    in input order -- one of ``w_i * loss_i``, the other of the ``w_i``
    -- with an exact zero normalized to ``0.0``. Overflow or invalid
    operations during the subsequent conversion, multiplication,
    logarithm, summation, or division, and non-finite intermediate values
    or results, raise FloatingPointError.

    The return value is a float. The inputs are not modified.
    Deterministic: same inputs, same result.
    """
    if not isinstance(y_true, list) or not isinstance(y_prob, list):
        raise ValueError("y_true and y_prob must be lists")
    if len(y_true) == 0 or len(y_prob) == 0:
        raise ValueError("y_true and y_prob must be non-empty lists")
    if len(y_true) != len(y_prob):
        raise ValueError("y_true and y_prob must have the same length")
    n = len(y_true)

    for value in y_true:
        if type(value) is not int:
            raise ValueError("y_true must contain only integers")
    classes = sorted(set(y_true))
    if len(classes) < 2:
        raise ValueError("y_true must contain at least two distinct classes")
    n_classes = len(classes)
    class_index = {label: j for j, label in enumerate(classes)}

    for row in y_prob:
        if not isinstance(row, list):
            raise ValueError("y_prob rows must be lists")
        if len(row) != n_classes:
            raise ValueError(
                "each y_prob row must have one entry per class"
            )
        for value in row:
            if type(value) not in (int, float):
                raise ValueError(
                    "y_prob must contain only finite numbers in [0, 1]"
                )
            # math.isfinite raises OverflowError for ints too large to
            # convert to float; such values fail the finite requirement.
            try:
                finite = math.isfinite(value)
            except OverflowError as exc:
                raise ValueError(
                    "y_prob must contain only finite numbers in [0, 1]"
                ) from exc
            if not finite or value < 0 or value > 1:
                raise ValueError(
                    "y_prob must contain only finite numbers in [0, 1]"
                )
        # The columns of each row must describe a probability distribution.
        try:
            row_sum = math.fsum(row)
        except OverflowError as exc:
            raise ValueError(
                "each y_prob row must sum to exactly 1.0"
            ) from exc
        if row_sum != 1.0:
            raise ValueError(
                "each y_prob row must sum to exactly 1.0"
            )

    if sample_weight is None:
        weights = [1.0] * n
    else:
        if not isinstance(sample_weight, list) or len(sample_weight) != n:
            raise ValueError(
                "sample_weight must be a list with the same length as "
                "y_true"
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
        weights = [_multiclass_log_loss_float(value) for value in sample_weight]

    # The weights have just been converted to float, so summing them is
    # still part of the input checks: overflow, invalid operations, a
    # non-finite total, and a non-positive total all raise ValueError.
    try:
        total_weight = math.fsum(weights)
    except (OverflowError, ValueError) as exc:
        raise ValueError("the total weight must be greater than 0") from exc
    if not math.isfinite(total_weight) or total_weight <= 0.0:
        raise ValueError("the total weight must be greater than 0")

    loss_terms = []
    for i in range(n):
        j = class_index[y_true[i]]
        p = _multiclass_log_loss_float(y_prob[i][j])
        p = min(max(p, 1e-15), 1.0 - 1e-15)
        try:
            sample_loss = -math.log(p)
            term = weights[i] * sample_loss
        except (OverflowError, ValueError) as exc:
            raise FloatingPointError(
                "non-finite value encountered during multiclass-log-loss "
                "computation"
            ) from exc
        if not math.isfinite(term):
            raise FloatingPointError(
                "non-finite value encountered during multiclass-log-loss "
                "computation"
            )
        loss_terms.append(term)

    try:
        total_loss = math.fsum(loss_terms)
        loss = total_loss / total_weight
    except (OverflowError, ValueError) as exc:
        raise FloatingPointError(
            "non-finite value encountered during multiclass-log-loss "
            "computation"
        ) from exc
    if not math.isfinite(total_loss) or not math.isfinite(loss):
        raise FloatingPointError(
            "non-finite value encountered during multiclass-log-loss "
            "computation"
        )
    if loss == 0:
        loss = 0.0
    return loss


def silhouette_score(X, labels):
    """Compute the mean silhouette coefficient of a clustering.

    ``X`` must be a non-empty rectangular ``list`` of non-empty ``list``
    rows whose elements are finite values of type exactly ``int`` or
    ``float`` (booleans and subclasses are rejected). ``labels`` must be
    a ``list`` of the same length whose elements have type exactly
    ``int``, and the number of distinct labels must lie in
    ``[2, len(X) - 1]``. Any violation -- including overflow during the
    finiteness checks -- raises ValueError.

    The distance between samples ``i`` and ``j`` is
    ``sqrt(math.fsum((X[i][k] - X[j][k]) ** 2))`` with terms accumulated
    in column order. Clusters are the label groups in ascending label
    order, keeping input order within each cluster. When the cluster of
    sample ``i`` holds only ``i``, ``s_i`` is ``0.0``; otherwise ``a_i``
    is the mean distance from ``i`` to the other samples of its cluster
    and ``b_i`` is the smallest, over the other clusters, mean distance
    from ``i`` to a cluster's samples, with every sum taken by
    ``math.fsum`` in cluster input order. With ``m = max(a_i, b_i)``,
    ``s_i`` is ``0.0`` when ``m == 0`` and ``(b_i - a_i) / m`` otherwise.
    The result is ``math.fsum(s_i in input order) / len(X)`` with an
    exact zero normalized to ``0.0``. Overflow, invalid operations, or
    non-finite intermediate values after validation raise
    FloatingPointError.

    The return value is a float. The inputs are not modified.
    Deterministic: same inputs, same result.
    """
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
            if type(value) not in (int, float):
                raise ValueError(
                    "X must contain only finite non-boolean numbers"
                )
            # math.isfinite raises OverflowError for ints too large to
            # convert to float; such values fail the finite requirement.
            try:
                finite = math.isfinite(value)
            except OverflowError as exc:
                raise ValueError(
                    "X must contain only finite non-boolean numbers"
                ) from exc
            if not finite:
                raise ValueError(
                    "X must contain only finite non-boolean numbers"
                )
    n = len(X)
    if not isinstance(labels, list) or len(labels) != n:
        raise ValueError("labels must be a list with the same length as X")
    for value in labels:
        if type(value) is not int:
            raise ValueError("labels must contain only integers")
    distinct = sorted(set(labels))
    if len(distinct) < 2 or len(distinct) > n - 1:
        raise ValueError(
            "labels must contain between 2 and len(X) - 1 distinct labels"
        )

    clusters = {}
    for i in range(n):
        clusters.setdefault(labels[i], []).append(i)
    groups = [clusters[label] for label in distinct]

    def fail():
        raise FloatingPointError(
            "non-finite value encountered during silhouette score"
        )

    # Pairwise distances; d(i, j) == d(j, i) exactly, so each unordered
    # pair is computed once with the smaller index first.
    distances = [[0.0] * n for _ in range(n)]
    for i in range(n):
        row_i = X[i]
        for j in range(i + 1, n):
            row_j = X[j]
            terms = []
            for k in range(width):
                try:
                    diff = row_i[k] - row_j[k]
                except (OverflowError, ValueError) as exc:
                    raise FloatingPointError(
                        "non-finite value encountered during silhouette "
                        "score"
                    ) from exc
                if isinstance(diff, float) and not math.isfinite(diff):
                    fail()
                try:
                    square = diff ** 2
                except (OverflowError, ValueError) as exc:
                    raise FloatingPointError(
                        "non-finite value encountered during silhouette "
                        "score"
                    ) from exc
                if isinstance(square, float) and not math.isfinite(square):
                    fail()
                terms.append(square)
            try:
                total = math.fsum(terms)
            except (OverflowError, ValueError) as exc:
                raise FloatingPointError(
                    "non-finite value encountered during silhouette score"
                ) from exc
            if not math.isfinite(total):
                fail()
            try:
                distance = math.sqrt(total)
            except (OverflowError, ValueError) as exc:
                raise FloatingPointError(
                    "non-finite value encountered during silhouette score"
                ) from exc
            if not math.isfinite(distance):
                fail()
            distances[i][j] = distance
            distances[j][i] = distance

    def checked_mean(total_terms, count):
        """``math.fsum(total_terms) / count`` with overflow, invalid
        operations, and non-finite values raising FloatingPointError."""
        try:
            total = math.fsum(total_terms)
        except (OverflowError, ValueError) as exc:
            raise FloatingPointError(
                "non-finite value encountered during silhouette score"
            ) from exc
        if not math.isfinite(total):
            fail()
        try:
            mean = total / count
        except (OverflowError, ValueError) as exc:
            raise FloatingPointError(
                "non-finite value encountered during silhouette score"
            ) from exc
        if not math.isfinite(mean):
            fail()
        return mean

    s_values = []
    for i in range(n):
        own = clusters[labels[i]]
        if len(own) == 1:
            s_values.append(0.0)
            continue
        a_i = checked_mean(
            [distances[i][j] for j in own if j != i], len(own) - 1
        )
        b_i = None
        for group in groups:
            if group is own:
                continue
            mean = checked_mean(
                [distances[i][j] for j in group], len(group)
            )
            if b_i is None or mean < b_i:
                b_i = mean
        m = max(a_i, b_i)
        if m == 0:
            s_values.append(0.0)
            continue
        try:
            s_i = (b_i - a_i) / m
        except (OverflowError, ValueError) as exc:
            raise FloatingPointError(
                "non-finite value encountered during silhouette score"
            ) from exc
        if not math.isfinite(s_i):
            fail()
        s_values.append(s_i)

    try:
        total = math.fsum(s_values)
    except (OverflowError, ValueError) as exc:
        raise FloatingPointError(
            "non-finite value encountered during silhouette score"
        ) from exc
    if not math.isfinite(total):
        fail()
    try:
        result = total / n
    except (OverflowError, ValueError) as exc:
        raise FloatingPointError(
            "non-finite value encountered during silhouette score"
        ) from exc
    if not math.isfinite(result):
        fail()
    if result == 0:
        result = 0.0
    return result


def davies_bouldin_score(X, labels) -> float:
    """Compute the Davies-Bouldin index of a clustering.

    ``X`` must be a non-empty rectangular ``list`` of non-empty ``list``
    rows whose elements are finite values of type exactly ``int`` or
    ``float`` (booleans and subclasses are rejected). ``labels`` must be
    a ``list`` of the same length whose elements have type exactly
    ``int``, and the number of distinct labels must lie in
    ``[2, len(X) - 1]``. Any violation -- including overflow during the
    finiteness checks -- raises ValueError.

    Clusters are the label groups in ascending label order, keeping input
    order within each cluster. The coordinates of centroid ``k`` are, in
    column order, ``math.fsum`` of that coordinate over the cluster's
    samples (in cluster input order) divided by the cluster size. The
    distance between two points is
    ``sqrt(math.fsum((a_j - b_j) ** 2))`` with terms accumulated in
    column order. ``S_k`` is the ``math.fsum`` (in cluster input order)
    of the distances from cluster ``k``'s samples to its centroid,
    divided by the cluster size. ``M_kl`` is the distance between
    centroids ``k`` and ``l``; if any pair of distinct clusters has
    coincident centroids (``M_kl == 0``), ValueError is raised. Otherwise,
    in cluster order, ``R_k = max_{l != k} (S_k + S_l) / M_kl`` and the
    result is ``math.fsum(R_k) / number_of_clusters`` with an exact zero
    normalized to ``0.0``. Overflow, invalid operations, or non-finite
    intermediate values after validation raise FloatingPointError.

    The return value is a float. The inputs are not modified.
    Deterministic: same inputs, same result.
    """
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
            if type(value) not in (int, float):
                raise ValueError(
                    "X must contain only finite non-boolean numbers"
                )
            # math.isfinite raises OverflowError for ints too large to
            # convert to float; such values fail the finite requirement.
            try:
                finite = math.isfinite(value)
            except OverflowError as exc:
                raise ValueError(
                    "X must contain only finite non-boolean numbers"
                ) from exc
            if not finite:
                raise ValueError(
                    "X must contain only finite non-boolean numbers"
                )
    n = len(X)
    if not isinstance(labels, list) or len(labels) != n:
        raise ValueError("labels must be a list with the same length as X")
    for value in labels:
        if type(value) is not int:
            raise ValueError("labels must contain only integers")
    distinct = sorted(set(labels))
    if len(distinct) < 2 or len(distinct) > n - 1:
        raise ValueError(
            "labels must contain between 2 and len(X) - 1 distinct labels"
        )

    clusters = {}
    for i in range(n):
        clusters.setdefault(labels[i], []).append(i)
    groups = [clusters[label] for label in distinct]

    def fail(exc=None):
        error = FloatingPointError(
            "non-finite value encountered during Davies-Bouldin score"
        )
        if exc is None:
            raise error
        raise error from exc

    def euclidean(a, b):
        """``sqrt(math.fsum((a_j - b_j) ** 2))`` in column order, with
        post-validation failures raising FloatingPointError."""
        terms = []
        for j in range(width):
            try:
                diff = a[j] - b[j]
                square = diff ** 2
            except (OverflowError, ValueError) as exc:
                fail(exc)
            if isinstance(diff, float) and not math.isfinite(diff):
                fail()
            if isinstance(square, float) and not math.isfinite(square):
                fail()
            terms.append(square)
        try:
            total = math.fsum(terms)
            distance = math.sqrt(total)
        except (OverflowError, ValueError) as exc:
            fail(exc)
        if not math.isfinite(total) or not math.isfinite(distance):
            fail()
        return distance

    # Centroids: each coordinate is the fsum over the cluster in input
    # order, divided by the cluster size.
    centroids = []
    for group in groups:
        size = len(group)
        coordinates = []
        for j in range(width):
            try:
                total = math.fsum(X[i][j] for i in group)
                coordinate = total / size
            except (OverflowError, ValueError) as exc:
                fail(exc)
            if not math.isfinite(total) or not math.isfinite(coordinate):
                fail()
            coordinates.append(coordinate)
        centroids.append(coordinates)

    # Within-cluster mean distances S_k (terms in cluster input order).
    scatter = []
    for k, group in enumerate(groups):
        size = len(group)
        terms = [euclidean(X[i], centroids[k]) for i in group]
        try:
            total = math.fsum(terms)
            mean = total / size
        except (OverflowError, ValueError) as exc:
            fail(exc)
        if not math.isfinite(total) or not math.isfinite(mean):
            fail()
        scatter.append(mean)

    # Centroid separations M_kl; coincident distinct centroids are invalid.
    k_clusters = len(groups)
    separations = [[0.0] * k_clusters for _ in range(k_clusters)]
    for k in range(k_clusters):
        for l in range(k + 1, k_clusters):
            distance = euclidean(centroids[k], centroids[l])
            if distance == 0.0:
                raise ValueError(
                    "distinct clusters must not have coincident centroids"
                )
            separations[k][l] = distance
            separations[l][k] = distance

    # R_k = max over l != k of (S_k + S_l) / M_kl, clusters in order.
    r_values = []
    for k in range(k_clusters):
        best = None
        for l in range(k_clusters):
            if l == k:
                continue
            try:
                numerator = scatter[k] + scatter[l]
                ratio = numerator / separations[k][l]
            except (OverflowError, ValueError) as exc:
                fail(exc)
            if not math.isfinite(numerator) or not math.isfinite(ratio):
                fail()
            if best is None or ratio > best:
                best = ratio
        r_values.append(best)

    try:
        total = math.fsum(r_values)
        result = total / k_clusters
    except (OverflowError, ValueError) as exc:
        fail(exc)
    if not math.isfinite(total) or not math.isfinite(result):
        fail()
    if result == 0:
        result = 0.0
    return result


def calinski_harabasz_score(X, labels) -> float:
    """Compute the Calinski-Harabasz index of a clustering.

    ``X`` must be a non-empty rectangular ``list`` of non-empty ``list``
    rows whose elements are finite values of type exactly ``int`` or
    ``float`` (booleans and subclasses are rejected). ``labels`` must be
    a ``list`` of the same length whose elements have type exactly
    ``int``; with ``n`` the number of samples and ``k`` the number of
    distinct labels, ``2 <= k <= n - 1`` must hold. Any violation --
    including overflow during the finiteness checks -- raises ValueError.

    Clusters are visited in ascending label order, keeping input order
    within each cluster. Column by column, ``math.fsum`` over all samples
    in input order gives the global centroid coordinate ``m_j`` (the sum
    divided by ``n``), and ``math.fsum`` over each cluster's samples in
    cluster input order gives that cluster's centroid coordinate
    ``m_cj`` (the sum divided by the cluster size); the centroid sums
    follow those same sample orders. The between-cluster dispersion is
    ``B = sum_c sum_j n_c * (m_cj - m_j) ** 2`` with the outer sum over
    clusters in ascending label order and the inner sum over columns;
    the within-cluster dispersion is
    ``W = sum_i sum_j (X[i][j] - m_{labels[i],j}) ** 2`` with the outer
    sum over samples in input order and the inner sum over columns,
    each accumulated by ``math.fsum``. When ``W == 0`` the result is
    ``1.0``; otherwise it is ``(B / (k - 1)) / (W / (n - k))``.
    Overflow, invalid operations, or non-finite intermediate or final
    values after validation raise FloatingPointError. An exact zero
    result is normalized to ``0.0``.

    The return value is a float. The inputs are not modified.
    Deterministic: same inputs, same result.
    """
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
            if type(value) not in (int, float):
                raise ValueError(
                    "X must contain only finite non-boolean numbers"
                )
            # math.isfinite raises OverflowError for ints too large to
            # convert to float; such values fail the finite requirement.
            try:
                finite = math.isfinite(value)
            except OverflowError as exc:
                raise ValueError(
                    "X must contain only finite non-boolean numbers"
                ) from exc
            if not finite:
                raise ValueError(
                    "X must contain only finite non-boolean numbers"
                )
    n = len(X)
    if not isinstance(labels, list) or len(labels) != n:
        raise ValueError("labels must be a list with the same length as X")
    for value in labels:
        if type(value) is not int:
            raise ValueError("labels must contain only integers")
    distinct = sorted(set(labels))
    k = len(distinct)
    if k < 2 or k > n - 1:
        raise ValueError(
            "labels must contain between 2 and len(X) - 1 distinct labels"
        )

    def fail(exc=None):
        error = FloatingPointError(
            "non-finite value encountered during Calinski-Harabasz score"
        )
        if exc is None:
            raise error
        raise error from exc

    # Validate once, then convert every entry to float so that all later
    # arithmetic and math.fsum accumulation operate on a single type.
    try:
        data = [[float(value) for value in row] for row in X]
    except (OverflowError, ValueError) as exc:
        fail(exc)
    for row in data:
        for value in row:
            if not math.isfinite(value):
                fail()

    clusters = {}
    for i in range(n):
        clusters.setdefault(labels[i], []).append(i)
    groups = [(label, clusters[label]) for label in distinct]

    # Global centroid: fsum over all samples in input order, per column.
    global_centroid = []
    for j in range(width):
        try:
            total = math.fsum(data[i][j] for i in range(n))
            coordinate = total / n
        except (OverflowError, ValueError) as exc:
            fail(exc)
        if not math.isfinite(total) or not math.isfinite(coordinate):
            fail()
        global_centroid.append(coordinate)

    # Cluster centroids: fsum over each cluster's samples in cluster
    # input order, per column.
    cluster_centroids = []
    for label, group in groups:
        size = len(group)
        coordinates = []
        for j in range(width):
            try:
                total = math.fsum(data[i][j] for i in group)
                coordinate = total / size
            except (OverflowError, ValueError) as exc:
                fail(exc)
            if not math.isfinite(total) or not math.isfinite(coordinate):
                fail()
            coordinates.append(coordinate)
        cluster_centroids.append(coordinates)
    centroid_of = {
        label: cluster_centroids[c] for c, (label, _) in enumerate(groups)
    }

    # B = fsum over clusters, then columns, of n_c * (m_cj - m_j) ** 2.
    b_terms = []
    for c, (label, group) in enumerate(groups):
        size = len(group)
        for j in range(width):
            try:
                diff = cluster_centroids[c][j] - global_centroid[j]
                term = size * diff ** 2
            except (OverflowError, ValueError) as exc:
                fail(exc)
            if not math.isfinite(diff) or not math.isfinite(term):
                fail()
            b_terms.append(term)
    try:
        B = math.fsum(b_terms)
    except (OverflowError, ValueError) as exc:
        fail(exc)
    if not math.isfinite(B):
        fail()

    # W = fsum over samples, then columns, of (x_ij - m_label,j) ** 2.
    w_terms = []
    for i in range(n):
        centroid = centroid_of[labels[i]]
        for j in range(width):
            try:
                diff = data[i][j] - centroid[j]
                term = diff ** 2
            except (OverflowError, ValueError) as exc:
                fail(exc)
            if not math.isfinite(diff) or not math.isfinite(term):
                fail()
            w_terms.append(term)
    try:
        W = math.fsum(w_terms)
    except (OverflowError, ValueError) as exc:
        fail(exc)
    if not math.isfinite(W):
        fail()

    if W == 0.0:
        return 1.0
    try:
        result = (B / (k - 1)) / (W / (n - k))
    except (OverflowError, ValueError, ZeroDivisionError) as exc:
        fail(exc)
    if not math.isfinite(result):
        fail()
    if result == 0:
        result = 0.0
    return result


def adjusted_rand_score(labels_true, labels_pred):
    """Return the adjusted Rand index of two integer label partitions.

    Both arguments must be non-empty lists of equal length whose elements
    are exactly ``int`` (booleans are rejected). The inputs are not
    modified. Deterministic: same inputs, same result.

    Rows and columns of the contingency table correspond to the distinct
    labels of ``labels_true`` and ``labels_pred`` respectively, each in
    ascending label order; contingency counts ``n_ij`` accumulate by
    sample index. With ``C(x) = x * (x - 1) // 2`` and ``N = C(n)``,
    ``I = sum C(n_ij)``, ``A = sum C(a_i)`` over row sums, and
    ``B = sum C(b_j)`` over column sums, the expected index ``E = A*B/N``,
    the mean index ``M = (A+B)/2``, and the adjusted Rand index is
    ``(I - E) / (M - E)``. Every step of this computation uses
    ``fractions.Fraction`` exactly; only the final value is converted to
    ``float``. When ``N == 0`` (fewer than two samples) or ``M == E``
    (indeterminate denominator), the result is ``1.0``; an exact zero
    result is normalized to ``0.0``.
    """
    n = _check_metric_vectors(labels_true, labels_pred)
    for value in labels_true:
        if type(value) is not int:
            raise ValueError("labels_true must contain only integers")
    for value in labels_pred:
        if type(value) is not int:
            raise ValueError("labels_pred must contain only integers")

    def comb2(x):
        return x * (x - 1) // 2

    row_labels = sorted(set(labels_true))
    col_labels = sorted(set(labels_pred))
    row_index = {label: i for i, label in enumerate(row_labels)}
    col_index = {label: j for j, label in enumerate(col_labels)}

    counts = [[0] * len(col_labels) for _ in row_labels]
    for i in range(n):
        counts[row_index[labels_true[i]]][col_index[labels_pred[i]]] += 1

    row_sums = [sum(row) for row in counts]
    col_sums = [
        sum(counts[i][j] for i in range(len(row_labels)))
        for j in range(len(col_labels))
    ]

    total = comb2(n)
    if total == 0:
        return 1.0

    index = sum(comb2(value) for row in counts for value in row)
    row_term = sum(comb2(value) for value in row_sums)
    col_term = sum(comb2(value) for value in col_sums)

    expected = Fraction(row_term * col_term, total)
    mean = Fraction(row_term + col_term, 2)
    denominator = mean - expected
    if denominator == 0:
        return 1.0
    result = float((Fraction(index) - expected) / denominator)
    if result == 0:
        return 0.0
    return result


def normalized_mutual_info_score(labels_true, labels_pred):
    """Return the normalized mutual information of two integer partitions.

    Both arguments must be non-empty lists of equal length whose elements
    are exactly ``int`` (booleans are rejected). The inputs are not
    modified. Deterministic: same inputs, same result.

    Rows and columns of the contingency table correspond to the distinct
    labels of ``labels_true`` and ``labels_pred`` respectively, each in
    ascending label order; contingency counts ``n_ij`` accumulate by
    sample index, with row sums ``a_i`` and column sums ``b_j``. The
    mutual information is the ``math.fsum`` of
    ``(n_ij / n) * math.log((n_ij * n) / (a_i * b_j))`` over the cells
    with ``n_ij > 0``, taken row by row and then column by column. The
    entropies are ``H_true = -math.fsum((a_i / n) * math.log(a_i / n))``
    in ascending true-label order and ``H_pred`` likewise from the
    ``b_j``. When both entropies are exactly zero the result is ``1.0``;
    when exactly one is zero the result is ``0.0``; otherwise the result
    is ``MI / math.sqrt(H_true * H_pred)``. An exact zero result is
    normalized to ``0.0``. Overflow, invalid operations, or non-finite
    intermediate values or results in the post-validation multiplication,
    division, ``math.log``, ``math.sqrt``, or ``math.fsum`` steps raise
    FloatingPointError.
    """
    n = _check_metric_vectors(labels_true, labels_pred)
    for value in labels_true:
        if type(value) is not int:
            raise ValueError("labels_true must contain only integers")
    for value in labels_pred:
        if type(value) is not int:
            raise ValueError("labels_pred must contain only integers")

    row_labels = sorted(set(labels_true))
    col_labels = sorted(set(labels_pred))
    row_index = {label: i for i, label in enumerate(row_labels)}
    col_index = {label: j for j, label in enumerate(col_labels)}

    counts = [[0] * len(col_labels) for _ in row_labels]
    for i in range(n):
        counts[row_index[labels_true[i]]][col_index[labels_pred[i]]] += 1

    row_sums = [sum(row) for row in counts]
    col_sums = [
        sum(counts[i][j] for i in range(len(row_labels)))
        for j in range(len(col_labels))
    ]

    def non_finite():
        return FloatingPointError(
            "non-finite value encountered during normalized mutual "
            "information"
        )

    def checked(value):
        if not math.isfinite(value):
            raise non_finite()
        return value

    mi_terms = []
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            n_ij = counts[i][j]
            if n_ij <= 0:
                continue
            try:
                ratio = (n_ij * n) / (row_sums[i] * col_sums[j])
                log_value = math.log(ratio)
                term = (n_ij / n) * log_value
            except (OverflowError, ValueError) as exc:
                raise non_finite() from exc
            checked(ratio)
            checked(log_value)
            mi_terms.append(checked(term))
    try:
        mi = math.fsum(mi_terms)
    except (OverflowError, ValueError) as exc:
        raise non_finite() from exc
    checked(mi)

    def entropy(sums):
        """``-math.fsum((s / n) * math.log(s / n))`` over ``sums``."""
        terms = []
        for s in sums:
            try:
                p = s / n
                term = p * math.log(p)
            except (OverflowError, ValueError) as exc:
                raise non_finite() from exc
            terms.append(checked(term))
        try:
            total = math.fsum(terms)
        except (OverflowError, ValueError) as exc:
            raise non_finite() from exc
        return checked(-total)

    h_true = entropy(row_sums)
    h_pred = entropy(col_sums)

    if h_true == 0 and h_pred == 0:
        return 1.0
    if h_true == 0 or h_pred == 0:
        return 0.0
    try:
        product = h_true * h_pred
        denominator = math.sqrt(product)
        result = mi / denominator
    except (OverflowError, ValueError) as exc:
        raise non_finite() from exc
    checked(product)
    checked(denominator)
    checked(result)
    if result == 0:
        return 0.0
    return result


def matthews_corrcoef(y_true, y_pred):
    """Return the Matthews correlation coefficient of two integer label
    vectors.

    Both arguments must be non-empty lists of equal length whose elements
    are exactly ``int`` (booleans and subclasses are rejected). The inputs
    are not modified. Deterministic: same inputs, same result.

    The labels are the sorted union of the labels appearing in either
    vector, defining the rows (true labels) and columns (predicted labels)
    of an integer contingency table ``C`` accumulated in sample order. In
    that label order ``t_k`` is the ``k``-th row sum, ``p_k`` the ``k``-th
    column sum, ``c`` the sum of the diagonal, and ``s`` the number of
    samples. With every sum taken as an exact Python integer in label
    order, ``u = c*s - sum(p_k*t_k)``, ``a = s*s - sum(p_k*p_k)``, and
    ``b = s*s - sum(t_k*t_k)``. When ``a`` or ``b`` is zero the result is
    ``0.0``; otherwise it is ``u / math.sqrt(a*b)``. An exact zero result
    is normalized to ``0.0``. Overflow, invalid operations, or non-finite
    values during the square root, the integer-to-float conversion, or the
    division raise FloatingPointError.
    """
    s = _check_metric_vectors(y_true, y_pred)
    for value in y_true:
        if type(value) is not int:
            raise ValueError("y_true must contain only integers")
    for value in y_pred:
        if type(value) is not int:
            raise ValueError("y_pred must contain only integers")

    labels = sorted(set(y_true) | set(y_pred))
    index = {label: k for k, label in enumerate(labels)}
    size = len(labels)

    counts = [[0 for _ in range(size)] for _ in range(size)]
    for i in range(s):
        counts[index[y_true[i]]][index[y_pred[i]]] += 1

    t = [0 for _ in range(size)]
    p = [0 for _ in range(size)]
    c = 0
    for k in range(size):
        row_sum = 0
        for col in range(size):
            row_sum += counts[k][col]
        t[k] = row_sum
        col_sum = 0
        for row in range(size):
            col_sum += counts[row][k]
        p[k] = col_sum
        c += counts[k][k]

    pt = 0
    pp = 0
    tt = 0
    for k in range(size):
        pt += p[k] * t[k]
        pp += p[k] * p[k]
        tt += t[k] * t[k]

    u = c * s - pt
    a = s * s - pp
    b = s * s - tt
    if a == 0 or b == 0:
        return 0.0

    def non_finite():
        return FloatingPointError(
            "non-finite value encountered during matthews corrcoef"
        )

    try:
        denominator = math.sqrt(a * b)
    except (OverflowError, ValueError) as exc:
        raise non_finite() from exc
    if not math.isfinite(denominator):
        raise non_finite()
    try:
        result = u / denominator
    except (OverflowError, ValueError) as exc:
        raise non_finite() from exc
    if not math.isfinite(result):
        raise non_finite()
    if result == 0:
        return 0.0
    return result


def cohen_kappa_score(y_true, y_pred, weights=None):
    """Return Cohen's kappa for two integer label vectors.

    Both ``y_true`` and ``y_pred`` must be non-empty lists of equal
    length whose elements are exactly ``int`` (booleans and subclasses
    are rejected). The inputs are not modified. Deterministic: same
    inputs, same result.

    ``weights`` must be ``None``, ``"linear"``, or ``"quadratic"``. The
    classes are the sorted union of the values appearing in either
    vector; counts are accumulated in sample order into a square integer
    matrix ``C`` with true classes as rows and predicted classes as
    columns. In class order ``r_i`` is the ``i``-th row sum and ``c_j``
    the ``j``-th column sum, ``n`` the number of samples, and ``K`` the
    number of classes. With a single class the result is ``1.0``.

    The weight matrix is ``d(i, i) = 0.0`` and, for ``i != j``,
    ``1.0`` when unweighted, ``abs(i - j) / (K - 1)`` for linear
    weights, and ``((i - j) / (K - 1)) ** 2`` for quadratic weights.
    Iterating rows then columns, ``O = math.fsum(C_ij * d_ij) / n`` and
    ``E = math.fsum(r_i * c_j * d_ij) / (n * n)``. When ``E`` is zero
    the result is ``1.0``; otherwise it is ``1.0 - O / E``. An exact
    zero result is normalized to ``0.0``. Overflow, invalid operations,
    or non-finite values during the multiplications, divisions, power,
    or ``math.fsum`` calls raise FloatingPointError.
    """
    n = _check_metric_vectors(y_true, y_pred)
    for value in y_true:
        if type(value) is not int:
            raise ValueError("y_true must contain only integers")
    for value in y_pred:
        if type(value) is not int:
            raise ValueError("y_pred must contain only integers")
    if weights not in (None, "linear", "quadratic"):
        raise ValueError("weights must be None, 'linear', or 'quadratic'")

    classes = sorted(set(y_true) | set(y_pred))
    index = {label: k for k, label in enumerate(classes)}
    k = len(classes)
    if k == 1:
        return 1.0

    counts = [[0 for _ in range(k)] for _ in range(k)]
    for i in range(n):
        counts[index[y_true[i]]][index[y_pred[i]]] += 1

    row_sums = [0 for _ in range(k)]
    col_sums = [0 for _ in range(k)]
    for i in range(k):
        row_total = 0
        col_total = 0
        for j in range(k):
            row_total += counts[i][j]
            col_total += counts[j][i]
        row_sums[i] = row_total
        col_sums[i] = col_total

    span = k - 1

    def distance(i, j):
        if i == j:
            return 0.0
        delta = i - j
        if weights is None:
            return 1.0
        if weights == "linear":
            return abs(delta) / span
        return (delta / span) ** 2

    def non_finite(exc):
        return FloatingPointError(
            "non-finite value encountered during cohen kappa"
        )

    observed_terms = []
    expected_terms = []
    try:
        for i in range(k):
            for j in range(k):
                weight = distance(i, j)
                observed_terms.append(counts[i][j] * weight)
                expected_terms.append(row_sums[i] * col_sums[j] * weight)
        observed = math.fsum(observed_terms)
        expected = math.fsum(expected_terms)
        if not math.isfinite(observed) or not math.isfinite(expected):
            raise non_finite(None)
        o_value = observed / n
        e_value = expected / (n * n)
        if not math.isfinite(o_value) or not math.isfinite(e_value):
            raise non_finite(None)
    except (OverflowError, ValueError) as exc:
        raise non_finite(exc) from exc

    if e_value == 0:
        return 1.0
    try:
        result = 1.0 - o_value / e_value
    except (OverflowError, ValueError) as exc:
        raise non_finite(exc) from exc
    if not math.isfinite(result):
        raise non_finite(None)
    if result == 0:
        return 0.0
    return result


def _balanced_accuracy_float(value):
    """Convert a validated number to float; overflow, invalid operations,
    and non-finite results raise FloatingPointError."""
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise FloatingPointError(
            "non-finite value encountered during balanced accuracy"
        ) from exc
    if not math.isfinite(result):
        raise FloatingPointError(
            "non-finite value encountered during balanced accuracy"
        )
    return result


def balanced_accuracy_score(y_true, y_pred, sample_weight=None,
                            adjusted=False):
    """Return the balanced accuracy of two integer label vectors.

    Both ``y_true`` and ``y_pred`` must be non-empty lists of equal
    length whose elements are exactly ``int`` (booleans are rejected).
    ``sample_weight`` must be ``None`` -- every sample then weighs
    ``1.0`` -- or a list of the same length whose elements are finite
    non-negative values of type exactly ``int`` or ``float`` (booleans
    are rejected). ``adjusted`` must be exactly ``True`` or ``False``.
    Any violation (including overflow during the finiteness checks)
    raises ValueError. The inputs are not modified.

    The classes ``C`` are the sorted distinct values of ``y_true``;
    labels appearing only in ``y_pred`` merely count as
    misclassifications. After validation the weights are converted to
    ``float``; in input order, ``math.fsum`` computes for each class
    the total weight ``W`` of its samples and the weight ``T`` of its
    correctly predicted samples. A class with ``W <= 0`` raises
    ValueError. With ``r = T / W`` per class, the balanced accuracy is
    ``B = math.fsum(r over C) / len(C)``. When ``adjusted`` is false
    the result is ``B``; when true and ``len(C) == 1`` the result is
    ``1.0``; otherwise it is ``(B - 1 / len(C)) / (1 - 1 / len(C))``.
    Overflow, invalid operations during the post-validation conversion
    or arithmetic, and non-finite intermediate values or results raise
    FloatingPointError. An exact zero result is normalized to ``0.0``.

    The return value is a float. Deterministic: same inputs, same
    result.
    """
    n = _check_metric_vectors(y_true, y_pred)
    for value in y_true:
        if type(value) is not int:
            raise ValueError("y_true must contain only integers")
    for value in y_pred:
        if type(value) is not int:
            raise ValueError("y_pred must contain only integers")

    if sample_weight is None:
        weights = [1.0] * n
    else:
        if not isinstance(sample_weight, list) or len(sample_weight) != n:
            raise ValueError(
                "sample_weight must be a list with the same length as "
                "y_true"
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
        weights = [
            _balanced_accuracy_float(value) for value in sample_weight
        ]

    if type(adjusted) is not bool:
        raise ValueError("adjusted must be a boolean")

    classes = sorted(set(y_true))
    index = {label: k for k, label in enumerate(classes)}
    k = len(classes)

    true_terms = [[] for _ in range(k)]
    correct_terms = [[] for _ in range(k)]
    for i in range(n):
        j = index[y_true[i]]
        true_terms[j].append(weights[i])
        if y_pred[i] == y_true[i]:
            correct_terms[j].append(weights[i])

    def non_finite(exc):
        return FloatingPointError(
            "non-finite value encountered during balanced accuracy"
        )

    recalls = []
    for j in range(k):
        try:
            total = math.fsum(true_terms[j])
            correct = math.fsum(correct_terms[j])
        except (OverflowError, ValueError) as exc:
            raise non_finite(exc) from exc
        if not math.isfinite(total) or not math.isfinite(correct):
            raise non_finite(None)
        if total <= 0.0:
            raise ValueError(
                "every class must have a positive total weight"
            )
        try:
            recall = correct / total
        except (OverflowError, ValueError) as exc:
            raise non_finite(exc) from exc
        if not math.isfinite(recall):
            raise non_finite(None)
        recalls.append(recall)

    try:
        score = math.fsum(recalls) / k
    except (OverflowError, ValueError) as exc:
        raise non_finite(exc) from exc
    if not math.isfinite(score):
        raise non_finite(None)

    if not adjusted:
        result = score
    elif k == 1:
        return 1.0
    else:
        try:
            chance = 1 / k
            result = (score - chance) / (1 - chance)
        except (OverflowError, ValueError) as exc:
            raise non_finite(exc) from exc
        if not math.isfinite(result):
            raise non_finite(None)

    if result == 0:
        return 0.0
    return result


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
