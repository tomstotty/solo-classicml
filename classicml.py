"""Minimal classic machine-learning library (Python standard library only).

Exports:
    LinearRegression -- deterministic ordinary least squares regression with
        an optional L2 (ridge) penalty, trained by full-batch gradient descent.
    LogisticRegression -- deterministic binary logistic regression with an
        optional L2 penalty, trained by full-batch gradient descent.
    KNeighborsClassifier -- deterministic k-nearest-neighbors classifier
        using squared Euclidean distances.
    StandardScaler -- deterministic standardization by column mean and
        population standard deviation.
    accuracy_score -- fraction of index-wise equal integer labels.
    mean_squared_error -- mean of squared index-wise differences.

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
import sys
from decimal import Decimal, ROUND_HALF_UP

__all__ = [
    "LinearRegression",
    "LogisticRegression",
    "KNeighborsClassifier",
    "StandardScaler",
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
        # float (booleans and subclasses excluded) and math.isfinite
        # holds; math.isfinite raises OverflowError for ints too large to
        # convert to float, which also fails the requirement.
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
                if type(value) is not int and type(value) is not float:
                    raise ValueError(
                        "X must contain only finite non-boolean numbers"
                    )
                try:
                    if not math.isfinite(value):
                        raise ValueError(
                            "X must contain only finite non-boolean numbers"
                        )
                except OverflowError as exc:
                    raise ValueError(
                        "X must contain only finite non-boolean numbers"
                    ) from exc
        return width

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


def _check_metric_inputs(y_true, y_pred):
    """Validate that both metric inputs are non-empty lists of equal length."""
    if (
        not isinstance(y_true, list)
        or not isinstance(y_pred, list)
        or len(y_true) == 0
        or len(y_true) != len(y_pred)
    ):
        raise ValueError(
            "y_true and y_pred must be non-empty lists of equal length"
        )
    return len(y_true)


def accuracy_score(y_true, y_pred):
    """Fraction of index-wise equal entries between two integer label lists.

    Both arguments must be non-empty lists of equal length whose elements
    are exactly ``int`` (booleans excluded). The inputs are not modified.
    """
    n = _check_metric_inputs(y_true, y_pred)
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


def _require_exact_finite(value, name):
    # Accept only values whose type is exactly int or float (booleans and
    # subclasses excluded) and for which math.isfinite holds; an
    # OverflowError from the finiteness check also fails the requirement.
    if type(value) is not int and type(value) is not float:
        raise ValueError(
            "%s must contain only finite non-boolean numbers" % name
        )
    try:
        if not math.isfinite(value):
            raise ValueError(
                "%s must contain only finite non-boolean numbers" % name
            )
    except OverflowError as exc:
        raise ValueError(
            "%s must contain only finite non-boolean numbers" % name
        ) from exc


def mean_squared_error(y_true, y_pred):
    """Mean of squared index-wise differences between two numeric lists.

    Both arguments must be non-empty lists of equal length whose elements
    are exactly ``int`` or ``float`` (booleans excluded) and finite. The
    inputs are not modified. Overflow, invalid operations, and non-finite
    intermediate values or results raise FloatingPointError.
    """
    n = _check_metric_inputs(y_true, y_pred)
    for value in y_true:
        _require_exact_finite(value, "y_true")
    for value in y_pred:
        _require_exact_finite(value, "y_pred")

    terms = []
    for i in range(n):
        try:
            d = y_true[i] - y_pred[i]
        except (OverflowError, ValueError) as exc:
            raise FloatingPointError(
                "non-finite value encountered during mean squared error"
            ) from exc
        try:
            if not math.isfinite(d):
                raise FloatingPointError(
                    "non-finite value encountered during mean squared error"
                )
        except OverflowError as exc:
            raise FloatingPointError(
                "non-finite value encountered during mean squared error"
            ) from exc
        try:
            square = d ** 2
        except (OverflowError, ValueError) as exc:
            raise FloatingPointError(
                "non-finite value encountered during mean squared error"
            ) from exc
        try:
            if not math.isfinite(square):
                raise FloatingPointError(
                    "non-finite value encountered during mean squared error"
                )
        except OverflowError as exc:
            raise FloatingPointError(
                "non-finite value encountered during mean squared error"
            ) from exc
        terms.append(square)

    try:
        result = math.fsum(terms) / n
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
