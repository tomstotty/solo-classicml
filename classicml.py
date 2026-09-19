"""Deterministic classic machine-learning models (Python standard library only)."""

import json
import math
import sys
from decimal import Decimal, ROUND_HALF_UP

__all__ = ["LinearRegression"]

_QUANT_10 = Decimal("0.0000000001")
_INPUT_KEYS = ("X", "y", "lr", "l2", "max_iter", "tol")


def _is_number(value):
    # bool is a subclass of int but is explicitly not accepted as a number
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_finite_number(value):
    return _is_number(value) and math.isfinite(value)


def _check_matrix(X, y=None):
    """Validate a non-empty rectangular matrix of finite non-bool numbers."""
    if not isinstance(X, list) or len(X) == 0:
        raise ValueError("X must be a non-empty list of rows")
    n_cols = None
    for row in X:
        if not isinstance(row, list) or len(row) == 0:
            raise ValueError("X must contain only non-empty lists")
        if n_cols is None:
            n_cols = len(row)
        elif len(row) != n_cols:
            raise ValueError("X must be rectangular")
        for value in row:
            if not _is_finite_number(value):
                raise ValueError("X contains a value that is not a finite number")
    if y is not None:
        if not isinstance(y, list) or len(y) != len(X):
            raise ValueError("y must be a list with the same length as X")
        for value in y:
            if not _is_finite_number(value):
                raise ValueError("y contains a value that is not a finite number")
    return n_cols


class LinearRegression:
    """Plain-batch-gradient-descent linear regression with optional L2 penalty."""

    def __init__(self, lr=0.01, l2=0.0, max_iter=1000, tol=1e-8):
        if not _is_finite_number(lr) or lr <= 0:
            raise ValueError("lr must be a finite number greater than 0")
        if not _is_finite_number(l2) or l2 < 0:
            raise ValueError("l2 must be a finite number greater than or equal to 0")
        if (
            not isinstance(max_iter, int)
            or isinstance(max_iter, bool)
            or max_iter < 1
        ):
            raise ValueError("max_iter must be an integer greater than or equal to 1")
        if not _is_finite_number(tol) or tol <= 0:
            raise ValueError("tol must be a finite number greater than 0")

        self.lr = float(lr)
        self.l2 = float(l2)
        self.max_iter = max_iter
        self.tol = float(tol)
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_cols = _check_matrix(X, y)
        m = len(X)

        # Parameters always start from zero.
        w = [0.0] * n_cols
        b = 0.0
        lr, l2 = self.lr, self.l2

        for _ in range(self.max_iter):
            preds = []
            for row in X:
                p = math.fsum(w[j] * row[j] for j in range(n_cols)) + b
                if not math.isfinite(p):
                    raise FloatingPointError(
                        "non-finite intermediate prediction encountered"
                    )
                preds.append(p)

            errors = [preds[i] - y[i] for i in range(m)]

            gw = []
            for j in range(n_cols):
                g = (
                    2.0 * math.fsum(errors[i] * X[i][j] for i in range(m)) / m
                    + 2.0 * l2 * w[j]
                )
                gw.append(g)
            gb = 2.0 * math.fsum(errors) / m

            if not all(math.isfinite(g) for g in gw) or not math.isfinite(gb):
                raise FloatingPointError("non-finite gradient encountered")

            # Simultaneous parameter update.
            new_w = [w[j] - lr * gw[j] for j in range(n_cols)]
            new_b = b - lr * gb

            if not all(math.isfinite(v) for v in new_w) or not math.isfinite(new_b):
                raise FloatingPointError("non-finite parameter value encountered")

            step = max(
                [abs(new_w[j] - w[j]) for j in range(n_cols)] + [abs(new_b - b)]
            )
            w, b = new_w, new_b

            if step <= self.tol:
                break

        self.w = w
        self.b = b
        return self

    def predict(self, X):
        if self.w is None:
            raise ValueError("the model must be fitted before calling predict")
        n_cols = _check_matrix(X)
        if n_cols != len(self.w):
            raise ValueError(
                "X has a different number of features than the fitted model"
            )

        result = []
        for row in X:
            p = math.fsum(self.w[j] * row[j] for j in range(n_cols)) + self.b
            if not math.isfinite(p):
                raise FloatingPointError("non-finite prediction encountered")
            quantized = Decimal(str(p)).quantize(
                _QUANT_10, rounding=ROUND_HALF_UP
            )
            value = float(quantized)
            if value == 0.0:
                value = 0.0  # normalize negative zero
            result.append(value)
        return result


def _format_fixed(value):
    """Format a float with exactly 10 decimals, normalizing negative zero."""
    if value == 0:
        value = 0.0
    return format(value, ".10f")


def _train_linear(raw_bytes):
    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("input is not valid UTF-8") from exc

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("input is not valid JSON") from exc

    if not isinstance(data, dict) or tuple(data.keys()) != _INPUT_KEYS:
        raise ValueError(
            "input keys must appear in the order X, y, lr, l2, max_iter, tol"
        )

    model = LinearRegression(
        lr=data["lr"],
        l2=data["l2"],
        max_iter=data["max_iter"],
        tol=data["tol"],
    )
    model.fit(data["X"], data["y"])

    w_values = ",".join(_format_fixed(v) for v in model.w)
    return (
        '{"class":"LinearRegression",'
        f'"lr":{_format_fixed(model.lr)},'
        f'"l2":{_format_fixed(model.l2)},'
        f'"max_iter":{model.max_iter},'
        f'"tol":{_format_fixed(model.tol)},'
        f'"w":[{w_values}],'
        f'"b":{_format_fixed(model.b)}}}'
    )


def main(argv):
    if len(argv) == 2 and argv[1] == "train-linear":
        try:
            payload = _train_linear(sys.stdin.buffer.read())
        except Exception as exc:  # all business/parse errors share one contract
            sys.stderr.write(f"ValueError: {exc}\n")
            return 2
        sys.stdout.buffer.write(payload.encode("utf-8"))
        return 0

    sys.stderr.write("ValueError: unknown command (expected: train-linear)\n")
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
