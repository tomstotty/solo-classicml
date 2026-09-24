"""Minimal classic machine-learning library (Python standard library only).

Exports:
    LinearRegression -- deterministic ordinary least squares regression with
        an optional L2 (ridge) penalty, trained by full-batch gradient descent.
    LogisticRegression -- deterministic binary logistic regression with an
        optional L2 penalty, trained by full-batch gradient descent.
    MultinomialLogisticRegression -- deterministic softmax multi-class
        logistic regression with an optional L2 penalty, trained by
        full-batch gradient descent.
    KNeighborsClassifier -- deterministic k-nearest-neighbors classifier
        using squared Euclidean distances.
    KNeighborsRegressor -- deterministic k-nearest-neighbors regressor
        using squared Euclidean distances with uniform or inverse-distance
        weights.
    DecisionTreeClassifier -- deterministic binary decision tree classifier
        using Gini impurity splits.
    RandomForestClassifier -- deterministic bagged forest of Gini decision
        trees with per-node random feature subsampling.
    ExtraTreesClassifier -- deterministic extremely-randomized forest of
        Gini decision trees using all samples per tree with per-node
        random feature and threshold selection.
    AdaBoostClassifier -- deterministic discrete AdaBoost of decision
        stumps over +/-1 labels.
    GradientBoostingRegressor -- deterministic gradient boosting for
        regression with level-wise decision stumps and a constant
        learning rate.
    GradientBoostingClassifier -- deterministic gradient boosting for
        binary classification with decision stumps, log-odds
        initialization and binary logistic pseudo-residuals.
    StandardScaler -- deterministic standardization by column mean and
        population standard deviation.
    KMeans -- deterministic k-means clustering with Lloyd's iterations and
        seeded centroid initialization.
    PCA -- deterministic first-principal-component projection for
        two-dimensional data.
    DBSCAN -- deterministic density-based clustering with an epsilon
        neighborhood and a minimum core-point sample count.
    AgglomerativeClustering -- deterministic bottom-up clustering with
        single, complete, or average linkage.
    GaussianMixture -- deterministic one-dimensional Gaussian mixture
        model fitted by expectation-maximization from sorted initial means.
    IsolationForest -- reusable deterministic isolation-forest anomaly
        detector with fit/score_samples/predict and a contamination-
        derived score threshold.
    accuracy_score -- fraction of positions where two integer label
        vectors agree.
    mean_squared_error -- weighted mean of squared element-wise
        differences of two finite real vectors.
    root_mean_squared_error -- square root of the weighted mean of squared
        element-wise differences of two finite real vectors.
    mean_absolute_error -- weighted mean of element-wise absolute
        differences of two finite real vectors.
    median_absolute_error -- weighted median of element-wise absolute
        differences of two finite real vectors.
    max_error -- maximum of the element-wise absolute differences of two
        finite real vectors.
    mean_squared_log_error -- weighted mean squared logarithmic error of
        two finite non-negative real vectors.
    root_mean_squared_log_error -- square root of the weighted mean
        squared logarithmic error of two finite non-negative real
        vectors.
    r2_score -- coefficient of determination of two finite real vectors,
        optionally weighted.
    explained_variance_score -- weighted explained variance regression
        score of two finite real vectors.
    concordance_correlation_coefficient -- Lin's weighted concordance
        correlation coefficient of two finite real vectors.
    mean_pinball_loss -- weighted mean pinball (quantile) loss of two
        finite real vectors at a quantile level alpha in [0, 1].
    d2_pinball_score -- fraction by which pinball loss improves over the
        weighted alpha-quantile constant baseline of the true vector.
    mean_absolute_scaled_error -- weighted mean absolute error of two
        finite real vectors scaled by the in-sample naive-forecast error
        of a training vector with seasonality m.
    mean_poisson_deviance -- weighted mean Poisson deviance of a
        non-negative real vector and a strictly positive prediction
        vector.
    mean_gamma_deviance -- weighted mean Gamma deviance of two
        strictly positive real vectors.
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
    roc_auc_score -- trapezoidal area under the roc_curve, with optional
        standardized partial AUC over an FPR range.
    multiclass_roc_auc_score -- one-vs-one ROC AUC of a multiclass score
        matrix, averaged as macro or weighted by class support.
    det_curve -- false positive/false negative rates and ascending
        thresholds for a binary score ranking.
    precision_recall_curve -- precision/recall pairs at descending score
        thresholds for a binary score ranking.
    average_precision_score -- stepwise area under the
        precision_recall_curve.
    multiclass_average_precision_score -- one-vs-rest average precision
        of a multiclass score matrix, per class or averaged macro or
        weighted by class support.
    calibration_curve -- per-bin positive-class weight fractions and mean
        predicted probabilities for a binary probability vector.
    expected_calibration_error -- weighted expected calibration error for
        a binary probability vector.
    adaptive_calibration_error -- weighted expected calibration error for
        a binary probability vector with equal-frequency bins.
    maximum_calibration_error -- maximum per-bin calibration gap for a
        binary probability vector with equal-width bins.
    brier_score_loss -- weighted mean squared probability error for a
        binary probability vector.
    brier_score_decomposition -- reliability, resolution, and
        uncertainty decomposition of the weighted mean squared
        probability error.
    log_loss -- weighted logistic (cross-entropy) loss for a binary
        probability vector with clamped probabilities.
    multiclass_log_loss -- weighted multiclass logistic (cross-entropy)
        loss over a probability matrix with one column per class, using
        the probability of the true class clamped away from zero.
    multiclass_brier_score_loss -- weighted multiclass Brier score loss
        over a probability matrix with one column per class, summed over
        columns without dividing by the number of classes.
    hinge_loss -- weighted binary hinge loss of a +/-1 label vector and
        a real-valued score vector.
    logit_loss -- weighted binary logistic (cross-entropy) loss of a
        binary label vector and an unconstrained real score vector.
    multiclass_hinge_loss -- weighted multiclass Crammer-Singer hinge
        loss over a score matrix with one column per class.
    silhouette_score -- mean silhouette coefficient of a clustering over
        a finite real matrix and an integer label vector.
    silhouette_samples -- per-sample silhouette coefficients of a
        clustering over a finite real matrix and an integer label
        vector, in input order.
    davies_bouldin_score -- Davies-Bouldin index of a clustering: the mean,
        over clusters in ascending label order, of the largest ratio of
        summed within-cluster mean distances to centroid separation.
    calinski_harabasz_score -- Calinski-Harabasz index of a clustering:
        the ratio of between-cluster dispersion to within-cluster
        dispersion, each divided by its degrees of freedom.
    adjusted_rand_score -- exact adjusted Rand index of two integer
        partitions, computed with fractions.Fraction and returned as float.
    fowlkes_mallows_score -- Fowlkes-Mallows index of two integer
        partitions: TP / sqrt(P * Q) from the exact integer contingency
        table.
    normalized_mutual_info_score -- mutual information of two integer
        partitions normalized by the geometric mean of their entropies.
    adjusted_mutual_info_score -- mutual information of two integer
        partitions adjusted for chance against the expected mutual
        information of the hypergeometric model.
    homogeneity_completeness_v_measure -- homogeneity, completeness, and
        V-measure of two integer partitions with a beta weighting.
    matthews_corrcoef -- Matthews correlation coefficient of two integer
        label vectors, computed from an exact integer contingency table.
    balanced_accuracy_score -- weighted mean per-class recall of two
        integer label vectors, optionally adjusted for chance.
    geometric_mean_score -- weighted geometric mean per-class recall of
        two integer label vectors, with a correction for zero recalls.
    top_k_accuracy_score -- weighted fraction of samples whose true class
        is among the k highest-scoring columns of a score matrix.
    ndcg_score -- normalized discounted cumulative gain at k between a
        non-negative relevance vector and a score vector, or the
        weighted mean, over the rows of a relevance matrix and a score
        matrix, of the per-row normalized discounted cumulative gain at
        k with linear or exponential gains.
    discounted_cumulative_gain_score -- weighted mean, over the rows of
        a non-negative relevance matrix and a score matrix, of the
        discounted cumulative gain at k with linear or exponential
        gains.
    label_ranking_average_precision_score -- weighted mean, over samples,
        of the average precision of a score ranking against a binary
        label matrix.
    coverage_error -- weighted mean, over samples, of the number of
        labels that must be included to cover all positive labels of a
        binary label matrix under a score ranking.
    label_ranking_loss -- weighted mean, over samples, of the fraction
        of positive-negative label pairs whose ordering is wrong (ties
        count half) under a score ranking.
    hamming_loss -- weighted mean, over samples, of the fraction of
        labels that differ between two binary multilabel matrices.
    mean_reciprocal_rank_score -- weighted mean, over samples, of the
        reciprocal rank of the first positive label among the k
        highest-scoring columns of a binary label matrix.
    continuous_ranked_probability_score -- weighted mean, over samples,
        of the continuous ranked probability score of an ensemble of
        predictions against one target per row.
    energy_score -- weighted mean, over samples, of the energy score of
        an ensemble of multivariate predictions with exponent beta
        against one multivariate target per sample.
    interval_score -- weighted mean interval score of central
        (1 - alpha) prediction intervals against one target per row.
    weighted_interval_score -- weighted mean, over samples, of the
        multi-interval weighted interval score combining a median
        absolute-error term with central prediction intervals at
        strictly increasing coverage levels.
    concordance_index -- weighted Harrell's concordance index between
        event times and risk scores over comparable sample pairs.
    integrated_brier_score -- integrated Brier score of survival
        probability estimates at evaluation times, using the
        inverse-probability-of-censoring weighting with the Kaplan-Meier
        estimator of the censoring distribution.
    cumulative_dynamic_auc -- cumulative/dynamic AUC of risk scores at
        evaluation times, using inverse-probability-of-censoring
        weighting with the Kaplan-Meier estimator of the censoring
        distribution.
    survival_brier_score -- Brier score of survival probability
        estimates at each evaluation time, using the
        inverse-probability-of-censoring weighting with the Kaplan-Meier
        estimator of the censoring distribution.
    isolation_forest_score -- deterministic isolation-forest anomaly
        score of each row of a finite real matrix, averaged over a
        seeded forest of random isolation trees.
    local_outlier_factor_score -- deterministic local outlier factor of
        each row of a finite real matrix from reachability distances
        averaged over the tie-inclusive k-nearest neighborhoods.
    optics_clustering -- deterministic OPTICS-style clustering of a
        finite real matrix returning integer labels in input order.
    dumps -- serialize a fitted KMeans/PCA/linear/scaler/tree/forest
        model (including MultinomialLogisticRegression,
        AgglomerativeClustering, and DBSCAN) to whitespace-free JSON
        text (quantized to 10 decimal places with ROUND_HALF_UP).
    loads -- reconstruct an independent fitted model from text produced
        by dumps; anything outside that byte format raises ValueError.

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
from collections import deque
from decimal import Decimal, ROUND_HALF_UP, localcontext
from fractions import Fraction

__all__ = [
    "LinearRegression",
    "LogisticRegression",
    "MultinomialLogisticRegression",
    "KNeighborsClassifier",
    "KNeighborsRegressor",
    "DecisionTreeClassifier",
    "DecisionTreeRegressor",
    "RandomForestRegressor",
    "RandomForestClassifier",
    "ExtraTreesClassifier",
    "AdaBoostClassifier",
    "GradientBoostingRegressor",
    "GradientBoostingClassifier",
    "LassoRegression",
    "ElasticNetRegression",
    "StandardScaler",
    "KMeans",
    "KMedoids",
    "PCA",
    "DBSCAN",
    "AgglomerativeClustering",
    "GaussianMixture",
    "IsolationForest",
    "accuracy_score",
    "mean_squared_error",
    "root_mean_squared_error",
    "mean_absolute_error",
    "median_absolute_error",
    "max_error",
    "mean_squared_log_error",
    "root_mean_squared_log_error",
    "r2_score",
    "explained_variance_score",
    "concordance_correlation_coefficient",
    "mean_absolute_percentage_error",
    "symmetric_mean_absolute_percentage_error",
    "median_absolute_percentage_error",
    "mean_squared_percentage_error",
    "mean_pinball_loss",
    "d2_pinball_score",
    "mean_huber_loss",
    "mean_log_cosh_loss",
    "mean_absolute_scaled_error",
    "mean_poisson_deviance",
    "mean_gamma_deviance",
    "mean_tweedie_deviance",
    "d2_tweedie_score",
    "precision_recall_fscore_support",
    "confusion_matrix",
    "multilabel_confusion_matrix",
    "precision_score",
    "recall_score",
    "f1_score",
    "fbeta_score",
    "roc_curve",
    "roc_auc_score",
    "multiclass_roc_auc_score",
    "det_curve",
    "precision_recall_curve",
    "average_precision_score",
    "multiclass_average_precision_score",
    "calibration_curve",
    "expected_calibration_error",
    "adaptive_calibration_error",
    "maximum_calibration_error",
    "brier_score_loss",
    "brier_score_decomposition",
    "log_loss",
    "multiclass_log_loss",
    "multiclass_brier_score_loss",
    "hinge_loss",
    "logit_loss",
    "multiclass_hinge_loss",
    "silhouette_score",
    "silhouette_samples",
    "davies_bouldin_score",
    "calinski_harabasz_score",
    "adjusted_rand_score",
    "fowlkes_mallows_score",
    "normalized_mutual_info_score",
    "adjusted_mutual_info_score",
    "homogeneity_completeness_v_measure",
    "matthews_corrcoef",
    "cohen_kappa_score",
    "balanced_accuracy_score",
    "geometric_mean_score",
    "top_k_accuracy_score",
    "ndcg_score",
    "discounted_cumulative_gain_score",
    "jaccard_score",
    "label_ranking_average_precision_score",
    "coverage_error",
    "label_ranking_loss",
    "hamming_loss",
    "mean_reciprocal_rank_score",
    "precision_at_k_score",
    "recall_at_k_score",
    "mean_average_precision_at_k_score",
    "continuous_ranked_probability_score",
    "energy_score",
    "interval_score",
    "weighted_interval_score",
    "concordance_index",
    "integrated_brier_score",
    "cumulative_dynamic_auc",
    "survival_brier_score",
    "isolation_forest_score",
    "local_outlier_factor_score",
    "optics_clustering",
    "dumps",
    "loads",
]

_QUANTUM = Decimal("1E-10")
_QUANTUM12 = Decimal("1E-12")
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


def _checked_parameter(value, name):
    """Validate a finite non-boolean hyperparameter, mapping integer
    conversion overflow (e.g. ``10**400``) to ValueError."""
    try:
        return _require_finite_number(value, name)
    except OverflowError as exc:
        raise ValueError(
            "%s must be a finite non-boolean real number" % name
        ) from exc


def _softmax(z):
    """Numerically stable softmax of a score list.

    ``a = max(z)``; ``u_k = exp(z_k - a)`` and ``p_k = u_k / fsum(u)``,
    evaluated in index order. Any arithmetic failure or non-finite
    result raises FloatingPointError.
    """
    a = z[0]
    for value in z[1:]:
        if value > a:
            a = value
    try:
        u = [math.exp(value - a) for value in z]
        total = math.fsum(u)
    except (OverflowError, ValueError) as exc:
        raise FloatingPointError(
            "non-finite value encountered during softmax"
        ) from exc
    if not math.isfinite(total) or total <= 0.0:
        raise FloatingPointError(
            "non-finite value encountered during softmax"
        )
    p = []
    for value in u:
        prob = value / total
        if not math.isfinite(prob):
            raise FloatingPointError(
                "non-finite value encountered during softmax"
            )
        p.append(prob)
    return p


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

    def predict_proba(self, X):
        """Return the predicted probability of the positive class (1)
        for each row of X.

        X must satisfy the same validation as in fit/predict: a
        non-empty rectangular matrix of finite non-boolean numbers with
        the same number of columns as the training data. Each row's
        score is ``z = math.fsum(w_j * x_j) + b`` computed in feature
        order, mapped through the numerically stable sigmoid (the
        ``z >= 0`` branch uses ``1 / (1 + exp(-z))``, otherwise
        ``e = exp(z); e / (1 + e)``). An OverflowError or ValueError
        from the conversion, multiply-add, exp, or fsum steps, or any
        non-finite intermediate or result, raises FloatingPointError.
        An exact zero result is returned as positive 0.0. The input is
        not modified and repeated calls return identical values.
        """
        if self.w is None or self.b is None:
            raise ValueError(
                "model must be fitted before predict_proba is called"
            )
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
            try:
                if z >= 0:
                    p = 1.0 / (1.0 + math.exp(-z))
                else:
                    e = math.exp(z)
                    p = e / (1.0 + e)
            except (OverflowError, ValueError) as exc:
                raise FloatingPointError(
                    "non-finite prediction encountered"
                ) from exc
            if not math.isfinite(p):
                raise FloatingPointError("non-finite prediction encountered")
            if p == 0:
                p = 0.0
            results.append(p)
        return results


def _check_multiclass_vector(y, n):
    """Validate an integer class-label vector with the same length as X.

    Every element has type exactly ``int`` (booleans are rejected) and
    at least two distinct labels must occur.
    """
    if not isinstance(y, list) or len(y) != n:
        raise ValueError("y must be a list with the same length as X")
    classes = set()
    for value in y:
        if type(value) is not int:
            raise ValueError("y must contain only integers")
        classes.add(value)
    if len(classes) < 2:
        raise ValueError("y must contain at least two distinct classes")
    return sorted(classes)


class MultinomialLogisticRegression:
    """Multinomial (multi-class) logistic regression with a softmax output.

    Fits one weight vector and intercept per class (classes sorted in
    ascending order) by full-batch gradient descent on the mean
    cross-entropy plus ``l2 * sum_k sum_j w_kj**2``; intercepts are not
    penalized. All K weight vectors and intercepts start at zero and
    are updated synchronously. Training uses no randomness.
    """

    def __init__(self, lr=0.01, l2=0.0, max_iter=1000, tol=1e-8):
        _checked_parameter(lr, "lr")
        if lr <= 0:
            raise ValueError("lr must be greater than 0")
        _checked_parameter(l2, "l2")
        if l2 < 0:
            raise ValueError("l2 must be non-negative")
        if isinstance(max_iter, bool) or not isinstance(max_iter, int):
            raise ValueError("max_iter must be an integer")
        if max_iter < 1:
            raise ValueError("max_iter must be at least 1")
        _checked_parameter(tol, "tol")
        if tol <= 0:
            raise ValueError("tol must be greater than 0")

        self.lr = lr
        self.l2 = l2
        self.max_iter = max_iter
        self.tol = tol
        self.classes = None
        self.W = None
        self.b = None

    def _scores(self, X, width):
        """Return the matrix ``z[i][k]`` of per-class scores for X.

        Each score is ``math.fsum(w_kj * x_ij for j in feature order)
        + b_k``, computed in sample then class order. Any arithmetic
        failure or non-finite result raises FloatingPointError.
        """
        classes = self.classes
        W = self.W
        b = self.b
        k_count = len(classes)
        scores = []
        for row in X:
            row_scores = []
            for k in range(k_count):
                try:
                    z = math.fsum(
                        W[k][j] * row[j] for j in range(width)
                    ) + b[k]
                except (OverflowError, ValueError) as exc:
                    raise FloatingPointError(
                        "non-finite prediction encountered"
                    ) from exc
                if not math.isfinite(z):
                    raise FloatingPointError(
                        "non-finite prediction encountered"
                    )
                row_scores.append(z)
            scores.append(row_scores)
        return scores

    def fit(self, X, y):
        # Reset first so a failed fit leaves the model unfitted.
        self.classes = None
        self.W = None
        self.b = None

        try:
            width = _check_matrix(X)
            classes = _check_multiclass_vector(y, len(X))
        except OverflowError as exc:
            raise ValueError(
                "X must contain only finite non-boolean numbers"
            ) from exc

        n = len(X)
        k_count = len(classes)
        W = [[0.0] * width for _ in range(k_count)]
        b = [0.0] * k_count
        lr = self.lr
        l2 = self.l2
        index = {label: k for k, label in enumerate(classes)}
        targets = [index[value] for value in y]

        for _ in range(self.max_iter):
            # Forward pass: probabilities in sample/class/feature order.
            probabilities = []
            for i in range(n):
                row = X[i]
                z = []
                for k in range(k_count):
                    try:
                        score = math.fsum(
                            W[k][j] * row[j] for j in range(width)
                        ) + b[k]
                    except (OverflowError, ValueError) as exc:
                        raise FloatingPointError(
                            "non-finite value encountered during fit"
                        ) from exc
                    if not math.isfinite(score):
                        raise FloatingPointError(
                            "non-finite value encountered during fit"
                        )
                    z.append(score)
                probabilities.append(_softmax(z))

            # Gradients accumulated over samples, then synchronous update.
            new_W = [[0.0] * width for _ in range(k_count)]
            new_b = [0.0] * k_count
            max_step = 0.0
            for k in range(k_count):
                for j in range(width):
                    try:
                        grad = (
                            math.fsum(
                                (
                                    probabilities[i][k]
                                    - (1.0 if targets[i] == k else 0.0)
                                )
                                * X[i][j]
                                for i in range(n)
                            )
                            / n
                            + 2.0 * l2 * W[k][j]
                        )
                    except (OverflowError, ValueError) as exc:
                        raise FloatingPointError(
                            "non-finite weight gradient encountered during fit"
                        ) from exc
                    if not math.isfinite(grad):
                        raise FloatingPointError(
                            "non-finite weight gradient encountered during fit"
                        )
                    try:
                        updated = W[k][j] - lr * grad
                    except (OverflowError, ValueError) as exc:
                        raise FloatingPointError(
                            "non-finite weight value encountered during fit"
                        ) from exc
                    if not math.isfinite(updated):
                        raise FloatingPointError(
                            "non-finite weight value encountered during fit"
                        )
                    new_W[k][j] = updated
                    step = abs(updated - W[k][j])
                    if step > max_step:
                        max_step = step

                try:
                    grad_b = (
                        math.fsum(
                            probabilities[i][k]
                            - (1.0 if targets[i] == k else 0.0)
                            for i in range(n)
                        )
                        / n
                    )
                except (OverflowError, ValueError) as exc:
                    raise FloatingPointError(
                        "non-finite bias gradient encountered during fit"
                    ) from exc
                if not math.isfinite(grad_b):
                    raise FloatingPointError(
                        "non-finite bias gradient encountered during fit"
                    )
                try:
                    updated_b = b[k] - lr * grad_b
                except (OverflowError, ValueError) as exc:
                    raise FloatingPointError(
                        "non-finite bias value encountered during fit"
                    ) from exc
                if not math.isfinite(updated_b):
                    raise FloatingPointError(
                        "non-finite bias value encountered during fit"
                    )
                new_b[k] = updated_b
                step_b = abs(updated_b - b[k])
                if step_b > max_step:
                    max_step = step_b

            W = new_W
            b = new_b

            if max_step <= self.tol:
                break

        self.classes = classes
        self.W = W
        self.b = b
        return self

    def _validate_prediction(self, X):
        if self.classes is None or self.W is None or self.b is None:
            raise ValueError("model must be fitted before prediction")
        try:
            width = _check_matrix(X)
        except OverflowError as exc:
            raise ValueError(
                "X must contain only finite non-boolean numbers"
            ) from exc
        if width != len(self.W[0]):
            raise ValueError(
                "X must have the same number of features as the training data"
            )
        return width

    def decision_function(self, X):
        """Return the per-class score matrix ``z`` (one row per sample,
        one column per class in ascending class order)."""
        width = self._validate_prediction(X)
        return self._scores(X, width)

    def predict_proba(self, X):
        """Return the softmax probability matrix (one row per sample,
        one column per class in ascending class order)."""
        width = self._validate_prediction(X)
        scores = self._scores(X, width)
        return [_softmax(row) for row in scores]

    def predict(self, X):
        """Return the class with the largest probability for each row.

        Ties are resolved toward the smaller class label (classes are
        stored in ascending order, so the first column attaining the
        maximum wins).
        """
        width = self._validate_prediction(X)
        scores = self._scores(X, width)
        predictions = []
        for row in scores:
            probabilities = _softmax(row)
            best_k = 0
            best_p = probabilities[0]
            for k in range(1, len(probabilities)):
                if probabilities[k] > best_p:
                    best_p = probabilities[k]
                    best_k = k
            predictions.append(self.classes[best_k])
        return predictions


def _check_label_vector(y, n):
    """Validate an integer class-label vector with the same length as X."""
    if not isinstance(y, list) or len(y) != n:
        raise ValueError("y must be a list with the same length as X")
    for value in y:
        if type(value) is not int:
            raise ValueError("y must contain only integers")


def _check_signed_label_vector(y, n):
    """Validate a +/-1 integer label vector with the same length as X.

    Every element has type exactly ``int`` and equals ``-1`` or ``1``, and
    both labels must occur.
    """
    if not isinstance(y, list) or len(y) != n:
        raise ValueError("y must be a list with the same length as X")
    seen_minus = False
    seen_plus = False
    for value in y:
        if type(value) is not int or value not in (-1, 1):
            raise ValueError("y must contain only the integers -1 and 1")
        if value == -1:
            seen_minus = True
        else:
            seen_plus = True
    if not seen_minus or not seen_plus:
        raise ValueError("y must contain both labels -1 and 1")


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

    def kneighbors(
        self, X, n_neighbors=None
    ) -> tuple[list[list[float]], list[list[int]]]:
        if self._X is None:
            raise ValueError(
                "model must be fitted before kneighbors is called"
            )
        width = _check_matrix(X)
        if width != self._width:
            raise ValueError(
                "X must have the same number of features as the training data"
            )
        if n_neighbors is not None and type(n_neighbors) is not int:
            raise ValueError("n_neighbors must be None or an integer")
        k = self.n_neighbors if n_neighbors is None else n_neighbors
        if k < 1 or k > len(self._X):
            raise ValueError(
                "n_neighbors must be between 1 and the number of "
                "training samples"
            )

        all_distances = []
        all_indices = []
        for query in X:
            pairs = []
            for train_index, train_row in enumerate(self._X):
                d2 = _squared_distance(query, train_row)
                try:
                    distance = math.sqrt(d2)
                except (OverflowError, ValueError) as exc:
                    raise FloatingPointError(
                        "non-finite value encountered during distance "
                        "computation"
                    ) from exc
                if not math.isfinite(distance):
                    raise FloatingPointError(
                        "non-finite value encountered during distance "
                        "computation"
                    )
                pairs.append((d2, distance, train_index))
            pairs.sort(key=lambda pair: (pair[0], pair[2]))
            all_distances.append(
                [0.0 if pair[1] == 0.0 else pair[1] for pair in pairs[:k]]
            )
            all_indices.append([int(pair[2]) for pair in pairs[:k]])
        return all_distances, all_indices


class KNeighborsRegressor:
    """Deterministic k-nearest-neighbors regressor.

    Distances are squared Euclidean distances accumulated with
    ``math.fsum`` in feature order. Neighbors are selected by ascending
    ``(distance, training row index)``. With ``weights="uniform"`` the
    prediction is the ``math.fsum`` of the neighbors' targets divided by
    their count. With ``weights="distance"``, any neighbor at exactly zero
    distance makes the prediction the equal-weight mean of all zero-distance
    neighbors; otherwise each target is weighted by ``1 / sqrt(distance)``
    and the numerator and denominator are each accumulated with
    ``math.fsum`` in neighbor order. An exact zero result is written as
    positive ``0.0``. No randomness is used.
    """

    def __init__(self, n_neighbors=5, weights="uniform"):
        if type(n_neighbors) is not int or n_neighbors < 1:
            raise ValueError("n_neighbors must be a positive integer")
        if type(weights) is not str or weights not in ("uniform", "distance"):
            raise ValueError('weights must be "uniform" or "distance"')
        self.n_neighbors = n_neighbors
        self.weights = weights
        self._X = None
        self._y = None
        self._width = None

    def fit(self, X, y):
        # Clear the previous fit up front so that a failed validation
        # leaves the model unfitted.
        self._X = None
        self._y = None
        self._width = None
        try:
            width = _check_gradient_matrix(X)
            _check_gradient_target(y, len(X))
        except OverflowError as exc:
            # An int too large to convert to float failed its finiteness
            # check: still a rejected input, hence ValueError.
            raise ValueError(
                "X and y must contain only finite non-boolean numbers"
            ) from exc
        if self.n_neighbors > len(X):
            raise ValueError(
                "n_neighbors must not exceed the number of training samples"
            )
        self._X = X
        self._y = y
        self._width = width
        return self

    def predict(self, X) -> list[float]:
        if self._X is None:
            raise ValueError("model must be fitted before predict is called")
        try:
            width = _check_gradient_matrix(X)
        except OverflowError as exc:
            raise ValueError(
                "X must contain only finite non-boolean numbers"
            ) from exc
        if width != self._width:
            raise ValueError(
                "X must have the same number of features as the training data"
            )

        results = []
        for query in X:
            distances = [
                _squared_distance(query, train_row) for train_row in self._X
            ]
            order = sorted(
                range(len(self._X)), key=lambda i: (distances[i], i)
            )
            neighbors = order[: self.n_neighbors]
            if self.weights == "uniform":
                results.append(self._equal_weight_mean(neighbors))
            else:
                results.append(self._distance_weighted_mean(neighbors, distances))
        return results

    def _equal_weight_mean(self, indices):
        """fsum of the targets at ``indices`` divided by their count."""
        try:
            total = math.fsum(self._y[i] for i in indices)
            result = total / len(indices)
        except (OverflowError, ValueError) as exc:
            raise FloatingPointError(
                "non-finite value encountered during k-nearest-neighbors "
                "prediction"
            ) from exc
        if not math.isfinite(result):
            raise FloatingPointError(
                "non-finite value encountered during k-nearest-neighbors "
                "prediction"
            )
        return _positive_zero(result)

    def _distance_weighted_mean(self, indices, distances):
        """Inverse-distance weighted mean; zero distances collapse to the
        equal-weight mean over all exactly coincident neighbors."""
        zero = [i for i in indices if distances[i] == 0]
        if zero:
            return self._equal_weight_mean(zero)

        weights = []
        products = []
        for i in indices:
            try:
                root = math.sqrt(distances[i])
                weight = 1.0 / root
                product = weight * self._y[i]
            except (OverflowError, ValueError, ZeroDivisionError) as exc:
                raise FloatingPointError(
                    "non-finite value encountered during k-nearest-neighbors "
                    "prediction"
                ) from exc
            if not (
                math.isfinite(root)
                and math.isfinite(weight)
                and math.isfinite(product)
            ):
                raise FloatingPointError(
                    "non-finite value encountered during k-nearest-neighbors "
                    "prediction"
                )
            weights.append(weight)
            products.append(product)
        try:
            numerator = math.fsum(products)
            denominator = math.fsum(weights)
            result = numerator / denominator
        except (OverflowError, ValueError, ZeroDivisionError) as exc:
            raise FloatingPointError(
                "non-finite value encountered during k-nearest-neighbors "
                "prediction"
            ) from exc
        if not math.isfinite(result):
            raise FloatingPointError(
                "non-finite value encountered during k-nearest-neighbors "
                "prediction"
            )
        return _positive_zero(result)


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


class _DecisionTreeRegressorNode:
    """Node of a DecisionTreeRegressor tree (leaf when feature is None)."""

    __slots__ = ("value", "feature", "threshold", "left", "right")

    def __init__(self, value):
        self.value = value
        self.feature = None
        self.threshold = None
        self.left = None
        self.right = None


def _regressor_tree_mean(values):
    """Mean of a non-empty list as ``math.fsum(values) / len(values)``.

    Any overflow, invalid operation, or non-finite result in the
    ``math.fsum`` or division steps raises FloatingPointError.
    """
    try:
        mean = math.fsum(values) / len(values)
    except (OverflowError, ValueError, ZeroDivisionError) as exc:
        raise FloatingPointError(
            "non-finite value encountered during decision tree regression"
        ) from exc
    if not math.isfinite(mean):
        raise FloatingPointError(
            "non-finite value encountered during decision tree regression"
        )
    return mean


def _regressor_tree_loss(values, mean):
    """Squared-error loss ``fsum((value - mean) ** 2)`` in list order.

    Any overflow, invalid operation, or non-finite intermediate or final
    value in the subtraction, power, or ``math.fsum`` steps raises
    FloatingPointError.
    """
    terms = []
    for value in values:
        try:
            deviation = value - mean
            term = deviation ** 2
        except (OverflowError, ValueError) as exc:
            raise FloatingPointError(
                "non-finite value encountered during decision tree "
                "regression"
            ) from exc
        if not _is_finite_or_int(term):
            raise FloatingPointError(
                "non-finite value encountered during decision tree "
                "regression"
            )
        terms.append(term)
    try:
        loss = math.fsum(terms)
    except (OverflowError, ValueError) as exc:
        raise FloatingPointError(
            "non-finite value encountered during decision tree regression"
        ) from exc
    if not math.isfinite(loss):
        raise FloatingPointError(
            "non-finite value encountered during decision tree regression"
        )
    return loss


class DecisionTreeRegressor:
    """Deterministic decision tree regressor with squared-error splits.

    Each node's value is the mean of its targets, computed as a
    sample-order ``math.fsum`` divided by the number of samples (the root
    is at depth 0). A node becomes a leaf when it reaches ``max_depth``,
    when it holds a single sample, or when no candidate split strictly
    improves on the node's loss. For every feature, the distinct values in
    ascending order -- except the maximum -- serve as thresholds ``t``
    with ``x <= t`` going to the left child. A candidate's loss is the sum
    of the two sides' ``math.fsum((y - side_mean) ** 2)`` in sample order;
    only losses strictly below the parent's same-formula loss are
    accepted, and the best candidate is the first in ascending
    ``(loss, feature index, t)`` order. Children are built left first,
    then right. No randomness is used.
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
        # Clear the previous fit up front so that a failed validation or
        # computation leaves the model unfitted.
        self._root = None
        self._n_features = None
        try:
            width = _check_gradient_matrix(X)
            _check_gradient_target(y, len(X))
        except OverflowError as exc:
            # An int too large to convert to float failed its finiteness
            # check: still a rejected input, hence ValueError.
            raise ValueError(
                "X and y must contain only finite non-boolean numbers"
            ) from exc
        root = self._build(X, y, list(range(len(X))), 0, width)
        self._root = root
        self._n_features = width
        return self

    def _build(self, X, y, indices, depth, width):
        targets = [y[i] for i in indices]
        node = _DecisionTreeRegressorNode(_regressor_tree_mean(targets))

        if len(indices) == 1:
            return node
        if self.max_depth is not None and depth >= self.max_depth:
            return node

        parent_loss = _regressor_tree_loss(targets, node.value)
        best = None  # (loss, feature index, threshold)
        for j in range(width):
            values = sorted(set(X[i][j] for i in indices))
            for t in values[:-1]:
                left_targets = []
                right_targets = []
                for i in indices:
                    if X[i][j] <= t:
                        left_targets.append(y[i])
                    else:
                        right_targets.append(y[i])
                left_mean = _regressor_tree_mean(left_targets)
                right_mean = _regressor_tree_mean(right_targets)
                left_loss = _regressor_tree_loss(left_targets, left_mean)
                right_loss = _regressor_tree_loss(right_targets, right_mean)
                try:
                    loss = left_loss + right_loss
                except (OverflowError, ValueError) as exc:
                    raise FloatingPointError(
                        "non-finite value encountered during decision tree "
                        "regression"
                    ) from exc
                if not math.isfinite(loss):
                    raise FloatingPointError(
                        "non-finite value encountered during decision tree "
                        "regression"
                    )
                if loss < parent_loss and (
                    best is None or (loss, j, t) < best
                ):
                    best = (loss, j, t)
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

    def predict(self, X) -> list[float]:
        if self._root is None:
            raise ValueError("model must be fitted before predict is called")
        try:
            width = _check_gradient_matrix(X)
        except OverflowError as exc:
            raise ValueError(
                "X must contain only finite non-boolean numbers"
            ) from exc
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
            results.append(_positive_zero(node.value))
        return results


class RandomForestRegressor:
    """Deterministic bagged forest of squared-error regression trees.

    The forest holds ``n_estimators`` trees, each grown exactly as
    ``DecisionTreeRegressor(max_depth=max_depth)`` except that at every
    non-leaf node only a random subset of features is searched. Node
    values are sample-order ``math.fsum`` means; a node becomes a leaf
    when it holds one sample, when it reaches ``max_depth``, or when no
    candidate split strictly improves the parent's squared-error loss;
    thresholds and the ascending ``(loss, feature index, t)`` tie-break
    follow the decision tree rules.

    Randomness comes from a single ``random.Random(seed)`` stream consumed
    in tree order. For each tree, ``n`` (the training set size) calls to
    ``rng.randrange(n)`` draw a bootstrap sample with replacement;
    repeated rows are kept in draw order. Then, in left-before-right
    recursion order, every node that is neither a single-sample node nor a
    depth-limited node consumes exactly one
    ``sorted(rng.sample(range(p), max_features))`` draw -- where ``p`` is
    the number of training features -- selecting the features searched at
    that node; this includes nodes that then become leaves because no
    selected feature offers a strict improvement. Single-sample and
    depth-limited nodes consume no sample. Prediction averages the tree
    leaves in tree order with
    ``math.fsum(leaf_values) / n_estimators``. The same parameters and
    inputs always give the same result.
    """

    def __init__(self, n_estimators=10, max_depth=None, max_features=1,
                 seed=0):
        if type(n_estimators) is not int:
            raise ValueError("n_estimators must be an integer")
        if n_estimators <= 0:
            raise ValueError("n_estimators must be greater than 0")
        if max_depth is not None:
            if type(max_depth) is not int or max_depth < 1:
                raise ValueError(
                    "max_depth must be None or a positive integer"
                )
        if type(max_features) is not int:
            raise ValueError("max_features must be an integer")
        if max_features <= 0:
            raise ValueError("max_features must be greater than 0")
        if type(seed) is not int:
            raise ValueError("seed must be an integer")

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.seed = seed
        self._trees = None
        self._n_features = None

    def fit(self, X, y):
        # Clear the previous fit up front so that a failed validation or
        # computation leaves the model unfitted.
        self._trees = None
        self._n_features = None
        try:
            width = _check_gradient_matrix(X)
            _check_gradient_target(y, len(X))
        except OverflowError as exc:
            # An int too large to convert to float failed its finiteness
            # check: still a rejected input, hence ValueError.
            raise ValueError(
                "X and y must contain only finite non-boolean numbers"
            ) from exc
        if self.max_features > width:
            raise ValueError(
                "max_features must not exceed the number of training features"
            )

        n = len(X)
        rng = random.Random(self.seed)
        trees = []
        for _ in range(self.n_estimators):
            indices = [rng.randrange(n) for _ in range(n)]
            trees.append(self._build(X, y, indices, 0, width, rng))

        self._trees = trees
        self._n_features = width
        return self

    def _build(self, X, y, indices, depth, width, rng):
        targets = [y[i] for i in indices]
        node = _DecisionTreeRegressorNode(_regressor_tree_mean(targets))

        if len(indices) == 1:
            return node
        if self.max_depth is not None and depth >= self.max_depth:
            return node

        features = sorted(rng.sample(range(width), self.max_features))

        parent_loss = _regressor_tree_loss(targets, node.value)
        best = None  # (loss, feature index, threshold)
        for j in features:
            values = sorted(set(X[i][j] for i in indices))
            for t in values[:-1]:
                left_targets = []
                right_targets = []
                for i in indices:
                    if X[i][j] <= t:
                        left_targets.append(y[i])
                    else:
                        right_targets.append(y[i])
                left_mean = _regressor_tree_mean(left_targets)
                right_mean = _regressor_tree_mean(right_targets)
                left_loss = _regressor_tree_loss(left_targets, left_mean)
                right_loss = _regressor_tree_loss(right_targets, right_mean)
                try:
                    loss = left_loss + right_loss
                except (OverflowError, ValueError) as exc:
                    raise FloatingPointError(
                        "non-finite value encountered during random forest "
                        "regression"
                    ) from exc
                if not math.isfinite(loss):
                    raise FloatingPointError(
                        "non-finite value encountered during random forest "
                        "regression"
                    )
                if loss < parent_loss and (
                    best is None or (loss, j, t) < best
                ):
                    best = (loss, j, t)
        if best is None:
            return node

        _, feature, threshold = best
        left_indices = [i for i in indices if X[i][feature] <= threshold]
        right_indices = [i for i in indices if X[i][feature] > threshold]
        node.feature = feature
        node.threshold = threshold
        node.left = self._build(
            X, y, left_indices, depth + 1, width, rng
        )
        node.right = self._build(
            X, y, right_indices, depth + 1, width, rng
        )
        return node

    def predict(self, X) -> list[float]:
        if self._trees is None:
            raise ValueError("model must be fitted before predict is called")
        try:
            width = _check_gradient_matrix(X)
        except OverflowError as exc:
            raise ValueError(
                "X must contain only finite non-boolean numbers"
            ) from exc
        if width != self._n_features:
            raise ValueError(
                "X must have the same number of features as the training data"
            )
        results = []
        for row in X:
            values = []
            for tree in self._trees:
                node = tree
                while node.feature is not None:
                    if row[node.feature] <= node.threshold:
                        node = node.left
                    else:
                        node = node.right
                values.append(node.value)
            try:
                prediction = math.fsum(values) / self.n_estimators
            except (OverflowError, ValueError, ZeroDivisionError) as exc:
                raise FloatingPointError(
                    "non-finite value encountered during random forest "
                    "regression"
                ) from exc
            if not math.isfinite(prediction):
                raise FloatingPointError(
                    "non-finite value encountered during random forest "
                    "regression"
                )
            results.append(_positive_zero(prediction))
        return results

    def permutation_importance(self, X, y, n_repeats=5, seed=0):
        """Return permutation feature importances via loss degradation.

        ``X`` must be a non-empty rectangular list of non-empty rows whose
        elements are finite values of type exactly ``int`` or ``float``
        (booleans and subclasses are rejected), and ``y`` must be a list
        with the same length and the same element rule; the number of
        columns must match the fitted model. ``n_repeats`` must be an
        exact positive integer and ``seed`` an exact integer. The model
        must already be fitted. Any violation raises ValueError.

        The baseline loss is ``b = mean_squared_error(y, predict(X))``.
        Randomness comes from a single ``random.Random(seed)`` stream: for
        each feature ``j`` in order and each repeat ``r`` in order, a fresh
        ``idx = list(range(n))`` is shuffled with ``rng.shuffle`` and a
        copy of ``X`` has its column ``j`` replaced by
        ``X[idx[i]][j]`` in row order; the importance is the permuted
        copy's loss minus ``b``. The return value is the triple
        ``(mean, std, raw)`` where ``raw`` is the per-feature list of
        per-repeat importances, ``mean[j] = fsum(raw[j]) / n_repeats``,
        and ``std[j] = sqrt(fsum((v - mean[j]) ** 2) / n_repeats)``
        (population standard deviation). Results are floats and an exact
        zero is normalized to positive ``0.0``. The model and the inputs
        are not modified. OverflowError or ValueError from the
        computation and any non-finite result raise FloatingPointError;
        a FloatingPointError propagates unchanged.
        """
        if self._trees is None:
            raise ValueError(
                "model must be fitted before permutation_importance is called"
            )
        if type(n_repeats) is not int or n_repeats <= 0:
            raise ValueError("n_repeats must be a positive integer")
        if type(seed) is not int:
            raise ValueError("seed must be an integer")
        try:
            width = _check_gradient_matrix(X)
            _check_gradient_target(y, len(X))
        except OverflowError as exc:
            # An int too large to convert to float failed its finiteness
            # check: still a rejected input, hence ValueError.
            raise ValueError(
                "X and y must contain only finite non-boolean numbers"
            ) from exc
        if width != self._n_features:
            raise ValueError(
                "X must have the same number of features as the training data"
            )

        n = len(X)
        try:
            baseline = mean_squared_error(y, self.predict(X))
            if not math.isfinite(baseline):
                raise FloatingPointError(
                    "non-finite value encountered during random forest "
                    "regression"
                )

            rng = random.Random(seed)
            raw = []
            for j in range(width):
                column = [row[j] for row in X]
                values = []
                for _ in range(n_repeats):
                    idx = list(range(n))
                    rng.shuffle(idx)
                    permuted = [list(row) for row in X]
                    for i in range(n):
                        permuted[i][j] = column[idx[i]]
                    loss = mean_squared_error(y, self.predict(permuted))
                    importance = loss - baseline
                    if not math.isfinite(importance):
                        raise FloatingPointError(
                            "non-finite value encountered during random "
                            "forest regression"
                        )
                    values.append(_positive_zero(float(importance)))
                raw.append(values)

            mean = []
            std = []
            for j in range(width):
                average = math.fsum(raw[j]) / n_repeats
                if not math.isfinite(average):
                    raise FloatingPointError(
                        "non-finite value encountered during random forest "
                        "regression"
                    )
                average = _positive_zero(average)
                mean.append(average)
                deviation = math.fsum(
                    (value - average) ** 2 for value in raw[j]
                ) / n_repeats
                standard_deviation = math.sqrt(deviation)
                if not math.isfinite(standard_deviation):
                    raise FloatingPointError(
                        "non-finite value encountered during random forest "
                        "regression"
                    )
                std.append(_positive_zero(standard_deviation))
        except FloatingPointError:
            # A FloatingPointError raised anywhere (including by predict or
            # mean_squared_error) propagates unchanged.
            raise
        except (OverflowError, ValueError) as exc:
            # Any other overflow or invalid-operation during the computation
            # is reported as FloatingPointError.
            raise FloatingPointError(
                "non-finite value encountered during random forest regression"
            ) from exc
        return mean, std, raw


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


class ExtraTreesClassifier:
    """Deterministic extremely-randomized forest of Gini classifiers.

    The forest holds ``n_estimators`` unconstrained decision trees. Unlike
    ``RandomForestClassifier``, every tree is grown on all ``n`` training
    samples (no bootstrap draws). Node labels are the majority label (ties
    go to the smallest label), a split sends ``x <= t`` left, a candidate's
    score is
    ``(|L| * gini(L) + |R| * gini(R)) / |S|``, only scores strictly below
    the parent Gini are accepted, the best candidate is the first in
    ascending ``(score, feature index, t)`` order, and a node becomes a
    leaf when it is pure or no candidate split improves the Gini.

    Randomness comes from a single ``random.Random(seed)`` stream consumed
    in tree order. At every non-pure node in left-before-right recursion
    order, ``sorted(rng.sample(range(p), max_features))`` -- where ``p`` is
    the number of training features -- selects the features examined at
    that node. Pure nodes consume no randomness. For each selected
    feature, its ascending distinct values
    among the node's samples are recorded as ``V``; a constant feature
    yields no candidate and consumes no randomness. Otherwise a single
    ``rng.randrange(len(V) - 1)`` call picks the index ``k`` and the
    threshold is ``V[k]`` (so the maximum value is never a threshold).
    Prediction votes over the trees in order for each input row; ties go
    to the smallest label. ``fit`` first clears any previously fitted
    state, so a failed fit leaves the model unfitted. The inputs are not
    modified and the same arguments always give the same result.
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

    def fit(self, X, y) -> ExtraTreesClassifier:
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
            # Each tree uses every sample exactly once: no bootstrap.
            indices = list(range(n))
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

        # One feature draw per non-pure node, in recursion order.
        features = sorted(rng.sample(range(width), self.max_features))

        parent_gini = _gini_impurity(labels)
        n = len(indices)
        best = None  # (score, feature index, threshold)
        for j in features:
            values = sorted(set(X[i][j] for i in indices))
            if len(values) < 2:
                # A constant feature offers no split and consumes no
                # threshold draw.
                continue
            # One random threshold per selected non-constant feature.
            threshold = values[rng.randrange(len(values) - 1)]
            left_labels = []
            right_labels = []
            for i in indices:
                if X[i][j] <= threshold:
                    left_labels.append(y[i])
                else:
                    right_labels.append(y[i])
            score = (
                len(left_labels) * _gini_impurity(left_labels)
                + len(right_labels) * _gini_impurity(right_labels)
            ) / n
            if score < parent_gini and (
                best is None or (score, j, threshold) < best
            ):
                best = (score, j, threshold)
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

    def predict(self, X) -> list[int]:
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


class _IsolationTreeNode:
    """Node of an isolation tree; leaves store only the sample count."""

    __slots__ = ("count", "feature", "threshold", "left", "right")

    def __init__(self, count):
        self.count = count
        self.feature = None
        self.threshold = None
        self.left = None
        self.right = None


def _isolation_c(s):
    """Return the average path length of an unsuccessful binary search
    tree lookup over ``s`` samples: ``c(1) = 0`` and, for ``s >= 2``,
    ``c(s) = 2*H(s-1) - 2*(s-1)/s`` where ``H(m)`` is the ``math.fsum``
    of ``1/k`` for ``k = 1..m``."""
    if s == 1:
        return 0.0
    harmonic = math.fsum(1.0 / k for k in range(1, s))
    return 2.0 * harmonic - 2.0 * (s - 1) / s


def _isolation_tree_build(X, indices, depth, max_depth, width, rng):
    node = _IsolationTreeNode(len(indices))
    if depth >= max_depth or len(indices) == 1:
        return node
    features = []
    for j in range(width):
        first = X[indices[0]][j]
        for i in indices[1:]:
            if X[i][j] != first:
                features.append(j)
                break
    if not features:
        return node
    feature = rng.choice(features)
    values = sorted(set(X[i][feature] for i in indices))
    threshold = rng.choice(values[:-1])
    left = [i for i in indices if X[i][feature] <= threshold]
    right = [i for i in indices if X[i][feature] > threshold]
    node.feature = feature
    node.threshold = threshold
    node.left = _isolation_tree_build(
        X, left, depth + 1, max_depth, width, rng
    )
    node.right = _isolation_tree_build(
        X, right, depth + 1, max_depth, width, rng
    )
    return node


def _isolation_path(node, row, c_values):
    """Return the path length of ``row``: the depth of the leaf it
    reaches plus ``c`` of the leaf's sample count."""
    depth = 0
    while node.feature is not None:
        depth += 1
        if row[node.feature] <= node.threshold:
            node = node.left
        else:
            node = node.right
    return depth + c_values[node.count]


def isolation_forest_score(X, n_estimators=100, max_samples=256, seed=0):
    """Return the isolation-forest anomaly score of each row of ``X``.

    ``n_estimators`` and ``max_samples`` must be exact integers with
    ``n_estimators > 0`` and ``max_samples >= 2``, and ``seed`` must be
    an exact integer. ``X`` is validated like ``RandomForestClassifier``
    training data and must have at least ``max_samples`` rows; any
    violation raises ValueError.

    Let ``T = n_estimators``, ``psi = max_samples``, ``F = math.fsum``
    and ``rng = random.Random(seed)``. Each of the ``T`` trees draws its
    subsample as ``rng.sample(range(n), psi)`` and is grown from depth 0:
    a node becomes a leaf (storing its sample count) when it reaches
    depth ``ceil(log2(psi))``, holds a single sample, or every column is
    constant; otherwise ``rng.choice`` first picks a feature from the
    ascending list of non-constant features and then a threshold from
    that column's distinct ascending values except the maximum, with
    ``x <= t`` going left and the rest right, recursing left before
    right. The path length of a sample is the depth of the leaf it
    reaches plus ``c`` of the leaf's sample count, and its score is
    ``2 ** (-F(paths in tree order) / T / c(psi))``. Scores come back as
    floats in input order with zero written as ``+0.0``; any
    OverflowError, ValueError, ZeroDivisionError or non-finite result
    after validation raises FloatingPointError. ``X`` is not modified
    and the same arguments always give the same result.
    """
    if type(n_estimators) is not int:
        raise ValueError("n_estimators must be an integer")
    if n_estimators <= 0:
        raise ValueError("n_estimators must be greater than 0")
    if type(max_samples) is not int:
        raise ValueError("max_samples must be an integer")
    if max_samples < 2:
        raise ValueError("max_samples must be at least 2")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    width = _check_tree_matrix(X)
    if len(X) < max_samples:
        raise ValueError("X must have at least max_samples rows")

    try:
        n = len(X)
        rng = random.Random(seed)
        max_depth = math.ceil(math.log2(max_samples))
        c_values = [0.0] + [
            _isolation_c(s) for s in range(1, max_samples + 1)
        ]
        c_psi = c_values[max_samples]
        paths = [[] for _ in range(n)]
        for _ in range(n_estimators):
            indices = rng.sample(range(n), max_samples)
            tree = _isolation_tree_build(
                X, indices, 0, max_depth, width, rng
            )
            for i in range(n):
                paths[i].append(_isolation_path(tree, X[i], c_values))
        scores = []
        for sample_paths in paths:
            score = 2.0 ** (
                -math.fsum(sample_paths) / n_estimators / c_psi
            )
            if not math.isfinite(score):
                raise FloatingPointError(
                    "non-finite value encountered during isolation"
                    " forest scoring"
                )
            scores.append(_positive_zero(score))
        return scores
    except (OverflowError, ValueError, ZeroDivisionError) as exc:
        raise FloatingPointError(
            "non-finite value encountered during isolation forest scoring"
        ) from exc


def local_outlier_factor_score(X, n_neighbors=20) -> list[float]:
    """Return the local outlier factor (LOF) of each row of ``X``.

    ``n_neighbors`` must be an exact integer ``k`` with
    ``1 <= k < len(X)`` and ``X`` must be a non-empty rectangular matrix
    with at least two rows, whose rows are non-empty lists and whose
    elements have type exactly ``int`` or are finite values of type
    exactly ``float`` (booleans and subclasses are rejected); any
    violation raises ValueError.

    Let ``F = math.fsum``. For each pair ``i != j`` the Euclidean
    distance is ``d(i, j) = sqrt(F((X[i][h] - X[j][h]) ** 2))`` summed
    over columns in ascending order. The other samples are sorted by
    ``(d(i, j), j)`` ascending and the ``k``-th entry gives ``kd_i``;
    the neighborhood ``N_i`` is ``{j != i: d(i, j) <= kd_i}`` (ties
    included). For each sample
    ``r_i = max(F(max(kd_j, d(i, j)) for j in N_i) / len(N_i),
    sys.float_info.epsilon)`` with the terms summed over ``j`` in
    ascending index order, and the returned factor is
    ``LOF_i = F(r_i / r_j for j in N_i) / len(N_i)``, again with the
    terms in ascending ``j`` order. Factors come back as floats in input
    order with zero written as ``+0.0``; any OverflowError, ValueError,
    ZeroDivisionError or non-finite intermediate value or result after
    validation raises FloatingPointError. ``X`` is not modified, only
    the standard library is used, and the same arguments always give
    the same result.
    """
    if type(n_neighbors) is not int:
        raise ValueError("n_neighbors must be an integer")
    width = _check_tree_matrix(X)
    if len(X) < 2:
        raise ValueError("X must have at least two rows")
    if n_neighbors < 1 or n_neighbors >= len(X):
        raise ValueError(
            "n_neighbors must satisfy 1 <= n_neighbors < len(X)"
        )

    try:
        n = len(X)
        epsilon = sys.float_info.epsilon
        distances = [[0.0] * n for _ in range(n)]
        for i in range(n):
            row = distances[i]
            for j in range(i + 1, n):
                d_ij = math.sqrt(
                    math.fsum(
                        (X[i][h] - X[j][h]) ** 2 for h in range(width)
                    )
                )
                if not math.isfinite(d_ij):
                    raise FloatingPointError(
                        "non-finite value encountered during local"
                        " outlier factor scoring"
                    )
                row[j] = d_ij
                distances[j][i] = d_ij
        k_distances = [0.0] * n
        neighborhoods = [None] * n
        for i in range(n):
            row = distances[i]
            ordered = sorted(
                (j for j in range(n) if j != i),
                key=lambda j: (row[j], j),
            )
            kd_i = row[ordered[n_neighbors - 1]]
            k_distances[i] = kd_i
            neighborhoods[i] = [j for j in ordered if row[j] <= kd_i]

        reachability = [0.0] * n
        for i in range(n):
            neighbors_i = sorted(neighborhoods[i])
            value = math.fsum(
                max(k_distances[j], distances[j][i]) for j in neighbors_i
            ) / len(neighbors_i)
            if not math.isfinite(value):
                raise FloatingPointError(
                    "non-finite value encountered during local outlier"
                    " factor scoring"
                )
            reachability[i] = max(value, epsilon)

        factors = []
        for i in range(n):
            neighbors_i = sorted(neighborhoods[i])
            factor = math.fsum(
                reachability[i] / reachability[j] for j in neighbors_i
            ) / len(neighbors_i)
            if not math.isfinite(factor):
                raise FloatingPointError(
                    "non-finite value encountered during local outlier"
                    " factor scoring"
                )
            factors.append(_positive_zero(factor))
        return factors
    except (OverflowError, ValueError, ZeroDivisionError) as exc:
        raise FloatingPointError(
            "non-finite value encountered during local outlier factor"
            " scoring"
        ) from exc


def optics_clustering(X, eps, min_samples=5) -> list[int]:
    """Return OPTICS-style cluster labels for the rows of ``X``.

    ``X`` must be a non-empty rectangular matrix whose rows are
    non-empty lists and whose elements have type exactly ``int`` or are
    finite values of type exactly ``float`` (booleans and subclasses
    are rejected), ``eps`` must be a positive exact ``int`` or a finite
    positive ``float``, and ``min_samples`` must be an exact ``int``
    with ``2 <= min_samples <= len(X)``; any violation raises
    ValueError.

    Let ``F = math.fsum``. The Euclidean distance is
    ``d(i, j) = sqrt(F((X[i][k] - X[j][k]) ** 2))`` summed over columns
    in ascending order, and the core distance of point ``i`` is the
    ``min_samples``-th entry of all points sorted by ``(d(i, j), j)``
    ascending. Reachability distances start undefined; each segment
    starts at the smallest unprocessed index, and after processing
    ``p`` every unprocessed ``q`` has its reachability distance
    strictly lowered by ``max(core_distance[p], d(p, q))`` when that
    value is smaller, with the next processed point always the
    candidate with the smallest ``(reachability, q)`` until no
    candidate remains. Labels are then assigned in processing order
    and returned in input order: a segment head or a point whose
    reachability distance exceeds ``eps`` opens a new cluster
    (numbered upward from 0) when its core distance is at most ``eps``
    and is labeled ``-1`` otherwise, and every other point joins the
    current cluster. Any OverflowError, ValueError, ZeroDivisionError
    or non-finite intermediate value or result after validation raises
    FloatingPointError. ``X`` is not modified, only the standard
    library is used, and the same arguments always give the same
    result.
    """
    width = _check_tree_matrix(X)
    if type(eps) is int:
        if eps <= 0:
            raise ValueError("eps must be positive")
    elif type(eps) is float and math.isfinite(eps):
        if eps <= 0.0:
            raise ValueError("eps must be positive")
    else:
        raise ValueError(
            "eps must be a positive int or a finite positive float"
        )
    if type(min_samples) is not int:
        raise ValueError("min_samples must be an integer")
    n = len(X)
    if min_samples < 2 or min_samples > n:
        raise ValueError(
            "min_samples must satisfy 2 <= min_samples <= len(X)"
        )

    try:
        distances = [[0.0] * n for _ in range(n)]
        for i in range(n):
            row = distances[i]
            for j in range(i + 1, n):
                d_ij = math.sqrt(
                    math.fsum(
                        (X[i][k] - X[j][k]) ** 2 for k in range(width)
                    )
                )
                if not math.isfinite(d_ij):
                    raise FloatingPointError(
                        "non-finite value encountered during optics"
                        " clustering"
                    )
                row[j] = d_ij
                distances[j][i] = d_ij

        core = [0.0] * n
        for i in range(n):
            row = distances[i]
            ordered = sorted(range(n), key=lambda j: (row[j], j))
            core[i] = row[ordered[min_samples - 1]]
            if not math.isfinite(core[i]):
                raise FloatingPointError(
                    "non-finite value encountered during optics"
                    " clustering"
                )

        processed = [False] * n
        reach = [None] * n
        segment_head = [False] * n
        order = []
        remaining = n
        while remaining:
            p = 0
            while processed[p]:
                p += 1
            segment_head[p] = True
            while True:
                processed[p] = True
                order.append(p)
                remaining -= 1
                row = distances[p]
                core_p = core[p]
                best = -1
                best_reach = None
                for q in range(n):
                    if processed[q]:
                        continue
                    r = max(core_p, row[q])
                    if not math.isfinite(r):
                        raise FloatingPointError(
                            "non-finite value encountered during optics"
                            " clustering"
                        )
                    rq = reach[q]
                    if rq is None or r < rq:
                        reach[q] = r
                        rq = r
                    if best_reach is None or rq < best_reach:
                        best_reach = rq
                        best = q
                if best < 0:
                    break
                p = best

        labels = [0] * n
        current = -1
        next_cluster = 0
        for p in order:
            if segment_head[p] or reach[p] > eps:
                if core[p] <= eps:
                    current = next_cluster
                    next_cluster += 1
                    labels[p] = current
                else:
                    labels[p] = -1
            else:
                labels[p] = current
        return labels
    except (OverflowError, ValueError, ZeroDivisionError) as exc:
        raise FloatingPointError(
            "non-finite value encountered during optics clustering"
        ) from exc


class IsolationForest:
    """Reusable deterministic isolation-forest anomaly detector.

    ``n_estimators`` and ``max_samples`` must be exact integers with
    ``n_estimators > 0`` and ``max_samples >= 2``, ``seed`` must be an
    exact integer, and ``contamination`` must be the string ``"auto"``
    or a finite non-boolean value of type exactly ``int`` or ``float``
    with ``0 < contamination <= 0.5``; any violation raises ValueError.

    ``fit(X)`` first clears any previously fitted state, so a failed
    fit leaves the model unfitted. ``X`` is validated like
    ``isolation_forest_score`` training data and must have at least
    ``max_samples`` rows; any violation raises ValueError. The forest is
    built exactly as in ``isolation_forest_score``: with
    ``T = n_estimators``, ``psi = max_samples`` and
    ``rng = random.Random(seed)``, each of the ``T`` trees draws its
    subsample as ``rng.sample(range(n), psi)`` and is grown from depth
    0, becoming a leaf (storing its sample count) at depth
    ``ceil(log2(psi))``, on a single sample, or when every column is
    constant; otherwise ``rng.choice`` picks a feature from the
    ascending non-constant features and then a threshold from that
    column's distinct ascending values except the maximum, with
    ``x <= t`` going left, recursing left before right. The path length
    of a sample is the depth of the leaf it reaches plus ``c`` of the
    leaf's sample count, and its score is
    ``2 ** (-math.fsum(paths in tree order) / T / c(psi))``.

    The decision threshold ``threshold_`` is ``0.5`` when
    ``contamination`` is ``"auto"``; otherwise, with
    ``m = ceil(contamination * n)``, it is the ``m``-th entry of the
    training scores sorted in descending order.

    ``score_samples(X)`` scores a non-empty matrix with the same number
    of columns as the training data using the saved trees, and
    ``predict(X)`` returns ``-1`` for rows whose score is at least
    ``threshold_`` and ``1`` otherwise. Calling either before fitting,
    or with invalid or wrongly-shaped ``X``, raises ValueError. Scores
    come back as floats in input order with zero written as ``+0.0``;
    any OverflowError, ValueError, ZeroDivisionError or non-finite
    result after validation raises FloatingPointError. ``X`` is not
    modified and the same arguments always give the same result.
    """

    def __init__(self, n_estimators=100, max_samples=256,
                 contamination="auto", seed=0):
        if type(n_estimators) is not int:
            raise ValueError("n_estimators must be an integer")
        if n_estimators <= 0:
            raise ValueError("n_estimators must be greater than 0")
        if type(max_samples) is not int:
            raise ValueError("max_samples must be an integer")
        if max_samples < 2:
            raise ValueError("max_samples must be at least 2")
        if type(contamination) is str:
            valid = contamination == "auto"
        elif type(contamination) is int or type(contamination) is float:
            valid = 0.0 < contamination <= 0.5
        else:
            valid = False
        if not valid:
            raise ValueError(
                'contamination must be "auto" or a finite number in'
                " (0, 0.5]"
            )
        if type(seed) is not int:
            raise ValueError("seed must be an integer")
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.seed = seed
        self._clear()

    def _clear(self):
        """Reset all fitted state so the model is unfitted."""
        self._trees = None
        self._c_values = None
        self._c_psi = None
        self._width = None
        self.threshold_ = None

    def fit(self, X) -> IsolationForest:
        """Fit the forest on ``X`` and return ``self``."""
        self._clear()
        width = _check_tree_matrix(X)
        if len(X) < self.max_samples:
            raise ValueError("X must have at least max_samples rows")
        try:
            n = len(X)
            rng = random.Random(self.seed)
            max_depth = math.ceil(math.log2(self.max_samples))
            c_values = [0.0] + [
                _isolation_c(s) for s in range(1, self.max_samples + 1)
            ]
            c_psi = c_values[self.max_samples]
            trees = []
            paths = [[] for _ in range(n)]
            for _ in range(self.n_estimators):
                indices = rng.sample(range(n), self.max_samples)
                tree = _isolation_tree_build(
                    X, indices, 0, max_depth, width, rng
                )
                trees.append(tree)
                for i in range(n):
                    paths[i].append(_isolation_path(tree, X[i], c_values))
            scores = []
            for sample_paths in paths:
                score = 2.0 ** (
                    -math.fsum(sample_paths) / self.n_estimators / c_psi
                )
                if not math.isfinite(score):
                    raise FloatingPointError(
                        "non-finite value encountered during isolation"
                        " forest scoring"
                    )
                scores.append(_positive_zero(score))
        except (OverflowError, ValueError, ZeroDivisionError) as exc:
            raise FloatingPointError(
                "non-finite value encountered during isolation forest"
                " scoring"
            ) from exc
        if self.contamination == "auto":
            threshold = 0.5
        else:
            m = math.ceil(self.contamination * n)
            threshold = sorted(scores, reverse=True)[m - 1]
        self._trees = trees
        self._c_values = c_values
        self._c_psi = c_psi
        self._width = width
        self.threshold_ = threshold
        return self

    def score_samples(self, X) -> list[float]:
        """Return the anomaly score of each row of ``X``."""
        if self._trees is None:
            raise ValueError("IsolationForest is not fitted")
        width = _check_tree_matrix(X)
        if width != self._width:
            raise ValueError(
                "X must have the same number of columns as the training"
                " data"
            )
        try:
            scores = []
            for row in X:
                score = 2.0 ** (
                    -math.fsum(
                        _isolation_path(tree, row, self._c_values)
                        for tree in self._trees
                    ) / len(self._trees) / self._c_psi
                )
                if not math.isfinite(score):
                    raise FloatingPointError(
                        "non-finite value encountered during isolation"
                        " forest scoring"
                    )
                scores.append(_positive_zero(score))
            return scores
        except (OverflowError, ValueError, ZeroDivisionError) as exc:
            raise FloatingPointError(
                "non-finite value encountered during isolation forest"
                " scoring"
            ) from exc

    def predict(self, X) -> list[int]:
        """Return ``-1`` for rows scoring at least ``threshold_``, else
        ``1``."""
        return [
            -1 if score >= self.threshold_ else 1
            for score in self.score_samples(X)
        ]

    def permutation_importance(self, X, n_repeats=5, seed=0) -> tuple:
        """Return permutation feature importances via score change.

        ``X`` must be a non-empty rectangular list of non-empty rows whose
        elements are finite values of type exactly ``int`` or ``float``
        (booleans and subclasses are rejected), and the number of columns
        must match the fitted model. ``n_repeats`` must be an exact
        positive integer and ``seed`` an exact integer. The model must
        already be fitted. Any violation raises ValueError.

        The baseline is ``b = score_samples(X)``. Randomness comes from a
        single ``random.Random(seed)`` stream: for each feature ``j`` in
        order and each repeat ``r`` in order, a fresh
        ``idx = list(range(n))`` is shuffled with ``rng.shuffle`` and a
        copy of ``X`` has its column ``j`` replaced by
        ``X[idx[i]][j]`` in row order; writing ``s`` for the copy's
        ``score_samples`` result,
        ``raw[j][r] = fsum(abs(s[i] - b[i]) for i in input order) / n``.
        The return value is the triple ``(mean, std, raw)`` where ``raw``
        is the per-feature list of per-repeat importances,
        ``mean[j] = fsum(raw[j]) / n_repeats``, and
        ``std[j] = sqrt(fsum((v - mean[j]) ** 2) / n_repeats)``
        (population standard deviation). Results are floats and an exact
        zero is normalized to positive ``0.0``. The model and the inputs
        are not modified and the same arguments always give the same
        result. Exceptions raised by ``score_samples`` propagate
        unchanged; any other OverflowError, ValueError or
        ZeroDivisionError during the computation and any non-finite
        result raise FloatingPointError.
        """
        if self._trees is None:
            raise ValueError(
                "model must be fitted before permutation_importance is called"
            )
        if type(n_repeats) is not int or n_repeats <= 0:
            raise ValueError("n_repeats must be a positive integer")
        if type(seed) is not int:
            raise ValueError("seed must be an integer")
        try:
            width = _check_gradient_matrix(X)
        except OverflowError as exc:
            # An int too large to convert to float failed its finiteness
            # check: still a rejected input, hence ValueError.
            raise ValueError(
                "X must contain only finite non-boolean numbers"
            ) from exc
        if width != self._width:
            raise ValueError(
                "X must have the same number of features as the training data"
            )

        n = len(X)
        # Exceptions from score_samples (including FloatingPointError)
        # propagate unchanged, so the scoring calls live outside the
        # conversion try below.
        baseline = self.score_samples(X)

        def permuted_matrix(j, idx, column):
            # Build the column-permuted copy. These are not score_samples
            # steps, so any OverflowError, ValueError or ZeroDivisionError
            # they raise is reported as FloatingPointError.
            try:
                permuted = [list(row) for row in X]
                for i in range(n):
                    permuted[i][j] = column[idx[i]]
                return permuted
            except FloatingPointError:
                raise
            except (OverflowError, ValueError, ZeroDivisionError) as exc:
                raise FloatingPointError(
                    "non-finite value encountered during isolation forest"
                    " scoring"
                ) from exc

        rng = random.Random(seed)
        permuted_scores = []
        for j in range(width):
            try:
                column = [row[j] for row in X]
            except FloatingPointError:
                raise
            except (OverflowError, ValueError, ZeroDivisionError) as exc:
                raise FloatingPointError(
                    "non-finite value encountered during isolation forest"
                    " scoring"
                ) from exc
            repeats = []
            for _ in range(n_repeats):
                idx = list(range(n))
                try:
                    rng.shuffle(idx)
                except FloatingPointError:
                    raise
                except (
                    OverflowError,
                    ValueError,
                    ZeroDivisionError,
                ) as exc:
                    # The shuffle is part of the computation rather than a
                    # score_samples call, so its arithmetic errors are
                    # reported as FloatingPointError like every other
                    # non-scoring step.
                    raise FloatingPointError(
                        "non-finite value encountered during isolation forest"
                        " scoring"
                    ) from exc
                permuted = permuted_matrix(j, idx, column)
                # Exceptions from score_samples (including
                # FloatingPointError) propagate unchanged, so this call
                # deliberately stays outside the conversion handler.
                repeats.append(self.score_samples(permuted))
            permuted_scores.append(repeats)

        try:
            raw = []
            for j in range(width):
                values = []
                for scores in permuted_scores[j]:
                    importance = math.fsum(
                        abs(scores[i] - baseline[i]) for i in range(n)
                    ) / n
                    if not math.isfinite(importance):
                        raise FloatingPointError(
                            "non-finite value encountered during isolation"
                            " forest scoring"
                        )
                    values.append(_positive_zero(float(importance)))
                raw.append(values)

            mean = []
            std = []
            for j in range(width):
                average = math.fsum(raw[j]) / n_repeats
                if not math.isfinite(average):
                    raise FloatingPointError(
                        "non-finite value encountered during isolation"
                        " forest scoring"
                    )
                average = _positive_zero(average)
                mean.append(average)
                deviation = math.fsum(
                    (value - average) ** 2 for value in raw[j]
                ) / n_repeats
                standard_deviation = math.sqrt(deviation)
                if not math.isfinite(standard_deviation):
                    raise FloatingPointError(
                        "non-finite value encountered during isolation"
                        " forest scoring"
                    )
                std.append(_positive_zero(standard_deviation))
        except FloatingPointError:
            # A FloatingPointError raised anywhere propagates unchanged.
            raise
        except (OverflowError, ValueError, ZeroDivisionError) as exc:
            # Any other overflow, invalid operation, or division by zero
            # during the computation is reported as FloatingPointError.
            raise FloatingPointError(
                "non-finite value encountered during isolation forest scoring"
            ) from exc
        return mean, std, raw


class AdaBoostClassifier:
    """Deterministic discrete AdaBoost (SAMME sign form) of decision stumps.

    Training maintains sample weights ``w_i`` with ``w_i = 1 / n`` initially.
    At each round the weak classifier is a threshold rule on a single
    feature: ``h = s`` when ``x <= t`` and ``-s`` otherwise. Candidates are
    enumerated by feature index in ascending order, then by threshold ``t``
    taken over all distinct ascending column values (the maximum yields a
    constant prediction), then by sign ``s = -1`` before ``s = 1``. For
    each candidate the weighted misclassification error ``e`` is the
    ``math.fsum``, taken in sample order, of the weights of the
    misclassified samples; the selected candidate is the first in ascending
    lexicographic ``(e, feature, t, s)`` order.

    If ``e >= 0.5`` training ends without adding that candidate; if no weak
    classifier has been saved yet, ``fit`` raises ValueError. Otherwise the
    candidate weight is ``a = 0.5 * log((1 - q) / q)`` where
    ``q = min(max(e, 1e-15), 1 - 1e-15)``, the weights are updated
    synchronously to ``w_i * exp(-a * y_i * h_i)`` and renormalized with
    ``math.fsum`` in sample order, and the weak classifier together with
    ``a`` is saved. When ``e == 0`` training ends after the save; otherwise
    it runs for at most ``n_estimators`` rounds.

    Prediction sums ``a * h`` over the saved weak classifiers in save order
    using ``math.fsum`` and returns ``1`` for a strictly positive sum and
    ``-1`` otherwise. No randomness is used.
    """

    def __init__(self, n_estimators=50):
        if type(n_estimators) is not int or n_estimators <= 0:
            raise ValueError("n_estimators must be a positive integer")
        self.n_estimators = n_estimators
        self._stumps = None
        self._n_features = None

    def fit(self, X, y):
        self._stumps = None
        self._n_features = None
        width = _check_tree_matrix(X)
        _check_signed_label_vector(y, len(X))

        n = len(X)
        weights = [1.0 / n for _ in range(n)]
        stumps = []
        for _ in range(self.n_estimators):
            best = None  # (error, feature index, threshold, sign)
            best_predictions = None
            for j in range(width):
                values = sorted(set(row[j] for row in X))
                for t in values:
                    for s in (-1, 1):
                        predictions = [
                            s if row[j] <= t else -s for row in X
                        ]
                        try:
                            error = math.fsum(
                                weights[i]
                                for i in range(n)
                                if predictions[i] != y[i]
                            )
                        except (OverflowError, ValueError) as exc:
                            raise FloatingPointError(
                                "non-finite value encountered during boosting"
                            ) from exc
                        if not math.isfinite(error):
                            raise FloatingPointError(
                                "non-finite value encountered during boosting"
                            )
                        candidate = (error, j, t, s)
                        if best is None or candidate < best:
                            best = candidate
                            best_predictions = predictions

            error, feature, threshold, sign = best
            if error >= 0.5:
                if not stumps:
                    raise ValueError(
                        "every weak classifier has error of at least 0.5"
                    )
                break

            q = min(max(error, 1e-15), 1.0 - 1e-15)
            try:
                alpha = 0.5 * math.log((1.0 - q) / q)
            except (OverflowError, ValueError) as exc:
                raise FloatingPointError(
                    "non-finite value encountered during boosting"
                ) from exc
            if not math.isfinite(alpha):
                raise FloatingPointError(
                    "non-finite value encountered during boosting"
                )
            updated = []
            for i in range(n):
                try:
                    value = weights[i] * math.exp(
                        -alpha * y[i] * best_predictions[i]
                    )
                except (OverflowError, ValueError) as exc:
                    raise FloatingPointError(
                        "non-finite value encountered during boosting"
                    ) from exc
                if not math.isfinite(value):
                    raise FloatingPointError(
                        "non-finite value encountered during boosting"
                    )
                updated.append(value)
            try:
                total = math.fsum(updated)
            except (OverflowError, ValueError) as exc:
                raise FloatingPointError(
                    "non-finite value encountered during boosting"
                ) from exc
            if not math.isfinite(total) or total <= 0.0:
                raise FloatingPointError(
                    "non-finite value encountered during boosting"
                )
            normalized = []
            for value in updated:
                factor = value / total
                if not math.isfinite(factor):
                    raise FloatingPointError(
                        "non-finite value encountered during boosting"
                    )
                normalized.append(factor)

            stumps.append((feature, threshold, sign, alpha))
            weights = normalized
            if error == 0.0:
                break

        self._stumps = stumps
        self._n_features = width
        return self

    def predict(self, X):
        if self._stumps is None:
            raise ValueError("model must be fitted before predict is called")
        width = _check_tree_matrix(X)
        if width != self._n_features:
            raise ValueError(
                "X must have the same number of features as the training data"
            )

        results = []
        for row in X:
            terms = []
            for feature, threshold, sign, alpha in self._stumps:
                vote = sign if row[feature] <= threshold else -sign
                try:
                    term = alpha * vote
                except (OverflowError, ValueError) as exc:
                    raise FloatingPointError(
                        "non-finite value encountered during boosting"
                    ) from exc
                if isinstance(term, float) and not math.isfinite(term):
                    raise FloatingPointError(
                        "non-finite value encountered during boosting"
                    )
                terms.append(term)
            try:
                score = math.fsum(terms)
            except (OverflowError, ValueError) as exc:
                raise FloatingPointError(
                    "non-finite value encountered during boosting"
                ) from exc
            if not math.isfinite(score):
                raise FloatingPointError(
                    "non-finite value encountered during boosting"
                )
            results.append(1 if score > 0 else -1)
        return results

    def decision_function(self, X) -> list[float]:
        """Return the signed margin of each row of X in input order.

        X must satisfy the same validation as in predict: a non-empty
        rectangular matrix of finite non-boolean numbers with the same
        number of columns as the training data, and the model must be
        fitted. Each saved weak classifier votes ``sign`` when
        ``x[feature] <= threshold`` and ``-sign`` otherwise; the margin
        is the ``math.fsum`` of ``alpha * vote`` over the weak
        classifiers in save order. An OverflowError, ValueError, or
        ZeroDivisionError from the multiplication or fsum, or any
        non-finite term or sum, raises FloatingPointError. An exact zero
        margin is returned as positive 0.0. The input and the model are
        not modified and repeated calls return identical values.
        """
        if self._stumps is None:
            raise ValueError(
                "model must be fitted before decision_function is called"
            )
        width = _check_tree_matrix(X)
        if width != self._n_features:
            raise ValueError(
                "X must have the same number of features as the training data"
            )

        results = []
        for row in X:
            terms = []
            for feature, threshold, sign, alpha in self._stumps:
                vote = sign if row[feature] <= threshold else -sign
                try:
                    term = alpha * vote
                except (
                    OverflowError,
                    ValueError,
                    ZeroDivisionError,
                ) as exc:
                    raise FloatingPointError(
                        "non-finite value encountered during boosting"
                    ) from exc
                if isinstance(term, float) and not math.isfinite(term):
                    raise FloatingPointError(
                        "non-finite value encountered during boosting"
                    )
                terms.append(term)
            try:
                score = math.fsum(terms)
            except (
                OverflowError,
                ValueError,
                ZeroDivisionError,
            ) as exc:
                raise FloatingPointError(
                    "non-finite value encountered during boosting"
                ) from exc
            if not math.isfinite(score):
                raise FloatingPointError(
                    "non-finite value encountered during boosting"
                )
            if score == 0:
                score = 0.0
            results.append(score)
        return results

    def predict_proba(self, X) -> list[list[float]]:
        """Return the class probabilities of each row of X in input
        order.

        X must satisfy the same validation as in predict (validation is
        delegated to :meth:`decision_function`). For the margin ``s`` the
        score ``z = 2.0 * s`` is mapped through the numerically stable
        sigmoid (the ``z >= 0`` branch uses ``1 / (1 + exp(-z))``,
        otherwise ``e = exp(z); e / (1 + e)``); each row returns
        ``[1.0 - p, p]`` whose columns correspond to labels -1 and 1 in
        that order. An OverflowError, ValueError, or ZeroDivisionError
        from the multiplication, exp, addition, subtraction, or
        division, or any non-finite intermediate or result, raises
        FloatingPointError. An exact zero probability is returned as
        positive 0.0. The input and the model are not modified and
        repeated calls return identical values.
        """
        scores = self.decision_function(X)
        results = []
        for score in scores:
            try:
                z = 2.0 * score
                if not math.isfinite(z):
                    raise FloatingPointError(
                        "non-finite value encountered during boosting"
                    )
                if z >= 0:
                    p = 1.0 / (1.0 + math.exp(-z))
                else:
                    e = math.exp(z)
                    p = e / (1.0 + e)
                negative = 1.0 - p
            except (
                OverflowError,
                ValueError,
                ZeroDivisionError,
            ) as exc:
                raise FloatingPointError(
                    "non-finite value encountered during boosting"
                ) from exc
            if not math.isfinite(p) or not math.isfinite(negative):
                raise FloatingPointError(
                    "non-finite value encountered during boosting"
                )
            if p == 0:
                p = 0.0
            if negative == 0:
                negative = 0.0
            results.append([negative, p])
        return results


def _is_finite_or_int(value):
    """Return True for an int of any size or a finite non-boolean float."""
    if type(value) is int:
        return True
    return type(value) is float and math.isfinite(value)


def _is_exact_finite_number(value):
    """Return True for an exact finite int or float; booleans and other
    types return False. An int too large to convert to float makes the
    finiteness check raise OverflowError."""
    if type(value) is int:
        return math.isfinite(value)
    return type(value) is float and math.isfinite(value)


def _require_exact_finite_positive(value, name):
    """Validate a finite, non-boolean, strictly positive exact int/float.

    An OverflowError from the finiteness check on an oversized integer is
    reported as ValueError like every other rejected value.
    """
    try:
        valid = (
            type(value) in (int, float)
            and not isinstance(value, bool)
            and math.isfinite(value)
            and value > 0
        )
    except OverflowError as exc:
        raise ValueError(
            "%s must be a finite non-boolean positive number" % name
        ) from exc
    if not valid:
        raise ValueError(
            "%s must be a finite non-boolean positive number" % name
        )


def _require_exact_finite_number(value, name):
    """Validate an exact finite int/float (booleans rejected).

    An OverflowError from the finiteness check on an oversized integer is
    reported as ValueError like every other rejected value.
    """
    try:
        valid = (
            type(value) in (int, float)
            and not isinstance(value, bool)
            and math.isfinite(value)
        )
    except OverflowError as exc:
        raise ValueError(
            "%s must be a finite non-boolean number" % name
        ) from exc
    if not valid:
        raise ValueError("%s must be a finite non-boolean number" % name)


def _positive_zero(value):
    """Normalize an exact zero (of either sign) to positive 0.0."""
    return 0.0 if value == 0 else value


def _check_gradient_matrix(X):
    """Validate a non-empty rectangular matrix of exact ints or finite
    floats (booleans and subclasses are rejected). Unlike the tree matrix
    check, integers are also passed through ``math.isfinite`` so that an
    int too large to convert to float raises OverflowError from the
    finiteness check, which the caller maps to ValueError."""
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
            if not _is_exact_finite_number(value):
                raise ValueError(
                    "X must contain only finite non-boolean numbers"
                )
    return width


def _check_gradient_target(y, n):
    """Validate a target vector of exact ints or finite floats with the
    same length as X (booleans and subclasses are rejected). An int too
    large to convert to float raises OverflowError from the finiteness
    check, which the caller maps to ValueError."""
    if not isinstance(y, list) or len(y) != n:
        raise ValueError("y must be a list with the same length as X")
    for value in y:
        if not _is_exact_finite_number(value):
            raise ValueError("y must contain only finite non-boolean numbers")


def _require_exact_positive_or_finite_float(value, name):
    """Validate a strictly positive parameter without float conversion.

    An exact ``int`` of any magnitude passes (its arithmetic use decides
    whether it overflows); a ``float`` must be finite and positive.
    Booleans, subclasses and every other type are ValueError.
    """
    if type(value) is int:
        if value <= 0:
            raise ValueError(
                "%s must be a finite non-boolean positive number" % name
            )
        return
    if type(value) is float and math.isfinite(value) and value > 0.0:
        return
    raise ValueError(
        "%s must be a finite non-boolean positive number" % name
    )


def _is_exact_int_or_finite_float(value):
    """Return True for an exact int of any magnitude or a finite
    non-boolean float; booleans, subclasses and other types return False.
    No integer is ever converted through float, so this never raises."""
    if type(value) is int:
        return True
    return type(value) is float and math.isfinite(value)


def _require_exact_unit_interval_number(value, name):
    """Validate a non-boolean exact int/float lying in the closed interval
    ``[0, 1]``. An exact ``int`` must be ``0`` or ``1``; a ``float`` must
    be finite and within the range. Booleans, subclasses and every other
    type are ValueError.
    """
    if isinstance(value, bool) or type(value) not in (int, float):
        raise ValueError(
            "%s must be a finite non-boolean number in [0, 1]" % name
        )
    if type(value) is float and not math.isfinite(value):
        raise ValueError(
            "%s must be a finite non-boolean number in [0, 1]" % name
        )
    if not (0 <= value <= 1):
        raise ValueError(
            "%s must be a finite non-boolean number in [0, 1]" % name
        )


def _check_lasso_matrix(X):
    """Validate a non-empty rectangular matrix whose elements are exact
    ints of any magnitude or finite floats (booleans and subclasses are
    rejected). Integers deliberately bypass ``math.isfinite`` so that an
    oversized int is accepted as input and only rejected through an
    arithmetic failure during fitting or prediction."""
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
            if not _is_exact_int_or_finite_float(value):
                raise ValueError(
                    "X must contain only finite non-boolean numbers"
                )
    return width


def _check_lasso_target(y, n):
    """Validate a target vector of exact ints (any magnitude) or finite
    floats with the same length as X (booleans and subclasses are
    rejected)."""
    if not isinstance(y, list) or len(y) != n:
        raise ValueError("y must be a list with the same length as X")
    for value in y:
        if not _is_exact_int_or_finite_float(value):
            raise ValueError("y must contain only finite non-boolean numbers")


def _check_gradient_binary_target(y, n):
    """Validate a binary target vector with the same length as X.

    Every element has type exactly ``int`` (booleans and subclasses are
    rejected) and equals ``0`` or ``1``, and both classes must occur.
    """
    if not isinstance(y, list) or len(y) != n:
        raise ValueError("y must be a list with the same length as X")
    seen_zero = False
    seen_one = False
    for value in y:
        if type(value) is not int or value not in (0, 1):
            raise ValueError("y must contain only the integers 0 and 1")
        if value == 0:
            seen_zero = True
        else:
            seen_one = True
    if not seen_zero or not seen_one:
        raise ValueError("y must contain both classes 0 and 1")


class GradientBoostingRegressor:
    """Deterministic gradient boosting for regression with level-wise
    decision stumps and a constant learning rate.

    The initial constant is ``c = fsum(y) / n`` and every prediction starts
    there. At each round the pseudo-residuals are ``r_i = y_i - v_i`` where
    ``v_i`` is the current prediction; candidate stumps are enumerated by
    feature index ``j`` in ascending order, then by threshold ``t`` taken
    over the distinct ascending values of column ``j`` except its maximum
    (the maximum would leave the right side empty). A stump assigns the mean
    residual of the samples with ``x_ij <= t`` to the left side (``a``) and
    the mean residual of the remaining samples to the right side (``b``),
    each mean a sample-order ``math.fsum`` divided by the side size. The
    selected stump is the first in ascending lexicographic
    ``(fsum((r_i - side_mean) ** 2 in sample order), j, t)`` order; the
    saved increments are ``learning_rate * a`` and ``learning_rate * b`` and
    the predictions are updated synchronously with them.

    A round ends training without saving a stump when no threshold exists
    on any feature (every column is constant); otherwise training stops
    after saving when the larger absolute value of the two increments is at
    most ``tol``, and in any case after at most ``n_estimators`` rounds.
    Prediction adds the saved increment of the side each sample belongs to,
    in save order, starting from ``c``. No randomness is used.
    """

    def __init__(self, n_estimators=100, learning_rate=0.1, tol=1e-8):
        if type(n_estimators) is not int or n_estimators <= 0:
            raise ValueError("n_estimators must be a positive integer")
        _require_exact_finite_positive(learning_rate, "learning_rate")
        _require_exact_finite_positive(tol, "tol")

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.tol = tol
        self._constant = None
        self._stumps = None
        self._n_features = None

    def fit(self, X, y):
        # Clear the previous fit up front so that a failed validation or
        # computation leaves the model unfitted.
        self._constant = None
        self._stumps = None
        self._n_features = None
        try:
            width = _check_gradient_matrix(X)
            _check_gradient_target(y, len(X))
        except OverflowError as exc:
            # An int too large to convert to float failed its finiteness
            # check: still a rejected input, hence ValueError.
            raise ValueError(
                "X and y must contain only finite non-boolean numbers"
            ) from exc

        n = len(X)
        try:
            constant = math.fsum(y) / n
        except (OverflowError, ValueError) as exc:
            raise FloatingPointError(
                "non-finite value encountered during gradient boosting"
            ) from exc
        if not math.isfinite(constant):
            raise FloatingPointError(
                "non-finite value encountered during gradient boosting"
            )
        constant = _positive_zero(constant)

        predictions = [constant for _ in range(n)]
        eta = self.learning_rate
        stumps = []

        for _ in range(self.n_estimators):
            residuals = []
            for i in range(n):
                try:
                    residual = y[i] - predictions[i]
                except (OverflowError, ValueError) as exc:
                    raise FloatingPointError(
                        "non-finite value encountered during gradient boosting"
                    ) from exc
                if not _is_finite_or_int(residual):
                    raise FloatingPointError(
                        "non-finite value encountered during gradient boosting"
                    )
                residuals.append(_positive_zero(residual))

            best = None  # (squared error, feature, threshold)
            best_left = None
            best_right = None
            for j in range(width):
                thresholds = sorted(set(row[j] for row in X))[:-1]
                for t in thresholds:
                    left = [
                        residuals[i] for i in range(n) if X[i][j] <= t
                    ]
                    right = [
                        residuals[i]
                        for i in range(n)
                        if not X[i][j] <= t
                    ]
                    left_mean, right_mean = self._side_means(left, right)

                    terms = []
                    for i in range(n):
                        mean = left_mean if X[i][j] <= t else right_mean
                        try:
                            deviation = residuals[i] - mean
                            term = deviation ** 2
                        except (OverflowError, ValueError) as exc:
                            raise FloatingPointError(
                                "non-finite value encountered during gradient "
                                "boosting"
                            ) from exc
                        if not _is_finite_or_int(term):
                            raise FloatingPointError(
                                "non-finite value encountered during gradient "
                                "boosting"
                            )
                        terms.append(term)
                    try:
                        error = math.fsum(terms)
                    except (OverflowError, ValueError) as exc:
                        raise FloatingPointError(
                            "non-finite value encountered during gradient "
                            "boosting"
                        ) from exc
                    if not math.isfinite(error):
                        raise FloatingPointError(
                            "non-finite value encountered during gradient "
                            "boosting"
                        )

                    candidate = (error, j, t)
                    if best is None or candidate < best:
                        best = candidate
                        best_left = left_mean
                        best_right = right_mean

            if best is None:
                # Every column is constant: no split is possible, so the
                # constant model is retained.
                break

            try:
                increment_left = eta * best_left
                increment_right = eta * best_right
            except (OverflowError, ValueError) as exc:
                raise FloatingPointError(
                    "non-finite value encountered during gradient boosting"
                ) from exc
            if not (
                _is_finite_or_int(increment_left)
                and _is_finite_or_int(increment_right)
            ):
                raise FloatingPointError(
                    "non-finite value encountered during gradient boosting"
                )
            increment_left = _positive_zero(increment_left)
            increment_right = _positive_zero(increment_right)

            _, feature, threshold = best
            stumps.append((feature, threshold, increment_left, increment_right))
            for i in range(n):
                increment = (
                    increment_left
                    if X[i][feature] <= threshold
                    else increment_right
                )
                try:
                    updated = predictions[i] + increment
                except (OverflowError, ValueError) as exc:
                    raise FloatingPointError(
                        "non-finite value encountered during gradient boosting"
                    ) from exc
                if not _is_finite_or_int(updated):
                    raise FloatingPointError(
                        "non-finite value encountered during gradient boosting"
                    )
                predictions[i] = _positive_zero(updated)

            if (
                max(abs(increment_left), abs(increment_right)) <= self.tol
            ):
                break

        self._constant = constant
        self._stumps = stumps
        self._n_features = width
        return self

    @staticmethod
    def _side_means(left, right):
        """Return the sample-order fsum means of the two non-empty sides."""
        means = []
        for side in (left, right):
            try:
                mean = math.fsum(side) / len(side)
            except (OverflowError, ValueError) as exc:
                raise FloatingPointError(
                    "non-finite value encountered during gradient boosting"
                ) from exc
            if not math.isfinite(mean):
                raise FloatingPointError(
                    "non-finite value encountered during gradient boosting"
                )
            means.append(_positive_zero(mean))
        return means[0], means[1]

    def predict(self, X) -> list[float]:
        if self._stumps is None:
            raise ValueError("model must be fitted before predict is called")
        try:
            width = _check_gradient_matrix(X)
        except OverflowError as exc:
            raise ValueError(
                "X must contain only finite non-boolean numbers"
            ) from exc
        if width != self._n_features:
            raise ValueError(
                "X must have the same number of features as the training data"
            )

        results = []
        for row in X:
            prediction = self._constant
            for feature, threshold, increment_left, increment_right in (
                self._stumps
            ):
                increment = (
                    increment_left
                    if row[feature] <= threshold
                    else increment_right
                )
                try:
                    prediction = prediction + increment
                except (OverflowError, ValueError) as exc:
                    raise FloatingPointError(
                        "non-finite value encountered during gradient boosting"
                    ) from exc
                if not _is_finite_or_int(prediction):
                    raise FloatingPointError(
                        "non-finite value encountered during gradient boosting"
                    )
                prediction = _positive_zero(prediction)
            results.append(prediction)
        return results

    def staged_predict(self, X) -> list[list[float]]:
        """Return the predictions after every prefix of the saved stumps.

        Validation matches ``predict``: the model must be fitted and ``X``
        must be a non-empty rectangular matrix of finite non-boolean exact
        int/float values with as many columns as the training data. The
        result is a fresh list of ``len(_stumps) + 1`` fresh lists, one per
        input row each: index 0 holds the initial constant for every row
        and index r holds the predictions after accumulating the first r
        stumps in save order. Each addition follows ``predict`` exactly,
        including the positive-zero normalization and FloatingPointError
        for overflow or non-finite results. Neither the argument nor the
        model is modified.
        """
        if self._stumps is None:
            raise ValueError(
                "model must be fitted before staged_predict is called"
            )
        try:
            width = _check_gradient_matrix(X)
        except OverflowError as exc:
            raise ValueError(
                "X must contain only finite non-boolean numbers"
            ) from exc
        if width != self._n_features:
            raise ValueError(
                "X must have the same number of features as the training data"
            )

        current = [self._constant for _ in X]
        stages = [current]
        for feature, threshold, increment_left, increment_right in (
            self._stumps
        ):
            updated = []
            for i, row in enumerate(X):
                increment = (
                    increment_left
                    if row[feature] <= threshold
                    else increment_right
                )
                try:
                    prediction = current[i] + increment
                except (OverflowError, ValueError) as exc:
                    raise FloatingPointError(
                        "non-finite value encountered during gradient boosting"
                    ) from exc
                if not _is_finite_or_int(prediction):
                    raise FloatingPointError(
                        "non-finite value encountered during gradient boosting"
                    )
                updated.append(_positive_zero(prediction))
            current = updated
            stages.append(current)
        return stages


class GradientBoostingClassifier:
    """Deterministic gradient boosting for binary classification with
    level-wise decision stumps and a constant learning rate.

    The initial score is the log-odds of the training base rate:
    ``q = min(max(fsum(y) / n, 1e-15), 1 - 1e-15)`` and
    ``c = log(q / (1 - q))``, and every score starts at ``c``. At each
    round the probability ``p_i`` is the numerically stable logistic
    sigmoid of the current score (the same formulation as
    ``LogisticRegression``) and the pseudo-residuals are
    ``r_i = y_i - p_i``. Candidate stumps are enumerated exactly as in
    ``GradientBoostingRegressor``: feature index ``j`` ascending, then the
    threshold ``t`` over the distinct ascending values of column ``j``
    except its maximum, with ``x_ij <= t`` on the left side; each side
    outputs the sample-order ``math.fsum`` mean of its residuals and the
    selected stump is the first in ascending lexicographic
    ``(fsum((r_i - side_output) ** 2 in sample order), j, t)`` order.
    The saved increments are ``learning_rate`` times the two side outputs
    and the scores are updated synchronously with them.

    Training ends without saving when no threshold exists (every column
    is constant); otherwise it stops after saving when the larger
    absolute increment is at most ``tol``, and in any case after at most
    ``n_estimators`` rounds. Prediction accumulates the saved increment
    of the side each sample belongs to, in save order, starting from
    ``c``, and returns ``1`` for a non-negative score and ``0``
    otherwise. No randomness is used.
    """

    def __init__(self, n_estimators=100, learning_rate=0.1, tol=1e-8):
        if type(n_estimators) is not int or n_estimators <= 0:
            raise ValueError("n_estimators must be a positive integer")
        _require_exact_finite_positive(learning_rate, "learning_rate")
        _require_exact_finite_positive(tol, "tol")

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.tol = tol
        self._constant = None
        self._stumps = None
        self._n_features = None

    def fit(self, X, y):
        # Clear the previous fit up front so that a failed validation or
        # computation leaves the model unfitted.
        self._constant = None
        self._stumps = None
        self._n_features = None
        try:
            width = _check_gradient_matrix(X)
        except OverflowError as exc:
            # An int too large to convert to float failed its finiteness
            # check: still a rejected input, hence ValueError.
            raise ValueError(
                "X must contain only finite non-boolean numbers"
            ) from exc
        _check_gradient_binary_target(y, len(X))

        n = len(X)
        try:
            q = math.fsum(y) / n
        except (OverflowError, ValueError) as exc:
            raise FloatingPointError(
                "non-finite value encountered during gradient boosting"
            ) from exc
        if q < 1e-15:
            q = 1e-15
        elif q > 1.0 - 1e-15:
            q = 1.0 - 1e-15
        try:
            constant = math.log(q / (1.0 - q))
        except (OverflowError, ValueError) as exc:
            raise FloatingPointError(
                "non-finite value encountered during gradient boosting"
            ) from exc
        if not math.isfinite(constant):
            raise FloatingPointError(
                "non-finite value encountered during gradient boosting"
            )
        constant = _positive_zero(constant)

        scores = [constant for _ in range(n)]
        eta = self.learning_rate
        stumps = []

        for _ in range(self.n_estimators):
            residuals = []
            for i in range(n):
                try:
                    probability = _sigmoid(scores[i])
                    residual = y[i] - probability
                except (OverflowError, ValueError) as exc:
                    raise FloatingPointError(
                        "non-finite value encountered during gradient boosting"
                    ) from exc
                if not _is_finite_or_int(residual):
                    raise FloatingPointError(
                        "non-finite value encountered during gradient boosting"
                    )
                residuals.append(_positive_zero(residual))

            best = None  # (squared error, feature, threshold)
            best_left = None
            best_right = None
            for j in range(width):
                thresholds = sorted(set(row[j] for row in X))[:-1]
                for t in thresholds:
                    left = [
                        residuals[i] for i in range(n) if X[i][j] <= t
                    ]
                    right = [
                        residuals[i]
                        for i in range(n)
                        if not X[i][j] <= t
                    ]
                    left_mean, right_mean = (
                        GradientBoostingRegressor._side_means(left, right)
                    )

                    terms = []
                    for i in range(n):
                        mean = left_mean if X[i][j] <= t else right_mean
                        try:
                            deviation = residuals[i] - mean
                            term = deviation ** 2
                        except (OverflowError, ValueError) as exc:
                            raise FloatingPointError(
                                "non-finite value encountered during gradient "
                                "boosting"
                            ) from exc
                        if not _is_finite_or_int(term):
                            raise FloatingPointError(
                                "non-finite value encountered during gradient "
                                "boosting"
                            )
                        terms.append(term)
                    try:
                        error = math.fsum(terms)
                    except (OverflowError, ValueError) as exc:
                        raise FloatingPointError(
                            "non-finite value encountered during gradient "
                            "boosting"
                        ) from exc
                    if not math.isfinite(error):
                        raise FloatingPointError(
                            "non-finite value encountered during gradient "
                            "boosting"
                        )

                    candidate = (error, j, t)
                    if best is None or candidate < best:
                        best = candidate
                        best_left = left_mean
                        best_right = right_mean

            if best is None:
                # Every column is constant: no split is possible, so the
                # constant model is retained.
                break

            try:
                increment_left = eta * best_left
                increment_right = eta * best_right
            except (OverflowError, ValueError) as exc:
                raise FloatingPointError(
                    "non-finite value encountered during gradient boosting"
                ) from exc
            if not (
                _is_finite_or_int(increment_left)
                and _is_finite_or_int(increment_right)
            ):
                raise FloatingPointError(
                    "non-finite value encountered during gradient boosting"
                )
            increment_left = _positive_zero(increment_left)
            increment_right = _positive_zero(increment_right)

            _, feature, threshold = best
            stumps.append((feature, threshold, increment_left, increment_right))
            for i in range(n):
                increment = (
                    increment_left
                    if X[i][feature] <= threshold
                    else increment_right
                )
                try:
                    updated = scores[i] + increment
                except (OverflowError, ValueError) as exc:
                    raise FloatingPointError(
                        "non-finite value encountered during gradient boosting"
                    ) from exc
                if not _is_finite_or_int(updated):
                    raise FloatingPointError(
                        "non-finite value encountered during gradient boosting"
                    )
                scores[i] = _positive_zero(updated)

            if (
                max(abs(increment_left), abs(increment_right)) <= self.tol
            ):
                break

        self._constant = constant
        self._stumps = stumps
        self._n_features = width
        return self

    def predict(self, X) -> list[int]:
        if self._stumps is None:
            raise ValueError("model must be fitted before predict is called")
        try:
            width = _check_gradient_matrix(X)
        except OverflowError as exc:
            raise ValueError(
                "X must contain only finite non-boolean numbers"
            ) from exc
        if width != self._n_features:
            raise ValueError(
                "X must have the same number of features as the training data"
            )

        results = []
        for row in X:
            score = self._constant
            for feature, threshold, increment_left, increment_right in (
                self._stumps
            ):
                increment = (
                    increment_left
                    if row[feature] <= threshold
                    else increment_right
                )
                try:
                    score = score + increment
                except (OverflowError, ValueError) as exc:
                    raise FloatingPointError(
                        "non-finite value encountered during gradient boosting"
                    ) from exc
                if not _is_finite_or_int(score):
                    raise FloatingPointError(
                        "non-finite value encountered during gradient boosting"
                    )
                score = _positive_zero(score)
            results.append(1 if score >= 0 else 0)
        return results

    def decision_function(self, X) -> list[float]:
        """Return the accumulated score of each row of X in input order.

        Validation matches ``predict`` exactly: the model must be fitted
        and ``X`` must be a non-empty rectangular matrix of finite
        non-boolean exact int/float values with as many columns as the
        training data. Each score starts at ``_constant`` and adds the
        saved left increment when ``row[feature] <= threshold`` and the
        right increment otherwise, for every stump in save order, using
        the same arithmetic, positive-zero normalization and
        FloatingPointError handling as ``predict``. The result is a fresh
        list of floats (an integer score is converted after the last
        accumulation; an overflowing conversion or a non-finite result
        raises FloatingPointError); an exact zero is returned as positive
        0.0. A score of at least zero predicts class 1, so
        ``predict`` and ``decision_function`` agree on every row. Neither
        the argument nor the model is modified.
        """
        if self._stumps is None:
            raise ValueError(
                "model must be fitted before decision_function is called"
            )
        try:
            width = _check_gradient_matrix(X)
        except OverflowError as exc:
            raise ValueError(
                "X must contain only finite non-boolean numbers"
            ) from exc
        if width != self._n_features:
            raise ValueError(
                "X must have the same number of features as the training data"
            )

        results = []
        for row in X:
            score = self._constant
            for feature, threshold, increment_left, increment_right in (
                self._stumps
            ):
                increment = (
                    increment_left
                    if row[feature] <= threshold
                    else increment_right
                )
                try:
                    score = score + increment
                except (
                    OverflowError,
                    ValueError,
                    ZeroDivisionError,
                ) as exc:
                    raise FloatingPointError(
                        "non-finite value encountered during gradient boosting"
                    ) from exc
                if not _is_finite_or_int(score):
                    raise FloatingPointError(
                        "non-finite value encountered during gradient boosting"
                    )
                score = _positive_zero(score)
            try:
                score = float(score)
            except (
                OverflowError,
                ValueError,
                ZeroDivisionError,
            ) as exc:
                raise FloatingPointError(
                    "non-finite value encountered during gradient boosting"
                ) from exc
            if not math.isfinite(score):
                raise FloatingPointError(
                    "non-finite value encountered during gradient boosting"
                )
            results.append(_positive_zero(score))
        return results

    def predict_proba(self, X) -> list[list[float]]:
        """Return the class probabilities of each row of X in input
        order.

        Validation is delegated to :meth:`decision_function`, which
        applies the same checks as ``predict``. For each score ``s`` the
        margin ``z = 2.0 * s`` is mapped through the numerically stable
        logistic sigmoid (the ``z >= 0`` branch uses
        ``p = 1 / (1 + exp(-z))``, otherwise ``e = exp(z)`` and
        ``p = e / (1 + e)``) and the row returns ``[1.0 - p, p]`` whose
        columns correspond to classes 0 and 1 in that order. An
        OverflowError, ValueError, or ZeroDivisionError from the
        multiplication, exp, addition, subtraction, or division, or any
        non-finite intermediate or result, raises FloatingPointError. An
        exact zero probability is returned as positive 0.0. The result
        is a fresh list of fresh lists; the input and the model are not
        modified and repeated calls return identical values.
        """
        scores = self.decision_function(X)
        results = []
        for score in scores:
            try:
                z = 2.0 * score
                if not math.isfinite(z):
                    raise FloatingPointError(
                        "non-finite value encountered during gradient boosting"
                    )
                if z >= 0:
                    p = 1.0 / (1.0 + math.exp(-z))
                else:
                    e = math.exp(z)
                    p = e / (1.0 + e)
                negative = 1.0 - p
            except (
                OverflowError,
                ValueError,
                ZeroDivisionError,
            ) as exc:
                raise FloatingPointError(
                    "non-finite value encountered during gradient boosting"
                ) from exc
            if not math.isfinite(p) or not math.isfinite(negative):
                raise FloatingPointError(
                    "non-finite value encountered during gradient boosting"
                )
            results.append([_positive_zero(negative), _positive_zero(p)])
        return results


class LassoRegression:
    """Linear regression with an L1 penalty, fitted by coordinate descent.

    Training starts from ``w = [0.0] * n_features`` and
    ``b = fsum(y) / n``. Each round first sets the intercept to
    ``b = fsum(y_i - fsum(w_j * x_ij)) / n`` and then visits the features
    ``j`` in ascending order. For each feature it computes
    ``r = fsum(x_ij * (y_i - b - fsum(w_k * x_ik for k != j))) / n`` and
    ``z = fsum(x_ij ** 2) / n``. When ``z`` is zero the weight becomes
    ``0.0``; otherwise the soft-thresholded value ``s`` is ``r - alpha``
    when ``r > alpha``, ``r + alpha`` when ``r < -alpha`` and ``0``
    otherwise, giving ``w_j = s / z``. Every sum iterates over indices in
    ascending order and uses ``math.fsum``. Training stops after a round
    whose largest absolute change in ``b`` or any weight is at most
    ``tol``, and in any case after at most ``max_iter`` rounds. No
    randomness is used.
    """

    def __init__(self, alpha=1.0, max_iter=1000, tol=1e-8):
        # alpha/tol accept an exact int of any magnitude (only their use
        # against finite float arithmetic decides whether it overflows) or
        # a strictly positive finite float; booleans and subclasses are
        # rejected without ever converting the int through float.
        _require_exact_positive_or_finite_float(alpha, "alpha")
        if type(max_iter) is not int or max_iter <= 0:
            raise ValueError("max_iter must be a positive integer")
        _require_exact_positive_or_finite_float(tol, "tol")

        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.w = None
        self.b = None

    def fit(self, X, y):
        # Clear any previous fit up front so that a failed validation or
        # computation leaves the model unfitted.
        self.w = None
        self.b = None
        width = _check_lasso_matrix(X)
        _check_lasso_target(y, len(X))

        n = len(X)
        w = [0.0] * width
        try:
            b = math.fsum(y) / n
        except (OverflowError, ValueError, ZeroDivisionError) as exc:
            raise FloatingPointError(
                "non-finite value encountered during lasso regression"
            ) from exc
        if not math.isfinite(b):
            raise FloatingPointError(
                "non-finite value encountered during lasso regression"
            )
        b = _positive_zero(b)
        alpha = self.alpha
        tol = self.tol

        try:
            for _ in range(self.max_iter):
                previous_b = b
                b = math.fsum(
                    y[i]
                    - math.fsum(w[j] * X[i][j] for j in range(width))
                    for i in range(n)
                ) / n
                if not math.isfinite(b):
                    raise FloatingPointError(
                        "non-finite value encountered during lasso regression"
                    )
                max_change = abs(b - previous_b)

                for j in range(width):
                    previous_w = w[j]
                    r = math.fsum(
                        X[i][j]
                        * (
                            y[i]
                            - b
                            - math.fsum(
                                w[k] * X[i][k]
                                for k in range(width)
                                if k != j
                            )
                        )
                        for i in range(n)
                    ) / n
                    z = math.fsum(X[i][j] ** 2 for i in range(n)) / n
                    if not math.isfinite(r) or not math.isfinite(z):
                        raise FloatingPointError(
                            "non-finite value encountered during lasso "
                            "regression"
                        )
                    if z == 0.0:
                        updated = 0.0
                    else:
                        if r > alpha:
                            s = r - alpha
                        elif r < -alpha:
                            s = r + alpha
                        else:
                            s = 0.0
                        updated = s / z
                        if not math.isfinite(updated):
                            raise FloatingPointError(
                                "non-finite value encountered during lasso "
                                "regression"
                            )
                    w[j] = updated
                    change = abs(updated - previous_w)
                    if change > max_change:
                        max_change = change

                b = _positive_zero(b)
                if max_change <= tol:
                    break
        except (OverflowError, ValueError, ZeroDivisionError) as exc:
            raise FloatingPointError(
                "non-finite value encountered during lasso regression"
            ) from exc

        self.w = w
        self.b = b
        return self

    def predict(self, X):
        if self.w is None or self.b is None:
            raise ValueError("model must be fitted before predict is called")
        # Integers of any magnitude are structurally legal input; only an
        # arithmetic failure below rejects such a row.
        width = _check_lasso_matrix(X)
        if width != len(self.w):
            raise ValueError(
                "X must have the same number of features as the training data"
            )

        results = []
        for row in X:
            try:
                prediction = (
                    math.fsum(self.w[j] * row[j] for j in range(width))
                    + self.b
                )
            except (
                OverflowError,
                ValueError,
                ZeroDivisionError,
            ) as exc:
                raise FloatingPointError(
                    "non-finite prediction encountered"
                ) from exc
            if not math.isfinite(prediction):
                raise FloatingPointError("non-finite prediction encountered")
            results.append(_positive_zero(prediction))
        return results


class ElasticNetRegression:
    """Linear regression with a combined L1/L2 penalty, fitted by
    coordinate descent.

    Training starts from ``w = [0.0] * n_features`` and
    ``b = fsum(y) / n``. Each round first sets the intercept to
    ``b = fsum(y_i - fsum(w_j * x_ij)) / n`` and then visits the features
    ``j`` in ascending order. For each feature it computes
    ``r = fsum(x_ij * (y_i - b - fsum(w_k * x_ik for k != j))) / n``,
    ``z = fsum(x_ij ** 2) / n + alpha * (1 - l1_ratio)`` and
    ``q = alpha * l1_ratio``. When ``z`` is zero the weight becomes
    ``0.0``; otherwise the soft-thresholded value is ``(r - q) / z`` when
    ``r > q``, ``(r + q) / z`` when ``r < -q`` and ``0.0`` otherwise.
    Every sum iterates over indices in ascending order and uses
    ``math.fsum``. Training stops after a round whose largest absolute
    change in ``b`` or any weight is at most ``tol``, and in any case
    after at most ``max_iter`` rounds. No randomness is used.
    """

    def __init__(
        self, alpha=1.0, l1_ratio=0.5, max_iter=1000, tol=1e-8
    ):
        # Validation mirrors LassoRegression for alpha/max_iter/tol;
        # l1_ratio additionally has to lie in the closed interval [0, 1].
        _require_exact_positive_or_finite_float(alpha, "alpha")
        _require_exact_unit_interval_number(l1_ratio, "l1_ratio")
        if type(max_iter) is not int or max_iter <= 0:
            raise ValueError("max_iter must be a positive integer")
        _require_exact_positive_or_finite_float(tol, "tol")

        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.w = None
        self.b = None

    def fit(self, X, y):
        # Clear any previous fit up front so that a failed validation or
        # computation leaves the model unfitted.
        self.w = None
        self.b = None
        width = _check_lasso_matrix(X)
        _check_lasso_target(y, len(X))

        n = len(X)
        w = [0.0] * width
        try:
            b = math.fsum(y) / n
        except (OverflowError, ValueError, ZeroDivisionError) as exc:
            raise FloatingPointError(
                "non-finite value encountered during elastic net regression"
            ) from exc
        if not math.isfinite(b):
            raise FloatingPointError(
                "non-finite value encountered during elastic net regression"
            )
        b = _positive_zero(b)
        alpha = self.alpha
        l1_ratio = self.l1_ratio
        tol = self.tol

        try:
            for _ in range(self.max_iter):
                previous_b = b
                b = math.fsum(
                    y[i]
                    - math.fsum(w[j] * X[i][j] for j in range(width))
                    for i in range(n)
                ) / n
                if not math.isfinite(b):
                    raise FloatingPointError(
                        "non-finite value encountered during elastic net "
                        "regression"
                    )
                max_change = abs(b - previous_b)

                for j in range(width):
                    previous_w = w[j]
                    r = math.fsum(
                        X[i][j]
                        * (
                            y[i]
                            - b
                            - math.fsum(
                                w[k] * X[i][k]
                                for k in range(width)
                                if k != j
                            )
                        )
                        for i in range(n)
                    ) / n
                    z = (
                        math.fsum(X[i][j] ** 2 for i in range(n)) / n
                        + alpha * (1 - l1_ratio)
                    )
                    q = alpha * l1_ratio
                    if not math.isfinite(r) or not math.isfinite(z):
                        raise FloatingPointError(
                            "non-finite value encountered during elastic net "
                            "regression"
                        )
                    if z == 0.0:
                        updated = 0.0
                    else:
                        if r > q:
                            s = r - q
                        elif r < -q:
                            s = r + q
                        else:
                            s = 0.0
                        updated = s / z
                        if not math.isfinite(updated):
                            raise FloatingPointError(
                                "non-finite value encountered during elastic "
                                "net regression"
                            )
                    w[j] = updated
                    change = abs(updated - previous_w)
                    if change > max_change:
                        max_change = change

                b = _positive_zero(b)
                if max_change <= tol:
                    break
        except (OverflowError, ValueError, ZeroDivisionError) as exc:
            raise FloatingPointError(
                "non-finite value encountered during elastic net regression"
            ) from exc

        self.w = w
        self.b = b
        return self

    def predict(self, X):
        if self.w is None or self.b is None:
            raise ValueError("model must be fitted before predict is called")
        # Integers of any magnitude are structurally legal input; only an
        # arithmetic failure below rejects such a row.
        width = _check_lasso_matrix(X)
        if width != len(self.w):
            raise ValueError(
                "X must have the same number of features as the training data"
            )

        results = []
        for row in X:
            try:
                prediction = (
                    math.fsum(self.w[j] * row[j] for j in range(width))
                    + self.b
                )
            except (
                OverflowError,
                ValueError,
                ZeroDivisionError,
            ) as exc:
                raise FloatingPointError(
                    "non-finite prediction encountered"
                ) from exc
            if not math.isfinite(prediction):
                raise FloatingPointError("non-finite prediction encountered")
            results.append(_positive_zero(prediction))
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

    After a successful fit, ``transform`` returns, for each input row, a
    fresh list of length ``n_clusters`` with the squared Euclidean
    distances to the centroids in ascending centroid index order; an exact
    zero is normalized to ``0.0``. ``fit_predict`` fits and then returns
    the prediction labels; an exception raised by ``fit`` propagates
    unchanged.
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

    def transform(self, X):
        if self.cluster_centers_ is None:
            raise ValueError("model must be fitted before transform is called")
        width = _check_exact_matrix(X)
        if width != self._n_features:
            raise ValueError(
                "X must have the same number of features as the training data"
            )

        results = []
        for row in X:
            distances = []
            for k in range(self.n_clusters):
                distance = _squared_distance(row, self.cluster_centers_[k])
                if distance == 0:
                    distance = 0.0
                distances.append(distance)
            results.append(distances)
        return results

    def fit_predict(self, X):
        self.fit(X)
        return self.predict(X)


def _euclidean_distance(a, b):
    """Euclidean distance ``sqrt(math.fsum((a[j] - b[j]) ** 2))`` between
    two equal-length rows, accumulated in ascending column order.

    Every subtraction, squaring, summation, and square-root step is
    checked; overflow, invalid operations, and non-finite intermediate
    values raise FloatingPointError.
    """
    terms = []
    for j in range(len(a)):
        try:
            diff = a[j] - b[j]
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
    try:
        result = math.sqrt(total)
    except (OverflowError, ValueError) as exc:
        raise FloatingPointError(
            "non-finite value encountered during distance computation"
        ) from exc
    if not math.isfinite(result):
        raise FloatingPointError(
            "non-finite value encountered during distance computation"
        )
    return result


class KMedoids:
    """Deterministic k-medoids clustering (PAM-style swap-free updates).

    The distance between two rows is
    ``d(a, b) = sqrt(math.fsum((a[j] - b[j]) ** 2))`` accumulated in
    ascending column order. The first medoid is sample 0; each subsequent
    medoid is the not-yet-chosen sample whose minimum distance to the
    already chosen medoids is largest (ties go to the smallest sample
    index). Each round assigns every sample, in row order, to the nearest
    medoid (ties go to the smallest medoid position), then synchronously
    replaces each cluster's medoid with the member minimizing the
    ``math.fsum`` sum of distances to the cluster's members accumulated
    in input order (ties go to the smallest original sample index);
    empty clusters keep their previous medoid. Rounds stop when the
    medoid indices are unchanged, after at most ``max_iter`` rounds.
    Neither ``fit`` nor ``predict`` modifies its input. The same
    parameters and inputs always give the same result.

    ``fit_predict`` fits and then returns the prediction labels as a
    fresh list; an exception raised by ``fit`` propagates unchanged.
    """

    def __init__(self, n_clusters=8, max_iter=300):
        if type(n_clusters) is not int:
            raise ValueError("n_clusters must be an integer")
        if n_clusters <= 0:
            raise ValueError("n_clusters must be greater than 0")
        if type(max_iter) is not int:
            raise ValueError("max_iter must be an integer")
        if max_iter <= 0:
            raise ValueError("max_iter must be greater than 0")

        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.medoid_indices_ = None
        self.cluster_centers_ = None
        self._n_features = None

    @staticmethod
    def _assign(row, medoids):
        """Return the position of the medoid closest to ``row``; ties go
        to the smallest medoid position."""
        best_index = 0
        best_distance = None
        for k in range(len(medoids)):
            distance = _euclidean_distance(row, medoids[k])
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_index = k
        return best_index

    def fit(self, X):
        self.medoid_indices_ = None
        self.cluster_centers_ = None
        self._n_features = None

        width = _check_exact_matrix(X)
        n = len(X)
        if self.n_clusters > n:
            raise ValueError(
                "n_clusters must not exceed the number of samples"
            )

        medoid_indices = [0]
        while len(medoid_indices) < self.n_clusters:
            best_index = None
            best_distance = None
            for i in range(n):
                if i in medoid_indices:
                    continue
                nearest = None
                for m in medoid_indices:
                    distance = _euclidean_distance(X[i], X[m])
                    if nearest is None or distance < nearest:
                        nearest = distance
                if best_distance is None or nearest > best_distance:
                    best_distance = nearest
                    best_index = i
            medoid_indices.append(best_index)

        for _ in range(self.max_iter):
            medoids = [X[m] for m in medoid_indices]
            clusters = [[] for _ in range(self.n_clusters)]
            for i in range(n):
                clusters[self._assign(X[i], medoids)].append(i)

            new_indices = []
            for k in range(self.n_clusters):
                members = clusters[k]
                if not members:
                    new_indices.append(medoid_indices[k])
                    continue
                best_member = None
                best_total = None
                for i in members:
                    try:
                        total = math.fsum(
                            _euclidean_distance(X[i], X[j]) for j in members
                        )
                    except (OverflowError, ValueError) as exc:
                        raise FloatingPointError(
                            "non-finite value encountered during fit"
                        ) from exc
                    if not math.isfinite(total):
                        raise FloatingPointError(
                            "non-finite value encountered during fit"
                        )
                    if best_total is None or total < best_total:
                        best_total = total
                        best_member = i
                new_indices.append(best_member)

            if new_indices == medoid_indices:
                break
            medoid_indices = new_indices

        self.medoid_indices_ = list(medoid_indices)
        self.cluster_centers_ = [list(X[m]) for m in medoid_indices]
        self._n_features = width
        return self

    def predict(self, X):
        if self.medoid_indices_ is None:
            raise ValueError("model must be fitted before predict is called")
        width = _check_exact_matrix(X)
        if width != self._n_features:
            raise ValueError(
                "X must have the same number of features as the training data"
            )

        return [self._assign(row, self.cluster_centers_) for row in X]

    def fit_predict(self, X):
        self.fit(X)
        return self.predict(X)


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
    ``inverse_transform`` maps each one-column row ``[z]`` back to
    ``[F([mu_j, z * v_j]) for j in (0, 1)]`` with an exact zero
    normalized to ``0.0``. ``reconstruction_error`` returns
    ``F((x_ij - r_ij) ** 2) / (2 * n)`` where ``R`` is
    ``inverse_transform(transform(X))``, the squared terms are
    accumulated in row-then-column order, and a zero result is
    normalized to ``0.0``. No method modifies its input or the fitted
    state. No randomness is used.
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

    @staticmethod
    def _validate_z(Z):
        # Non-empty single-column list[list] whose elements have type
        # exactly int (any size) or finite float; booleans and
        # subclasses are rejected.
        if not isinstance(Z, list) or len(Z) == 0:
            raise ValueError("Z must be a non-empty list of rows")
        for row in Z:
            if not isinstance(row, list):
                raise ValueError("Z rows must be lists")
            if len(row) != 1:
                raise ValueError("Z must have exactly one column")
            value = row[0]
            if type(value) is int:
                continue
            if type(value) is float and math.isfinite(value):
                continue
            raise ValueError(
                "Z must contain only finite non-boolean numbers"
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

    def inverse_transform(self, Z):
        if self.mean_ is None or self.components_ is None:
            raise ValueError(
                "PCA must be fitted before inverse_transform is called"
            )
        self._validate_z(Z)

        mean = self.mean_
        v = self.components_[0]

        results = []
        for row in Z:
            z = row[0]
            reconstructed = []
            for j in range(2):
                try:
                    q = z * v[j]
                    value = math.fsum([mean[j], q])
                except (OverflowError, ValueError, ZeroDivisionError) as exc:
                    raise FloatingPointError(
                        "non-finite value encountered "
                        "during inverse_transform"
                    ) from exc
                if not math.isfinite(value):
                    raise FloatingPointError(
                        "non-finite value encountered "
                        "during inverse_transform"
                    )
                if value == 0:
                    value = 0.0
                reconstructed.append(value)
            results.append(reconstructed)
        return results

    def reconstruction_error(self, X):
        if self.mean_ is None or self.components_ is None:
            raise ValueError(
                "PCA must be fitted before reconstruction_error is called"
            )
        self._validate(X)

        # transform and inverse_transform propagate their own exceptions
        # unchanged; validation has already ruled out non-numeric inputs.
        Z = self.transform(X)
        R = self.inverse_transform(Z)
        n = len(X)

        terms = []
        for i in range(n):
            for j in range(2):
                try:
                    diff = X[i][j] - R[i][j]
                    term = diff ** 2
                except (OverflowError, ValueError, ZeroDivisionError) as exc:
                    raise FloatingPointError(
                        "non-finite value encountered "
                        "during reconstruction_error"
                    ) from exc
                if isinstance(term, float) and not math.isfinite(term):
                    raise FloatingPointError(
                        "non-finite value encountered "
                        "during reconstruction_error"
                    )
                terms.append(term)

        try:
            error = math.fsum(terms) / (2 * n)
        except (OverflowError, ValueError, ZeroDivisionError) as exc:
            raise FloatingPointError(
                "non-finite value encountered during reconstruction_error"
            ) from exc
        if not math.isfinite(error):
            raise FloatingPointError(
                "non-finite value encountered during reconstruction_error"
            )
        if error == 0:
            error = 0.0
        return error

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class DBSCAN:
    """Deterministic density-based spatial clustering.

    Two points are neighbors when their squared Euclidean distance,
    ``math.fsum((X[i][k] - X[j][k]) ** 2)`` accumulated in ascending
    column order, is at most ``eps ** 2``; each point is its own neighbor.
    A point whose neighbor count is at least ``min_samples`` is a core
    point. Core points are scanned in ascending index order, and each
    newly discovered core starts a cluster: its not-yet-clustered core
    neighbors are traversed FIFO, enqueued in ascending neighbor index
    order (each core point is enqueued at most once), and every core point
    reached joins the cluster. Cluster numbers increase from zero in
    discovery order. Afterwards the non-core points are processed in
    ascending index order and assigned the smallest cluster label among
    their core neighbors, or ``-1`` when they have none. ``labels_`` is
    ``None`` initially and after a failed fit; a successful fit stores an
    integer list in input order. Neither ``fit`` nor ``fit_predict``
    modifies the input. No randomness is used.
    """

    def __init__(self, eps=0.5, min_samples=5):
        if type(eps) not in (int, float):
            raise ValueError(
                "eps must be a finite non-boolean int or float greater than 0"
            )
        try:
            eps_finite = math.isfinite(eps)
        except OverflowError:
            # math.isfinite raises OverflowError on exact ints too large to
            # convert to float; treat them like any other invalid eps.
            raise ValueError(
                "eps must be a finite non-boolean int or float greater than 0"
            )
        if not eps_finite or eps <= 0:
            raise ValueError(
                "eps must be a finite non-boolean int or float greater than 0"
            )
        if type(min_samples) is not int or min_samples <= 0:
            raise ValueError("min_samples must be a positive integer")

        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None

    @staticmethod
    def _squared_distance(row_a, row_b):
        """Squared Euclidean distance with terms accumulated in column
        order via ``math.fsum``; overflow or a non-finite result raises
        FloatingPointError."""
        terms = []
        for k in range(len(row_a)):
            try:
                diff = row_a[k] - row_b[k]
            except (OverflowError, ValueError) as exc:
                raise FloatingPointError(
                    "non-finite value encountered during distance computation"
                ) from exc
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

    def fit(self, X):
        self.labels_ = None

        _check_exact_matrix(X)
        n = len(X)

        try:
            eps_squared = self.eps ** 2
        except (OverflowError, ValueError) as exc:
            raise FloatingPointError(
                "non-finite value encountered during fit"
            ) from exc
        if isinstance(eps_squared, float) and not math.isfinite(eps_squared):
            raise FloatingPointError(
                "non-finite value encountered during fit"
            )

        # Neighbor lists, each in strictly ascending index order; a point is
        # always its own neighbor.
        neighbors = []
        for i in range(n):
            row_neighbors = []
            for j in range(n):
                if self._squared_distance(X[i], X[j]) <= eps_squared:
                    row_neighbors.append(j)
            neighbors.append(row_neighbors)

        core = [
            len(row_neighbors) >= self.min_samples
            for row_neighbors in neighbors
        ]

        labels = [-1] * n
        next_label = 0
        # Core points that have already been reached by a cluster.
        clustered_core = [False] * n

        for i in range(n):
            if not core[i] or clustered_core[i]:
                continue
            label = next_label
            next_label += 1
            labels[i] = label
            clustered_core[i] = True
            queue = deque()
            # Enqueue this core's not-yet-clustered core neighbors; the
            # neighbor list is already ascending, so enqueue order is
            # ascending and every core point is enqueued at most once.
            for j in neighbors[i]:
                if core[j] and not clustered_core[j]:
                    clustered_core[j] = True
                    labels[j] = label
                    queue.append(j)
            while queue:
                point = queue.popleft()
                for j in neighbors[point]:
                    if core[j] and not clustered_core[j]:
                        clustered_core[j] = True
                        labels[j] = label
                        queue.append(j)

        # Border points take the smallest cluster label among their core
        # neighbors; points with none remain noise (-1). Core labels never
        # change here.
        for i in range(n):
            if core[i]:
                continue
            border_label = -1
            for j in neighbors[i]:
                if core[j]:
                    candidate = labels[j]
                    if border_label == -1 or candidate < border_label:
                        border_label = candidate
            labels[i] = border_label

        self.labels_ = labels
        return self

    def fit_predict(self, X):
        self.fit(X)
        return list(self.labels_)


class AgglomerativeClustering:
    """Deterministic agglomerative (bottom-up) clustering with selectable
    single, complete, or average linkage.

    Initially every input index is its own cluster, represented by a
    tuple of member indices in strictly ascending order. The pairwise
    distance is ``sqrt(math.fsum((X[i][k] - X[j][k]) ** 2))`` with the
    squared terms accumulated in ascending column order. The distance
    ``D(A, B)`` between two clusters enumerates every cross-cluster pair
    with members of ``A`` on the outer loop and members of ``B`` on the
    inner loop: single linkage takes the smallest pairwise distance,
    complete linkage the largest, and average linkage their
    ``math.fsum`` total divided by the number of pairs.

    Each round the surviving clusters are sorted by their member tuple,
    and the pair with the lexicographically smallest
    ``(distance, A, B)`` tuple -- distance first, then the two member
    tuples -- is merged; the merged members stay in ascending order.
    Leaves have IDs ``0`` through ``n - 1``; the cluster created by
    merge ``r`` (counted from zero) gets ID ``n + r``. Each merge
    appends ``[id(A), id(B)]`` to ``children_`` in the order fixed by
    the winning key, and the merge distance as a ``float`` to
    ``distances_``. Merging repeats until ``n_clusters`` clusters
    remain, so both lists have length ``n - n_clusters`` and are empty
    when no merge occurs. The final clusters are numbered from zero in
    ascending order of their smallest member index, and ``labels_`` is
    an integer list in input order. The input is never modified, and no
    randomness is used.

    ``fit`` clears ``labels_``, ``children_``, and ``distances_``
    before validation, leaves all three ``None`` if fitting fails, and
    returns ``self``; ``fit_predict`` calls ``fit`` and returns a fresh
    copy of ``labels_``. Any subtraction, squaring, ``math.fsum``,
    square root, or division that raises
    ``OverflowError``/``ValueError`` or yields a non-finite value raises
    ``FloatingPointError``.
    """

    def __init__(self, n_clusters=2, linkage="average"):
        if type(n_clusters) is not int or n_clusters <= 0:
            raise ValueError("n_clusters must be a positive integer")
        if type(linkage) is not str or linkage not in (
            "single",
            "complete",
            "average",
        ):
            raise ValueError(
                'linkage must be "single", "complete", or "average"'
            )
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.labels_ = None
        self.children_ = None
        self.distances_ = None

    @staticmethod
    def _distance(row_a, row_b):
        """Euclidean distance; overflow or a non-finite result raises
        FloatingPointError."""
        terms = []
        for k in range(len(row_a)):
            try:
                diff = row_a[k] - row_b[k]
            except (OverflowError, ValueError) as exc:
                raise FloatingPointError(
                    "non-finite value encountered during distance computation"
                ) from exc
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
        try:
            result = math.sqrt(total)
        except (OverflowError, ValueError) as exc:
            raise FloatingPointError(
                "non-finite value encountered during distance computation"
            ) from exc
        if not math.isfinite(result):
            raise FloatingPointError(
                "non-finite value encountered during distance computation"
            )
        return result

    def _cluster_distance(self, members_a, members_b, X):
        """Linkage distance over all cross-cluster pairwise distances,
        enumerated with A members on the outer loop and B members on the
        inner loop. Single linkage takes the minimum, complete linkage
        the maximum, and average linkage the ``math.fsum`` mean; any
        arithmetic failure or non-finite result raises
        FloatingPointError."""
        if self.linkage == "average":
            terms = []
            for i in members_a:
                for j in members_b:
                    terms.append(self._distance(X[i], X[j]))
            try:
                total = math.fsum(terms)
            except (OverflowError, ValueError) as exc:
                raise FloatingPointError(
                    "non-finite value encountered during linkage computation"
                ) from exc
            if not math.isfinite(total):
                raise FloatingPointError(
                    "non-finite value encountered during linkage computation"
                )
            try:
                result = total / len(terms)
            except (OverflowError, ValueError, ZeroDivisionError) as exc:
                raise FloatingPointError(
                    "non-finite value encountered during linkage computation"
                ) from exc
            if not math.isfinite(result):
                raise FloatingPointError(
                    "non-finite value encountered during linkage computation"
                )
            return result

        best = None
        for i in members_a:
            for j in members_b:
                distance = self._distance(X[i], X[j])
                if best is None:
                    best = distance
                elif self.linkage == "single":
                    if distance < best:
                        best = distance
                else:  # complete
                    if distance > best:
                        best = distance
        return best

    def fit(self, X):
        # Clear before validation so a failed fit always leaves the
        # fitted attributes as None.
        self.labels_ = None
        self.children_ = None
        self.distances_ = None
        _check_exact_matrix(X)
        n = len(X)
        if n < self.n_clusters:
            raise ValueError(
                "n_clusters must not exceed the number of samples"
            )

        clusters = [(i,) for i in range(n)]
        # Current node ID of each cluster: leaves keep 0..n-1 and the
        # r-th merge creates node n+r.
        cluster_ids = list(range(n))
        children = []
        distances = []
        while len(clusters) > self.n_clusters:
            # Sort clusters by their ascending member tuples before each
            # round so distance ties resolve lexicographically.
            order = sorted(range(len(clusters)), key=lambda idx: clusters[idx])
            clusters = [clusters[idx] for idx in order]
            cluster_ids = [cluster_ids[idx] for idx in order]
            best_key = None
            best_pair = None
            for a_index in range(len(clusters)):
                for b_index in range(a_index + 1, len(clusters)):
                    members_a = clusters[a_index]
                    members_b = clusters[b_index]
                    distance = self._cluster_distance(
                        members_a, members_b, X
                    )
                    key = (distance, members_a, members_b)
                    if best_key is None or key < best_key:
                        best_key = key
                        best_pair = (a_index, b_index)
            a_index, b_index = best_pair
            distance = float(best_key[0])
            merged = tuple(
                sorted(clusters[a_index] + clusters[b_index])
            )
            children.append(
                [cluster_ids[a_index], cluster_ids[b_index]]
            )
            distances.append(distance)
            merged_id = n + len(children) - 1
            del clusters[b_index]
            del cluster_ids[b_index]
            del clusters[a_index]
            del cluster_ids[a_index]
            clusters.append(merged)
            cluster_ids.append(merged_id)

        # Number final clusters from zero by ascending smallest member.
        clusters.sort()
        labels = [0] * n
        for label, members in enumerate(clusters):
            for index in members:
                labels[index] = label
        self.labels_ = labels
        self.children_ = children
        self.distances_ = distances
        return self

    def fit_predict(self, X):
        self.fit(X)
        return list(self.labels_)


class GaussianMixture:
    """Deterministic one-dimensional Gaussian mixture model fitted by the
    expectation-maximization algorithm.

    The ``n_components`` mixture weights, means, and variances start from
    weights ``1 / n_components``, means equal to the ``n_components``
    smallest training values in ascending order, and variances ``1.0``.
    Each round computes, for every sample ``x_i`` and component ``j``, the
    log responsibility

        ell_ij = log(w_j) - (log(2*pi*v_j) + (x_i - m_j)**2 / v_j) / 2

    and then the softmax responsibilities

        r_ij = exp(ell_ij - a_i) / sum_h exp(ell_ih - a_i),

    where ``a_i = max_j ell_ij``; ties for the largest log responsibility
    go to the smallest component index. The component parameters are
    updated synchronously from

        N_j = sum_i r_ij,  w_j = N_j / n,
        m_j = sum_i r_ij * x_i / N_j,
        v_j = max(sum_i r_ij * (x_i - m_j)**2 / N_j, 1e-12),

    with all sums accumulated in ascending index order via
    ``math.fsum``. Rounds stop when the largest absolute change across the
    three parameter groups is at most ``1e-6``, after at most 100 rounds.
    An empty component, an arithmetic ``OverflowError``/``ValueError``, or
    a non-finite result raises FloatingPointError. Neither ``fit`` nor
    ``predict`` modifies its input, and the same parameters and inputs
    always give the same result.

    After a successful fit, ``predict`` returns a tuple of the index of
    the component with the largest responsibility for each input value
    (ties go to the smallest index) and the responsibility matrix with
    one row per input value and one column per component in ascending
    component order.

    After a successful fit, ``score_samples`` returns, for each input
    value in input order, the log of the mixture density

        s = a + log(sum_j exp(l_j - a)),  a = max_j l_j,

    with ``l_j`` the per-component log responsibility and the sum taken
    over components in ascending index order via ``math.fsum``;
    ``score`` returns the ``math.fsum`` mean of those values. Any
    arithmetic failure or non-finite intermediate value raises
    FloatingPointError; an exact zero is reported as positive ``0.0``.

    After a successful fit, ``aic`` and ``bic`` return the information
    criteria ``2*p - 2*L`` and ``log(n)*p - 2*L`` respectively, where
    ``s`` is the list returned by ``score_samples`` in input order,
    ``L = fsum(s)``, ``n`` is the number of samples, and
    ``p = 3*n_components - 1`` free parameters. Any arithmetic failure
    or non-finite intermediate value raises FloatingPointError; an exact
    zero is reported as positive ``0.0``.
    """

    def __init__(self, n_components=2):
        if type(n_components) is not int or n_components <= 0:
            raise ValueError("n_components must be a positive integer")
        self.n_components = n_components
        self.weights_ = None
        self.means_ = None
        self.variances_ = None

    @staticmethod
    def _check_1d_vector(X):
        """Validate a non-empty vector whose elements have type exactly
        ``int`` or ``float`` (booleans and subclasses rejected), with
        finite values; an overflow raised while checking finiteness of a
        huge integer is reported as ValueError too."""
        if not isinstance(X, list) or len(X) == 0:
            raise ValueError("X must be a non-empty list")
        for value in X:
            if type(value) not in (int, float):
                raise ValueError(
                    "X must contain only finite int or float elements"
                )
            try:
                finite = math.isfinite(value)
            except OverflowError as exc:
                raise ValueError(
                    "X must contain only finite int or float elements"
                ) from exc
            if not finite:
                raise ValueError(
                    "X must contain only finite int or float elements"
                )

    @staticmethod
    def _log_responsibilities(X, weights, means, variances):
        """Return the n-by-k matrix of unnormalized log responsibilities;
        any arithmetic failure or non-finite result raises
        FloatingPointError."""
        k = len(weights)
        log_ell = []
        for x in X:
            row = []
            for j in range(k):
                try:
                    diff = x - means[j]
                    square = diff * diff
                    ell = (
                        math.log(weights[j])
                        - (
                            math.log(2.0 * math.pi * variances[j])
                            + square / variances[j]
                        )
                        / 2.0
                    )
                except (OverflowError, ValueError) as exc:
                    raise FloatingPointError(
                        "non-finite value encountered during fitting"
                    ) from exc
                if not math.isfinite(ell):
                    raise FloatingPointError(
                        "non-finite value encountered during fitting"
                    )
                row.append(ell)
            log_ell.append(row)
        return log_ell

    @classmethod
    def _responsibilities(cls, X, weights, means, variances):
        """Return the softmax responsibility matrix for ``X``; any
        arithmetic failure or non-finite result raises
        FloatingPointError."""
        log_ell = cls._log_responsibilities(
            X, weights, means, variances
        )
        responsibilities = []
        for row in log_ell:
            largest = row[0]
            for ell in row[1:]:
                if ell > largest:
                    largest = ell
            try:
                exps = [math.exp(ell - largest) for ell in row]
                total = math.fsum(exps)
            except (OverflowError, ValueError) as exc:
                raise FloatingPointError(
                    "non-finite value encountered during fitting"
                ) from exc
            if not math.isfinite(total) or total <= 0.0:
                raise FloatingPointError(
                    "non-finite value encountered during fitting"
                )
            scaled = []
            for exponent in exps:
                value = exponent / total
                if not math.isfinite(value):
                    raise FloatingPointError(
                        "non-finite value encountered during fitting"
                    )
                scaled.append(value)
            responsibilities.append(scaled)
        return responsibilities

    def fit(self, X):
        # Reset first so a failed fit leaves the model unfitted.
        self.weights_ = None
        self.means_ = None
        self.variances_ = None

        self._check_1d_vector(X)
        n = len(X)
        k = self.n_components
        if n < k:
            raise ValueError(
                "n_components must not exceed the number of samples"
            )

        weights = [1.0 / k for _ in range(k)]
        means = sorted(X)[:k]
        variances = [1.0 for _ in range(k)]

        for _ in range(100):
            responsibilities = self._responsibilities(
                X, weights, means, variances
            )

            counts = []
            new_weights = []
            new_means = []
            for j in range(k):
                try:
                    count = math.fsum(
                        responsibilities[i][j] for i in range(n)
                    )
                except (OverflowError, ValueError) as exc:
                    raise FloatingPointError(
                        "non-finite value encountered during fitting"
                    ) from exc
                if not math.isfinite(count) or count <= 0.0:
                    raise FloatingPointError(
                        "non-finite value encountered during fitting"
                    )
                counts.append(count)

                try:
                    weight = count / n
                    weighted_total = math.fsum(
                        responsibilities[i][j] * X[i] for i in range(n)
                    )
                    mean = weighted_total / count
                except (OverflowError, ValueError) as exc:
                    raise FloatingPointError(
                        "non-finite value encountered during fitting"
                    ) from exc
                if not (
                    math.isfinite(weight) and math.isfinite(mean)
                ):
                    raise FloatingPointError(
                        "non-finite value encountered during fitting"
                    )
                new_weights.append(weight)
                new_means.append(mean)

            new_variances = []
            for j in range(k):
                mean = new_means[j]
                try:
                    squared_total = math.fsum(
                        responsibilities[i][j] * (X[i] - mean) ** 2
                        for i in range(n)
                    )
                    variance = max(squared_total / counts[j], 1e-12)
                except (OverflowError, ValueError) as exc:
                    raise FloatingPointError(
                        "non-finite value encountered during fitting"
                    ) from exc
                if not math.isfinite(variance):
                    raise FloatingPointError(
                        "non-finite value encountered during fitting"
                    )
                new_variances.append(variance)

            max_change = 0.0
            for j in range(k):
                for old, new in (
                    (weights[j], new_weights[j]),
                    (means[j], new_means[j]),
                    (variances[j], new_variances[j]),
                ):
                    try:
                        change = abs(new - old)
                    except (OverflowError, ValueError) as exc:
                        raise FloatingPointError(
                            "non-finite value encountered during fitting"
                        ) from exc
                    if not math.isfinite(change):
                        raise FloatingPointError(
                            "non-finite value encountered during fitting"
                        )
                    if change > max_change:
                        max_change = change

            weights = new_weights
            means = new_means
            variances = new_variances
            if max_change <= 1e-6:
                break

        self.weights_ = weights
        self.means_ = means
        self.variances_ = variances
        return self

    def predict(self, X):
        if self.weights_ is None:
            raise ValueError("model must be fitted before predict is called")
        self._check_1d_vector(X)

        responsibilities = self._responsibilities(
            X, self.weights_, self.means_, self.variances_
        )
        labels = []
        for row in responsibilities:
            best_index = 0
            best_value = row[0]
            for j in range(1, self.n_components):
                if row[j] > best_value:
                    best_value = row[j]
                    best_index = j
            labels.append(best_index)
        return labels, responsibilities

    def score_samples(self, X) -> list[float]:
        """Return the log mixture density of each input value in input
        order.

        For each value ``x`` and component ``j`` the log responsibility
        ``l_j`` is computed in ascending component order; with
        ``a = max_j l_j`` the result is ``s = a + log(fsum(exp(l_j -
        a)))``, all via :mod:`math`. An arithmetic
        ``OverflowError``/``ValueError``/``ZeroDivisionError`` or any
        non-finite intermediate value, result, or fitted mean raises
        FloatingPointError. The input is not modified.
        """
        if self.weights_ is None:
            raise ValueError(
                "model must be fitted before score_samples is called"
            )
        self._check_1d_vector(X)

        for mean in self.means_:
            if not math.isfinite(mean):
                raise FloatingPointError(
                    "non-finite value encountered while scoring"
                )

        scores = []
        k = self.n_components
        for x in X:
            log_ells = []
            largest = None
            for j in range(k):
                try:
                    ell = (
                        math.log(self.weights_[j])
                        - (
                            math.log(
                                2.0 * math.pi * self.variances_[j]
                            )
                            + (x - self.means_[j]) ** 2
                            / self.variances_[j]
                        )
                        / 2.0
                    )
                except (
                    OverflowError,
                    ValueError,
                    ZeroDivisionError,
                ) as exc:
                    raise FloatingPointError(
                        "non-finite value encountered while scoring"
                    ) from exc
                if not math.isfinite(ell):
                    raise FloatingPointError(
                        "non-finite value encountered while scoring"
                    )
                log_ells.append(ell)
                if largest is None or ell > largest:
                    largest = ell

            try:
                shifted = [math.exp(ell - largest) for ell in log_ells]
                total = math.fsum(shifted)
                score = largest + math.log(total)
            except (
                OverflowError,
                ValueError,
                ZeroDivisionError,
            ) as exc:
                raise FloatingPointError(
                    "non-finite value encountered while scoring"
                ) from exc
            if not math.isfinite(score):
                raise FloatingPointError(
                    "non-finite value encountered while scoring"
                )
            if score == 0.0:
                score = 0.0
            scores.append(score)
        return scores

    def score(self, X) -> float:
        """Return the ``math.fsum`` mean of :meth:`score_samples`."""
        if self.weights_ is None:
            raise ValueError(
                "model must be fitted before score is called"
            )
        samples = self.score_samples(X)
        try:
            result = math.fsum(samples) / len(X)
        except (
            OverflowError,
            ValueError,
            ZeroDivisionError,
        ) as exc:
            raise FloatingPointError(
                "non-finite value encountered while scoring"
            ) from exc
        if not math.isfinite(result):
            raise FloatingPointError(
                "non-finite value encountered while scoring"
            )
        if result == 0.0:
            result = 0.0
        return result

    def _information_criterion(self, X, log_penalty):
        """Shared AIC/BIC machinery; ``log_penalty`` is the penalty per
        free parameter (``2.0`` for AIC, ``math.log`` applied to the
        sample count for BIC)."""
        samples = self.score_samples(X)
        n = len(X)
        p = 3 * self.n_components - 1
        try:
            total = math.fsum(samples)
            penalty = (
                log_penalty(n) if callable(log_penalty) else log_penalty
            )
        except (
            OverflowError,
            ValueError,
            ZeroDivisionError,
        ) as exc:
            raise FloatingPointError(
                "non-finite value encountered while scoring"
            ) from exc
        if not math.isfinite(total) or not math.isfinite(penalty):
            raise FloatingPointError(
                "non-finite value encountered while scoring"
            )
        try:
            result = penalty * p - 2.0 * total
        except (
            OverflowError,
            ValueError,
            ZeroDivisionError,
        ) as exc:
            raise FloatingPointError(
                "non-finite value encountered while scoring"
            ) from exc
        if not math.isfinite(result):
            raise FloatingPointError(
                "non-finite value encountered while scoring"
            )
        if result == 0.0:
            result = 0.0
        return result

    def aic(self, X) -> float:
        """Return the Akaike information criterion ``2*p - 2*L``.

        ``L`` is the ``math.fsum`` of the values returned by
        :meth:`score_samples` and ``p = 3*n_components - 1`` free
        parameters. The model must be fitted; validation of ``X`` is
        delegated to :meth:`score_samples`. Any arithmetic failure or
        non-finite intermediate value raises FloatingPointError; an exact
        zero is reported as positive ``0.0``.
        """
        if self.weights_ is None:
            raise ValueError(
                "model must be fitted before aic is called"
            )
        return self._information_criterion(X, 2.0)

    def bic(self, X) -> float:
        """Return the Bayesian information criterion ``log(n)*p - 2*L``.

        ``L`` is the ``math.fsum`` of the values returned by
        :meth:`score_samples`, ``n`` is the number of samples, and
        ``p = 3*n_components - 1`` free parameters. The model must be
        fitted; validation of ``X`` is delegated to
        :meth:`score_samples`. Any arithmetic failure or non-finite
        intermediate value raises FloatingPointError; an exact zero is
        reported as positive ``0.0``.
        """
        if self.weights_ is None:
            raise ValueError(
                "model must be fitted before bic is called"
            )
        return self._information_criterion(X, math.log)


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


def mean_squared_error(y_true, y_pred, sample_weight=None) -> float:
    """Return the (optionally weighted) mean of the squared differences.

    ``y_true`` and ``y_pred`` must be non-empty lists of equal length
    whose elements are finite values of type exactly ``int`` or
    ``float`` (booleans are rejected). ``sample_weight`` must be
    ``None`` or a list of the same length whose elements are finite
    non-negative values of type exactly ``int`` or ``float``. Any
    container, length, type, range, or finiteness violation (including
    ``OverflowError`` raised by ``math.isfinite``) raises ValueError.

    When ``sample_weight`` is ``None``, the original values are used in
    input order: ``d_i = y_true_i - y_pred_i``, ``q_i = d_i ** 2``, and
    the result is ``math.fsum(q) / n``; integer values are not narrowed
    to ``float`` first, so results for very large legal integers are
    unchanged. Overflow, invalid operations, and non-finite
    intermediate values raise FloatingPointError.

    When weights are supplied, the values and weights are converted to
    ``float`` in input order; a conversion failure or a non-finite
    converted value raises FloatingPointError. ``math.fsum`` computes
    the total weight ``W = sum(w_i)``; if that summation overflows or
    is invalid, is non-finite, or ``W`` is less than or equal to zero,
    a ValueError is raised. For each index, ``d_i = y_true_i -
    y_pred_i``, ``q_i = d_i ** 2``, and ``u_i = w_i * q_i``; in input
    order, ``math.fsum`` computes ``S = sum(u_i)`` and the result is
    ``S / W``. Overflow, invalid operations, or division by zero
    during the arithmetic and summation, and non-finite intermediate
    values or results, raise FloatingPointError.

    An exact zero result is normalized to positive ``0.0``. The return
    value is a float. The inputs are not modified. Deterministic: same
    inputs, same result.
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

    if not isinstance(sample_weight, list) or len(sample_weight) != n:
        raise ValueError(
            "sample_weight must be a list with the same length as y_true"
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
        w = [float(value) for value in sample_weight]
    except (OverflowError, ValueError) as exc:
        raise FloatingPointError(
            "non-finite value encountered during mean squared error"
        ) from exc
    for values in (t, p, w):
        for value in values:
            if not math.isfinite(value):
                raise FloatingPointError(
                    "non-finite value encountered during mean squared error"
                )

    try:
        total_weight = math.fsum(w)
    except (OverflowError, ValueError) as exc:
        raise ValueError("the total weight must be greater than 0") from exc
    if not math.isfinite(total_weight) or total_weight <= 0.0:
        raise ValueError("the total weight must be greater than 0")

    terms = []
    for i in range(n):
        try:
            d = t[i] - p[i]
            q = d ** 2
            u = w[i] * q
        except (OverflowError, ValueError) as exc:
            raise FloatingPointError(
                "non-finite value encountered during mean squared error"
            ) from exc
        if (
            not math.isfinite(d)
            or not math.isfinite(q)
            or not math.isfinite(u)
        ):
            raise FloatingPointError(
                "non-finite value encountered during mean squared error"
            )
        terms.append(u)

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
        result = total / total_weight
    except (OverflowError, ValueError, ZeroDivisionError) as exc:
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


def root_mean_squared_error(y_true, y_pred, sample_weight=None) -> float:
    """Return the weighted root mean squared error.

    ``y_true`` and ``y_pred`` must be non-empty lists of equal length
    whose elements are finite values of type exactly ``int`` or
    ``float`` (booleans are rejected). ``sample_weight`` must be
    ``None`` -- every sample then weighs ``1.0`` -- or a list of the
    same length whose elements are finite non-negative values of type
    exactly ``int`` or ``float``. Any container, length, type, range,
    or finiteness violation (including ``OverflowError`` raised by
    ``math.isfinite``) raises ValueError.

    After validation, the values and weights are converted to ``float``
    in input order. ``math.fsum`` computes the total weight
    ``W = sum(w_i)``; if that summation overflows or is invalid, is
    non-finite, or ``W`` is less than or equal to zero, a ValueError is
    raised. For each index, ``d_i = y_true_i - y_pred_i``,
    ``q_i = d_i ** 2``, and ``u_i = w_i * q_i``; in input order,
    ``math.fsum`` computes ``S = sum(u_i)``, and the result is
    ``math.sqrt(S / W)``. Overflow, invalid operations, or division by
    zero during the post-validation conversion, arithmetic, summation,
    or square root, and non-finite intermediate values or results,
    raise FloatingPointError. An exact zero result is normalized to
    positive ``0.0``.

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
            "non-finite value encountered during root mean squared error"
        ) from exc
    for values in (t, p, w):
        for value in values:
            if not math.isfinite(value):
                raise FloatingPointError(
                    "non-finite value encountered during root mean squared "
                    "error"
                )

    try:
        total_weight = math.fsum(w)
    except (OverflowError, ValueError) as exc:
        raise ValueError("the total weight must be greater than 0") from exc
    if not math.isfinite(total_weight) or total_weight <= 0.0:
        raise ValueError("the total weight must be greater than 0")

    terms = []
    for i in range(n):
        try:
            d = t[i] - p[i]
            q = d ** 2
            u = w[i] * q
        except (OverflowError, ValueError) as exc:
            raise FloatingPointError(
                "non-finite value encountered during root mean squared error"
            ) from exc
        if (
            not math.isfinite(d)
            or not math.isfinite(q)
            or not math.isfinite(u)
        ):
            raise FloatingPointError(
                "non-finite value encountered during root mean squared error"
            )
        terms.append(u)

    try:
        total = math.fsum(terms)
    except (OverflowError, ValueError) as exc:
        raise FloatingPointError(
            "non-finite value encountered during root mean squared error"
        ) from exc
    if not math.isfinite(total):
        raise FloatingPointError(
            "non-finite value encountered during root mean squared error"
        )
    try:
        mean = total / total_weight
    except (OverflowError, ValueError, ZeroDivisionError) as exc:
        raise FloatingPointError(
            "non-finite value encountered during root mean squared error"
        ) from exc
    if not math.isfinite(mean):
        raise FloatingPointError(
            "non-finite value encountered during root mean squared error"
        )
    try:
        result = math.sqrt(mean)
    except (OverflowError, ValueError) as exc:
        raise FloatingPointError(
            "non-finite value encountered during root mean squared error"
        ) from exc
    if not math.isfinite(result):
        raise FloatingPointError(
            "non-finite value encountered during root mean squared error"
        )
    if result == 0:
        result = 0.0
    return result


def mean_absolute_error(y_true, y_pred, sample_weight=None) -> float:
    """Return the weighted mean of the element-wise absolute differences.

    ``y_true`` and ``y_pred`` must be non-empty lists of equal length
    whose elements are finite values of type exactly ``int`` or
    ``float`` (booleans are rejected). ``sample_weight`` must be
    ``None`` -- every sample then weighs ``1.0`` -- or a list of the
    same length whose elements are finite non-negative values of type
    exactly ``int`` or ``float`` (booleans are rejected). Any
    container, length, type, range, or finiteness violation (including
    ``OverflowError`` raised by ``math.isfinite``) raises ValueError.

    After validation, the values and weights are converted to
    ``float`` in input order. ``math.fsum`` computes the total weight
    ``W = sum(w_i)``; if that summation overflows or is invalid, is
    non-finite, or ``W`` is less than or equal to zero, a ValueError is
    raised. For each index, ``d_i = abs(y_true_i - y_pred_i)`` and
    ``l_i = w_i * d_i``; in input order, ``math.fsum`` computes
    ``L = sum(l_i)`` and the result is ``L / W``. Overflow, invalid
    operations during the post-validation conversion, subtraction,
    absolute value, multiplication, the ``L`` summation, or the
    division, and non-finite intermediate values or results raise
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
            "non-finite value encountered during mean absolute error"
        ) from exc
    for values in (t, p, w):
        for value in values:
            if not math.isfinite(value):
                raise FloatingPointError(
                    "non-finite value encountered during mean absolute error"
                )

    try:
        total_weight = math.fsum(w)
    except (OverflowError, ValueError) as exc:
        raise ValueError("the total weight must be greater than 0") from exc
    if not math.isfinite(total_weight) or total_weight <= 0.0:
        raise ValueError("the total weight must be greater than 0")

    loss_terms = []
    for i in range(n):
        try:
            d = abs(t[i] - p[i])
            term = w[i] * d
        except (OverflowError, ValueError) as exc:
            raise FloatingPointError(
                "non-finite value encountered during mean absolute error"
            ) from exc
        if not math.isfinite(d) or not math.isfinite(term):
            raise FloatingPointError(
                "non-finite value encountered during mean absolute error"
            )
        loss_terms.append(term)

    try:
        total_loss = math.fsum(loss_terms)
    except (OverflowError, ValueError) as exc:
        raise FloatingPointError(
            "non-finite value encountered during mean absolute error"
        ) from exc
    if not math.isfinite(total_loss):
        raise FloatingPointError(
            "non-finite value encountered during mean absolute error"
        )
    try:
        result = total_loss / total_weight
    except (OverflowError, ValueError, ZeroDivisionError) as exc:
        raise FloatingPointError(
            "non-finite value encountered during mean absolute error"
        ) from exc
    if not math.isfinite(result):
        raise FloatingPointError(
            "non-finite value encountered during mean absolute error"
        )
    if result == 0:
        result = 0.0
    return result


def max_error(y_true, y_pred) -> float:
    """Return the maximum of the element-wise absolute differences.

    Both arguments must be non-empty lists of equal length whose elements
    are finite values of type exactly ``int`` or ``float`` (booleans and
    other numeric subclasses are rejected). The inputs are not modified.
    After validation both vectors are converted to ``float`` in input
    order; the conversion, subtraction, ``abs``, and comparison are all
    checked -- overflow, invalid operations, and non-finite intermediate
    values raise FloatingPointError. An exact zero result is normalized
    to ``0.0``. Deterministic: same inputs, same result.
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

    t = []
    p = []
    for i in range(n):
        try:
            t.append(float(y_true[i]))
            p.append(float(y_pred[i]))
        except (OverflowError, ValueError) as exc:
            raise FloatingPointError(
                "non-finite value encountered during max error"
            ) from exc

    best = None
    for i in range(n):
        try:
            r = abs(t[i] - p[i])
        except (OverflowError, ValueError) as exc:
            raise FloatingPointError(
                "non-finite value encountered during max error"
            ) from exc
        if not math.isfinite(r):
            raise FloatingPointError(
                "non-finite value encountered during max error"
            )
        if best is None:
            best = r
        else:
            try:
                larger = r > best
            except (OverflowError, ValueError) as exc:
                raise FloatingPointError(
                    "non-finite value encountered during max error"
                ) from exc
            if larger:
                best = r
    result = float(best)
    if result == 0:
        result = 0.0
    return result


def median_absolute_error(y_true, y_pred, sample_weight=None) -> float:
    """Return the weighted median of the element-wise absolute differences.

    ``y_true`` and ``y_pred`` must be non-empty lists of equal length
    whose elements are finite values of type exactly ``int`` or
    ``float`` (booleans are rejected). ``sample_weight`` must be
    ``None`` -- every sample then weighs ``1.0`` -- or a list of the
    same length whose elements are finite non-negative values of type
    exactly ``int`` or ``float`` (booleans are rejected). Any
    container, length, type, range, or finiteness violation (including
    ``OverflowError`` raised by ``math.isfinite``) raises ValueError.

    After validation, the values and weights are converted to
    ``float`` in input order. ``math.fsum`` computes the total weight
    ``W = sum(w_i)``; if that summation overflows or is invalid, is
    non-finite, or ``W`` is less than or equal to zero, a ValueError is
    raised. For each index, ``r_i = abs(y_true_i - y_pred_i)`` and the
    items are sorted by ``(r_i, i)`` ascending. Starting from an empty
    prefix, the weight prefix is recomputed with ``math.fsum`` after
    each included item (zero-weight items are retained); the result is
    the first ``r_i`` whose prefix is greater than or equal to
    ``W / 2``. A prefix exactly equal to ``W / 2`` still takes the
    current ``r_i`` -- adjacent values are never averaged. Overflow,
    invalid operations during the post-validation conversion,
    subtraction, absolute value, sorting, the division, or a prefix
    summation, and non-finite intermediate values raise
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
            "non-finite value encountered during median absolute error"
        ) from exc
    for values in (t, p, w):
        for value in values:
            if not math.isfinite(value):
                raise FloatingPointError(
                    "non-finite value encountered during median absolute error"
                )

    try:
        total_weight = math.fsum(w)
    except (OverflowError, ValueError) as exc:
        raise ValueError("the total weight must be greater than 0") from exc
    if not math.isfinite(total_weight) or total_weight <= 0.0:
        raise ValueError("the total weight must be greater than 0")

    items = []
    for i in range(n):
        try:
            r = abs(t[i] - p[i])
        except (OverflowError, ValueError) as exc:
            raise FloatingPointError(
                "non-finite value encountered during median absolute error"
            ) from exc
        if not math.isfinite(r):
            raise FloatingPointError(
                "non-finite value encountered during median absolute error"
            )
        items.append((r, i))

    try:
        items.sort()
    except (OverflowError, ValueError, TypeError) as exc:
        raise FloatingPointError(
            "non-finite value encountered during median absolute error"
        ) from exc

    try:
        half_weight = total_weight / 2.0
    except (OverflowError, ValueError) as exc:
        raise FloatingPointError(
            "non-finite value encountered during median absolute error"
        ) from exc
    if not math.isfinite(half_weight):
        raise FloatingPointError(
            "non-finite value encountered during median absolute error"
        )

    included = []
    result = None
    for r, i in items:
        included.append(w[i])
        try:
            prefix = math.fsum(included)
        except (OverflowError, ValueError) as exc:
            raise FloatingPointError(
                "non-finite value encountered during median absolute error"
            ) from exc
        if not math.isfinite(prefix):
            raise FloatingPointError(
                "non-finite value encountered during median absolute error"
            )
        if prefix >= half_weight:
            result = r
            break

    if result == 0:
        result = 0.0
    return result


def mean_squared_log_error(y_true, y_pred, sample_weight=None) -> float:
    """Return the weighted mean squared logarithmic error.

    ``y_true`` and ``y_pred`` must be non-empty lists of equal length
    whose elements are finite non-negative values of type exactly
    ``int`` or ``float`` (booleans are rejected). ``sample_weight`` must
    be ``None`` -- every sample then weighs ``1.0`` -- or a list of the
    same length whose elements follow the same rules. Any container,
    length, type, range, or finiteness violation (including
    ``OverflowError`` raised by ``math.isfinite``) raises ValueError.

    After validation, the values and weights are converted to ``float``
    in input order. ``math.fsum`` computes the total weight
    ``W = sum(w_i)``; if that summation overflows or is invalid, is
    non-finite, or ``W`` is less than or equal to zero, a ValueError is
    raised. For each index,
    ``e_i = log1p(y_pred_i) - log1p(y_true_i)`` and, in input order,
    ``math.fsum`` computes ``L = sum(w_i * e_i ** 2)``; the result is
    ``L / W``. Overflow, invalid operations during the post-validation
    conversion or arithmetic, and non-finite intermediate values or
    results raise FloatingPointError. An exact zero result is normalized
    to ``0.0``.

    The return value is a float. The inputs are not modified.
    Deterministic: same inputs, same result.
    """
    n = _check_metric_vectors(y_true, y_pred)
    for name, values in (("y_true", y_true), ("y_pred", y_pred)):
        for value in values:
            if type(value) not in (int, float):
                raise ValueError(
                    "%s must contain only finite non-negative non-boolean "
                    "numbers" % name
                )
            # math.isfinite raises OverflowError for ints too large to
            # convert to float; such values fail the finite requirement.
            try:
                finite = math.isfinite(value)
            except OverflowError as exc:
                raise ValueError(
                    "%s must contain only finite non-negative non-boolean "
                    "numbers" % name
                ) from exc
            if not finite or value < 0:
                raise ValueError(
                    "%s must contain only finite non-negative non-boolean "
                    "numbers" % name
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
            "non-finite value encountered during mean squared log error"
        ) from exc
    for values in (t, p, w):
        for value in values:
            if not math.isfinite(value):
                raise FloatingPointError(
                    "non-finite value encountered during mean squared log "
                    "error"
                )

    try:
        total_weight = math.fsum(w)
    except (OverflowError, ValueError) as exc:
        raise ValueError("the total weight must be greater than 0") from exc
    if not math.isfinite(total_weight) or total_weight <= 0.0:
        raise ValueError("the total weight must be greater than 0")

    error_terms = []
    for i in range(n):
        try:
            e = math.log1p(p[i]) - math.log1p(t[i])
            term = w[i] * e ** 2
        except (OverflowError, ValueError) as exc:
            raise FloatingPointError(
                "non-finite value encountered during mean squared log error"
            ) from exc
        if not math.isfinite(e) or not math.isfinite(term):
            raise FloatingPointError(
                "non-finite value encountered during mean squared log error"
            )
        error_terms.append(term)

    try:
        total_error = math.fsum(error_terms)
    except (OverflowError, ValueError) as exc:
        raise FloatingPointError(
            "non-finite value encountered during mean squared log error"
        ) from exc
    if not math.isfinite(total_error):
        raise FloatingPointError(
            "non-finite value encountered during mean squared log error"
        )
    try:
        result = total_error / total_weight
    except (OverflowError, ValueError, ZeroDivisionError) as exc:
        raise FloatingPointError(
            "non-finite value encountered during mean squared log error"
        ) from exc
    if not math.isfinite(result):
        raise FloatingPointError(
            "non-finite value encountered during mean squared log error"
        )
    if result == 0:
        result = 0.0
    return result


def root_mean_squared_log_error(y_true, y_pred, sample_weight=None) -> float:
    """Return the weighted root mean squared logarithmic error.

    ``y_true``, ``y_pred``, and ``sample_weight`` follow exactly the
    validation rules of :func:`mean_squared_log_error`: ``y_true`` and
    ``y_pred`` must be non-empty lists of equal length whose elements
    are finite non-negative values of type exactly ``int`` or ``float``
    (booleans are rejected), and ``sample_weight`` must be ``None`` --
    every sample then weighs ``1.0`` -- or a list of the same length
    whose elements follow the same rules, with a total weight greater
    than zero. Any container, length, type, range, or finiteness
    violation raises ValueError; every exception raised by the
    underlying ``mean_squared_log_error`` call propagates unchanged.

    The implementation obtains the mean squared logarithmic error
    ``m`` from ``mean_squared_log_error(y_true, y_pred, sample_weight)``
    and returns ``math.sqrt(m)``. ``m`` must be a finite non-negative
    float; if it is not, if ``math.sqrt`` raises ``OverflowError`` or
    ``ValueError``, or if the square root is non-finite, a
    FloatingPointError is raised. An exact zero ``m`` yields positive
    ``0.0``.

    The return value is a float. The inputs are not modified.
    Deterministic: same inputs, same result.
    """
    m = mean_squared_log_error(y_true, y_pred, sample_weight)
    if type(m) is not float or not math.isfinite(m) or m < 0:
        raise FloatingPointError(
            "non-finite value encountered during root mean squared log "
            "error"
        )
    try:
        result = math.sqrt(m)
    except (OverflowError, ValueError) as exc:
        raise FloatingPointError(
            "non-finite value encountered during root mean squared log "
            "error"
        ) from exc
    if not math.isfinite(result):
        raise FloatingPointError(
            "non-finite value encountered during root mean squared log "
            "error"
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


def _evs_float(value):
    """Convert a validated number to float; overflow, invalid operations,
    and non-finite results raise FloatingPointError."""
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise FloatingPointError(
            "non-finite value encountered during explained variance "
            "score computation"
        ) from exc
    if not math.isfinite(result):
        raise FloatingPointError(
            "non-finite value encountered during explained variance "
            "score computation"
        )
    return result


def explained_variance_score(y_true, y_pred, sample_weight=None,
                             force_finite=True) -> float:
    """Return the explained variance regression score.

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
    in input (sample) order, ``math.fsum`` computes the total weight
    ``W = sum(w_i)``, the weighted mean
    ``m = sum(w_i * y_true_i) / W``, the per-sample residuals
    ``r_i = y_true_i - y_pred_i``, their weighted mean
    ``q = sum(w_i * r_i) / W``, the residual variance
    ``N = sum(w_i * (r_i - q) ** 2)``, and the variance of the targets
    ``D = sum(w_i * (y_true_i - m) ** 2)``. When ``D`` is non-zero the
    result is ``1.0 - N / D``. When ``D`` and ``N`` are both zero the
    result is ``1.0``. When only ``D`` is zero the result is ``0.0`` if
    ``force_finite`` is true and ``float("-inf")`` otherwise. Overflow,
    invalid operations during the post-validation conversion,
    subtraction, multiplication, squaring, ``math.fsum`` or division, and
    non-finite intermediate values or results raise FloatingPointError
    (the negative-infinity result above excepted). An exact zero result
    is normalized to ``0.0``.

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
        weights = [_evs_float(value) for value in sample_weight]

    if type(force_finite) is not bool:
        raise ValueError("force_finite must be a boolean")

    try:
        total_weight = math.fsum(weights)
    except (OverflowError, ValueError) as exc:
        raise FloatingPointError(
            "non-finite value encountered during explained variance "
            "score computation"
        ) from exc
    if not math.isfinite(total_weight):
        raise FloatingPointError(
            "non-finite value encountered during explained variance "
            "score computation"
        )
    if total_weight <= 0.0:
        raise ValueError("the total weight must be greater than 0")

    yt = [_evs_float(value) for value in y_true]
    yp = [_evs_float(value) for value in y_pred]

    mean_terms = []
    residual_terms = []
    for i in range(n):
        try:
            mean_term = weights[i] * yt[i]
        except (OverflowError, ValueError) as exc:
            raise FloatingPointError(
                "non-finite value encountered during explained variance "
                "score computation"
            ) from exc
        if not math.isfinite(mean_term):
            raise FloatingPointError(
                "non-finite value encountered during explained variance "
                "score computation"
            )
        mean_terms.append(mean_term)
        try:
            residual = yt[i] - yp[i]
        except (OverflowError, ValueError) as exc:
            raise FloatingPointError(
                "non-finite value encountered during explained variance "
                "score computation"
            ) from exc
        if not math.isfinite(residual):
            raise FloatingPointError(
                "non-finite value encountered during explained variance "
                "score computation"
            )
        residual_terms.append(residual)
    try:
        mean = math.fsum(mean_terms) / total_weight
    except (OverflowError, ValueError, ZeroDivisionError) as exc:
        raise FloatingPointError(
            "non-finite value encountered during explained variance "
            "score computation"
        ) from exc
    if not math.isfinite(mean):
        raise FloatingPointError(
            "non-finite value encountered during explained variance "
            "score computation"
        )

    weighted_residual_terms = []
    for i in range(n):
        try:
            term = weights[i] * residual_terms[i]
        except (OverflowError, ValueError) as exc:
            raise FloatingPointError(
                "non-finite value encountered during explained variance "
                "score computation"
            ) from exc
        if not math.isfinite(term):
            raise FloatingPointError(
                "non-finite value encountered during explained variance "
                "score computation"
            )
        weighted_residual_terms.append(term)
    try:
        residual_mean = math.fsum(weighted_residual_terms) / total_weight
    except (OverflowError, ValueError, ZeroDivisionError) as exc:
        raise FloatingPointError(
            "non-finite value encountered during explained variance "
            "score computation"
        ) from exc
    if not math.isfinite(residual_mean):
        raise FloatingPointError(
            "non-finite value encountered during explained variance "
            "score computation"
        )

    numerator_terms = []
    denominator_terms = []
    for i in range(n):
        try:
            residual_offset = residual_terms[i] - residual_mean
            target_offset = yt[i] - mean
            numerator_term = weights[i] * residual_offset ** 2
            denominator_term = weights[i] * target_offset ** 2
        except (OverflowError, ValueError) as exc:
            raise FloatingPointError(
                "non-finite value encountered during explained variance "
                "score computation"
            ) from exc
        if not math.isfinite(numerator_term) or not math.isfinite(
                denominator_term):
            raise FloatingPointError(
                "non-finite value encountered during explained variance "
                "score computation"
            )
        numerator_terms.append(numerator_term)
        denominator_terms.append(denominator_term)
    try:
        numerator = math.fsum(numerator_terms)
        denominator = math.fsum(denominator_terms)
    except (OverflowError, ValueError) as exc:
        raise FloatingPointError(
            "non-finite value encountered during explained variance "
            "score computation"
        ) from exc
    if not math.isfinite(numerator) or not math.isfinite(denominator):
        raise FloatingPointError(
            "non-finite value encountered during explained variance "
            "score computation"
        )

    if denominator != 0.0:
        try:
            result = 1.0 - numerator / denominator
        except (OverflowError, ValueError, ZeroDivisionError) as exc:
            raise FloatingPointError(
                "non-finite value encountered during explained variance "
                "score computation"
            ) from exc
        if not math.isfinite(result):
            raise FloatingPointError(
                "non-finite value encountered during explained variance "
                "score computation"
            )
        if result == 0:
            result = 0.0
        return result
    if numerator == 0.0:
        return 1.0
    if force_finite:
        return 0.0
    return float("-inf")


def _ccc_float(value):
    """Convert a validated number to float; overflow, invalid operations,
    and non-finite results raise FloatingPointError."""
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise FloatingPointError(
            "non-finite value encountered during concordance correlation "
            "coefficient computation"
        ) from exc
    if not math.isfinite(result):
        raise FloatingPointError(
            "non-finite value encountered during concordance correlation "
            "coefficient computation"
        )
    return result


def concordance_correlation_coefficient(
        y_true, y_pred, sample_weight=None) -> float:
    """Return Lin's weighted concordance correlation coefficient.

    ``y_true`` and ``y_pred`` must be non-empty lists of equal length
    whose elements are finite values of type exactly ``int`` or
    ``float`` (booleans are rejected). ``sample_weight`` must be
    ``None`` -- every sample then weighs ``1.0`` -- or a list of the
    same length whose elements are finite non-negative values of type
    exactly ``int`` or ``float`` (booleans are rejected). The total
    weight must be greater than zero. Any violation (including
    ``OverflowError`` raised by ``math.isfinite``) raises ValueError.

    After validation, the values and weights are converted to
    ``float``. In input order, with ``t`` the true value, ``p`` the
    prediction, and ``w`` the weight, ``math.fsum`` computes the total
    weight ``W = sum(w)``, the weighted means ``mt = sum(w * t) / W``
    and ``mp = sum(w * p) / W``, the weighted variances
    ``Vt = sum(w * (t - mt) ** 2) / W`` and
    ``Vp = sum(w * (p - mp) ** 2) / W``, and the weighted covariance
    ``C = sum(w * (t - mt) * (p - mp)) / W``. The result is
    ``2 * C / (Vt + Vp + (mt - mp) ** 2)``; when the denominator is
    exactly zero the result is ``1.0``. Overflow, invalid operations
    during the post-validation conversion, subtraction, multiplication,
    squaring, ``math.fsum`` or division, and non-finite intermediate
    values or results raise FloatingPointError. An exact zero result is
    normalized to positive ``0.0``; the result is not clipped.

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
        weights = [_ccc_float(value) for value in sample_weight]

    try:
        total_weight = math.fsum(weights)
    except (OverflowError, ValueError) as exc:
        raise ValueError("the total weight must be greater than 0") from exc
    if not math.isfinite(total_weight) or total_weight <= 0.0:
        raise ValueError("the total weight must be greater than 0")

    t = [_ccc_float(value) for value in y_true]
    p = [_ccc_float(value) for value in y_pred]

    true_mean_terms = []
    pred_mean_terms = []
    for i in range(n):
        try:
            true_term = weights[i] * t[i]
            pred_term = weights[i] * p[i]
        except (OverflowError, ValueError) as exc:
            raise FloatingPointError(
                "non-finite value encountered during concordance "
                "correlation coefficient computation"
            ) from exc
        if not math.isfinite(true_term) or not math.isfinite(pred_term):
            raise FloatingPointError(
                "non-finite value encountered during concordance "
                "correlation coefficient computation"
            )
        true_mean_terms.append(true_term)
        pred_mean_terms.append(pred_term)
    try:
        mt = math.fsum(true_mean_terms) / total_weight
        mp = math.fsum(pred_mean_terms) / total_weight
    except (OverflowError, ValueError, ZeroDivisionError) as exc:
        raise FloatingPointError(
            "non-finite value encountered during concordance correlation "
            "coefficient computation"
        ) from exc
    if not math.isfinite(mt) or not math.isfinite(mp):
        raise FloatingPointError(
            "non-finite value encountered during concordance correlation "
            "coefficient computation"
        )

    vt_terms = []
    vp_terms = []
    cov_terms = []
    for i in range(n):
        try:
            true_offset = t[i] - mt
            pred_offset = p[i] - mp
            vt_term = weights[i] * true_offset ** 2
            vp_term = weights[i] * pred_offset ** 2
            cov_term = weights[i] * true_offset * pred_offset
        except (OverflowError, ValueError) as exc:
            raise FloatingPointError(
                "non-finite value encountered during concordance "
                "correlation coefficient computation"
            ) from exc
        if (not math.isfinite(vt_term) or not math.isfinite(vp_term)
                or not math.isfinite(cov_term)):
            raise FloatingPointError(
                "non-finite value encountered during concordance "
                "correlation coefficient computation"
            )
        vt_terms.append(vt_term)
        vp_terms.append(vp_term)
        cov_terms.append(cov_term)
    try:
        vt = math.fsum(vt_terms) / total_weight
        vp = math.fsum(vp_terms) / total_weight
        covariance = math.fsum(cov_terms) / total_weight
    except (OverflowError, ValueError, ZeroDivisionError) as exc:
        raise FloatingPointError(
            "non-finite value encountered during concordance correlation "
            "coefficient computation"
        ) from exc
    if (not math.isfinite(vt) or not math.isfinite(vp)
            or not math.isfinite(covariance)):
        raise FloatingPointError(
            "non-finite value encountered during concordance correlation "
            "coefficient computation"
        )

    try:
        mean_gap_squared = (mt - mp) ** 2
        denominator = vt + vp + mean_gap_squared
    except (OverflowError, ValueError) as exc:
        raise FloatingPointError(
            "non-finite value encountered during concordance correlation "
            "coefficient computation"
        ) from exc
    if not math.isfinite(denominator):
        raise FloatingPointError(
            "non-finite value encountered during concordance correlation "
            "coefficient computation"
        )

    if denominator == 0.0:
        return 1.0
    try:
        result = 2.0 * covariance / denominator
    except (OverflowError, ValueError, ZeroDivisionError) as exc:
        raise FloatingPointError(
            "non-finite value encountered during concordance correlation "
            "coefficient computation"
        ) from exc
    if not math.isfinite(result):
        raise FloatingPointError(
            "non-finite value encountered during concordance correlation "
            "coefficient computation"
        )
    if result == 0:
        result = 0.0
    return result


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


def symmetric_mean_absolute_percentage_error(
    y_true, y_pred, sample_weight=None
) -> float:
    """Return the weighted symmetric mean absolute percentage error (SMAPE).

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
    value, ``p`` the prediction, and ``w`` the weight, the symmetric
    absolute percentage error is ``e = 0.0`` when
    ``abs(t) + abs(p) == 0`` and
    ``e = 2.0 * abs(t - p) / (abs(t) + abs(p))`` otherwise.
    ``math.fsum`` computes ``W = sum(w)`` and ``L = sum(w * e)``. If
    summing ``W`` raises OverflowError or ValueError, or ``W`` is
    non-finite or less than or equal to zero, a ValueError is raised;
    otherwise the result is ``L / W``. Overflow, invalid operations
    during the post-validation conversion, absolute value, arithmetic,
    or ``L`` summation, and non-finite intermediate values or results
    raise FloatingPointError. An exact zero result is normalized to
    ``0.0``.

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
            "non-finite value encountered during symmetric mean absolute "
            "percentage error"
        ) from exc
    for values in (t, p, w):
        for value in values:
            if not math.isfinite(value):
                raise FloatingPointError(
                    "non-finite value encountered during symmetric mean "
                    "absolute percentage error"
                )

    loss_terms = []
    for i in range(n):
        try:
            abs_true = abs(t[i])
            abs_pred = abs(p[i])
            denom = abs_true + abs_pred
            abs_diff = abs(t[i] - p[i])
        except (OverflowError, ValueError) as exc:
            raise FloatingPointError(
                "non-finite value encountered during symmetric mean "
                "absolute percentage error"
            ) from exc
        if (
            not math.isfinite(abs_true)
            or not math.isfinite(abs_pred)
            or not math.isfinite(denom)
            or not math.isfinite(abs_diff)
        ):
            raise FloatingPointError(
                "non-finite value encountered during symmetric mean "
                "absolute percentage error"
            )
        if denom == 0.0:
            error = 0.0
        else:
            try:
                error = 2.0 * abs_diff / denom
            except (OverflowError, ValueError, ZeroDivisionError) as exc:
                raise FloatingPointError(
                    "non-finite value encountered during symmetric mean "
                    "absolute percentage error"
                ) from exc
        if not math.isfinite(error):
            raise FloatingPointError(
                "non-finite value encountered during symmetric mean "
                "absolute percentage error"
            )
        try:
            term = w[i] * error
        except (OverflowError, ValueError) as exc:
            raise FloatingPointError(
                "non-finite value encountered during symmetric mean "
                "absolute percentage error"
            ) from exc
        if not math.isfinite(term):
            raise FloatingPointError(
                "non-finite value encountered during symmetric mean "
                "absolute percentage error"
            )
        loss_terms.append(term)

    try:
        total_weight = math.fsum(w)
    except (OverflowError, ValueError) as exc:
        raise ValueError("the total weight must be greater than 0") from exc
    if not math.isfinite(total_weight) or total_weight <= 0.0:
        raise ValueError("the total weight must be greater than 0")
    try:
        total_loss = math.fsum(loss_terms)
    except (OverflowError, ValueError) as exc:
        raise FloatingPointError(
            "non-finite value encountered during symmetric mean absolute "
            "percentage error"
        ) from exc
    if not math.isfinite(total_loss):
        raise FloatingPointError(
            "non-finite value encountered during symmetric mean absolute "
            "percentage error"
        )
    try:
        result = total_loss / total_weight
    except (OverflowError, ValueError, ZeroDivisionError) as exc:
        raise FloatingPointError(
            "non-finite value encountered during symmetric mean absolute "
            "percentage error"
        ) from exc
    if not math.isfinite(result):
        raise FloatingPointError(
            "non-finite value encountered during symmetric mean absolute "
            "percentage error"
        )
    if result == 0:
        result = 0.0
    return result


def median_absolute_percentage_error(y_true, y_pred, sample_weight=None) -> float:
    """Return the weighted median of the element-wise absolute percentage
    errors.

    ``y_true`` and ``y_pred`` must be non-empty lists of equal length
    whose elements are finite values of type exactly ``int`` or
    ``float`` (booleans are rejected). ``sample_weight`` must be
    ``None`` -- every sample then weighs ``1.0`` -- or a list of the
    same length whose elements are finite non-negative values of type
    exactly ``int`` or ``float`` (booleans are rejected). Any
    container, length, type, range, or finiteness violation (including
    ``OverflowError`` raised by ``math.isfinite``) raises ValueError.

    After validation, the values and weights are converted to
    ``float`` in input order. ``math.fsum`` computes the total weight
    ``W = sum(w_i)``; if that summation overflows or is invalid, is
    non-finite, or ``W`` is less than or equal to zero, a ValueError is
    raised. For each index,
    ``e_i = abs(t_i - p_i) / max(abs(t_i), sys.float_info.epsilon)``
    and the items are sorted by ``(e_i, i)`` ascending. Starting from
    an empty prefix, the weight prefix is recomputed with
    ``math.fsum`` after each included item (zero-weight items are
    retained); the result is the first ``e_i`` whose prefix is greater
    than or equal to ``W / 2``. A prefix exactly equal to ``W / 2``
    still takes the current ``e_i`` -- adjacent values are never
    averaged. Overflow, invalid operations during the post-validation
    conversion, subtraction, absolute value, sorting, the division, or
    a prefix summation, and non-finite intermediate values raise
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
            "non-finite value encountered during median absolute "
            "percentage error"
        ) from exc
    for values in (t, p, w):
        for value in values:
            if not math.isfinite(value):
                raise FloatingPointError(
                    "non-finite value encountered during median absolute "
                    "percentage error"
                )

    try:
        total_weight = math.fsum(w)
    except (OverflowError, ValueError) as exc:
        raise ValueError("the total weight must be greater than 0") from exc
    if not math.isfinite(total_weight) or total_weight <= 0.0:
        raise ValueError("the total weight must be greater than 0")

    items = []
    for i in range(n):
        try:
            abs_true = abs(t[i])
            abs_diff = abs(t[i] - p[i])
        except (OverflowError, ValueError) as exc:
            raise FloatingPointError(
                "non-finite value encountered during median absolute "
                "percentage error"
            ) from exc
        if not math.isfinite(abs_true) or not math.isfinite(abs_diff):
            raise FloatingPointError(
                "non-finite value encountered during median absolute "
                "percentage error"
            )
        denom = max(abs_true, sys.float_info.epsilon)
        try:
            e = abs_diff / denom
        except (OverflowError, ValueError, ZeroDivisionError) as exc:
            raise FloatingPointError(
                "non-finite value encountered during median absolute "
                "percentage error"
            ) from exc
        if not math.isfinite(e):
            raise FloatingPointError(
                "non-finite value encountered during median absolute "
                "percentage error"
            )
        items.append((e, i))

    try:
        items.sort()
    except (OverflowError, ValueError, TypeError) as exc:
        raise FloatingPointError(
            "non-finite value encountered during median absolute "
            "percentage error"
        ) from exc

    try:
        half_weight = total_weight / 2.0
    except (OverflowError, ValueError) as exc:
        raise FloatingPointError(
            "non-finite value encountered during median absolute "
            "percentage error"
        ) from exc
    if not math.isfinite(half_weight):
        raise FloatingPointError(
            "non-finite value encountered during median absolute "
            "percentage error"
        )

    included = []
    result = None
    for e, i in items:
        included.append(w[i])
        try:
            prefix = math.fsum(included)
        except (OverflowError, ValueError) as exc:
            raise FloatingPointError(
                "non-finite value encountered during median absolute "
                "percentage error"
            ) from exc
        if not math.isfinite(prefix):
            raise FloatingPointError(
                "non-finite value encountered during median absolute "
                "percentage error"
            )
        if prefix >= half_weight:
            result = e
            break

    if result == 0:
        result = 0.0
    return result


def mean_squared_percentage_error(y_true, y_pred, sample_weight=None) -> float:
    """Return the weighted mean squared percentage error (MSPE).

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
    value, ``p`` the prediction, and ``w`` the weight, the squared
    percentage error is
    ``e = (t - p) / max(abs(t), sys.float_info.epsilon)`` -- when
    ``t`` is zero, ``sys.float_info.epsilon`` serves as the
    denominator -- and ``math.fsum`` computes both ``W = sum(w)`` and
    ``L = sum(w * e ** 2)``. When ``math.fsum`` overflows or fails
    while computing ``W``, or ``W`` is not finite or ``W <= 0``, a
    ValueError is raised. The result is ``L / W``. Apart from the
    validation and the total-weight checks above, overflow, invalid
    operations, or division by zero during the post-validation
    conversion or arithmetic, and non-finite intermediate values or
    results, raise FloatingPointError. An exact zero result is
    normalized to ``0.0``.

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
            "non-finite value encountered during mean squared "
            "percentage error"
        ) from exc
    for values in (t, p, w):
        for value in values:
            if not math.isfinite(value):
                raise FloatingPointError(
                    "non-finite value encountered during mean squared "
                    "percentage error"
                )

    weight_terms = []
    loss_terms = []
    for i in range(n):
        try:
            abs_true = abs(t[i])
            diff = t[i] - p[i]
        except (OverflowError, ValueError) as exc:
            raise FloatingPointError(
                "non-finite value encountered during mean squared "
                "percentage error"
            ) from exc
        if not math.isfinite(abs_true) or not math.isfinite(diff):
            raise FloatingPointError(
                "non-finite value encountered during mean squared "
                "percentage error"
            )
        denom = max(abs_true, sys.float_info.epsilon)
        try:
            error = diff / denom
            squared = error ** 2
            term = w[i] * squared
        except (OverflowError, ValueError, ZeroDivisionError) as exc:
            raise FloatingPointError(
                "non-finite value encountered during mean squared "
                "percentage error"
            ) from exc
        if (
            not math.isfinite(error)
            or not math.isfinite(squared)
            or not math.isfinite(term)
        ):
            raise FloatingPointError(
                "non-finite value encountered during mean squared "
                "percentage error"
            )
        weight_terms.append(w[i])
        loss_terms.append(term)

    try:
        total_weight = math.fsum(weight_terms)
    except (OverflowError, ValueError) as exc:
        raise ValueError("the total weight must be a finite number") from exc
    if not math.isfinite(total_weight):
        raise ValueError("the total weight must be a finite number")
    if total_weight <= 0.0:
        raise ValueError("the total weight must be greater than 0")
    try:
        total_loss = math.fsum(loss_terms)
    except (OverflowError, ValueError) as exc:
        raise FloatingPointError(
            "non-finite value encountered during mean squared "
            "percentage error"
        ) from exc
    if not math.isfinite(total_loss):
        raise FloatingPointError(
            "non-finite value encountered during mean squared "
            "percentage error"
        )
    try:
        result = total_loss / total_weight
    except (OverflowError, ValueError, ZeroDivisionError) as exc:
        raise FloatingPointError(
            "non-finite value encountered during mean squared "
            "percentage error"
        ) from exc
    if not math.isfinite(result):
        raise FloatingPointError(
            "non-finite value encountered during mean squared "
            "percentage error"
        )
    if result == 0:
        result = 0.0
    return result


def mean_pinball_loss(y_true, y_pred, alpha=0.5, sample_weight=None) -> float:
    """Return the weighted mean pinball (quantile) loss.

    ``y_true`` and ``y_pred`` must be non-empty lists of equal length
    whose elements are finite values of type exactly ``int`` or
    ``float`` (booleans are rejected). ``alpha`` must be a finite value
    of type exactly ``int`` or ``float`` (booleans are rejected) in the
    closed interval ``[0, 1]``. ``sample_weight`` must be ``None`` --
    every sample then weighs ``1.0`` -- or a list of the same length
    whose elements are finite non-negative values of type exactly
    ``int`` or ``float`` (booleans are rejected). Any container, length,
    type, range, or finiteness violation (including ``OverflowError``
    raised by ``math.isfinite``) raises ValueError.

    After validation, the values, alpha, and weights are converted to
    ``float`` in input order. ``math.fsum`` computes the total weight
    ``W = sum(w_i)``; if that summation overflows or is invalid, is
    non-finite, or ``W`` is less than or equal to zero, a ValueError is
    raised. For each index, with ``d = y_true_i - y_pred_i``, the
    per-sample loss is ``l = alpha * d`` when ``d >= 0`` and
    ``l = (alpha - 1.0) * d`` otherwise. In input order, ``math.fsum``
    computes ``L = sum(w_i * l_i)`` and the result is ``L / W``.
    Overflow, invalid operations during the post-validation conversion
    or arithmetic, and non-finite intermediate values or results raise
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

    if type(alpha) not in (int, float):
        raise ValueError(
            "alpha must be a finite non-boolean number in [0, 1]"
        )
    try:
        alpha_finite = math.isfinite(alpha)
    except OverflowError as exc:
        raise ValueError(
            "alpha must be a finite non-boolean number in [0, 1]"
        ) from exc
    if not alpha_finite or alpha < 0 or alpha > 1:
        raise ValueError(
            "alpha must be a finite non-boolean number in [0, 1]"
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
        a = float(alpha)
        if sample_weight is None:
            w = [1.0] * n
        else:
            w = [float(value) for value in sample_weight]
    except (OverflowError, ValueError) as exc:
        raise FloatingPointError(
            "non-finite value encountered during mean pinball loss"
        ) from exc
    for values in (t, p, w):
        for value in values:
            if not math.isfinite(value):
                raise FloatingPointError(
                    "non-finite value encountered during mean pinball loss"
                )
    if not math.isfinite(a):
        raise FloatingPointError(
            "non-finite value encountered during mean pinball loss"
        )

    try:
        total_weight = math.fsum(w)
    except (OverflowError, ValueError) as exc:
        raise ValueError("the total weight must be greater than 0") from exc
    if not math.isfinite(total_weight) or total_weight <= 0.0:
        raise ValueError("the total weight must be greater than 0")

    loss_terms = []
    for i in range(n):
        try:
            d = t[i] - p[i]
            if d >= 0:
                loss = a * d
            else:
                loss = (a - 1.0) * d
            term = w[i] * loss
        except (OverflowError, ValueError) as exc:
            raise FloatingPointError(
                "non-finite value encountered during mean pinball loss"
            ) from exc
        if not math.isfinite(loss) or not math.isfinite(term):
            raise FloatingPointError(
                "non-finite value encountered during mean pinball loss"
            )
        loss_terms.append(term)

    try:
        total_loss = math.fsum(loss_terms)
    except (OverflowError, ValueError) as exc:
        raise FloatingPointError(
            "non-finite value encountered during mean pinball loss"
        ) from exc
    if not math.isfinite(total_loss):
        raise FloatingPointError(
            "non-finite value encountered during mean pinball loss"
        )
    try:
        result = total_loss / total_weight
    except (OverflowError, ValueError, ZeroDivisionError) as exc:
        raise FloatingPointError(
            "non-finite value encountered during mean pinball loss"
        ) from exc
    if not math.isfinite(result):
        raise FloatingPointError(
            "non-finite value encountered during mean pinball loss"
        )
    if result == 0:
        result = 0.0
    return result


def d2_pinball_score(y_true, y_pred, alpha=0.5, sample_weight=None) -> float:
    """Return the D^2 pinball score: relative improvement over a constant.

    The score is ``1 - L / L0`` where ``L`` is the weighted pinball loss
    of ``y_pred`` and ``L0`` is the weighted pinball loss of the constant
    baseline ``q``, the weighted ``alpha``-quantile of ``y_true``. It
    measures how much the quantile predictions improve on always
    predicting that constant.

    ``y_true`` and ``y_pred`` must be non-empty lists of equal length
    whose elements are finite values of type exactly ``int`` or
    ``float`` (booleans are rejected). ``alpha`` must be a finite value
    of type exactly ``int`` or ``float`` (booleans are rejected) in the
    closed interval ``[0, 1]``. ``sample_weight`` must be ``None`` --
    every sample then weighs ``1.0`` -- or a list of the same length
    whose elements are finite non-negative values of type exactly
    ``int`` or ``float`` (booleans are rejected). Any container, length,
    type, range, or finiteness violation (including ``OverflowError``
    raised by ``math.isfinite``) raises ValueError.

    After validation, the values, alpha, and weights are converted to
    ``float`` in input order. ``math.fsum`` computes the total weight
    ``W = sum(w_i)``; if that summation overflows or is invalid, is
    non-finite, or ``W`` is less than or equal to zero, a ValueError is
    raised. The triples ``(t_i, i, w_i)`` are sorted ascending by
    ``(t_i, i)`` and ``q`` is the first ``t_i`` whose running prefix
    weight (computed with ``math.fsum``) reaches or exceeds
    ``alpha * W``. With ``rho(d) = alpha * d`` for ``d >= 0`` and
    ``rho(d) = (alpha - 1.0) * d`` otherwise, ``math.fsum`` in original
    sample order computes ``L = sum(w_i * rho(t_i - p_i))`` and
    ``L0 = sum(w_i * rho(t_i - q))``. When ``L0`` is non-zero the result
    is ``1.0 - L / L0``; when ``L0`` and ``L`` are both zero the result
    is ``1.0``; when only ``L0`` is zero the result is ``0.0``.

    Overflow, invalid operations, or division by zero during the
    post-validation conversion, sorting, prefix summation, loss
    arithmetic, or final division, and any non-finite intermediate value
    or result raise FloatingPointError. An exact zero result is
    normalized to ``0.0``.

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

    if type(alpha) not in (int, float):
        raise ValueError(
            "alpha must be a finite non-boolean number in [0, 1]"
        )
    try:
        alpha_finite = math.isfinite(alpha)
    except OverflowError as exc:
        raise ValueError(
            "alpha must be a finite non-boolean number in [0, 1]"
        ) from exc
    if not alpha_finite or alpha < 0 or alpha > 1:
        raise ValueError(
            "alpha must be a finite non-boolean number in [0, 1]"
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
        a = float(alpha)
        if sample_weight is None:
            w = [1.0] * n
        else:
            w = [float(value) for value in sample_weight]
    except (OverflowError, ValueError) as exc:
        raise FloatingPointError(
            "non-finite value encountered during d2 pinball score"
        ) from exc
    for values in (t, p, w):
        for value in values:
            if not math.isfinite(value):
                raise FloatingPointError(
                    "non-finite value encountered during d2 pinball score"
                )
    if not math.isfinite(a):
        raise FloatingPointError(
            "non-finite value encountered during d2 pinball score"
        )

    try:
        total_weight = math.fsum(w)
    except (OverflowError, ValueError) as exc:
        raise ValueError("the total weight must be greater than 0") from exc
    if not math.isfinite(total_weight) or total_weight <= 0.0:
        raise ValueError("the total weight must be greater than 0")

    try:
        target = a * total_weight
        if not math.isfinite(target):
            raise FloatingPointError(
                "non-finite value encountered during d2 pinball score"
            )

        # Sort (t_i, i, w_i) ascending by (t_i, i); the unique index
        # makes ties deterministic. q is the first t_i whose running
        # prefix weight, accumulated with math.fsum, reaches alpha * W.
        triples = sorted((t[i], i, w[i]) for i in range(n))
        prefix_weights = []
        q = None
        for t_i, _, w_i in triples:
            prefix_weights.append(w_i)
            prefix_weight = math.fsum(prefix_weights)
            if not math.isfinite(prefix_weight):
                raise FloatingPointError(
                    "non-finite value encountered during d2 pinball score"
                )
            if prefix_weight >= target:
                q = t_i
                break
        if q is None or not math.isfinite(q):
            raise FloatingPointError(
                "non-finite value encountered during d2 pinball score"
            )

        model_terms = []
        baseline_terms = []
        for i in range(n):
            d = t[i] - p[i]
            if d >= 0:
                rho = a * d
            else:
                rho = (a - 1.0) * d
            if not math.isfinite(rho):
                raise FloatingPointError(
                    "non-finite value encountered during d2 pinball score"
                )
            term = w[i] * rho
            if not math.isfinite(term):
                raise FloatingPointError(
                    "non-finite value encountered during d2 pinball score"
                )
            model_terms.append(term)

            d0 = t[i] - q
            if d0 >= 0:
                rho0 = a * d0
            else:
                rho0 = (a - 1.0) * d0
            if not math.isfinite(rho0):
                raise FloatingPointError(
                    "non-finite value encountered during d2 pinball score"
                )
            term0 = w[i] * rho0
            if not math.isfinite(term0):
                raise FloatingPointError(
                    "non-finite value encountered during d2 pinball score"
                )
            baseline_terms.append(term0)

        model_loss = math.fsum(model_terms)
        baseline_loss = math.fsum(baseline_terms)
        if not math.isfinite(model_loss) or not math.isfinite(baseline_loss):
            raise FloatingPointError(
                "non-finite value encountered during d2 pinball score"
            )

        if baseline_loss != 0.0:
            result = 1.0 - model_loss / baseline_loss
        elif model_loss == 0.0:
            result = 1.0
        else:
            result = 0.0
        if not math.isfinite(result):
            raise FloatingPointError(
                "non-finite value encountered during d2 pinball score"
            )
    except (OverflowError, ValueError, ZeroDivisionError) as exc:
        raise FloatingPointError(
            "non-finite value encountered during d2 pinball score"
        ) from exc

    if result == 0:
        result = 0.0
    return result


def mean_huber_loss(y_true, y_pred, delta=1.0, sample_weight=None) -> float:
    """Return the weighted mean Huber loss.

    ``y_true`` and ``y_pred`` must be non-empty lists of equal length
    whose elements are finite values of type exactly ``int`` or
    ``float`` (booleans are rejected). ``delta`` must be a finite value
    of type exactly ``int`` or ``float`` (booleans are rejected) and
    strictly greater than zero. ``sample_weight`` must be ``None`` --
    every sample then weighs ``1.0`` -- or a list of the same length
    whose elements are finite non-negative values of type exactly
    ``int`` or ``float`` (booleans are rejected). Any container, length,
    type, range, or finiteness violation (including ``OverflowError``
    raised by ``math.isfinite``) raises ValueError.

    After validation, the values, delta, and weights are converted to
    ``float`` in input order. ``math.fsum`` computes the total weight
    ``W = sum(w_i)``; if that summation overflows or is invalid, is
    non-finite, or ``W`` is less than or equal to zero, a ValueError is
    raised. For each index, with ``r = y_true_i - y_pred_i`` and
    ``a = abs(r)``, the per-sample loss is ``l = 0.5 * r * r`` when
    ``a <= delta`` and ``l = delta * (a - 0.5 * delta)`` otherwise. In
    input order, ``math.fsum`` computes ``L = sum(w_i * l_i)`` and the
    result is ``L / W``. Overflow, invalid operations during the
    post-validation conversion or arithmetic, and non-finite
    intermediate values or results raise FloatingPointError. An exact
    zero result is normalized to ``0.0``.

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

    if type(delta) not in (int, float):
        raise ValueError("delta must be a finite non-boolean number > 0")
    try:
        delta_finite = math.isfinite(delta)
    except OverflowError as exc:
        raise ValueError(
            "delta must be a finite non-boolean number > 0"
        ) from exc
    if not delta_finite or delta <= 0:
        raise ValueError("delta must be a finite non-boolean number > 0")

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
        d = float(delta)
        if sample_weight is None:
            w = [1.0] * n
        else:
            w = [float(value) for value in sample_weight]
    except (OverflowError, ValueError) as exc:
        raise FloatingPointError(
            "non-finite value encountered during mean huber loss"
        ) from exc
    for values in (t, p, w):
        for value in values:
            if not math.isfinite(value):
                raise FloatingPointError(
                    "non-finite value encountered during mean huber loss"
                )
    if not math.isfinite(d):
        raise FloatingPointError(
            "non-finite value encountered during mean huber loss"
        )

    try:
        total_weight = math.fsum(w)
    except (OverflowError, ValueError) as exc:
        raise ValueError("the total weight must be greater than 0") from exc
    if not math.isfinite(total_weight) or total_weight <= 0.0:
        raise ValueError("the total weight must be greater than 0")

    loss_terms = []
    for i in range(n):
        try:
            r = t[i] - p[i]
            a = abs(r)
            if a <= d:
                loss = 0.5 * r * r
            else:
                loss = d * (a - 0.5 * d)
            term = w[i] * loss
        except (OverflowError, ValueError) as exc:
            raise FloatingPointError(
                "non-finite value encountered during mean huber loss"
            ) from exc
        if not math.isfinite(r) or not math.isfinite(a):
            raise FloatingPointError(
                "non-finite value encountered during mean huber loss"
            )
        if not math.isfinite(loss) or not math.isfinite(term):
            raise FloatingPointError(
                "non-finite value encountered during mean huber loss"
            )
        loss_terms.append(term)

    try:
        total_loss = math.fsum(loss_terms)
    except (OverflowError, ValueError) as exc:
        raise FloatingPointError(
            "non-finite value encountered during mean huber loss"
        ) from exc
    if not math.isfinite(total_loss):
        raise FloatingPointError(
            "non-finite value encountered during mean huber loss"
        )
    try:
        result = total_loss / total_weight
    except (OverflowError, ValueError, ZeroDivisionError) as exc:
        raise FloatingPointError(
            "non-finite value encountered during mean huber loss"
        ) from exc
    if not math.isfinite(result):
        raise FloatingPointError(
            "non-finite value encountered during mean huber loss"
        )
    if result == 0:
        result = 0.0
    return result


def mean_log_cosh_loss(y_true, y_pred, sample_weight=None) -> float:
    """Return the weighted mean logarithm of the hyperbolic cosine loss.

    ``y_true`` and ``y_pred`` must be non-empty lists of equal length
    whose elements are finite values of type exactly ``int`` or
    ``float`` (booleans are rejected). ``sample_weight`` must be
    ``None`` -- every sample then weighs ``1.0`` -- or a list of the
    same length whose elements are finite non-negative values of type
    exactly ``int`` or ``float`` (booleans are rejected). Any
    container, length, type, range, or finiteness violation (including
    ``OverflowError`` raised by ``math.isfinite``) raises ValueError.

    After validation, the values and weights are converted to ``float``
    in input order. ``math.fsum`` computes the total weight
    ``W = sum(w_i)``; if that summation overflows or is invalid, is
    non-finite, or ``W`` is less than or equal to zero, a ValueError is
    raised. For each index, with ``r = y_true_i - y_pred_i`` and
    ``a = abs(r)``, the per-sample loss is computed with the stable
    form ``l = a + math.log1p(math.exp(-2.0 * a)) - math.log(2.0)``.
    In input order, ``math.fsum`` computes ``L = sum(w_i * l_i)`` and
    the result is ``L / W``. Overflow, invalid operations during the
    post-validation conversion or arithmetic, and non-finite
    intermediate values or results raise FloatingPointError. An exact
    zero result is normalized to ``0.0``.

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
            "non-finite value encountered during mean log cosh loss"
        ) from exc
    for values in (t, p, w):
        for value in values:
            if not math.isfinite(value):
                raise FloatingPointError(
                    "non-finite value encountered during mean log cosh loss"
                )

    try:
        total_weight = math.fsum(w)
    except (OverflowError, ValueError) as exc:
        raise ValueError("the total weight must be greater than 0") from exc
    if not math.isfinite(total_weight) or total_weight <= 0.0:
        raise ValueError("the total weight must be greater than 0")

    log_two = math.log(2.0)
    loss_terms = []
    for i in range(n):
        try:
            r = t[i] - p[i]
            a = abs(r)
            z = -2.0 * a
            if not math.isfinite(z):
                raise FloatingPointError(
                    "non-finite value encountered during mean log cosh loss"
                )
            loss = a + math.log1p(math.exp(z)) - log_two
            term = w[i] * loss
        except (OverflowError, ValueError) as exc:
            raise FloatingPointError(
                "non-finite value encountered during mean log cosh loss"
            ) from exc
        if not math.isfinite(r) or not math.isfinite(a):
            raise FloatingPointError(
                "non-finite value encountered during mean log cosh loss"
            )
        if not math.isfinite(loss) or not math.isfinite(term):
            raise FloatingPointError(
                "non-finite value encountered during mean log cosh loss"
            )
        loss_terms.append(term)

    try:
        total_loss = math.fsum(loss_terms)
    except (OverflowError, ValueError) as exc:
        raise FloatingPointError(
            "non-finite value encountered during mean log cosh loss"
        ) from exc
    if not math.isfinite(total_loss):
        raise FloatingPointError(
            "non-finite value encountered during mean log cosh loss"
        )
    try:
        result = total_loss / total_weight
    except (OverflowError, ValueError, ZeroDivisionError) as exc:
        raise FloatingPointError(
            "non-finite value encountered during mean log cosh loss"
        ) from exc
    if not math.isfinite(result):
        raise FloatingPointError(
            "non-finite value encountered during mean log cosh loss"
        )
    if result == 0:
        result = 0.0
    return result


def mean_absolute_scaled_error(
    y_true, y_pred, y_train, m=1, sample_weight=None
) -> float:
    """Return the mean absolute scaled error (MASE).

    ``y_true`` and ``y_pred`` must be non-empty lists of equal length
    whose elements are finite values of type exactly ``int`` or
    ``float`` (booleans are rejected). ``y_train`` must be a list of
    the same kind of values with more than ``m`` elements. ``m`` must
    be a value of type exactly ``int`` greater than or equal to 1.
    ``sample_weight`` must be ``None`` -- every sample then weighs
    ``1.0`` -- or a list of the same length as ``y_true`` whose
    elements are finite non-negative values of type exactly ``int`` or
    ``float`` (booleans are rejected). Any container, length, type,
    range, or finiteness violation (including ``OverflowError`` raised
    by ``math.isfinite``) raises ValueError.

    After validation, the values and weights are converted to ``float``
    in input order, giving ``t``, ``p``, ``s``, and ``w`` with
    ``n = len(s)``. In ascending index order, ``math.fsum`` computes
    the total weight ``W = sum(w_i)``; if that summation overflows or
    is invalid, is non-finite, or ``W`` is less than or equal to zero,
    a ValueError is raised. The in-sample naive-forecast scale is
    ``D = sum(abs(s_i - s_{i-m}) for i in m..n-1) / (n - m)``; a zero
    ``D`` raises ValueError. The weighted mean absolute error is
    ``N = sum(w_i * abs(t_i - p_i)) / W`` and the result is ``N / D``.
    Overflow, invalid operations during the post-validation conversion
    or arithmetic, and non-finite intermediate values or results raise
    FloatingPointError. An exact zero result is normalized to ``0.0``.

    The return value is a float. The inputs are not modified.
    Deterministic: same inputs, same result.
    """
    n_samples = _check_metric_vectors(y_true, y_pred)
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

    if type(m) is not int or m < 1:
        raise ValueError("m must be a positive integer")
    if not isinstance(y_train, list):
        raise ValueError("y_train must be a list")
    if len(y_train) <= m:
        raise ValueError("y_train must have more elements than m")
    for value in y_train:
        if type(value) not in (int, float):
            raise ValueError(
                "y_train must contain only finite non-boolean numbers"
            )
        # math.isfinite raises OverflowError for ints too large to
        # convert to float; such values fail the finite requirement.
        try:
            finite = math.isfinite(value)
        except OverflowError as exc:
            raise ValueError(
                "y_train must contain only finite non-boolean numbers"
            ) from exc
        if not finite:
            raise ValueError(
                "y_train must contain only finite non-boolean numbers"
            )

    if sample_weight is not None:
        if not isinstance(sample_weight, list) or len(sample_weight) != n_samples:
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
        s = [float(value) for value in y_train]
        if sample_weight is None:
            w = [1.0] * n_samples
        else:
            w = [float(value) for value in sample_weight]
    except (OverflowError, ValueError) as exc:
        raise FloatingPointError(
            "non-finite value encountered during mean absolute scaled error"
        ) from exc
    for values in (t, p, s, w):
        for value in values:
            if not math.isfinite(value):
                raise FloatingPointError(
                    "non-finite value encountered during mean absolute "
                    "scaled error"
                )
    n = len(s)

    try:
        total_weight = math.fsum(w)
    except (OverflowError, ValueError) as exc:
        raise ValueError("the total weight must be greater than 0") from exc
    if not math.isfinite(total_weight) or total_weight <= 0.0:
        raise ValueError("the total weight must be greater than 0")

    scale_terms = []
    for i in range(m, n):
        try:
            term = abs(s[i] - s[i - m])
        except (OverflowError, ValueError) as exc:
            raise FloatingPointError(
                "non-finite value encountered during mean absolute scaled "
                "error"
            ) from exc
        if not math.isfinite(term):
            raise FloatingPointError(
                "non-finite value encountered during mean absolute scaled "
                "error"
            )
        scale_terms.append(term)
    try:
        scale_sum = math.fsum(scale_terms)
    except (OverflowError, ValueError) as exc:
        raise FloatingPointError(
            "non-finite value encountered during mean absolute scaled error"
        ) from exc
    if not math.isfinite(scale_sum):
        raise FloatingPointError(
            "non-finite value encountered during mean absolute scaled error"
        )
    try:
        scale = scale_sum / (n - m)
    except (OverflowError, ValueError, ZeroDivisionError) as exc:
        raise FloatingPointError(
            "non-finite value encountered during mean absolute scaled error"
        ) from exc
    if not math.isfinite(scale):
        raise FloatingPointError(
            "non-finite value encountered during mean absolute scaled error"
        )
    if scale == 0:
        raise ValueError("the naive forecast scale must be greater than 0")

    error_terms = []
    for i in range(n_samples):
        try:
            term = w[i] * abs(t[i] - p[i])
        except (OverflowError, ValueError) as exc:
            raise FloatingPointError(
                "non-finite value encountered during mean absolute scaled "
                "error"
            ) from exc
        if not math.isfinite(term):
            raise FloatingPointError(
                "non-finite value encountered during mean absolute scaled "
                "error"
            )
        error_terms.append(term)
    try:
        error_sum = math.fsum(error_terms)
    except (OverflowError, ValueError) as exc:
        raise FloatingPointError(
            "non-finite value encountered during mean absolute scaled error"
        ) from exc
    if not math.isfinite(error_sum):
        raise FloatingPointError(
            "non-finite value encountered during mean absolute scaled error"
        )
    try:
        numerator = error_sum / total_weight
    except (OverflowError, ValueError, ZeroDivisionError) as exc:
        raise FloatingPointError(
            "non-finite value encountered during mean absolute scaled error"
        ) from exc
    if not math.isfinite(numerator):
        raise FloatingPointError(
            "non-finite value encountered during mean absolute scaled error"
        )
    try:
        result = numerator / scale
    except (OverflowError, ValueError, ZeroDivisionError) as exc:
        raise FloatingPointError(
            "non-finite value encountered during mean absolute scaled error"
        ) from exc
    if not math.isfinite(result):
        raise FloatingPointError(
            "non-finite value encountered during mean absolute scaled error"
        )
    if result == 0:
        result = 0.0
    return result


def mean_poisson_deviance(y_true, y_pred, sample_weight=None) -> float:
    """Return the weighted mean Poisson deviance.

    ``y_true`` and ``y_pred`` must be non-empty lists of equal length
    whose elements are finite values of type exactly ``int`` or
    ``float`` (booleans are rejected). Every element of ``y_true`` must
    be greater than or equal to zero and every element of ``y_pred``
    must be strictly greater than zero. ``sample_weight`` must be
    ``None`` -- every sample then weighs ``1.0`` -- or a list of the
    same length whose elements are finite non-negative values of type
    exactly ``int`` or ``float`` (booleans are rejected). Any container,
    length, type, range, or finiteness violation (including
    ``OverflowError`` raised by ``math.isfinite``) raises ValueError.

    After validation, the values and weights are converted to ``float``
    in input order. ``math.fsum`` computes the total weight
    ``W = sum(w_i)``; if that summation overflows or is invalid, is
    non-finite, or ``W`` is less than or equal to zero, a ValueError is
    raised. For each index, with ``t = y_true_i`` and ``p = y_pred_i``,
    the per-sample deviance is ``d = 2.0 * p`` when ``t == 0`` and
    ``d = 2.0 * (t * log(t / p) - t + p)`` otherwise; the deviance must
    be finite before the weighted term ``w_i * d`` is formed. In input
    order, ``math.fsum`` computes ``L = sum(w_i * d_i)`` and the result
    is ``L / W``. Overflow, invalid operations during the
    post-validation conversion or arithmetic, and non-finite
    intermediate values or results raise FloatingPointError. An exact
    zero result is normalized to ``0.0``.

    The return value is a float. The inputs are not modified.
    Deterministic: same inputs, same result.
    """
    n = _check_metric_vectors(y_true, y_pred)
    for value in y_true:
        if type(value) not in (int, float):
            raise ValueError(
                "y_true must contain only finite non-boolean numbers "
                "greater than or equal to 0"
            )
        # math.isfinite raises OverflowError for ints too large to
        # convert to float; such values fail the finite requirement.
        try:
            finite = math.isfinite(value)
        except OverflowError as exc:
            raise ValueError(
                "y_true must contain only finite non-boolean numbers "
                "greater than or equal to 0"
            ) from exc
        if not finite or value < 0:
            raise ValueError(
                "y_true must contain only finite non-boolean numbers "
                "greater than or equal to 0"
            )
    for value in y_pred:
        if type(value) not in (int, float):
            raise ValueError(
                "y_pred must contain only finite non-boolean numbers "
                "greater than 0"
            )
        # math.isfinite raises OverflowError for ints too large to
        # convert to float; such values fail the finite requirement.
        try:
            finite = math.isfinite(value)
        except OverflowError as exc:
            raise ValueError(
                "y_pred must contain only finite non-boolean numbers "
                "greater than 0"
            ) from exc
        if not finite or value <= 0:
            raise ValueError(
                "y_pred must contain only finite non-boolean numbers "
                "greater than 0"
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
            "non-finite value encountered during mean Poisson deviance"
        ) from exc
    for values in (t, p, w):
        for value in values:
            if not math.isfinite(value):
                raise FloatingPointError(
                    "non-finite value encountered during mean Poisson "
                    "deviance"
                )

    try:
        total_weight = math.fsum(w)
    except (OverflowError, ValueError) as exc:
        raise ValueError("the total weight must be greater than 0") from exc
    if not math.isfinite(total_weight) or total_weight <= 0.0:
        raise ValueError("the total weight must be greater than 0")

    deviance_terms = []
    for i in range(n):
        try:
            if t[i] == 0:
                d = 2.0 * p[i]
            else:
                d = 2.0 * (t[i] * math.log(t[i] / p[i]) - t[i] + p[i])
        except (OverflowError, ValueError) as exc:
            raise FloatingPointError(
                "non-finite value encountered during mean Poisson deviance"
            ) from exc
        if not math.isfinite(d):
            raise FloatingPointError(
                "non-finite value encountered during mean Poisson deviance"
            )
        try:
            term = w[i] * d
        except (OverflowError, ValueError) as exc:
            raise FloatingPointError(
                "non-finite value encountered during mean Poisson deviance"
            ) from exc
        if not math.isfinite(term):
            raise FloatingPointError(
                "non-finite value encountered during mean Poisson deviance"
            )
        deviance_terms.append(term)

    try:
        total_deviance = math.fsum(deviance_terms)
    except (OverflowError, ValueError) as exc:
        raise FloatingPointError(
            "non-finite value encountered during mean Poisson deviance"
        ) from exc
    if not math.isfinite(total_deviance):
        raise FloatingPointError(
            "non-finite value encountered during mean Poisson deviance"
        )
    try:
        result = total_deviance / total_weight
    except (OverflowError, ValueError, ZeroDivisionError) as exc:
        raise FloatingPointError(
            "non-finite value encountered during mean Poisson deviance"
        ) from exc
    if not math.isfinite(result):
        raise FloatingPointError(
            "non-finite value encountered during mean Poisson deviance"
        )
    if result == 0:
        result = 0.0
    return result


def mean_gamma_deviance(y_true, y_pred, sample_weight=None) -> float:
    """Return the weighted mean Gamma deviance.

    ``y_true`` and ``y_pred`` must be non-empty lists of equal length
    whose elements are finite values of type exactly ``int`` or
    ``float`` (booleans are rejected). Every element of both vectors
    must be strictly greater than zero. ``sample_weight`` must be
    ``None`` -- every sample then weighs ``1.0`` -- or a list of the
    same length whose elements are finite non-negative values of type
    exactly ``int`` or ``float`` (booleans are rejected). Any container,
    length, type, range, or finiteness violation (including
    ``OverflowError`` raised by ``math.isfinite``) raises ValueError.

    After validation, the values and weights are converted to ``float``
    in input order. ``math.fsum`` computes the total weight
    ``W = sum(w_i)``; if that summation overflows or is invalid, is
    non-finite, or ``W`` is less than or equal to zero, a ValueError is
    raised. For each index, with ``t = y_true_i`` and ``p = y_pred_i``,
    the per-sample deviance is ``d = 2.0 * (log(p / t) + t / p - 1.0)``;
    the deviance must be finite before the weighted term ``w_i * d`` is
    formed. In input order, ``math.fsum`` computes ``L = sum(w_i * d_i)``
    and the result is ``L / W``. Overflow, invalid operations, or
    division by zero during the post-validation conversion or
    arithmetic, and non-finite intermediate values or results, raise
    FloatingPointError. An exact zero result is normalized to ``0.0``.

    The return value is a float. The inputs are not modified.
    Deterministic: same inputs, same result.
    """
    n = _check_metric_vectors(y_true, y_pred)
    for value in y_true:
        if type(value) not in (int, float):
            raise ValueError(
                "y_true must contain only finite non-boolean numbers "
                "greater than 0"
            )
        # math.isfinite raises OverflowError for ints too large to
        # convert to float; such values fail the finite requirement.
        try:
            finite = math.isfinite(value)
        except OverflowError as exc:
            raise ValueError(
                "y_true must contain only finite non-boolean numbers "
                "greater than 0"
            ) from exc
        if not finite or value <= 0:
            raise ValueError(
                "y_true must contain only finite non-boolean numbers "
                "greater than 0"
            )
    for value in y_pred:
        if type(value) not in (int, float):
            raise ValueError(
                "y_pred must contain only finite non-boolean numbers "
                "greater than 0"
            )
        # math.isfinite raises OverflowError for ints too large to
        # convert to float; such values fail the finite requirement.
        try:
            finite = math.isfinite(value)
        except OverflowError as exc:
            raise ValueError(
                "y_pred must contain only finite non-boolean numbers "
                "greater than 0"
            ) from exc
        if not finite or value <= 0:
            raise ValueError(
                "y_pred must contain only finite non-boolean numbers "
                "greater than 0"
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
            "non-finite value encountered during mean gamma deviance"
        ) from exc
    for values in (t, p, w):
        for value in values:
            if not math.isfinite(value):
                raise FloatingPointError(
                    "non-finite value encountered during mean gamma "
                    "deviance"
                )

    try:
        total_weight = math.fsum(w)
    except (OverflowError, ValueError) as exc:
        raise ValueError("the total weight must be greater than 0") from exc
    if not math.isfinite(total_weight) or total_weight <= 0.0:
        raise ValueError("the total weight must be greater than 0")

    deviance_terms = []
    for i in range(n):
        try:
            a = p[i] / t[i]
            b = t[i] / p[i]
            c = math.log(a)
            d = 2.0 * (c + b - 1.0)
        except (OverflowError, ValueError, ZeroDivisionError) as exc:
            raise FloatingPointError(
                "non-finite value encountered during mean gamma deviance"
            ) from exc
        if not math.isfinite(d):
            raise FloatingPointError(
                "non-finite value encountered during mean gamma deviance"
            )
        try:
            term = w[i] * d
        except (OverflowError, ValueError) as exc:
            raise FloatingPointError(
                "non-finite value encountered during mean gamma deviance"
            ) from exc
        if not math.isfinite(term):
            raise FloatingPointError(
                "non-finite value encountered during mean gamma deviance"
            )
        deviance_terms.append(term)

    try:
        total_deviance = math.fsum(deviance_terms)
    except (OverflowError, ValueError) as exc:
        raise FloatingPointError(
            "non-finite value encountered during mean gamma deviance"
        ) from exc
    if not math.isfinite(total_deviance):
        raise FloatingPointError(
            "non-finite value encountered during mean gamma deviance"
        )
    try:
        result = total_deviance / total_weight
    except (OverflowError, ValueError, ZeroDivisionError) as exc:
        raise FloatingPointError(
            "non-finite value encountered during mean gamma deviance"
        ) from exc
    if not math.isfinite(result):
        raise FloatingPointError(
            "non-finite value encountered during mean gamma deviance"
        )
    if result == 0:
        result = 0.0
    return result


def mean_tweedie_deviance(y_true, y_pred, power=1.5, sample_weight=None) -> float:
    """Return the weighted mean Tweedie deviance.

    ``y_true`` and ``y_pred`` must be non-empty lists of equal length
    whose elements are finite values of type exactly ``int`` or
    ``float`` (booleans are rejected). Every element of ``y_true`` must
    be greater than or equal to zero and every element of ``y_pred``
    must be strictly greater than zero. ``power`` must be a finite
    value of type exactly ``int`` or ``float`` (booleans are rejected)
    strictly between 1 and 2. ``sample_weight`` must be ``None`` --
    every sample then weighs ``1.0`` -- or a list of the same length
    whose elements are finite non-negative values of type exactly
    ``int`` or ``float`` (booleans are rejected). Any container,
    length, type, range, or finiteness violation (including
    ``OverflowError`` raised by ``math.isfinite``) raises ValueError.

    After validation, the values, weights, and ``power`` are converted
    to ``float`` in input order. ``math.fsum`` computes the total
    weight ``W = sum(w_i)``; if that summation overflows or is invalid,
    is non-finite, or ``W`` is less than or equal to zero, a ValueError
    is raised. For each index, with ``q = float(power)``,
    ``t = y_true_i``, and ``p = y_pred_i``, the per-sample deviance is
    ``d = 2 * (t**(2-q)/((1-q)*(2-q)) - t*p**(1-q)/(1-q)
    + p**(2-q)/(2-q))``; the deviance must be finite before the
    weighted term ``w_i * d`` is formed. In input order, ``math.fsum``
    computes ``L = sum(w_i * d_i)`` and the result is ``L / W``.
    Overflow, invalid operations, or division by zero during the
    post-validation conversion or arithmetic, and non-finite
    intermediate values or results, raise FloatingPointError. An exact
    zero result is normalized to ``0.0``.

    The return value is a float. The inputs are not modified.
    Deterministic: same inputs, same result.
    """
    n = _check_metric_vectors(y_true, y_pred)
    for value in y_true:
        if type(value) not in (int, float):
            raise ValueError(
                "y_true must contain only finite non-boolean numbers "
                "greater than or equal to 0"
            )
        # math.isfinite raises OverflowError for ints too large to
        # convert to float; such values fail the finite requirement.
        try:
            finite = math.isfinite(value)
        except OverflowError as exc:
            raise ValueError(
                "y_true must contain only finite non-boolean numbers "
                "greater than or equal to 0"
            ) from exc
        if not finite or value < 0:
            raise ValueError(
                "y_true must contain only finite non-boolean numbers "
                "greater than or equal to 0"
            )
    for value in y_pred:
        if type(value) not in (int, float):
            raise ValueError(
                "y_pred must contain only finite non-boolean numbers "
                "greater than 0"
            )
        # math.isfinite raises OverflowError for ints too large to
        # convert to float; such values fail the finite requirement.
        try:
            finite = math.isfinite(value)
        except OverflowError as exc:
            raise ValueError(
                "y_pred must contain only finite non-boolean numbers "
                "greater than 0"
            ) from exc
        if not finite or value <= 0:
            raise ValueError(
                "y_pred must contain only finite non-boolean numbers "
                "greater than 0"
            )

    if type(power) not in (int, float):
        raise ValueError(
            "power must be a finite non-boolean number strictly "
            "between 1 and 2"
        )
    # math.isfinite raises OverflowError for ints too large to
    # convert to float; such values fail the finite requirement.
    try:
        finite = math.isfinite(power)
    except OverflowError as exc:
        raise ValueError(
            "power must be a finite non-boolean number strictly "
            "between 1 and 2"
        ) from exc
    if not finite or power <= 1 or power >= 2:
        raise ValueError(
            "power must be a finite non-boolean number strictly "
            "between 1 and 2"
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
        q = float(power)
        if sample_weight is None:
            w = [1.0] * n
        else:
            w = [float(value) for value in sample_weight]
    except (OverflowError, ValueError) as exc:
        raise FloatingPointError(
            "non-finite value encountered during mean Tweedie deviance"
        ) from exc
    for values in (t, p, w):
        for value in values:
            if not math.isfinite(value):
                raise FloatingPointError(
                    "non-finite value encountered during mean Tweedie "
                    "deviance"
                )
    if not math.isfinite(q):
        raise FloatingPointError(
            "non-finite value encountered during mean Tweedie deviance"
        )

    try:
        total_weight = math.fsum(w)
    except (OverflowError, ValueError) as exc:
        raise ValueError("the total weight must be greater than 0") from exc
    if not math.isfinite(total_weight) or total_weight <= 0.0:
        raise ValueError("the total weight must be greater than 0")

    deviance_terms = []
    for i in range(n):
        try:
            d = 2 * (
                t[i] ** (2 - q) / ((1 - q) * (2 - q))
                - t[i] * p[i] ** (1 - q) / (1 - q)
                + p[i] ** (2 - q) / (2 - q)
            )
        except (OverflowError, ValueError, ZeroDivisionError) as exc:
            raise FloatingPointError(
                "non-finite value encountered during mean Tweedie deviance"
            ) from exc
        if not math.isfinite(d):
            raise FloatingPointError(
                "non-finite value encountered during mean Tweedie deviance"
            )
        try:
            term = w[i] * d
        except (OverflowError, ValueError) as exc:
            raise FloatingPointError(
                "non-finite value encountered during mean Tweedie deviance"
            ) from exc
        if not math.isfinite(term):
            raise FloatingPointError(
                "non-finite value encountered during mean Tweedie deviance"
            )
        deviance_terms.append(term)

    try:
        total_deviance = math.fsum(deviance_terms)
    except (OverflowError, ValueError) as exc:
        raise FloatingPointError(
            "non-finite value encountered during mean Tweedie deviance"
        ) from exc
    if not math.isfinite(total_deviance):
        raise FloatingPointError(
            "non-finite value encountered during mean Tweedie deviance"
        )
    try:
        result = total_deviance / total_weight
    except (OverflowError, ValueError, ZeroDivisionError) as exc:
        raise FloatingPointError(
            "non-finite value encountered during mean Tweedie deviance"
        ) from exc
    if not math.isfinite(result):
        raise FloatingPointError(
            "non-finite value encountered during mean Tweedie deviance"
        )
    if result == 0:
        result = 0.0
    return result


def d2_tweedie_score(y_true, y_pred, power=1.5, sample_weight=None) -> float:
    """Return the D^2 Tweedie score: relative improvement over a mean.

    The score is ``1 - D / D0`` where ``D`` is the mean Tweedie
    deviance of ``y_pred`` and ``D0`` is the mean Tweedie deviance of
    the constant baseline ``mu``, the weighted mean of ``y_true``. It
    measures how much the predictions improve on always predicting
    that constant.

    All validation of the four arguments -- containers, lengths,
    exact types, booleans, finiteness, value ranges, the total weight,
    and the ValueError boundaries -- is exactly that of
    ``mean_tweedie_deviance``: that function is first called with the
    same arguments to obtain ``D`` and any exception it raises
    propagates unchanged.

    After validation, the true values and weights (``None`` means
    every sample weighs ``1.0``) are converted to ``float`` in input
    order. ``math.fsum`` computes the total weight ``W = sum(w_i)``;
    if that summation overflows or is invalid, is non-finite, or ``W``
    is less than or equal to zero, a ValueError is raised.
    ``math.fsum`` computes ``sum(w_i * y_true_i)`` and the weighted
    mean ``mu`` is that sum divided by ``W``. When ``mu`` is exactly
    zero the baseline deviance ``D0`` is ``0.0``; otherwise ``D0`` is
    ``mean_tweedie_deviance(y_true, [mu] * len(y_true), power,
    sample_weight)``. When ``D0`` is non-zero the result is
    ``1.0 - D / D0``; when ``D0`` and ``D`` are both zero the result
    is ``1.0``; when only ``D0`` is zero the result is ``0.0``.

    Overflow, invalid operations, or division by zero during the
    post-validation conversion, weighted-mean multiplication,
    summation, division, or final division, and any non-finite
    intermediate value or result raise FloatingPointError. An exact
    zero result is normalized to ``0.0``.

    The return value is a float. The inputs are not modified.
    Deterministic: same inputs, same result.
    """
    model_deviance = mean_tweedie_deviance(
        y_true, y_pred, power, sample_weight
    )

    n = len(y_true)
    try:
        t = [float(value) for value in y_true]
        if sample_weight is None:
            w = [1.0] * n
        else:
            w = [float(value) for value in sample_weight]
    except (OverflowError, ValueError) as exc:
        raise FloatingPointError(
            "non-finite value encountered during d2 Tweedie score"
        ) from exc
    for values in (t, w):
        for value in values:
            if not math.isfinite(value):
                raise FloatingPointError(
                    "non-finite value encountered during d2 Tweedie score"
                )

    try:
        total_weight = math.fsum(w)
    except (OverflowError, ValueError) as exc:
        raise ValueError("the total weight must be greater than 0") from exc
    if not math.isfinite(total_weight) or total_weight <= 0.0:
        raise ValueError("the total weight must be greater than 0")

    weighted_terms = []
    for i in range(n):
        try:
            term = w[i] * t[i]
        except (OverflowError, ValueError) as exc:
            raise FloatingPointError(
                "non-finite value encountered during d2 Tweedie score"
            ) from exc
        if not math.isfinite(term):
            raise FloatingPointError(
                "non-finite value encountered during d2 Tweedie score"
            )
        weighted_terms.append(term)
    try:
        weighted_sum = math.fsum(weighted_terms)
    except (OverflowError, ValueError) as exc:
        raise FloatingPointError(
            "non-finite value encountered during d2 Tweedie score"
        ) from exc
    if not math.isfinite(weighted_sum):
        raise FloatingPointError(
            "non-finite value encountered during d2 Tweedie score"
        )
    try:
        mu = weighted_sum / total_weight
    except (OverflowError, ValueError, ZeroDivisionError) as exc:
        raise FloatingPointError(
            "non-finite value encountered during d2 Tweedie score"
        ) from exc
    if not math.isfinite(mu):
        raise FloatingPointError(
            "non-finite value encountered during d2 Tweedie score"
        )

    if mu == 0.0:
        baseline_deviance = 0.0
    else:
        baseline = [mu] * n
        baseline_deviance = mean_tweedie_deviance(
            y_true, baseline, power, sample_weight
        )

    try:
        if baseline_deviance != 0.0:
            result = 1.0 - model_deviance / baseline_deviance
        elif model_deviance == 0.0:
            result = 1.0
        else:
            result = 0.0
        if not math.isfinite(result):
            raise FloatingPointError(
                "non-finite value encountered during d2 Tweedie score"
            )
    except (OverflowError, ValueError, ZeroDivisionError) as exc:
        raise FloatingPointError(
            "non-finite value encountered during d2 Tweedie score"
        ) from exc

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


def multilabel_confusion_matrix(
    y_true, y_pred, labels=None, sample_weight=None
) -> list:
    """Compute one ``[[TN, FP], [FN, TP]]`` confusion matrix per label.

    ``y_true`` and ``y_pred`` must be non-empty lists of equal length whose
    elements have type exactly ``int`` (booleans are rejected). When
    ``labels`` is ``None`` the labels are the sorted union of the labels
    appearing in either vector, in ascending order; otherwise ``labels``
    must be a non-empty list of distinct values of type exactly ``int``
    (booleans are rejected), kept in the given order, and may include
    labels that never appear or omit labels that do. ``sample_weight``
    must be ``None`` or a list of the same length whose elements are
    finite non-negative values of type exactly ``int`` or ``float``
    (booleans are rejected). Any violation -- including the finiteness
    checks, which raise OverflowError for integers too large to convert
    to float -- raises ValueError.

    For each label ``c`` the matrix is ``[[TN, FP], [FN, TP]]``: TN counts
    samples that are ``c`` on neither side, FP samples predicted ``c`` but
    not truly ``c``, FN samples truly ``c`` but not predicted ``c``, and
    TP samples that are ``c`` on both sides. Without weights the four
    cells hold exact ``int`` counts. With weights each cell sums the
    weights of its samples, converted to float, with ``math.fsum`` in
    sample order; empty cells and exact zero sums are ``+0.0``. Overflow
    or an invalid operation during a weight sum, or a non-finite sum,
    raises FloatingPointError.

    The matrices are returned as a three-level list in label order. The
    inputs are not modified. Deterministic: same inputs, same result.
    """
    n = _check_metric_vectors(y_true, y_pred)
    for value in y_true:
        if type(value) is not int:
            raise ValueError("y_true must contain only integers")
    for value in y_pred:
        if type(value) is not int:
            raise ValueError("y_pred must contain only integers")

    if labels is None:
        label_order = sorted(set(y_true) | set(y_pred))
    else:
        if not isinstance(labels, list) or len(labels) == 0:
            raise ValueError("labels must be a non-empty list of integers")
        label_order = []
        seen = set()
        for value in labels:
            if type(value) is not int or value in seen:
                raise ValueError(
                    "labels must contain only distinct integers"
                )
            seen.add(value)
            label_order.append(value)

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

    result = []
    if weighted:
        # Per-cell weight lists, accumulated in sample order, so each cell
        # is reduced with math.fsum in that exact order.
        for c in label_order:
            terms = [[[] for _ in range(2)] for _ in range(2)]
            for i in range(n):
                terms[y_true[i] == c][y_pred[i] == c].append(
                    float(sample_weight[i])
                )
            matrix = [[0.0, 0.0], [0.0, 0.0]]
            for r in range(2):
                for k in range(2):
                    cell_terms = terms[r][k]
                    if not cell_terms:
                        continue
                    try:
                        total = math.fsum(cell_terms)
                    except (OverflowError, ValueError) as exc:
                        raise FloatingPointError(
                            "non-finite value encountered during "
                            "multilabel confusion matrix"
                        ) from exc
                    if not math.isfinite(total):
                        raise FloatingPointError(
                            "non-finite value encountered during "
                            "multilabel confusion matrix"
                        )
                    # math.fsum of exact zeros returns 0.0; normalize a
                    # negative-zero result to +0.0.
                    matrix[r][k] = total + 0.0
            result.append(matrix)
    else:
        for c in label_order:
            matrix = [[0, 0], [0, 0]]
            for i in range(n):
                matrix[y_true[i] == c][y_pred[i] == c] += 1
            result.append(matrix)
    return result


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


def fbeta_score(
    y_true, y_pred, beta=1.0, average="binary", pos_label=1, zero_division=0
):
    """Compute the F-beta score for integer labels.

    ``y_true`` and ``y_pred`` must be non-empty lists of equal length whose
    elements have type exactly ``int`` (booleans are rejected). The label
    set is the sorted union of the labels appearing in either vector.

    Validation of ``y_true``, ``y_pred``, ``average``, ``pos_label``, and
    ``zero_division`` (including the label order, the binary requirements,
    and the per-class precision/recall counts) is exactly the same as in
    :func:`precision_recall_fscore_support`; any violation raises
    ValueError. ``beta`` must be a value of type exactly ``int`` or
    ``float`` (booleans rejected) that is finite and strictly greater than
    zero; otherwise ValueError.

    Letting ``q = beta * beta`` and using the same per-class precision
    ``P`` and recall ``R`` as :func:`precision_recall_fscore_support`, the
    score is ``F = (1 + q) * P * R / (q * P + R)``; a zero denominator
    yields ``float(zero_division)``. With ``average=None`` the return is a
    list of floats aligned to the sorted labels. ``"binary"`` reports the
    ``pos_label`` class, ``"micro"`` pools the counts first, ``"macro"``
    averages the per-class F values with ``math.fsum`` in sorted-label
    order, and ``"weighted"`` weights them by support and divides by the
    sample count; each of these modes returns a float.

    If any conversion or computation after validation raises
    OverflowError or ValueError, or any intermediate value or result is
    non-finite, FloatingPointError is raised. An exact zero is returned as
    ``0.0``. The inputs are not modified. Deterministic: same inputs, same
    result.
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
    if type(beta) not in (int, float):
        raise ValueError(
            "beta must be a finite non-boolean number greater than 0"
        )
    # math.isfinite raises OverflowError for ints too large to convert to
    # float; such values fail the finite requirement.
    try:
        beta_finite = math.isfinite(beta)
    except OverflowError as exc:
        raise ValueError(
            "beta must be a finite non-boolean number greater than 0"
        ) from exc
    if not beta_finite or beta <= 0:
        raise ValueError(
            "beta must be a finite non-boolean number greater than 0"
        )

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
    support = {label: 0 for label in labels}
    for i in range(n):
        true_label = y_true[i]
        pred_label = y_pred[i]
        support[true_label] += 1
        if true_label == pred_label:
            tp[true_label] += 1
        else:
            fp[pred_label] += 1

    fill = float(zero_division)

    try:
        q = float(beta) * float(beta)
    except (OverflowError, ValueError) as exc:
        raise FloatingPointError(
            "non-finite value encountered during fbeta_score"
        ) from exc
    if not math.isfinite(q):
        raise FloatingPointError(
            "non-finite value encountered during fbeta_score"
        )

    def pr_for(tp_count, fp_count, support_count):
        """Precision and recall from one class's (or pooled) counts."""
        precision_denominator = tp_count + fp_count
        if precision_denominator == 0:
            p = fill
        else:
            p = tp_count / precision_denominator
        if support_count == 0:
            r = fill
        else:
            r = tp_count / support_count
        return p, r

    def fbeta_for(p, r):
        """F-beta from precision and recall:
        ``(1 + q) * P * R / (q * P + R)``."""
        try:
            numerator = (1.0 + q) * p * r
            denominator = q * p + r
        except (OverflowError, ValueError) as exc:
            raise FloatingPointError(
                "non-finite value encountered during fbeta_score"
            ) from exc
        if not math.isfinite(numerator) or not math.isfinite(denominator):
            raise FloatingPointError(
                "non-finite value encountered during fbeta_score"
            )
        if denominator == 0:
            return fill
        try:
            f = numerator / denominator
        except (OverflowError, ValueError, ZeroDivisionError) as exc:
            raise FloatingPointError(
                "non-finite value encountered during fbeta_score"
            ) from exc
        if not math.isfinite(f):
            raise FloatingPointError(
                "non-finite value encountered during fbeta_score"
            )
        if f == 0:
            f = 0.0
        return f

    if average is None:
        return [
            fbeta_for(*pr_for(tp[label], fp[label], support[label]))
            for label in labels
        ]

    if average == "binary":
        return fbeta_for(
            *pr_for(tp[pos_label], fp[pos_label], support[pos_label])
        )

    if average == "micro":
        return fbeta_for(*pr_for(sum(tp.values()), sum(fp.values()), n))

    fscores = [
        fbeta_for(*pr_for(tp[label], fp[label], support[label]))
        for label in labels
    ]

    if average == "macro":
        try:
            result = math.fsum(fscores) / len(labels)
        except (OverflowError, ValueError, ZeroDivisionError) as exc:
            raise FloatingPointError(
                "non-finite value encountered during fbeta_score"
            ) from exc
        if not math.isfinite(result):
            raise FloatingPointError(
                "non-finite value encountered during fbeta_score"
            )
        if result == 0:
            result = 0.0
        return result

    # weighted
    try:
        result = math.fsum(
            fscores[k] * support[labels[k]] for k in range(len(labels))
        ) / n
    except (OverflowError, ValueError, ZeroDivisionError) as exc:
        raise FloatingPointError(
            "non-finite value encountered during fbeta_score"
        ) from exc
    if not math.isfinite(result):
        raise FloatingPointError(
            "non-finite value encountered during fbeta_score"
        )
    if result == 0:
        result = 0.0
    return result


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


def roc_auc_score(
    y_true, y_score, pos_label=1, sample_weight=None, max_fpr=None
) -> float:
    """Return the (optionally partial) area under :func:`roc_curve`.

    Validation of ``y_true``, ``y_score``, ``pos_label`` and
    ``sample_weight`` -- including weighted ROC, tied-score groups, and
    input invariance -- and their exceptions are exactly those of
    :func:`roc_curve`. ``max_fpr`` must be ``None`` or a finite value of
    type exactly ``int`` or ``float`` (booleans are rejected) with
    ``0 < max_fpr <= 1``; any violation raises ValueError, and an
    OverflowError raised by the finiteness check is treated the same
    way.

    With ``max_fpr`` equal to ``None`` or ``1``, the area is the full
    ``math.fsum`` -- in curve order -- of the trapezoid terms
    ``(fpr[k + 1] - fpr[k]) * (tpr[k] + tpr[k + 1]) / 2`` over adjacent
    curve points.

    With ``0 < max_fpr < 1``, the curve is cut at FPR ``max_fpr``: ``k``
    is the first index with ``fpr[k] > max_fpr``, and the TPR at the cut
    is linearly interpolated between points ``k - 1`` and ``k``. The
    interpolation point is appended after the points at indices ``0``
    through ``k - 1`` and their trapezoid areas are summed in order with
    ``math.fsum`` to give ``A``. The result is the standardized partial
    area ``0.5 * (1 + (A - Amin) / (Amax - Amin))`` with
    ``Amin = max_fpr ** 2 / 2`` and ``Amax = max_fpr``.

    An exact zero result is normalized to ``0.0``. After validation,
    interpolation, arithmetic, exponentiation, or ``math.fsum`` raising
    OverflowError, ValueError, or ZeroDivisionError, or producing a
    non-finite value, raises FloatingPointError.

    The return value is a float. The inputs are not modified.
    Deterministic: same inputs, same result.
    """
    fpr, tpr, _ = roc_curve(
        y_true, y_score, pos_label=pos_label, sample_weight=sample_weight
    )
    if max_fpr is not None:
        if type(max_fpr) not in (int, float):
            raise ValueError(
                "max_fpr must be None or a finite non-boolean number "
                "with 0 < max_fpr <= 1"
            )
        try:
            finite = math.isfinite(max_fpr)
        except OverflowError as exc:
            raise ValueError(
                "max_fpr must be None or a finite non-boolean number "
                "with 0 < max_fpr <= 1"
            ) from exc
        if not finite or not (0 < max_fpr <= 1):
            raise ValueError(
                "max_fpr must be None or a finite non-boolean number "
                "with 0 < max_fpr <= 1"
            )

    try:
        terms = []
        if max_fpr is None or max_fpr == 1:
            upper = len(fpr) - 1
        else:
            k = 0
            while fpr[k] <= max_fpr:
                k += 1
            cut_tpr = (
                tpr[k - 1]
                + (tpr[k] - tpr[k - 1])
                * (max_fpr - fpr[k - 1])
                / (fpr[k] - fpr[k - 1])
            )
            if not math.isfinite(cut_tpr):
                raise FloatingPointError(
                    "non-finite value encountered during roc auc "
                    "computation"
                )
            upper = k - 1
        for j in range(upper):
            term = (
                (fpr[j + 1] - fpr[j])
                * (tpr[j] + tpr[j + 1])
                / 2
            )
            if not math.isfinite(term):
                raise FloatingPointError(
                    "non-finite value encountered during roc auc "
                    "computation"
                )
            terms.append(term)
        if max_fpr is not None and max_fpr != 1:
            term = (
                (max_fpr - fpr[k - 1])
                * (tpr[k - 1] + cut_tpr)
                / 2
            )
            if not math.isfinite(term):
                raise FloatingPointError(
                    "non-finite value encountered during roc auc "
                    "computation"
                )
            terms.append(term)
        auc = math.fsum(terms)
        if not math.isfinite(auc):
            raise FloatingPointError(
                "non-finite value encountered during roc auc computation"
            )
        if max_fpr is not None and max_fpr != 1:
            amin = max_fpr ** 2 / 2
            amax = max_fpr
            auc = 0.5 * (1 + (auc - amin) / (amax - amin))
        if not math.isfinite(auc):
            raise FloatingPointError(
                "non-finite value encountered during roc auc computation"
            )
        if auc == 0:
            auc = 0.0
    except (OverflowError, ValueError, ZeroDivisionError) as exc:
        raise FloatingPointError(
            "non-finite value encountered during roc auc computation"
        ) from exc
    return auc


# ``from __future__ import annotations`` stores annotations as strings;
# expose the builtin ``float`` as the runtime return annotation.
roc_auc_score.__annotations__["return"] = float


def multiclass_roc_auc_score(
    y_true, y_score, average="macro", sample_weight=None
) -> float:
    """Return the one-vs-one multiclass ROC AUC, macro or weighted.

    ``y_true`` and ``y_score`` must be non-empty lists of equal length.
    ``y_true`` must contain values of type exactly ``int`` (booleans are
    rejected) drawn from at least three distinct labels; the classes are
    the sorted distinct labels, in ascending order, and each column of
    ``y_score`` corresponds to one class in that order. Every ``y_score``
    row must be a list whose length equals the number of classes and whose
    elements are finite values in the closed interval ``[0, 1]`` with type
    exactly ``int`` or ``float`` (booleans are rejected); the elements of
    each row, in column order, must sum to exactly ``1.0`` under
    ``math.fsum``. ``average`` must be exactly ``"macro"`` or
    ``"weighted"``. ``sample_weight`` must be ``None`` -- every sample
    then weighs ``1.0`` -- or a list of the same length whose elements
    are finite non-negative values of type exactly ``int`` or ``float``
    (booleans are rejected); the total weight of every class must be
    greater than zero. Any violation -- container, length, class, shape,
    type, range, row-sum, average, or per-class-weight checks, including
    an OverflowError raised by a finiteness check -- raises ValueError.

    The classes are taken in ascending order and every unordered pair
    ``(a, b)`` with ``a < b`` is considered in that order. For each pair,
    only the samples whose true class is ``a`` or ``b`` are retained and
    :func:`roc_auc_score` is called twice on the retained samples: once
    with the class-``a`` score column and ``pos_label=a``, and once with
    the class-``b`` score column and ``pos_label=b``; the pair value is
    the mean of the two AUCs. With ``"macro"`` averaging, the result is
    the mean of the pair values, accumulated in pair order with
    ``math.fsum``. With ``"weighted"`` averaging, each pair is weighted
    by the sum of the weights of its two classes and the result is the
    ``math.fsum`` -- in pair order -- of the weighted pair values divided
    by the ``math.fsum`` -- in the same order -- of the pair weights; an
    exact zero result is normalized to ``0.0``.

    An OverflowError, ValueError, or ZeroDivisionError raised during the
    computation, or any non-finite intermediate value or result, raises
    FloatingPointError. The return value is a float. The inputs are not
    modified. Deterministic: same inputs, same result.
    """
    if not isinstance(y_true, list) or not isinstance(y_score, list):
        raise ValueError("y_true and y_score must be lists")
    if len(y_true) == 0 or len(y_score) == 0:
        raise ValueError("y_true and y_score must be non-empty lists")
    if len(y_true) != len(y_score):
        raise ValueError("y_true and y_score must have the same length")
    n = len(y_true)

    for value in y_true:
        if type(value) is not int:
            raise ValueError("y_true must contain only integers")
    classes = sorted(set(y_true))
    if len(classes) < 3:
        raise ValueError(
            "y_true must contain at least three distinct classes"
        )
    n_classes = len(classes)
    class_index = {label: j for j, label in enumerate(classes)}

    for row in y_score:
        if not isinstance(row, list):
            raise ValueError("y_score rows must be lists")
        if len(row) != n_classes:
            raise ValueError(
                "each y_score row must have one entry per class"
            )
        for value in row:
            if type(value) not in (int, float):
                raise ValueError(
                    "y_score must contain only finite numbers in [0, 1]"
                )
            # math.isfinite raises OverflowError for ints too large to
            # convert to float; such values fail the finite requirement.
            try:
                finite = math.isfinite(value)
            except OverflowError as exc:
                raise ValueError(
                    "y_score must contain only finite numbers in [0, 1]"
                ) from exc
            if not finite or value < 0 or value > 1:
                raise ValueError(
                    "y_score must contain only finite numbers in [0, 1]"
                )
        # The columns of each row must describe a probability distribution.
        try:
            row_sum = math.fsum(row)
        except OverflowError as exc:
            raise ValueError(
                "each y_score row must sum to exactly 1.0"
            ) from exc
        if row_sum != 1.0:
            raise ValueError("each y_score row must sum to exactly 1.0")

    if average not in ("macro", "weighted"):
        raise ValueError('average must be "macro" or "weighted"')

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
        try:
            weights = [float(value) for value in sample_weight]
        except (OverflowError, ValueError) as exc:
            raise ValueError(
                "sample_weight must contain only finite non-negative "
                "non-boolean numbers"
            ) from exc
        for value in weights:
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    "sample_weight must contain only finite non-negative "
                    "non-boolean numbers"
                )

    # Each class needs a strictly positive weight, otherwise the
    # one-vs-one AUC for a pair touching it is undefined.
    class_weight_terms = [[] for _ in range(n_classes)]
    for i in range(n):
        class_weight_terms[class_index[y_true[i]]].append(weights[i])
    class_weights = []
    for terms in class_weight_terms:
        try:
            total = math.fsum(terms)
        except (OverflowError, ValueError) as exc:
            raise ValueError(
                "the total weight of each class must be greater than 0"
            ) from exc
        if not math.isfinite(total) or total <= 0.0:
            raise ValueError(
                "the total weight of each class must be greater than 0"
            )
        class_weights.append(total)

    pair_values = []
    pair_weights = []
    for p in range(n_classes):
        for q in range(p + 1, n_classes):
            label_a = classes[p]
            label_b = classes[q]
            pair_true = []
            score_a = []
            score_b = []
            pair_weights_local = []
            for i in range(n):
                if y_true[i] == label_a or y_true[i] == label_b:
                    pair_true.append(y_true[i])
                    score_a.append(y_score[i][p])
                    score_b.append(y_score[i][q])
                    pair_weights_local.append(weights[i])
            try:
                auc_a = roc_auc_score(
                    pair_true,
                    score_a,
                    pos_label=label_a,
                    sample_weight=pair_weights_local,
                )
                auc_b = roc_auc_score(
                    pair_true,
                    score_b,
                    pos_label=label_b,
                    sample_weight=pair_weights_local,
                )
                if not math.isfinite(auc_a) or not math.isfinite(auc_b):
                    raise FloatingPointError(
                        "non-finite value encountered during multiclass-"
                        "roc-auc computation"
                    )
                pair_value = (auc_a + auc_b) / 2
                if not math.isfinite(pair_value):
                    raise FloatingPointError(
                        "non-finite value encountered during multiclass-"
                        "roc-auc computation"
                    )
                if average == "macro":
                    pair_values.append(pair_value)
                else:
                    pair_weight = class_weights[p] + class_weights[q]
                    if not math.isfinite(pair_weight):
                        raise FloatingPointError(
                            "non-finite value encountered during "
                            "multiclass-roc-auc computation"
                        )
                    pair_values.append(pair_value * pair_weight)
                    pair_weights.append(pair_weight)
            except (OverflowError, ValueError, ZeroDivisionError) as exc:
                raise FloatingPointError(
                    "non-finite value encountered during multiclass-roc-"
                    "auc computation"
                ) from exc

    try:
        if average == "macro":
            result = math.fsum(pair_values) / len(pair_values)
        else:
            numerator = math.fsum(pair_values)
            denominator = math.fsum(pair_weights)
            result = numerator / denominator
        if not math.isfinite(result):
            raise FloatingPointError(
                "non-finite value encountered during multiclass-roc-auc "
                "computation"
            )
        if result == 0:
            result = 0.0
    except (OverflowError, ValueError, ZeroDivisionError) as exc:
        raise FloatingPointError(
            "non-finite value encountered during multiclass-roc-auc "
            "computation"
        ) from exc
    return result


# ``from __future__ import annotations`` stores annotations as strings;
# expose the builtin ``float`` as the runtime return annotation.
multiclass_roc_auc_score.__annotations__["return"] = float


def _det_float(value):
    """Convert a validated score/weight to float; overflow, invalid
    operations, and non-finite results raise FloatingPointError."""
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise FloatingPointError(
            "non-finite value encountered during det computation"
        ) from exc
    if not math.isfinite(result):
        raise FloatingPointError(
            "non-finite value encountered during det computation"
        )
    return result


def det_curve(y_true, y_score, pos_label=1, sample_weight=None) -> tuple:
    """Compute false positive/false negative rates at ascending thresholds.

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

    The thresholds are the distinct score values in ascending order, each
    converted to ``float``; samples sharing a score always form one
    threshold group. At a threshold ``t`` every sample whose score is
    greater than or equal to ``t`` is judged positive: ``FP`` is the
    ``math.fsum`` -- taken in input order -- of the weights of the
    negative-class samples judged positive, and ``FN`` is the analogous
    sum for positive-class samples judged negative (score below ``t``).
    ``FPR`` is ``FP`` divided by the total negative-class weight and
    ``FNR`` is ``FN`` divided by the total positive-class weight. An exact
    zero rate is normalized to ``0.0``. Overflow, invalid operations, or
    non-finite intermediate values after validation raise
    FloatingPointError.

    The return value is ``(fpr, fnr, thresholds)``, three equally long
    lists of floats. The inputs are not modified. Deterministic: same
    inputs, same result.
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
        weights = [_det_float(value) for value in sample_weight]

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
            "non-finite value encountered during det computation"
        ) from exc
    if not math.isfinite(pos_total) or not math.isfinite(neg_total):
        raise FloatingPointError(
            "non-finite value encountered during det computation"
        )
    if pos_total <= 0.0 or neg_total <= 0.0:
        raise ValueError(
            "the total weight of each class must be greater than 0"
        )

    thresholds = [_det_float(value) for value in sorted(set(y_score))]

    fprs = []
    fnrs = []
    for threshold in thresholds:
        fp_terms = []
        fn_terms = []
        for i in range(n):
            if y_score[i] >= threshold:
                if y_true[i] != pos_label:
                    fp_terms.append(weights[i])
            elif y_true[i] == pos_label:
                fn_terms.append(weights[i])
        try:
            fp = math.fsum(fp_terms)
            fn = math.fsum(fn_terms)
            fpr_value = fp / neg_total
            fnr_value = fn / pos_total
        except (OverflowError, ValueError, ZeroDivisionError) as exc:
            raise FloatingPointError(
                "non-finite value encountered during det computation"
            ) from exc
        if not math.isfinite(fpr_value) or not math.isfinite(fnr_value):
            raise FloatingPointError(
                "non-finite value encountered during det computation"
            )
        if fpr_value == 0:
            fpr_value = 0.0
        if fnr_value == 0:
            fnr_value = 0.0
        fprs.append(fpr_value)
        fnrs.append(fnr_value)
    return fprs, fnrs, thresholds


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


def multiclass_average_precision_score(
    y_true, y_score, average="macro", sample_weight=None
):
    """Return the one-vs-rest multiclass average precision.

    ``y_true`` and ``y_score`` must be non-empty lists of equal length.
    ``y_true`` must contain values of type exactly ``int`` (booleans are
    rejected) drawn from at least three distinct labels; the classes are
    the sorted distinct labels, in ascending order, and each column of
    ``y_score`` corresponds to one class in that order. Every ``y_score``
    row must be a list whose length equals the number of classes and whose
    elements are finite values in the closed interval ``[0, 1]`` with type
    exactly ``int`` or ``float`` (booleans are rejected); the elements of
    each row, in column order, must sum to exactly ``1.0`` under
    ``math.fsum``. ``average`` must be exactly ``None``, ``"macro"``, or
    ``"weighted"``. ``sample_weight`` must be ``None`` -- every sample
    then weighs ``1.0`` -- or a list of the same length whose elements
    are finite non-negative values of type exactly ``int`` or ``float``
    (booleans are rejected); the total weight of every class, accumulated
    in input order with ``math.fsum``, must be finite and greater than
    zero. Any violation -- container, length, class, shape, type, range,
    row-sum, average, or per-class-weight checks, including an
    OverflowError raised by a finiteness check -- raises ValueError.

    For each class, in ascending class order, a binary label vector is
    built with ``1`` at samples whose true class is that class and ``0``
    elsewhere; :func:`average_precision_score` is then called on that
    vector, the class's score column, ``pos_label=1``, and the same
    sample weights. With ``average`` ``None`` the return is the list of
    per-class average precisions in class order. With ``"macro"`` the
    result is the ``math.fsum`` of the per-class values divided by the
    number of classes. With ``"weighted"`` the result is the
    ``math.fsum`` -- in class order -- of each per-class value times its
    class weight, divided by the ``math.fsum`` of the class weights; an
    exact zero result is normalized to ``0.0``. The two averaging modes
    return a float.

    An OverflowError, ValueError, ZeroDivisionError, or ArithmeticError
    raised during post-validation conversion, the per-class calls, or the
    multiplications, divisions, or ``math.fsum`` steps, as well as any
    non-finite intermediate value or result, raises FloatingPointError.
    The inputs are not modified. Deterministic: same inputs, same result.
    """
    if not isinstance(y_true, list) or not isinstance(y_score, list):
        raise ValueError("y_true and y_score must be lists")
    if len(y_true) == 0 or len(y_score) == 0:
        raise ValueError("y_true and y_score must be non-empty lists")
    if len(y_true) != len(y_score):
        raise ValueError("y_true and y_score must have the same length")
    n = len(y_true)

    for value in y_true:
        if type(value) is not int:
            raise ValueError("y_true must contain only integers")
    classes = sorted(set(y_true))
    if len(classes) < 3:
        raise ValueError(
            "y_true must contain at least three distinct classes"
        )
    n_classes = len(classes)
    class_index = {label: j for j, label in enumerate(classes)}

    for row in y_score:
        if not isinstance(row, list):
            raise ValueError("y_score rows must be lists")
        if len(row) != n_classes:
            raise ValueError(
                "each y_score row must have one entry per class"
            )
        for value in row:
            if type(value) not in (int, float):
                raise ValueError(
                    "y_score must contain only finite numbers in [0, 1]"
                )
            # math.isfinite raises OverflowError for ints too large to
            # convert to float; such values fail the finite requirement.
            try:
                finite = math.isfinite(value)
            except OverflowError as exc:
                raise ValueError(
                    "y_score must contain only finite numbers in [0, 1]"
                ) from exc
            if not finite or value < 0 or value > 1:
                raise ValueError(
                    "y_score must contain only finite numbers in [0, 1]"
                )
        # The columns of each row must describe a probability distribution.
        try:
            row_sum = math.fsum(row)
        except OverflowError as exc:
            raise ValueError(
                "each y_score row must sum to exactly 1.0"
            ) from exc
        if row_sum != 1.0:
            raise ValueError("each y_score row must sum to exactly 1.0")

    if average not in (None, "macro", "weighted"):
        raise ValueError('average must be None, "macro", or "weighted"')

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
        try:
            weights = [float(value) for value in sample_weight]
        except (OverflowError, ValueError) as exc:
            raise FloatingPointError(
                "non-finite value encountered during multiclass-average-"
                "precision computation"
            ) from exc
        for value in weights:
            if not math.isfinite(value) or value < 0.0:
                raise FloatingPointError(
                    "non-finite value encountered during multiclass-average-"
                    "precision computation"
                )

    # Every class needs a strictly positive weight, otherwise its
    # one-vs-rest average precision is undefined.
    class_weight_terms = [[] for _ in range(n_classes)]
    for i in range(n):
        class_weight_terms[class_index[y_true[i]]].append(weights[i])
    class_weights = []
    for terms in class_weight_terms:
        try:
            total = math.fsum(terms)
        except (OverflowError, ValueError) as exc:
            raise ValueError(
                "the total weight of each class must be greater than 0"
            ) from exc
        if not math.isfinite(total) or total <= 0.0:
            raise ValueError(
                "the total weight of each class must be greater than 0"
            )
        class_weights.append(total)

    def fail(exc):
        return FloatingPointError(
            "non-finite value encountered during multiclass-average-"
            "precision computation"
        )

    scores = []
    for j in range(n_classes):
        binary_true = [1 if y_true[i] == classes[j] else 0 for i in range(n)]
        class_scores = [y_score[i][j] for i in range(n)]
        try:
            ap = average_precision_score(
                binary_true,
                class_scores,
                pos_label=1,
                sample_weight=weights,
            )
        except (OverflowError, ValueError, ZeroDivisionError) as exc:
            raise fail(exc) from exc
        if not math.isfinite(ap):
            raise fail(None)
        scores.append(ap)

    if average is None:
        return scores

    try:
        if average == "macro":
            result = math.fsum(scores) / n_classes
        else:
            numerator = math.fsum(
                scores[j] * class_weights[j] for j in range(n_classes)
            )
            denominator = math.fsum(class_weights)
            result = numerator / denominator
        if not math.isfinite(result):
            raise fail(None)
        if result == 0:
            result = 0.0
    except (OverflowError, ValueError, ZeroDivisionError) as exc:
        raise fail(exc) from exc
    return result


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


def expected_calibration_error(
    y_true, y_prob, n_bins=5, pos_label=1, sample_weight=None
):
    """Compute the weighted expected calibration error.

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
    Within a bin, samples are accumulated in input order with
    ``math.fsum`` to obtain the total weight ``W``, the positive-class
    weight ``T``, and the weighted probability sum ``P`` (the
    ``math.fsum`` of the ``w * p`` products). Bins with ``W == 0`` are
    omitted; for each retained bin the gap is ``abs(T / W - P / W)``.
    Bins are visited in ascending bin index and their ``W * gap``
    contributions are accumulated with ``math.fsum``; the result is that
    sum divided by the total weight of all samples, with an exact zero
    normalized to ``0.0``. Overflow, invalid operations during
    post-validation conversion or arithmetic, division by zero, and
    non-finite intermediate values or results raise FloatingPointError.

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
    except (OverflowError, ValueError, ZeroDivisionError) as exc:
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
    # unbounded) while still letting bins be visited in index order.
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

    contributions = []
    for bin_index in sorted(bins):
        w_terms, pos_terms, wp_terms = bins[bin_index]
        try:
            bin_weight = math.fsum(w_terms)
            pos_weight = math.fsum(pos_terms)
            prob_weight = math.fsum(wp_terms)
        except (OverflowError, ValueError, ZeroDivisionError) as exc:
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
            gap = abs(fraction - mean_prob)
            contribution = bin_weight * gap
        except (
            OverflowError,
            ValueError,
            ZeroDivisionError,
        ) as exc:
            raise FloatingPointError(
                "non-finite value encountered during calibration "
                "computation"
            ) from exc
        if (
            not math.isfinite(fraction)
            or not math.isfinite(mean_prob)
            or not math.isfinite(gap)
            or not math.isfinite(contribution)
        ):
            raise FloatingPointError(
                "non-finite value encountered during calibration "
                "computation"
            )
        contributions.append(contribution)

    try:
        weighted_gap = math.fsum(contributions)
        result = weighted_gap / total_weight
    except (OverflowError, ValueError, ZeroDivisionError) as exc:
        raise FloatingPointError(
            "non-finite value encountered during calibration computation"
        ) from exc
    if not math.isfinite(weighted_gap) or not math.isfinite(result):
        raise FloatingPointError(
            "non-finite value encountered during calibration computation"
        )
    if result == 0:
        result = 0.0
    return result


def adaptive_calibration_error(
    y_true, y_prob, n_bins=5, pos_label=1, sample_weight=None
) -> float:
    """Compute the weighted expected calibration error with adaptive bins.

    ``y_true`` and ``y_prob`` must be non-empty lists of equal length.
    ``y_true`` must contain values of type exactly ``int`` (booleans are
    rejected) drawn from exactly two distinct labels, and ``pos_label``
    must be an exact ``int`` equal to one of them; the other label is the
    negative class. ``y_prob`` must contain finite values in the closed
    interval ``[0, 1]`` whose type is exactly ``int`` or ``float``
    (booleans are rejected; the only accepted ``int`` values are ``0``
    and ``1``). ``n_bins`` must be a positive exact ``int`` not greater
    than the number of samples. ``sample_weight`` must be ``None`` --
    every sample then weighs ``1.0`` -- or a list of the same length
    whose elements are finite non-negative values of type exactly
    ``int`` or ``float`` (booleans are rejected). The total weight must
    be greater than zero. Any violation (including overflow during the
    finiteness checks) raises ValueError.

    Samples are ordered by the ascending key ``(y_prob[i], i)``, so
    equal probabilities are separated by their original index. Bin ``r``
    (``r`` starting at zero) holds the sorted positions in the
    half-open range
    ``[floor(r * n / n_bins), floor((r + 1) * n / n_bins))``; samples
    sharing a probability at a boundary are thus split according to
    their original indices. Within each bin the samples are visited in
    that sorted order and accumulated with ``math.fsum`` to obtain the
    total weight ``W``, the positive-class weight ``T``, and the
    weighted probability sum ``P`` (the ``math.fsum`` of the ``w * p``
    products). Bins with ``W == 0`` are omitted; each retained bin
    contributes ``W * abs(T / W - P / W)``. The contributions are
    summed with ``math.fsum`` in ascending bin order and divided by the
    total weight of all samples, with an exact zero normalized to
    ``0.0``. Overflow, invalid operations during post-validation
    conversion or arithmetic, division by zero, and non-finite
    intermediate values or results raise FloatingPointError.

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

    if type(n_bins) is not int or n_bins < 1:
        raise ValueError("n_bins must be a positive integer")
    if n_bins > n:
        raise ValueError(
            "n_bins must not exceed the number of samples"
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
        try:
            weights = [_cal_float(value) for value in sample_weight]
        except ZeroDivisionError as exc:
            raise FloatingPointError(
                "non-finite value encountered during calibration "
                "computation"
            ) from exc

    try:
        total_weight = math.fsum(weights)
    except (OverflowError, ValueError, ZeroDivisionError) as exc:
        raise FloatingPointError(
            "non-finite value encountered during calibration computation"
        ) from exc
    if not math.isfinite(total_weight):
        raise FloatingPointError(
            "non-finite value encountered during calibration computation"
        )
    if total_weight <= 0.0:
        raise ValueError("the total weight must be greater than 0")

    # Ascending (probability, original index) order; the index tiebreaker
    # is deterministic and splits equal-probability samples across bin
    # boundaries by their original indices.
    order = sorted(range(n), key=lambda i: (y_prob[i], i))

    contributions = []
    for r in range(n_bins):
        start = (r * n) // n_bins
        end = ((r + 1) * n) // n_bins
        w_terms = []
        pos_terms = []
        wp_terms = []
        for position in range(start, end):
            i = order[position]
            p = y_prob[i]
            w = weights[i]
            try:
                wp = w * p
            except (
                OverflowError,
                ValueError,
                ZeroDivisionError,
            ) as exc:
                raise FloatingPointError(
                    "non-finite value encountered during calibration "
                    "computation"
                ) from exc
            if isinstance(wp, float) and not math.isfinite(wp):
                raise FloatingPointError(
                    "non-finite value encountered during calibration "
                    "computation"
                )
            w_terms.append(w)
            if y_true[i] == pos_label:
                pos_terms.append(w)
            wp_terms.append(wp)

        try:
            bin_weight = math.fsum(w_terms)
            pos_weight = math.fsum(pos_terms)
            prob_weight = math.fsum(wp_terms)
        except (OverflowError, ValueError, ZeroDivisionError) as exc:
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
            gap = abs(fraction - mean_prob)
            contribution = bin_weight * gap
        except (
            OverflowError,
            ValueError,
            ZeroDivisionError,
        ) as exc:
            raise FloatingPointError(
                "non-finite value encountered during calibration "
                "computation"
            ) from exc
        if (
            not math.isfinite(fraction)
            or not math.isfinite(mean_prob)
            or not math.isfinite(gap)
            or not math.isfinite(contribution)
        ):
            raise FloatingPointError(
                "non-finite value encountered during calibration "
                "computation"
            )
        contributions.append(contribution)

    try:
        weighted_gap = math.fsum(contributions)
        result = weighted_gap / total_weight
    except (OverflowError, ValueError, ZeroDivisionError) as exc:
        raise FloatingPointError(
            "non-finite value encountered during calibration computation"
        ) from exc
    if not math.isfinite(weighted_gap) or not math.isfinite(result):
        raise FloatingPointError(
            "non-finite value encountered during calibration computation"
        )
    if result == 0:
        result = 0.0
    return result


def maximum_calibration_error(
    y_true, y_prob, n_bins=5, pos_label=1, sample_weight=None
) -> float:
    """Compute the maximum per-bin calibration gap.

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
    Bins with ``W == 0`` are omitted; each retained bin has the gap
    ``abs(T / W - P / W)``, and the return value is the largest such
    gap, with an exact zero normalized to ``0.0``. Overflow, invalid
    operations during post-validation conversion or arithmetic, division
    by zero, and non-finite intermediate values or results raise
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
        try:
            weights = [_cal_float(value) for value in sample_weight]
        except ZeroDivisionError as exc:
            raise FloatingPointError(
                "non-finite value encountered during calibration "
                "computation"
            ) from exc

    try:
        total_weight = math.fsum(weights)
    except (OverflowError, ValueError, ZeroDivisionError) as exc:
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
    # unbounded) while still letting bins be visited in index order.
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
        except (OverflowError, ValueError, ZeroDivisionError) as exc:
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

    maximum = None
    for bin_index in sorted(bins):
        w_terms, pos_terms, wp_terms = bins[bin_index]
        try:
            bin_weight = math.fsum(w_terms)
            pos_weight = math.fsum(pos_terms)
            prob_weight = math.fsum(wp_terms)
        except (OverflowError, ValueError, ZeroDivisionError) as exc:
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
            gap = abs(fraction - mean_prob)
        except (
            OverflowError,
            ValueError,
            ZeroDivisionError,
        ) as exc:
            raise FloatingPointError(
                "non-finite value encountered during calibration "
                "computation"
            ) from exc
        if (
            not math.isfinite(fraction)
            or not math.isfinite(mean_prob)
            or not math.isfinite(gap)
        ):
            raise FloatingPointError(
                "non-finite value encountered during calibration "
                "computation"
            )
        if maximum is None or gap > maximum:
            maximum = gap

    # The total weight is positive, so at least one bin was retained.
    result = float(maximum)
    if not math.isfinite(result):
        raise FloatingPointError(
            "non-finite value encountered during calibration computation"
        )
    if result == 0:
        result = 0.0
    return result


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


def brier_score_decomposition(
    y_true, y_prob, n_bins=5, pos_label=1, sample_weight=None
) -> tuple:
    """Decompose the weighted mean squared probability error.

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

    After validation the probabilities and weights are converted to
    ``float``. Letting ``F`` denote ``math.fsum``, in input order the
    total weight is ``W = F(w)`` and the positive-class weight is
    ``T = F(w of positive samples)``, with overall positive rate
    ``o = T / W``. Equal-width bins are indexed
    ``min(int(p * n_bins), n_bins - 1)``. Bins are visited in ascending
    bin index; within a bin, samples are accumulated in input order with
    ``F`` to obtain the bin total weight ``Wb``, the bin positive-class
    weight ``Tb``, and the bin weighted probability sum
    ``Pb = F(w * p)``. Bins with ``Wb == 0`` are omitted; for each
    retained bin ``ob = Tb / Wb`` and ``pb = Pb / Wb``. In ascending bin
    index order the reliability is
    ``R = F(Wb * (pb - ob) ** 2) / W``, the resolution is
    ``S = F(Wb * (ob - o) ** 2) / W``, and the uncertainty is
    ``U = o * (1 - o)``. An exact zero is normalized to ``0.0``.
    Overflow, invalid operations during post-validation conversion or
    arithmetic, division by zero, and non-finite intermediate values or
    results raise FloatingPointError.

    The return value is ``(reliability, resolution, uncertainty)``,
    three floats. The inputs are not modified. Deterministic: same
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

    probabilities = [_cal_float(value) for value in y_prob]

    try:
        total_weight = math.fsum(weights)
        positive_weight = math.fsum(
            weights[i] for i in range(n) if y_true[i] == pos_label
        )
    except (
        OverflowError,
        ValueError,
        ZeroDivisionError,
    ) as exc:
        raise FloatingPointError(
            "non-finite value encountered during calibration computation"
        ) from exc
    if (
        not math.isfinite(total_weight)
        or not math.isfinite(positive_weight)
    ):
        raise FloatingPointError(
            "non-finite value encountered during calibration computation"
        )
    if total_weight <= 0.0:
        raise ValueError("the total weight must be greater than 0")

    try:
        overall_rate = positive_weight / total_weight
    except (
        OverflowError,
        ValueError,
        ZeroDivisionError,
    ) as exc:
        raise FloatingPointError(
            "non-finite value encountered during calibration computation"
        ) from exc
    if not math.isfinite(overall_rate):
        raise FloatingPointError(
            "non-finite value encountered during calibration computation"
        )

    # Each occupied bin maps to its W, T, and w*p term lists, all kept in
    # input order. A dict avoids allocating n_bins lists (n_bins is
    # unbounded) while still letting bins be visited in index order.
    bins = {}
    last_bin = n_bins - 1
    for i in range(n):
        p = probabilities[i]
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
        except (OverflowError, ValueError, ZeroDivisionError) as exc:
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

    reliability_terms = []
    resolution_terms = []
    for bin_index in sorted(bins):
        w_terms, pos_terms, wp_terms = bins[bin_index]
        try:
            bin_weight = math.fsum(w_terms)
            bin_positive_weight = math.fsum(pos_terms)
            bin_prob_weight = math.fsum(wp_terms)
        except (
            OverflowError,
            ValueError,
            ZeroDivisionError,
        ) as exc:
            raise FloatingPointError(
                "non-finite value encountered during calibration "
                "computation"
            ) from exc
        if (
            not math.isfinite(bin_weight)
            or not math.isfinite(bin_positive_weight)
            or not math.isfinite(bin_prob_weight)
        ):
            raise FloatingPointError(
                "non-finite value encountered during calibration "
                "computation"
            )
        if bin_weight == 0.0:
            continue
        try:
            bin_rate = bin_positive_weight / bin_weight
            bin_mean_prob = bin_prob_weight / bin_weight
            reliability_gap = bin_mean_prob - bin_rate
            resolution_gap = bin_rate - overall_rate
            reliability_term = bin_weight * reliability_gap ** 2
            resolution_term = bin_weight * resolution_gap ** 2
        except (
            OverflowError,
            ValueError,
            ZeroDivisionError,
        ) as exc:
            raise FloatingPointError(
                "non-finite value encountered during calibration "
                "computation"
            ) from exc
        if (
            not math.isfinite(bin_rate)
            or not math.isfinite(bin_mean_prob)
            or not math.isfinite(reliability_gap)
            or not math.isfinite(resolution_gap)
            or not math.isfinite(reliability_term)
            or not math.isfinite(resolution_term)
        ):
            raise FloatingPointError(
                "non-finite value encountered during calibration "
                "computation"
            )
        reliability_terms.append(reliability_term)
        resolution_terms.append(resolution_term)

    try:
        reliability = math.fsum(reliability_terms) / total_weight
        resolution = math.fsum(resolution_terms) / total_weight
        uncertainty = overall_rate * (1.0 - overall_rate)
    except (
        OverflowError,
        ValueError,
        ZeroDivisionError,
    ) as exc:
        raise FloatingPointError(
            "non-finite value encountered during calibration computation"
        ) from exc
    if (
        not math.isfinite(reliability)
        or not math.isfinite(resolution)
        or not math.isfinite(uncertainty)
    ):
        raise FloatingPointError(
            "non-finite value encountered during calibration computation"
        )
    if reliability == 0:
        reliability = 0.0
    if resolution == 0:
        resolution = 0.0
    if uncertainty == 0:
        uncertainty = 0.0
    return reliability, resolution, uncertainty


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


def _multiclass_brier_float(value):
    """Convert a validated probability/weight to float; overflow, invalid
    operations, and non-finite results raise FloatingPointError."""
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise FloatingPointError(
            "non-finite value encountered during multiclass-brier-score "
            "computation"
        ) from exc
    if not math.isfinite(result):
        raise FloatingPointError(
            "non-finite value encountered during multiclass-brier-score "
            "computation"
        )
    return result


def multiclass_brier_score_loss(y_true, y_prob, sample_weight=None) -> float:
    """Compute the weighted multiclass Brier score loss.

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
    (booleans are rejected). Any violation -- container, length, class,
    shape, type, range, row-sum, or finiteness checks, including overflow
    during those checks -- raises ValueError.

    After validation, the probabilities and weights are converted to
    ``float``. For each sample and each column, ``t`` is ``1.0`` when the
    label equals that column's class and ``0.0`` otherwise, and the
    per-sample squared error ``q_i`` is the ``math.fsum``, in column
    order, of ``(p - t) ** 2``. The result is the quotient of two
    ``math.fsum`` sums accumulated in sample order -- one of
    ``w_i * q_i``, the other of the ``w_i`` -- with an exact zero
    normalized to ``0.0``. The column sum is not divided by the number of
    classes. Overflow or an invalid operation during the weight
    summation, a non-finite total weight, or a total weight that is not
    greater than zero raises ValueError. Overflow or invalid operations
    during any other conversion, subtraction, squaring, multiplication,
    summation, or division, and non-finite intermediate values or
    results, raise FloatingPointError.

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
        weights = [_multiclass_brier_float(value) for value in sample_weight]

    # The weights have just been converted to float, so summing them is
    # still part of the input checks: overflow, invalid operations, a
    # non-finite total, and a non-positive total all raise ValueError.
    try:
        total_weight = math.fsum(weights)
    except (OverflowError, ValueError) as exc:
        raise ValueError("the total weight must be greater than 0") from exc
    if not math.isfinite(total_weight) or total_weight <= 0.0:
        raise ValueError("the total weight must be greater than 0")

    error_terms = []
    for i in range(n):
        j = class_index[y_true[i]]
        column_terms = []
        for k in range(n_classes):
            p = _multiclass_brier_float(y_prob[i][k])
            t = 1.0 if k == j else 0.0
            try:
                residual = p - t
                term = residual ** 2
            except (OverflowError, ValueError) as exc:
                raise FloatingPointError(
                    "non-finite value encountered during multiclass-brier-"
                    "score computation"
                ) from exc
            if not math.isfinite(term):
                raise FloatingPointError(
                    "non-finite value encountered during multiclass-brier-"
                    "score computation"
                )
            column_terms.append(term)
        try:
            q_i = math.fsum(column_terms)
            weighted = weights[i] * q_i
        except (OverflowError, ValueError) as exc:
            raise FloatingPointError(
                "non-finite value encountered during multiclass-brier-score "
                "computation"
            ) from exc
        if not math.isfinite(q_i) or not math.isfinite(weighted):
            raise FloatingPointError(
                "non-finite value encountered during multiclass-brier-score "
                "computation"
            )
        error_terms.append(weighted)

    try:
        total_error = math.fsum(error_terms)
        loss = total_error / total_weight
    except (OverflowError, ValueError, ZeroDivisionError) as exc:
        raise FloatingPointError(
            "non-finite value encountered during multiclass-brier-score "
            "computation"
        ) from exc
    if not math.isfinite(total_error) or not math.isfinite(loss):
        raise FloatingPointError(
            "non-finite value encountered during multiclass-brier-score "
            "computation"
        )
    if loss == 0:
        loss = 0.0
    return loss


def _multiclass_hinge_loss_float(value):
    """Convert a validated score/weight to float; overflow, invalid
    operations, and non-finite results raise FloatingPointError."""
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise FloatingPointError(
            "non-finite value encountered during multiclass-hinge-loss "
            "computation"
        ) from exc
    if not math.isfinite(result):
        raise FloatingPointError(
            "non-finite value encountered during multiclass-hinge-loss "
            "computation"
        )
    return result


def _hinge_loss_float(value):
    """Convert a validated score/weight to float; overflow, invalid
    operations, and non-finite results raise FloatingPointError."""
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise FloatingPointError(
            "non-finite value encountered during hinge loss computation"
        ) from exc
    if not math.isfinite(result):
        raise FloatingPointError(
            "non-finite value encountered during hinge loss computation"
        )
    return result


def hinge_loss(y_true, y_pred, sample_weight=None) -> float:
    """Compute the weighted binary hinge loss.

    ``y_true`` and ``y_pred`` must be non-empty lists of equal length.
    Every element of ``y_true`` must have type exactly ``int`` (booleans
    are rejected) and equal ``1`` or ``-1``. Every element of ``y_pred``
    must be a finite value of type exactly ``int`` or ``float``
    (booleans are rejected). ``sample_weight`` must be ``None`` -- every
    sample then weighs ``1.0`` -- or a list of the same length whose
    elements are finite non-negative values of type exactly ``int`` or
    ``float`` (booleans are rejected). Any violation -- container,
    length, type, range, or finiteness checks, including overflow during
    those checks -- raises ValueError.

    After validation, the scores and weights are converted to ``float``
    in input order and the converted weights are summed with
    ``math.fsum``; overflow or an invalid operation during that
    summation, a non-finite total, or a total that is not greater than
    zero raises ValueError. For each sample, in input order,
    ``z_i = 1.0 - y_true_i * score_i``, ``loss_i = max(0.0, z_i)``, and
    ``term_i = weight_i * loss_i``. The result is
    ``math.fsum(term_i) / W`` with both sums accumulated in input
    order; an exact zero is normalized to positive ``0.0``. Overflow or
    invalid operations during the conversion, multiplication,
    subtraction, maximum, term multiplication, loss summation, or
    division, and non-finite intermediate values or results, raise
    FloatingPointError.

    The return value is a float. The inputs are not modified.
    Deterministic: same inputs, same result.
    """
    n = _check_metric_vectors(y_true, y_pred)
    for value in y_true:
        if type(value) is not int or value not in (-1, 1):
            raise ValueError("y_true must contain only the integers -1 and 1")
    for value in y_pred:
        if type(value) not in (int, float):
            raise ValueError(
                "y_pred must contain only finite non-boolean numbers"
            )
        # math.isfinite raises OverflowError for ints too large to
        # convert to float; such values fail the finite requirement.
        try:
            finite = math.isfinite(value)
        except OverflowError as exc:
            raise ValueError(
                "y_pred must contain only finite non-boolean numbers"
            ) from exc
        if not finite:
            raise ValueError(
                "y_pred must contain only finite non-boolean numbers"
            )

    has_weights = sample_weight is not None
    if has_weights:
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

    # After validation, the scores and weights are converted to float
    # in input order (scores first, then weights); a conversion failure
    # or a non-finite converted value raises FloatingPointError.
    scores = [_hinge_loss_float(value) for value in y_pred]
    if has_weights:
        weights = [_hinge_loss_float(value) for value in sample_weight]
    else:
        weights = [1.0] * n

    # The weights have just been converted to float, so summing them is
    # still part of the input checks: overflow, invalid operations, a
    # non-finite total, and a non-positive total all raise ValueError.
    try:
        total_weight = math.fsum(weights)
    except (OverflowError, ValueError) as exc:
        raise ValueError("the total weight must be greater than 0") from exc
    if not math.isfinite(total_weight) or total_weight <= 0.0:
        raise ValueError("the total weight must be greater than 0")

    terms = []
    for i in range(n):
        try:
            z = 1.0 - y_true[i] * scores[i]
            loss = max(0.0, z)
            term = weights[i] * loss
        except (OverflowError, ValueError) as exc:
            raise FloatingPointError(
                "non-finite value encountered during hinge loss computation"
            ) from exc
        if (
            not math.isfinite(z)
            or not math.isfinite(loss)
            or not math.isfinite(term)
        ):
            raise FloatingPointError(
                "non-finite value encountered during hinge loss computation"
            )
        terms.append(term)

    try:
        total_loss = math.fsum(terms)
        result = total_loss / total_weight
    except (OverflowError, ValueError, ZeroDivisionError) as exc:
        raise FloatingPointError(
            "non-finite value encountered during hinge loss computation"
        ) from exc
    if not math.isfinite(total_loss) or not math.isfinite(result):
        raise FloatingPointError(
            "non-finite value encountered during hinge loss computation"
        )
    if result == 0:
        result = 0.0
    return result


def _logit_loss_float(value):
    """Convert a validated score/weight to float; overflow, invalid
    operations, and non-finite results raise FloatingPointError."""
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise FloatingPointError(
            "non-finite value encountered during logit-loss computation"
        ) from exc
    if not math.isfinite(result):
        raise FloatingPointError(
            "non-finite value encountered during logit-loss computation"
        )
    return result


def logit_loss(y_true, y_score, pos_label=1, sample_weight=None) -> float:
    """Compute the weighted binary logistic (cross-entropy) loss of scores.

    ``y_true`` and ``y_score`` must be non-empty lists of equal length.
    ``y_true`` must contain values of type exactly ``int`` (booleans are
    rejected) drawn from exactly two distinct labels, and ``pos_label``
    must be an exact ``int`` equal to one of them. Every element of
    ``y_score`` must be a finite value of type exactly ``int`` or
    ``float`` (booleans are rejected). ``sample_weight`` must be
    ``None`` -- every sample then weighs ``1.0`` -- or a list of the same
    length whose elements are finite non-negative values of type exactly
    ``int`` or ``float`` (booleans are rejected). Any violation --
    container, length, type, label, range, or finiteness checks,
    including overflow during those checks -- raises ValueError.

    After validation, the scores and weights are converted to ``float``
    in input order (scores first, then weights) and the converted
    weights are summed with ``math.fsum``; overflow or an invalid operation during that
    summation, a non-finite total, or a total that is not greater than
    zero raises ValueError. For each sample, in input order, ``t_i`` is
    ``1.0`` when the label equals ``pos_label`` and ``0.0`` otherwise,
    ``z_i`` is the score, and the per-sample loss is
    ``max(z_i, 0.0) - t_i * z_i + math.log1p(math.exp(-abs(z_i)))``.
    The result is ``math.fsum(w_i * loss_i) / W`` with both sums
    accumulated in input order; an exact zero is normalized to positive
    ``0.0``. Overflow or invalid operations during the conversion,
    multiplication, maximum, absolute value, exponent, logarithm,
    addition, summation, or division, and non-finite intermediate values
    or results, raise FloatingPointError.

    The return value is a float. The inputs are not modified.
    Deterministic: same inputs, same result.
    """
    if not isinstance(y_true, list) or not isinstance(y_score, list):
        raise ValueError("y_true and y_score must be lists")
    if len(y_true) == 0 or len(y_score) == 0:
        raise ValueError("y_true and y_score must be non-empty lists")
    if len(y_true) != len(y_score):
        raise ValueError("y_true and y_score must have the same length")
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

    for value in y_score:
        if type(value) not in (int, float):
            raise ValueError(
                "y_score must contain only finite non-boolean numbers"
            )
        # math.isfinite raises OverflowError for ints too large to
        # convert to float; such values fail the finite requirement.
        try:
            finite = math.isfinite(value)
        except OverflowError as exc:
            raise ValueError(
                "y_score must contain only finite non-boolean numbers"
            ) from exc
        if not finite:
            raise ValueError(
                "y_score must contain only finite non-boolean numbers"
            )

    has_weights = sample_weight is not None
    if has_weights:
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

    # After validation, the scores and weights are converted to float
    # in input order (scores first, then weights); a conversion failure
    # or a non-finite converted value raises FloatingPointError.
    scores = [_logit_loss_float(value) for value in y_score]
    if has_weights:
        weights = [_logit_loss_float(value) for value in sample_weight]
    else:
        weights = [1.0] * n

    # The weights have just been converted to float, so summing them is
    # still part of the input checks: overflow, invalid operations, a
    # non-finite total, and a non-positive total all raise ValueError.
    try:
        total_weight = math.fsum(weights)
    except (OverflowError, ValueError) as exc:
        raise ValueError("the total weight must be greater than 0") from exc
    if not math.isfinite(total_weight) or total_weight <= 0.0:
        raise ValueError("the total weight must be greater than 0")

    terms = []
    for i in range(n):
        z = scores[i]
        t = 1.0 if y_true[i] == pos_label else 0.0
        try:
            sample_loss = (
                max(z, 0.0)
                - t * z
                + math.log1p(math.exp(-abs(z)))
            )
            term = weights[i] * sample_loss
        except (OverflowError, ValueError) as exc:
            raise FloatingPointError(
                "non-finite value encountered during logit-loss computation"
            ) from exc
        if not math.isfinite(sample_loss) or not math.isfinite(term):
            raise FloatingPointError(
                "non-finite value encountered during logit-loss computation"
            )
        terms.append(term)

    try:
        total_loss = math.fsum(terms)
        result = total_loss / total_weight
    except (OverflowError, ValueError, ZeroDivisionError) as exc:
        raise FloatingPointError(
            "non-finite value encountered during logit-loss computation"
        ) from exc
    if not math.isfinite(total_loss) or not math.isfinite(result):
        raise FloatingPointError(
            "non-finite value encountered during logit-loss computation"
        )
    if result == 0:
        result = 0.0
    return result


def multiclass_hinge_loss(y_true, y_score, sample_weight=None) -> float:
    """Compute the weighted multiclass Crammer-Singer hinge loss.

    ``y_true`` and ``y_score`` must be non-empty lists of equal length.
    ``y_true`` must contain values of type exactly ``int`` (booleans are
    rejected) drawn from at least two distinct labels; the classes are the
    sorted distinct labels, in ascending order, and each column of
    ``y_score`` corresponds to one class in that order. Every ``y_score``
    row must be a list whose length equals the number of classes and whose
    elements are finite values with type exactly ``int`` or ``float``
    (booleans are rejected). ``sample_weight`` must be ``None`` -- every
    sample then weighs ``1.0`` -- or a list of the same length whose
    elements are finite non-negative values of type exactly ``int`` or
    ``float`` (booleans are rejected). The total weight must be greater
    than zero. Any violation -- container, length, shape, type, range,
    total-weight, or finiteness checks, including overflow during those
    checks -- raises ValueError.

    After validation, the scores and weights are converted to ``float``
    and the converted weights are summed with ``math.fsum`` in input
    order; overflow or an invalid operation during that summation, a
    non-finite total, or a total that is not greater than zero raises
    ValueError. For each sample, in input order, ``c`` is the score in
    the true-class column, ``r`` is the largest score among the other
    columns taken in ascending class order, and the per-sample loss is
    ``max(0.0, 1.0 + r - c)``. The result is the quotient of two
    ``math.fsum`` sums, both accumulated in input order -- one of
    ``w_i * loss_i``, the other of the ``w_i`` -- with an exact zero
    normalized to ``0.0``. Overflow or invalid operations during the
    subsequent conversion, addition, subtraction, multiplication,
    summation, or division, and non-finite intermediate values or
    results, raise FloatingPointError.

    The return value is a float. The inputs are not modified.
    Deterministic: same inputs, same result.
    """
    if not isinstance(y_true, list) or not isinstance(y_score, list):
        raise ValueError("y_true and y_score must be lists")
    if len(y_true) == 0 or len(y_score) == 0:
        raise ValueError("y_true and y_score must be non-empty lists")
    if len(y_true) != len(y_score):
        raise ValueError("y_true and y_score must have the same length")
    n = len(y_true)

    for value in y_true:
        if type(value) is not int:
            raise ValueError("y_true must contain only integers")
    classes = sorted(set(y_true))
    if len(classes) < 2:
        raise ValueError("y_true must contain at least two distinct classes")
    n_classes = len(classes)
    class_index = {label: j for j, label in enumerate(classes)}

    for row in y_score:
        if not isinstance(row, list):
            raise ValueError("y_score rows must be lists")
        if len(row) != n_classes:
            raise ValueError(
                "each y_score row must have one entry per class"
            )
        for value in row:
            if type(value) not in (int, float):
                raise ValueError(
                    "y_score must contain only finite non-boolean numbers"
                )
            # math.isfinite raises OverflowError for ints too large to
            # convert to float; such values fail the finite requirement.
            try:
                finite = math.isfinite(value)
            except OverflowError as exc:
                raise ValueError(
                    "y_score must contain only finite non-boolean numbers"
                ) from exc
            if not finite:
                raise ValueError(
                    "y_score must contain only finite non-boolean numbers"
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
        weights = [_multiclass_hinge_loss_float(value) for value in sample_weight]

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
        true_j = class_index[y_true[i]]
        c = _multiclass_hinge_loss_float(y_score[i][true_j])
        r = None
        for j in range(n_classes):
            if j == true_j:
                continue
            value = _multiclass_hinge_loss_float(y_score[i][j])
            if r is None or value > r:
                r = value
        try:
            margin = 1.0 + r - c
            sample_loss = margin if margin > 0.0 else 0.0
            term = weights[i] * sample_loss
        except (OverflowError, ValueError) as exc:
            raise FloatingPointError(
                "non-finite value encountered during multiclass-hinge-loss "
                "computation"
            ) from exc
        if (
            not math.isfinite(r)
            or not math.isfinite(c)
            or not math.isfinite(margin)
            or not math.isfinite(sample_loss)
            or not math.isfinite(term)
        ):
            raise FloatingPointError(
                "non-finite value encountered during multiclass-hinge-loss "
                "computation"
            )
        loss_terms.append(term)

    try:
        total_loss = math.fsum(loss_terms)
        loss = total_loss / total_weight
    except (OverflowError, ValueError) as exc:
        raise FloatingPointError(
            "non-finite value encountered during multiclass-hinge-loss "
            "computation"
        ) from exc
    if not math.isfinite(total_loss) or not math.isfinite(loss):
        raise FloatingPointError(
            "non-finite value encountered during multiclass-hinge-loss "
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


def silhouette_samples(X, labels) -> "list[float]":
    """Compute the silhouette coefficient of each sample.

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
    The coefficients are returned as a new ``float`` list in input
    order, with an exact zero written as positive ``0.0``. Overflow,
    invalid operations, or non-finite intermediate values after
    validation raise FloatingPointError.

    The inputs are not modified. Deterministic: same inputs, same
    result.
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
            "non-finite value encountered during silhouette samples"
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
                        "samples"
                    ) from exc
                if isinstance(diff, float) and not math.isfinite(diff):
                    fail()
                try:
                    square = diff ** 2
                except (OverflowError, ValueError) as exc:
                    raise FloatingPointError(
                        "non-finite value encountered during silhouette "
                        "samples"
                    ) from exc
                if isinstance(square, float) and not math.isfinite(square):
                    fail()
                terms.append(square)
            try:
                total = math.fsum(terms)
            except (OverflowError, ValueError) as exc:
                raise FloatingPointError(
                    "non-finite value encountered during silhouette samples"
                ) from exc
            if not math.isfinite(total):
                fail()
            try:
                distance = math.sqrt(total)
            except (OverflowError, ValueError) as exc:
                raise FloatingPointError(
                    "non-finite value encountered during silhouette samples"
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
                "non-finite value encountered during silhouette samples"
            ) from exc
        if not math.isfinite(total):
            fail()
        try:
            mean = total / count
        except (OverflowError, ValueError) as exc:
            raise FloatingPointError(
                "non-finite value encountered during silhouette samples"
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
                "non-finite value encountered during silhouette samples"
            ) from exc
        if not math.isfinite(s_i):
            fail()
        s_values.append(s_i)

    return [0.0 if value == 0 else float(value) for value in s_values]


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


def fowlkes_mallows_score(labels_true, labels_pred):
    """Return the Fowlkes-Mallows index of two integer label partitions.

    Both arguments must be non-empty lists of equal length whose elements
    are exactly ``int`` (booleans are rejected). The inputs are not
    modified. Deterministic: same inputs, same result.

    Rows and columns of the contingency table correspond to the distinct
    labels of ``labels_true`` and ``labels_pred`` respectively, each in
    ascending label order; contingency counts ``n_ij`` accumulate by
    sample index, with row sums ``a_i`` and column sums ``b_j`` taken in
    that order. With ``C(x) = x * (x - 1) // 2``, ``TP = sum C(n_ij)``,
    ``P = sum C(a_i)``, and ``Q = sum C(b_j)`` are accumulated exactly as
    Python integers in that order. When ``P * Q == 0`` the result is
    ``0.0``; otherwise the result is ``TP / math.sqrt(P * Q)`` as a
    ``float``, with an exact zero normalized to ``0.0``. If ``math.sqrt``,
    the integer-to-float conversion, or the division raises
    ``OverflowError``/``ValueError``, or if the denominator or the result
    is not finite, ``FloatingPointError`` is raised.
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

    tp = sum(comb2(value) for row in counts for value in row)
    p = sum(comb2(value) for value in row_sums)
    q = sum(comb2(value) for value in col_sums)

    if p * q == 0:
        return 0.0
    try:
        denominator = math.sqrt(p * q)
        result = tp / denominator
    except (OverflowError, ValueError) as exc:
        raise FloatingPointError(
            "fowlkes_mallows_score result is not representable"
        ) from exc
    if not math.isfinite(denominator) or not math.isfinite(result):
        raise FloatingPointError(
            "fowlkes_mallows_score result is not finite"
        )
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


def adjusted_mutual_info_score(labels_true, labels_pred):
    """Return the adjusted mutual information of two integer partitions.

    Both arguments must be non-empty lists of equal length whose elements
    are exactly ``int`` (booleans are rejected). The inputs are not
    modified. Deterministic: same inputs, same result.

    Rows and columns of the contingency table correspond to the distinct
    labels of ``labels_true`` and ``labels_pred`` respectively, each in
    ascending label order; contingency counts ``n_ij`` accumulate by
    sample index, with row sums ``a_i``, column sums ``b_j``, and ``n``
    the sample count. The mutual information ``MI`` and the entropies
    ``H_true`` and ``H_pred`` are computed exactly as in
    ``normalized_mutual_info_score``. The expected mutual information is
    the ``math.fsum``, over rows ``i``, columns ``j``, and integers ``q``
    from ``max(1, a_i + b_j - n)`` to ``min(a_i, b_j)`` (each in ascending
    order), of ``float(P) * (q / n) * math.log((n * q) / (a_i * b_j))``
    where ``P = Fraction(comb(a_i, q) * comb(n - a_i, b_j - q),
    comb(n, b_j))``. With ``d = (H_true + H_pred) / 2 - EMI`` the result
    is ``1.0`` when ``d == 0`` and ``(MI - EMI) / d`` otherwise. The
    return value is a float; an exact zero result is normalized to
    ``0.0``. Overflow, invalid operations, or non-finite intermediate
    values or results in the post-validation ``Fraction``-to-float
    conversion, multiplication, division, ``math.log``, or ``math.fsum``
    steps raise FloatingPointError.
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
            "non-finite value encountered during adjusted mutual "
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

    emi_terms = []
    for i in range(len(row_labels)):
        a_i = row_sums[i]
        for j in range(len(col_labels)):
            b_j = col_sums[j]
            for q in range(max(1, a_i + b_j - n), min(a_i, b_j) + 1):
                try:
                    probability = Fraction(
                        math.comb(a_i, q) * math.comb(n - a_i, b_j - q),
                        math.comb(n, b_j),
                    )
                    p_float = float(probability)
                    log_value = math.log((n * q) / (a_i * b_j))
                    term = p_float * (q / n) * log_value
                except (OverflowError, ValueError) as exc:
                    raise non_finite() from exc
                checked(p_float)
                checked(log_value)
                emi_terms.append(checked(term))
    try:
        emi = math.fsum(emi_terms)
    except (OverflowError, ValueError) as exc:
        raise non_finite() from exc
    checked(emi)

    try:
        d = (h_true + h_pred) / 2 - emi
    except (OverflowError, ValueError) as exc:
        raise non_finite() from exc
    checked(d)
    if d == 0:
        return 1.0
    try:
        numerator = mi - emi
        result = numerator / d
    except (OverflowError, ValueError) as exc:
        raise non_finite() from exc
    checked(numerator)
    checked(result)
    if result == 0:
        return 0.0
    return result


def homogeneity_completeness_v_measure(labels_true, labels_pred, beta=1.0):
    """Return the homogeneity, completeness, and V-measure of two partitions.

    Both label arguments must be non-empty lists of equal length whose
    elements are exactly ``int`` (booleans are rejected). ``beta`` must be
    a finite, positive, non-boolean number of type exactly ``int`` or
    ``float``. The inputs are not modified. Deterministic: same inputs,
    same result.

    Rows and columns of the contingency table correspond to the distinct
    labels of ``labels_true`` and ``labels_pred`` respectively, each in
    ascending label order; contingency counts ``n_ij`` accumulate by
    sample index, with row sums ``a_i``, column sums ``b_j``, and ``n``
    the sample count. The entropies are
    ``H_t = -math.fsum((a_i / n) * math.log(a_i / n))`` and
    ``H_p = -math.fsum((b_j / n) * math.log(b_j / n))``; the mutual
    information is the ``math.fsum``, row by row then column by column,
    of ``(n_ij / n) * math.log((n_ij * n) / (a_i * b_j))`` over the cells
    with ``n_ij > 0``. Homogeneity is ``1.0`` when ``H_t == 0`` and
    ``MI / H_t`` otherwise; completeness is ``1.0`` when ``H_p == 0`` and
    ``MI / H_p`` otherwise. With ``d = beta * h + c`` the V-measure is
    ``0.0`` when ``d == 0`` and ``(1 + beta) * h * c / d`` otherwise. All
    three results are floats; exact zeros are normalized to ``0.0``.
    Overflow, invalid operations, or non-finite values during the
    post-validation float conversion, multiplication, division,
    ``math.log``, or ``math.fsum`` steps raise FloatingPointError.
    """
    n = _check_metric_vectors(labels_true, labels_pred)
    for value in labels_true:
        if type(value) is not int:
            raise ValueError("labels_true must contain only integers")
    for value in labels_pred:
        if type(value) is not int:
            raise ValueError("labels_pred must contain only integers")
    if type(beta) not in (int, float) or isinstance(beta, bool):
        raise ValueError("beta must be a non-boolean int or float")
    try:
        if not math.isfinite(beta) or beta <= 0:
            raise ValueError("beta must be a finite positive number")
    except OverflowError as exc:
        raise ValueError("beta must be a finite positive number") from exc

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

    def fp_error(exc=None):
        error = FloatingPointError(
            "non-finite value encountered during "
            "homogeneity/completeness/V-measure"
        )
        if exc is not None:
            raise error from exc
        raise error

    def checked(value):
        if not math.isfinite(value):
            fp_error()
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
                fp_error(exc)
            checked(ratio)
            checked(log_value)
            mi_terms.append(checked(term))
    try:
        mi = math.fsum(mi_terms)
    except (OverflowError, ValueError) as exc:
        fp_error(exc)
    checked(mi)

    def entropy(sums):
        """``-math.fsum((s / n) * math.log(s / n))`` over ``sums``."""
        terms = []
        for s in sums:
            try:
                p = s / n
                term = p * math.log(p)
            except (OverflowError, ValueError) as exc:
                fp_error(exc)
            terms.append(checked(term))
        try:
            total = math.fsum(terms)
        except (OverflowError, ValueError) as exc:
            fp_error(exc)
        return checked(-total)

    h_t = entropy(row_sums)
    h_p = entropy(col_sums)

    if h_t == 0.0:
        h = 1.0
    else:
        try:
            h = mi / h_t
        except (OverflowError, ValueError, ZeroDivisionError) as exc:
            fp_error(exc)
        checked(h)
    if h_p == 0.0:
        c = 1.0
    else:
        try:
            c = mi / h_p
        except (OverflowError, ValueError, ZeroDivisionError) as exc:
            fp_error(exc)
        checked(c)

    try:
        beta_f = float(beta)
        d = beta_f * h + c
    except (OverflowError, ValueError) as exc:
        fp_error(exc)
    checked(beta_f)
    checked(d)
    if d == 0.0:
        v = 0.0
    else:
        try:
            v = ((1.0 + beta_f) * h * c) / d
        except (OverflowError, ValueError, ZeroDivisionError) as exc:
            fp_error(exc)
        checked(v)

    if h == 0.0:
        h = 0.0
    if c == 0.0:
        c = 0.0
    if v == 0.0:
        v = 0.0
    return (h, c, v)


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


def _geometric_mean_float(value):
    """Convert a validated number to float; overflow, invalid operations,
    and non-finite results raise FloatingPointError."""
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise FloatingPointError(
            "non-finite value encountered during geometric mean"
        ) from exc
    if not math.isfinite(result):
        raise FloatingPointError(
            "non-finite value encountered during geometric mean"
        )
    return result


def geometric_mean_score(
    y_true, y_pred, correction=0.0, sample_weight=None
) -> float:
    """Return the geometric mean of the per-class recalls.

    Both ``y_true`` and ``y_pred`` must be non-empty lists of equal
    length whose elements are exactly ``int`` (booleans are rejected),
    and ``y_true`` must hold at least two distinct classes.
    ``correction`` must be a finite value of type exactly ``int`` or
    ``float`` (booleans are rejected) with ``0 <= correction <= 1``.
    ``sample_weight`` must be ``None`` -- every sample then weighs
    ``1.0`` -- or a list of the same length whose elements are finite
    non-negative values of type exactly ``int`` or ``float`` (booleans
    are rejected). Any container, length, type, range, or finiteness
    violation (including ``OverflowError`` raised by the finiteness
    checks) raises ValueError. The inputs are not modified.

    The classes ``C`` are the distinct values of ``y_true`` in ascending
    order; labels appearing only in ``y_pred`` merely count as
    misclassifications. After validation the weights and ``correction``
    are converted to ``float``; in class order, and in sample order
    within each class, ``math.fsum`` computes for each class the total
    weight ``W`` of its samples and the weight ``T`` of its correctly
    predicted samples. A class with ``W <= 0`` raises ValueError. With
    ``r = T / W`` per class, a zero ``r`` is replaced by ``correction``;
    if any ``r`` is still zero the result is positive ``0.0``. Otherwise
    the result is ``math.exp`` of the mean -- in class order, via
    ``math.fsum`` -- of ``math.log(r)`` over the classes. Overflow,
    invalid operations, or division by zero during the post-validation
    conversion, summation, division, logarithm, or exponentiation, and
    non-finite intermediate values or results, raise
    FloatingPointError.

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

    classes = sorted(set(y_true))
    if len(classes) < 2:
        raise ValueError("y_true must contain at least two classes")
    index = {label: k for k, label in enumerate(classes)}
    k = len(classes)

    if type(correction) not in (int, float):
        raise ValueError(
            "correction must be a finite non-boolean number in [0, 1]"
        )
    # math.isfinite raises OverflowError for ints too large to convert
    # to float; such values fail the finite requirement.
    try:
        finite = math.isfinite(correction)
    except OverflowError as exc:
        raise ValueError(
            "correction must be a finite non-boolean number in [0, 1]"
        ) from exc
    if not finite or not (0 <= correction <= 1):
        raise ValueError(
            "correction must be a finite non-boolean number in [0, 1]"
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
        weights = [
            _geometric_mean_float(value) for value in sample_weight
        ]

    correction_value = _geometric_mean_float(correction)

    true_terms = [[] for _ in range(k)]
    correct_terms = [[] for _ in range(k)]
    for i in range(n):
        j = index[y_true[i]]
        true_terms[j].append(weights[i])
        if y_pred[i] == y_true[i]:
            correct_terms[j].append(weights[i])

    def non_finite(exc):
        return FloatingPointError(
            "non-finite value encountered during geometric mean"
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
        except (OverflowError, ValueError, ZeroDivisionError) as exc:
            raise non_finite(exc) from exc
        if not math.isfinite(recall):
            raise non_finite(None)
        if recall == 0:
            recall = correction_value
        recalls.append(recall)

    for recall in recalls:
        if recall == 0:
            return 0.0

    log_terms = []
    for recall in recalls:
        try:
            log_value = math.log(recall)
        except (OverflowError, ValueError) as exc:
            raise non_finite(exc) from exc
        if not math.isfinite(log_value):
            raise non_finite(None)
        log_terms.append(log_value)

    try:
        result = math.exp(math.fsum(log_terms) / k)
    except (OverflowError, ValueError, ZeroDivisionError) as exc:
        raise non_finite(exc) from exc
    if not math.isfinite(result):
        raise non_finite(None)
    if result == 0:
        return 0.0
    return result


# ``from __future__ import annotations`` stores annotations as strings;
# expose the builtin ``float`` as the runtime return annotation.
geometric_mean_score.__annotations__["return"] = float


def top_k_accuracy_score(y_true, y_score, k=2, sample_weight=None) -> float:
    """Return the weighted fraction of samples whose true class is among
    the ``k`` highest-scoring columns.

    ``y_true`` must be a non-empty list whose elements are exactly
    ``int`` (booleans are rejected) and which holds at least two
    distinct classes. The classes are the distinct values of ``y_true``
    in ascending order; they correspond to the columns of ``y_score``.
    ``y_score`` must be a list of the same length whose rows are lists
    with one entry per class and whose elements are finite values of
    type exactly ``int`` or ``float`` (booleans are rejected). ``k``
    must be exactly ``int`` with ``1 <= k <= number of classes``.
    ``sample_weight`` must be ``None`` -- every sample then weighs
    ``1.0`` -- or a list of the same length whose elements are finite
    non-negative values of type exactly ``int`` or ``float`` (booleans
    are rejected). Any container, length, shape, type, range, or
    finiteness violation (including ``OverflowError`` raised by
    ``math.isfinite``) raises ValueError.

    For each sample the column indices are ordered by ascending
    ``(-score, class)`` -- ties in score therefore favor the smaller
    class -- and the sample is a hit when the column of its true label
    is among the first ``k`` entries. After validation the weights are
    converted to ``float``; in input order, ``math.fsum`` computes the
    total weight ``W`` and the hit weight ``C``. Overflow, invalid
    operations, or a non-finite value during the ``W`` summation, and
    ``W <= 0``, raise ValueError; overflow, invalid operations, and
    non-finite intermediate values or results during the post-validation
    conversion, the ``C`` summation, and the division raise
    FloatingPointError. The result is ``C / W``; an exact zero result is
    normalized to ``0.0``.

    The return value is a float. The inputs are not modified.
    Deterministic: same inputs, same result.
    """
    if not isinstance(y_true, list) or len(y_true) == 0:
        raise ValueError("y_true must be a non-empty list")
    for value in y_true:
        if type(value) is not int:
            raise ValueError("y_true must contain only integers")
    classes = sorted(set(y_true))
    n_classes = len(classes)
    if n_classes < 2:
        raise ValueError("y_true must contain at least two classes")
    n = len(y_true)

    if not isinstance(y_score, list) or len(y_score) != n:
        raise ValueError(
            "y_score must be a list with the same length as y_true"
        )
    for row in y_score:
        if not isinstance(row, list) or len(row) != n_classes:
            raise ValueError(
                "y_score rows must be lists with one entry per class"
            )
        for value in row:
            if type(value) not in (int, float):
                raise ValueError(
                    "y_score must contain only finite non-boolean numbers"
                )
            # math.isfinite raises OverflowError for ints too large to
            # convert to float; such values fail the finite requirement.
            try:
                finite = math.isfinite(value)
            except OverflowError as exc:
                raise ValueError(
                    "y_score must contain only finite non-boolean numbers"
                ) from exc
            if not finite:
                raise ValueError(
                    "y_score must contain only finite non-boolean numbers"
                )

    if type(k) is not int:
        raise ValueError("k must be an integer")
    if k < 1 or k > n_classes:
        raise ValueError("k must be between 1 and the number of classes")

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

    def non_finite(exc):
        return FloatingPointError(
            "non-finite value encountered during top-k accuracy"
        )

    if sample_weight is None:
        weights = [1.0] * n
    else:
        try:
            weights = [float(value) for value in sample_weight]
        except (OverflowError, ValueError) as exc:
            raise non_finite(exc) from exc
        for value in weights:
            if not math.isfinite(value):
                raise non_finite(None)

    try:
        total_weight = math.fsum(weights)
    except (OverflowError, ValueError) as exc:
        raise ValueError("the total weight must be greater than 0") from exc
    if not math.isfinite(total_weight) or total_weight <= 0.0:
        raise ValueError("the total weight must be greater than 0")

    index = {label: j for j, label in enumerate(classes)}
    hit_terms = []
    for i in range(n):
        row = y_score[i]
        order = sorted(
            range(n_classes), key=lambda j: (-row[j], classes[j])
        )
        position = [0] * n_classes
        for rank, j in enumerate(order):
            position[j] = rank
        if position[index[y_true[i]]] < k:
            hit_terms.append(weights[i])

    try:
        hit_weight = math.fsum(hit_terms)
    except (OverflowError, ValueError) as exc:
        raise non_finite(exc) from exc
    if not math.isfinite(hit_weight):
        raise non_finite(None)
    try:
        result = hit_weight / total_weight
    except (OverflowError, ValueError, ZeroDivisionError) as exc:
        raise non_finite(exc) from exc
    if not math.isfinite(result):
        raise non_finite(None)
    if result == 0:
        return 0.0
    return result


def ndcg_score(y_true, y_score, k=None, gain="exponential", sample_weight=None) -> float:
    """Return the normalized discounted cumulative gain at ``k``.

    Two input shapes are supported.

    One-dimensional form: ``y_true`` is a non-empty list whose
    elements are not lists; ``y_true`` and ``y_score`` must be
    non-empty lists of equal length. ``y_true`` elements must be
    finite non-negative values of type exactly ``int`` or ``float``
    (booleans are rejected); ``y_score`` elements must be finite
    values of type exactly ``int`` or ``float`` (booleans are
    rejected). ``k`` must be ``None`` or an exact ``int`` (booleans
    are rejected); with ``None`` the list length ``m`` is used,
    otherwise ``1 <= k <= m`` is required. The newly added
    ``gain`` and ``sample_weight`` parameters must keep their
    default values in this form or ValueError is raised; with the
    defaults the behavior is identical to the one-vector metric.

    Two-dimensional form: ``y_true`` must be a non-empty rectangular
    list of rows with at least two columns; its elements must be
    finite non-negative values of type exactly ``int`` or ``float``
    (booleans are rejected). ``y_score`` must be a list of the same
    shape whose elements are finite values of type exactly ``int`` or
    ``float`` (booleans are rejected). With ``k`` ``None`` the number
    of columns ``m`` is used; otherwise ``k`` must be an exact ``int``
    (booleans are rejected) with ``1 <= k <= m``. ``gain`` must be
    ``"linear"`` or ``"exponential"``. ``sample_weight`` must be
    ``None`` -- every row then weighs ``1.0`` -- or a list with the
    same length as the number of rows whose elements are finite
    non-negative values of type exactly ``int`` or ``float``
    (booleans are rejected). Any container, shape, length, type,
    sign, range, or finiteness violation (including ``OverflowError``
    raised by ``math.isfinite``) raises ValueError.

    For row ``i`` the predicted order is the first ``k`` indices of
    ``range(m)`` sorted ascending by ``(-y_score[i][j], j)``; the
    ideal order is obtained the same way using ``(-y_true[i][j], j)``,
    so ties favor the smaller index. For rank ``r`` starting at zero
    the gain is ``g(x) = x`` for the linear variant and
    ``g(x) = 2.0 ** x - 1.0`` for the exponential variant, and an
    order's discounted gain is
    ``D = math.fsum(g(y_true[i][j]) / math.log2(r + 2))`` over the
    retained columns in ascending rank. The row value is ``0.0`` when
    the ideal discounted gain is zero and ``D(predicted) /
    D(ideal)`` otherwise. After validation the weights are converted
    to ``float`` in row order; ``math.fsum`` computes the total
    weight ``W = sum(w_i)`` and the weighted total
    ``S = sum(w_i * row_value_i)`` in row order, and the result is
    ``S / W``. Overflow or invalid operations during the ``W``
    summation, a non-finite ``W``, or ``W <= 0`` raise ValueError;
    overflow, invalid operations, and non-finite values during the
    post-validation conversion, the sorting, the gain and discount
    computation, the ``math.fsum`` summations, the weighted
    multiplication, the row ratios, or the final division raise
    FloatingPointError. An exact zero result is normalized to ``0.0``.

    The return value is a float. The inputs are not modified.
    Deterministic: same inputs, same result.
    """
    if not isinstance(y_true, list) or not isinstance(y_score, list):
        raise ValueError("y_true and y_score must be lists")
    if len(y_true) == 0 or len(y_score) == 0:
        raise ValueError("y_true and y_score must be non-empty lists")

    two_dimensional = all(isinstance(row, list) for row in y_true)

    if not two_dimensional:
        # The new parameters are not available in the one-dimensional
        # form; with the defaults the historical behavior is preserved.
        if gain != "exponential" or sample_weight is not None:
            raise ValueError(
                "gain and sample_weight require two-dimensional inputs"
            )
        m = len(y_true)
        if len(y_score) != m:
            raise ValueError("y_true and y_score must have the same length")
        for value in y_true:
            if type(value) not in (int, float):
                raise ValueError(
                    "y_true must contain only finite non-negative "
                    "non-boolean numbers"
                )
            # math.isfinite raises OverflowError for ints too large to
            # convert to float; such values fail the finite requirement.
            try:
                finite = math.isfinite(value)
            except OverflowError as exc:
                raise ValueError(
                    "y_true must contain only finite non-negative "
                    "non-boolean numbers"
                ) from exc
            if not finite or value < 0:
                raise ValueError(
                    "y_true must contain only finite non-negative "
                    "non-boolean numbers"
                )
        for value in y_score:
            if type(value) not in (int, float):
                raise ValueError(
                    "y_score must contain only finite non-boolean numbers"
                )
            # math.isfinite raises OverflowError for ints too large to
            # convert to float; such values fail the finite requirement.
            try:
                finite = math.isfinite(value)
            except OverflowError as exc:
                raise ValueError(
                    "y_score must contain only finite non-boolean numbers"
                ) from exc
            if not finite:
                raise ValueError(
                    "y_score must contain only finite non-boolean numbers"
                )

        if k is None:
            k = m
        else:
            if type(k) is not int:
                raise ValueError("k must be None or an integer")
            if k < 1 or k > m:
                raise ValueError(
                    "k must be between 1 and the length of the vectors"
                )

        predicted = sorted(range(m), key=lambda j: (-y_score[j], j))[:k]
        ideal = sorted(range(m), key=lambda j: (-y_true[j], j))[:k]

        def dcg(order):
            terms = []
            for r, j in enumerate(order):
                try:
                    gain_value = 2.0 ** y_true[j] - 1.0
                    discount = math.log2(r + 2)
                    term = gain_value / discount
                except (OverflowError, ValueError) as exc:
                    raise FloatingPointError(
                        "non-finite value encountered during ndcg score"
                    ) from exc
                if (
                    not math.isfinite(gain_value)
                    or not math.isfinite(discount)
                    or not math.isfinite(term)
                ):
                    raise FloatingPointError(
                        "non-finite value encountered during ndcg score"
                    )
                terms.append(term)
            try:
                total = math.fsum(terms)
            except (OverflowError, ValueError) as exc:
                raise FloatingPointError(
                    "non-finite value encountered during ndcg score"
                ) from exc
            if not math.isfinite(total):
                raise FloatingPointError(
                    "non-finite value encountered during ndcg score"
                )
            return total

        a = dcg(predicted)
        b = dcg(ideal)
        if b == 0:
            return 0.0
        try:
            result = a / b
        except (OverflowError, ValueError, ZeroDivisionError) as exc:
            raise FloatingPointError(
                "non-finite value encountered during ndcg score"
            ) from exc
        if not math.isfinite(result):
            raise FloatingPointError(
                "non-finite value encountered during ndcg score"
            )
        if result == 0:
            return 0.0
        return result

    width = None
    for row in y_true:
        if len(row) == 0:
            raise ValueError("y_true rows must be non-empty lists")
        if width is None:
            width = len(row)
        elif len(row) != width:
            raise ValueError("y_true must be rectangular")
        for value in row:
            if type(value) not in (int, float):
                raise ValueError(
                    "y_true must contain only finite non-negative "
                    "non-boolean numbers"
                )
            # math.isfinite raises OverflowError for ints too large to
            # convert to float; such values fail the finite requirement.
            try:
                finite = math.isfinite(value)
            except OverflowError as exc:
                raise ValueError(
                    "y_true must contain only finite non-negative "
                    "non-boolean numbers"
                ) from exc
            if not finite or value < 0:
                raise ValueError(
                    "y_true must contain only finite non-negative "
                    "non-boolean numbers"
                )
    if width < 2:
        raise ValueError("y_true must have at least two columns")
    n = len(y_true)

    if len(y_score) != n:
        raise ValueError(
            "y_score must be a list with the same length as y_true"
        )
    for row in y_score:
        if not isinstance(row, list) or len(row) != width:
            raise ValueError("y_score must have the same shape as y_true")
        for value in row:
            if type(value) not in (int, float):
                raise ValueError(
                    "y_score must contain only finite non-boolean numbers"
                )
            # math.isfinite raises OverflowError for ints too large to
            # convert to float; such values fail the finite requirement.
            try:
                finite = math.isfinite(value)
            except OverflowError as exc:
                raise ValueError(
                    "y_score must contain only finite non-boolean numbers"
                ) from exc
            if not finite:
                raise ValueError(
                    "y_score must contain only finite non-boolean numbers"
                )

    if k is None:
        k = width
    else:
        if type(k) is not int:
            raise ValueError("k must be None or an integer")
        if k < 1 or k > width:
            raise ValueError(
                "k must be between 1 and the number of columns"
            )

    if gain not in ("linear", "exponential"):
        raise ValueError("gain must be 'linear' or 'exponential'")

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

    def non_finite(exc):
        return FloatingPointError(
            "non-finite value encountered during ndcg score"
        )

    if sample_weight is None:
        weights = [1.0] * n
    else:
        try:
            weights = [float(value) for value in sample_weight]
        except (OverflowError, ValueError) as exc:
            raise non_finite(exc) from exc
        for value in weights:
            if not math.isfinite(value):
                raise non_finite(None)

    try:
        total_weight = math.fsum(weights)
    except (OverflowError, ValueError) as exc:
        raise ValueError("the total weight must be greater than 0") from exc
    if not math.isfinite(total_weight) or total_weight <= 0.0:
        raise ValueError("the total weight must be greater than 0")

    def dcg(order, labels):
        terms = []
        for r, j in enumerate(order):
            try:
                if gain == "linear":
                    relevance = float(labels[j])
                else:
                    relevance = 2.0 ** labels[j] - 1.0
                discount = math.log2(r + 2)
                term = relevance / discount
            except (OverflowError, ValueError, ZeroDivisionError) as exc:
                raise non_finite(exc) from exc
            if (
                not math.isfinite(relevance)
                or not math.isfinite(discount)
                or not math.isfinite(term)
            ):
                raise non_finite(None)
            terms.append(term)
        try:
            total = math.fsum(terms)
        except (OverflowError, ValueError) as exc:
            raise non_finite(exc) from exc
        if not math.isfinite(total):
            raise non_finite(None)
        return total

    row_scores = []
    for i in range(n):
        labels = y_true[i]
        scores = y_score[i]
        try:
            predicted = sorted(
                range(width), key=lambda j: (-scores[j], j)
            )[:k]
            ideal = sorted(
                range(width), key=lambda j: (-labels[j], j)
            )[:k]
        except (OverflowError, ValueError) as exc:
            raise non_finite(exc) from exc
        predicted_dcg = dcg(predicted, labels)
        ideal_dcg = dcg(ideal, labels)
        if ideal_dcg == 0.0:
            row_scores.append(0.0)
            continue
        try:
            row_value = predicted_dcg / ideal_dcg
        except (OverflowError, ValueError, ZeroDivisionError) as exc:
            raise non_finite(exc) from exc
        if not math.isfinite(row_value):
            raise non_finite(None)
        row_scores.append(row_value)

    weighted_terms = []
    for i in range(n):
        try:
            term = weights[i] * row_scores[i]
        except (OverflowError, ValueError) as exc:
            raise non_finite(exc) from exc
        if not math.isfinite(term):
            raise non_finite(None)
        weighted_terms.append(term)
    try:
        weighted_total = math.fsum(weighted_terms)
    except (OverflowError, ValueError) as exc:
        raise non_finite(exc) from exc
    if not math.isfinite(weighted_total):
        raise non_finite(None)
    try:
        result = weighted_total / total_weight
    except (OverflowError, ValueError, ZeroDivisionError) as exc:
        raise non_finite(exc) from exc
    if not math.isfinite(result):
        raise non_finite(None)
    if result == 0:
        return 0.0
    return result


def discounted_cumulative_gain_score(
    y_true, y_score, k=None, gain="exponential", sample_weight=None
) -> float:
    """Return the weighted mean per-row discounted cumulative gain at ``k``.

    ``y_true`` must be a non-empty rectangular list of rows with at
    least two columns; its elements must be finite non-negative values
    of type exactly ``int`` or ``float`` (booleans are rejected).
    ``y_score`` must be a list of the same shape whose elements are
    finite values of type exactly ``int`` or ``float`` (booleans are
    rejected). With ``k`` ``None`` the number of columns ``m`` is used;
    otherwise ``k`` must be an exact ``int`` (booleans are rejected)
    with ``1 <= k <= m``. ``gain`` must be ``"linear"`` or
    ``"exponential"``. ``sample_weight`` must be ``None`` -- every row
    then weighs ``1.0`` -- or a list with the same length as the number
    of rows whose elements are finite non-negative values of type
    exactly ``int`` or ``float`` (booleans are rejected). Any container,
    shape, length, type, value, range, or finiteness violation
    (including ``OverflowError`` raised by ``math.isfinite``) raises
    ValueError.

    For row ``i`` the column indices are sorted ascending by
    ``(-y_score[i][j], j)`` and only the first ``k`` are retained, so
    ties favor the smaller index. For rank ``r`` starting at zero the
    gain is ``g(x) = x`` for the linear variant and
    ``g(x) = 2.0 ** x - 1.0`` for the exponential variant, and the row
    score is
    ``d_i = math.fsum(g(y_true[i][j]) / math.log2(r + 2))`` over the
    retained columns in ascending rank. After validation the weights
    are converted to ``float`` in row order; ``math.fsum`` computes the
    total weight ``W = sum(w_i)`` and the weighted score
    ``S = sum(w_i * d_i)`` in row order, and the result is ``S / W``.
    Overflow or invalid operations during the ``W`` summation, a
    non-finite ``W``, or ``W <= 0`` raise ValueError; overflow, invalid
    operations, and non-finite values during the post-validation
    conversion, the gain and discount computation, the weighted
    multiplication, the ``S`` summation, or the final division raise
    FloatingPointError. An exact zero result is normalized to ``0.0``.

    The return value is a float. The inputs are not modified.
    Deterministic: same inputs, same result.
    """
    if not isinstance(y_true, list) or len(y_true) == 0:
        raise ValueError("y_true must be a non-empty list of rows")
    width = None
    for row in y_true:
        if not isinstance(row, list) or len(row) == 0:
            raise ValueError("y_true rows must be non-empty lists")
        if width is None:
            width = len(row)
        elif len(row) != width:
            raise ValueError("y_true must be rectangular")
        for value in row:
            if type(value) not in (int, float):
                raise ValueError(
                    "y_true must contain only finite non-negative "
                    "non-boolean numbers"
                )
            # math.isfinite raises OverflowError for ints too large to
            # convert to float; such values fail the finite requirement.
            try:
                finite = math.isfinite(value)
            except OverflowError as exc:
                raise ValueError(
                    "y_true must contain only finite non-negative "
                    "non-boolean numbers"
                ) from exc
            if not finite or value < 0:
                raise ValueError(
                    "y_true must contain only finite non-negative "
                    "non-boolean numbers"
                )
    if width < 2:
        raise ValueError("y_true must have at least two columns")
    n = len(y_true)

    if not isinstance(y_score, list) or len(y_score) != n:
        raise ValueError(
            "y_score must be a list with the same length as y_true"
        )
    for row in y_score:
        if not isinstance(row, list) or len(row) != width:
            raise ValueError("y_score must have the same shape as y_true")
        for value in row:
            if type(value) not in (int, float):
                raise ValueError(
                    "y_score must contain only finite non-boolean numbers"
                )
            # math.isfinite raises OverflowError for ints too large to
            # convert to float; such values fail the finite requirement.
            try:
                finite = math.isfinite(value)
            except OverflowError as exc:
                raise ValueError(
                    "y_score must contain only finite non-boolean numbers"
                ) from exc
            if not finite:
                raise ValueError(
                    "y_score must contain only finite non-boolean numbers"
                )

    if k is None:
        k = width
    else:
        if type(k) is not int:
            raise ValueError("k must be None or an integer")
        if k < 1 or k > width:
            raise ValueError(
                "k must be between 1 and the number of columns"
            )

    if gain not in ("linear", "exponential"):
        raise ValueError("gain must be 'linear' or 'exponential'")

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

    def non_finite(exc):
        return FloatingPointError(
            "non-finite value encountered during "
            "discounted cumulative gain score"
        )

    if sample_weight is None:
        weights = [1.0] * n
    else:
        try:
            weights = [float(value) for value in sample_weight]
        except (OverflowError, ValueError) as exc:
            raise non_finite(exc) from exc
        for value in weights:
            if not math.isfinite(value):
                raise non_finite(None)

    try:
        total_weight = math.fsum(weights)
    except (OverflowError, ValueError) as exc:
        raise ValueError("the total weight must be greater than 0") from exc
    if not math.isfinite(total_weight) or total_weight <= 0.0:
        raise ValueError("the total weight must be greater than 0")

    row_scores = []
    for i in range(n):
        labels = y_true[i]
        scores = y_score[i]
        try:
            top = sorted(
                range(width), key=lambda j: (-scores[j], j)
            )[:k]
        except (OverflowError, ValueError) as exc:
            raise non_finite(exc) from exc
        terms = []
        for r, j in enumerate(top):
            try:
                if gain == "linear":
                    relevance = float(labels[j])
                else:
                    relevance = 2.0 ** labels[j] - 1.0
                discount = math.log2(r + 2)
                term = relevance / discount
            except (OverflowError, ValueError, ZeroDivisionError) as exc:
                raise non_finite(exc) from exc
            if (
                not math.isfinite(relevance)
                or not math.isfinite(discount)
                or not math.isfinite(term)
            ):
                raise non_finite(None)
            terms.append(term)
        try:
            row_score = math.fsum(terms)
        except (OverflowError, ValueError) as exc:
            raise non_finite(exc) from exc
        if not math.isfinite(row_score):
            raise non_finite(None)
        row_scores.append(row_score)

    weighted_terms = []
    for i in range(n):
        try:
            term = weights[i] * row_scores[i]
        except (OverflowError, ValueError) as exc:
            raise non_finite(exc) from exc
        if not math.isfinite(term):
            raise non_finite(None)
        weighted_terms.append(term)
    try:
        weighted_total = math.fsum(weighted_terms)
    except (OverflowError, ValueError) as exc:
        raise non_finite(exc) from exc
    if not math.isfinite(weighted_total):
        raise non_finite(None)
    try:
        result = weighted_total / total_weight
    except (OverflowError, ValueError, ZeroDivisionError) as exc:
        raise non_finite(exc) from exc
    if not math.isfinite(result):
        raise non_finite(None)
    if result == 0:
        return 0.0
    return result


_JACCARD_AVERAGES = (None, "micro", "macro", "weighted")


def jaccard_score(y_true, y_pred, labels=None, average=None):
    """Return the Jaccard similarity score of two integer label vectors.

    ``y_true`` and ``y_pred`` must be non-empty lists of equal length
    whose elements have type exactly ``int`` (booleans are rejected).
    The inputs are not modified. Deterministic: same inputs, same result.

    When ``labels`` is ``None`` the label set is the sorted union of the
    labels appearing in either vector. Otherwise ``labels`` must be a
    non-empty list of distinct exact ``int`` values, kept in the given
    order, that contains every label appearing in either vector.
    ``average`` must be one of ``None``, ``"micro"``, ``"macro"``, or
    ``"weighted"``. Any violation raises ValueError.

    Counts are accumulated in sample order. For each label ``c``:
    ``tp`` counts true ``c`` predicted ``c``, ``fp`` counts true
    non-``c`` predicted ``c``, ``fn`` counts true ``c`` predicted
    non-``c``, and ``support`` counts true ``c``. The label's Jaccard
    score is ``J = tp / (tp + fp + fn)``; when the denominator is zero
    the score is ``0.0``.

    With ``average`` ``None`` the return is a list of floats aligned to
    the labels in label order. ``"micro"`` pools ``tp``, ``fp``, and
    ``fn`` across the labels in label order and scores the pooled
    counts. ``"macro"`` is the ``math.fsum`` mean, in label order, of
    the per-label scores; ``"weighted"`` divides the ``math.fsum``, in
    label order, of ``J * support`` by the number of samples. The three
    averaging modes return a float. An exact zero result is normalized
    to ``0.0``. Overflow, invalid operations, or non-finite values
    during the divisions or ``math.fsum`` steps raise
    FloatingPointError.
    """
    n = _check_metric_vectors(y_true, y_pred)
    for value in y_true:
        if type(value) is not int:
            raise ValueError("y_true must contain only integers")
    for value in y_pred:
        if type(value) is not int:
            raise ValueError("y_pred must contain only integers")
    if average not in _JACCARD_AVERAGES:
        raise ValueError(
            "average must be one of None, 'micro', 'macro', 'weighted'"
        )

    if labels is None:
        label_order = sorted(set(y_true) | set(y_pred))
    else:
        if not isinstance(labels, list) or len(labels) == 0:
            raise ValueError("labels must be a non-empty list")
        label_order = []
        seen = set()
        for value in labels:
            if type(value) is not int:
                raise ValueError("labels must contain only integers")
            if value in seen:
                raise ValueError("labels must not contain duplicates")
            seen.add(value)
            label_order.append(value)
        present = set(y_true) | set(y_pred)
        if any(value not in seen for value in present):
            raise ValueError(
                "labels must contain every label appearing in the inputs"
            )

    tp = {label: 0 for label in label_order}
    fp = {label: 0 for label in label_order}
    fn = {label: 0 for label in label_order}
    support = {label: 0 for label in label_order}
    for i in range(n):
        true_label = y_true[i]
        pred_label = y_pred[i]
        support[true_label] += 1
        if true_label == pred_label:
            tp[true_label] += 1
        else:
            fn[true_label] += 1
            fp[pred_label] += 1

    def non_finite(exc):
        return FloatingPointError(
            "non-finite value encountered during jaccard score"
        )

    def jaccard_for(tp_count, fp_count, fn_count):
        denominator = tp_count + fp_count + fn_count
        if denominator == 0:
            return 0.0
        try:
            score = tp_count / denominator
        except (OverflowError, ValueError) as exc:
            raise non_finite(exc) from exc
        if not math.isfinite(score):
            raise non_finite(None)
        if score == 0:
            return 0.0
        return score

    if average == "micro":
        total_tp = 0
        total_fp = 0
        total_fn = 0
        for label in label_order:
            total_tp += tp[label]
            total_fp += fp[label]
            total_fn += fn[label]
        return jaccard_for(total_tp, total_fp, total_fn)

    scores = [
        jaccard_for(tp[label], fp[label], fn[label]) for label in label_order
    ]

    if average is None:
        return scores

    if average == "macro":
        try:
            result = math.fsum(scores) / len(label_order)
        except (OverflowError, ValueError) as exc:
            raise non_finite(exc) from exc
        if not math.isfinite(result):
            raise non_finite(None)
        if result == 0:
            return 0.0
        return result

    try:
        result = math.fsum(
            scores[k] * support[label_order[k]]
            for k in range(len(label_order))
        ) / n
    except (OverflowError, ValueError) as exc:
        raise non_finite(exc) from exc
    if not math.isfinite(result):
        raise non_finite(None)
    if result == 0:
        return 0.0
    return result


def label_ranking_average_precision_score(
    y_true, y_score, sample_weight=None
) -> float:
    """Return the weighted mean per-sample average precision of the scores.

    ``y_true`` must be a non-empty rectangular list of rows with at least
    two columns whose elements are exactly the integers ``0`` and ``1``
    (booleans are rejected), and every row must contain at least one
    positive (``1``) label. ``y_score`` must be a list of the same shape
    whose elements are finite values of type exactly ``int`` or
    ``float`` (booleans are rejected). ``sample_weight`` must be
    ``None`` -- every sample then weighs ``1.0`` -- or a list with the
    same length as the number of rows whose elements are finite
    non-negative values of type exactly ``int`` or ``float`` (booleans
    are rejected). Any container, shape, length, type, value, or
    finiteness violation (including ``OverflowError`` raised by
    ``math.isfinite``) raises ValueError.

    For row ``i`` and each positive-label column ``j`` (in ascending
    column order), ``A`` is the number of columns whose score is not
    less than the score of column ``j`` and ``R`` is the number of
    positive labels among those columns; the row score is
    ``q_i = math.fsum(R / A for the positive columns) / number of
    positive labels``. After validation the weights are converted to
    ``float`` in row order; ``math.fsum`` computes the total weight
    ``W = sum(w_i)`` and the weighted score ``S = sum(w_i * q_i)``, and
    the result is ``S / W``. Overflow or invalid operations during the
    ``W`` summation, a non-finite ``W``, or ``W <= 0`` raise
    ValueError; overflow, invalid operations, and non-finite values
    during the post-validation conversion, the score divisions and
    ``math.fsum`` steps, the weighted multiplication, the ``S``
    summation, or the final division raise FloatingPointError. An exact
    zero result is normalized to ``0.0``.

    The return value is a float. The inputs are not modified.
    Deterministic: same inputs, same result.
    """
    if not isinstance(y_true, list) or len(y_true) == 0:
        raise ValueError("y_true must be a non-empty list of rows")
    width = None
    for row in y_true:
        if not isinstance(row, list) or len(row) == 0:
            raise ValueError("y_true rows must be non-empty lists")
        if width is None:
            width = len(row)
        elif len(row) != width:
            raise ValueError("y_true must be rectangular")
        for value in row:
            if type(value) is not int or value not in (0, 1):
                raise ValueError(
                    "y_true must contain only the integers 0 and 1"
                )
    if width < 2:
        raise ValueError("y_true must have at least two columns")
    for row in y_true:
        if not any(row):
            raise ValueError(
                "every y_true row must contain at least one positive label"
            )
    n = len(y_true)

    if not isinstance(y_score, list) or len(y_score) != n:
        raise ValueError(
            "y_score must be a list with the same length as y_true"
        )
    for row in y_score:
        if not isinstance(row, list) or len(row) != width:
            raise ValueError("y_score must have the same shape as y_true")
        for value in row:
            if type(value) not in (int, float):
                raise ValueError(
                    "y_score must contain only finite non-boolean numbers"
                )
            # math.isfinite raises OverflowError for ints too large to
            # convert to float; such values fail the finite requirement.
            try:
                finite = math.isfinite(value)
            except OverflowError as exc:
                raise ValueError(
                    "y_score must contain only finite non-boolean numbers"
                ) from exc
            if not finite:
                raise ValueError(
                    "y_score must contain only finite non-boolean numbers"
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

    def non_finite(exc):
        return FloatingPointError(
            "non-finite value encountered during label ranking average "
            "precision"
        )

    if sample_weight is None:
        weights = [1.0] * n
    else:
        try:
            weights = [float(value) for value in sample_weight]
        except (OverflowError, ValueError) as exc:
            raise non_finite(exc) from exc
        for value in weights:
            if not math.isfinite(value):
                raise non_finite(None)

    try:
        total_weight = math.fsum(weights)
    except (OverflowError, ValueError) as exc:
        raise ValueError("the total weight must be greater than 0") from exc
    if not math.isfinite(total_weight) or total_weight <= 0.0:
        raise ValueError("the total weight must be greater than 0")

    row_scores = []
    for i in range(n):
        labels = y_true[i]
        scores = y_score[i]
        positive_columns = [j for j in range(width) if labels[j] == 1]
        precision_terms = []
        for j in positive_columns:
            threshold = scores[j]
            ranked = [k for k in range(width) if scores[k] >= threshold]
            count_above = len(ranked)
            relevant_above = sum(1 for k in ranked if labels[k] == 1)
            try:
                precision = relevant_above / count_above
            except (OverflowError, ValueError) as exc:
                raise non_finite(exc) from exc
            if not math.isfinite(precision):
                raise non_finite(None)
            precision_terms.append(precision)
        try:
            summed = math.fsum(precision_terms)
            row_score = summed / len(positive_columns)
        except (OverflowError, ValueError) as exc:
            raise non_finite(exc) from exc
        if not math.isfinite(summed) or not math.isfinite(row_score):
            raise non_finite(None)
        row_scores.append(row_score)

    weighted_terms = []
    for i in range(n):
        try:
            term = weights[i] * row_scores[i]
        except (OverflowError, ValueError) as exc:
            raise non_finite(exc) from exc
        if not math.isfinite(term):
            raise non_finite(None)
        weighted_terms.append(term)
    try:
        weighted_total = math.fsum(weighted_terms)
    except (OverflowError, ValueError) as exc:
        raise non_finite(exc) from exc
    if not math.isfinite(weighted_total):
        raise non_finite(None)
    try:
        result = weighted_total / total_weight
    except (OverflowError, ValueError, ZeroDivisionError) as exc:
        raise non_finite(exc) from exc
    if not math.isfinite(result):
        raise non_finite(None)
    if result == 0:
        return 0.0
    return result


def coverage_error(y_true, y_score, sample_weight=None) -> float:
    """Return the weighted mean per-sample coverage error of the scores.

    ``y_true`` must be a non-empty rectangular list of rows with at least
    two columns whose elements are exactly the integers ``0`` and ``1``
    (booleans are rejected), and every row must contain at least one
    positive (``1``) label. ``y_score`` must be a list of the same shape
    whose elements are finite values of type exactly ``int`` or
    ``float`` (booleans are rejected). ``sample_weight`` must be
    ``None`` -- every sample then weighs ``1.0`` -- or a list with the
    same length as the number of rows whose elements are finite
    non-negative values of type exactly ``int`` or ``float`` (booleans
    are rejected). Any container, shape, length, type, value, or
    finiteness violation (including ``OverflowError`` raised by
    ``math.isfinite``) raises ValueError.

    For row ``i``, ``t`` is the minimum score among the positive-label
    columns and the row score ``q_i`` is the number of columns whose
    score is greater than or equal to ``t`` (ties at the threshold are
    all counted). After validation the weights are converted to
    ``float`` in row order; ``math.fsum`` computes the total weight
    ``W = sum(w_i)`` and the weighted score ``S = sum(w_i * q_i)``, and
    the result is ``S / W``. Overflow or invalid operations during the
    ``W`` summation, a non-finite ``W``, or ``W <= 0`` raise ValueError;
    overflow, invalid operations, and non-finite values during the
    post-validation conversion, the weighted multiplication, the ``S``
    summation, or the final division raise FloatingPointError.

    The return value is a float. The inputs are not modified.
    Deterministic: same inputs, same result.
    """
    if not isinstance(y_true, list) or len(y_true) == 0:
        raise ValueError("y_true must be a non-empty list of rows")
    width = None
    for row in y_true:
        if not isinstance(row, list) or len(row) == 0:
            raise ValueError("y_true rows must be non-empty lists")
        if width is None:
            width = len(row)
        elif len(row) != width:
            raise ValueError("y_true must be rectangular")
        for value in row:
            if type(value) is not int or value not in (0, 1):
                raise ValueError(
                    "y_true must contain only the integers 0 and 1"
                )
    if width < 2:
        raise ValueError("y_true must have at least two columns")
    for row in y_true:
        if not any(row):
            raise ValueError(
                "every y_true row must contain at least one positive label"
            )
    n = len(y_true)

    if not isinstance(y_score, list) or len(y_score) != n:
        raise ValueError(
            "y_score must be a list with the same length as y_true"
        )
    for row in y_score:
        if not isinstance(row, list) or len(row) != width:
            raise ValueError("y_score must have the same shape as y_true")
        for value in row:
            if type(value) not in (int, float):
                raise ValueError(
                    "y_score must contain only finite non-boolean numbers"
                )
            # math.isfinite raises OverflowError for ints too large to
            # convert to float; such values fail the finite requirement.
            try:
                finite = math.isfinite(value)
            except OverflowError as exc:
                raise ValueError(
                    "y_score must contain only finite non-boolean numbers"
                ) from exc
            if not finite:
                raise ValueError(
                    "y_score must contain only finite non-boolean numbers"
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

    def non_finite(exc):
        return FloatingPointError(
            "non-finite value encountered during coverage error"
        )

    if sample_weight is None:
        weights = [1.0] * n
    else:
        try:
            weights = [float(value) for value in sample_weight]
        except (OverflowError, ValueError) as exc:
            raise non_finite(exc) from exc
        for value in weights:
            if not math.isfinite(value):
                raise non_finite(None)

    try:
        total_weight = math.fsum(weights)
    except (OverflowError, ValueError) as exc:
        raise ValueError("the total weight must be greater than 0") from exc
    if not math.isfinite(total_weight) or total_weight <= 0.0:
        raise ValueError("the total weight must be greater than 0")

    row_scores = []
    for i in range(n):
        labels = y_true[i]
        scores = y_score[i]
        threshold = min(
            scores[j] for j in range(width) if labels[j] == 1
        )
        row_scores.append(sum(1 for score in scores if score >= threshold))

    weighted_terms = []
    for i in range(n):
        try:
            term = weights[i] * row_scores[i]
        except (OverflowError, ValueError) as exc:
            raise non_finite(exc) from exc
        if not math.isfinite(term):
            raise non_finite(None)
        weighted_terms.append(term)
    try:
        weighted_total = math.fsum(weighted_terms)
    except (OverflowError, ValueError) as exc:
        raise non_finite(exc) from exc
    if not math.isfinite(weighted_total):
        raise non_finite(None)
    try:
        result = weighted_total / total_weight
    except (OverflowError, ValueError, ZeroDivisionError) as exc:
        raise non_finite(exc) from exc
    if not math.isfinite(result):
        raise non_finite(None)
    return float(result)


def label_ranking_loss(y_true, y_score, sample_weight=None) -> float:
    """Return the weighted mean per-sample ranking loss of the scores.

    ``y_true`` must be a non-empty rectangular list of rows with at least
    two columns whose elements are exactly the integers ``0`` and ``1``
    (booleans are rejected); unlike the average-precision and coverage
    metrics, rows without a positive or without a negative label are
    allowed and contribute ``0.0``. ``y_score`` must be a list of the
    same shape whose elements are finite values of type exactly ``int``
    or ``float`` (booleans are rejected). ``sample_weight`` must be
    ``None`` -- every sample then weighs ``1.0`` -- or a list with the
    same length as the number of rows whose elements are finite
    non-negative values of type exactly ``int`` or ``float`` (booleans
    are rejected). Any container, shape, length, type, value, or
    finiteness violation (including ``OverflowError`` raised by
    ``math.isfinite``) raises ValueError.

    For row ``i`` let ``P`` be the positive-label columns and ``N`` the
    negative-label columns, both taken in ascending column order. If
    either is empty the row score is ``q_i = 0.0``; otherwise the pairs
    are traversed in ``P`` then ``N`` order, and each positive-negative
    pair contributes ``1.0`` when the positive score is lower than the
    negative score, ``0.5`` on a tie, and ``0.0`` when it is higher;
    ``math.fsum`` sums the contributions and the row score is that sum
    divided by ``|P| * |N|``. After validation the weights are converted
    to ``float`` in row order; ``math.fsum`` computes the total weight
    ``W = sum(w_i)`` and the weighted score ``S = sum(w_i * q_i)``, and
    the result is ``S / W``. Overflow or invalid operations during the
    ``W`` summation, a non-finite ``W``, or ``W <= 0`` raise ValueError;
    overflow, invalid operations, and non-finite values during the
    post-validation conversion, the pair summation and division, the
    weighted multiplication, the ``S`` summation, or the final division
    raise FloatingPointError. An exact zero result is normalized to
    ``0.0``.

    The return value is a float. The inputs are not modified.
    Deterministic: same inputs, same result.
    """
    if not isinstance(y_true, list) or len(y_true) == 0:
        raise ValueError("y_true must be a non-empty list of rows")
    width = None
    for row in y_true:
        if not isinstance(row, list) or len(row) == 0:
            raise ValueError("y_true rows must be non-empty lists")
        if width is None:
            width = len(row)
        elif len(row) != width:
            raise ValueError("y_true must be rectangular")
        for value in row:
            if type(value) is not int or value not in (0, 1):
                raise ValueError(
                    "y_true must contain only the integers 0 and 1"
                )
    if width < 2:
        raise ValueError("y_true must have at least two columns")
    n = len(y_true)

    if not isinstance(y_score, list) or len(y_score) != n:
        raise ValueError(
            "y_score must be a list with the same length as y_true"
        )
    for row in y_score:
        if not isinstance(row, list) or len(row) != width:
            raise ValueError("y_score must have the same shape as y_true")
        for value in row:
            if type(value) not in (int, float):
                raise ValueError(
                    "y_score must contain only finite non-boolean numbers"
                )
            # math.isfinite raises OverflowError for ints too large to
            # convert to float; such values fail the finite requirement.
            try:
                finite = math.isfinite(value)
            except OverflowError as exc:
                raise ValueError(
                    "y_score must contain only finite non-boolean numbers"
                ) from exc
            if not finite:
                raise ValueError(
                    "y_score must contain only finite non-boolean numbers"
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

    def non_finite(exc):
        return FloatingPointError(
            "non-finite value encountered during label ranking loss"
        )

    if sample_weight is None:
        weights = [1.0] * n
    else:
        try:
            weights = [float(value) for value in sample_weight]
        except (OverflowError, ValueError) as exc:
            raise non_finite(exc) from exc
        for value in weights:
            if not math.isfinite(value):
                raise non_finite(None)

    try:
        total_weight = math.fsum(weights)
    except (OverflowError, ValueError) as exc:
        raise ValueError("the total weight must be greater than 0") from exc
    if not math.isfinite(total_weight) or total_weight <= 0.0:
        raise ValueError("the total weight must be greater than 0")

    row_scores = []
    for i in range(n):
        labels = y_true[i]
        scores = y_score[i]
        positive_columns = [j for j in range(width) if labels[j] == 1]
        negative_columns = [j for j in range(width) if labels[j] == 0]
        if not positive_columns or not negative_columns:
            row_scores.append(0.0)
            continue
        pair_terms = []
        for j in positive_columns:
            positive_score = scores[j]
            for k in negative_columns:
                negative_score = scores[k]
                if positive_score < negative_score:
                    pair_terms.append(1.0)
                elif positive_score == negative_score:
                    pair_terms.append(0.5)
                else:
                    pair_terms.append(0.0)
        try:
            summed = math.fsum(pair_terms)
            row_score = summed / (
                len(positive_columns) * len(negative_columns)
            )
        except (OverflowError, ValueError) as exc:
            raise non_finite(exc) from exc
        if not math.isfinite(summed) or not math.isfinite(row_score):
            raise non_finite(None)
        row_scores.append(row_score)

    weighted_terms = []
    for i in range(n):
        try:
            term = weights[i] * row_scores[i]
        except (OverflowError, ValueError) as exc:
            raise non_finite(exc) from exc
        if not math.isfinite(term):
            raise non_finite(None)
        weighted_terms.append(term)
    try:
        weighted_total = math.fsum(weighted_terms)
    except (OverflowError, ValueError) as exc:
        raise non_finite(exc) from exc
    if not math.isfinite(weighted_total):
        raise non_finite(None)
    try:
        result = weighted_total / total_weight
    except (OverflowError, ValueError, ZeroDivisionError) as exc:
        raise non_finite(exc) from exc
    if not math.isfinite(result):
        raise non_finite(None)
    if result == 0:
        return 0.0
    return result


def hamming_loss(y_true, y_pred, sample_weight=None) -> float:
    """Return the weighted Hamming loss of two binary multilabel matrices.

    ``y_true`` and ``y_pred`` must each be a non-empty list of non-empty
    lists, have identical numbers of rows and columns (at least one
    column), and every element must be an integer whose type is exactly
    ``int`` with value ``0`` or ``1``; booleans (and any other type),
    non-list containers, an empty matrix, ragged rows, or differing
    shapes raise ValueError. ``sample_weight`` must be ``None`` -- every
    row then weighs ``1.0`` -- or a list with the same length as the
    number of rows whose elements are finite non-negative values of type
    exactly ``int`` or ``float`` (booleans are rejected); any other
    container, length, type, value, or finiteness violation (including
    ``OverflowError`` raised by ``math.isfinite``) raises ValueError.

    For row ``i`` let ``e_i`` be the number of columns where the two
    matrices differ; the row loss is ``q_i = e_i / n_columns``. After
    validation the weights are converted to ``float`` in row order, and
    ``math.fsum`` computes the total weight ``W = sum(w_i)``; overflow
    or invalid operations during that summation, a non-finite ``W``, or
    ``W <= 0`` raise ValueError. The weighted loss
    ``L = sum(w_i * q_i)`` is likewise accumulated with ``math.fsum`` in
    row order, and the result is ``L / W``. Overflow, invalid
    operations, or non-finite values during the post-validation weight
    conversion, the row division, the weighted multiplication, either
    subsequent ``math.fsum``, or the final division raise
    FloatingPointError. An exact zero result is normalized to ``0.0``.

    The return value is a float. The inputs are not modified.
    Deterministic: same inputs, same result.
    """
    if not isinstance(y_true, list) or len(y_true) == 0:
        raise ValueError("y_true must be a non-empty list of rows")
    if not isinstance(y_pred, list) or len(y_pred) == 0:
        raise ValueError("y_pred must be a non-empty list of rows")
    n = len(y_true)
    if len(y_pred) != n:
        raise ValueError("y_true and y_pred must have the same number of rows")

    width = None
    for row in y_true:
        if not isinstance(row, list) or len(row) == 0:
            raise ValueError("y_true rows must be non-empty lists")
        if width is None:
            width = len(row)
        elif len(row) != width:
            raise ValueError("y_true must be rectangular")
        for value in row:
            if type(value) is not int or value not in (0, 1):
                raise ValueError(
                    "y_true must contain only the integers 0 and 1"
                )

    for row in y_pred:
        if not isinstance(row, list) or len(row) == 0:
            raise ValueError("y_pred rows must be non-empty lists")
        if len(row) != width:
            raise ValueError("y_pred must have the same shape as y_true")
        for value in row:
            if type(value) is not int or value not in (0, 1):
                raise ValueError(
                    "y_pred must contain only the integers 0 and 1"
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

    def non_finite(exc):
        return FloatingPointError(
            "non-finite value encountered during hamming loss"
        )

    if sample_weight is None:
        weights = [1.0] * n
    else:
        try:
            weights = [float(value) for value in sample_weight]
        except (OverflowError, ValueError) as exc:
            raise non_finite(exc) from exc
        for value in weights:
            if not math.isfinite(value):
                raise non_finite(None)

    try:
        total_weight = math.fsum(weights)
    except (OverflowError, ValueError) as exc:
        raise ValueError("the total weight must be greater than 0") from exc
    if not math.isfinite(total_weight) or total_weight <= 0.0:
        raise ValueError("the total weight must be greater than 0")

    row_losses = []
    for i in range(n):
        true_row = y_true[i]
        pred_row = y_pred[i]
        mismatches = 0
        for j in range(width):
            if true_row[j] != pred_row[j]:
                mismatches += 1
        try:
            row_loss = mismatches / width
        except (OverflowError, ValueError) as exc:
            raise non_finite(exc) from exc
        if not math.isfinite(row_loss):
            raise non_finite(None)
        row_losses.append(row_loss)

    weighted_terms = []
    for i in range(n):
        try:
            term = weights[i] * row_losses[i]
        except (OverflowError, ValueError) as exc:
            raise non_finite(exc) from exc
        if not math.isfinite(term):
            raise non_finite(None)
        weighted_terms.append(term)
    try:
        weighted_total = math.fsum(weighted_terms)
    except (OverflowError, ValueError) as exc:
        raise non_finite(exc) from exc
    if not math.isfinite(weighted_total):
        raise non_finite(None)
    try:
        result = weighted_total / total_weight
    except (OverflowError, ValueError, ZeroDivisionError) as exc:
        raise non_finite(exc) from exc
    if not math.isfinite(result):
        raise non_finite(None)
    if result == 0:
        return 0.0
    return result


def mean_reciprocal_rank_score(
    y_true, y_score, k=None, sample_weight=None
) -> float:
    """Return the weighted mean reciprocal rank of the scores.

    ``y_true`` must be a non-empty rectangular list of rows with at least
    two columns whose elements are exactly the integers ``0`` and ``1``
    (booleans are rejected). Rows without a positive label are allowed.
    ``y_score`` must be a list of the same shape whose elements are
    finite values of type exactly ``int`` or ``float`` (booleans are
    rejected). ``k`` must be ``None`` or an exact ``int`` (booleans are
    rejected); with ``None`` the number of columns ``m`` is used,
    otherwise ``1 <= k <= m`` is required. ``sample_weight`` must be
    ``None`` -- every sample then weighs ``1.0`` -- or a list with the
    same length as the number of rows whose elements are finite
    non-negative values of type exactly ``int`` or ``float`` (booleans
    are rejected). Any container, shape, length, type, value, range, or
    finiteness violation (including ``OverflowError`` raised by
    ``math.isfinite``) raises ValueError.

    For row ``i`` the column indices are sorted ascending by
    ``(-y_score[i][j], j)`` and only the first ``k`` are retained, so
    ties favor the smaller index. Let ``r`` be the one-based position of
    the first retained column whose true label is ``1``; the row score
    is ``q_i = 1.0 / r`` when such a column exists and ``0.0`` when no
    positive label appears in the top ``k``. After validation the
    weights are converted to ``float`` in row order; ``math.fsum``
    computes the total weight ``W = sum(w_i)`` and the weighted score
    ``S = sum(w_i * q_i)``, and the result is ``S / W``. Overflow or
    invalid operations during the ``W`` summation, a non-finite ``W``,
    or ``W <= 0`` raise ValueError; overflow, invalid operations, and
    non-finite values during the post-validation conversion, the
    reciprocal division, the weighted multiplication, the ``S``
    summation, or the final division raise FloatingPointError. An exact
    zero result is normalized to ``0.0``.

    The return value is a float. The inputs are not modified.
    Deterministic: same inputs, same result.
    """
    if not isinstance(y_true, list) or len(y_true) == 0:
        raise ValueError("y_true must be a non-empty list of rows")
    width = None
    for row in y_true:
        if not isinstance(row, list) or len(row) == 0:
            raise ValueError("y_true rows must be non-empty lists")
        if width is None:
            width = len(row)
        elif len(row) != width:
            raise ValueError("y_true must be rectangular")
        for value in row:
            if type(value) is not int or value not in (0, 1):
                raise ValueError(
                    "y_true must contain only the integers 0 and 1"
                )
    if width < 2:
        raise ValueError("y_true must have at least two columns")
    n = len(y_true)

    if not isinstance(y_score, list) or len(y_score) != n:
        raise ValueError(
            "y_score must be a list with the same length as y_true"
        )
    for row in y_score:
        if not isinstance(row, list) or len(row) != width:
            raise ValueError("y_score must have the same shape as y_true")
        for value in row:
            if type(value) not in (int, float):
                raise ValueError(
                    "y_score must contain only finite non-boolean numbers"
                )
            # math.isfinite raises OverflowError for ints too large to
            # convert to float; such values fail the finite requirement.
            try:
                finite = math.isfinite(value)
            except OverflowError as exc:
                raise ValueError(
                    "y_score must contain only finite non-boolean numbers"
                ) from exc
            if not finite:
                raise ValueError(
                    "y_score must contain only finite non-boolean numbers"
                )

    if k is None:
        k = width
    else:
        if type(k) is not int:
            raise ValueError("k must be None or an integer")
        if k < 1 or k > width:
            raise ValueError(
                "k must be between 1 and the number of columns"
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

    def non_finite(exc):
        return FloatingPointError(
            "non-finite value encountered during mean reciprocal rank score"
        )

    if sample_weight is None:
        weights = [1.0] * n
    else:
        try:
            weights = [float(value) for value in sample_weight]
        except (OverflowError, ValueError) as exc:
            raise non_finite(exc) from exc
        for value in weights:
            if not math.isfinite(value):
                raise non_finite(None)

    try:
        total_weight = math.fsum(weights)
    except (OverflowError, ValueError) as exc:
        raise ValueError("the total weight must be greater than 0") from exc
    if not math.isfinite(total_weight) or total_weight <= 0.0:
        raise ValueError("the total weight must be greater than 0")

    row_scores = []
    for i in range(n):
        labels = y_true[i]
        scores = y_score[i]
        order = sorted(range(width), key=lambda j: (-scores[j], j))[:k]
        row_score = 0.0
        for r, j in enumerate(order, start=1):
            if labels[j] == 1:
                try:
                    row_score = 1.0 / r
                except (OverflowError, ValueError, ZeroDivisionError) as exc:
                    raise non_finite(exc) from exc
                if not math.isfinite(row_score):
                    raise non_finite(None)
                break
        row_scores.append(row_score)

    weighted_terms = []
    for i in range(n):
        try:
            term = weights[i] * row_scores[i]
        except (OverflowError, ValueError) as exc:
            raise non_finite(exc) from exc
        if not math.isfinite(term):
            raise non_finite(None)
        weighted_terms.append(term)
    try:
        weighted_total = math.fsum(weighted_terms)
    except (OverflowError, ValueError) as exc:
        raise non_finite(exc) from exc
    if not math.isfinite(weighted_total):
        raise non_finite(None)
    try:
        result = weighted_total / total_weight
    except (OverflowError, ValueError, ZeroDivisionError) as exc:
        raise non_finite(exc) from exc
    if not math.isfinite(result):
        raise non_finite(None)
    if result == 0:
        return 0.0
    return result


def _check_ranking_at_k_inputs(y_true, y_score, k):
    """Validate and return ``(n, width)`` for precision/recall at k.

    ``y_true`` must be a non-empty rectangular list of rows with at
    least two columns whose elements are exactly the integers ``0``
    and ``1`` (booleans are rejected); ``y_score`` must have the same
    shape and contain only finite values of type exactly ``int`` or
    ``float`` (booleans are rejected). ``k`` must be an exact ``int``
    with ``1 <= k <= width``. Any violation raises ValueError.
    """
    if not isinstance(y_true, list) or len(y_true) == 0:
        raise ValueError("y_true must be a non-empty list of rows")
    width = None
    for row in y_true:
        if not isinstance(row, list) or len(row) == 0:
            raise ValueError("y_true rows must be non-empty lists")
        if width is None:
            width = len(row)
        elif len(row) != width:
            raise ValueError("y_true must be rectangular")
        for value in row:
            if type(value) is not int or value not in (0, 1):
                raise ValueError(
                    "y_true must contain only the integers 0 and 1"
                )
    if width < 2:
        raise ValueError("y_true must have at least two columns")
    n = len(y_true)

    if not isinstance(y_score, list) or len(y_score) != n:
        raise ValueError(
            "y_score must be a list with the same length as y_true"
        )
    for row in y_score:
        if not isinstance(row, list) or len(row) != width:
            raise ValueError("y_score must have the same shape as y_true")
        for value in row:
            if type(value) not in (int, float):
                raise ValueError(
                    "y_score must contain only finite non-boolean numbers"
                )
            # math.isfinite raises OverflowError for ints too large to
            # convert to float; such values fail the finite requirement.
            try:
                finite = math.isfinite(value)
            except OverflowError as exc:
                raise ValueError(
                    "y_score must contain only finite non-boolean numbers"
                ) from exc
            if not finite:
                raise ValueError(
                    "y_score must contain only finite non-boolean numbers"
                )

    if type(k) is not int:
        raise ValueError("k must be an integer")
    if k < 1 or k > width:
        raise ValueError(
            "k must be between 1 and the number of columns"
        )
    return n, width


def _weighted_mean_at_k(y_true, y_score, k, sample_weight, mode):
    """Shared weighted-mean implementation for precision/recall at k."""
    n, width = _check_ranking_at_k_inputs(y_true, y_score, k)

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

    if mode == "precision":
        label = "precision at k score"
    elif mode == "recall":
        label = "recall at k score"
    else:
        label = "mean average precision at k score"

    def non_finite(exc):
        return FloatingPointError(
            "non-finite value encountered during " + label
        )

    if sample_weight is None:
        weights = [1.0] * n
    else:
        try:
            weights = [float(value) for value in sample_weight]
        except (OverflowError, ValueError) as exc:
            raise non_finite(exc) from exc
        for value in weights:
            if not math.isfinite(value):
                raise non_finite(None)

    try:
        total_weight = math.fsum(weights)
    except (OverflowError, ValueError) as exc:
        raise ValueError("the total weight must be greater than 0") from exc
    if not math.isfinite(total_weight) or total_weight <= 0.0:
        raise ValueError("the total weight must be greater than 0")

    row_scores = []
    for i in range(n):
        labels = y_true[i]
        scores = y_score[i]
        try:
            top = sorted(
                range(width), key=lambda j: (-scores[j], j)
            )[:k]
        except (OverflowError, ValueError) as exc:
            raise non_finite(exc) from exc
        hits = 0
        if mode == "map":
            positives = 0
            for j in range(width):
                if labels[j] == 1:
                    positives += 1
            if positives == 0:
                row_score = 0.0
            else:
                cumulative = 0
                hit_count = 0
                for rank, j in enumerate(top, start=1):
                    if labels[j] == 1:
                        hit_count += 1
                        try:
                            cumulative += hit_count / rank
                        except (
                            OverflowError,
                            ValueError,
                            ZeroDivisionError,
                        ) as exc:
                            raise non_finite(exc) from exc
                try:
                    row_score = cumulative / min(positives, k)
                except (
                    OverflowError,
                    ValueError,
                    ZeroDivisionError,
                ) as exc:
                    raise non_finite(exc) from exc
        else:
            for j in top:
                if labels[j] == 1:
                    hits += 1
        if mode == "precision":
            try:
                row_score = hits / k
            except (OverflowError, ValueError, ZeroDivisionError) as exc:
                raise non_finite(exc) from exc
        elif mode == "recall":
            positives = 0
            for j in range(width):
                if labels[j] == 1:
                    positives += 1
            if positives > 0:
                try:
                    row_score = hits / positives
                except (
                    OverflowError,
                    ValueError,
                    ZeroDivisionError,
                ) as exc:
                    raise non_finite(exc) from exc
            else:
                row_score = 0.0
        if not math.isfinite(row_score):
            raise non_finite(None)
        row_scores.append(row_score)

    weighted_terms = []
    for i in range(n):
        try:
            term = weights[i] * row_scores[i]
        except (OverflowError, ValueError) as exc:
            raise non_finite(exc) from exc
        if not math.isfinite(term):
            raise non_finite(None)
        weighted_terms.append(term)
    try:
        weighted_total = math.fsum(weighted_terms)
    except (OverflowError, ValueError) as exc:
        raise non_finite(exc) from exc
    if not math.isfinite(weighted_total):
        raise non_finite(None)
    try:
        result = weighted_total / total_weight
    except (OverflowError, ValueError, ZeroDivisionError) as exc:
        raise non_finite(exc) from exc
    if not math.isfinite(result):
        raise non_finite(None)
    if result == 0:
        return 0.0
    return result


def precision_at_k_score(
    y_true, y_score, k=1, sample_weight=None
) -> float:
    """Return the weighted mean per-sample precision at ``k``.

    ``y_true`` must be a non-empty rectangular list of rows with at
    least two columns whose elements are exactly the integers ``0`` and
    ``1`` (booleans are rejected). Rows without a positive label are
    allowed. ``y_score`` must be a list of the same shape whose
    elements are finite values of type exactly ``int`` or ``float``
    (booleans are rejected). ``k`` must be an exact ``int`` (booleans
    are rejected) with ``1 <= k <= m`` where ``m`` is the number of
    columns. ``sample_weight`` must be ``None`` -- every sample then
    weighs ``1.0`` -- or a list with the same length as the number of
    rows whose elements are finite non-negative values of type exactly
    ``int`` or ``float`` (booleans are rejected). Any container, shape,
    length, type, value, range, or finiteness violation (including
    ``OverflowError`` raised by ``math.isfinite``) raises ValueError.

    For row ``i`` the column indices are sorted ascending by
    ``(-y_score[i][j], j)`` and only the first ``k`` are retained, so
    ties favor the smaller index. Let ``h`` be the number of retained
    columns whose true label is ``1``; the row precision is
    ``q_i = h / k``. After validation the weights are converted to
    ``float`` in row order; ``math.fsum`` computes the total weight
    ``W = sum(w_i)`` and the weighted score ``S = sum(w_i * q_i)``, and
    the result is ``S / W``. Overflow or invalid operations during the
    ``W`` summation, a non-finite ``W``, or ``W <= 0`` raise
    ValueError; overflow, invalid operations, and non-finite values
    during the post-validation conversion, the precision division, the
    weighted multiplication, the ``S`` summation, or the final division
    raise FloatingPointError. An exact zero result is normalized to
    ``0.0``.

    The return value is a float. The inputs are not modified.
    Deterministic: same inputs, same result.
    """
    return _weighted_mean_at_k(
        y_true, y_score, k, sample_weight, "precision"
    )


def recall_at_k_score(
    y_true, y_score, k=1, sample_weight=None
) -> float:
    """Return the weighted mean per-sample recall at ``k``.

    ``y_true`` must be a non-empty rectangular list of rows with at
    least two columns whose elements are exactly the integers ``0`` and
    ``1`` (booleans are rejected). Rows without a positive label are
    allowed and contribute a row recall of ``0.0``. ``y_score`` must be
    a list of the same shape whose elements are finite values of type
    exactly ``int`` or ``float`` (booleans are rejected). ``k`` must be
    an exact ``int`` (booleans are rejected) with ``1 <= k <= m`` where
    ``m`` is the number of columns. ``sample_weight`` must be ``None``
    -- every sample then weighs ``1.0`` -- or a list with the same
    length as the number of rows whose elements are finite
    non-negative values of type exactly ``int`` or ``float`` (booleans
    are rejected). Any container, shape, length, type, value, range, or
    finiteness violation (including ``OverflowError`` raised by
    ``math.isfinite``) raises ValueError.

    For row ``i`` the column indices are sorted ascending by
    ``(-y_score[i][j], j)`` and only the first ``k`` are retained, so
    ties favor the smaller index. Let ``h`` be the number of retained
    columns whose true label is ``1`` and ``p`` the number of positive
    labels in the whole row; the row recall is ``q_i = h / p`` when
    ``p > 0`` and ``0.0`` otherwise. After validation the weights are
    converted to ``float`` in row order; ``math.fsum`` computes the
    total weight ``W = sum(w_i)`` and the weighted score
    ``S = sum(w_i * q_i)``, and the result is ``S / W``. Overflow or
    invalid operations during the ``W`` summation, a non-finite ``W``,
    or ``W <= 0`` raise ValueError; overflow, invalid operations, and
    non-finite values during the post-validation conversion, the recall
    division, the weighted multiplication, the ``S`` summation, or the
    final division raise FloatingPointError. An exact zero result is
    normalized to ``0.0``.

    The return value is a float. The inputs are not modified.
    Deterministic: same inputs, same result.
    """
    return _weighted_mean_at_k(
        y_true, y_score, k, sample_weight, "recall"
    )


def mean_average_precision_at_k_score(
    y_true, y_score, k=1, sample_weight=None
) -> float:
    """Return the weighted mean per-sample average precision at ``k``.

    ``y_true`` must be a non-empty rectangular list of rows with at
    least two columns whose elements are exactly the integers ``0`` and
    ``1`` (booleans are rejected). ``y_score`` must be a list of the
    same shape whose elements are finite values of type exactly ``int``
    or ``float`` (booleans are rejected). ``k`` must be an exact ``int``
    (booleans are rejected) with ``1 <= k <= m`` where ``m`` is the
    number of columns. ``sample_weight`` must be ``None`` -- every
    sample then weighs ``1.0`` -- or a list with the same length as the
    number of rows whose elements are finite non-negative values of
    type exactly ``int`` or ``float`` (booleans are rejected). Any
    container, shape, length, type, value, range, or finiteness
    violation (including ``OverflowError`` raised by
    ``math.isfinite``) raises ValueError.

    For row ``i`` the column indices are sorted ascending by
    ``(-y_score[i][j], j)`` and only the first ``k`` are retained, so
    ties favor the smaller index. Walking the retained columns in rank
    order ``r = 1, ..., k``, every time the true label is ``1`` a hit
    counter ``h`` is incremented and ``h / r`` is accumulated. Let
    ``p`` be the number of positive labels in the whole row; the row
    score is ``q_i = 0.0`` when ``p == 0`` and the accumulated value
    divided by ``min(p, k)`` otherwise. After validation the weights
    are converted to ``float`` in row order; ``math.fsum`` computes the
    total weight ``W = sum(w_i)`` and the weighted score
    ``S = sum(w_i * q_i)``, and the result is ``S / W``. Overflow or
    invalid operations during the ``W`` summation, a non-finite ``W``,
    or ``W <= 0`` raise ValueError; overflow, invalid operations, and
    non-finite values during the post-validation conversion, the score
    negation, the divisions and accumulations, the weighted
    multiplication, the ``S`` summation, or the final division raise
    FloatingPointError. An exact zero result is normalized to ``0.0``.

    The return value is a float. The inputs are not modified.
    Deterministic: same inputs, same result.
    """
    return _weighted_mean_at_k(
        y_true, y_score, k, sample_weight, "map"
    )


def continuous_ranked_probability_score(
    y_true, y_samples, sample_weight=None, ensemble_weight=None
) -> float:
    """Return the weighted mean continuous ranked probability score.

    ``y_true`` must be a non-empty list whose elements are finite values
    of type exactly ``int`` or ``float`` (booleans are rejected).
    ``y_samples`` must be a list of rows of the same height as
    ``y_true`` and a common width of at least one column; its elements
    are likewise finite values of type exactly ``int`` or ``float``
    (booleans are rejected). ``sample_weight`` must be ``None`` -- every
    sample then weighs ``1.0`` -- or a list with the same length as
    ``y_true``; ``ensemble_weight`` must be ``None`` -- every ensemble
    member then weighs ``1.0`` -- or a list with the same length as the
    number of columns. Weight entries must be finite non-negative values
    of type exactly ``int`` or ``float`` (booleans are rejected). Any
    container, shape, length, type, range, or finiteness violation
    (including ``OverflowError`` raised by ``math.isfinite``) raises
    ValueError.

    After validation the values and weights are converted to ``float``
    in input order. ``math.fsum`` computes the sample-weight total
    ``W = fsum(w_i)`` and the ensemble-weight total
    ``V = fsum(v_j)``; if either summation overflows or is invalid, is
    non-finite, or its total is less than or equal to zero, a
    ValueError is raised. For row ``i`` with target ``t_i`` and members
    ``x_ij``, the member term is summed in column order ``j`` as
    ``A = fsum(v_j * abs(x_ij - t_i)) / V`` and the pairwise term is
    summed with ``j`` outer and ``k`` inner as
    ``B = fsum(v_j * v_k * abs(x_ij - x_ik)) / (2 * V * V)``; the row
    score is ``q_i = A - B``. The result is
    ``fsum(w_i * q_i in row order) / W``. Overflow or invalid
    operations during the post-validation conversion, absolute value,
    arithmetic, or ``math.fsum`` steps, a zero division, and non-finite
    intermediate values or results raise FloatingPointError. An exact
    zero result is normalized to ``0.0``.

    The return value is a float. The inputs are not modified.
    Deterministic: same inputs, same result.
    """
    if not isinstance(y_true, list) or len(y_true) == 0:
        raise ValueError("y_true must be a non-empty list")
    n = len(y_true)
    for value in y_true:
        if type(value) not in (int, float):
            raise ValueError(
                "y_true must contain only finite non-boolean numbers"
            )
        # math.isfinite raises OverflowError for ints too large to
        # convert to float; such values fail the finite requirement.
        try:
            finite = math.isfinite(value)
        except OverflowError as exc:
            raise ValueError(
                "y_true must contain only finite non-boolean numbers"
            ) from exc
        if not finite:
            raise ValueError(
                "y_true must contain only finite non-boolean numbers"
            )

    if not isinstance(y_samples, list) or len(y_samples) != n:
        raise ValueError(
            "y_samples must be a list of rows with the same length as "
            "y_true"
        )
    width = None
    for row in y_samples:
        if not isinstance(row, list) or len(row) == 0:
            raise ValueError("y_samples rows must be non-empty lists")
        if width is None:
            width = len(row)
        elif len(row) != width:
            raise ValueError("y_samples must be rectangular")
        for value in row:
            if type(value) not in (int, float):
                raise ValueError(
                    "y_samples must contain only finite non-boolean "
                    "numbers"
                )
            # math.isfinite raises OverflowError for ints too large to
            # convert to float; such values fail the finite requirement.
            try:
                finite = math.isfinite(value)
            except OverflowError as exc:
                raise ValueError(
                    "y_samples must contain only finite non-boolean "
                    "numbers"
                ) from exc
            if not finite:
                raise ValueError(
                    "y_samples must contain only finite non-boolean "
                    "numbers"
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

    if ensemble_weight is not None:
        if (
            not isinstance(ensemble_weight, list)
            or len(ensemble_weight) != width
        ):
            raise ValueError(
                "ensemble_weight must be a list with the same length as "
                "the number of columns of y_samples"
            )
        for value in ensemble_weight:
            if type(value) not in (int, float):
                raise ValueError(
                    "ensemble_weight must contain only finite "
                    "non-negative non-boolean numbers"
                )
            try:
                finite = math.isfinite(value)
            except OverflowError as exc:
                raise ValueError(
                    "ensemble_weight must contain only finite "
                    "non-negative non-boolean numbers"
                ) from exc
            if not finite or value < 0:
                raise ValueError(
                    "ensemble_weight must contain only finite "
                    "non-negative non-boolean numbers"
                )

    def non_finite(exc):
        return FloatingPointError(
            "non-finite value encountered during continuous ranked "
            "probability score"
        )

    try:
        t = [float(value) for value in y_true]
        x = [[float(value) for value in row] for row in y_samples]
        if sample_weight is None:
            w = [1.0] * n
        else:
            w = [float(value) for value in sample_weight]
        if ensemble_weight is None:
            v = [1.0] * width
        else:
            v = [float(value) for value in ensemble_weight]
    except (OverflowError, ValueError) as exc:
        raise non_finite(exc) from exc
    for values in [t, w, v] + x:
        for value in values:
            if not math.isfinite(value):
                raise non_finite(None)

    try:
        total_sample_weight = math.fsum(w)
        total_ensemble_weight = math.fsum(v)
    except (OverflowError, ValueError) as exc:
        raise ValueError("the total weight must be greater than 0") from exc
    if (
        not math.isfinite(total_sample_weight)
        or total_sample_weight <= 0.0
        or not math.isfinite(total_ensemble_weight)
        or total_ensemble_weight <= 0.0
    ):
        raise ValueError("the total weight must be greater than 0")

    row_scores = []
    for i in range(n):
        member_terms = []
        pairwise_terms = []
        for j in range(width):
            try:
                member_distance = abs(x[i][j] - t[i])
                member_terms.append(v[j] * member_distance)
            except (OverflowError, ValueError) as exc:
                raise non_finite(exc) from exc
            if not math.isfinite(member_distance) or not math.isfinite(
                member_terms[-1]
            ):
                raise non_finite(None)
            for k in range(width):
                try:
                    pairwise_distance = abs(x[i][j] - x[i][k])
                    pairwise_terms.append(
                        v[j] * v[k] * pairwise_distance
                    )
                except (OverflowError, ValueError) as exc:
                    raise non_finite(exc) from exc
                if not math.isfinite(pairwise_distance) or not math.isfinite(
                    pairwise_terms[-1]
                ):
                    raise non_finite(None)
        try:
            member_sum = math.fsum(member_terms)
            pairwise_sum = math.fsum(pairwise_terms)
        except (OverflowError, ValueError) as exc:
            raise non_finite(exc) from exc
        if not math.isfinite(member_sum) or not math.isfinite(pairwise_sum):
            raise non_finite(None)
        try:
            member_part = member_sum / total_ensemble_weight
            pairwise_denominator = (
                2.0 * total_ensemble_weight * total_ensemble_weight
            )
            pairwise_part = pairwise_sum / pairwise_denominator
            row_score = member_part - pairwise_part
        except (
            OverflowError,
            ValueError,
            ZeroDivisionError,
        ) as exc:
            raise non_finite(exc) from exc
        if (
            not math.isfinite(pairwise_denominator)
            or not math.isfinite(member_part)
            or not math.isfinite(pairwise_part)
            or not math.isfinite(row_score)
        ):
            raise non_finite(None)
        row_scores.append(row_score)

    weighted_terms = []
    for i in range(n):
        try:
            term = w[i] * row_scores[i]
        except (OverflowError, ValueError) as exc:
            raise non_finite(exc) from exc
        if not math.isfinite(term):
            raise non_finite(None)
        weighted_terms.append(term)
    try:
        weighted_total = math.fsum(weighted_terms)
    except (OverflowError, ValueError) as exc:
        raise non_finite(exc) from exc
    if not math.isfinite(weighted_total):
        raise non_finite(None)
    try:
        result = weighted_total / total_sample_weight
    except (OverflowError, ValueError, ZeroDivisionError) as exc:
        raise non_finite(exc) from exc
    if not math.isfinite(result):
        raise non_finite(None)
    if result == 0:
        return 0.0
    return result


# ``from __future__ import annotations`` stores annotations as strings;
# expose the builtin ``float`` as the runtime return annotation.
continuous_ranked_probability_score.__annotations__["return"] = float


def energy_score(
    y_true, y_samples, beta=1.0, sample_weight=None, ensemble_weight=None
) -> float:
    """Return the weighted mean energy score of an ensemble of forecasts.

    ``y_true`` must be a non-empty list of non-empty rows (shape n x d);
    all rows must share the same width ``d``. ``y_samples`` must be a
    list of ``n`` groups; each group must be a non-empty list of ``m``
    rows of width ``d``, and all groups must share the same size ``m``.
    Every data element must be a finite value of type exactly ``int`` or
    ``float`` (booleans are rejected). ``beta`` must be a finite value
    of type exactly ``int`` or ``float`` (booleans are rejected)
    strictly between ``0`` and ``2``. ``sample_weight`` must be
    ``None`` -- every sample then weighs ``1.0`` -- or a list of length
    ``n``; ``ensemble_weight`` must be ``None`` -- every ensemble member
    then weighs ``1.0`` -- or a list of length ``m``. Weight entries
    must be finite non-negative values of type exactly ``int`` or
    ``float`` (booleans are rejected). Any container, shape, length,
    type, range, or finiteness violation (including ``OverflowError``
    raised by ``math.isfinite``) raises ValueError.

    After validation the values and weights are converted to ``float``
    in input order and ``F`` denotes ``math.fsum``. The weight totals
    are ``W = F(w_i)`` and ``V = F(v_j)``; if either summation
    overflows or is invalid, is non-finite, or its total is less than
    or equal to zero, a ValueError is raised. For points ``a`` and
    ``b`` the distance is
    ``D(a, b) = sqrt(F((a_h - b_h) ** 2 for increasing h)) ** beta``.
    For sample ``i`` with target ``y_i`` and members ``x_ij``,
    ``A = F(v_j * D(x_ij, y_i) for increasing j) / V`` and
    ``B = F(v_j * v_k * D(x_ij, x_ik) with j outer and k inner) /
    (2 * V * V)``; the row score is ``q_i = A - B``. The result is
    ``F(w_i * q_i for increasing i) / W``. Any other conversion,
    square root, power, arithmetic, or ``math.fsum`` step that raises
    ``OverflowError``, ``ValueError``, or ``ZeroDivisionError``, and
    any non-finite intermediate value or result, raises
    FloatingPointError. An exact zero result is normalized to ``0.0``.

    The return value is a float. The inputs are not modified.
    Deterministic: same inputs, same result.
    """
    if not isinstance(y_true, list) or len(y_true) == 0:
        raise ValueError("y_true must be a non-empty list of rows")
    n = len(y_true)
    d = None
    for row in y_true:
        if not isinstance(row, list) or len(row) == 0:
            raise ValueError("y_true rows must be non-empty lists")
        if d is None:
            d = len(row)
        elif len(row) != d:
            raise ValueError("y_true must be rectangular")
        for value in row:
            if type(value) not in (int, float):
                raise ValueError(
                    "y_true must contain only finite non-boolean numbers"
                )
            # math.isfinite raises OverflowError for ints too large to
            # convert to float; such values fail the finite requirement.
            try:
                finite = math.isfinite(value)
            except OverflowError as exc:
                raise ValueError(
                    "y_true must contain only finite non-boolean numbers"
                ) from exc
            if not finite:
                raise ValueError(
                    "y_true must contain only finite non-boolean numbers"
                )

    if not isinstance(y_samples, list) or len(y_samples) != n:
        raise ValueError(
            "y_samples must be a list with one group per row of y_true"
        )
    m = None
    for group in y_samples:
        if not isinstance(group, list) or len(group) == 0:
            raise ValueError("y_samples groups must be non-empty lists of rows")
        if m is None:
            m = len(group)
        elif len(group) != m:
            raise ValueError("y_samples groups must share a common size")
        for row in group:
            if not isinstance(row, list) or len(row) != d:
                raise ValueError(
                    "y_samples rows must be lists with the same width as "
                    "the rows of y_true"
                )
            for value in row:
                if type(value) not in (int, float):
                    raise ValueError(
                        "y_samples must contain only finite non-boolean "
                        "numbers"
                    )
                try:
                    finite = math.isfinite(value)
                except OverflowError as exc:
                    raise ValueError(
                        "y_samples must contain only finite non-boolean "
                        "numbers"
                    ) from exc
                if not finite:
                    raise ValueError(
                        "y_samples must contain only finite non-boolean "
                        "numbers"
                    )

    if type(beta) not in (int, float):
        raise ValueError(
            "beta must be a finite non-boolean number with 0 < beta < 2"
        )
    try:
        beta_finite = math.isfinite(beta)
    except OverflowError as exc:
        raise ValueError(
            "beta must be a finite non-boolean number with 0 < beta < 2"
        ) from exc
    if not beta_finite or not 0 < beta < 2:
        raise ValueError(
            "beta must be a finite non-boolean number with 0 < beta < 2"
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

    if ensemble_weight is not None:
        if (
            not isinstance(ensemble_weight, list)
            or len(ensemble_weight) != m
        ):
            raise ValueError(
                "ensemble_weight must be a list with the same length as "
                "the number of members of y_samples"
            )
        for value in ensemble_weight:
            if type(value) not in (int, float):
                raise ValueError(
                    "ensemble_weight must contain only finite "
                    "non-negative non-boolean numbers"
                )
            try:
                finite = math.isfinite(value)
            except OverflowError as exc:
                raise ValueError(
                    "ensemble_weight must contain only finite "
                    "non-negative non-boolean numbers"
                ) from exc
            if not finite or value < 0:
                raise ValueError(
                    "ensemble_weight must contain only finite "
                    "non-negative non-boolean numbers"
                )

    def non_finite(exc):
        return FloatingPointError(
            "non-finite value encountered during energy score"
        )

    try:
        y = [[float(value) for value in row] for row in y_true]
        x = [
            [[float(value) for value in row] for row in group]
            for group in y_samples
        ]
        if sample_weight is None:
            w = [1.0] * n
        else:
            w = [float(value) for value in sample_weight]
        if ensemble_weight is None:
            v = [1.0] * m
        else:
            v = [float(value) for value in ensemble_weight]
        beta_value = float(beta)
    except (OverflowError, ValueError) as exc:
        raise non_finite(exc) from exc
    for values in [w, v] + y + [row for group in x for row in group]:
        for value in values:
            if not math.isfinite(value):
                raise non_finite(None)
    if not math.isfinite(beta_value):
        raise non_finite(None)

    try:
        total_sample_weight = math.fsum(w)
        total_ensemble_weight = math.fsum(v)
    except (OverflowError, ValueError) as exc:
        raise ValueError("the total weight must be greater than 0") from exc
    if (
        not math.isfinite(total_sample_weight)
        or total_sample_weight <= 0.0
        or not math.isfinite(total_ensemble_weight)
        or total_ensemble_weight <= 0.0
    ):
        raise ValueError("the total weight must be greater than 0")

    def distance(a, b):
        squared_terms = []
        for h in range(d):
            try:
                term = (a[h] - b[h]) ** 2
            except (OverflowError, ValueError) as exc:
                raise non_finite(exc) from exc
            if not math.isfinite(term):
                raise non_finite(None)
            squared_terms.append(term)
        try:
            squared_sum = math.fsum(squared_terms)
            root = math.sqrt(squared_sum)
            value = root ** beta_value
        except (OverflowError, ValueError, ZeroDivisionError) as exc:
            raise non_finite(exc) from exc
        if (
            not math.isfinite(squared_sum)
            or not math.isfinite(root)
            or not math.isfinite(value)
        ):
            raise non_finite(None)
        return value

    row_scores = []
    for i in range(n):
        member_terms = []
        pairwise_terms = []
        for j in range(m):
            try:
                member_term = v[j] * distance(x[i][j], y[i])
            except (OverflowError, ValueError) as exc:
                raise non_finite(exc) from exc
            if not math.isfinite(member_term):
                raise non_finite(None)
            member_terms.append(member_term)
            for k in range(m):
                try:
                    pairwise_term = v[j] * v[k] * distance(
                        x[i][j], x[i][k]
                    )
                except (OverflowError, ValueError) as exc:
                    raise non_finite(exc) from exc
                if not math.isfinite(pairwise_term):
                    raise non_finite(None)
                pairwise_terms.append(pairwise_term)
        try:
            member_sum = math.fsum(member_terms)
            pairwise_sum = math.fsum(pairwise_terms)
        except (OverflowError, ValueError) as exc:
            raise non_finite(exc) from exc
        if not math.isfinite(member_sum) or not math.isfinite(pairwise_sum):
            raise non_finite(None)
        try:
            member_part = member_sum / total_ensemble_weight
            pairwise_denominator = (
                2.0 * total_ensemble_weight * total_ensemble_weight
            )
            pairwise_part = pairwise_sum / pairwise_denominator
            row_score = member_part - pairwise_part
        except (
            OverflowError,
            ValueError,
            ZeroDivisionError,
        ) as exc:
            raise non_finite(exc) from exc
        if (
            not math.isfinite(pairwise_denominator)
            or not math.isfinite(member_part)
            or not math.isfinite(pairwise_part)
            or not math.isfinite(row_score)
        ):
            raise non_finite(None)
        row_scores.append(row_score)

    weighted_terms = []
    for i in range(n):
        try:
            term = w[i] * row_scores[i]
        except (OverflowError, ValueError) as exc:
            raise non_finite(exc) from exc
        if not math.isfinite(term):
            raise non_finite(None)
        weighted_terms.append(term)
    try:
        weighted_total = math.fsum(weighted_terms)
    except (OverflowError, ValueError) as exc:
        raise non_finite(exc) from exc
    if not math.isfinite(weighted_total):
        raise non_finite(None)
    try:
        result = weighted_total / total_sample_weight
    except (OverflowError, ValueError, ZeroDivisionError) as exc:
        raise non_finite(exc) from exc
    if not math.isfinite(result):
        raise non_finite(None)
    if result == 0:
        return 0.0
    return result


# ``from __future__ import annotations`` stores annotations as strings;
# expose the builtin ``float`` as the runtime return annotation.
energy_score.__annotations__["return"] = float


def interval_score(
    y_true, y_lower, y_upper, alpha=0.05, sample_weight=None
) -> float:
    """Return the weighted mean interval score.

    ``y_true``, ``y_lower``, and ``y_upper`` must be non-empty lists of
    the same length whose elements are finite values of type exactly
    ``int`` or ``float`` (booleans are rejected); in addition
    ``y_lower[i] <= y_upper[i]`` must hold for every index. ``alpha``
    must be a finite value of type exactly ``int`` or ``float``
    (booleans are rejected) with ``0 < alpha < 1``. ``sample_weight``
    must be ``None`` -- every sample then weighs ``1.0`` -- or a list
    with the same length as the other inputs whose elements are finite
    non-negative values of type exactly ``int`` or ``float`` (booleans
    are rejected). Any container, length, type, range, or finiteness
    violation (including ``OverflowError`` raised by ``math.isfinite``)
    raises ValueError.

    After validation the values and weights are converted to ``float``
    in input order. ``math.fsum`` computes the total weight
    ``W = fsum(w_i)``; if that summation overflows or is invalid, is
    non-finite, or its total is less than or equal to zero, a ValueError
    is raised. For sample ``i`` with target ``t``, lower bound ``l``,
    and upper bound ``u`` the width term is ``s = u - l``; when
    ``t < l`` the penalty ``(2 / alpha) * (l - t)`` is added, when
    ``t > u`` the penalty ``(2 / alpha) * (t - u)`` is added, and no
    penalty is added otherwise. The terms are combined as
    ``L = fsum(w_i * s_i in input order)`` and the result is ``L / W``.
    Apart from the ``W`` check, overflow or invalid operations during
    the post-validation conversion, the bound comparisons, the
    arithmetic, or the ``L`` summation, a zero division, and non-finite
    intermediate values or results raise FloatingPointError. An exact
    zero result is normalized to ``0.0``.

    The return value is a float. The inputs are not modified.
    Deterministic: same inputs, same result.
    """
    if (
        not isinstance(y_true, list)
        or not isinstance(y_lower, list)
        or not isinstance(y_upper, list)
    ):
        raise ValueError(
            "y_true, y_lower, and y_upper must be non-empty lists of the "
            "same length"
        )
    n = len(y_true)
    if n == 0 or len(y_lower) != n or len(y_upper) != n:
        raise ValueError(
            "y_true, y_lower, and y_upper must be non-empty lists of the "
            "same length"
        )
    for name, values in (
        ("y_true", y_true),
        ("y_lower", y_lower),
        ("y_upper", y_upper),
    ):
        for value in values:
            if type(value) not in (int, float):
                raise ValueError(
                    name + " must contain only finite non-boolean numbers"
                )
            # math.isfinite raises OverflowError for ints too large to
            # convert to float; such values fail the finite requirement.
            try:
                finite = math.isfinite(value)
            except OverflowError as exc:
                raise ValueError(
                    name + " must contain only finite non-boolean numbers"
                ) from exc
            if not finite:
                raise ValueError(
                    name + " must contain only finite non-boolean numbers"
                )
    for i in range(n):
        if not y_lower[i] <= y_upper[i]:
            raise ValueError(
                "each y_lower element must be less than or equal to the "
                "corresponding y_upper element"
            )

    if type(alpha) not in (int, float):
        raise ValueError("alpha must be a finite number with 0 < alpha < 1")
    try:
        finite = math.isfinite(alpha)
    except OverflowError as exc:
        raise ValueError(
            "alpha must be a finite number with 0 < alpha < 1"
        ) from exc
    if not finite or not 0 < alpha < 1:
        raise ValueError("alpha must be a finite number with 0 < alpha < 1")

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

    def non_finite(exc):
        return FloatingPointError(
            "non-finite value encountered during interval score"
        )

    try:
        t = [float(value) for value in y_true]
        l = [float(value) for value in y_lower]
        u = [float(value) for value in y_upper]
        if sample_weight is None:
            w = [1.0] * n
        else:
            w = [float(value) for value in sample_weight]
        a = float(alpha)
    except (OverflowError, ValueError) as exc:
        raise non_finite(exc) from exc
    for values in (t, l, u, w):
        for value in values:
            if not math.isfinite(value):
                raise non_finite(None)
    if not math.isfinite(a):
        raise non_finite(None)

    try:
        total_weight = math.fsum(w)
    except (OverflowError, ValueError) as exc:
        raise ValueError("the total weight must be greater than 0") from exc
    if not math.isfinite(total_weight) or total_weight <= 0.0:
        raise ValueError("the total weight must be greater than 0")

    terms = []
    for i in range(n):
        try:
            width = u[i] - l[i]
            if t[i] < l[i]:
                width += (2.0 / a) * (l[i] - t[i])
            elif t[i] > u[i]:
                width += (2.0 / a) * (t[i] - u[i])
            term = w[i] * width
        except (
            OverflowError,
            ValueError,
            ZeroDivisionError,
        ) as exc:
            raise non_finite(exc) from exc
        if not math.isfinite(width) or not math.isfinite(term):
            raise non_finite(None)
        terms.append(term)

    try:
        total = math.fsum(terms)
    except (OverflowError, ValueError) as exc:
        raise non_finite(exc) from exc
    if not math.isfinite(total):
        raise non_finite(None)
    try:
        result = total / total_weight
    except (OverflowError, ValueError, ZeroDivisionError) as exc:
        raise non_finite(exc) from exc
    if not math.isfinite(result):
        raise non_finite(None)
    if result == 0:
        return 0.0
    return result


# ``from __future__ import annotations`` stores annotations as strings;
# expose the builtin ``float`` as the runtime return annotation.
interval_score.__annotations__["return"] = float


def weighted_interval_score(y, m, lo, hi, a, w=None) -> float:
    """Return the weighted mean multi-interval weighted interval score.

    ``y`` and ``m`` must be non-empty lists of the same length whose
    elements are finite values of type exactly ``int`` or ``float``
    (booleans are rejected); ``y`` holds the targets and ``m`` the
    median predictions. ``a`` must be a non-empty list of finite values
    of type exactly ``int`` or ``float`` (booleans are rejected), each
    strictly between ``0`` and ``1``, in strictly increasing order.
    ``lo`` and ``hi`` must be lists of rows with the same height as
    ``y`` and a common width equal to ``len(a)``; their elements are
    likewise finite values of type exactly ``int`` or ``float``
    (booleans are rejected), and ``lo[i][k] <= hi[i][k]`` must hold for
    every pair. ``w`` must be ``None`` -- every sample then weighs
    ``1.0`` -- or a list with the same length as ``y`` whose elements
    are finite non-negative values of type exactly ``int`` or ``float``
    (booleans are rejected). Any container, length, shape, type, range,
    ordering, or finiteness violation (including ``OverflowError``
    raised by ``math.isfinite``) raises ValueError.

    After validation the values and weights are converted to ``float``
    in input order. ``math.fsum`` computes the total weight
    ``W = fsum(w_i)``; if that summation overflows or is invalid, is
    non-finite, or its total is less than or equal to zero, a ValueError
    is raised. For sample ``i`` and interval ``k`` the interval width
    is ``I = hi[i][k] - lo[i][k]``; when ``y[i] < lo[i][k]`` the penalty
    ``2 * (lo[i][k] - y[i]) / a[k]`` is added, when
    ``y[i] > hi[i][k]`` the penalty ``2 * (y[i] - hi[i][k]) / a[k]`` is
    added, and no penalty is added otherwise. The per-sample score is
    ``q_i = (0.5 * abs(y[i] - m[i]) + fsum((a[k] / 2) * I_k in
    increasing-k order)) / (len(a) + 0.5)``. The result is
    ``fsum(w_i * q_i in sample order) / W``. Apart from the ``W``
    check, overflow or invalid operations during the post-validation
    conversion, comparisons, absolute value, arithmetic, or
    ``math.fsum`` steps, and non-finite intermediate values or results
    raise FloatingPointError. An exact zero result is normalized to
    ``0.0``.

    The return value is a float. The inputs are not modified.
    Deterministic: same inputs, same result.
    """
    if not isinstance(y, list) or len(y) == 0:
        raise ValueError("y must be a non-empty list")
    n = len(y)
    if not isinstance(m, list) or len(m) != n:
        raise ValueError("m must be a list with the same length as y")
    for name, values in (("y", y), ("m", m)):
        for value in values:
            if type(value) not in (int, float):
                raise ValueError(
                    name + " must contain only finite non-boolean numbers"
                )
            # math.isfinite raises OverflowError for ints too large to
            # convert to float; such values fail the finite requirement.
            try:
                finite = math.isfinite(value)
            except OverflowError as exc:
                raise ValueError(
                    name + " must contain only finite non-boolean numbers"
                ) from exc
            if not finite:
                raise ValueError(
                    name + " must contain only finite non-boolean numbers"
                )

    if not isinstance(a, list) or len(a) == 0:
        raise ValueError(
            "a must be a non-empty list of strictly increasing numbers "
            "with 0 < a[k] < 1"
        )
    previous = None
    for value in a:
        if type(value) not in (int, float):
            raise ValueError(
                "a must contain only finite non-boolean numbers with "
                "0 < a[k] < 1"
            )
        try:
            finite = math.isfinite(value)
        except OverflowError as exc:
            raise ValueError(
                "a must contain only finite non-boolean numbers with "
                "0 < a[k] < 1"
            ) from exc
        if not finite or not 0 < value < 1:
            raise ValueError(
                "a must be a non-empty list of strictly increasing "
                "numbers with 0 < a[k] < 1"
            )
        if previous is not None and not value > previous:
            raise ValueError(
                "a must be a non-empty list of strictly increasing "
                "numbers with 0 < a[k] < 1"
            )
        previous = value
    p = len(a)

    if not isinstance(lo, list) or len(lo) != n:
        raise ValueError(
            "lo must be a list of rows with the same length as y and the "
            "same width as a"
        )
    if not isinstance(hi, list) or len(hi) != n:
        raise ValueError(
            "hi must be a list of rows with the same length as y and the "
            "same width as a"
        )
    for name, rows in (("lo", lo), ("hi", hi)):
        for row in rows:
            if not isinstance(row, list) or len(row) != p:
                raise ValueError(
                    name
                    + " must be a rectangular matrix with one row per "
                    "sample and one column per entry of a"
                )
            for value in row:
                if type(value) not in (int, float):
                    raise ValueError(
                        name
                        + " must contain only finite non-boolean numbers"
                    )
                try:
                    finite = math.isfinite(value)
                except OverflowError as exc:
                    raise ValueError(
                        name
                        + " must contain only finite non-boolean numbers"
                    ) from exc
                if not finite:
                    raise ValueError(
                        name
                        + " must contain only finite non-boolean numbers"
                    )
    for i in range(n):
        for k in range(p):
            if not lo[i][k] <= hi[i][k]:
                raise ValueError(
                    "each lo element must be less than or equal to the "
                    "corresponding hi element"
                )

    if w is not None:
        if not isinstance(w, list) or len(w) != n:
            raise ValueError(
                "w must be a list with the same length as y"
            )
        for value in w:
            if type(value) not in (int, float):
                raise ValueError(
                    "w must contain only finite non-negative "
                    "non-boolean numbers"
                )
            try:
                finite = math.isfinite(value)
            except OverflowError as exc:
                raise ValueError(
                    "w must contain only finite non-negative "
                    "non-boolean numbers"
                ) from exc
            if not finite or value < 0:
                raise ValueError(
                    "w must contain only finite non-negative "
                    "non-boolean numbers"
                )

    def non_finite(exc):
        return FloatingPointError(
            "non-finite value encountered during weighted interval score"
        )

    try:
        t = [float(value) for value in y]
        med = [float(value) for value in m]
        lower = [[float(value) for value in row] for row in lo]
        upper = [[float(value) for value in row] for row in hi]
        alphas = [float(value) for value in a]
        if w is None:
            weights = [1.0] * n
        else:
            weights = [float(value) for value in w]
    except (OverflowError, ValueError) as exc:
        raise non_finite(exc) from exc
    for values in [t, med, alphas, weights] + lower + upper:
        for value in values:
            if not math.isfinite(value):
                raise non_finite(None)

    try:
        total_weight = math.fsum(weights)
    except (OverflowError, ValueError) as exc:
        raise ValueError("the total weight must be greater than 0") from exc
    if not math.isfinite(total_weight) or total_weight <= 0.0:
        raise ValueError("the total weight must be greater than 0")

    row_scores = []
    for i in range(n):
        interval_terms = []
        for k in range(p):
            try:
                width = upper[i][k] - lower[i][k]
                if t[i] < lower[i][k]:
                    width += (2.0 / alphas[k]) * (lower[i][k] - t[i])
                elif t[i] > upper[i][k]:
                    width += (2.0 / alphas[k]) * (t[i] - upper[i][k])
                term = (alphas[k] / 2.0) * width
            except (
                OverflowError,
                ValueError,
                ZeroDivisionError,
            ) as exc:
                raise non_finite(exc) from exc
            if not math.isfinite(width) or not math.isfinite(term):
                raise non_finite(None)
            interval_terms.append(term)
        try:
            median_distance = abs(t[i] - med[i])
            median_term = 0.5 * median_distance
            interval_sum = math.fsum(interval_terms)
            numerator = median_term + interval_sum
            row_score = numerator / (p + 0.5)
        except (OverflowError, ValueError) as exc:
            raise non_finite(exc) from exc
        if (
            not math.isfinite(median_distance)
            or not math.isfinite(median_term)
            or not math.isfinite(interval_sum)
            or not math.isfinite(numerator)
            or not math.isfinite(row_score)
        ):
            raise non_finite(None)
        row_scores.append(row_score)

    weighted_terms = []
    for i in range(n):
        try:
            term = weights[i] * row_scores[i]
        except (OverflowError, ValueError) as exc:
            raise non_finite(exc) from exc
        if not math.isfinite(term):
            raise non_finite(None)
        weighted_terms.append(term)
    try:
        weighted_total = math.fsum(weighted_terms)
    except (OverflowError, ValueError) as exc:
        raise non_finite(exc) from exc
    if not math.isfinite(weighted_total):
        raise non_finite(None)
    try:
        result = weighted_total / total_weight
    except (OverflowError, ValueError, ZeroDivisionError) as exc:
        raise non_finite(exc) from exc
    if not math.isfinite(result):
        raise non_finite(None)
    if result == 0:
        return 0.0
    return result


# ``from __future__ import annotations`` stores annotations as strings;
# expose the builtin ``float`` as the runtime return annotation.
weighted_interval_score.__annotations__["return"] = float


def concordance_index(event_time, risk_score, event_observed=None, sample_weight=None) -> float:
    """Return the (optionally weighted) concordance index.

    ``event_time`` and ``risk_score`` must be non-empty lists of equal
    length whose elements are finite values of type exactly ``int`` or
    ``float`` (booleans are rejected); every event time must be
    non-negative. ``event_observed`` must be ``None`` or a list of the
    same length whose elements are exactly the integers ``0`` or ``1``
    (booleans are rejected); ``None`` is treated as all ones.
    ``sample_weight`` must be ``None`` or a list of the same length
    whose elements are finite non-negative values of type exactly
    ``int`` or ``float``; ``None`` is treated as all ``1.0``. Any
    container, length, type, range, or finiteness violation (including
    ``OverflowError`` raised by ``math.isfinite``) raises ValueError.

    After validation the values are converted to ``float``. Pairs are
    scanned with ``i`` outer and ``j`` inner for ``i < j``: a pair is
    comparable only when the earlier time belongs to a sample with an
    observed event (``time_i < time_j`` and ``event_i == 1``, or
    symmetrically); pairs with equal times or whose earlier sample has
    no observed event are ignored. For a comparable pair with earlier
    event sample ``a`` and other sample ``b``, ``q = w_a * w_b`` adds
    ``q`` to the concordant weight when ``risk_a > risk_b``, ``0.5 * q``
    when the risks are equal, and nothing otherwise; ``q`` always adds
    to the total weight. ``math.fsum`` computes the total weight ``W``
    and the concordant weight ``C``; ``W <= 0`` raises ValueError,
    otherwise the result is ``C / W``. Overflow, invalid operations, or
    division by zero during the post-validation conversion, arithmetic,
    or summation, and non-finite intermediate values or results, raise
    FloatingPointError.

    An exact zero result is normalized to positive ``0.0``. The return
    value is a float. The inputs are not modified. Deterministic: same
    inputs, same result.
    """
    n = _check_metric_vectors(event_time, risk_score)
    for name, values in (("event_time", event_time), ("risk_score", risk_score)):
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
    for value in event_time:
        if value < 0:
            raise ValueError("event_time must contain only non-negative values")

    if event_observed is None:
        events = [1] * n
    else:
        if not isinstance(event_observed, list):
            raise ValueError("event_observed must be None or a list")
        if len(event_observed) != n:
            raise ValueError(
                "event_observed must have the same length as event_time"
            )
        events = []
        for value in event_observed:
            if type(value) is not int or value not in (0, 1):
                raise ValueError("event_observed must contain only 0 or 1")
            events.append(value)

    if sample_weight is None:
        weights = [1.0] * n
    else:
        if not isinstance(sample_weight, list):
            raise ValueError("sample_weight must be None or a list")
        if len(sample_weight) != n:
            raise ValueError(
                "sample_weight must have the same length as event_time"
            )
        weights = []
        for value in sample_weight:
            if type(value) not in (int, float):
                raise ValueError(
                    "sample_weight must contain only finite non-boolean numbers"
                )
            try:
                finite = math.isfinite(value)
            except OverflowError as exc:
                raise ValueError(
                    "sample_weight must contain only finite non-boolean numbers"
                ) from exc
            if not finite:
                raise ValueError(
                    "sample_weight must contain only finite non-boolean numbers"
                )
            if value < 0:
                raise ValueError(
                    "sample_weight must contain only non-negative values"
                )
            weights.append(value)

    def non_finite():
        return FloatingPointError(
            "non-finite value encountered during concordance index"
        )

    times = []
    risks = []
    for i in range(n):
        try:
            time_value = float(event_time[i])
            risk_value = float(risk_score[i])
            weight_value = float(weights[i])
        except (OverflowError, ValueError) as exc:
            raise non_finite() from exc
        if not math.isfinite(time_value):
            raise non_finite()
        if not math.isfinite(risk_value):
            raise non_finite()
        if not math.isfinite(weight_value):
            raise non_finite()
        times.append(time_value)
        risks.append(risk_value)
        weights[i] = weight_value

    total_terms = []
    concordant_terms = []
    for i in range(n):
        for j in range(i + 1, n):
            if times[i] < times[j] and events[i] == 1:
                a, b = i, j
            elif times[j] < times[i] and events[j] == 1:
                a, b = j, i
            else:
                continue
            try:
                q = weights[a] * weights[b]
            except (OverflowError, ValueError) as exc:
                raise non_finite() from exc
            if not math.isfinite(q):
                raise non_finite()
            total_terms.append(q)
            if risks[a] > risks[b]:
                concordant_terms.append(q)
            elif risks[a] == risks[b]:
                concordant_terms.append(0.5 * q)

    try:
        total_weight = math.fsum(total_terms)
    except (OverflowError, ValueError) as exc:
        raise non_finite() from exc
    if not math.isfinite(total_weight):
        raise non_finite()
    if total_weight <= 0:
        raise ValueError("no comparable pairs with positive total weight")
    try:
        concordant_weight = math.fsum(concordant_terms)
    except (OverflowError, ValueError) as exc:
        raise non_finite() from exc
    if not math.isfinite(concordant_weight):
        raise non_finite()
    try:
        result = concordant_weight / total_weight
    except (OverflowError, ValueError, ZeroDivisionError) as exc:
        raise non_finite() from exc
    if not math.isfinite(result):
        raise non_finite()
    if result == 0:
        return 0.0
    return result


# ``from __future__ import annotations`` stores annotations as strings;
# expose the builtin ``float`` as the runtime return annotation.
concordance_index.__annotations__["return"] = float


def integrated_brier_score(
    event_time, event_observed, survival_prob, times
) -> float:
    """Return the integrated Brier score of survival predictions.

    ``event_time`` (``T``) must be a non-empty list of finite
    non-negative values of type exactly ``int`` or ``float`` (booleans
    are rejected). ``event_observed`` (``E``) must be a list of the same
    length whose elements are exactly the integers ``0`` or ``1``
    (booleans are rejected); ``1`` marks an observed event and ``0`` a
    censoring. ``times`` (``t``) must be a list of at least two finite
    non-negative values of type exactly ``int`` or ``float`` in strictly
    increasing order. ``survival_prob`` (``S``) must be a
    ``len(T)`` by ``len(t)`` two-dimensional list whose entries are
    finite values of type exactly ``int`` or ``float`` in ``[0, 1]``;
    each row must be non-increasing. Any container, length, type,
    range, ordering, or finiteness violation (including
    ``OverflowError`` raised by ``math.isfinite``) raises ValueError.

    After validation all values are converted to ``float``. For the
    sorted unique event times ``u``, the Kaplan-Meier estimator of the
    censoring distribution is built as ``R = #(T >= u)``,
    ``D = #(T == u and E == 0)`` and
    ``G(u) = G(u-) * (1 - D / R)`` starting from ``G = 1``. At each
    evaluation time ``t_j`` the Brier score is the ``math.fsum`` over
    samples in ascending order of
    ``E_i * I(T_i <= t_j) * S_ij**2 / G(T_i-)`` plus
    ``I(T_i > t_j) * (1 - S_ij)**2 / G(t_j)``, divided by the number of
    samples; ``G(t_j)`` is the right-continuous step value of the
    estimator. A required censoring probability that is less than or
    equal to zero raises ValueError.

    The trapezoidal rule integrates the Brier scores over ``times`` and
    the result is divided by ``t[-1] - t[0]``. Overflow, invalid
    operations, or division by zero during the post-validation
    conversion, arithmetic, or summation, and non-finite intermediate
    values or results, raise FloatingPointError. An exact zero result is
    normalized to positive ``0.0``. The return value is a float. The
    inputs are not modified. Deterministic: same inputs, same result.
    """
    if not isinstance(event_time, list) or not isinstance(event_observed, list):
        raise ValueError("event_time and event_observed must be lists")
    n = len(event_time)
    if n == 0:
        raise ValueError("event_time must be a non-empty list")
    if len(event_observed) != n:
        raise ValueError(
            "event_observed must have the same length as event_time"
        )
    for value in event_time:
        if type(value) not in (int, float):
            raise ValueError(
                "event_time must contain only finite non-boolean numbers"
            )
        # math.isfinite raises OverflowError for ints too large to
        # convert to float; such values fail the finite requirement.
        try:
            finite = math.isfinite(value)
        except OverflowError as exc:
            raise ValueError(
                "event_time must contain only finite non-boolean numbers"
            ) from exc
        if not finite:
            raise ValueError(
                "event_time must contain only finite non-boolean numbers"
            )
        if value < 0:
            raise ValueError("event_time must contain only non-negative values")
    for value in event_observed:
        if type(value) is not int or value not in (0, 1):
            raise ValueError("event_observed must contain only 0 or 1")

    if not isinstance(times, list) or len(times) < 2:
        raise ValueError("times must be a list of at least two values")
    previous_time = None
    for value in times:
        if type(value) not in (int, float):
            raise ValueError(
                "times must contain only finite non-boolean numbers"
            )
        try:
            finite = math.isfinite(value)
        except OverflowError as exc:
            raise ValueError(
                "times must contain only finite non-boolean numbers"
            ) from exc
        if not finite:
            raise ValueError(
                "times must contain only finite non-boolean numbers"
            )
        if value < 0:
            raise ValueError("times must contain only non-negative values")
        if previous_time is not None and not value > previous_time:
            raise ValueError("times must be strictly increasing")
        previous_time = value

    if not isinstance(survival_prob, list) or len(survival_prob) != n:
        raise ValueError(
            "survival_prob must be a list with one row per event time"
        )
    m = len(times)
    for row in survival_prob:
        if not isinstance(row, list) or len(row) != m:
            raise ValueError(
                "each survival_prob row must have one value per time"
            )
        previous_prob = None
        for value in row:
            if type(value) not in (int, float):
                raise ValueError(
                    "survival_prob must contain only finite non-boolean "
                    "numbers in [0, 1]"
                )
            try:
                finite = math.isfinite(value)
            except OverflowError as exc:
                raise ValueError(
                    "survival_prob must contain only finite non-boolean "
                    "numbers in [0, 1]"
                ) from exc
            if not finite or value < 0 or value > 1:
                raise ValueError(
                    "survival_prob must contain only finite non-boolean "
                    "numbers in [0, 1]"
                )
            if previous_prob is not None and value > previous_prob:
                raise ValueError(
                    "each survival_prob row must be non-increasing"
                )
            previous_prob = value

    def non_finite():
        return FloatingPointError(
            "non-finite value encountered during integrated brier score"
        )

    try:
        event_times = [float(value) for value in event_time]
        eval_times = [float(value) for value in times]
        survival = [[float(value) for value in row] for row in survival_prob]
    except (OverflowError, ValueError) as exc:
        raise non_finite() from exc
    for value in event_times:
        if not math.isfinite(value):
            raise non_finite()
    for value in eval_times:
        if not math.isfinite(value):
            raise non_finite()
    for row in survival:
        for value in row:
            if not math.isfinite(value):
                raise non_finite()

    # Kaplan-Meier estimator G of the censoring distribution:
    # g_before[u] is G(u-) and g_after[u] is G(u).
    unique_times = sorted(set(event_times))
    g_before = {}
    g_after = {}
    g = 1.0
    for u in unique_times:
        g_before[u] = g
        at_risk = 0
        censored = 0
        for i in range(n):
            if event_times[i] >= u:
                at_risk += 1
            if event_times[i] == u and event_observed[i] == 0:
                censored += 1
        try:
            g = g * (1.0 - censored / at_risk)
        except (OverflowError, ValueError, ZeroDivisionError) as exc:
            raise non_finite() from exc
        if not math.isfinite(g):
            raise non_finite()
        g_after[u] = g

    def g_at(value):
        # Right-continuous step value: largest unique time <= value.
        lo, hi = 0, len(unique_times)
        while lo < hi:
            mid = (lo + hi) // 2
            if unique_times[mid] <= value:
                lo = mid + 1
            else:
                hi = mid
        if lo == 0:
            return 1.0
        return g_after[unique_times[lo - 1]]

    brier = []
    for j in range(m):
        t_j = eval_times[j]
        g_tj = None
        terms = []
        for i in range(n):
            s_ij = survival[i][j]
            term = 0.0
            if event_observed[i] == 1 and event_times[i] <= t_j:
                weight = g_before[event_times[i]]
                if weight <= 0.0:
                    raise ValueError(
                        "the censoring survival probability must be "
                        "greater than 0"
                    )
                try:
                    term = term + (s_ij * s_ij) / weight
                except (OverflowError, ValueError, ZeroDivisionError) as exc:
                    raise non_finite() from exc
            if event_times[i] > t_j:
                if g_tj is None:
                    g_tj = g_at(t_j)
                    if g_tj <= 0.0:
                        raise ValueError(
                            "the censoring survival probability must be "
                            "greater than 0"
                        )
                try:
                    term = term + ((1.0 - s_ij) * (1.0 - s_ij)) / g_tj
                except (OverflowError, ValueError, ZeroDivisionError) as exc:
                    raise non_finite() from exc
            if not math.isfinite(term):
                raise non_finite()
            terms.append(term)
        try:
            b_j = math.fsum(terms) / n
        except (OverflowError, ValueError, ZeroDivisionError) as exc:
            raise non_finite() from exc
        if not math.isfinite(b_j):
            raise non_finite()
        brier.append(b_j)

    areas = []
    for j in range(m - 1):
        try:
            area = (
                (eval_times[j + 1] - eval_times[j])
                * (brier[j] + brier[j + 1])
                / 2.0
            )
        except (OverflowError, ValueError, ZeroDivisionError) as exc:
            raise non_finite() from exc
        if not math.isfinite(area):
            raise non_finite()
        areas.append(area)
    try:
        total = math.fsum(areas)
        span = eval_times[-1] - eval_times[0]
        result = total / span
    except (OverflowError, ValueError, ZeroDivisionError) as exc:
        raise non_finite() from exc
    if not math.isfinite(total) or not math.isfinite(span):
        raise non_finite()
    if not math.isfinite(result):
        raise non_finite()
    if result == 0:
        return 0.0
    return result


# ``from __future__ import annotations`` stores annotations as strings;
# expose the builtin ``float`` as the runtime return annotation.
integrated_brier_score.__annotations__["return"] = float


def cumulative_dynamic_auc(
    event_time, event_observed, risk_score, times
) -> list[float]:
    """Return the cumulative/dynamic AUC of risk scores at each time.

    ``event_time`` (``T``) must be a non-empty list of finite
    non-negative values of type exactly ``int`` or ``float`` (booleans
    are rejected). ``event_observed`` (``E``) must be a list of the same
    length whose elements are exactly the integers ``0`` or ``1``
    (booleans are rejected); ``1`` marks an observed event and ``0`` a
    censoring. ``risk_score`` (``R``) must be a list of the same length
    of finite values of type exactly ``int`` or ``float``. ``times``
    (``t``) must be a non-empty list of finite non-negative values of
    type exactly ``int`` or ``float`` in strictly increasing order. Any
    container, length, type, range, ordering, or finiteness violation
    (including ``OverflowError`` raised by ``math.isfinite``) raises
    ValueError.

    After validation all values are converted to ``float``. For the
    sorted unique event times ``u``, the Kaplan-Meier estimator of the
    censoring distribution is built as ``R = #(T >= u)``,
    ``D = #(T == u and E == 0)`` and ``G(u) = G(u-) * (1 - D / R)``
    starting from ``G = 1``; the estimator is right-continuous and
    ``G(T_i-)`` is retained for every sample. At each evaluation time
    ``t`` the cases are the samples with ``E_i == 1`` and
    ``T_i <= t`` and the controls are the samples with ``T_j > t``;
    with ``w_i = 1 / G(T_i-)`` the sums are scanned with ``i`` outer
    and ``j`` inner: ``math.fsum`` computes ``W`` as the sum of ``w_i``
    repeated once per control and ``C`` as the sum of ``w_i`` when
    ``R_i > R_j``, ``0.5 * w_i`` when ``R_i == R_j``, and ``0.0``
    otherwise. The result at that time is ``C / W``. No cases, no
    controls, a required censoring probability less than or equal to
    zero, or ``W <= 0`` raises ValueError.

    Overflow, invalid operations, or division by zero during the
    post-validation conversion, the Kaplan-Meier computation, the
    arithmetic, or the summation, and non-finite intermediate values or
    results, raise FloatingPointError. An exact zero result is
    normalized to positive ``0.0``. The return value is a list of
    floats, one per evaluation time. The inputs are not modified.
    Deterministic: same inputs, same result.
    """
    if not isinstance(event_time, list) or not isinstance(event_observed, list):
        raise ValueError("event_time and event_observed must be lists")
    if not isinstance(risk_score, list):
        raise ValueError("risk_score must be a list")
    n = len(event_time)
    if n == 0:
        raise ValueError("event_time must be a non-empty list")
    if len(event_observed) != n:
        raise ValueError(
            "event_observed must have the same length as event_time"
        )
    if len(risk_score) != n:
        raise ValueError(
            "risk_score must have the same length as event_time"
        )
    for value in event_time:
        if type(value) not in (int, float):
            raise ValueError(
                "event_time must contain only finite non-boolean numbers"
            )
        # math.isfinite raises OverflowError for ints too large to
        # convert to float; such values fail the finite requirement.
        try:
            finite = math.isfinite(value)
        except OverflowError as exc:
            raise ValueError(
                "event_time must contain only finite non-boolean numbers"
            ) from exc
        if not finite:
            raise ValueError(
                "event_time must contain only finite non-boolean numbers"
            )
        if value < 0:
            raise ValueError("event_time must contain only non-negative values")
    for value in event_observed:
        if type(value) is not int or value not in (0, 1):
            raise ValueError("event_observed must contain only 0 or 1")
    for value in risk_score:
        if type(value) not in (int, float):
            raise ValueError(
                "risk_score must contain only finite non-boolean numbers"
            )
        try:
            finite = math.isfinite(value)
        except OverflowError as exc:
            raise ValueError(
                "risk_score must contain only finite non-boolean numbers"
            ) from exc
        if not finite:
            raise ValueError(
                "risk_score must contain only finite non-boolean numbers"
            )

    if not isinstance(times, list) or len(times) == 0:
        raise ValueError("times must be a non-empty list")
    previous_time = None
    for value in times:
        if type(value) not in (int, float):
            raise ValueError(
                "times must contain only finite non-boolean numbers"
            )
        try:
            finite = math.isfinite(value)
        except OverflowError as exc:
            raise ValueError(
                "times must contain only finite non-boolean numbers"
            ) from exc
        if not finite:
            raise ValueError(
                "times must contain only finite non-boolean numbers"
            )
        if value < 0:
            raise ValueError("times must contain only non-negative values")
        if previous_time is not None and not value > previous_time:
            raise ValueError("times must be strictly increasing")
        previous_time = value

    def non_finite():
        return FloatingPointError(
            "non-finite value encountered during cumulative dynamic auc"
        )

    try:
        event_times = [float(value) for value in event_time]
        risks = [float(value) for value in risk_score]
        eval_times = [float(value) for value in times]
    except (OverflowError, ValueError) as exc:
        raise non_finite() from exc
    for value in event_times:
        if not math.isfinite(value):
            raise non_finite()
    for value in risks:
        if not math.isfinite(value):
            raise non_finite()
    for value in eval_times:
        if not math.isfinite(value):
            raise non_finite()

    # Kaplan-Meier estimator G of the censoring distribution:
    # g_before[u] is G(u-) and g_after[u] is G(u).
    unique_times = sorted(set(event_times))
    g_before = {}
    g = 1.0
    for u in unique_times:
        g_before[u] = g
        at_risk = 0
        censored = 0
        for i in range(n):
            if event_times[i] >= u:
                at_risk += 1
            if event_times[i] == u and event_observed[i] == 0:
                censored += 1
        try:
            g = g * (1.0 - censored / at_risk)
        except (OverflowError, ValueError, ZeroDivisionError) as exc:
            raise non_finite() from exc
        if not math.isfinite(g):
            raise non_finite()

    aucs = []
    for t_j in eval_times:
        cases = []
        controls = []
        for i in range(n):
            if event_observed[i] == 1 and event_times[i] <= t_j:
                cases.append(i)
            if event_times[i] > t_j:
                controls.append(i)
        if not cases:
            raise ValueError("no cases at an evaluation time")
        if not controls:
            raise ValueError("no controls at an evaluation time")
        weights = []
        for i in cases:
            weight = g_before[event_times[i]]
            if weight <= 0.0:
                raise ValueError(
                    "the censoring survival probability must be "
                    "greater than 0"
                )
            try:
                w_i = 1.0 / weight
            except (OverflowError, ValueError, ZeroDivisionError) as exc:
                raise non_finite() from exc
            if not math.isfinite(w_i):
                raise non_finite()
            weights.append(w_i)
        total_terms = []
        concordant_terms = []
        for case_index, i in enumerate(cases):
            w_i = weights[case_index]
            for j in controls:
                total_terms.append(w_i)
                if risks[i] > risks[j]:
                    concordant_terms.append(w_i)
                elif risks[i] == risks[j]:
                    try:
                        concordant_terms.append(0.5 * w_i)
                    except (OverflowError, ValueError) as exc:
                        raise non_finite() from exc
                else:
                    concordant_terms.append(0.0)
        try:
            total_weight = math.fsum(total_terms)
        except (OverflowError, ValueError) as exc:
            raise non_finite() from exc
        if not math.isfinite(total_weight):
            raise non_finite()
        if total_weight <= 0:
            raise ValueError("no positive total weight at an evaluation time")
        try:
            concordant_weight = math.fsum(concordant_terms)
        except (OverflowError, ValueError) as exc:
            raise non_finite() from exc
        if not math.isfinite(concordant_weight):
            raise non_finite()
        try:
            result = concordant_weight / total_weight
        except (OverflowError, ValueError, ZeroDivisionError) as exc:
            raise non_finite() from exc
        if not math.isfinite(result):
            raise non_finite()
        if result == 0:
            result = 0.0
        aucs.append(result)
    return aucs


# ``from __future__ import annotations`` stores annotations as strings;
# expose the builtin generic as the runtime return annotation.
cumulative_dynamic_auc.__annotations__["return"] = list[float]


def survival_brier_score(
    event_time, event_observed, survival_prob, times
) -> list[float]:
    """Return the Brier score of survival predictions at each time.

    ``event_time`` (``T``) must be a non-empty list of finite
    non-negative values of type exactly ``int`` or ``float`` (booleans
    are rejected). ``event_observed`` (``E``) must be a list of the same
    length whose elements are exactly the integers ``0`` or ``1``
    (booleans are rejected); ``1`` marks an observed event and ``0`` a
    censoring. ``times`` (``t``) must be a non-empty list of finite
    non-negative values of type exactly ``int`` or ``float`` in strictly
    increasing order. ``survival_prob`` (``S``) must be a
    ``len(T)`` by ``len(t)`` two-dimensional list whose entries are
    finite values of type exactly ``int`` or ``float`` in ``[0, 1]``;
    each row must be non-increasing. Any container, length, type,
    range, ordering, or finiteness violation (including
    ``OverflowError`` raised by ``math.isfinite``) raises ValueError.

    After validation all values are converted to ``float``. For the
    sorted unique event times ``u``, the Kaplan-Meier estimator of the
    censoring distribution is built as ``R = #(T >= u)``,
    ``D = #(T == u and E == 0)`` and
    ``G(u) = G(u-) * (1 - D / R)`` starting from ``G = 1``; the
    estimator is right-continuous and ``G(T_i-)`` is retained for every
    sample. At each evaluation time ``t_j`` the Brier score is the
    ``math.fsum`` over samples in ascending order of
    ``E_i * I(T_i <= t_j) * S_ij**2 / G(T_i-)`` plus
    ``I(T_i > t_j) * (1 - S_ij)**2 / G(t_j)``, divided by the number of
    samples; ``G(t_j)`` is the right-continuous step value of the
    estimator. A required censoring probability that is less than or
    equal to zero raises ValueError.

    Overflow, invalid operations, or division by zero during the
    post-validation conversion, the Kaplan-Meier computation, the
    arithmetic, the squaring, or the summation, and non-finite
    intermediate values or results, raise FloatingPointError. An exact
    zero score is normalized to positive ``0.0``. The return value is a
    list of floats, one per evaluation time, in the order of ``times``.
    The inputs are not modified. Deterministic: same inputs, same
    result.
    """
    if not isinstance(event_time, list) or not isinstance(event_observed, list):
        raise ValueError("event_time and event_observed must be lists")
    n = len(event_time)
    if n == 0:
        raise ValueError("event_time must be a non-empty list")
    if len(event_observed) != n:
        raise ValueError(
            "event_observed must have the same length as event_time"
        )
    for value in event_time:
        if type(value) not in (int, float):
            raise ValueError(
                "event_time must contain only finite non-boolean numbers"
            )
        # math.isfinite raises OverflowError for ints too large to
        # convert to float; such values fail the finite requirement.
        try:
            finite = math.isfinite(value)
        except OverflowError as exc:
            raise ValueError(
                "event_time must contain only finite non-boolean numbers"
            ) from exc
        if not finite:
            raise ValueError(
                "event_time must contain only finite non-boolean numbers"
            )
        if value < 0:
            raise ValueError("event_time must contain only non-negative values")
    for value in event_observed:
        if type(value) is not int or value not in (0, 1):
            raise ValueError("event_observed must contain only 0 or 1")

    if not isinstance(times, list) or len(times) == 0:
        raise ValueError("times must be a non-empty list")
    previous_time = None
    for value in times:
        if type(value) not in (int, float):
            raise ValueError(
                "times must contain only finite non-boolean numbers"
            )
        try:
            finite = math.isfinite(value)
        except OverflowError as exc:
            raise ValueError(
                "times must contain only finite non-boolean numbers"
            ) from exc
        if not finite:
            raise ValueError(
                "times must contain only finite non-boolean numbers"
            )
        if value < 0:
            raise ValueError("times must contain only non-negative values")
        if previous_time is not None and not value > previous_time:
            raise ValueError("times must be strictly increasing")
        previous_time = value

    if not isinstance(survival_prob, list) or len(survival_prob) != n:
        raise ValueError(
            "survival_prob must be a list with one row per event time"
        )
    m = len(times)
    for row in survival_prob:
        if not isinstance(row, list) or len(row) != m:
            raise ValueError(
                "each survival_prob row must have one value per time"
            )
        previous_prob = None
        for value in row:
            if type(value) not in (int, float):
                raise ValueError(
                    "survival_prob must contain only finite non-boolean "
                    "numbers in [0, 1]"
                )
            try:
                finite = math.isfinite(value)
            except OverflowError as exc:
                raise ValueError(
                    "survival_prob must contain only finite non-boolean "
                    "numbers in [0, 1]"
                ) from exc
            if not finite or value < 0 or value > 1:
                raise ValueError(
                    "survival_prob must contain only finite non-boolean "
                    "numbers in [0, 1]"
                )
            if previous_prob is not None and value > previous_prob:
                raise ValueError(
                    "each survival_prob row must be non-increasing"
                )
            previous_prob = value

    def non_finite():
        return FloatingPointError(
            "non-finite value encountered during survival brier score"
        )

    try:
        event_times = [float(value) for value in event_time]
        eval_times = [float(value) for value in times]
        survival = [[float(value) for value in row] for row in survival_prob]
    except (OverflowError, ValueError) as exc:
        raise non_finite() from exc
    for value in event_times:
        if not math.isfinite(value):
            raise non_finite()
    for value in eval_times:
        if not math.isfinite(value):
            raise non_finite()
    for row in survival:
        for value in row:
            if not math.isfinite(value):
                raise non_finite()

    # Kaplan-Meier estimator G of the censoring distribution:
    # g_before[u] is G(u-) and g_after[u] is G(u).
    unique_times = sorted(set(event_times))
    g_before = {}
    g_after = {}
    g = 1.0
    for u in unique_times:
        g_before[u] = g
        at_risk = 0
        censored = 0
        for i in range(n):
            if event_times[i] >= u:
                at_risk += 1
            if event_times[i] == u and event_observed[i] == 0:
                censored += 1
        try:
            g = g * (1.0 - censored / at_risk)
        except (OverflowError, ValueError, ZeroDivisionError) as exc:
            raise non_finite() from exc
        if not math.isfinite(g):
            raise non_finite()
        g_after[u] = g

    def g_at(value):
        # Right-continuous step value: largest unique time <= value.
        lo, hi = 0, len(unique_times)
        while lo < hi:
            mid = (lo + hi) // 2
            if unique_times[mid] <= value:
                lo = mid + 1
            else:
                hi = mid
        if lo == 0:
            return 1.0
        return g_after[unique_times[lo - 1]]

    brier = []
    for j in range(m):
        t_j = eval_times[j]
        g_tj = None
        terms = []
        for i in range(n):
            s_ij = survival[i][j]
            term = 0.0
            if event_observed[i] == 1 and event_times[i] <= t_j:
                weight = g_before[event_times[i]]
                if weight <= 0.0:
                    raise ValueError(
                        "the censoring survival probability must be "
                        "greater than 0"
                    )
                try:
                    term = term + (s_ij * s_ij) / weight
                except (OverflowError, ValueError, ZeroDivisionError) as exc:
                    raise non_finite() from exc
            if event_times[i] > t_j:
                if g_tj is None:
                    g_tj = g_at(t_j)
                    if g_tj <= 0.0:
                        raise ValueError(
                            "the censoring survival probability must be "
                            "greater than 0"
                        )
                try:
                    term = term + ((1.0 - s_ij) * (1.0 - s_ij)) / g_tj
                except (OverflowError, ValueError, ZeroDivisionError) as exc:
                    raise non_finite() from exc
            if not math.isfinite(term):
                raise non_finite()
            terms.append(term)
        try:
            b_j = math.fsum(terms) / n
        except (OverflowError, ValueError, ZeroDivisionError) as exc:
            raise non_finite() from exc
        if not math.isfinite(b_j):
            raise non_finite()
        if b_j == 0:
            b_j = 0.0
        brier.append(b_j)
    return brier


# ``from __future__ import annotations`` stores annotations as strings;
# expose the builtin generic as the runtime return annotation.
survival_brier_score.__annotations__["return"] = list[float]


_SERIAL_KEYS_KMEANS = (
    "class",
    "n_clusters",
    "max_iter",
    "tol",
    "seed",
    "cluster_centers",
)
_SERIAL_KEYS_PCA = ("class", "mean", "components")
_SERIAL_KEYS_LINEAR = ("class", "lr", "l2", "max_iter", "tol", "w", "b")
_SERIAL_KEYS_LASSO = ("class", "alpha", "max_iter", "tol", "w", "b")
_SERIAL_KEYS_ELASTIC_NET = (
    "class",
    "alpha",
    "l1_ratio",
    "max_iter",
    "tol",
    "w",
    "b",
)
_SERIAL_KEYS_MULTINOMIAL = (
    "class",
    "lr",
    "l2",
    "max_iter",
    "tol",
    "classes",
    "W",
    "b",
)
_SERIAL_KEYS_SCALER = ("class", "n_features_in", "mean", "scale")
_SERIAL_KEYS_TREE = ("class", "max_depth", "n_features_in", "tree")
_SERIAL_KEYS_TREE_NODE = ("label", "feature", "threshold", "left", "right")
_SERIAL_KEYS_TREE_REGRESSOR = (
    "class",
    "max_depth",
    "n_features_in",
    "tree",
)
_SERIAL_KEYS_GAUSSIAN = (
    "class",
    "n_components",
    "weights",
    "means",
    "variances",
)
_SERIAL_KEYS_FOREST = (
    "class",
    "n_estimators",
    "max_features",
    "seed",
    "n_features_in",
    "trees",
)
_SERIAL_KEYS_FOREST_REGRESSOR = (
    "class",
    "n_estimators",
    "max_depth",
    "max_features",
    "seed",
    "n_features_in",
    "trees",
)
_SERIAL_KEYS_FOREST_REGRESSOR_NODE = (
    "value",
    "feature",
    "threshold",
    "left",
    "right",
)
_SERIAL_KEYS_ISOLATION_FOREST = (
    "class",
    "n_estimators",
    "max_samples",
    "contamination",
    "seed",
    "n_features_in",
    "threshold",
    "trees",
)
_SERIAL_KEYS_ISOLATION_NODE = (
    "count",
    "feature",
    "threshold",
    "left",
    "right",
)
_SERIAL_KEYS_ADABOOST = (
    "class",
    "n_estimators",
    "n_features_in",
    "stumps",
)
_SERIAL_KEYS_STUMP = ("feature", "threshold", "sign", "alpha")
_SERIAL_KEYS_BOOSTING = (
    "class",
    "n_estimators",
    "learning_rate",
    "tol",
    "n_features_in",
    "constant",
    "stumps",
)
_SERIAL_KEYS_BOOSTING_STUMP = ("feature", "threshold", "left", "right")
_SERIAL_KEYS_KNN_REGRESSOR = (
    "class",
    "n_neighbors",
    "weights",
    "n_features_in",
    "X",
    "y",
)

_SERIAL_KEYS_KNN_CLASSIFIER = (
    "class",
    "n_neighbors",
    "n_features_in",
    "X",
    "y",
)

_SERIAL_KEYS_AGGLOMERATIVE = (
    "class",
    "n_clusters",
    "linkage",
    "labels",
    "children",
    "distances",
)

_SERIAL_KEYS_DBSCAN = (
    "class",
    "eps",
    "min_samples",
    "labels",
)


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


def _quantize_fixed12(value):
    """Quantize a finite non-boolean real to 12 decimal places (HALF_UP)
    and return its canonical fixed-point JSON lexical form.

    The value first passes through ``float``; then
    ``Decimal(str(v)).quantize(1E-12, ROUND_HALF_UP)`` performs the
    rounding. Negative zero normalizes to ``0.000000000000``. Every
    rejection (booleans, non-numbers, non-finite or unconvertibly large
    values) is a ValueError.
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
            _QUANTUM12, rounding=ROUND_HALF_UP
        )
    if decimal_value == 0:
        return "0.000000000000"
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


def _dumps_knn_regressor(model):
    """Serialize a fitted KNeighborsRegressor.

    The construction parameters are re-validated exactly as ``__init__``
    does (positive exact-int ``n_neighbors``, ``weights`` in
    ``{"uniform", "distance"}``), the model must be fitted, and
    ``n_features_in`` (the fitted width) must be a positive exact int.
    ``X`` is a non-empty rectangular list of rows with at least
    ``n_neighbors`` rows, each of length ``n_features_in``; ``y`` is a
    list of the same length. Every element of ``X`` and ``y`` must be a
    finite value of type exactly ``float`` (booleans, ints, and
    subclasses are rejected) whose fixed 10-decimal HALF_UP quantization
    converts back with ``float`` to exactly the original value.
    """
    if model._X is None or model._y is None or model._width is None:
        raise ValueError(
            "KNeighborsRegressor must be fitted before dumps is called"
        )
    n_neighbors = model.n_neighbors
    weights = model.weights
    n_features = model._width
    if type(n_neighbors) is not int or n_neighbors < 1:
        raise ValueError("n_neighbors must be a positive integer")
    if type(weights) is not str or weights not in ("uniform", "distance"):
        raise ValueError('weights must be "uniform" or "distance"')
    if type(n_features) is not int or n_features < 1:
        raise ValueError("n_features_in must be a positive integer")

    X = model._X
    y = model._y
    if not isinstance(X, list) or len(X) == 0:
        raise ValueError("X must be a non-empty list of rows")
    if len(X) < n_neighbors:
        raise ValueError(
            "number of training rows must be at least n_neighbors"
        )
    for row in X:
        if not isinstance(row, list) or len(row) != n_features:
            raise ValueError("every X row must have length n_features_in")
        for value in row:
            _quantize_float_exact(value, "X")
    if not isinstance(y, list) or len(y) != len(X):
        raise ValueError("y must be a list with the same length as X")
    for value in y:
        _quantize_float_exact(value, "y")

    x_text = (
        "["
        + ",".join(
            "["
            + ",".join(_quantize_float_exact(v, "X") for v in row)
            + "]"
            for row in X
        )
        + "]"
    )
    y_text = "[" + ",".join(_quantize_float_exact(v, "y") for v in y) + "]"
    return (
        '{"class":"KNeighborsRegressor","n_neighbors":'
        + str(n_neighbors)
        + ',"weights":'
        + json.dumps(weights)
        + ',"n_features_in":'
        + str(n_features)
        + ',"X":'
        + x_text
        + ',"y":'
        + y_text
        + "}"
    )


def _dumps_knn_classifier(model):
    """Serialize a fitted KNeighborsClassifier.

    ``n_neighbors`` is re-validated exactly as ``__init__`` does (a
    positive exact int, booleans rejected), the model must be fitted,
    and ``n_features_in`` (the fitted width) must be a positive exact
    int. ``X`` is a non-empty rectangular list of rows with at least
    ``n_neighbors`` rows, each of length ``n_features_in``; ``y`` is a
    list of the same length. Every element of ``X`` must be a finite
    value of type exactly ``float`` whose fixed 10-decimal HALF_UP
    quantization converts back with ``float`` to exactly the original
    value; every element of ``y`` must be an exact ``int`` (booleans
    rejected), emitted as a JSON integer with no leading zeros.
    """
    if model._X is None or model._y is None or model._width is None:
        raise ValueError(
            "KNeighborsClassifier must be fitted before dumps is called"
        )
    n_neighbors = model.n_neighbors
    n_features = model._width
    if type(n_neighbors) is not int or n_neighbors < 1:
        raise ValueError("n_neighbors must be a positive integer")
    if type(n_features) is not int or n_features < 1:
        raise ValueError("n_features_in must be a positive integer")

    X = model._X
    y = model._y
    if not isinstance(X, list) or len(X) == 0:
        raise ValueError("X must be a non-empty list of rows")
    if len(X) < n_neighbors:
        raise ValueError(
            "number of training rows must be at least n_neighbors"
        )
    for row in X:
        if not isinstance(row, list) or len(row) != n_features:
            raise ValueError("every X row must have length n_features_in")
        for value in row:
            _quantize_float_exact(value, "X")
    if not isinstance(y, list) or len(y) != len(X):
        raise ValueError("y must be a list with the same length as X")
    for value in y:
        if type(value) is not int:
            raise ValueError("y must contain only integers")

    x_text = (
        "["
        + ",".join(
            "["
            + ",".join(_quantize_float_exact(v, "X") for v in row)
            + "]"
            for row in X
        )
        + "]"
    )
    y_text = "[" + ",".join(str(v) for v in y) + "]"
    return (
        '{"class":"KNeighborsClassifier","n_neighbors":'
        + str(n_neighbors)
        + ',"n_features_in":'
        + str(n_features)
        + ',"X":'
        + x_text
        + ',"y":'
        + y_text
        + "}"
    )


def _agglomerative_members(n, children):
    """Replay ``children`` of an agglomerative clustering and return the
    final surviving clusters as ascending member tuples.

    Leaves are the singleton clusters ``0`` ... ``n - 1``; merge ``r``
    references two clusters still alive at that round, ordered so the
    first id's member tuple is strictly smaller than the second's, and
    creates node ``n + r``. Every referenced id must name an existing
    live node. Raises ValueError on any inconsistency.
    """
    members = {i: (i,) for i in range(n)}
    alive = set(range(n))
    for r, pair in enumerate(children):
        merged_id = n + r
        if len(pair) != 2:
            raise ValueError("each children entry must be an integer pair")
        left_id, right_id = pair
        if left_id not in alive or right_id not in alive:
            raise ValueError(
                "each merge must reference two different surviving clusters"
            )
        if left_id < 0 or left_id >= merged_id or right_id < 0 or right_id >= merged_id:
            raise ValueError("children entries must reference existing nodes")
        left_members = members[left_id]
        right_members = members[right_id]
        if not left_members < right_members:
            raise ValueError(
                "each merge must reference its two clusters in ascending "
                "member-tuple order"
            )
        merged = tuple(sorted(left_members + right_members))
        alive.discard(left_id)
        alive.discard(right_id)
        members[merged_id] = merged
        alive.add(merged_id)
    return sorted(members[node_id] for node_id in alive)


def _validate_agglomerative_state(model):
    """Validate a fitted AgglomerativeClustering's parameters and state.

    Confirms the construction parameters exactly as ``__init__`` does
    (positive exact-int ``n_clusters``, linkage one of
    ``single``/``complete``/``average``), that the model is fitted with
    non-empty integer ``labels_`` of length ``n`` (``n_clusters <= n``),
    that ``children_`` has exactly ``n - n_clusters`` integer pairs which
    replay as valid merges over leaves ``0`` ... ``n - 1`` (the r-th pair
    references two distinct surviving clusters and creates node
    ``n + r``), that ``distances_`` has the same length with entries of
    type exactly ``int`` or ``float`` (booleans and subclasses rejected):
    non-negative ints of arbitrary size, never converted through float,
    or finite non-negative floats, and that ``labels_`` equals the
    clusters numbered by ascending smallest member in input order.
    Returns ``(n_clusters, labels, children, distances)``.
    """
    n_clusters = model.n_clusters
    linkage = model.linkage
    if type(n_clusters) is not int or n_clusters <= 0:
        raise ValueError("n_clusters must be a positive integer")
    if type(linkage) is not str or linkage not in (
        "single",
        "complete",
        "average",
    ):
        raise ValueError(
            'linkage must be "single", "complete", or "average"'
        )

    labels = model.labels_
    children = model.children_
    distances = model.distances_
    if labels is None or children is None or distances is None:
        raise ValueError(
            "AgglomerativeClustering must be fitted before dumps is called"
        )
    if not isinstance(labels, list) or len(labels) == 0:
        raise ValueError("labels_ must be a non-empty integer list")
    n = len(labels)
    for value in labels:
        if type(value) is not int:
            raise ValueError("labels_ must contain only integers")
    if n_clusters > n:
        raise ValueError(
            "n_clusters must not exceed the number of labeled samples"
        )

    if not isinstance(children, list) or len(children) != n - n_clusters:
        raise ValueError(
            "children_ must have length n - n_clusters"
        )
    for pair in children:
        if (
            not isinstance(pair, list)
            or len(pair) != 2
            or type(pair[0]) is not int
            or type(pair[1]) is not int
        ):
            raise ValueError(
                "every children_ entry must be a pair of integers"
            )

    clusters = _agglomerative_members(n, children)
    if len(clusters) != n_clusters:
        raise ValueError(
            "the merges must leave exactly n_clusters clusters"
        )
    expected_labels = [0] * n
    for label, cluster_members in enumerate(clusters):
        for index in cluster_members:
            expected_labels[index] = label
    if labels != expected_labels:
        raise ValueError(
            "labels_ must number the clusters by ascending smallest member"
        )

    if not isinstance(distances, list) or len(distances) != len(children):
        raise ValueError(
            "distances_ must have the same length as children_"
        )
    for value in distances:
        # Exact types only -- bool and any int/float subclass are rejected
        # (bool is a subclass of int). Arbitrarily large non-negative ints
        # are accepted without ever passing through float.
        if type(value) is int:
            if value < 0:
                raise ValueError(
                    "distances_ must contain only finite non-negative numbers"
                )
            continue
        if type(value) is float:
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    "distances_ must contain only finite non-negative numbers"
                )
            continue
        raise ValueError(
            "distances_ must contain only finite non-negative numbers"
        )
    return n_clusters, labels, children, distances


def _int_to_exact_str(value):
    """Exact decimal spelling of a non-negative ``int`` of any size.

    Chunked formatting bypasses CPython's 4300-digit ``str(int)`` safety
    limit; the value is never converted through ``float``.
    """
    if value == 0:
        return "0"
    base = 1000000000
    parts = []
    while value:
        parts.append(str(value % base))
        value //= base
    parts.reverse()
    return parts[0] + "".join(part.zfill(9) for part in parts[1:])


def _int_from_decimal_text(token):
    """Parse a non-negative ASCII digit string into an ``int`` of any
    size, chunked to bypass CPython's 4300-digit ``int(str)`` limit."""
    value = 0
    for start in range(0, len(token), 9):
        chunk = token[start:start + 9]
        value = value * (10 ** len(chunk)) + int(chunk)
    return value


def _quantize_agglomerative_distance(value):
    """Quantize one agglomerative merge distance to its canonical text.

    The value must have type exactly ``int`` or ``float`` (booleans and
    any subclasses are rejected). A non-negative exact ``int`` may be
    arbitrarily large and is never converted to ``float`` (its decimal
    spelling is built chunkwise to bypass CPython's integer digit
    limit); a ``float`` must be finite and non-negative. Quantization is
    ``Decimal(str(v)).quantize(1E-10, ROUND_HALF_UP)`` -- deliberately
    ``str(v)`` directly, so an exact integer keeps its exact decimal
    spelling rather than a float-rounded one. Negative zero normalizes to
    ``0.0000000000``.
    """
    if type(value) is int:
        if value < 0:
            raise ValueError(
                "distances_ must contain only finite non-negative numbers"
            )
        token = _int_to_exact_str(value)
    elif type(value) is float:
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(
                "distances_ must contain only finite non-negative numbers"
            )
        token = str(value)
    else:
        raise ValueError(
            "distances_ must contain only finite non-negative numbers"
        )
    with localcontext() as ctx:
        ctx.prec = max(400, len(token) + 20)
        decimal_value = Decimal(token).quantize(
            _QUANTUM, rounding=ROUND_HALF_UP
        )
    if decimal_value == 0:
        return "0.0000000000"
    return format(decimal_value, "f")


def _dumps_agglomerative(model):
    """Serialize a fitted AgglomerativeClustering.

    The top-level keys are ``class``, ``n_clusters``, ``linkage``,
    ``labels``, ``children``, ``distances`` in that order; ``class`` is
    ``"AgglomerativeClustering"``, ``n_clusters`` is a positive JSON
    integer no larger than the label count, and ``linkage`` is
    ``"single"``, ``"complete"``, or ``"average"``. ``labels`` is a
    non-empty JSON integer array; ``children`` has ``n - n_clusters``
    integer pairs replaying the merges with leaves ``0`` ... ``n - 1``
    (the r-th pair merges two distinct surviving clusters into node
    ``n + r``); ``distances`` is an array of the same length whose
    entries have type exactly ``int`` or ``float`` (booleans and
    subclasses rejected): non-negative ints of arbitrary size -- never
    converted through float -- or finite non-negative floats, quantized
    to 10 decimal places via ``Decimal(str(v))`` with ROUND_HALF_UP
    (negative zero becomes ``0.0000000000``).
    """
    n_clusters, labels, children, distances = (
        _validate_agglomerative_state(model)
    )
    labels_text = "[" + ",".join(str(value) for value in labels) + "]"
    children_text = (
        "["
        + ",".join(
            "[" + str(pair[0]) + "," + str(pair[1]) + "]"
            for pair in children
        )
        + "]"
    )
    distances_text = (
        "["
        + ",".join(
            _quantize_agglomerative_distance(value) for value in distances
        )
        + "]"
    )
    return (
        '{"class":"AgglomerativeClustering","n_clusters":'
        + str(n_clusters)
        + ',"linkage":'
        + json.dumps(model.linkage)
        + ',"labels":'
        + labels_text
        + ',"children":'
        + children_text
        + ',"distances":'
        + distances_text
        + "}"
    )


def _validate_dbscan_state(model):
    """Validate a fitted DBSCAN's construction parameters and state.

    Confirms ``eps`` exactly as ``__init__`` does (a value of type
    exactly ``int`` or ``float``, booleans rejected, finite, strictly
    positive -- exact ints keep their exact decimal spelling), that
    ``min_samples`` is a positive exact int, and that ``labels_`` is a
    non-empty exact-int list containing only -1 and cluster labels
    numbered consecutively from zero in first-appearance order. Returns
    ``(eps, min_samples, labels)``.
    """
    eps = model.eps
    min_samples = model.min_samples
    labels = model.labels_
    if type(eps) is int:
        if eps <= 0:
            raise ValueError(
                "eps must be a finite non-boolean int or float greater than 0"
            )
    elif type(eps) is float:
        if not math.isfinite(eps) or eps <= 0.0:
            raise ValueError(
                "eps must be a finite non-boolean int or float greater than 0"
            )
    else:
        raise ValueError(
            "eps must be a finite non-boolean int or float greater than 0"
        )
    if type(min_samples) is not int or min_samples <= 0:
        raise ValueError("min_samples must be a positive integer")

    if not isinstance(labels, list) or len(labels) == 0:
        raise ValueError("labels_ must be a non-empty integer list")
    next_label = 0
    seen = set()
    for value in labels:
        if type(value) is not int:
            raise ValueError("labels_ must contain only integers")
        if value == -1:
            continue
        if value < 0:
            raise ValueError(
                "labels_ must contain only -1 or consecutive cluster labels"
            )
        if value not in seen:
            if value != next_label:
                raise ValueError(
                    "non-negative labels must first appear in order "
                    "0, 1, 2, ..."
                )
            seen.add(value)
            next_label += 1
    return eps, min_samples, labels


def _quantize_dbscan_eps(value):
    """Quantize a DBSCAN ``eps`` to its canonical fixed 10-decimal text.

    The value has type exactly ``int`` or ``float`` (booleans and
    subclasses rejected), is finite and strictly positive, and is
    quantized via ``Decimal(str(v))`` with ROUND_HALF_UP. The quantized
    text must stay strictly positive and convert back with ``float`` to
    exactly the original value.
    """
    if type(value) is int:
        if value <= 0:
            raise ValueError(
                "eps must be a finite non-boolean int or float greater than 0"
            )
        token = _int_to_exact_str(value)
    elif type(value) is float:
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(
                "eps must be a finite non-boolean int or float greater than 0"
            )
        token = str(value)
    else:
        raise ValueError(
            "eps must be a finite non-boolean int or float greater than 0"
        )
    with localcontext() as ctx:
        ctx.prec = max(400, len(token) + 20)
        decimal_value = Decimal(token).quantize(
            _QUANTUM, rounding=ROUND_HALF_UP
        )
    if decimal_value <= 0:
        raise ValueError("eps must remain positive after quantization")
    text = format(decimal_value, "f")
    try:
        roundtrip = float(text)
    except OverflowError as exc:
        raise ValueError(
            "eps must survive 10-decimal quantization unchanged"
        ) from exc
    if roundtrip != value:
        raise ValueError(
            "eps must survive 10-decimal quantization unchanged"
        )
    return text


def _dumps_dbscan(model):
    """Serialize a fitted DBSCAN.

    The top-level keys are ``class``, ``eps``, ``min_samples``,
    ``labels`` in that order; ``class`` is ``"DBSCAN"``, ``eps`` is a
    strictly positive finite exact int/float (booleans rejected)
    quantized to 10 decimal places via ``Decimal(str(v))`` with
    ROUND_HALF_UP -- the quantized text stays positive and converts back
    with ``float`` to exactly the original value -- ``min_samples`` is a
    positive JSON integer, and ``labels`` is a non-empty JSON integer
    array containing only -1 and cluster labels numbered consecutively
    from zero in first-appearance order.
    """
    eps, min_samples, labels = _validate_dbscan_state(model)
    eps_text = _quantize_dbscan_eps(eps)
    labels_text = "[" + ",".join(str(value) for value in labels) + "]"
    return (
        '{"class":"DBSCAN","eps":'
        + eps_text
        + ',"min_samples":'
        + str(min_samples)
        + ',"labels":'
        + labels_text
        + "}"
    )


def dumps(model):
    """Serialize a fitted KMeans, PCA, LinearRegression,
    LogisticRegression, LassoRegression, ElasticNetRegression,
    MultinomialLogisticRegression, StandardScaler,
    DecisionTreeClassifier, DecisionTreeRegressor,
    RandomForestClassifier, ExtraTreesClassifier,
    RandomForestRegressor, AdaBoostClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier,
    GaussianMixture, KNeighborsRegressor,
    KNeighborsClassifier, AgglomerativeClustering, DBSCAN, or
    IsolationForest model to compact JSON text.

    The result contains no whitespace and no trailing newline. Integers
    (``n_clusters``, ``max_iter``, ``seed``, ``n_features_in``) are emitted
    as JSON integers; ``lr``, ``l2``, ``tol``, ``b`` and every array
    coordinate is quantized to 10 decimal places with ROUND_HALF_UP
    (negative zero becomes ``0.0000000000``). A positive ``lr``/``tol``
    that quantizes to zero is rejected, as are any non-fitted models,
    unsupported objects, invalid construction parameters, and malformed
    or non-finite state. The argument is not modified.

    For DecisionTreeClassifier, ``max_depth`` is a positive JSON integer
    or the string ``"none"``; each tree node carries the keys ``label``,
    ``feature``, ``threshold``, ``left``, ``right`` in that order, with
    leaves encoded as ``feature`` -1, ``threshold`` 0.0000000000 and
    empty ``left``/``right`` arrays.

    For DecisionTreeRegressor the top-level keys are ``class``,
    ``max_depth``, ``n_features_in``, ``tree`` in that order; ``class``
    is ``"DecisionTreeRegressor"``, ``max_depth`` is a positive JSON
    integer or ``"none"``, and ``n_features_in`` is a positive JSON
    integer. The single ``tree`` is encoded exactly like one
    RandomForestRegressor tree: node keys ``value``, ``feature``,
    ``threshold``, ``left``, ``right`` in that order, leaves as
    ``feature`` -1 with ``threshold`` 0.0000000000 and empty children,
    internal features in ``[0, n_features_in)``, and every node
    ``value``/internal ``threshold`` a finite exact int/float quantized
    to 10 decimal places via ``Decimal(str(v))`` with ROUND_HALF_UP
    (negative zero becomes ``0.0000000000``) whose quantized text
    converts back with ``float`` to exactly the original value.

    For RandomForestClassifier the top-level keys are ``class``,
    ``n_estimators``, ``max_features``, ``seed``, ``n_features_in``,
    ``trees`` in that order, with exactly ``n_estimators`` trees whose
    nodes use the same node encoding as DecisionTreeClassifier.
    ExtraTreesClassifier uses this exact same format, differing only in
    ``class`` being ``"ExtraTreesClassifier"``.

    For RandomForestRegressor the top-level keys are ``class``,
    ``n_estimators``, ``max_depth``, ``max_features``, ``seed``,
    ``n_features_in``, ``trees`` in that order; ``class`` is
    ``"RandomForestRegressor"``, ``seed`` is a JSON integer,
    ``max_depth`` is a positive JSON integer or ``"none"``, and the
    remaining numeric fields are positive JSON integers with
    ``max_features <= n_features_in`` and exactly ``n_estimators`` trees.
    Each tree node carries the keys ``value``, ``feature``,
    ``threshold``, ``left``, ``right`` in that order; leaves are encoded
    as ``feature`` -1, ``threshold`` 0.0000000000 and empty
    ``left``/``right`` arrays, and internal nodes carry a feature in
    ``[0, n_features_in)`` and node children. Every node ``value`` and
    internal ``threshold`` is a finite exact int/float quantized to 10
    decimal places via ``Decimal(str(v))`` with ROUND_HALF_UP (negative
    zero becomes ``0.0000000000``); the quantized text must convert back
    with ``float`` to exactly the original value.

    For AdaBoostClassifier the top-level keys are ``class``,
    ``n_estimators``, ``n_features_in``, ``stumps`` in that order;
    ``stumps`` is a non-empty array of at most ``n_estimators`` stumps,
    each with the keys ``feature``, ``threshold``, ``sign``, ``alpha``
    in that order. ``feature`` is an integer in ``[0, n_features_in)``,
    ``sign`` is -1 or 1, and ``threshold``/``alpha`` are exact finite
    int/float values with ``alpha`` strictly positive, quantized to 10
    decimal places; the quantized text must convert back with ``float``
    to exactly the original value.

    For GradientBoostingRegressor and GradientBoostingClassifier the
    top-level keys are ``class``,
    ``n_estimators``, ``learning_rate``, ``tol``, ``n_features_in``,
    ``constant``, ``stumps`` in that order; ``class`` is the same-named
    string of the fitted model; the two integers are
    positive JSON integers, ``learning_rate`` and ``tol`` are strictly
    positive finite exact int/float values quantized to 10 decimal
    places that must remain positive after quantization, and
    ``constant`` is a finite exact int/float quantized the same way
    (negative zero becomes ``0.0000000000``); the quantized text must
    convert back with ``float`` to exactly the original value.
    ``stumps`` is an array of zero to ``n_estimators`` stumps, each an
    object with the keys ``feature``, ``threshold``, ``left``, ``right``
    in that order: an integer ``feature`` in ``[0, n_features_in)`` and
    three finite exact int/float values quantized to 10 decimal places,
    each of which must convert back with ``float`` to exactly the
    original value so a reloaded model predicts identically.

    For GaussianMixture the top-level keys are ``class``,
    ``n_components``, ``weights``, ``means``, ``variances`` in that
    order; ``n_components`` is a positive JSON integer and the three
    arrays are non-empty lists of that length whose elements are exact
    int/float values (booleans rejected). Each coordinate is converted
    through ``float`` and quantized to 12 decimal places with
    ROUND_HALF_UP (negative zero becomes ``0.000000000000``); weights
    must be positive, means finite, and variances at least ``1e-12``,
    with neither weights nor variances quantizing to zero.

    For KNeighborsRegressor the top-level keys are ``class``,
    ``n_neighbors``, ``weights``, ``n_features_in``, ``X``, ``y`` in
    that order; ``class`` is ``"KNeighborsRegressor"``, ``n_neighbors``
    and ``n_features_in`` are positive JSON integers, and ``weights``
    is ``"uniform"`` or ``"distance"``. ``X`` is a non-empty 2-D array
    with at least ``n_neighbors`` rows, each of length
    ``n_features_in``, and ``y`` is a 1-D array of the same length.
    Every element of ``X`` and ``y`` must have type exactly ``float``
    and be finite; each is quantized to 10 decimal places with
    ROUND_HALF_UP via ``Decimal(str(v))`` (negative zero becomes
    ``0.0000000000``), and the quantized text must convert back with
    ``float`` to exactly the original value or dumps raises ValueError.

    For LassoRegression the top-level keys are ``class``, ``alpha``,
    ``max_iter``, ``tol``, ``w``, ``b`` in that order; ``class`` is
    ``"LassoRegression"``, ``max_iter`` is a positive JSON integer, and
    ``alpha``/``tol`` are strictly positive exact ints (of any
    magnitude) or finite floats. ``w`` is a non-empty 1-D array whose
    elements and ``b`` are exact ints or finite floats. Every number is
    quantized to 10 decimal places via ``Decimal(str(v))`` with
    ROUND_HALF_UP (never through ``float`` first; negative zero becomes
    ``0.0000000000``); the fixed ``alpha``/``tol`` text must convert
    back with ``float`` to a finite strictly positive number but need
    not equal the original value, while the fixed ``w``/``b`` text must
    convert back to exactly the original finite value.

    For ElasticNetRegression the top-level keys are ``class``,
    ``alpha``, ``l1_ratio``, ``max_iter``, ``tol``, ``w``, ``b`` in
    that order; ``class`` is ``"ElasticNetRegression"`` and
    ``max_iter`` is a positive JSON integer. ``alpha``/``tol`` are
    strictly positive exact ints (of any magnitude) or finite floats
    and ``l1_ratio`` is an exact int or finite float in ``[0, 1]``.
    ``w`` is a non-empty 1-D array whose elements and ``b`` are exact
    ints or finite floats. Every number is quantized to 10 decimal
    places via ``Decimal(str(v))`` with ROUND_HALF_UP (never through
    ``float`` first; negative zero becomes ``0.0000000000``); the fixed
    ``alpha``/``l1_ratio``/``tol`` text must convert back with
    ``float`` to a legal parameter value (``alpha``/``tol`` strictly
    positive, ``l1_ratio`` in ``[0, 1]``) but need not equal the
    original value, while the fixed ``w``/``b`` text must convert back
    to exactly the original finite value.

    For MultinomialLogisticRegression the top-level keys are ``class``,
    ``lr``, ``l2``, ``max_iter``, ``tol``, ``classes``, ``W``, ``b`` in
    that order; ``class`` is ``"MultinomialLogisticRegression"``.
    ``lr`` and ``tol`` are strictly positive and ``l2`` non-negative
    finite values (all quantized to 10 decimal places, with ``lr`` and
    ``tol`` remaining positive after quantization), and ``max_iter`` is a
    positive JSON integer. ``classes`` is an ascending list of at least
    two distinct JSON integers; ``W`` is a non-empty rectangular 2-D
    array with one row per class (its column count is the feature count)
    and ``b`` is an array of the same length. Every coordinate of ``W``
    and ``b`` is quantized to 10 decimal places with ROUND_HALF_UP
    (negative zero becomes ``0.0000000000``).

    For KNeighborsClassifier the top-level keys are ``class``,
    ``n_neighbors``, ``n_features_in``, ``X``, ``y`` in that order;
    ``class`` is ``"KNeighborsClassifier"`` and ``n_neighbors`` and
    ``n_features_in`` are positive JSON integers. ``X`` is a non-empty
    2-D array with at least ``n_neighbors`` rows, each of length
    ``n_features_in``, whose elements have type exactly ``float`` and
    are finite; each is quantized to 10 decimal places with
    ROUND_HALF_UP via ``Decimal(str(v))`` (negative zero becomes
    ``0.0000000000``), and the quantized text must convert back with
    ``float`` to exactly the original value or dumps raises ValueError.
    ``y`` is a 1-D array of the same length as ``X`` whose elements
    have type exactly ``int`` (booleans rejected) and are emitted as
    JSON integers with no leading zeros.

    For AgglomerativeClustering the top-level keys are ``class``,
    ``n_clusters``, ``linkage``, ``labels``, ``children``,
    ``distances`` in that order; ``class`` is
    ``"AgglomerativeClustering"``, ``n_clusters`` is a positive JSON
    integer no larger than the label count, and ``linkage`` is
    ``"single"``, ``"complete"``, or ``"average"``. ``labels`` is a
    non-empty JSON integer array of length ``n``. ``children`` has
    ``n - n_clusters`` integer pairs: leaves are ``0`` ... ``n - 1``,
    the r-th pair must reference two distinct surviving clusters by
    ascending member tuple and creates merge node ``n + r``. ``labels``
    must equal the clusters numbered from zero by ascending smallest
    member in input order. ``distances`` has the same length as
    ``children``; each entry has type exactly ``int`` or ``float``
    (booleans and subclasses rejected) -- a non-negative int of
    arbitrary size, never converted through float, or a finite
    non-negative float -- quantized to 10 decimal places via
    ``Decimal(str(v))`` with ROUND_HALF_UP (negative zero becomes
    ``0.0000000000``).

    For DBSCAN the top-level keys are ``class``, ``eps``,
    ``min_samples``, ``labels`` in that order; ``class`` is
    ``"DBSCAN"``, ``eps`` is a strictly positive finite exact int/float
    (booleans rejected) quantized to 10 decimal places via
    ``Decimal(str(v))`` with ROUND_HALF_UP, staying positive and
    converting back with ``float`` to exactly the original value,
    ``min_samples`` is a positive JSON integer, and ``labels`` is a
    non-empty JSON integer array containing only -1 and cluster labels
    numbered consecutively from zero in first-appearance order.

    For IsolationForest the top-level keys are ``class``,
    ``n_estimators``, ``max_samples``, ``contamination``, ``seed``,
    ``n_features_in``, ``threshold``, ``trees`` in that order; ``class``
    is ``"IsolationForest"``, the construction fields are re-validated
    exactly as ``__init__`` performs the checks (``n_estimators`` a
    positive JSON integer, ``max_samples`` a JSON integer of at least
    2, ``contamination`` the string ``"auto"`` or a finite exact
    int/float in ``(0, 0.5]`` quantized to 10 decimal places (the
    quantized text must convert back with ``float`` to exactly the
    original value), and
    ``seed`` a JSON integer), ``n_features_in`` is a positive JSON
    integer, and the decision ``threshold`` is a finite float
    quantized to 10 decimal places via ``Decimal(str(v))`` with
    ROUND_HALF_UP (negative zero becomes ``0.0000000000``); the
    quantized threshold text must convert back with ``float`` to
    exactly the original value. The ``trees`` array holds exactly
    ``n_estimators`` trees. Each node carries the keys ``count``,
    ``feature``, ``threshold``, ``left``, ``right`` in that order:
    ``count`` is a JSON integer in ``[1, max_samples]``; leaves are
    encoded as ``feature`` -1, ``threshold`` ``0.0000000000`` and
    empty ``left``/``right`` arrays; internal nodes carry a feature in
    ``[0, n_features_in)``, a finite exact int/float threshold
    quantized the same way (its quantized text converting back
    exactly), node children, and a count equal to the sum of the
    children's counts. Every tree root has count ``max_samples``.
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

    if isinstance(model, (LinearRegression, LogisticRegression)):
        try:
            return _dumps_linear(model)
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError("invalid linear model state") from exc

    if isinstance(model, LassoRegression):
        try:
            return _dumps_lasso(model)
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError("invalid LassoRegression state") from exc

    if isinstance(model, ElasticNetRegression):
        try:
            return _dumps_elastic_net(model)
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError("invalid ElasticNetRegression state") from exc

    if isinstance(model, MultinomialLogisticRegression):
        try:
            return _dumps_multinomial(model)
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(
                "invalid MultinomialLogisticRegression state"
            ) from exc

    if isinstance(model, StandardScaler):
        try:
            return _dumps_scaler(model)
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError("invalid StandardScaler state") from exc

    if isinstance(model, DecisionTreeClassifier):
        try:
            return _dumps_tree(model)
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError("invalid DecisionTreeClassifier state") from exc

    if isinstance(model, RandomForestClassifier):
        try:
            return _dumps_forest(model, "RandomForestClassifier")
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError("invalid RandomForestClassifier state") from exc

    if isinstance(model, ExtraTreesClassifier):
        try:
            return _dumps_forest(model, "ExtraTreesClassifier")
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError("invalid ExtraTreesClassifier state") from exc

    if isinstance(model, RandomForestRegressor):
        try:
            return _dumps_forest_regressor(model)
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError("invalid RandomForestRegressor state") from exc

    if isinstance(model, DecisionTreeRegressor):
        try:
            return _dumps_tree_regressor(model)
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError("invalid DecisionTreeRegressor state") from exc

    if isinstance(model, AdaBoostClassifier):
        try:
            return _dumps_adaboost(model)
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError("invalid AdaBoostClassifier state") from exc

    if isinstance(model, GradientBoostingRegressor):
        try:
            return _dumps_boosting(model, "GradientBoostingRegressor")
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(
                "invalid GradientBoostingRegressor state"
            ) from exc

    if isinstance(model, GradientBoostingClassifier):
        try:
            return _dumps_boosting(model, "GradientBoostingClassifier")
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(
                "invalid GradientBoostingClassifier state"
            ) from exc

    if isinstance(model, GaussianMixture):
        try:
            return _dumps_gaussian(model)
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError("invalid GaussianMixture state") from exc

    if isinstance(model, KNeighborsRegressor):
        try:
            return _dumps_knn_regressor(model)
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(
                "invalid KNeighborsRegressor state"
            ) from exc

    if isinstance(model, KNeighborsClassifier):
        try:
            return _dumps_knn_classifier(model)
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(
                "invalid KNeighborsClassifier state"
            ) from exc

    if isinstance(model, AgglomerativeClustering):
        try:
            return _dumps_agglomerative(model)
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(
                "invalid AgglomerativeClustering state"
            ) from exc

    if isinstance(model, DBSCAN):
        try:
            return _dumps_dbscan(model)
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError("invalid DBSCAN state") from exc

    if isinstance(model, IsolationForest):
        try:
            return _dumps_isolation_forest(model)
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError("invalid IsolationForest state") from exc

    raise ValueError(
        "dumps only supports fitted KMeans, PCA, LinearRegression, "
        "LogisticRegression, LassoRegression, ElasticNetRegression, "
        "MultinomialLogisticRegression, "
        "StandardScaler, DecisionTreeClassifier, DecisionTreeRegressor, "
        "RandomForestClassifier, ExtraTreesClassifier, "
        "RandomForestRegressor, "
        "AdaBoostClassifier, GradientBoostingRegressor, "
        "GradientBoostingClassifier, "
        "GaussianMixture, KNeighborsRegressor, "
        "KNeighborsClassifier, AgglomerativeClustering, DBSCAN, and "
        "IsolationForest models"
    )


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


def _dumps_linear(model):
    """Serialize a fitted LinearRegression or LogisticRegression.

    Both classes share the same constructor signature and fitted state
    (``w``/``b``), so one encoder covers both; the emitted ``class`` is
    the model's own class name.
    """
    w = model.w
    b = model.b
    if w is None or b is None:
        raise ValueError("model must be fitted before dumps is called")
    lr = model.lr
    l2 = model.l2
    max_iter = model.max_iter
    tol = model.tol
    # Re-validate the construction parameters exactly as __init__ does.
    if (
        not _is_finite_number(lr)
        or lr <= 0
        or not _is_finite_number(l2)
        or l2 < 0
        or type(max_iter) is not int
        or max_iter < 1
        or not _is_finite_number(tol)
        or tol <= 0
    ):
        raise ValueError("model has invalid construction parameters")
    lr_text = _quantize_fixed(lr)
    if Decimal(lr_text) == 0:
        raise ValueError(
            "lr must remain positive after quantization to 10 decimals"
        )
    l2_text = _quantize_fixed(l2)
    tol_text = _quantize_fixed(tol)
    if Decimal(tol_text) == 0:
        raise ValueError(
            "tol must remain positive after quantization to 10 decimals"
        )
    if not isinstance(w, list) or len(w) == 0:
        raise ValueError("w must be a non-empty list")
    w_text = "[" + ",".join(_quantize_fixed(v) for v in w) + "]"
    b_text = _quantize_fixed(b)
    return (
        '{"class":"'
        + type(model).__name__
        + '","lr":'
        + lr_text
        + ',"l2":'
        + l2_text
        + ',"max_iter":'
        + str(max_iter)
        + ',"tol":'
        + tol_text
        + ',"w":'
        + w_text
        + ',"b":'
        + b_text
        + "}"
    )


def _dumps_lasso(model):
    """Serialize a fitted LassoRegression.

    The top-level keys are ``class``, ``alpha``, ``max_iter``, ``tol``,
    ``w``, ``b`` in that order; ``class`` is ``"LassoRegression"`` and
    ``max_iter`` is a positive JSON integer. ``alpha``/``tol`` are exact
    ints of any magnitude or strictly positive finite floats; each
    coordinate of the non-empty 1-D ``w`` and ``b`` is an exact int or a
    finite float (booleans and subclasses rejected). Every number is
    formatted via ``Decimal(str(v))`` quantized to 10 decimal places with
    ROUND_HALF_UP (negative zero becomes ``0.0000000000``), never passing
    the original value through ``float`` first. The fixed ``alpha``/
    ``tol`` text only has to convert back with ``float`` to a finite
    strictly positive number (it need not equal the original value); the
    fixed ``w``/``b`` text must convert back with ``float`` to exactly
    the original finite value, or dumps raises ValueError.
    """
    w = model.w
    b = model.b
    if w is None or b is None:
        raise ValueError("model must be fitted before dumps is called")
    alpha = model.alpha
    tol = model.tol
    max_iter = model.max_iter
    # Re-validate the construction parameters exactly as __init__ does.
    try:
        _require_exact_positive_or_finite_float(alpha, "alpha")
        _require_exact_positive_or_finite_float(tol, "tol")
    except ValueError:
        raise ValueError("model has invalid construction parameters")
    if type(max_iter) is not int or max_iter <= 0:
        raise ValueError("model has invalid construction parameters")

    # alpha/tol are quantized for display only: the fixed text only has
    # to read back as a finite strictly positive float, not as the
    # original value. w/b are fitted state and must survive the round
    # trip exactly.
    alpha_text = _quantize_lasso_number(
        alpha, "alpha", positive=True, require_equal=False
    )
    tol_text = _quantize_lasso_number(
        tol, "tol", positive=True, require_equal=False
    )

    if not isinstance(w, list) or len(w) == 0:
        raise ValueError("w must be a non-empty list")
    w_text = (
        "["
        + ",".join(
            _quantize_lasso_number(value, "w element") for value in w
        )
        + "]"
    )
    b_text = _quantize_lasso_number(b, "b")

    return (
        '{"class":"LassoRegression","alpha":'
        + alpha_text
        + ',"max_iter":'
        + str(max_iter)
        + ',"tol":'
        + tol_text
        + ',"w":'
        + w_text
        + ',"b":'
        + b_text
        + "}"
    )


def _quantize_lasso_number(value, name, positive=False, require_equal=True):
    """Quantize an exact int/float to its canonical 10-decimal JSON form.

    Unlike :func:`_quantize_fixed`, the value is never converted through
    ``float`` beforehand: ``Decimal(str(value))`` carries an exact int of
    arbitrary magnitude into the ROUND_HALF_UP quantization, and negative
    zero becomes ``0.0000000000``. The fixed text must convert back with
    ``float`` to a finite number; when ``require_equal`` is set (the
    default, used for fitted state such as ``w``/``b``) it must also be
    exactly equal to ``value``. Construction parameters such as
    ``alpha``/``tol`` pass ``require_equal=False``: their readback only
    has to be finite and, when ``positive`` is set, strictly positive.
    """
    if isinstance(value, bool) or type(value) not in (int, float):
        raise ValueError("%s must be a finite non-boolean number" % name)
    if type(value) is float and not math.isfinite(value):
        raise ValueError("%s must be a finite non-boolean number" % name)
    token = str(value)
    try:
        with localcontext() as ctx:
            ctx.prec = max(50, len(token) + 50)
            quantized = Decimal(token).quantize(
                _QUANTUM, rounding=ROUND_HALF_UP
            )
    except (ArithmeticError, ValueError) as exc:
        raise ValueError("%s must be a finite non-boolean number" % name) \
            from exc
    text = "0.0000000000" if quantized == 0 else format(quantized, "f")
    try:
        converted = float(text)
    except (OverflowError, ValueError) as exc:
        raise ValueError(
            "%s must equal its float serialization" % name
        ) from exc
    if not math.isfinite(converted):
        raise ValueError(
            "%s must equal its float serialization" % name
        )
    if require_equal and converted != value:
        raise ValueError(
            "%s must equal its float serialization" % name
        )
    if positive and converted <= 0.0:
        raise ValueError("%s must be greater than 0" % name)
    return text


def _dumps_elastic_net(model):
    """Serialize a fitted ElasticNetRegression.

    The top-level keys are ``class``, ``alpha``, ``l1_ratio``,
    ``max_iter``, ``tol``, ``w``, ``b`` in that order; ``class`` is
    ``"ElasticNetRegression"`` and ``max_iter`` is a positive JSON
    integer. ``alpha``/``tol`` are exact ints of any magnitude or
    strictly positive finite floats and ``l1_ratio`` is an exact int or
    finite float in ``[0, 1]``; each coordinate of the non-empty 1-D
    ``w`` and ``b`` is an exact int or a finite float (booleans and
    subclasses rejected). Every number is formatted via
    ``Decimal(str(v))`` quantized to 10 decimal places with
    ROUND_HALF_UP (negative zero becomes ``0.0000000000``), never
    passing the original value through ``float`` first. The fixed
    ``alpha``/``l1_ratio``/``tol`` text only has to read back as a
    legal parameter value (strictly positive for ``alpha``/``tol``,
    within ``[0, 1]`` for ``l1_ratio``), not as the original value;
    the fixed ``w``/``b`` text must convert back with ``float`` to
    exactly the original finite value, or dumps raises ValueError.
    """
    w = model.w
    b = model.b
    if w is None or b is None:
        raise ValueError("model must be fitted before dumps is called")
    alpha = model.alpha
    l1_ratio = model.l1_ratio
    max_iter = model.max_iter
    tol = model.tol
    # Re-validate the construction parameters exactly as __init__ does.
    try:
        _require_exact_positive_or_finite_float(alpha, "alpha")
        _require_exact_unit_interval_number(l1_ratio, "l1_ratio")
        _require_exact_positive_or_finite_float(tol, "tol")
    except ValueError:
        raise ValueError("model has invalid construction parameters")
    if type(max_iter) is not int or max_iter <= 0:
        raise ValueError("model has invalid construction parameters")

    # alpha/l1_ratio/tol are quantized for display only: the fixed text
    # only has to read back as a legal parameter value, not as the
    # original value. w/b are fitted state and must survive the round
    # trip exactly.
    alpha_text = _quantize_lasso_number(
        alpha, "alpha", positive=True, require_equal=False
    )
    l1_ratio_text = _quantize_lasso_number(
        l1_ratio, "l1_ratio", require_equal=False
    )
    if not 0.0 <= float(l1_ratio_text) <= 1.0:
        raise ValueError(
            "l1_ratio must remain in [0, 1] after quantization to "
            "10 decimals"
        )
    tol_text = _quantize_lasso_number(
        tol, "tol", positive=True, require_equal=False
    )

    if not isinstance(w, list) or len(w) == 0:
        raise ValueError("w must be a non-empty list")
    w_text = (
        "["
        + ",".join(
            _quantize_lasso_number(value, "w element") for value in w
        )
        + "]"
    )
    b_text = _quantize_lasso_number(b, "b")

    return (
        '{"class":"ElasticNetRegression","alpha":'
        + alpha_text
        + ',"l1_ratio":'
        + l1_ratio_text
        + ',"max_iter":'
        + str(max_iter)
        + ',"tol":'
        + tol_text
        + ',"w":'
        + w_text
        + ',"b":'
        + b_text
        + "}"
    )


def _dumps_multinomial(model):
    """Serialize a fitted MultinomialLogisticRegression.

    The top-level keys are ``class``, ``lr``, ``l2``, ``max_iter``,
    ``tol``, ``classes``, ``W``, ``b`` in that order; ``class`` is
    ``"MultinomialLogisticRegression"``. The construction parameters are
    re-validated exactly as ``__init__`` does (strictly positive finite
    ``lr``/``tol``, non-negative finite ``l2``, positive exact-int
    ``max_iter``). ``classes`` is an ascending list of at least two
    distinct JSON integers; ``W`` is a non-empty rectangular matrix with
    one row per class and at least one feature column, and ``b`` a vector
    of the same length. Every coordinate is a finite non-boolean number
    quantized via ``_quantize_fixed`` -- ``Decimal(str(float(v)))`` with
    ROUND_HALF_UP to 10 decimal places (negative zero becomes
    ``0.0000000000``).
    """
    classes = model.classes
    W = model.W
    b = model.b
    if classes is None or W is None or b is None:
        raise ValueError(
            "MultinomialLogisticRegression must be fitted before dumps "
            "is called"
        )
    lr = model.lr
    l2 = model.l2
    max_iter = model.max_iter
    tol = model.tol
    # Re-validate the construction parameters exactly as __init__ does.
    if (
        not _is_finite_number(lr)
        or lr <= 0
        or not _is_finite_number(l2)
        or l2 < 0
        or isinstance(max_iter, bool)
        or type(max_iter) is not int
        or max_iter < 1
        or not _is_finite_number(tol)
        or tol <= 0
    ):
        raise ValueError("model has invalid construction parameters")
    lr_text = _quantize_fixed(lr)
    if Decimal(lr_text) == 0:
        raise ValueError(
            "lr must remain positive after quantization to 10 decimals"
        )
    l2_text = _quantize_fixed(l2)
    tol_text = _quantize_fixed(tol)
    if Decimal(tol_text) == 0:
        raise ValueError(
            "tol must remain positive after quantization to 10 decimals"
        )

    if not isinstance(classes, list) or len(classes) < 2:
        raise ValueError(
            "classes must be a list of at least two distinct integers"
        )
    previous = None
    for label in classes:
        if type(label) is not int:
            raise ValueError("classes must contain only integers")
        if previous is not None and label <= previous:
            raise ValueError("classes must be strictly ascending and unique")
        previous = label

    if not isinstance(W, list) or len(W) == 0:
        raise ValueError("W must be a non-empty list of rows")
    if len(W) != len(classes):
        raise ValueError("W must have one row per class")
    width = None
    encoded_rows = []
    for row in W:
        if not isinstance(row, list) or len(row) == 0:
            raise ValueError("W rows must be non-empty lists")
        if width is None:
            width = len(row)
        elif len(row) != width:
            raise ValueError("W must be rectangular")
        encoded_rows.append(
            "[" + ",".join(_quantize_fixed(v) for v in row) + "]"
        )
    if not isinstance(b, list) or len(b) != len(classes):
        raise ValueError("b must be a list with one entry per class")
    b_encoded = [_quantize_fixed(v) for v in b]

    classes_text = "[" + ",".join(str(label) for label in classes) + "]"
    w_text = "[" + ",".join(encoded_rows) + "]"
    b_text = "[" + ",".join(b_encoded) + "]"
    return (
        '{"class":"MultinomialLogisticRegression","lr":'
        + lr_text
        + ',"l2":'
        + l2_text
        + ',"max_iter":'
        + str(max_iter)
        + ',"tol":'
        + tol_text
        + ',"classes":'
        + classes_text
        + ',"W":'
        + w_text
        + ',"b":'
        + b_text
        + "}"
    )


def _dumps_scaler(model):
    """Serialize a fitted StandardScaler.

    ``mean_`` and ``scale_`` must be non-empty lists of equal length,
    every element an exact finite int/float, ``n_features_in_`` a positive
    exact int matching that length, and every scale strictly positive.
    Coordinates are quantized literally with
    ``Decimal(str(v)).quantize(1E-10, ROUND_HALF_UP)`` (exact ints keep
    their full magnitude instead of passing through a binary float).
    """
    mean = model.mean_
    scale = model.scale_
    n_features = model.n_features_in_
    if mean is None or scale is None or n_features is None:
        raise ValueError(
            "StandardScaler must be fitted before dumps is called"
        )
    if type(n_features) is not int or n_features <= 0:
        raise ValueError("n_features_in_ must be a positive integer")
    if not isinstance(mean, list) or len(mean) == 0:
        raise ValueError("mean_ must be a non-empty list")
    if not isinstance(scale, list) or len(scale) != len(mean):
        raise ValueError("scale_ must be a list the same length as mean_")
    if len(mean) != n_features:
        raise ValueError(
            "state array length must equal n_features_in_"
        )
    for value in mean:
        _require_scaler_number(value, "mean_")
    for value in scale:
        _require_scaler_number(value, "scale_")
        if value <= 0:
            raise ValueError("scale_ must be strictly positive")
    mean_text = "[" + ",".join(_quantize_state_number(v) for v in mean) + "]"
    scale_text = (
        "[" + ",".join(_quantize_state_number(v) for v in scale) + "]"
    )
    return (
        '{"class":"StandardScaler","n_features_in":'
        + str(n_features)
        + ',"mean":'
        + mean_text
        + ',"scale":'
        + scale_text
        + "}"
    )


def _require_scaler_number(value, name):
    """Exact int (always finite) or finite exact float; reject booleans."""
    if type(value) is int:
        return
    if type(value) is float and math.isfinite(value):
        return
    raise ValueError("%s must contain only finite numbers" % name)


def _quantize_state_number(value):
    """Quantize an exact int/float to 10 fixed decimals via the literal
    ``Decimal(str(v))`` with ROUND_HALF_UP. Negative zero normalizes to
    ``0.0000000000``. The decimal context covers the full digit count of
    any exact integer."""
    token = str(value)
    with localcontext() as ctx:
        ctx.prec = max(400, len(token.lstrip("-")) + 20)
        decimal_value = Decimal(token).quantize(
            _QUANTUM, rounding=ROUND_HALF_UP
        )
    if decimal_value == 0:
        return "0.0000000000"
    return format(decimal_value, "f")


def _quantize_float_exact(value, name):
    """Quantize an exact finite ``float`` to 10 fixed decimals.

    Unlike :func:`_quantize_state_number`, the type must be exactly
    ``float`` (booleans, ints, and subclasses are rejected), the value
    must be finite, and the canonical fixed-point text must convert back
    with ``float`` to exactly the original value -- so a value that loses
    information at 10 decimals is rejected. Rounding is
    ``Decimal(str(v)).quantize(1E-10, ROUND_HALF_UP)``; negative zero
    normalizes to ``0.0000000000``.
    """
    if type(value) is not float or not math.isfinite(value):
        raise ValueError("%s must contain only finite floats" % name)
    token = str(value)
    with localcontext() as ctx:
        ctx.prec = max(400, len(token.lstrip("-")) + 20)
        decimal_value = Decimal(token).quantize(
            _QUANTUM, rounding=ROUND_HALF_UP
        )
    if decimal_value == 0:
        text = "0.0000000000"
    else:
        text = format(decimal_value, "f")
    if float(text) != value:
        raise ValueError(
            "%s value does not round-trip at 10 decimal places" % name
        )
    return text


def _dumps_tree(model):
    """Serialize a fitted DecisionTreeClassifier.

    ``max_depth`` is emitted as a positive JSON integer or the string
    ``"none"``; ``n_features_in`` is a positive JSON integer taken from
    the fitted feature count. Nodes are encoded by ``_encode_tree_node``.
    """
    root = model._root
    n_features = model._n_features
    if root is None or n_features is None:
        raise ValueError(
            "DecisionTreeClassifier must be fitted before dumps is called"
        )
    max_depth = model.max_depth
    # Re-validate the construction parameter exactly as __init__ does.
    if max_depth is not None and (
        type(max_depth) is not int or max_depth < 1
    ):
        raise ValueError("max_depth must be None or a positive integer")
    if type(n_features) is not int or n_features <= 0:
        raise ValueError("n_features_in must be a positive integer")
    if max_depth is None:
        max_depth_text = '"none"'
    else:
        max_depth_text = str(max_depth)
    return (
        '{"class":"DecisionTreeClassifier","max_depth":'
        + max_depth_text
        + ',"n_features_in":'
        + str(n_features)
        + ',"tree":'
        + _encode_tree_node(root, n_features)
        + "}"
    )


def _encode_tree_node(node, n_features):
    """Encode one tree node with the keys label, feature, threshold,
    left, right in that order.

    Leaves (``feature is None``) are emitted as ``"feature":-1``,
    ``"threshold":0.0000000000``, ``"left":[]``, ``"right":[]`` and must
    carry no threshold or children. Internal nodes need an exact integer
    feature in ``[0, n_features)``, an exact finite int/float threshold,
    and node children. Labels must be exact integers.
    """
    if not isinstance(node, _DecisionTreeNode):
        raise ValueError("tree nodes must be _DecisionTreeNode instances")
    label = node.label
    if type(label) is not int:
        raise ValueError("tree node labels must be integers")
    feature = node.feature
    if feature is None:
        if (
            node.threshold is not None
            or node.left is not None
            or node.right is not None
        ):
            raise ValueError(
                "leaf nodes must have no threshold or children"
            )
        return (
            '{"label":'
            + str(label)
            + ',"feature":-1,"threshold":0.0000000000,"left":[],"right":[]}'
        )
    if type(feature) is not int or not 0 <= feature < n_features:
        raise ValueError("internal node feature must be in [0, n_features_in)")
    _require_scaler_number(node.threshold, "threshold")
    threshold_text = _quantize_state_number(node.threshold)
    left_text = _encode_tree_node(node.left, n_features)
    right_text = _encode_tree_node(node.right, n_features)
    return (
        '{"label":'
        + str(label)
        + ',"feature":'
        + str(feature)
        + ',"threshold":'
        + threshold_text
        + ',"left":'
        + left_text
        + ',"right":'
        + right_text
        + "}"
    )


def _dumps_forest(model, class_name):
    """Serialize a fitted RandomForestClassifier or ExtraTreesClassifier.

    The top-level keys are class, n_estimators, max_features, seed,
    n_features_in, trees in that order; n_estimators and n_features_in are
    positive JSON integers, 1 <= max_features <= n_features_in, and the
    ``trees`` array holds exactly n_estimators node trees encoded by
    ``_encode_tree_node``. The two classes share this format exactly,
    differing only in the ``class`` string. The construction parameters
    are re-validated exactly as ``__init__`` performs the checks, and the
    fitted state must match them in shape.
    """
    trees = model._trees
    n_features = model._n_features
    if trees is None or n_features is None:
        raise ValueError(
            class_name + " must be fitted before dumps is called"
        )
    n_estimators = model.n_estimators
    max_features = model.max_features
    seed = model.seed
    if (
        type(n_estimators) is not int
        or n_estimators <= 0
        or type(max_features) is not int
        or max_features <= 0
        or type(seed) is not int
    ):
        raise ValueError(
            class_name + " has invalid construction parameters"
        )
    if type(n_features) is not int or n_features <= 0:
        raise ValueError("n_features_in must be a positive integer")
    if max_features > n_features:
        raise ValueError(
            "max_features must not exceed n_features_in"
        )
    if not isinstance(trees, list) or len(trees) != n_estimators:
        raise ValueError("trees must have length n_estimators")
    trees_text = "[" + ",".join(
        _encode_tree_node(tree, n_features) for tree in trees
    ) + "]"
    return (
        '{"class":"' + class_name + '","n_estimators":'
        + str(n_estimators)
        + ',"max_features":'
        + str(max_features)
        + ',"seed":'
        + str(seed)
        + ',"n_features_in":'
        + str(n_features)
        + ',"trees":'
        + trees_text
        + "}"
    )


def _dumps_tree_regressor(model):
    """Serialize a fitted DecisionTreeRegressor.

    The top-level keys are class, max_depth, n_features_in, tree in that
    order; class is ``"DecisionTreeRegressor"``, ``max_depth`` is emitted
    as a positive JSON integer or the string ``"none"``, and
    ``n_features_in`` is a positive JSON integer taken from the fitted
    feature count. The single ``tree`` uses exactly the RandomForestRegressor
    node encoding (``_encode_forest_regressor_node``): node keys
    value, feature, threshold, left, right in that order, leaves encoded
    as feature -1 with a zero threshold and empty children, internal
    features in ``[0, n_features_in)``, and every value/threshold a finite
    exact int/float quantized to 10 decimal places with ROUND_HALF_UP
    (negative zero becomes ``0.0000000000``) whose quantized text converts
    back with ``float`` to exactly the original value. The construction
    parameter is re-validated exactly as ``__init__`` performs the check.
    """
    root = model._root
    n_features = model._n_features
    if root is None or n_features is None:
        raise ValueError(
            "DecisionTreeRegressor must be fitted before dumps is called"
        )
    max_depth = model.max_depth
    if max_depth is not None and (
        type(max_depth) is not int or max_depth < 1
    ):
        raise ValueError("max_depth must be None or a positive integer")
    if type(n_features) is not int or n_features <= 0:
        raise ValueError("n_features_in must be a positive integer")
    if max_depth is None:
        max_depth_text = '"none"'
    else:
        max_depth_text = str(max_depth)
    return (
        '{"class":"DecisionTreeRegressor","max_depth":'
        + max_depth_text
        + ',"n_features_in":'
        + str(n_features)
        + ',"tree":'
        + _encode_forest_regressor_node(root, n_features)
        + "}"
    )


def _dumps_forest_regressor(model):
    """Serialize a fitted RandomForestRegressor.

    The top-level keys are class, n_estimators, max_depth, max_features,
    seed, n_features_in, trees in that order; class is
    ``"RandomForestRegressor"``, seed is a JSON integer, max_depth is a
    positive JSON integer or the string ``"none"``, and the remaining
    numeric fields (n_estimators, max_features, n_features_in) are
    positive JSON integers with ``max_features <= n_features_in``. The
    ``trees`` array holds exactly n_estimators node trees encoded by
    ``_encode_forest_regressor_node``. The construction parameters are
    re-validated exactly as ``__init__`` performs the checks, and the
    fitted state must match them in shape.
    """
    trees = model._trees
    n_features = model._n_features
    if trees is None or n_features is None:
        raise ValueError(
            "RandomForestRegressor must be fitted before dumps is called"
        )
    n_estimators = model.n_estimators
    max_depth = model.max_depth
    max_features = model.max_features
    seed = model.seed
    if (
        type(n_estimators) is not int
        or n_estimators <= 0
        or type(max_features) is not int
        or max_features <= 0
        or type(seed) is not int
    ):
        raise ValueError(
            "RandomForestRegressor has invalid construction parameters"
        )
    if max_depth is not None and (
        type(max_depth) is not int or max_depth < 1
    ):
        raise ValueError("max_depth must be None or a positive integer")
    if type(n_features) is not int or n_features <= 0:
        raise ValueError("n_features_in must be a positive integer")
    if max_features > n_features:
        raise ValueError(
            "max_features must not exceed n_features_in"
        )
    if not isinstance(trees, list) or len(trees) != n_estimators:
        raise ValueError("trees must have length n_estimators")
    if max_depth is None:
        max_depth_text = '"none"'
    else:
        max_depth_text = str(max_depth)
    trees_text = "[" + ",".join(
        _encode_forest_regressor_node(tree, n_features) for tree in trees
    ) + "]"
    return (
        '{"class":"RandomForestRegressor","n_estimators":'
        + str(n_estimators)
        + ',"max_depth":'
        + max_depth_text
        + ',"max_features":'
        + str(max_features)
        + ',"seed":'
        + str(seed)
        + ',"n_features_in":'
        + str(n_features)
        + ',"trees":'
        + trees_text
        + "}"
    )


def _encode_forest_regressor_node(node, n_features):
    """Encode one regressor tree node with the keys value, feature,
    threshold, left, right in that order.

    Leaves (``feature is None``) are emitted as ``"feature":-1``,
    ``"threshold":0.0000000000``, ``"left":[]``, ``"right":[]`` and must
    carry no threshold or children. Internal nodes need an exact integer
    feature in ``[0, n_features)``, an exact finite int/float threshold,
    and node children. Every node ``value`` and every internal
    ``threshold`` is quantized to 10 decimals via
    ``Decimal(str(v))`` with ROUND_HALF_UP (negative zero becomes
    ``0.0000000000``); the quantized text must convert back with
    ``float`` to exactly the original value so a reloaded forest predicts
    each value identically.
    """
    if not isinstance(node, _DecisionTreeRegressorNode):
        raise ValueError(
            "tree nodes must be _DecisionTreeRegressorNode instances"
        )
    value = node.value
    _require_scaler_number(value, "node value")
    value_text = _quantize_forest_regressor_number(value, "node value")
    feature = node.feature
    if feature is None:
        if (
            node.threshold is not None
            or node.left is not None
            or node.right is not None
        ):
            raise ValueError(
                "leaf nodes must have no threshold or children"
            )
        return (
            '{"value":'
            + value_text
            + ',"feature":-1,"threshold":0.0000000000,"left":[],"right":[]}'
        )
    if type(feature) is not int or not 0 <= feature < n_features:
        raise ValueError("internal node feature must be in [0, n_features_in)")
    _require_scaler_number(node.threshold, "threshold")
    threshold_text = _quantize_forest_regressor_number(
        node.threshold, "threshold"
    )
    left_text = _encode_forest_regressor_node(node.left, n_features)
    right_text = _encode_forest_regressor_node(node.right, n_features)
    return (
        '{"value":'
        + value_text
        + ',"feature":'
        + str(feature)
        + ',"threshold":'
        + threshold_text
        + ',"left":'
        + left_text
        + ',"right":'
        + right_text
        + "}"
    )


def _quantize_forest_regressor_number(value, name):
    """Quantize an exact finite int/float regressor node coordinate to
    10 fixed decimals via ``Decimal(str(v))`` with ROUND_HALF_UP (negative
    zero becomes ``0.0000000000``). The quantized text must convert back
    with ``float`` to exactly the original value, otherwise ValueError is
    raised because a loaded forest would not predict that value
    identically."""
    token = _quantize_state_number(value)
    if float(token) != value:
        raise ValueError(
            "%s must equal its 10-decimal quantization" % name
        )
    return token


def _dumps_isolation_forest(model):
    """Serialize a fitted IsolationForest.

    The top-level keys are class, n_estimators, max_samples,
    contamination, seed, n_features_in, threshold, trees in that order;
    class is ``"IsolationForest"``, the four construction fields
    (``n_estimators``, ``max_samples``, ``contamination``, ``seed``) are
    re-validated exactly as ``__init__`` performs the checks,
    ``n_features_in`` is a positive JSON integer, a numeric
    ``contamination`` is quantized to 10 decimals and the quantized text
    must convert back with ``float`` to exactly the original value, the
    decision ``threshold`` is a finite float quantized to 10 decimals,
    and the
    ``trees`` array holds exactly n_estimators node trees encoded by
    ``_encode_isolation_node``. The fitted state must match the
    construction parameters in shape.
    """
    trees = model._trees
    n_features = model._width
    if trees is None or n_features is None or model.threshold_ is None:
        raise ValueError(
            "IsolationForest must be fitted before dumps is called"
        )
    n_estimators = model.n_estimators
    max_samples = model.max_samples
    contamination = model.contamination
    seed = model.seed
    if (
        type(n_estimators) is not int
        or n_estimators <= 0
        or type(max_samples) is not int
        or max_samples < 2
        or type(seed) is not int
    ):
        raise ValueError(
            "IsolationForest has invalid construction parameters"
        )
    if type(contamination) is str:
        if contamination != "auto":
            raise ValueError(
                'contamination must be "auto" or a finite number in'
                " (0, 0.5]"
            )
        contamination_text = '"auto"'
    elif type(contamination) in (int, float) and (
        0.0 < contamination <= 0.5
    ):
        contamination_text = _quantize_state_number(
            contamination
        )
        quantized_contamination = float(contamination_text)
        if not 0.0 < quantized_contamination <= 0.5:
            raise ValueError(
                "contamination must remain in (0, 0.5] after quantization"
                " to 10 decimals"
            )
        if quantized_contamination != contamination:
            raise ValueError(
                "contamination must equal its 10-decimal quantization"
            )
    else:
        raise ValueError(
            'contamination must be "auto" or a finite number in (0, 0.5]'
        )
    if type(n_features) is not int or n_features <= 0:
        raise ValueError("n_features_in must be a positive integer")
    threshold = model.threshold_
    if type(threshold) is not float or not math.isfinite(threshold):
        raise ValueError("threshold must be a finite float")
    threshold_text = _quantize_state_number(threshold)
    if float(threshold_text) != threshold:
        raise ValueError(
            "threshold must equal its 10-decimal quantization"
        )
    if not isinstance(trees, list) or len(trees) != n_estimators:
        raise ValueError("trees must have length n_estimators")
    trees_text = "[" + ",".join(
        _encode_isolation_node(tree, n_features, max_samples)
        for tree in trees
    ) + "]"
    for tree in trees:
        if tree.count != max_samples:
            raise ValueError("tree root count must equal max_samples")
    return (
        '{"class":"IsolationForest","n_estimators":'
        + str(n_estimators)
        + ',"max_samples":'
        + str(max_samples)
        + ',"contamination":'
        + contamination_text
        + ',"seed":'
        + str(seed)
        + ',"n_features_in":'
        + str(n_features)
        + ',"threshold":'
        + threshold_text
        + ',"trees":'
        + trees_text
        + "}"
    )


def _encode_isolation_node(node, n_features, max_samples):
    """Encode one isolation tree node with the keys count, feature,
    threshold, left, right in that order.

    ``count`` is an exact integer in ``[1, max_samples]``. Leaves
    (``feature is None``) are emitted as ``feature`` -1, ``threshold``
    ``0.0000000000`` and empty ``left``/``right`` arrays. Internal nodes
    need an exact integer feature in ``[0, n_features)``, node children
    whose counts sum to ``count``, and a finite exact int/float threshold
    quantized to 10 decimals via ``Decimal(str(v))`` with ROUND_HALF_UP
    (negative zero becomes ``0.0000000000``); the quantized text must
    convert back with ``float`` to exactly the original threshold so a
    reloaded forest predicts identically.
    """
    if not isinstance(node, _IsolationTreeNode):
        raise ValueError(
            "tree nodes must be _IsolationTreeNode instances"
        )
    count = node.count
    if type(count) is not int or not 1 <= count <= max_samples:
        raise ValueError("node count must be an integer in [1, max_samples]")
    feature = node.feature
    if feature is None:
        if (
            node.threshold is not None
            or node.left is not None
            or node.right is not None
        ):
            raise ValueError(
                "leaf nodes must have no threshold or children"
            )
        return (
            '{"count":'
            + str(count)
            + ',"feature":-1,"threshold":0.0000000000,"left":[],"right":[]}'
        )
    if type(feature) is not int or not 0 <= feature < n_features:
        raise ValueError("internal node feature must be in [0, n_features_in)")
    threshold_text = _quantize_forest_regressor_number(
        node.threshold, "threshold"
    )
    left_text = _encode_isolation_node(node.left, n_features, max_samples)
    right_text = _encode_isolation_node(node.right, n_features, max_samples)
    if node.left.count + node.right.count != count:
        raise ValueError(
            "internal node count must equal the sum of its children"
        )
    return (
        '{"count":'
        + str(count)
        + ',"feature":'
        + str(feature)
        + ',"threshold":'
        + threshold_text
        + ',"left":'
        + left_text
        + ',"right":'
        + right_text
        + "}"
    )


def _dumps_adaboost(model):
    """Serialize a fitted AdaBoostClassifier.

    The top-level keys are class, n_estimators, n_features_in, stumps in
    that order; ``n_estimators`` and ``n_features_in`` are positive JSON
    integers and ``stumps`` is a non-empty array of at most
    ``n_estimators`` stumps encoded by ``_encode_stump``. The
    construction parameter is re-validated exactly as ``__init__``
    performs the check, and the fitted state must match it in shape.
    """
    stumps = model._stumps
    n_features = model._n_features
    if stumps is None or n_features is None:
        raise ValueError(
            "AdaBoostClassifier must be fitted before dumps is called"
        )
    n_estimators = model.n_estimators
    # Re-validate the construction parameter exactly as __init__ does.
    if type(n_estimators) is not int or n_estimators <= 0:
        raise ValueError("n_estimators must be a positive integer")
    if type(n_features) is not int or n_features <= 0:
        raise ValueError("n_features_in must be a positive integer")
    if (
        not isinstance(stumps, list)
        or len(stumps) == 0
        or len(stumps) > n_estimators
    ):
        raise ValueError(
            "stumps must be a non-empty list of at most n_estimators "
            "stumps"
        )
    stumps_text = "[" + ",".join(
        _encode_stump(stump, n_features) for stump in stumps
    ) + "]"
    return (
        '{"class":"AdaBoostClassifier","n_estimators":'
        + str(n_estimators)
        + ',"n_features_in":'
        + str(n_features)
        + ',"stumps":'
        + stumps_text
        + "}"
    )


def _encode_stump(stump, n_features):
    """Encode one decision stump with the keys feature, threshold, sign,
    alpha in that order.

    ``feature`` is an exact integer in ``[0, n_features)`` and ``sign``
    is -1 or 1. ``threshold`` and ``alpha`` must be exact finite
    int/float values (booleans rejected) with ``alpha`` strictly
    positive; both are quantized to 10 decimals and the quantized text
    must convert back with ``float`` to exactly the original value so a
    loaded model predicts identically.
    """
    if type(stump) is not tuple or len(stump) != 4:
        raise ValueError(
            "stumps must be (feature, threshold, sign, alpha) 4-tuples"
        )
    feature, threshold, sign, alpha = stump
    if type(feature) is not int or not 0 <= feature < n_features:
        raise ValueError("stump feature must be in [0, n_features_in)")
    if type(sign) is not int or sign not in (-1, 1):
        raise ValueError("stump sign must be -1 or 1")
    _require_scaler_number(threshold, "threshold")
    _require_scaler_number(alpha, "alpha")
    if alpha <= 0:
        raise ValueError("stump alpha must be strictly positive")
    threshold_text = _quantize_stump_number(threshold, "threshold")
    alpha_text = _quantize_stump_number(alpha, "alpha")
    return (
        '{"feature":'
        + str(feature)
        + ',"threshold":'
        + threshold_text
        + ',"sign":'
        + str(sign)
        + ',"alpha":'
        + alpha_text
        + "}"
    )


def _quantize_stump_number(value, name):
    """Quantize an exact int/float stump coordinate to 10 fixed decimals
    via ``Decimal(str(v))`` with ROUND_HALF_UP (negative zero becomes
    ``0.0000000000``). The quantized text must convert back with
    ``float`` to exactly the original value, otherwise ValueError is
    raised because a loaded stump would not predict identically."""
    token = _quantize_state_number(value)
    if float(token) != value:
        raise ValueError(
            "stump %s must equal its 10-decimal quantization" % name
        )
    return token


def _quantize_boosting_number(value, name):
    """Quantize an exact int/float boosting coordinate to 10 fixed
    decimals via ``Decimal(str(v))`` with ROUND_HALF_UP (negative zero
    becomes ``0.0000000000``). The quantized text must convert back with
    ``float`` to exactly the original value, otherwise ValueError is
    raised because a model reloaded from the text would not predict
    identically."""
    token = _quantize_state_number(value)
    if float(token) != value:
        raise ValueError(
            "boosting %s must equal its 10-decimal quantization" % name
        )
    return token


def _dumps_boosting(model, class_name):
    """Serialize a fitted GradientBoostingRegressor or
    GradientBoostingClassifier (``class_name`` selects the format tag).

    The top-level keys are class, n_estimators, learning_rate, tol,
    n_features_in, constant, stumps in that order; ``n_estimators`` and
    ``n_features_in`` are positive JSON integers, ``learning_rate`` and
    ``tol`` are strictly positive finite exact int/float values quantized
    to 10 decimals (they must remain positive after quantization), and
    ``constant`` is a finite exact int/float quantized the same way whose
    quantized text must convert back with ``float`` to exactly the
    original value. ``stumps`` is a list of zero to ``n_estimators``
    four-tuples encoded as objects with the keys feature, threshold,
    left, right in that order; ``feature`` is an exact integer in
    ``[0, n_features_in)`` and the other three are finite exact int/float
    values (booleans rejected), each quantized to 10 decimals with
    negative zero normalized to ``0.0000000000`` and each required to
    survive the ``float`` round trip exactly so a reloaded model predicts
    identically.
    """
    constant = model._constant
    stumps = model._stumps
    n_features = model._n_features
    if constant is None or stumps is None or n_features is None:
        raise ValueError(
            class_name + " must be fitted before dumps is "
            "called"
        )
    n_estimators = model.n_estimators
    learning_rate = model.learning_rate
    tol = model.tol
    # Re-validate the construction parameters exactly as __init__ does.
    if type(n_estimators) is not int or n_estimators <= 0:
        raise ValueError("n_estimators must be a positive integer")
    _require_exact_finite_positive(learning_rate, "learning_rate")
    _require_exact_finite_positive(tol, "tol")
    if type(n_features) is not int or n_features <= 0:
        raise ValueError("n_features_in must be a positive integer")
    _require_exact_finite_number(constant, "constant")
    if not isinstance(stumps, list) or not 0 <= len(stumps) <= n_estimators:
        raise ValueError(
            "stumps must be a list of at most n_estimators stumps"
        )
    learning_rate_text = _quantize_state_number(learning_rate)
    tol_text = _quantize_state_number(tol)
    if Decimal(learning_rate_text) == 0:
        raise ValueError(
            "learning_rate must remain positive after quantization to 10 "
            "decimals"
        )
    if Decimal(tol_text) == 0:
        raise ValueError(
            "tol must remain positive after quantization to 10 decimals"
        )
    constant_text = _quantize_boosting_number(constant, "constant")
    stumps_text = "[" + ",".join(
        _encode_boosting_stump(stump, n_features) for stump in stumps
    ) + "]"
    return (
        '{"class":"'
        + class_name
        + '","n_estimators":'
        + str(n_estimators)
        + ',"learning_rate":'
        + learning_rate_text
        + ',"tol":'
        + tol_text
        + ',"n_features_in":'
        + str(n_features)
        + ',"constant":'
        + constant_text
        + ',"stumps":'
        + stumps_text
        + "}"
    )


def _encode_boosting_stump(stump, n_features):
    """Encode one boosting stump as an object with the keys feature,
    threshold, left, right in that order.

    ``feature`` is an exact integer in ``[0, n_features)``; the
    threshold and the two side increments are finite exact int/float
    values (booleans rejected), quantized to 10 decimals with negative
    zero normalized to ``0.0000000000`` and each required to survive the
    ``float`` round trip exactly so a reloaded model predicts
    identically.
    """
    if type(stump) is not tuple or len(stump) != 4:
        raise ValueError(
            "stumps must be (feature, threshold, left, right) 4-tuples"
        )
    feature, threshold, left, right = stump
    if type(feature) is not int or not 0 <= feature < n_features:
        raise ValueError("stump feature must be in [0, n_features_in)")
    _require_exact_finite_number(threshold, "stump threshold")
    _require_exact_finite_number(left, "stump left increment")
    _require_exact_finite_number(right, "stump right increment")
    return (
        '{"feature":'
        + str(feature)
        + ',"threshold":'
        + _quantize_boosting_number(threshold, "stump threshold")
        + ',"left":'
        + _quantize_boosting_number(left, "stump left increment")
        + ',"right":'
        + _quantize_boosting_number(right, "stump right increment")
        + "}"
    )


def _dumps_gaussian(model):
    """Serialize a fitted GaussianMixture.

    The top-level keys are class, n_components, weights, means, variances
    in that order; ``n_components`` is a positive JSON integer and all
    three arrays are non-empty lists of that same length whose elements
    are exact finite int/float values (booleans rejected). Weights must be
    strictly positive (also after 12-decimal quantization) and variances
    at least ``1e-12``. Every coordinate passes through ``float`` before
    being quantized to 12 decimals with ROUND_HALF_UP.
    """
    weights = model.weights_
    means = model.means_
    variances = model.variances_
    if weights is None or means is None or variances is None:
        raise ValueError(
            "GaussianMixture must be fitted before dumps is called"
        )
    n_components = model.n_components
    if type(n_components) is not int or n_components <= 0:
        raise ValueError("n_components must be a positive integer")
    for array, name in (
        (weights, "weights_"),
        (means, "means_"),
        (variances, "variances_"),
    ):
        if not isinstance(array, list) or len(array) == 0:
            raise ValueError("%s must be a non-empty list" % name)
        if len(array) != n_components:
            raise ValueError(
                "%s length must equal n_components" % name
            )
        for value in array:
            if isinstance(value, bool) or type(value) not in (int, float):
                raise ValueError(
                    "%s elements must be int or float" % name
                )
    weights_text = "[" + ",".join(
        _quantize_gaussian_value(value, "weights_", positive=True)
        for value in weights
    ) + "]"
    means_text = "[" + ",".join(
        _quantize_gaussian_value(value, "means_") for value in means
    ) + "]"
    variances_text_parts = []
    for value in variances:
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "variances_ elements must be finite numbers"
            ) from exc
        if not math.isfinite(number) or number < 1e-12:
            raise ValueError(
                "variances_ elements must be finite and at least 1e-12"
            )
        variances_text_parts.append(
            _quantize_gaussian_value(value, "variances_", positive=True)
        )
    variances_text = "[" + ",".join(variances_text_parts) + "]"
    return (
        '{"class":"GaussianMixture","n_components":'
        + str(n_components)
        + ',"weights":'
        + weights_text
        + ',"means":'
        + means_text
        + ',"variances":'
        + variances_text
        + "}"
    )


def _quantize_gaussian_value(value, name, positive=False):
    """Quantize one GaussianMixture coordinate to 12 fixed decimals.

    The element is first converted with ``float`` and must stay finite;
    strictly positive coordinates must quantize to a nonzero value.
    """
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "%s elements must be finite numbers" % name
        ) from exc
    if not math.isfinite(number):
        raise ValueError("%s elements must be finite" % name)
    if positive and number <= 0.0:
        raise ValueError("%s elements must be strictly positive" % name)
    token = _quantize_fixed12(number)
    if positive and Decimal(token) == 0:
        raise ValueError(
            "%s elements must remain positive after quantization to 12 "
            "decimals" % name
        )
    return token


_JSON_INT_RE = re.compile(r"^(0|-?[1-9][0-9]*)$")
_JSON_FLOAT_RE = re.compile(r"^-?(0|[1-9][0-9]*)\.[0-9]{10}$")
_JSON_FLOAT12_RE = re.compile(r"^-?(0|[1-9][0-9]*)\.[0-9]{12}$")
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
    decimals that are neither 10 nor 12 digits). Booleans/null are parsed
    so that their presence as values can be rejected by the structural
    validators.
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
        if _JSON_FLOAT12_RE.match(token):
            return ("fixed12", token)
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
    if isinstance(node, tuple) and node[0] in ("int", "fixed", "fixed12"):
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


def _expect_agglomerative_distance(entry):
    """Parse one AgglomerativeClustering ``distances`` element.

    Accepts a non-negative fixed 10-decimal number. Values whose
    conversion to ``float`` is non-finite are kept as an exact ``int``
    only when their fraction is all zero (a huge integer distance, which
    dumps never converts through float); finite values and fractional
    values are returned as ``float``. Negative tokens are rejected.
    """
    if not (
        isinstance(entry, tuple)
        and entry[0] == "fixed"
        and _JSON_FLOAT_RE.match(entry[1])
    ):
        raise ValueError(
            "distances element must be a fixed 10-decimal JSON number"
        )
    token = entry[1]
    if token[0] == "-":
        raise ValueError("distances must be non-negative")
    value = float(token)
    if math.isfinite(value):
        return value
    integer_part, fraction = token.split(".")
    if set(fraction) == {"0"}:
        # Chunked parsing keeps the value exact even beyond CPython's
        # 4300-digit integer string-conversion limit.
        return _int_from_decimal_text(integer_part)
    raise ValueError(
        "out-of-range distances must have an all-zero fraction"
    )


def _expect_fixed12(entry, name):
    if not (
        isinstance(entry, tuple)
        and entry[0] == "fixed12"
        and _JSON_FLOAT12_RE.match(entry[1])
    ):
        raise ValueError("%s must be a fixed 12-decimal JSON number" % name)
    token = entry[1]
    value = float(token)
    if not math.isfinite(value):
        raise ValueError("%s must be finite" % name)
    # dumps normalizes negative zero to "0.000000000000".
    if value == 0.0 and token[0] == "-":
        raise ValueError("%s must not be negative zero" % name)
    return value


def _expect_fixed12_vector(node, length, name):
    if not isinstance(node, list) or len(node) != length:
        raise ValueError("%s must have length %d" % (name, length))
    return [_expect_fixed12(item, name + " element") for item in node]


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


def _load_linear(pairs, model_class, class_name):
    keys = tuple(key for key, _ in pairs)
    if keys != _SERIAL_KEYS_LINEAR:
        raise ValueError("%s JSON must have exactly the serialized keys "
                         "in the serialized order" % class_name)
    data = _convert(pairs)

    class_entry = data["class"]
    if not isinstance(class_entry, str) or class_entry != class_name:
        raise ValueError('class must be "%s"' % class_name)

    lr = _expect_fixed(data["lr"], "lr")
    if lr <= 0.0:
        raise ValueError("lr must be greater than 0")
    l2 = _expect_fixed(data["l2"], "l2")
    if l2 < 0.0:
        raise ValueError("l2 must be non-negative")
    max_iter = _expect_int(data["max_iter"], "max_iter")
    if max_iter < 1:
        raise ValueError("max_iter must be at least 1")
    tol = _expect_fixed(data["tol"], "tol")
    if tol <= 0.0:
        raise ValueError("tol must be greater than 0")

    w_node = data["w"]
    if not isinstance(w_node, list) or len(w_node) == 0:
        raise ValueError("w must be a non-empty array")
    w = [_expect_fixed(v, "w element") for v in w_node]

    b = _expect_fixed(data["b"], "b")

    model = model_class(lr=lr, l2=l2, max_iter=max_iter, tol=tol)
    model.w = list(w)
    model.b = b
    return model


def _load_lasso(pairs):
    keys = tuple(key for key, _ in pairs)
    if keys != _SERIAL_KEYS_LASSO:
        raise ValueError(
            "LassoRegression JSON must have exactly the serialized keys "
            "in the serialized order"
        )
    data = _convert(pairs)

    class_entry = data["class"]
    if not isinstance(class_entry, str) or class_entry != "LassoRegression":
        raise ValueError('class must be "LassoRegression"')

    alpha = _expect_fixed(data["alpha"], "alpha")
    if alpha <= 0.0:
        raise ValueError("alpha must be greater than 0")
    max_iter = _expect_int(data["max_iter"], "max_iter")
    if max_iter < 1:
        raise ValueError("max_iter must be at least 1")
    tol = _expect_fixed(data["tol"], "tol")
    if tol <= 0.0:
        raise ValueError("tol must be greater than 0")

    w_node = data["w"]
    if not isinstance(w_node, list) or len(w_node) == 0:
        raise ValueError("w must be a non-empty array")
    w = [_expect_fixed(value, "w element") for value in w_node]

    b = _expect_fixed(data["b"], "b")

    model = LassoRegression(
        alpha=alpha, max_iter=max_iter, tol=tol
    )
    model.w = list(w)
    model.b = b
    return model


def _load_elastic_net(pairs):
    keys = tuple(key for key, _ in pairs)
    if keys != _SERIAL_KEYS_ELASTIC_NET:
        raise ValueError(
            "ElasticNetRegression JSON must have exactly the serialized "
            "keys in the serialized order"
        )
    data = _convert(pairs)

    class_entry = data["class"]
    if (
        not isinstance(class_entry, str)
        or class_entry != "ElasticNetRegression"
    ):
        raise ValueError('class must be "ElasticNetRegression"')

    alpha = _expect_fixed(data["alpha"], "alpha")
    if alpha <= 0.0:
        raise ValueError("alpha must be greater than 0")
    l1_ratio = _expect_fixed(data["l1_ratio"], "l1_ratio")
    if not 0.0 <= l1_ratio <= 1.0:
        raise ValueError("l1_ratio must be in [0, 1]")
    max_iter = _expect_int(data["max_iter"], "max_iter")
    if max_iter < 1:
        raise ValueError("max_iter must be at least 1")
    tol = _expect_fixed(data["tol"], "tol")
    if tol <= 0.0:
        raise ValueError("tol must be greater than 0")

    w_node = data["w"]
    if not isinstance(w_node, list) or len(w_node) == 0:
        raise ValueError("w must be a non-empty array")
    w = [_expect_fixed(value, "w element") for value in w_node]

    b = _expect_fixed(data["b"], "b")

    model = ElasticNetRegression(
        alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter, tol=tol
    )
    model.w = list(w)
    model.b = b
    return model


def _load_multinomial(pairs):
    keys = tuple(key for key, _ in pairs)
    if keys != _SERIAL_KEYS_MULTINOMIAL:
        raise ValueError(
            "MultinomialLogisticRegression JSON must have exactly the "
            "serialized keys in the serialized order"
        )
    data = _convert(pairs)

    class_name = data["class"]
    if not isinstance(class_name, str):
        raise ValueError('class must be "MultinomialLogisticRegression"')
    if class_name != "MultinomialLogisticRegression":
        raise ValueError('class must be "MultinomialLogisticRegression"')

    lr = _expect_fixed(data["lr"], "lr")
    if lr <= 0.0:
        raise ValueError("lr must be greater than 0")
    l2 = _expect_fixed(data["l2"], "l2")
    if l2 < 0.0:
        raise ValueError("l2 must be non-negative")
    max_iter = _expect_int(data["max_iter"], "max_iter")
    if max_iter < 1:
        raise ValueError("max_iter must be at least 1")
    tol = _expect_fixed(data["tol"], "tol")
    if tol <= 0.0:
        raise ValueError("tol must be greater than 0")

    classes_node = data["classes"]
    if not isinstance(classes_node, list) or len(classes_node) < 2:
        raise ValueError(
            "classes must be an array of at least two JSON integers"
        )
    classes = []
    previous = None
    for entry in classes_node:
        label = _expect_int(entry, "classes element")
        if previous is not None and label <= previous:
            raise ValueError(
                "classes must be strictly ascending and unique"
            )
        classes.append(label)
        previous = label

    w_node = data["W"]
    if not isinstance(w_node, list) or len(w_node) != len(classes):
        raise ValueError("W must have one row per class")
    W = []
    width = None
    for row in w_node:
        if not isinstance(row, list) or len(row) == 0:
            raise ValueError("W rows must be non-empty arrays")
        if width is None:
            width = len(row)
        elif len(row) != width:
            raise ValueError("W must be rectangular")
        W.append([_expect_fixed(v, "W coordinate") for v in row])

    b = _expect_fixed_vector(data["b"], len(classes), "b")

    model = MultinomialLogisticRegression(
        lr=lr, l2=l2, max_iter=max_iter, tol=tol
    )
    model.classes = list(classes)
    model.W = [list(row) for row in W]
    model.b = list(b)
    return model


def _load_scaler(pairs):
    keys = tuple(key for key, _ in pairs)
    if keys != _SERIAL_KEYS_SCALER:
        raise ValueError(
            "StandardScaler JSON must have exactly the serialized keys "
            "in the serialized order"
        )
    data = _convert(pairs)

    class_name = data["class"]
    if not isinstance(class_name, str) or class_name != "StandardScaler":
        raise ValueError('class must be "StandardScaler"')

    n_features = _expect_int(data["n_features_in"], "n_features_in")
    if n_features <= 0:
        raise ValueError("n_features_in must be greater than 0")

    mean = _expect_fixed_vector(data["mean"], n_features, "mean")
    scale = _expect_fixed_vector(data["scale"], n_features, "scale")
    for value in scale:
        if value <= 0.0:
            raise ValueError("scale must be strictly positive")

    model = StandardScaler()
    model.mean_ = list(mean)
    model.scale_ = list(scale)
    model.n_features_in_ = n_features
    return model


def _load_tree(pairs):
    keys = tuple(key for key, _ in pairs)
    if keys != _SERIAL_KEYS_TREE:
        raise ValueError(
            "DecisionTreeClassifier JSON must have exactly the serialized "
            "keys in the serialized order"
        )
    data = _convert(pairs)

    class_name = data["class"]
    if not isinstance(class_name, str) or class_name != "DecisionTreeClassifier":
        raise ValueError('class must be "DecisionTreeClassifier"')

    max_depth_entry = data["max_depth"]
    if isinstance(max_depth_entry, str):
        if max_depth_entry != "none":
            raise ValueError('max_depth must be a positive integer or "none"')
        max_depth = None
    else:
        max_depth = _expect_int(max_depth_entry, "max_depth")
        if max_depth < 1:
            raise ValueError("max_depth must be a positive integer")

    n_features = _expect_int(data["n_features_in"], "n_features_in")
    if n_features <= 0:
        raise ValueError("n_features_in must be greater than 0")

    root = _load_tree_node(data["tree"], n_features)

    model = DecisionTreeClassifier(max_depth=max_depth)
    model._root = root
    model._n_features = n_features
    return model


def _load_tree_node(node, n_features):
    """Rebuild one tree node from its converted JSON object.

    The node must have exactly the keys label, feature, threshold, left,
    right in that order. A leaf has ``feature`` -1, ``threshold``
    0.0000000000 and empty ``left``/``right`` arrays; an internal node
    has a feature in ``[0, n_features)`` and node children.
    """
    if not isinstance(node, dict) or tuple(node.keys()) != _SERIAL_KEYS_TREE_NODE:
        raise ValueError(
            "tree nodes must have exactly the keys "
            "label, feature, threshold, left, right in that order"
        )
    label = _expect_int(node["label"], "node label")
    feature = _expect_int(node["feature"], "node feature")
    if feature == -1:
        threshold = _expect_fixed(node["threshold"], "leaf threshold")
        if threshold != 0.0:
            raise ValueError("leaf threshold must be 0.0000000000")
        if node["left"] != [] or node["right"] != []:
            raise ValueError("leaf children must be empty arrays")
        return _DecisionTreeNode(label)
    if feature < 0 or feature >= n_features:
        raise ValueError("node feature must be in [0, n_features_in)")
    threshold = _expect_fixed(node["threshold"], "node threshold")
    result = _DecisionTreeNode(label)
    result.feature = feature
    result.threshold = threshold
    result.left = _load_tree_node(node["left"], n_features)
    result.right = _load_tree_node(node["right"], n_features)
    return result


def _load_forest(pairs, class_name):
    keys = tuple(key for key, _ in pairs)
    if keys != _SERIAL_KEYS_FOREST:
        raise ValueError(
            class_name + " JSON must have exactly the serialized "
            "keys in the serialized order"
        )
    data = _convert(pairs)

    class_entry_name = data["class"]
    if (
        not isinstance(class_entry_name, str)
        or class_entry_name != class_name
    ):
        raise ValueError('class must be "' + class_name + '"')

    n_estimators = _expect_int(data["n_estimators"], "n_estimators")
    if n_estimators <= 0:
        raise ValueError("n_estimators must be greater than 0")
    max_features = _expect_int(data["max_features"], "max_features")
    seed = _expect_int(data["seed"], "seed")
    n_features = _expect_int(data["n_features_in"], "n_features_in")
    if n_features <= 0:
        raise ValueError("n_features_in must be greater than 0")
    if max_features < 1 or max_features > n_features:
        raise ValueError(
            "max_features must satisfy 1 <= max_features <= n_features_in"
        )

    trees_node = data["trees"]
    if not isinstance(trees_node, list) or len(trees_node) != n_estimators:
        raise ValueError("trees must have length n_estimators")
    trees = [_load_tree_node(tree, n_features) for tree in trees_node]

    if class_name == "ExtraTreesClassifier":
        model = ExtraTreesClassifier(
            n_estimators=n_estimators,
            max_features=max_features,
            seed=seed,
        )
    else:
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_features=max_features,
            seed=seed,
        )
    model._trees = trees
    model._n_features = n_features
    return model


def _load_tree_regressor(pairs):
    keys = tuple(key for key, _ in pairs)
    if keys != _SERIAL_KEYS_TREE_REGRESSOR:
        raise ValueError(
            "DecisionTreeRegressor JSON must have exactly the serialized "
            "keys in the serialized order"
        )
    data = _convert(pairs)

    class_name = data["class"]
    if not isinstance(class_name, str) or class_name != "DecisionTreeRegressor":
        raise ValueError('class must be "DecisionTreeRegressor"')

    max_depth_entry = data["max_depth"]
    if isinstance(max_depth_entry, str):
        if max_depth_entry != "none":
            raise ValueError('max_depth must be a positive integer or "none"')
        max_depth = None
    else:
        max_depth = _expect_int(max_depth_entry, "max_depth")
        if max_depth < 1:
            raise ValueError("max_depth must be a positive integer")

    n_features = _expect_int(data["n_features_in"], "n_features_in")
    if n_features <= 0:
        raise ValueError("n_features_in must be greater than 0")

    root = _load_forest_regressor_node(data["tree"], n_features)

    model = DecisionTreeRegressor(max_depth=max_depth)
    model._root = root
    model._n_features = n_features
    return model


def _load_forest_regressor(pairs):
    keys = tuple(key for key, _ in pairs)
    if keys != _SERIAL_KEYS_FOREST_REGRESSOR:
        raise ValueError(
            "RandomForestRegressor JSON must have exactly the serialized "
            "keys in the serialized order"
        )
    data = _convert(pairs)

    class_name = data["class"]
    if not isinstance(class_name, str) or class_name != "RandomForestRegressor":
        raise ValueError('class must be "RandomForestRegressor"')

    n_estimators = _expect_int(data["n_estimators"], "n_estimators")
    if n_estimators <= 0:
        raise ValueError("n_estimators must be greater than 0")

    max_depth_entry = data["max_depth"]
    if isinstance(max_depth_entry, str):
        if max_depth_entry != "none":
            raise ValueError('max_depth must be a positive integer or "none"')
        max_depth = None
    else:
        max_depth = _expect_int(max_depth_entry, "max_depth")
        if max_depth < 1:
            raise ValueError("max_depth must be a positive integer")

    max_features = _expect_int(data["max_features"], "max_features")
    seed = _expect_int(data["seed"], "seed")
    n_features = _expect_int(data["n_features_in"], "n_features_in")
    if n_features <= 0:
        raise ValueError("n_features_in must be greater than 0")
    if max_features < 1 or max_features > n_features:
        raise ValueError(
            "max_features must satisfy 1 <= max_features <= n_features_in"
        )

    trees_node = data["trees"]
    if not isinstance(trees_node, list) or len(trees_node) != n_estimators:
        raise ValueError("trees must have length n_estimators")
    trees = [
        _load_forest_regressor_node(tree, n_features) for tree in trees_node
    ]

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features=max_features,
        seed=seed,
    )
    model._trees = trees
    model._n_features = n_features
    return model


def _load_forest_regressor_node(node, n_features):
    """Rebuild one regressor tree node from its converted JSON object.

    The node must have exactly the keys value, feature, threshold, left,
    right in that order. A leaf has ``feature`` -1, ``threshold``
    0.0000000000 and empty ``left``/``right`` arrays; an internal node
    has a feature in ``[0, n_features)`` and node children. Node values
    and internal thresholds are finite fixed 10-decimal numbers.
    """
    if (
        not isinstance(node, dict)
        or tuple(node.keys()) != _SERIAL_KEYS_FOREST_REGRESSOR_NODE
    ):
        raise ValueError(
            "tree nodes must have exactly the keys "
            "value, feature, threshold, left, right in that order"
        )
    value = _expect_fixed(node["value"], "node value")
    feature = _expect_int(node["feature"], "node feature")
    if feature == -1:
        threshold = _expect_fixed(node["threshold"], "leaf threshold")
        if threshold != 0.0:
            raise ValueError("leaf threshold must be 0.0000000000")
        if node["left"] != [] or node["right"] != []:
            raise ValueError("leaf children must be empty arrays")
        return _DecisionTreeRegressorNode(value)
    if feature < 0 or feature >= n_features:
        raise ValueError("node feature must be in [0, n_features_in)")
    threshold = _expect_fixed(node["threshold"], "node threshold")
    result = _DecisionTreeRegressorNode(value)
    result.feature = feature
    result.threshold = threshold
    result.left = _load_forest_regressor_node(node["left"], n_features)
    result.right = _load_forest_regressor_node(node["right"], n_features)
    return result


def _load_isolation_forest(pairs):
    keys = tuple(key for key, _ in pairs)
    if keys != _SERIAL_KEYS_ISOLATION_FOREST:
        raise ValueError(
            "IsolationForest JSON must have exactly the serialized keys "
            "in the serialized order"
        )
    data = _convert(pairs)

    class_name = data["class"]
    if not isinstance(class_name, str) or class_name != "IsolationForest":
        raise ValueError('class must be "IsolationForest"')

    n_estimators = _expect_int(data["n_estimators"], "n_estimators")
    if n_estimators <= 0:
        raise ValueError("n_estimators must be greater than 0")
    max_samples = _expect_int(data["max_samples"], "max_samples")
    if max_samples < 2:
        raise ValueError("max_samples must be at least 2")

    contamination_entry = data["contamination"]
    if isinstance(contamination_entry, str):
        if contamination_entry != "auto":
            raise ValueError(
                'contamination must be "auto" or a finite number in'
                " (0, 0.5]"
            )
        contamination = "auto"
    else:
        contamination = _expect_fixed(
            contamination_entry, "contamination"
        )
        if not 0.0 < contamination <= 0.5:
            raise ValueError("contamination must be in (0, 0.5]")

    seed = _expect_int(data["seed"], "seed")
    n_features = _expect_int(data["n_features_in"], "n_features_in")
    if n_features <= 0:
        raise ValueError("n_features_in must be greater than 0")
    threshold = _expect_fixed(data["threshold"], "threshold")

    trees_node = data["trees"]
    if not isinstance(trees_node, list) or len(trees_node) != n_estimators:
        raise ValueError("trees must have length n_estimators")
    trees = [
        _load_isolation_node(tree, n_features, max_samples)
        for tree in trees_node
    ]
    for tree in trees:
        if tree.count != max_samples:
            raise ValueError("tree root count must equal max_samples")

    model = IsolationForest(
        n_estimators=n_estimators,
        max_samples=max_samples,
        contamination=contamination,
        seed=seed,
    )
    c_values = [0.0] + [
        _isolation_c(s) for s in range(1, max_samples + 1)
    ]
    model._trees = trees
    model._c_values = c_values
    model._c_psi = c_values[max_samples]
    model._width = n_features
    model.threshold_ = threshold
    return model


def _load_isolation_node(node, n_features, max_samples):
    """Rebuild one isolation tree node from its converted JSON object.

    The node must have exactly the keys count, feature, threshold, left,
    right in that order. ``count`` is a JSON integer in
    ``[1, max_samples]``. A leaf has ``feature`` -1, ``threshold``
    ``0.0000000000`` and empty ``left``/``right`` arrays; an internal
    node has a feature in ``[0, n_features)``, node children whose
    counts sum to ``count``, and a finite fixed 10-decimal threshold.
    """
    if (
        not isinstance(node, dict)
        or tuple(node.keys()) != _SERIAL_KEYS_ISOLATION_NODE
    ):
        raise ValueError(
            "tree nodes must have exactly the keys "
            "count, feature, threshold, left, right in that order"
        )
    count = _expect_int(node["count"], "node count")
    if not 1 <= count <= max_samples:
        raise ValueError("node count must be in [1, max_samples]")
    feature = _expect_int(node["feature"], "node feature")
    if feature == -1:
        threshold = _expect_fixed(node["threshold"], "leaf threshold")
        if threshold != 0.0:
            raise ValueError("leaf threshold must be 0.0000000000")
        if node["left"] != [] or node["right"] != []:
            raise ValueError("leaf children must be empty arrays")
        return _IsolationTreeNode(count)
    if feature < 0 or feature >= n_features:
        raise ValueError("node feature must be in [0, n_features_in)")
    threshold = _expect_fixed(node["threshold"], "node threshold")
    result = _IsolationTreeNode(count)
    result.feature = feature
    result.threshold = threshold
    result.left = _load_isolation_node(
        node["left"], n_features, max_samples
    )
    result.right = _load_isolation_node(
        node["right"], n_features, max_samples
    )
    if result.left.count + result.right.count != count:
        raise ValueError(
            "internal node count must equal the sum of its children"
        )
    return result


def _load_adaboost(pairs):
    keys = tuple(key for key, _ in pairs)
    if keys != _SERIAL_KEYS_ADABOOST:
        raise ValueError(
            "AdaBoostClassifier JSON must have exactly the serialized "
            "keys in the serialized order"
        )
    data = _convert(pairs)

    class_name = data["class"]
    if not isinstance(class_name, str) or class_name != "AdaBoostClassifier":
        raise ValueError('class must be "AdaBoostClassifier"')

    n_estimators = _expect_int(data["n_estimators"], "n_estimators")
    if n_estimators <= 0:
        raise ValueError("n_estimators must be greater than 0")
    n_features = _expect_int(data["n_features_in"], "n_features_in")
    if n_features <= 0:
        raise ValueError("n_features_in must be greater than 0")

    stumps_node = data["stumps"]
    if (
        not isinstance(stumps_node, list)
        or len(stumps_node) == 0
        or len(stumps_node) > n_estimators
    ):
        raise ValueError(
            "stumps must be a non-empty array of at most n_estimators "
            "stumps"
        )
    stumps = [_load_stump(stump, n_features) for stump in stumps_node]

    model = AdaBoostClassifier(n_estimators=n_estimators)
    model._stumps = stumps
    model._n_features = n_features
    return model


def _load_stump(node, n_features):
    """Rebuild one decision stump from its converted JSON object.

    The stump must have exactly the keys feature, threshold, sign, alpha
    in that order: an integer feature in ``[0, n_features)``, a fixed
    10-decimal threshold, a sign of -1 or 1, and a strictly positive
    fixed 10-decimal alpha.
    """
    if not isinstance(node, dict) or tuple(node.keys()) != _SERIAL_KEYS_STUMP:
        raise ValueError(
            "stumps must have exactly the keys "
            "feature, threshold, sign, alpha in that order"
        )
    feature = _expect_int(node["feature"], "stump feature")
    if feature < 0 or feature >= n_features:
        raise ValueError("stump feature must be in [0, n_features_in)")
    threshold = _expect_fixed(node["threshold"], "stump threshold")
    sign = _expect_int(node["sign"], "stump sign")
    if sign not in (-1, 1):
        raise ValueError("stump sign must be -1 or 1")
    alpha = _expect_fixed(node["alpha"], "stump alpha")
    if alpha <= 0.0:
        raise ValueError("stump alpha must be greater than 0")
    return (feature, threshold, sign, alpha)


def _load_boosting(pairs, class_name):
    keys = tuple(key for key, _ in pairs)
    if keys != _SERIAL_KEYS_BOOSTING:
        raise ValueError(
            class_name + " JSON must have exactly the "
            "serialized keys in the serialized order"
        )
    data = _convert(pairs)

    class_entry = data["class"]
    if not isinstance(class_entry, str) or class_entry != class_name:
        raise ValueError('class must be "%s"' % class_name)

    n_estimators = _expect_int(data["n_estimators"], "n_estimators")
    if n_estimators <= 0:
        raise ValueError("n_estimators must be greater than 0")
    learning_rate = _expect_fixed(data["learning_rate"], "learning_rate")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be greater than 0")
    tol = _expect_fixed(data["tol"], "tol")
    if tol <= 0.0:
        raise ValueError("tol must be greater than 0")
    n_features = _expect_int(data["n_features_in"], "n_features_in")
    if n_features <= 0:
        raise ValueError("n_features_in must be greater than 0")
    constant = _expect_fixed(data["constant"], "constant")

    stumps_node = data["stumps"]
    if not isinstance(stumps_node, list) or len(stumps_node) > n_estimators:
        raise ValueError(
            "stumps must be an array of at most n_estimators stumps"
        )
    stumps = [
        _load_boosting_stump(stump, n_features) for stump in stumps_node
    ]

    model_class = {
        "GradientBoostingRegressor": GradientBoostingRegressor,
        "GradientBoostingClassifier": GradientBoostingClassifier,
    }[class_name]
    model = model_class(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        tol=tol,
    )
    model._constant = constant
    model._stumps = stumps
    model._n_features = n_features
    return model


def _load_boosting_stump(node, n_features):
    """Rebuild one boosting stump as a (feature, threshold, left, right)
    tuple from its converted JSON object.

    The object must have exactly the keys feature, threshold, left, right
    in that order: an integer feature in ``[0, n_features)`` followed by
    three fixed 10-decimal numbers.
    """
    if (
        not isinstance(node, dict)
        or tuple(node.keys()) != _SERIAL_KEYS_BOOSTING_STUMP
    ):
        raise ValueError(
            "stumps must have exactly the keys "
            "feature, threshold, left, right in that order"
        )
    feature = _expect_int(node["feature"], "stump feature")
    if feature < 0 or feature >= n_features:
        raise ValueError("stump feature must be in [0, n_features_in)")
    threshold = _expect_fixed(node["threshold"], "stump threshold")
    left = _expect_fixed(node["left"], "stump left increment")
    right = _expect_fixed(node["right"], "stump right increment")
    return (feature, threshold, left, right)


def _load_gaussian(pairs):
    keys = tuple(key for key, _ in pairs)
    if keys != _SERIAL_KEYS_GAUSSIAN:
        raise ValueError(
            "GaussianMixture JSON must have exactly the serialized keys "
            "in the serialized order"
        )
    data = _convert(pairs)

    class_name = data["class"]
    if not isinstance(class_name, str) or class_name != "GaussianMixture":
        raise ValueError('class must be "GaussianMixture"')

    n_components = _expect_int(data["n_components"], "n_components")
    if n_components <= 0:
        raise ValueError("n_components must be greater than 0")

    weights = _expect_fixed12_vector(
        data["weights"], n_components, "weights"
    )
    for value in weights:
        if value <= 0.0:
            raise ValueError("weights must be strictly positive")

    means = _expect_fixed12_vector(data["means"], n_components, "means")

    variances = _expect_fixed12_vector(
        data["variances"], n_components, "variances"
    )
    for value in variances:
        if value < 1e-12:
            raise ValueError("variances must be at least 1e-12")

    model = GaussianMixture(n_components=n_components)
    model.weights_ = list(weights)
    model.means_ = list(means)
    model.variances_ = list(variances)
    return model


def _load_knn_regressor(pairs):
    keys = tuple(key for key, _ in pairs)
    if keys != _SERIAL_KEYS_KNN_REGRESSOR:
        raise ValueError(
            "KNeighborsRegressor JSON must have exactly the serialized "
            "keys in the serialized order"
        )
    data = _convert(pairs)

    class_name = data["class"]
    if not isinstance(class_name, str) or class_name != "KNeighborsRegressor":
        raise ValueError('class must be "KNeighborsRegressor"')

    n_neighbors = _expect_int(data["n_neighbors"], "n_neighbors")
    if n_neighbors < 1:
        raise ValueError("n_neighbors must be a positive integer")

    weights = data["weights"]
    if not isinstance(weights, str) or weights not in ("uniform", "distance"):
        raise ValueError('weights must be "uniform" or "distance"')

    n_features = _expect_int(data["n_features_in"], "n_features_in")
    if n_features < 1:
        raise ValueError("n_features_in must be a positive integer")

    x_node = data["X"]
    if not isinstance(x_node, list) or len(x_node) == 0:
        raise ValueError("X must be a non-empty array of rows")
    if len(x_node) < n_neighbors:
        raise ValueError("X must have at least n_neighbors rows")
    X = []
    for row in x_node:
        if not isinstance(row, list) or len(row) != n_features:
            raise ValueError("every X row must have length n_features_in")
        X.append([_expect_fixed(v, "X element") for v in row])

    y_node = data["y"]
    if not isinstance(y_node, list) or len(y_node) != len(X):
        raise ValueError("y must be an array with the same length as X")
    y = [_expect_fixed(v, "y element") for v in y_node]

    model = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights)
    model._X = X
    model._y = y
    model._width = n_features
    return model


def _load_knn_classifier(pairs):
    keys = tuple(key for key, _ in pairs)
    if keys != _SERIAL_KEYS_KNN_CLASSIFIER:
        raise ValueError(
            "KNeighborsClassifier JSON must have exactly the serialized "
            "keys in the serialized order"
        )
    data = _convert(pairs)

    class_name = data["class"]
    if not isinstance(class_name, str) or class_name != "KNeighborsClassifier":
        raise ValueError('class must be "KNeighborsClassifier"')

    n_neighbors = _expect_int(data["n_neighbors"], "n_neighbors")
    if n_neighbors < 1:
        raise ValueError("n_neighbors must be a positive integer")

    n_features = _expect_int(data["n_features_in"], "n_features_in")
    if n_features < 1:
        raise ValueError("n_features_in must be a positive integer")

    x_node = data["X"]
    if not isinstance(x_node, list) or len(x_node) == 0:
        raise ValueError("X must be a non-empty array of rows")
    if len(x_node) < n_neighbors:
        raise ValueError("X must have at least n_neighbors rows")
    X = []
    for row in x_node:
        if not isinstance(row, list) or len(row) != n_features:
            raise ValueError("every X row must have length n_features_in")
        X.append([_expect_fixed(v, "X element") for v in row])

    y_node = data["y"]
    if not isinstance(y_node, list) or len(y_node) != len(X):
        raise ValueError("y must be an array with the same length as X")
    y = [_expect_int(v, "y element") for v in y_node]

    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model._X = X
    model._y = y
    model._width = n_features
    return model


def _load_agglomerative(pairs):
    keys = tuple(key for key, _ in pairs)
    if keys != _SERIAL_KEYS_AGGLOMERATIVE:
        raise ValueError(
            "AgglomerativeClustering JSON must have exactly the serialized "
            "keys in the serialized order"
        )
    data = _convert(pairs)

    class_name = data["class"]
    if not isinstance(class_name, str) or class_name != "AgglomerativeClustering":
        raise ValueError('class must be "AgglomerativeClustering"')

    n_clusters = _expect_int(data["n_clusters"], "n_clusters")
    if n_clusters <= 0:
        raise ValueError("n_clusters must be a positive integer")

    linkage = data["linkage"]
    if not isinstance(linkage, str) or linkage not in (
        "single",
        "complete",
        "average",
    ):
        raise ValueError(
            'linkage must be "single", "complete", or "average"'
        )

    labels_node = data["labels"]
    if not isinstance(labels_node, list) or len(labels_node) == 0:
        raise ValueError("labels must be a non-empty array")
    n = len(labels_node)
    if n_clusters > n:
        raise ValueError("n_clusters must not exceed the number of labels")
    labels = [_expect_int(value, "labels element") for value in labels_node]

    children_node = data["children"]
    merge_count = n - n_clusters
    if not isinstance(children_node, list) or len(children_node) != merge_count:
        raise ValueError("children must have length n - n_clusters")
    children = []
    for pair in children_node:
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError("every children entry must be an integer pair")
        children.append(
            [
                _expect_int(pair[0], "children element"),
                _expect_int(pair[1], "children element"),
            ]
        )

    # Replay the merges: the r-th pair must reference two distinct
    # surviving clusters over leaves 0..n-1, creating node n+r.
    clusters = _agglomerative_members(n, children)
    if len(clusters) != n_clusters:
        raise ValueError(
            "the merges must leave exactly n_clusters clusters"
        )
    expected_labels = [0] * n
    for label, cluster_members in enumerate(clusters):
        for index in cluster_members:
            expected_labels[index] = label
    if labels != expected_labels:
        raise ValueError(
            "labels must number the clusters by ascending smallest member"
        )

    distances_node = data["distances"]
    if not isinstance(distances_node, list) or len(distances_node) != merge_count:
        raise ValueError("distances must have the same length as children")
    distances = []
    for value in distances_node:
        distances.append(_expect_agglomerative_distance(value))

    model = AgglomerativeClustering(
        n_clusters=n_clusters, linkage=linkage
    )
    # Fresh lists (including fresh inner pairs) so the fitted state never
    # shares storage with the parsed payload.
    model.labels_ = list(labels)
    model.children_ = [list(pair) for pair in children]
    model.distances_ = list(distances)
    return model


def _load_dbscan(pairs):
    keys = tuple(key for key, _ in pairs)
    if keys != _SERIAL_KEYS_DBSCAN:
        raise ValueError(
            "DBSCAN JSON must have exactly the serialized keys in the "
            "serialized order"
        )
    data = _convert(pairs)

    class_name = data["class"]
    if not isinstance(class_name, str) or class_name != "DBSCAN":
        raise ValueError('class must be "DBSCAN"')

    eps_entry = data["eps"]
    if not (
        isinstance(eps_entry, tuple)
        and eps_entry[0] == "fixed"
        and _JSON_FLOAT_RE.match(eps_entry[1])
    ):
        raise ValueError("eps must be a fixed 10-decimal JSON number")
    eps_token = eps_entry[1]
    if eps_token[0] == "-":
        raise ValueError("eps must be greater than 0")
    eps = float(eps_token)
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError("eps must be a finite number greater than 0")

    min_samples = _expect_int(data["min_samples"], "min_samples")
    if min_samples <= 0:
        raise ValueError("min_samples must be a positive integer")

    labels_node = data["labels"]
    if not isinstance(labels_node, list) or len(labels_node) == 0:
        raise ValueError("labels must be a non-empty array")
    labels = [_expect_int(value, "labels element") for value in labels_node]
    next_label = 0
    seen = set()
    for value in labels:
        if value == -1:
            continue
        if value < 0:
            raise ValueError(
                "labels must contain only -1 or consecutive cluster labels"
            )
        if value not in seen:
            if value != next_label:
                raise ValueError(
                    "non-negative labels must first appear in order "
                    "0, 1, 2, ..."
                )
            seen.add(value)
            next_label += 1

    model = DBSCAN(eps=eps, min_samples=min_samples)
    # A fresh list so the fitted state never shares storage with the
    # parsed payload.
    model.labels_ = list(labels)
    return model


def loads(text):
    """Reconstruct a fitted KMeans, PCA, LinearRegression,
    LogisticRegression, LassoRegression, ElasticNetRegression,
    MultinomialLogisticRegression, StandardScaler,
    DecisionTreeClassifier, DecisionTreeRegressor,
    RandomForestClassifier, ExtraTreesClassifier,
    RandomForestRegressor, AdaBoostClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier,
    GaussianMixture, KNeighborsRegressor, KNeighborsClassifier,
    AgglomerativeClustering, DBSCAN, or IsolationForest from text
    produced by dumps.

    Only the exact byte format emitted by :func:`dumps` is accepted: a
    ``str`` holding compact JSON with no whitespace, no duplicate keys,
    the exact key sets in order, JSON integers for integer parameters,
    10-decimal fixed-point numbers for floats (12-decimal for
    GaussianMixture), and consistent array shapes. Anything else --
    including non-str input (str subclasses included), empty strings,
    parse failures, booleans, exponent notation, non-finite values, or
    illegal parameters -- raises ValueError. The returned model is
    independent of the input and fitted; KMeans recovers its column
    count from centroid width, the linear models from the length of
    ``w``, LassoRegression likewise from the length of its non-empty
    ``w`` (copied rather than shared) with strictly positive
    ``alpha``/``tol`` and a positive ``max_iter``,
    ElasticNetRegression likewise from the length of its non-empty
    ``w`` (copied rather than shared) with strictly positive
    ``alpha``/``tol``, an ``l1_ratio`` in ``[0, 1]`` and a positive
    ``max_iter``,
    MultinomialLogisticRegression from the column count of ``W``
    (with its ``classes``/``W``/``b`` arrays copied rather than shared),
    StandardScaler/DecisionTreeClassifier/DecisionTreeRegressor/
    RandomForestClassifier/ExtraTreesClassifier/RandomForestRegressor
    from ``n_features_in``,
    AdaBoostClassifier from ``n_features_in``,
    GradientBoostingRegressor/GradientBoostingClassifier from
    ``n_features_in``,
    GaussianMixture from ``n_components``, KNeighborsRegressor from
    ``n_features_in`` with its stored ``X``/``y`` arrays copied rather
    than shared, KNeighborsClassifier from ``n_features_in`` with its
    stored ``X``/``y`` arrays copied rather than shared (``X``
    elements are finite fixed 10-decimal numbers and ``y`` elements are
    JSON integers), and AgglomerativeClustering from the length of
    ``labels`` with its ``labels``, ``children`` (inner pairs included),
    and ``distances`` lists copied rather than shared; the children
    merges are replayed and ``labels`` must number the resulting
    clusters by ascending smallest member. Distances that are finite as
    ``float`` come back as ``float``; a distance whose magnitude is
    outside the finite float range comes back as an exact ``int`` only
    when its fraction is all zero. DBSCAN is reconstructed from
    ``eps`` (a fixed 10-decimal number), ``min_samples`` (a positive
    JSON integer), and a fresh copy of ``labels`` containing only -1
    and labels consecutive from zero in first-appearance order.
    IsolationForest is reconstructed from ``n_features_in`` with the
    four construction fields (``n_estimators``, ``max_samples``,
    ``contamination`` as ``"auto"`` or a fixed 10-decimal number, and
    ``seed``), the fixed 10-decimal decision ``threshold``, and
    exactly ``n_estimators`` trees rebuilt from fresh
    ``_IsolationTreeNode`` instances (no shared structure); each node
    count is a JSON integer in ``[1, max_samples]``, leaves carry
    ``feature`` -1, a zero threshold and empty children, internal
    nodes carry a feature in ``[0, n_features)``, fixed 10-decimal
    threshold and children whose counts sum to the node count, and
    every tree root has count ``max_samples``.
    The argument is not modified.
    """
    if type(text) is not str or len(text) == 0:
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
    if class_entry == "LinearRegression":
        try:
            return _load_linear(pairs, LinearRegression, "LinearRegression")
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError("malformed serialized model") from exc
    if class_entry == "LogisticRegression":
        try:
            return _load_linear(pairs, LogisticRegression, "LogisticRegression")
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError("malformed serialized model") from exc
    if class_entry == "LassoRegression":
        try:
            return _load_lasso(pairs)
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError("malformed serialized model") from exc
    if class_entry == "ElasticNetRegression":
        try:
            return _load_elastic_net(pairs)
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError("malformed serialized model") from exc
    if class_entry == "MultinomialLogisticRegression":
        try:
            return _load_multinomial(pairs)
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError("malformed serialized model") from exc
    if class_entry == "StandardScaler":
        try:
            return _load_scaler(pairs)
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError("malformed serialized model") from exc
    if class_entry == "DecisionTreeClassifier":
        try:
            return _load_tree(pairs)
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError("malformed serialized model") from exc
    if class_entry == "RandomForestClassifier":
        try:
            return _load_forest(pairs, "RandomForestClassifier")
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError("malformed serialized model") from exc
    if class_entry == "ExtraTreesClassifier":
        try:
            return _load_forest(pairs, "ExtraTreesClassifier")
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError("malformed serialized model") from exc
    if class_entry == "RandomForestRegressor":
        try:
            return _load_forest_regressor(pairs)
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError("malformed serialized model") from exc
    if class_entry == "DecisionTreeRegressor":
        try:
            return _load_tree_regressor(pairs)
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError("malformed serialized model") from exc
    if class_entry == "AdaBoostClassifier":
        try:
            return _load_adaboost(pairs)
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError("malformed serialized model") from exc
    if class_entry == "GradientBoostingRegressor":
        try:
            return _load_boosting(pairs, "GradientBoostingRegressor")
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError("malformed serialized model") from exc
    if class_entry == "GradientBoostingClassifier":
        try:
            return _load_boosting(pairs, "GradientBoostingClassifier")
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError("malformed serialized model") from exc
    if class_entry == "GaussianMixture":
        try:
            return _load_gaussian(pairs)
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError("malformed serialized model") from exc
    if class_entry == "KNeighborsRegressor":
        try:
            return _load_knn_regressor(pairs)
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError("malformed serialized model") from exc
    if class_entry == "KNeighborsClassifier":
        try:
            return _load_knn_classifier(pairs)
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError("malformed serialized model") from exc
    if class_entry == "AgglomerativeClustering":
        try:
            return _load_agglomerative(pairs)
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError("malformed serialized model") from exc
    if class_entry == "DBSCAN":
        try:
            return _load_dbscan(pairs)
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError("malformed serialized model") from exc
    if class_entry == "IsolationForest":
        try:
            return _load_isolation_forest(pairs)
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
