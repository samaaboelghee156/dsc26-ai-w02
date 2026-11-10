"""Decision metrics for a mule-risk score.

Keep the public function names and signatures unchanged: the DSC grader imports
these functions directly.
"""
from __future__ import annotations

import math

import numpy as np


def _validated_arrays(y_true, scores) -> tuple[np.ndarray, np.ndarray]:
    """Validate inputs shared by all metric functions and return 1-D arrays."""
    try:
        labels = np.asarray(y_true)
        raw_scores = np.asarray(scores)
    except (TypeError, ValueError) as exc:
        raise ValueError("y_true and scores must be one-dimensional sequences") from exc

    if labels.ndim != 1 or raw_scores.ndim != 1:
        raise ValueError("y_true and scores must be one-dimensional")
    if labels.size == 0 or raw_scores.size == 0:
        raise ValueError("y_true and scores must not be empty")
    if labels.size != raw_scores.size:
        raise ValueError("y_true and scores must have the same length")

    # Accept numeric 0/1 labels (including bool and 0.0/1.0), but not strings
    # such as "1", missing values, or any other class.
    try:
        valid_labels = np.isin(labels, [0, 1])
        if not bool(np.all(valid_labels)):
            raise ValueError("every label must be 0 or 1")
        labels = labels.astype(int)
    except (TypeError, ValueError) as exc:
        if isinstance(exc, ValueError) and str(exc) == "every label must be 0 or 1":
            raise
        raise ValueError("every label must be 0 or 1") from exc

    # Convert scores to real numbers so comparisons and sorting behave
    # consistently. Infinite scores are permitted; NaN scores are not.
    try:
        score_values = np.asarray(raw_scores, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("scores must be numeric") from exc
    if np.isnan(score_values).any():
        raise ValueError("scores must not contain NaN")

    return labels, score_values


def confusion_at(y_true, scores, threshold: float) -> dict:
    """Return TP, FP, TN and FN counts at ``threshold``.

    A score equal to the threshold is flagged.
    """
    labels, score_values = _validated_arrays(y_true, scores)
    flagged = score_values >= threshold
    positive = labels == 1
    negative = ~positive

    return {
        "tp": int(np.count_nonzero(flagged & positive)),
        "fp": int(np.count_nonzero(flagged & negative)),
        "tn": int(np.count_nonzero(~flagged & negative)),
        "fn": int(np.count_nonzero(~flagged & positive)),
    }


def precision_recall_at(y_true, scores, threshold: float) -> tuple[float, float]:
    """Return (precision, recall) at ``threshold``."""
    counts = confusion_at(y_true, scores, threshold)
    predicted_positive = counts["tp"] + counts["fp"]
    actual_positive = counts["tp"] + counts["fn"]

    precision = counts["tp"] / predicted_positive if predicted_positive else 0.0
    recall = counts["tp"] / actual_positive if actual_positive else 0.0
    return float(precision), float(recall)


def precision_at_k(y_true, scores, k: int) -> float:
    """Share of positives among the k highest scores.

    Tied scores retain their original input order. If k exceeds the number of
    cases, all cases are used.
    """
    labels, score_values = _validated_arrays(y_true, scores)
    if k <= 0:
        raise ValueError("k must be greater than 0")

    k_used = min(int(k), labels.size)
    # A stable sort preserves the original order for equal scores.
    order = np.argsort(-score_values, kind="stable")
    return float(np.mean(labels[order[:k_used]]))


def expected_cost(
    y_true,
    scores,
    threshold: float,
    *,
    cost_fp: float,
    cost_fn: float,
) -> float:
    """Return FP * cost_fp + FN * cost_fn at ``threshold``."""
    counts = confusion_at(y_true, scores, threshold)
    return float(counts["fp"] * cost_fp + counts["fn"] * cost_fn)


def best_threshold(
    y_true,
    scores,
    *,
    cost_fp: float,
    cost_fn: float,
) -> float:
    """Return the minimum-cost distinct score, preferring the highest on ties.

    ``math.inf`` is also considered; it represents flagging no cases.
    """
    _, score_values = _validated_arrays(y_true, scores)
    candidates = np.append(np.unique(score_values), math.inf)

    best_t = float(candidates[0])
    best_cost = expected_cost(
        y_true, score_values, best_t, cost_fp=cost_fp, cost_fn=cost_fn
    )

    for candidate in candidates[1:]:
        threshold = float(candidate)
        cost = expected_cost(
            y_true, score_values, threshold, cost_fp=cost_fp, cost_fn=cost_fn
        )
        if cost < best_cost or (cost == best_cost and threshold > best_t):
            best_t = threshold
            best_cost = cost

    return best_t


def threshold_for_precision(
    y_true,
    scores,
    min_precision: float,
) -> float | None:
    """Return the lowest distinct score meeting ``min_precision``, if any."""
    _, score_values = _validated_arrays(y_true, scores)

    for candidate in np.unique(score_values):
        threshold = float(candidate)
        precision, _ = precision_recall_at(y_true, score_values, threshold)
        if precision >= min_precision:
            return threshold

    return None
