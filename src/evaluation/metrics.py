"""Compute classification metrics from predicted and true labels.

Pure functions over label lists — no config, no I/O, no API calls, so every
function here is testable against hand-written inputs.

Metrics are reported per class rather than as a single accuracy figure. The
dev-eval split is imbalanced (10 genuine_regression against 3 infra), so an
aggregate number is dominated by the largest class and hides failure on the
smallest.
"""
from collections import defaultdict


def build_confusion_matrix(
        true_labels: list[str],
        predicted_labels: list[str],
        categories: list[str],
) -> dict[str, dict[str, int]]:
    """Count predictions by (true, predicted) pair.

    Args:
        true_labels: Ground-truth label per example.
        predicted_labels: Predicted label per example, same order and length.
        categories: All valid categories, fixing row and column order so the
            matrix shape does not depend on which labels happen to appear.

    Returns:
        Nested dict where matrix[true][predicted] is a count. Every category
        appears as both a row and a column, including categories with no
        examples.

    Raises:
        ValueError: If the two label lists differ in length, or contain a
            label not in categories.
    """
    if len(true_labels) != len(predicted_labels):
        raise ValueError(
            f"Length mismatch: {len(true_labels)} true labels, "
            f"{len(predicted_labels)} predicted labels")

    for label in true_labels + predicted_labels:
        if label not in categories:
            raise ValueError(
                f"Unknown label {label!r}; expected one of {categories}")

    confusion_matrix = {t: {p: 0 for p in categories} for t in categories}
    for true_label, predicted_label in zip(true_labels, predicted_labels):
        confusion_matrix[true_label][predicted_label] += 1

    return confusion_matrix


def per_class_metrics(
        matrix: dict[str, dict[str, int]],
        categories: list[str],
) -> dict[str, dict[str, float | int]]:
    """Derive precision, recall, F1 and support from a confusion matrix.

    Support is included because it is what makes the other three
    interpretable: recall of 0.67 over 3 examples is two correct answers, not
    evidence of a 67% success rate.

    Args:
        matrix: Output of build_confusion_matrix.
        categories: All valid categories.

    Returns:
        Per category: precision, recall, f1 (floats) and support (int, the
        number of examples whose true label is this category). Precision and
        recall are 0.0 when their denominator is zero — the model never
        predicted the class, or the class has no examples.
    """
    i: int = 0
    true_positive = [i for category in matrix for predicted_value in category if i == predicted_value]


