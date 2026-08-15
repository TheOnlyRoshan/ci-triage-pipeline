"""Compute classification metrics from predicted and true labels.

Pure functions over label lists — no config, no I/O, no API calls, so every
function here is testable against hand-written inputs.

Metrics are reported per class rather than as a single accuracy figure. The
dev-eval split is imbalanced (10 genuine_regression against 3 infra), so an
aggregate number is dominated by the largest class and hides failure on the
smallest.
"""

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
    metrics: dict[str, dict[str, float | int]] = {}

    for category in categories:
        true_positives = matrix[category][category]
        false_positives = sum(
            matrix[other][category] for other in categories if other != category
        )
        false_negatives = sum(
            matrix[category][other] for other in categories if other != category
        )
        support = sum(matrix[category].values())

        predicted_total = true_positives + false_positives
        precision = true_positives / predicted_total if predicted_total else 0.0

        actual_total = true_positives + false_negatives
        recall = true_positives / actual_total if actual_total else 0.0

        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

        metrics[category] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support,
        }

    return metrics

def format_report(
        metrics: dict[str, dict[str, float | int]],
        matrix: dict[str, dict[str, int]],
        categories: list[str],
) -> str:
    """Render metrics and confusion matrix as an aligned text report.

    Returns a string rather than printing, so the caller can send it to a
    terminal, a file, or EXPERIMENTS.md without this function knowing which.

    Accuracy and both baselines are derived from the matrix rather than passed
    in. A hardcoded baseline goes stale the moment the split changes, and a
    separately-supplied accuracy can disagree with the table beneath it.

    Args:
        metrics: Output of per_class_metrics.
        matrix: Output of build_confusion_matrix.
        categories: All valid categories, fixing display order.

    Returns:
        Multi-line report: per-class table, confusion matrix, baselines, and
        a low-support warning when any class has fewer than five examples.
    """
    total = sum(sum(row.values()) for row in matrix.values())
    correct = sum(matrix[c][c] for c in categories)
    accuracy = correct / total if total else 0.0

    name_width = max(len(c) for c in categories) + 2
    rule = "-" * (name_width + 40)

    lines = ["CLASSIFICATION REPORT", "=" * 21, ""]

    lines.append(
        f"{'Class':<{name_width}}{'Precision':>10}{'Recall':>10}{'F1':>10}{'Support':>10}"
    )
    lines.append(rule)

    for category in categories:
        m = metrics[category]
        lines.append(
            f"{category:<{name_width}}"
            f"{m['precision']:>10.2f}"
            f"{m['recall']:>10.2f}"
            f"{m['f1']:>10.2f}"
            f"{m['support']:>10}"
        )

    lines.append(rule)
    lines.append(f"{'Overall accuracy':<{name_width}}{accuracy:>10.2f}{'':>20}{total:>10}")

    lines.append("")
    lines.append("CONFUSION MATRIX")
    lines.append("(rows = true label, columns = predicted)")
    lines.append("")

    header = f"{'true \\ pred':<{name_width}}"
    for category in categories:
        header += f"{category[:9]:>10}"
    lines.append(header)
    lines.append(rule)

    for true_label in categories:
        row = f"{true_label:<{name_width}}"
        for predicted_label in categories:
            row += f"{matrix[true_label][predicted_label]:>10}"
        lines.append(row)

    lines.append("")
    lines.append("BASELINES")
    random_baseline = 1 / len(categories) if categories else 0.0
    majority_class = max(categories, key=lambda c: metrics[c]['support'])
    majority_baseline = metrics[majority_class]['support'] / total if total else 0.0
    lines.append(f"{'Random (' + str(len(categories)) + ' classes)':<{name_width}}{random_baseline:>10.2f}")
    lines.append(f"{'Majority class':<{name_width}}{majority_baseline:>10.2f}   ({majority_class})")

    low_support = [c for c in categories if metrics[c]['support'] < 5]
    if low_support:
        lines.append("")
        lines.append(
            f"NOTE: low support for {', '.join(low_support)}. With fewer than five "
            f"examples, one misclassification moves recall by 20+ points."
        )

    return "\n".join(lines)