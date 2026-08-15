"""Run the labeler over a dataset split and report classification metrics.

Predictions go through the label store, so the first run pays for one API call
per example and subsequent runs are free unless the model, prompt version, or
preprocessing variant changes.

Defaults to the dev-eval split. The final-check split is a holdout and must be
requested explicitly: every number you look at and act on fits your decisions
a little more tightly to that split, so it stops being an honest estimate.

Run from the project root:
    python -m tools.run_eval [--split dev_eval] [--output report.txt]
"""
import argparse
from pathlib import Path

from dotenv import load_dotenv

from src.config_loader import load_config, find_project_root
from src.dataset.loader import Example, load_examples
from src.evaluation.metrics import build_confusion_matrix, per_class_metrics, format_report
from src.labeler.label_store import get_or_create
from src.labeler.log_preprocessor import preprocess_log


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Namespace with split (str) and output (str | None).
    """
    parser = argparse.ArgumentParser(
        description="Label a dataset split and report classification metrics.")
    parser.add_argument("--split", type=str, default="dev_eval", choices=['exemplars', 'dev_eval', 'final_check'],
                        help="Split to run.")
    parser.add_argument("--output", type=str, default=None, help="Output report.")
    return parser.parse_args()


def collect_predictions(
        examples: list[Example],
        config,
) -> tuple[list[str], list[str]]:
    """Label every example and return true and predicted labels in order.

    Predictions go through the label store, so only examples not already
    labelled under the current key incur an API call. Progress is printed per
    example because a full uncached run takes several minutes.

    The two returned lists are positionally aligned: index i of each refers to
    the same example. Nothing enforces that beyond both being appended in the
    same iteration, so neither list may be sorted or filtered independently.

    Args:
        examples: Examples to label, from load_examples.
        config: Loaded PipelineConfig.

    Returns:
        (true_labels, predicted_labels), same length and order as examples.

    Raises:
        ValueError: If a model response cannot be parsed. Examples already
            labelled are persisted, so a re-run resumes rather than restarts.
    """
    true_labels = []
    predicted_labels = []
    for serial_number, example in enumerate(examples, start=1):
        preprocessed = preprocess_log(example.log_text, config)
        result = get_or_create(example.example_id, preprocessed, config)
        true_labels.append(example.true_label)
        predicted_labels.append(result.label)
        status = 'ok' if example.true_label == result.label else 'MISS'
        print(f"[{serial_number:>2}/{len(examples)}] "
              f"{example.example_id:<26} -> {result.label:<20} "
              f"(true: {example.true_label:<20}) {status}")
    return true_labels, predicted_labels


def main() -> None:
    """CLI entry point: load a split, label it, print and optionally save the report."""
    load_dotenv()
    config = load_config(find_project_root() / "config.yaml")
    args = parse_args()
    examples = load_examples(config, args.split)
    print(f"Loaded {len(examples)} examples.")
    true_labels, predicted_labels = collect_predictions(examples, config)
    matrix = build_confusion_matrix(true_labels, predicted_labels, config.categories)
    metrics = per_class_metrics(matrix, config.categories)
    report = format_report(metrics, matrix, config.categories)
    print(report)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding='utf-8')
        print(f"\nReport written to {output_path}")


if __name__ == "__main__":
    main()
