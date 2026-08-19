"""Manual smoke test for the dataset loader and label store.

Loads the dev-eval split, labels the first example twice, and reports whether
the second call was served from the store. Not a unit test, since it makes a
live API call on the first run.

Run from the project root:
    python -m tools.smoke_test
"""
import time

from dotenv import load_dotenv

from src.config_loader import load_config, find_project_root
from src.dataset.loader import load_examples
from src.labeler.label_store import get_or_create, find_label
from src.labeler.log_preprocessor import preprocess_log


def main() -> None:
    """Load one example, label it twice, and check the second call was cached."""
    load_dotenv()
    config = load_config(find_project_root() / 'config.yaml')

    examples = load_examples(config, 'dev_eval')
    print(f"Loaded {len(examples)} dev-eval examples")

    example = examples[0]
    print(f"Using: {example.example_id} (true label: {example.true_label})")

    log_text = preprocess_log(example.log_text, config)

    print(f"In store before first call: {find_label(example.example_id, config)}")

    start = time.perf_counter()
    first = get_or_create(example.example_id, log_text, config)
    print(f"First call:  {first.label} @ {first.confidence}  ({time.perf_counter() - start:.2f}s)")

    start = time.perf_counter()
    second = get_or_create(example.example_id, log_text, config)
    print(f"Second call: {second.label} @ {second.confidence}  ({time.perf_counter() - start:.2f}s)")

    print(f"Correct: {first.label == example.true_label}")


if __name__ == "__main__":
    main()
