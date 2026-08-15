"""Read labelled examples from the dataset directory.

Each example is a pair of files in a per-class folder: an <n>.log holding the
CI output and an <n>.json sidecar holding its metadata. The sidecar also
carries fields that state the answer outright (injected_fault, failing_tests);
Example deliberately does not hold them, so they cannot reach a prompt.
"""
import json
from pathlib import Path

from pydantic import BaseModel

from src.config_loader import find_project_root
from src.categories import Category


class Example(BaseModel):
    """One dataset example, narrowed to the fields the pipeline may use.

    The split between labeler-visible and evaluation-only fields is the reason
    this model exists rather than passing raw sidecar dicts around: only
    log_text may ever reach a prompt, and true_label is read solely when
    scoring predictions.

    Attributes:
        example_id: Stable ID from the sidecar, e.g. 'flaky_test_006'. Taken
            from the file's contents rather than its name, which is just '006'.
        log_text: Raw CI log. The only field a prompt may contain.
        true_label: Ground truth, for evaluation only. Literal-constrained, so
            a misfiled or misspelled category fails at load rather than
            producing a class that silently never matches.
    """
    example_id: str
    log_text: str  # labeler-visible
    true_label: Category


def load_metadata(dataset_dir: Path) -> list[dict]:
    """Load all example metadata JSONs from the dataset directory.

    Walks dataset_dir/<label>/<n>.json, parses each file, validates that the
    'label' field matches its parent folder name, and attaches the absolute
    path of the example's log file under the 'log_path' key.

    Args:
        dataset_dir: Root dataset directory containing one subfolder per class.

    Returns:
        All examples as metadata dicts (one per JSON file).

    Raises:
        ValueError: If a record's 'label' field does not match its folder name.
    """
    records = []
    for file_path in dataset_dir.glob('*/*.json'):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            data['log_path'] = (file_path.parent / data['log_file']).resolve()
            if data['label'] != file_path.parent.name:
                raise ValueError(
                    f"Label mismatch in {file_path}: found '{data['label']}', folder '{file_path.parent.name}'")
            records.append(data)
    # print(f"Found {len(records)} json files in {dataset_dir}")
    return records


def load_examples(config, split: str | None = None) -> list[Example]:
    """Load dataset examples, optionally filtered to one split.

    Args:
        config: Loaded PipelineConfig.
        split: 'exemplars', 'final_check', 'dev_eval', or None for all.
               'dev_eval' is derived as the complement of the other two.

    Returns:
        Examples sorted by example_id.

    Raises:
        ValueError: On an unknown split, or if a configured ID has no file on disk.
    """
    if split not in {'exemplars', 'final_check', 'dev_eval', None}:
        raise ValueError(
            f"Invalid split {split!r}. Expected 'exemplars', 'final_check', "
            f"'dev_eval', or None.")

    records = load_metadata(find_project_root() / config.dataset.local_dir)

    exemplar_ids = {i for ids in config.split.exemplars.values() for i in ids}
    final_check_ids = set(config.split.final_check)

    expected_ids = None
    if split == 'exemplars':
        expected_ids = exemplar_ids
    elif split == 'final_check':
        expected_ids = final_check_ids

    if expected_ids is not None:
        selected = [r for r in records if r['id'] in expected_ids]
    elif split == 'dev_eval':
        selected = [r for r in records
                    if r['id'] not in exemplar_ids and r['id'] not in final_check_ids]
    else:
        selected = records

    if expected_ids is not None and len(selected) != len(expected_ids):
        missing = sorted(expected_ids - {r['id'] for r in selected})
        raise ValueError(
            f"Split '{split}': config lists {len(expected_ids)} IDs but "
            f"{len(selected)} were found on disk. Missing: {missing}")

    examples = [
        Example(
            example_id=r['id'],
            log_text=r['log_path'].read_text(encoding='utf-8'),
            true_label=r['label'],
        )
        for r in selected
    ]
    return sorted(examples, key=lambda e: e.example_id)