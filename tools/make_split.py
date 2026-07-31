"""Proposes the exemplar / final-check / dev-eval split for config.yaml.

Usage: python tools/make_split.py --dataset-dir data/dataset --seed 42
"""
import json
import random
from collections import defaultdict
from pathlib import Path


def normalize(s: str) -> str:
    """Normalize a string for comparison: lowercase, collapse all whitespace to single spaces.

        Args:
            s: Raw string (e.g., an injected_fault description).

        Returns:
            Normalized string; differences in case and spacing are erased.
        """
    return " ".join(s.lower().split())


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
            else:
                records.append(data)
    print(f"Found {len(records)} json files in {dataset_dir}")
    return records


def propose_exemplar_candidates(examples: list[dict]) -> dict[str, list[dict]]:
    """Rank exemplar candidates per class for human curation.

        Groups examples by label, computes each log's line count (stored as
        'log_lines'), sorts synthetic-first then shortest-first, and keeps the
        top 5 per class. Final exemplar selection is a human decision; this
        function only shortlists.

        Args:
            examples: Metadata dicts from load_metadata.

        Returns:
            Mapping of label -> top 5 candidate records, ranked.
        """
    exemplar_candidates: defaultdict[str, list[dict]] = defaultdict(list)
    for record in examples:
        label = record['label']
        exemplar_candidates[label].append(record)
    for label, items in exemplar_candidates.items():
        for r in items:
            with open(r['log_path'], 'r', encoding='utf-8', errors='ignore') as f:
                r['log_lines'] = sum(1 for _ in f)
        items.sort(key=lambda x: (0 if x.get('log_source') == 'synthetic' else 1, x['log_lines']))
        items[:] = items[:5]
    return exemplar_candidates


def pick_final_check(examples: list[dict], exemplar_ids: set[str],
                     seed: int, quotas: dict[str, int]) -> list[str]:
    """Randomly select the stratified, touch-once final-check set.

        Excludes exemplars, groups the remaining pool by label, and samples
        quota[label] IDs per class without replacement using a seeded RNG.
        Pools are sorted before sampling so results are reproducible across
        filesystems and platforms.

        Args:
            examples: Metadata dicts from load_metadata.
            exemplar_ids: IDs reserved as few-shot exemplars (excluded from the pool).
            seed: RNG seed; same seed + same dataset = same selection.
            quotas: Mapping of label -> number of examples to select.

        Returns:
            Sorted list of selected final-check IDs.

        Raises:
            ValueError: If a class has fewer eligible examples than its quota.
        """
    pool: defaultdict[str, list[str]] = defaultdict(list)
    final_check_ids: list[str] = []
    rng = random.Random(seed)
    for example in examples:
        if example['id'] not in exemplar_ids:
            pool[example['label']].append(example['id'])
    for label, k in quotas.items():
        class_ids = sorted(pool[label])
        try:
            picked = rng.sample(class_ids, k)
        except ValueError:
            raise ValueError(f"Class '{label}' has {len(class_ids)} eligible examples, need {k}")
        final_check_ids.extend(picked)
    return sorted(final_check_ids)


def check_leakage(examples: list[dict],
                  exemplar_ids: set[str]) -> list[tuple[str, str, str]]:
    """Detect near-duplicate leakage across the exemplar/eval boundary.

        Compares every exemplar against every eval example (non-exemplar) by
        normalized 'injected_fault' string. A match means the model would be
        evaluated on an example it effectively saw in its prompt.

        Args:
            examples: Metadata dicts from load_metadata.
            exemplar_ids: IDs used as few-shot exemplars.

        Returns:
            Flagged (exemplar_id, eval_id, reason) tuples; empty list = no leakage.
        """
    exemplars = [r for r in examples if r['id'] in exemplar_ids]
    evals = [r for r in examples if r['id'] not in exemplar_ids]

    flagged = []  # result list, starts empty
    for ex in exemplars:
        for ev in evals:
            if normalize(ex['injected_fault']) == normalize(ev['injected_fault']):
                flagged.append((ex['id'], ev['id'], "identical injected_fault"))
    return flagged


def emit_yaml(exemplars: dict[str, list[str]], final_check: list[str], seed: int) -> str:
    """Format the split as the YAML block to paste into config.yaml.

        Classes and IDs are emitted in sorted order so repeated runs produce
        byte-identical output (clean git diffs). The dev-eval set is not listed:
        it is derived as the complement of exemplars + final_check.

        Args:
            exemplars: Mapping of label -> chosen exemplar IDs.
            final_check: Selected final-check IDs.
            seed: The selection seed, recorded for provenance.

        Returns:
            YAML text for a top-level 'split:' section.
        """
    lines = ["split:", f"  selection_seed: {seed}", "  exemplars:"]
    for label in sorted(exemplars):
        ids = ", ".join(sorted(exemplars[label]))
        lines.append(f"    {label}: [{ids}]")
    fc = ", ".join(sorted(final_check))
    lines.append(f"  final_check: [{fc}]")
    return "\n".join(lines)


def main() -> None:
    path = Path("/Users/roshanpandey/Python Projects/ci-triage-dataset/dataset")  # temporary; argparse replaces this
    examples = load_metadata(path)
    candidates = propose_exemplar_candidates(examples)

    exemplars = {
        "infra": ["infra_006", "infra_008", "infra_002"],
        "genuine_regression": ["genuine_regression_015", "genuine_regression_013", "genuine_regression_017"],
        "transient": ["transient_014", "transient_011", "transient_010"],
        "flaky_test": ["flaky_test_006", "flaky_test_009", "flaky_test_013"],
    }
    exemplar_ids = {i for ids in exemplars.values() for i in ids}

    final_check_ids = pick_final_check(examples, exemplar_ids, seed=42,
                                       quotas={"genuine_regression": 3, "flaky_test": 3, "transient": 2, "infra": 2})

    for label, items in candidates.items():  # present
        print(f"\n=== {label} ===")
        for r in items:
            print(f"{r['id']:<28} {r['log_source']:<10} {r['log_lines']:>4}  {r['injected_fault'][:80]}")

    print(f"\nThe final_check_ids are {final_check_ids}, and the size is {len(final_check_ids)}")

    leaked_items = check_leakage(examples, exemplar_ids)
    if leaked_items:
        print(f"Found leakage: {leaked_items}")
    else:
        print("No leakage found")

    print("\n--- paste into config.yaml ---")
    print(emit_yaml(exemplars, final_check_ids, seed=42))
    print("--- end ---")


if __name__ == "__main__":
    main()
