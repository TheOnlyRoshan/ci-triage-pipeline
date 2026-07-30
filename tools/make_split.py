"""Proposes the exemplar / final-check / dev-eval split for config.yaml.

Usage: python tools/make_split.py --dataset-dir data/dataset --seed 42
"""
import json
import random
from collections import defaultdict
from pathlib import Path


def load_metadata(dataset_dir: Path) -> list[dict]:
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


def main() -> None:
    path = Path("/Users/roshanpandey/Python Projects/ci-triage-dataset/dataset")  # temporary; argparse replaces this
    examples = load_metadata(path)
    candidates = propose_exemplar_candidates(examples)
    exemplar_ids = {
        "infra_006", "infra_008", "infra_002",
        "genuine_regression_015", "genuine_regression_013", "genuine_regression_017",
        "transient_014", "transient_011", "transient_010",
        "flaky_test_006", "flaky_test_009", "flaky_test_013",
    }
    final_check_ids = pick_final_check(examples, exemplar_ids, seed=42,
                                       quotas={"genuine_regression": 3, "flaky_test": 3, "transient": 2, "infra": 2})
    for label, items in candidates.items():  # present
        print(f"\n=== {label} ===")
        for r in items:
            print(f"{r['id']:<28} {r['log_source']:<10} {r['log_lines']:>4}  {r['injected_fault'][:80]}")

    print(f"The final_check_ids are {final_check_ids}, and the size is {len(final_check_ids)}")


if __name__ == "__main__":
    main()
