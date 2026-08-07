"""Verifies the committed exemplar / final-check split in config.yaml.

Regenerates the final-check selection from the committed seed and asserts
it matches the committed split (reproducibility check).

Usage: python make_split.py [--config config.yaml]
"""
import argparse
import random
from collections import Counter, defaultdict
from pathlib import Path

from src.config_loader import load_config
from src.dataset.loader import load_metadata


def normalize(s: str) -> str:
    """Normalize a string for comparison: lowercase, collapse all whitespace to single spaces.

        Args:
            s: Raw string (e.g., an injected_fault description).

        Returns:
            Normalized string; differences in case and spacing are erased.
        """
    return " ".join(s.lower().split())

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
            seed: Base RNG seed; each class samples from its own RNG seeded
                with "seed:label", so results are independent of class order.
            quotas: Mapping of label -> number of examples to select.

        Returns:
            Sorted list of selected final-check IDs.

        Raises:
            ValueError: If a class has fewer eligible examples than its quota.
        """
    pool: defaultdict[str, list[str]] = defaultdict(list)
    final_check_ids: list[str] = []
    for example in examples:
        if example['id'] not in exemplar_ids:
            pool[example['label']].append(example['id'])
    for label, k in quotas.items():
        rng = random.Random(f"{seed}:{label}")
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
    """Verify the committed split against the dataset and the seed.

    Three checks: every committed final-check ID exists in the dataset; no
    exemplar shares an injected_fault with an eval example; and the final-check
    selection regenerates exactly from the committed seed. Prints the split as
    YAML for comparison against config.yaml.

    Raises:
        ValueError: If any check fails.
    """
    parser = argparse.ArgumentParser(description="Verify the committed split in config.yaml")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    args = parser.parse_args()

    config = load_config(args.config)
    seed = config.split.selection_seed
    exemplars = config.split.exemplars
    committed_final_check = sorted(config.split.final_check)

    examples = load_metadata(config.dataset.local_dir)
    label_by_id = {r['id']: r['label'] for r in examples}

    missing = [i for i in committed_final_check if i not in label_by_id]
    if missing:
        raise ValueError(f"final_check IDs not found in dataset: {missing}")

    exemplar_ids = {i for ids in exemplars.values() for i in ids}
    quotas = dict(Counter(label_by_id[i] for i in committed_final_check))

    final_check_ids = pick_final_check(examples, exemplar_ids, seed=seed, quotas=quotas)

    leaked_items = check_leakage(examples, exemplar_ids)
    if leaked_items:
        raise ValueError(f"Semantic leakage found: {leaked_items}")
    print("No semantic leakage found")

    if final_check_ids != committed_final_check:
        raise ValueError(
            f"Split NOT reproducible from seed {seed}.\n"
            f"  regenerated: {final_check_ids}\n"
            f"  committed:   {committed_final_check}")
    print(f"Split verified reproducible from seed {seed} "
          f"({len(final_check_ids)} final-check IDs match config.yaml)")

    print("\n--- committed split as YAML (should match config.yaml) ---")
    print(emit_yaml(exemplars, final_check_ids, seed=seed))
    print("--- end ---")


if __name__ == "__main__":
    main()
