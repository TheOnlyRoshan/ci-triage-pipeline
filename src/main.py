"""Command-line entry point: triage a single CI job.

Takes a job ID as an argument rather than reading it from config, since the
job under triage is per-run data, not configuration. In Phase 2 the trigger
supplies the same value automatically.

Run from the project root:
    python -m src.main <job_id> [--example_id ID] [--no_store]
"""

import argparse

from dotenv import load_dotenv

from src.auth import get_secret
from src.config_loader import find_project_root, load_config
from src.github.job_fetcher import get_failed_step
from src.github.log_fetcher import fetch_job_log, extract_step_window
from src.labeler.label_store import get_or_create
from src.labeler.llm_labeler import LabelResult, label_log
from src.labeler.log_preprocessor import preprocess_log


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Namespace with job_id (str), example_id (str | None) and
        no_store (bool).
    """
    parser = argparse.ArgumentParser(description="Process job ID via CLI")
    parser.add_argument("job_id", type=str, help="The unique identifier for the target job.")
    parser.add_argument("--example_id", type=str, default=None, help="The unique identifier for the target example.")
    parser.add_argument("--no_store", action="store_true", help="Do not store the label.")
    return parser.parse_args()

def run(job_id: str, example_id: str | None, use_store: bool, config) -> LabelResult:
    """Fetch, preprocess and classify a single CI job.

    Args:
        job_id: GitHub Actions job ID to triage.
        example_id: Key to store the result under. Defaults to
            f'job_{job_id}' when None, so ad-hoc runs are still cached and
            stay distinguishable from dataset example IDs.
        use_store: When False, calls the model directly and does not read or
            write the store. Used when iterating on the prompt, where a cache
            hit would mask the change being tested.
        config: Loaded PipelineConfig.

    Returns:
        The classification for this job.

    Raises:
        ValueError: If the job has no single failed step, or the model
            response cannot be parsed.
    """
    token = get_secret(config.github.token_env_var)
    step = get_failed_step(job_id, config.github.owner, config.github.repo, token)
    log = fetch_job_log(job_id, config.github.owner, config.github.repo, token)
    preprocessed = preprocess_log(extract_step_window(log, step), config)

    if use_store:
        return get_or_create(example_id or f"job_{job_id}", preprocessed, config)
    else:
        return label_log(preprocessed, config)

def main() -> None:
    """CLI entry point: parse arguments, run the pipeline, print the result."""
    load_dotenv()
    config = load_config(find_project_root() / 'config.yaml')
    args = parse_args()
    result = run(args.job_id, args.example_id, not args.no_store, config)
    print(f"Label:      {result.label}")
    print(f"Confidence: {result.confidence}")
    print(f"Rationale:  {result.rationale}")

if __name__ == "__main__":
    main()

