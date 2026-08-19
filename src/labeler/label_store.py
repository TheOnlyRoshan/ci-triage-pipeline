import json
from datetime import datetime, timezone
from pathlib import Path

from src.config_loader import find_project_root
from src.labeler.llm_labeler import LabelResult, label_log


def _store_path(config) -> Path:
    """Resolve the label store path relative to the project root.

    Args:
        config: Loaded PipelineConfig.

    Returns:
        Absolute path to the JSONL store.
    """
    return find_project_root() / config.label_store.path


def build_key(example_id: str, config) -> dict:
    """Build the four-field identity of a labelling request.

    A stored label is reusable only when all four fields match. Changing the
    prompt, the model, or the preprocessing flags means a different question
    was asked, so a stale answer must not be returned for it.

    Args:
        example_id: Dataset example ID, e.g. 'flaky_test_006'.
        config: Loaded PipelineConfig; reads prompt.version, llm.model and
            preprocessing.variant.

    Returns:
        Dict with keys example_id, prompt_version, model,
        preprocessing_variant.
    """
    return dict(
        example_id=example_id,
        prompt_version=config.prompt.version,
        model=config.llm.model,
        preprocessing_variant=config.preprocessing.variant,
    )


def find_label(example_id: str, config) -> LabelResult | None:
    """Look up a stored label matching the full four-field key.

    All of example_id, prompt_version, model and preprocessing_variant must
    match. A partial match is not a hit: changing any one of them means the
    question asked of the model was different, so a stored answer from the
    other configuration does not apply.

    Args:
        example_id: Dataset example ID, e.g. 'flaky_test_006'.
        config: Loaded PipelineConfig; supplies the other three key fields.

    Returns:
        The stored LabelResult, or None if the store has no matching record
        (including when the store file does not exist yet).
        When several rows match, the most recent is returned, since the store is append-only.
    """
    path = _store_path(config)
    if not path.exists():
        return None
    key = build_key(example_id, config)
    matched_rows = []

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parsed_line = json.loads(line)
            if all(parsed_line[k] == v for k, v in key.items()):
                matched_rows.append(parsed_line)
    if not matched_rows:
        return None
    return LabelResult(**matched_rows[-1])


def append_label(example_id: str, result: LabelResult, config) -> None:
    """Append one labelling result to the store.

    Opens in append mode so existing rows are never rewritten; the store is a
    log of every label ever produced, not a table of current values. Creates
    the parent directory on first write so a fresh clone works without manual
    setup.

    Args:
        example_id: Dataset example ID.
        result: The validated classification to store.
        config: Loaded PipelineConfig.
    """
    path_to_file = _store_path(config)
    path_to_file.parent.mkdir(parents=True, exist_ok=True)
    iso_format_z = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
    record = {
        **build_key(example_id, config),
        **result.model_dump(),
        "created_at": iso_format_z,
    }
    with open(path_to_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record) + "\n")


def get_or_create(example_id: str, log_text: str, config) -> LabelResult:
    """Return a stored label if one exists, otherwise label and store.

    The module's public entry point. Callers never call the labeler directly,
    so the check-miss-call-save sequence lives in exactly one place and cannot
    be half-applied by a caller who forgets the lookup.

    Nothing is stored when labelling fails. A parse error is not an answer,
    and caching it would make the failure permanent until the store is edited
    by hand.

    Args:
        example_id: Dataset example ID.
        log_text: Preprocessed log. Only used on a cache miss.
        config: Loaded PipelineConfig.

    Returns:
        The stored or freshly generated LabelResult.

    Raises:
        ValueError: If the model response cannot be parsed or validated.
    """
    existing = find_label(example_id, config)
    if existing is not None:
        return existing
    result = label_log(log_text, config)
    append_label(example_id, result, config)
    return result
