"""Assemble the classification prompt from a versioned template.

Prompt text lives in prompts/<version>.txt rather than in code, so versions can
be diffed and referenced from EXPERIMENTS.md. The version string is also part
of the label store's cache key: editing the prompt invalidates labels produced
by the previous version instead of silently mixing them.
"""
from src.config_loader import find_project_root, load_config


def load_prompt_template(config) -> str:
    """Read the prompt template named by config.prompt.version.

    Not cached. Re-reading a small file per call costs nothing and means an
    edit to the template takes effect on the next run rather than requiring a
    restart.

    Args:
        config: Loaded PipelineConfig; reads prompt.prompts_dir and
            prompt.version.

    Returns:
        The raw template text, with its {log} placeholder unfilled.

    Raises:
        FileNotFoundError: If the file is absent, naming the resolved path — a
            typo in prompt.version should fail loudly, not yield an empty
            prompt.
    """
    prompt_file = find_project_root() / config.prompt.prompts_dir / f"{config.prompt.version}.txt"
    try:
        return prompt_file.read_text(encoding="utf-8")
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Prompt template not found: {prompt_file}. "
            f"Check prompt.version in config.yaml."
        ) from e


def build_prompt(log_text: str, config) -> str:
    """Substitute a preprocessed log into the prompt template.

    log_text is inserted verbatim. Nothing is stripped or truncated here —
    log shaping belongs to log_preprocessor, and splitting that responsibility
    across two modules would make any change in output impossible to attribute.

    The template escapes its literal JSON braces as '{{' / '}}' so .format()
    leaves them intact. Braces inside log_text are safe: .format() parses only
    the template, never the substituted values. Do not call .format() again on
    the result — the log's own braces (e.g. 'bash -e {0}') would then be read
    as placeholders.

    Nothing identifying the example — ground truth, example ID, filename,
    folder — is included. Only the log text.

    Args:
        log_text: Preprocessed log, from preprocess_log.
        config: Loaded PipelineConfig.

    Returns:
        The complete prompt to send to the model.
    """
    template = load_prompt_template(config)
    return template.format(log=log_text)
