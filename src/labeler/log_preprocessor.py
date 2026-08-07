"""Reduce a raw step log to bounded, LLM-ready text.

Two stages, both pure string transformations driven entirely by config:
noise stripping (remove content with no classification signal) and windowing
(cap the line count). No network access and no API calls, so both are testable
against string fixtures alone.
"""
import re


def strip_noise_lines(raw_text: str, config) -> str:
    """Remove log lines and line fragments that carry no classification signal.

    Structural patterns (timestamps, ANSI escapes, '##[group]' markers) always
    apply — they are provably signal-free. Env-block and pip-output patterns
    are content judgements and stay behind config flags, because they are only
    safe for some failure classes: the runner's pythonLocation and
    LD_LIBRARY_PATH are exactly the evidence that distinguishes an 'infra'
    failure. Both default to false; the ablation is EXPERIMENTS.md E1.

    Every pattern must be anchored '^...*$' so a whole line collapses to empty.
    re.sub deletes only the matched substring, so an unanchored pattern like
    '^\\s*pythonLocation:' leaves ' /opt/hostedtoolcache/...' orphaned on the
    line, which then survives the blank-line filter.

    The blank-line check runs AFTER substitution and must stay there: a
    timestamp-only line is not blank until its timestamp has been stripped.

    Args:
        raw_text: Step log text, typically from extract_step_window.
        config: Loaded PipelineConfig; reads config.preprocessing.

    Returns:
        Surviving lines joined by newlines.
    """
    cleaned_lines = []
    patterns = list(config.preprocessing.structural_patterns)
    if config.preprocessing.strip_env_block:
        patterns += config.preprocessing.env_block_patterns
    if config.preprocessing.strip_pip_output:
        patterns += config.preprocessing.pip_output_patterns
    for line in raw_text.splitlines():
        for pattern in patterns:
            line = re.sub(pattern, "", line)
        if not line.strip():
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def extract_relevant_window(clean_text: str, config) -> str:
    """Cap a log at head_lines + tail_lines, marking what was dropped.

    Both ends are kept because signal location varies by failure class: an
    'infra' failure dies during environment setup (signal at the head), a test
    failure ends in an assertion and summary (signal at the tail).

    Logs that already fit are returned untouched. That guard is not just an
    optimisation — without it, a log shorter than head + tail would emit
    lines[:head] plus lines[-tail:], and negative-index clamping would repeat
    content silently.

    The truncation marker matters to the model, not just the reader: without
    it the two windows read as contiguous and invite invented causal links
    between unrelated events.

    Args:
        clean_text: Noise-stripped log text.
        config: Loaded PipelineConfig; reads head_lines, tail_lines and
            truncation_marker from config.preprocessing.

    Returns:
        Either clean_text unchanged, or head + marker + tail joined by
        newlines.
    """
    cleaned_lines = clean_text.splitlines()
    number_of_head_lines = config.preprocessing.head_lines
    number_of_tail_lines = config.preprocessing.tail_lines
    if len(cleaned_lines) <= number_of_head_lines + number_of_tail_lines:
        return clean_text
    header_lines = cleaned_lines[:number_of_head_lines]
    skipped_lines = len(cleaned_lines) - number_of_head_lines - number_of_tail_lines
    marker = config.preprocessing.truncation_marker.format(n=skipped_lines)
    tail_lines = cleaned_lines[-number_of_tail_lines:]
    return "\n".join(header_lines + [marker] + tail_lines)


def preprocess_log(raw_text: str, config) -> str:
    """Run the full preprocessing chain: strip noise, then window.

    This is the module's public entry point. Callers use it instead of the two
    stages directly so the order is owned in one place — and so a future third
    stage reaches every caller at once, rather than only the ones that
    remembered to add it. Divergence here between eval and production code
    would be train/serve skew.

    Args:
        raw_text: Step log text, typically from extract_step_window.
        config: Loaded PipelineConfig.

    Returns:
        Preprocessed log text, ready to substitute into a prompt.
    """
    clean_text = strip_noise_lines(raw_text, config)
    return extract_relevant_window(clean_text, config)
