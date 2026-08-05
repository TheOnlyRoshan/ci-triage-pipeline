import re


def strip_noise_lines(raw_text: str, config) -> str:
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
    clean_text = strip_noise_lines(raw_text, config)
    return extract_relevant_window(clean_text, config)
