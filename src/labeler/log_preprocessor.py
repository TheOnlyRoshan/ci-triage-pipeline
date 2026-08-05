import re

from dotenv import load_dotenv

from src.config_loader import load_config, find_project_root, PipelineConfig
from src.github.auth import get_github_token
from src.github.job_fetcher import get_failed_step
from src.github.log_fetcher import fetch_job_log, extract_step_window


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


load_dotenv()
config = load_config(find_project_root() / 'config.yaml')
token = get_github_token(config)
log = fetch_job_log(config.github.job_id, config.github.owner, config.github.repo, token)
step = get_failed_step(config.github.job_id, config.github.owner, config.github.repo, token)
#stripped_noise_lines = strip_noise_lines(extract_step_window(log, step), config)
#extracted_relevant_window = extract_relevant_window(stripped_noise_lines, config)
#print('Log after extracting relevant window:', extracted_relevant_window)
stripped_noise_lines = strip_noise_lines(extract_step_window(log, step), config)
extracted_relevant_window = extract_relevant_window(stripped_noise_lines, config)
print('Log after extracting relevant window:', extracted_relevant_window)
result = preprocess_log(extract_step_window(log, step), config)
print('Preprocessed log:', result)
