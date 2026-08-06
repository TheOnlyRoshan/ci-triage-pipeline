import json
from typing import Literal

import anthropic
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

from src.auth import get_secret
from src.config_loader import load_config, find_project_root
from src.github.job_fetcher import get_failed_step
from src.github.log_fetcher import fetch_job_log, extract_step_window
from src.labeler.log_preprocessor import preprocess_log
from src.labeler.prompt_builder import build_prompt


class LabelResult(BaseModel):
    label: Literal['flaky_test', 'genuine_regression', 'infra', 'transient']
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str


def call_llm(prompt: str, config) -> str:
    llm_api_token = get_secret(config.llm.api_key_env_var)
    client = anthropic.Anthropic(api_key=llm_api_token)
    response = client.messages.create(model=config.llm.model, max_tokens=config.llm.max_tokens,
                                      temperature=config.llm.temperature,
                                      messages=[{'role': 'user', 'content': prompt}])
    return response.content[0].text


def parse_label_response(raw_response: str) -> LabelResult:
    text = raw_response.strip()

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(
            f"No JSON object found in response: {raw_response!r}"
        )

    json_text = text[start:end + 1]

    try:
        parsed = json.loads(json_text)
        return LabelResult(**parsed)
    except (json.JSONDecodeError, ValidationError) as e:
        raise ValueError(
            f"Could not parse label from response: {raw_response!r}"
        ) from e

def label_log(log_text: str, config) -> LabelResult:
    prompt = build_prompt(log_text, config)
    raw = call_llm(prompt, config)
    return parse_label_response(raw)

load_dotenv()
config = load_config(find_project_root() / 'config.yaml')
github_token = get_secret(config.github.token_env_var)
log = fetch_job_log(config.github.job_id, config.github.owner, config.github.repo, github_token)
step = get_failed_step(config.github.job_id, config.github.owner, config.github.repo, github_token)
result = preprocess_log(extract_step_window(log, step), config)
print('Label:', label_log(result, config))