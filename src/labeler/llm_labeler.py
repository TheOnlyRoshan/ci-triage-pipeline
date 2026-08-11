"""Classify a preprocessed CI log by calling the Anthropic API.

The first module in the pipeline that is non-deterministic and costs money.
Model, temperature and max_tokens all come from config so a run is
reproducible and so the model can be varied for the three-model comparison
(EXPERIMENTS.md E2).
"""
import json
from typing import Literal

import anthropic

from pydantic import BaseModel, Field, ValidationError

from src.auth import get_secret
from src.labeler.prompt_builder import build_prompt


class LabelResult(BaseModel):
    """One validated classification of a CI failure.

    The Literal constraint on `label` is the point of this model: it closes the
    category set at parse time, so a response of "flaky" or an invented
    category like "timeout" fails validation instead of flowing into the
    confusion matrix as a value that silently matches nothing.

    Attributes:
        label: One of the four categories.
        confidence: Model's self-reported confidence, 0.0 to 1.0.
        rationale: Short justification citing evidence from the log. Kept for
            debugging — when a class is systematically misread, the rationales
            are what explain why.
    """
    label: Literal['flaky_test', 'genuine_regression', 'infra', 'transient']
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str


def call_llm(prompt: str, config) -> str:
    """Send one prompt to the Anthropic Messages API and return the raw text.

    Deliberately does no parsing — the fetch/parse split mirrors log_fetcher,
    so parse_label_response can be tested against fixture strings with no
    network access. No retry logic either: an API error should surface rather
    than be silently absorbed.

    Args:
        prompt: Complete prompt from build_prompt.
        config: Loaded PipelineConfig; reads config.llm.

    Returns:
        Text of the first content block of the response.

    Raises:
        RuntimeError: If the API key environment variable is unset.
        anthropic.APIError: On an API-level failure.
    """
    llm_api_token = get_secret(config.llm.api_key_env_var)
    client = anthropic.Anthropic(api_key=llm_api_token)
    response = client.messages.create(model=config.llm.model, max_tokens=config.llm.max_tokens,
                                      temperature=config.llm.temperature,
                                      messages=[{'role': 'user', 'content': prompt}])
    return response.content[0].text


def parse_label_response(raw_response: str) -> LabelResult:
    """Parse and validate a model response into a LabelResult.

    Slices from the first '{' to the last '}' before parsing. That single move
    absorbs both markdown fences and any preamble sentence, since a ```json
    fence sits before the first brace and its closing fence after the last.
    Nested objects are unaffected — they are between those bounds by
    definition.

    No retry and no fallback label. Re-asking until the response parses would
    discard exactly the cases where the model was genuinely confused, which is
    the signal the confusion matrix exists to show; defaulting to a category
    would fabricate data outright. A parse failure is a result, not an error to
    paper over.

    Args:
        raw_response: Unmodified text from call_llm.

    Returns:
        A validated LabelResult.

    Raises:
        ValueError: If no JSON object is present, if the slice is not valid
            JSON, or if the fields fail validation. The raw response is
            embedded in the message — knowing the model rambled, truncated, or
            invented a category is the difference between a one-look fix and a
            blind one.
    """
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
    """Classify a preprocessed log: build prompt, call model, parse response.

    The module's public entry point. Expects log_text to be preprocessed
    already; running preprocessing here as well would give two modules
    authority over log shaping.

    Args:
        log_text: Preprocessed log, from preprocess_log.
        config: Loaded PipelineConfig.

    Returns:
        A validated LabelResult.

    Raises:
        ValueError: If the response cannot be parsed or validated.
    """
    prompt = build_prompt(log_text, config)
    raw = call_llm(prompt, config)
    return parse_label_response(raw)
