"""Load and validate config.yaml into typed Pydantic models.

Every tunable in the pipeline lives in config.yaml; nothing is hardcoded.
Validation happens at load time so a bad regex, an out-of-range temperature or 
an overlapping split fails immediately, rather than midway through a run that 
has already spent API calls.
"""
import re
from pathlib import Path

import yaml
from pydantic import BaseModel, HttpUrl, model_validator, Field, field_validator
from src.categories import CATEGORIES


def find_project_root(marker: str = "config.yaml") -> Path:
    """Walk up from this file until a directory containing `marker` is found.

    Lets every other path in the project be written relative to the repo root,
    so the pipeline runs the same regardless of the current working directory.

    Args:
        marker: Filename that identifies the project root.

    Returns:
        The directory containing `marker`, which is the root itself and not
        the file.

    Raises:
        FileNotFoundError: If no parent directory contains `marker`.
    """
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / marker).exists():
            return parent
    raise FileNotFoundError(f"Could not find {marker} in any parent directory")


class GithubConfig(BaseModel):
    """GitHub repository and API access settings.

    Attributes:
        token_env_var: Name of the environment variable holding the token.
            The name only; the value is resolved at runtime by auth.get_secret.
        owner: Repository owner (user or org).
        repo: Repository name.
    """
    token_env_var: str
    owner: str
    repo: str


class DatasetConfig(BaseModel):
    """Location and pinned version of the labelled dataset.

    Attributes:
        repo: URL of the dataset repository.
        pinned_sha: Commit the dataset is pinned to, so results stay tied to a
            known dataset state.
        local_dir: Dataset path relative to the project root.
    """
    repo: HttpUrl
    pinned_sha: str
    local_dir: Path


class SplitConfig(BaseModel):
    """The committed exemplar / final-check split.

    Only two of the three splits are listed. Dev-eval is derived downstream as
    the complement, which is why it cannot drift out of sync with the others.

    Attributes:
        selection_seed: Seed the final-check selection was drawn with, kept so
            the split can be regenerated and verified.
        exemplars: Label -> exemplar IDs held out as few-shot candidates.
        final_check: IDs reserved as a touch-once holdout, never used for
            tuning.
    """
    selection_seed: int
    exemplars: dict[str, list[str]]
    final_check: list[str]

    @model_validator(mode="after")
    def check_no_leakage(self):
        """Reject configs where an ID appears in both exemplars and final_check.

        Dev-eval is derived as "everything else" downstream, so it cannot
        collide by construction; only this boundary needs guarding.
        """
        exemplar_ids = {ex_id for ids in self.exemplars.values() for ex_id in ids}
        overlap = exemplar_ids & set(self.final_check)
        if overlap:
            raise ValueError(
                f"Split leakage: IDs present in both exemplars and final_check: {sorted(overlap)}"
            )
        return self


class PreprocessConfig(BaseModel):
    """Log preprocessing patterns, flags and window sizes.

    Attributes:
        structural_patterns: Always applied. Timestamps, ANSI escapes, and
            group markers.
        strip_env_block: Whether to remove the runner env block.
        env_block_patterns: Patterns used when strip_env_block is true.
        strip_pip_output: Whether to remove pip install output.
        pip_output_patterns: Patterns used when strip_pip_output is true. These
            enumerate known pip line shapes rather than matching structurally,
            so some output survives when the flag is on.
        variant: Human-readable name for the current flag combination, written
            into every stored label so runs stay distinguishable. Nothing
            enforces that it matches the flags, so set it by hand when
            toggling.
        head_lines: Lines kept from the start of a log.
        tail_lines: Lines kept from the end.
        truncation_marker: Template for the elision marker; '{n}' is filled
            with the number of lines dropped.
    """
    structural_patterns: list[str]
    strip_env_block: bool
    env_block_patterns: list[str]
    strip_pip_output: bool
    pip_output_patterns: list[str]
    variant: str
    head_lines: int = Field(gt=0)
    tail_lines: int = Field(gt=0)
    truncation_marker: str

    @field_validator('structural_patterns', 'env_block_patterns', 'pip_output_patterns')
    @classmethod
    def patterns_must_compile(cls, patterns: list[str]) -> list[str]:
        """Reject any pattern that is not a valid regex.

        A malformed pattern would otherwise raise mid-run, after work has
        already been done. Note this checks only that a pattern compiles, not
        that it is anchored correctly. An unanchored pattern is valid regex
        and will still leave residue on the line.

        Args:
            patterns: Regex strings from one of the pattern lists.

        Returns:
            The list unchanged.

        Raises:
            ValueError: If any pattern fails to compile.
        """
        for p in patterns:
            try:
                re.compile(p)
            except re.error as e:
                raise ValueError(f"Invalid regex {p!r}: {e}") from e
        return patterns


class PromptConfig(BaseModel):
    """Which prompt template to use.

    Attributes:
        version: Template filename stem, e.g. 'v1' for prompts/v1.txt. Also
            recorded with every label, so changing it invalidates labels
            generated under the previous version.
        prompts_dir: Directory holding templates, relative to project root.
    """
    version: str
    prompts_dir: str


class LlmConfig(BaseModel):
    """Model and sampling settings for the labeler.

    Attributes:
        model: API model identifier. Recorded with every label so results from
            different models are never conflated.
        max_tokens: Response cap.
        temperature: Sampling temperature; 0.0 for reproducibility.
        api_key_env_var: Name of the environment variable holding the API key.
    """
    model: str
    max_tokens: int = Field(gt=0)
    temperature: float = Field(ge=0.0, le=1.0)
    api_key_env_var: str


class LabelStoreConfig(BaseModel):
    """Where predicted labels are persisted.

    Attributes:
        path: JSONL file path relative to the project root.
    """
    path: Path


class PipelineConfig(BaseModel):
    """Root config object: the fully validated contents of config.yaml."""
    categories: list[str]
    dataset: DatasetConfig
    split: SplitConfig
    github: GithubConfig
    preprocessing: PreprocessConfig
    prompt: PromptConfig
    llm: LlmConfig
    label_store: LabelStoreConfig

    @model_validator(mode='after')
    def categories_match_code(self) -> 'PipelineConfig':
        """Reject a config whose categories disagree with src.categories.

        The Literal in src.categories cannot read config, so the two are
        independent declarations of the same set. Comparing them at load time
        turns a silent mismatch into an immediate startup error. Without it,
        the mismatch would surface as a Pydantic validation failure deep
        inside a labelling run, after API calls have been paid for.

        Returns:
            The validated config.

        Raises:
            ValueError: If the two sets differ.
        """
        if set(self.categories) != set(CATEGORIES):
            raise ValueError(
                f"config categories {sorted(self.categories)} do not match "
                f"src.categories {sorted(CATEGORIES)}")
        return self


def load_config(file_path: Path) -> PipelineConfig:
    """Parse and validate config.yaml.

    Args:
        file_path: Path to the config file.

    Returns:
        A validated PipelineConfig.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is empty.
        pydantic.ValidationError: If any field is missing or invalid.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            raw_data = yaml.safe_load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Configuration file not found at: {file_path.resolve()}") from e
    if raw_data is None:
        raise ValueError(f"Config file is empty: {file_path}")
    return PipelineConfig(**raw_data)
