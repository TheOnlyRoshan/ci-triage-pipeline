from pydantic import BaseModel, HttpUrl, model_validator
from pathlib import Path

import yaml


def find_project_root(marker: str = "config.yaml") -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / marker).exists():
            return parent
    raise FileNotFoundError(f"Could not find {marker} in any parent directory")

class GithubConfig(BaseModel):
    token_env_var: str
    owner: str
    repo: str
    job_id: str


class DatasetConfig(BaseModel):
    repo: HttpUrl
    pinned_sha: str
    local_dir: Path


class SplitConfig(BaseModel):
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


class PipelineConfig(BaseModel):
    dataset: DatasetConfig
    split: SplitConfig
    github: GithubConfig


def load_config(file_path: Path) -> PipelineConfig:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            raw_data = yaml.safe_load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Configuration file not found at: {file_path.resolve()}") from e
    if raw_data is None:
        raise ValueError(f"Config file is empty: {file_path}")
    return PipelineConfig(**raw_data)