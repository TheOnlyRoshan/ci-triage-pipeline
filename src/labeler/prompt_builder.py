from src.config_loader import find_project_root


def load_prompt_template(config) -> str:
    prompt_file = find_project_root() / config.prompt.prompts_dir / f"{config.prompt.version}.txt"
    try:
        return prompt_file.read_text(encoding="utf-8")
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Prompt template not found: {prompt_file}. "
            f"Check prompt.version in config.yaml."
        ) from e


def build_prompt(log_text: str, config) -> str:
    template = load_prompt_template(config)
    return template.format(log=log_text)
