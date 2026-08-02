import os


def get_github_token(config: dict) -> str:
    env_var_name = config.github.token_env_var
    try:
        return os.environ[env_var_name]
    except KeyError as e:
        raise RuntimeError(
            f"Environment variable {env_var_name} not set. "
            f"Create a fine-grained PAT with Actions:Read scope and export it, "
            f"or add it to your .env file."
        ) from e
