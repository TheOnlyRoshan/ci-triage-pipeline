import os


def get_secret(env_var_name: str) -> str:
    try:
        return os.environ[env_var_name]
    except KeyError as e:
        raise RuntimeError(
            f"Environment variable {env_var_name} not set. "
            f"Create a fine-grained PAT with Actions:Read scope and export it, "
            f"or add it to your .env file."
        ) from e