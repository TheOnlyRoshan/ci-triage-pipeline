"""Secret lookup from the process environment.

Secrets are never stored in config.yaml; config holds only the *name* of the
environment variable, and this module resolves the name to its value. Callers
pass their own variable name (e.g. config.github.token_env_var), so this module
stays agnostic about which service a secret belongs to.
"""
import os


def get_secret(env_var_name: str) -> str:
    """Read a secret from an environment variable.

    Args:
        env_var_name: Name of the environment variable to read, typically
            sourced from config (e.g. 'GITHUB_TOKEN', 'ANTHROPIC_API_KEY').

    Returns:
        The variable's value.

    Raises:
        RuntimeError: If the variable is not set. Fails loudly rather than
            returning None, so a missing secret surfaces at startup instead of
            as an opaque 401 later.
    """
    try:
        return os.environ[env_var_name]
    except KeyError as e:
        raise RuntimeError(
            f"Environment variable {env_var_name} not set. "
            f"Create a fine-grained PAT with Actions:Read scope and export it, "
            f"or add it to your .env file."
        ) from e
