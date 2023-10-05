"""Utility functions"""
import os
from typing import Any, Dict, Optional, Set


def env_var_is_set(env_var: str) -> bool:
    """Check if an environment variable is set.

    Args:
        env_var (str): The name of the environment variable.

    Returns:
        bool: True if the environment variable is set, False otherwise.
    """
    return env_var in os.environ and os.environ[env_var] not in (
        "",
        "0",
        "false",
        "False",
    )


def get_from_dict_or_env(
    data: Dict[str, Any], key: str, env_key: str, default: Optional[str] = None
) -> str:
    """Get a value from a dictionary or an environment variable."""
    if key in data and data[key]:
        return data[key]
    else:
        return get_from_env(key, env_key, default=default)


def get_from_env(key: str, env_key: str, default: Optional[str] = None) -> str:
    """Get a value from a dictionary or an environment variable."""
    if env_key in os.environ and os.environ[env_key]:
        return os.environ[env_key]
    elif default is not None:
        return default
    else:
        raise ValueError(
            f"Did not find {key}, please add an environment variable"
            f" `{env_key}` which contains it, or pass"
            f"  `{key}` as a named parameter."
        )


def get_pydantic_field_names(pydantic_cls: Any) -> Set:
    """Get field names, including aliases, for a pydantic class.

    Args:
        pydantic_cls: Pydantic class."""
    all_required_field_names = set()
    for field in pydantic_cls.__fields__.values():
        all_required_field_names.add(field.name)
        if field.has_alias:
            all_required_field_names.add(field.alias)
    return all_required_field_names
