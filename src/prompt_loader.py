"""Prompt template loader: reads prompts from external YAML files."""
import yaml
from pathlib import Path

import config

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

_cache: dict[str, dict] = {}


def _load(template_name: str) -> dict:
    """Load and cache a YAML prompt template file."""
    if template_name in _cache:
        return _cache[template_name]

    path = _PROMPTS_DIR / f"{template_name}.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"Prompt template '{template_name}' not found at {path}. "
            f"Available templates: {[p.stem for p in _PROMPTS_DIR.glob('*.yaml')]}"
        )

    with open(path) as f:
        data = yaml.safe_load(f)

    _cache[template_name] = data
    return data


def get(key: str) -> str:
    """Get a prompt string by key from the active template.

    Uses the PROMPT_TEMPLATE setting from config (default: "default").

    Args:
        key: The prompt key (e.g. "system_prompt", "judge_prompt").

    Returns:
        The prompt template string.
    """
    template_name = getattr(config, "PROMPT_TEMPLATE", "default")
    data = _load(template_name)
    if key not in data:
        raise KeyError(
            f"Prompt key '{key}' not found in template '{template_name}'. "
            f"Available keys: {list(data.keys())}"
        )
    return data[key]


def reload():
    """Clear the prompt cache so templates are re-read on next access."""
    _cache.clear()
