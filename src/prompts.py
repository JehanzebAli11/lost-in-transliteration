"""Load prompt templates from prompts/ (string.Template files, ``$name`` slots)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from string import Template

import yaml

from .common import REPO_ROOT

PROMPTS_DIR = REPO_ROOT / "prompts"


@lru_cache(maxsize=None)
def _load(rel: str) -> Template:
    return Template((PROMPTS_DIR / rel).read_text(encoding="utf-8"))


def render(rel: str, **values) -> str:
    """Fill a prompt; every ``$slot`` must be provided (fails loudly otherwise)."""
    return _load(rel).substitute(**{k: str(v) for k, v in values.items()})


@lru_cache(maxsize=None)
def load_yaml(rel: str) -> dict:
    with open(PROMPTS_DIR / rel, encoding="utf-8") as f:
        return yaml.safe_load(f)


def variant_description(variant: str) -> str:
    return load_yaml("translation/variants.yaml")[variant].strip()


def literals_text(literals: list[str]) -> str:
    lits = [l for l in literals if l]
    return ", ".join(f'"{l}"' for l in lits) if lits else "(none)"


def prompt_path(rel: str) -> Path:
    return PROMPTS_DIR / rel
