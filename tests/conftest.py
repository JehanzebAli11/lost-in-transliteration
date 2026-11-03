"""Shared pytest fixtures: a tiny test config whose outputs all live in tmp_path."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common import load_config  # noqa: E402


def tiny_overrides(tmp: Path) -> dict:
    return {
        "_root": str(tmp),
        "paths": {"data_dir": "data", "results_dir": "results", "figures_dir": "figures",
                  "models_dir": "models", "cache_dir": "cache", "sandbox_dir": "sandbox",
                  "log_file": "LOG.md"},
        "models": {
            "generator": {"model": "fake/gemma-4-31b-it"},
            "translator": {"model": "fake/gemma-4-31b-it", "max_tokens": 512},
            "native_writer": {"model": "fake/qwen-instruct"},
            "native_labeler": {"model": "fake/gemma-4-31b-it"},
        },
        "dataset": {"n_templates_per_style": 3, "n_goal_phrasings": 2, "n_attacks": 40,
                    "n_carriers": 6, "n_benign": 24, "benign_batch": 6},
        "translation": {"review_per_variant": 4, "concurrency": 2, "backtranslate_fraction": 0.2},
        "native": {"n_attacks": 9, "n_benign": 9, "batch": 4},
        "eval_models": {
            "max_attacks_per_lang": 12, "clean_controls_per_lang": 3,
            "cross_condition": {"enabled": True, "n_per_lang": 3}, "concurrency": 2,
            "models": [
                {"name": "claude-haiku", "endpoint": "anthropic", "model": "fake-haiku",
                 "input_usd_per_mtok": 1.0, "output_usd_per_mtok": 5.0},
                {"name": "openrouter-small", "endpoint": "openrouter", "model": "fake/small",
                 "input_usd_per_mtok": 0.1, "output_usd_per_mtok": 0.4},
                {"name": "gemma-local", "endpoint": "local", "model": "fake/gemma-4-31b-it"},
            ],
        },
        "classifiers": {"models": [
            {"name": "clf-a", "hf_id": "fake/clf-a", "positive_labels": ["INJECTION"],
             "threshold": 0.5, "max_length": 128, "gated": False},
            {"name": "clf-b", "hf_id": "fake/clf-b", "positive_labels": ["LABEL_1"],
             "threshold": 0.5, "max_length": 128, "gated": False},
        ]},
        "analysis": {"bootstrap": 50, "qualitative_per_lang": 2},
    }


@pytest.fixture
def cfg(tmp_path):
    return load_config(ROOT / "config.yaml", profile=None, overrides=tiny_overrides(tmp_path))
