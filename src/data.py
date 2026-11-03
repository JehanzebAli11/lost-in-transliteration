"""Dataset file layout, loaders and the carrier-embedding helper.

File layout (PLAN §2), relative to ``paths.data_dir``::

    templates/goal_payloads.json   templates/templates.jsonl
    carriers/{en,ur,roman_ur,mixed}.jsonl
    {en,ur,roman_ur,mixed}.jsonl            attacks (label 1)
    benign_{en,ur,roman_ur,mixed}.jsonl     benign controls (label 0)
    native.jsonl                            §3.6, test-only, attacks + benign
    splits.json  human_review_sample.csv  phrase_translations.json

Every item row shares these keys: id, source_id, lang, condition ("main"|"native"),
label (1 attack / 0 benign), split ("train"|"test"|"heldout_style"), text.
Attacks also carry: template_id, style, goal, payload_id, canary, secret, phrase,
literals, carrier_id, position.
"""
from __future__ import annotations

import re
from pathlib import Path

from .common import LANGS, Config, read_json, read_jsonl

SENT_SPLIT_RE = re.compile(r"(?<=[.!?۔؟])\s+")


def attacks_path(cfg: Config, lang: str) -> Path:
    return cfg.data_dir / f"{lang}.jsonl"


def benign_path(cfg: Config, lang: str) -> Path:
    return cfg.data_dir / f"benign_{lang}.jsonl"


def carriers_path(cfg: Config, lang: str) -> Path:
    return cfg.data_dir / "carriers" / f"{lang}.jsonl"


def native_path(cfg: Config) -> Path:
    return cfg.data_dir / "native.jsonl"


def splits_path(cfg: Config) -> Path:
    return cfg.data_dir / "splits.json"


def load_attacks(cfg: Config, langs=LANGS) -> list[dict]:
    rows = []
    for lang in langs:
        rows += read_jsonl(attacks_path(cfg, lang))
    return rows


def load_benign(cfg: Config, langs=LANGS) -> list[dict]:
    rows = []
    for lang in langs:
        rows += read_jsonl(benign_path(cfg, lang))
    return rows


def load_native(cfg: Config) -> list[dict]:
    return read_jsonl(native_path(cfg))


def load_all_items(cfg: Config, include_native: bool = True) -> list[dict]:
    items = load_attacks(cfg) + load_benign(cfg)
    if include_native:
        items += load_native(cfg)
    return items


def load_carriers(cfg: Config, lang: str) -> dict[str, dict]:
    return {c["carrier_id"]: c for c in read_jsonl(carriers_path(cfg, lang))}


def load_splits(cfg: Config) -> dict:
    return read_json(splits_path(cfg), default={}) or {}


def eval_group(item: dict) -> str:
    """Which reporting group an item belongs to (PLAN §5.4): main splits or native."""
    if item.get("condition") == "native":
        return "native"
    return item.get("split", "test")


def lang_condition(item: dict) -> str:
    """Language label used in tables: the variant, or 'native' for §3.6 items."""
    return "native" if item.get("condition") == "native" else item["lang"]


# ----------------------------------------------------------------------------- embedding
def split_sentences(text: str) -> list[str]:
    parts = [p for p in SENT_SPLIT_RE.split(text.strip()) if p]
    return parts or [text]


def embed(carrier: str, injection: str, position: str) -> str:
    """Insert an injection into a carrier document at start / middle / end (§3.3)."""
    carrier = carrier.strip()
    injection = injection.strip()
    if position == "start":
        return f"{injection}\n\n{carrier}"
    if position == "end":
        return f"{carrier}\n\n{injection}"
    if position == "middle":
        sents = split_sentences(carrier)
        if len(sents) < 2:
            half = len(carrier) // 2
            cut = carrier.rfind(" ", 0, half)
            cut = half if cut == -1 else cut
            return f"{carrier[:cut].rstrip()}\n\n{injection}\n\n{carrier[cut:].lstrip()}"
        mid = len(sents) // 2
        return " ".join(sents[:mid]) + f"\n\n{injection}\n\n" + " ".join(sents[mid:])
    raise ValueError(f"unknown position {position!r}")
