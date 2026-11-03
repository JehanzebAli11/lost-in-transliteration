"""Shared helpers: config loading (with profiles), paths, JSONL I/O, seeding, LOG.md.

Every stage script takes ``--config`` (default ``config.yaml``) and ``--profile``
(default: env ``LIT_PROFILE``). A profile is a block under ``profiles:`` in the config
that is deep-merged over the base config, so ``LIT_PROFILE=smoke`` shrinks every stage
and redirects all outputs to ``runs/smoke/``.
"""
from __future__ import annotations

import argparse
import copy
import datetime as _dt
import hashlib
import json
import os
import random
import threading
from pathlib import Path
from typing import Any, Iterable, Iterator

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
TODO_PREFIX = "TODO_VERIFY"
LANGS = ["en", "ur", "roman_ur", "mixed"]
VARIANTS = ["ur", "roman_ur", "mixed"]  # non-English variants


class ConfigTodoError(RuntimeError):
    """A config value still carries a TODO_VERIFY marker."""


# ----------------------------------------------------------------------------- config
def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


class Config(dict):
    """dict with attribute-free helpers for paths and TODO checks."""

    @property
    def root(self) -> Path:
        return Path(self.get("_root", REPO_ROOT))

    def path(self, key: str) -> Path:
        p = Path(self["paths"][key])
        return p if p.is_absolute() else self.root / p

    @property
    def data_dir(self) -> Path:
        return self.path("data_dir")

    @property
    def results_dir(self) -> Path:
        return self.path("results_dir")

    @property
    def figures_dir(self) -> Path:
        return self.path("figures_dir")

    @property
    def models_dir(self) -> Path:
        return self.path("models_dir")

    @property
    def cache_dir(self) -> Path:
        return self.path("cache_dir")

    @property
    def sandbox_dir(self) -> Path:
        return self.path("sandbox_dir")

    @property
    def log_file(self) -> Path:
        return self.path("log_file")

    @property
    def seed(self) -> int:
        return int(self.get("seed", 0))


def load_config(path: str | Path | None = None, profile: str | None = None,
                overrides: dict | None = None) -> Config:
    path = Path(path or os.environ.get("LIT_CONFIG", REPO_ROOT / "config.yaml"))
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    profiles = raw.pop("profiles", {}) or {}
    profile = profile if profile is not None else os.environ.get("LIT_PROFILE") or None
    if profile:
        if profile not in profiles:
            raise KeyError(f"profile {profile!r} not in {path} (have: {sorted(profiles)})")
        raw = deep_merge(raw, profiles[profile])
    if overrides:
        raw = deep_merge(raw, overrides)
    raw.setdefault("_root", str(path.resolve().parent))
    raw["_profile"] = profile
    return Config(raw)


def is_todo(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.strip().startswith(TODO_PREFIX))


def require_verified(value: Any, what: str) -> Any:
    """Refuse to run with a value that still says TODO_VERIFY (or is missing)."""
    if is_todo(value):
        raise ConfigTodoError(
            f"{what} is not verified yet (value={value!r}). Check it on the Spark and "
            f"replace the TODO_VERIFY marker in config.yaml (see RUNBOOK.md step 1).")
    return value


def stage_parser(description: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--config", default=None, help="path to config.yaml")
    ap.add_argument("--profile", default=None, help="profile under profiles: (env LIT_PROFILE)")
    ap.add_argument("--limit", type=int, default=None, help="cap items processed (debugging)")
    return ap


def load_dotenv_if_present(cfg: Config | None = None) -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover
        return
    root = cfg.root if cfg is not None else REPO_ROOT
    env = root / ".env"
    if env.exists():
        load_dotenv(env, override=False)


# ----------------------------------------------------------------------------- io
def read_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    n = 0
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    os.replace(tmp, path)
    return n


class JsonlAppender:
    """Thread-safe append-only JSONL writer, used for resumable long jobs."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def append(self, row: dict) -> None:
        line = json.dumps(row, ensure_ascii=False) + "\n"
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line)
                f.flush()


def read_json(path: str | Path, default: Any = None) -> Any:
    path = Path(path)
    if not path.exists():
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=False)
        f.write("\n")
    os.replace(tmp, path)


# ----------------------------------------------------------------------------- misc
def stable_hash(*parts: Any) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(json.dumps(p, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8"))
        h.update(b"\x1f")
    return h.hexdigest()


def rng_for(seed: int, *salt: Any) -> random.Random:
    """Deterministic RNG derived from the global seed and a salt."""
    return random.Random(int(stable_hash(seed, *salt)[:16], 16))


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed % (2**32))
    except ImportError:  # pragma: no cover
        pass
    try:
        import torch  # noqa: F401
        torch.manual_seed(seed)
    except Exception:  # pragma: no cover - torch optional
        pass


def now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).astimezone().isoformat(timespec="seconds")


_log_lock = threading.Lock()


def log_event(cfg: Config, message: str, *, stage: str | None = None, echo: bool = True) -> None:
    """Append a timestamped line to LOG.md (PLAN §0 rule 3)."""
    line = f"- {now_iso()}" + (f" [{stage}]" if stage else "") + f" {message}\n"
    if echo:
        print(line.rstrip())
    path = cfg.log_file
    path.parent.mkdir(parents=True, exist_ok=True)
    with _log_lock:
        new = not path.exists()
        with open(path, "a", encoding="utf-8") as f:
            if new:
                f.write("# LOG\n\nTimestamped record of what ran, what broke and decisions made.\n\n")
            f.write(line)


def chunked(seq: list, n: int) -> Iterator[list]:
    for i in range(0, len(seq), n):
        yield seq[i:i + n]
