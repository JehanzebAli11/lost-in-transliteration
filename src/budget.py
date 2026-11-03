"""Budget control (PLAN §9): hard USD cap, pre-call cost estimate, spend.json, disk cache.

* ``Budget.check(est)`` raises :class:`BudgetExceeded` if the running total plus the
  estimate would pass ``stop_fraction`` (default 95%) of ``max_usd``. Callers catch it,
  finish the stage with what they have, and log it.
* ``Budget.record(...)`` adds the *actual* cost from the provider's token usage and
  rewrites ``results/spend.json`` atomically after every call, so a crash never loses
  the total and the cap holds across separate runs.
* ``DiskCache`` stores every response keyed by a hash of (provider, model, request),
  so reruns cost nothing.
* Paid models with no verified price are refused outright (:class:`PriceMissing`):
  we never make a paid call whose cost we cannot bound.
"""
from __future__ import annotations

import json
import math
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .common import is_todo, now_iso, read_json, stable_hash, write_json


class BudgetExceeded(RuntimeError):
    pass


class PriceMissing(RuntimeError):
    pass


@dataclass(frozen=True)
class Price:
    input_per_mtok: float
    output_per_mtok: float

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens * self.input_per_mtok + output_tokens * self.output_per_mtok) / 1e6


FREE = Price(0.0, 0.0)


def price_from_cfg(entry: dict, free: bool = False) -> Price | None:
    """Build a Price from a config entry; None if not verified (TODO / null)."""
    if free:
        return FREE
    i, o = entry.get("input_usd_per_mtok"), entry.get("output_usd_per_mtok")
    if is_todo(i) or is_todo(o):
        return None
    return Price(float(i), float(o))


def estimate_tokens(text: str) -> int:
    """Deliberately conservative token estimate.

    ASCII text ~3 chars/token; non-ASCII (Urdu script) can be close to 1 char/token in
    English-centric tokenizers, so count it at 1 char/token. Over-estimating only makes
    the budget guard stricter.
    """
    if not text:
        return 0
    non_ascii = sum(1 for ch in text if ord(ch) > 127)
    ascii_ = len(text) - non_ascii
    return int(math.ceil(ascii_ / 3.0 + non_ascii * 1.0)) + 8


class Budget:
    def __init__(self, max_usd: float, spend_path: str | Path, stop_fraction: float = 0.95):
        self.max_usd = float(max_usd)
        self.stop_fraction = float(stop_fraction)
        self.spend_path = Path(spend_path)
        self._lock = threading.Lock()
        state = read_json(self.spend_path, default=None)
        self.state: dict[str, Any] = state or {
            "max_usd": self.max_usd,
            "stop_fraction": self.stop_fraction,
            "total_usd": 0.0,
            "calls": 0,
            "cached_calls": 0,
            "by_model": {},
            "by_stage": {},
            "updated": now_iso(),
        }
        self.state["max_usd"] = self.max_usd
        self.state["stop_fraction"] = self.stop_fraction

    @classmethod
    def from_config(cls, cfg) -> "Budget":
        b = cfg.get("budget", {})
        return cls(b.get("max_usd", 40), cfg.results_dir / "spend.json", b.get("stop_fraction", 0.95))

    @property
    def total(self) -> float:
        return float(self.state["total_usd"])

    @property
    def hard_stop_usd(self) -> float:
        return self.max_usd * self.stop_fraction

    @property
    def remaining(self) -> float:
        return self.hard_stop_usd - self.total

    def estimate(self, price: Price | None, prompt_text: str, max_output_tokens: int) -> float:
        if price is None:
            raise PriceMissing("price not verified")
        return price.cost(estimate_tokens(prompt_text), int(max_output_tokens))

    def check(self, est_usd: float, *, label: str = "") -> None:
        with self._lock:
            if self.total + est_usd > self.hard_stop_usd:
                raise BudgetExceeded(
                    f"budget stop: spent ${self.total:.4f} + est ${est_usd:.4f} would pass "
                    f"{self.stop_fraction:.0%} of ${self.max_usd:.2f} cap ({label})")

    def record(self, *, model: str, stage: str, input_tokens: int, output_tokens: int,
               cost_usd: float, cached: bool = False) -> None:
        with self._lock:
            s = self.state
            if cached:
                s["cached_calls"] = s.get("cached_calls", 0) + 1
            else:
                s["calls"] += 1
                s["total_usd"] = round(s["total_usd"] + cost_usd, 8)
                m = s["by_model"].setdefault(model, {"calls": 0, "input_tokens": 0,
                                                     "output_tokens": 0, "usd": 0.0})
                m["calls"] += 1
                m["input_tokens"] += int(input_tokens)
                m["output_tokens"] += int(output_tokens)
                m["usd"] = round(m["usd"] + cost_usd, 8)
                st = s["by_stage"].setdefault(stage, {"calls": 0, "usd": 0.0})
                st["calls"] += 1
                st["usd"] = round(st["usd"] + cost_usd, 8)
            s["updated"] = now_iso()
            write_json(self.spend_path, s)


class DiskCache:
    """Content-addressed JSON cache: ``<dir>/<ab>/<hash>.json``."""

    def __init__(self, directory: str | Path):
        self.dir = Path(directory)

    @staticmethod
    def key(*parts: Any) -> str:
        return stable_hash(*parts)

    def _path(self, key: str) -> Path:
        return self.dir / key[:2] / f"{key}.json"

    def get(self, key: str) -> dict | None:
        p = self._path(key)
        if not p.exists():
            return None
        try:
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return None

    def set(self, key: str, value: dict) -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False)
        os.replace(tmp, p)
