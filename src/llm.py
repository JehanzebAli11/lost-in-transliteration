"""Injectable chat clients.

Every model call in the project goes through a :class:`ChatClient`:

* :class:`OpenAICompatClient` – vLLM on the Spark (``http://localhost:8000/v1``),
  OpenRouter, or any other OpenAI-compatible server.
* :class:`AnthropicClient` – the Anthropic Messages API (evaluation only; PLAN rule 8).
* :class:`ManagedClient` – wraps any client with the disk cache and the budget guard.

Tests pass fakes with the same ``chat()`` signature, so no stage ever needs a network.
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Protocol

from .budget import Budget, BudgetExceeded, DiskCache, Price, PriceMissing, price_from_cfg
from .common import Config, require_verified

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_reasoning(text: str | None) -> str:
    """Remove ``<think>...</think>`` blocks some open models emit, and trim."""
    if not text:
        return ""
    text = THINK_RE.sub("", text)
    if "</think>" in text:  # unterminated opening tag was swallowed by the template
        text = text.split("</think>", 1)[1]
    return text.strip()


@dataclass
class ChatResult:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    finish_reason: str | None = None
    cached: bool = False
    extra: dict = field(default_factory=dict)


class ChatClient(Protocol):
    def chat(self, *, model: str, system: str | None, user: str, temperature: float,
             max_tokens: int, seed: int | None = None,
             extra_body: dict | None = None) -> ChatResult: ...


# ----------------------------------------------------------------------------- real clients
class OpenAICompatClient:
    """OpenAI-compatible chat completions (vLLM, OpenRouter)."""

    def __init__(self, base_url: str, api_key: str | None = None, timeout: float = 300.0,
                 default_headers: dict | None = None, sdk_client: Any = None):
        if sdk_client is None:
            from openai import OpenAI
            sdk_client = OpenAI(base_url=base_url, api_key=api_key or "EMPTY", timeout=timeout,
                                default_headers=default_headers or None, max_retries=2)
        self._c = sdk_client
        self.base_url = base_url

    def chat(self, *, model, system, user, temperature, max_tokens, seed=None, extra_body=None):
        messages = ([{"role": "system", "content": system}] if system else []) + \
                   [{"role": "user", "content": user}]
        kwargs: dict[str, Any] = dict(model=model, messages=messages, temperature=temperature,
                                      max_tokens=max_tokens)
        if seed is not None:
            kwargs["seed"] = seed
        if extra_body:
            kwargs["extra_body"] = extra_body
        resp = self._c.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        usage = getattr(resp, "usage", None)
        return ChatResult(
            text=strip_reasoning(choice.message.content),
            input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            model=getattr(resp, "model", model) or model,
            finish_reason=getattr(choice, "finish_reason", None),
        )

    def list_models(self) -> list[str]:
        return [m.id for m in self._c.models.list().data]


class AnthropicClient:
    """Anthropic Messages API. Used only to *evaluate* models, never to make data.

    Temperature is not sendable on the current API version; responses are sampled at
    the API default (see chat()).
    """
    sends_temperature = False

    def __init__(self, api_key: str | None = None, timeout: float = 120.0, sdk_client: Any = None):
        if sdk_client is None:
            import anthropic
            sdk_client = anthropic.Anthropic(api_key=api_key, timeout=timeout, max_retries=2)
        self._c = sdk_client

    def chat(self, *, model, system, user, temperature, max_tokens, seed=None, extra_body=None):
        # API version 2026-10-21 (anthropic SDK >= 2.0) rejects temperature/top_p/top_k with a
        # 400, and Claude 4.7+ models only sample at the default (1.0) anyway, so the
        # requested temperature is deliberately NOT sent. eval_models records this per row.
        # https://platform.claude.com/docs/en/api/migration-2026-10-21
        kwargs: dict[str, Any] = dict(model=model, max_tokens=max_tokens,
                                      messages=[{"role": "user", "content": user}])
        if system:
            kwargs["system"] = system
        if extra_body:
            kwargs["extra_body"] = extra_body
        resp = self._c.messages.create(**kwargs)
        text = "".join(getattr(b, "text", "") for b in resp.content
                       if getattr(b, "type", "text") == "text")
        usage = resp.usage
        return ChatResult(text=text.strip(),
                          input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
                          output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
                          model=getattr(resp, "model", model) or model,
                          finish_reason=getattr(resp, "stop_reason", None))

    def list_models(self) -> list[str]:
        return [m.id for m in self._c.models.list().data]


# ----------------------------------------------------------------------------- managed wrapper
NON_RETRYABLE = (BudgetExceeded, PriceMissing, KeyboardInterrupt, TypeError, ValueError,
                 KeyError, AttributeError, NotImplementedError)


def is_retryable(e: Exception) -> bool:
    """Retry network errors, 408/409/429 and 5xx; never client errors or bugs."""
    if isinstance(e, NON_RETRYABLE):
        return False
    code = getattr(e, "status_code", None) or getattr(getattr(e, "response", None), "status_code", None)
    if isinstance(code, int) and 400 <= code < 500 and code not in (408, 409, 429):
        return False
    return True


class ManagedClient:
    """Cache + budget + retry around any ChatClient."""

    def __init__(self, inner: ChatClient, *, endpoint: str, stage: str,
                 price: Price | None, budget: Budget | None, cache: DiskCache | None,
                 retries: int = 4, sleep: Callable[[float], None] = time.sleep):
        self.inner = inner
        self.endpoint = endpoint
        self.stage = stage
        self.price = price
        self.budget = budget
        self.cache = cache
        self.retries = retries
        self.sleep = sleep

    def chat(self, *, model, system, user, temperature, max_tokens, seed=None, extra_body=None):
        key = DiskCache.key(self.endpoint, model, system, user, temperature, max_tokens, seed,
                            extra_body or {})
        if self.cache is not None:
            hit = self.cache.get(key)
            if hit is not None:
                res = ChatResult(**{k: v for k, v in hit.items() if k in ChatResult.__dataclass_fields__})
                res.cached = True
                if self.budget is not None:
                    self.budget.record(model=model, stage=self.stage, input_tokens=0,
                                       output_tokens=0, cost_usd=0.0, cached=True)
                return res
        if self.price is None:
            raise PriceMissing(f"no verified price for {self.endpoint}:{model}; refusing paid call")
        if self.budget is not None and (self.price.input_per_mtok or self.price.output_per_mtok):
            est = self.budget.estimate(self.price, (system or "") + user, max_tokens)
            self.budget.check(est, label=f"{self.endpoint}:{model}")
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                res = self.inner.chat(model=model, system=system, user=user,
                                      temperature=temperature, max_tokens=max_tokens,
                                      seed=seed, extra_body=extra_body)
                break
            except Exception as e:  # network / rate limit / 5xx are retried
                last = e
                if not is_retryable(e) or attempt == self.retries:
                    raise
                self.sleep(min(60.0, 2.0 ** attempt))
        else:  # pragma: no cover
            raise last  # type: ignore[misc]
        cost = self.price.cost(res.input_tokens, res.output_tokens)
        if self.budget is not None:
            self.budget.record(model=model, stage=self.stage, input_tokens=res.input_tokens,
                               output_tokens=res.output_tokens, cost_usd=cost)
        if self.cache is not None:
            d = asdict(res)
            d["cached"] = False
            self.cache.set(key, d)
        return res


# ----------------------------------------------------------------------------- factory
@dataclass
class ModelSpec:
    """A resolved model role: which client to call and with which parameters."""
    name: str
    endpoint: str
    model: str
    temperature: float = 0.0
    max_tokens: int = 512
    extra_body: dict | None = None
    client: ChatClient | None = None

    def chat(self, user: str, system: str | None = None, *, seed: int | None = None,
             temperature: float | None = None, max_tokens: int | None = None) -> ChatResult:
        assert self.client is not None, "ModelSpec has no client"
        return self.client.chat(model=self.model, system=system, user=user,
                                temperature=self.temperature if temperature is None else temperature,
                                max_tokens=self.max_tokens if max_tokens is None else max_tokens,
                                seed=seed, extra_body=self.extra_body)


def build_inner_client(cfg: Config, endpoint_name: str) -> ChatClient:
    ep = cfg["endpoints"][endpoint_name]
    key = os.environ.get(ep["api_key_env"]) if ep.get("api_key_env") else None
    if ep.get("api_key_env") and not key:
        raise RuntimeError(f"endpoint {endpoint_name}: env var {ep['api_key_env']} is not set")
    if ep["provider"] == "anthropic":
        if ep.get("base_url"):  # only for tests / proxies
            import anthropic
            return AnthropicClient(sdk_client=anthropic.Anthropic(api_key=key, base_url=ep["base_url"],
                                                                  timeout=ep.get("timeout", 120), max_retries=0))
        return AnthropicClient(api_key=key, timeout=ep.get("timeout", 120))
    if ep["provider"] == "openai_compat":
        return OpenAICompatClient(base_url=ep["base_url"], api_key=key,
                                  timeout=ep.get("timeout", 300),
                                  default_headers=ep.get("headers"))
    raise ValueError(f"unknown provider {ep['provider']!r}")


def resolve_model(cfg: Config, entry: dict, *, stage: str, budget: Budget | None = None,
                  cache: DiskCache | None = None, inner: ChatClient | None = None,
                  name: str | None = None) -> ModelSpec:
    """Turn a config entry ({endpoint, model, ...}) into a ready-to-call ModelSpec.

    ``inner`` injects a client (tests); otherwise one is built from ``endpoints:``.
    Endpoints marked ``free: true`` (the Spark's local vLLM) cost $0 and need no price.
    """
    endpoint = entry["endpoint"]
    ep = cfg["endpoints"][endpoint]
    model = require_verified(entry.get("model"), f"model id for {name or stage} ({endpoint})")
    free = bool(ep.get("free", False))
    price = price_from_cfg(entry, free=free)
    if inner is None:
        inner = build_inner_client(cfg, endpoint)
    client = ManagedClient(inner, endpoint=endpoint, stage=stage, price=price, budget=budget,
                           cache=cache)
    return ModelSpec(name=name or entry.get("name") or model, endpoint=endpoint, model=model,
                     temperature=float(entry.get("temperature", 0.0)),
                     max_tokens=int(entry.get("max_tokens", 512)),
                     extra_body=entry.get("extra_body") or None, client=client)


def resolve_role(cfg: Config, role: str, *, stage: str, budget: Budget | None = None,
                 cache: DiskCache | None = None, inner: ChatClient | None = None) -> ModelSpec:
    return resolve_model(cfg, cfg["models"][role], stage=stage, budget=budget, cache=cache,
                         inner=inner, name=role)


class ProvenanceError(RuntimeError):
    """PLAN rule 8: a closed/evaluated model was configured to create or translate data."""


def assert_open_weight(cfg: Config, spec: ModelSpec) -> ModelSpec:
    """Refuse data-making roles that point at Anthropic or a known closed model id."""
    ep = cfg["endpoints"][spec.endpoint]
    if ep["provider"] == "anthropic":
        raise ProvenanceError(f"{spec.name}: data-making role uses the Anthropic endpoint")
    low = spec.model.lower()
    for marker in cfg.get("closed_model_markers", []):
        if marker.lower() in low:
            raise ProvenanceError(f"{spec.name}: model id {spec.model!r} looks closed "
                                  f"(matches {marker!r}); only open-weight models may make data")
    return spec


def data_model(cfg: Config, role: str, *, stage: str, inner: ChatClient | None = None,
               budget: Budget | None = None) -> ModelSpec:
    """Resolve a data-making role (generator/translator/native_*) with the rule-8 guard."""
    spec = resolve_role(cfg, role, stage=stage, budget=budget,
                        cache=default_cache(cfg, f"llm/{role}"), inner=inner)
    return assert_open_weight(cfg, spec)


def default_cache(cfg: Config, sub: str) -> DiskCache:
    return DiskCache(cfg.cache_dir / sub)
