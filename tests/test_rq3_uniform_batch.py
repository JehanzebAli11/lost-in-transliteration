"""RQ3 at one shared max_tokens (owner decision 2026-11-09, option A): Claude Haiku 5.5 through the
Batch API (superseding its OpenRouter rows) and the offline uniform estimate. Offline: fake batch
client, sockets refused, a fake key."""
from __future__ import annotations

import copy

import pytest

from src import eval_models as em
from src.common import read_jsonl
from tests.fakes import FakeLLM

# --------------------------------------------------------------------------- Haiku through the Batch API
jzapi = pytest.importorskip("jzapi")
from src import eval_models_claude_batch as cb  # noqa: E402
from src import rq3_uniform as ru  # noqa: E402
from tests.test_eval_claude_batch import FakeBatchClient, jz, raw, run  # noqa: E402,F401
from tests.test_eval_phase1 import built  # noqa: E402,F401  (fixture)

CLIENTS = ("claude-haiku", "openrouter-small", "gemma-local")


def test_haiku_spec_and_defaults(jz, monkeypatch):
    cfg = jz
    assert [s.name for s in cb.model_specs(cfg)] == ["claude-sonnet", "claude-opus"]   # Haiku opt-in
    (h,) = cb.model_specs(cfg, only=["claude-haiku"])
    assert h.role == "haiku" and h.max_tokens == 1024 and h.extra == {"thinking": {"type": "disabled"}}
    p = cb.build_requests(cfg, h, em.build_tasks(cfg)[:2])[0]["params"]
    assert p["model"] == "claude-haiku-5-5" and p["thinking"] == {"type": "disabled"} and p["max_tokens"] == 1024
    assert not {"temperature", "top_p", "top_k", "output_config"} & set(p)
    assert cb.model_specs(cfg, only=["claude-haiku"], haiku_max_tokens=512)[0].max_tokens == 512
    seen = {}
    monkeypatch.setattr(cb, "load_config", lambda *a, **k: cfg)
    monkeypatch.setattr(cb, "run", lambda c, mode, **kw: seen.update(kw, mode=mode) or [])
    cb.main(["--mode", "sanity", "--only", "claude-haiku", "--haiku-max-tokens", "700"])
    assert seen["haiku_max_tokens"] == 700 and seen["only"] == ["claude-haiku"]
    cb.main(["--mode", "full"])
    assert seen["haiku_max_tokens"] == 1024


def test_haiku_batch_supersedes_the_openrouter_rows(jz):
    cfg = jz
    em.run(cfg, clients={"claude-haiku": FakeLLM()}, only=["claude-haiku"])     # the old route, max 200
    old = {r["key"]: r for r in raw(cfg, "claude-haiku")}
    n = len(old)
    assert {r["endpoint"] for r in old.values()} == {"anthropic"}
    fake = FakeBatchClient()
    with pytest.raises(cb.SanityGateError):
        run(cfg, "full", fake, only=["claude-haiku"])
    st = run(cfg, "sanity", fake, only=["claude-haiku"], round_=3)[0]
    assert st["batch"] == "rq3_claude-haiku_sanity_r3" and st["new_rows"] == 20 and st["superseded_rows"] == 20
    rows = raw(cfg, "claude-haiku")
    assert len(rows) == n and len({r["key"] for r in rows}) == n
    new = [r for r in rows if r["endpoint"] == "anthropic_batch"]
    assert len(new) == 20 and {r["max_tokens"] for r in new} == {1024}
    assert all(r["model_id"] == "claude-haiku-5-5" and r["temperature"] == "api_default"
               and r["request_params"] == {"thinking": {"type": "disabled"}} and r["stop_reason"] == "end_turn"
               for r in new)
    sup = read_jsonl(cfg.results_dir / "rq3_claude_superseded.jsonl")
    assert len(sup) == 20 and all(old[r["key"]] == {k: v for k, v in r.items()
                                                    if k not in ("reason", "superseded_by_batch", "superseded_at")}
                                  for r in sup)
    assert all("max_tokens 1024 via anthropic_batch" in r["reason"] and "endpoint anthropic)" in r["reason"]
               for r in sup)
    run(cfg, "full", fake, only=["claude-haiku"], round_=2)
    rows = raw(cfg, "claude-haiku")
    assert len(rows) == n and {r["endpoint"] for r in rows} == {"anthropic_batch"}
    assert len(read_jsonl(cfg.results_dir / "rq3_claude_superseded.jsonl")) == n
    assert [len(b) for b in fake.batches.values()] == [20, n - 20]


def test_endpoint_change_alone_triggers_a_rerun(jz):
    cfg = jz
    em.run(cfg, clients={"claude-haiku": FakeLLM()}, only=["claude-haiku"])     # max_tokens 200
    (h,) = cb.model_specs(cfg, only=["claude-haiku"], haiku_max_tokens=200)
    assert len(cb.todo_tasks(cfg, h, em.build_tasks(cfg))) == len(em.build_tasks(cfg))


def test_uniform_estimate_is_offline_and_shows_both_headrooms(jz, capsys):
    cfg = jz
    em.run(cfg, clients={m: FakeLLM() for m in CLIENTS})
    fake = FakeBatchClient()
    run(cfg, "sanity", fake, only=["claude-opus"])
    run(cfg, "full", fake, only=["claude-opus"])                                  # Opus already at 1024
    or_cfg = copy.deepcopy(cfg)
    or_cfg["eval_models"]["models"] = [
        {"name": "openrouter-small", "endpoint": "openrouter", "model": "fake/small",
         "input_usd_per_mtok": 0.2, "output_usd_per_mtok": 1.25},
        {"name": "gemma", "endpoint": "openrouter", "model": "fake/gemma", "input_usd_per_mtok": 0.2,
         "output_usd_per_mtok": 0.4}]
    or_cfg["budget"] = {"max_usd": 40, "stop_fraction": 0.95, "openrouter_key_limit_usd": 5}
    before = {p: p.read_bytes() for p in cfg.root.rglob("*") if p.is_file()}
    out = ru.estimate(cfg, or_cfg)
    text = capsys.readouterr().out
    assert {p: p.read_bytes() for p in cfg.root.rglob("*") if p.is_file()} == before   # writes nothing
    n = len(em.build_tasks(cfg))
    m = out["models"]
    assert m["claude-opus"]["n_todo"] == 0 and m["claude-opus"]["full_usd"] == 0
    assert m["claude-haiku"]["n_todo"] == n and m["claude-sonnet"]["n_todo"] == n
    assert m["openrouter-small"]["n_todo"] == n and m["gemma"]["n_todo"] == n      # no "gemma" rows yet
    assert m["claude-haiku"]["n_sanity"] == 20 and 0 < m["claude-haiku"]["sanity_usd"] < m["claude-haiku"]["full_usd"]
    tasks = em.build_tasks(cfg)
    by_hand = sum((cb.estimate_tokens("".join(em.build_prompt(cfg, t))) * 0.2 + 1024 * 1.25) / 1e6 for t in tasks)
    assert m["openrouter-small"]["full_usd"] == pytest.approx(by_hand)
    assert out["jzapi_headroom_after"] == pytest.approx(out["jzapi_headroom"] - out["claude_worst_usd"])
    assert out["lit_headroom_after"] == pytest.approx(out["lit_headroom"] - out["claude_worst_usd"]
                                                      - out["openrouter_worst_usd"])
    assert "jzapi lit_rq3" in text and "LIT spend.json" in text and "headroom" in text and "OpenRouter key" in text
    assert out["fits"] is True
