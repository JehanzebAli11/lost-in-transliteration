"""Run stages through the real OpenAI and Anthropic SDK clients against a local fake server."""
import pytest

from src import build_dataset as bd, check_connectivity as cc, eval_models as em, translate as tr
from src.common import read_json, read_jsonl
from src.data import attacks_path
from tests.fake_http import serve


@pytest.fixture
def http(cfg, monkeypatch):
    srv, url, llm = serve()
    for ep in ("local", "native_local", "openrouter"):
        cfg["endpoints"][ep]["base_url"] = f"{url}/v1"
    cfg["endpoints"]["anthropic"]["base_url"] = url
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    yield cfg, llm
    srv.shutdown()


def test_stages_over_http(http):
    cfg, llm = http
    bd.run_generate(cfg)          # no inner client: real OpenAICompatClient
    bd.run_expand(cfg)
    tr.run_items(cfg, kinds=["attacks", "carriers"])
    assert len(read_jsonl(attacks_path(cfg, "roman_ur"))) == cfg["dataset"]["n_attacks"]
    stats = em.run(cfg)           # real Anthropic + OpenAI-compatible clients
    assert {s["status"] for s in stats} == {"ok"}
    raw = read_jsonl(cfg.results_dir / "models_raw.jsonl")
    assert {r["endpoint"] for r in raw} == {"anthropic", "openrouter", "local"}
    spend = read_json(cfg.results_dir / "spend.json")
    assert spend["by_model"]["fake-haiku"]["usd"] > 0
    # cache: a second run makes no HTTP calls
    n = len(llm.calls)
    for p in ["models_raw.jsonl"]:
        (cfg.results_dir / p).unlink()
    em.run(cfg)
    assert len(llm.calls) == n


def test_connectivity_over_http(http):
    cfg, _ = http
    lines = cc.check(cfg, hf_info=lambda mid: "ok")
    assert any(l.startswith("OK   local") and "fake/gemma-4-31b-it" in l for l in lines)
