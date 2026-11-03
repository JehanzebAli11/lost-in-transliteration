import csv
import json

import pytest

from src import build_dataset as bd, build_native as bn, eval_classifiers as ec, eval_models as em
from src import eval_valar as ev, translate as tr
from src.budget import Budget
from src.common import read_json, read_jsonl
from tests.fakes import FakeClassifier, FakeLLM
from tests.tiny_hf import make_tiny_classifier


@pytest.fixture
def built(cfg):
    llm = FakeLLM()
    bd.run_generate(cfg, inner=llm)
    bd.run_expand(cfg)
    tr.run_phrases(cfg, inner=llm)
    tr.run_items(cfg, inner=llm)
    bn.run_native(cfg, writer_inner=llm, labeler_inner=llm)
    return cfg


def test_classifiers_with_fakes(built):
    cfg = built
    summ = ec.run(cfg, classifiers={"clf-a": FakeClassifier("clf-a"), "clf-b": FakeClassifier("clf-b", 0.5)})
    sets = {(s["system"], s["group_set"], s["lang_condition"]) for s in summ}
    for lang in ("en", "ur", "roman_ur", "mixed"):
        for gs in ("all", "test", "heldout_style", "train"):
            assert ("clf-a", gs, lang) in sets
    assert ("clf-a", "native", "native") in sets
    raw = read_jsonl(cfg.results_dir / "classifiers" / "clf-a.jsonl")
    assert {r["label"] for r in raw} == {0, 1}
    assert (cfg.results_dir / "classifiers_by_style.csv").stat().st_size > 0
    # rerun skips already-scored classifiers
    ec.run(cfg, classifiers={"clf-a": FakeClassifier("clf-a")})


def test_embedded_condition(built):
    cfg = built
    cfg["classifiers"]["embedded_condition"] = True
    ec.run(cfg, classifiers={"clf-a": FakeClassifier("clf-a")})
    raw = read_jsonl(cfg.results_dir / "classifiers" / "clf-a.jsonl")
    assert {"standalone", "embedded"} <= {r["input"] for r in raw}


def test_hf_classifier_real_code_path(tmp_path):
    path = make_tiny_classifier(tmp_path / "tiny")
    clf = ec.HFClassifier("tiny", str(path), ["INJECTION"], max_length=16, device="cpu", batch_size=3)
    long = "fixture alpha " * 60  # forces several overflow windows
    scores = clf.score(["hai ka ko", long, "کتاب پانی", "x"])
    assert len(scores) == 4 and all(0.0 <= s <= 1.0 for s in scores)
    with pytest.raises(ValueError):
        ec.HFClassifier("tiny", str(path), ["NOPE"], device="cpu")


def test_build_classifiers_skips_todo_and_gated(cfg, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    cfg["classifiers"]["models"] = [
        {"name": "todo", "hf_id": "TODO_VERIFY: x", "positive_labels": ["A"]},
        {"name": "gated", "hf_id": "x/y", "positive_labels": ["A"], "gated": True},
        {"name": "off", "enabled": False, "hf_id": "x/z", "positive_labels": ["A"]}]
    assert ec.build_classifiers(cfg) == {}
    log = (cfg.log_file).read_text()
    assert "SKIP classifier todo" in log and "SKIP classifier gated" in log


def test_valar_skip_and_http(built, monkeypatch):
    cfg = built
    monkeypatch.delenv("VALAR_API_URL", raising=False)
    monkeypatch.delenv("VALAR_API_KEY", raising=False)
    assert ev.run(cfg)["status"] == "skipped"
    assert read_json(cfg.results_dir / "valar_status.json")["status"] == "skipped"
    calls = []

    def post(url, json, headers):
        calls.append(headers)
        return {"result": {"score": 0.9 if "access" in json["text"] else 0.1}}
    vc = dict(cfg["valar"], response_score_path="result.score")
    sys_ = ev.ValarHTTP("http://valar.test/scan", "k", vc, post=post)
    assert ev.run(cfg, system=sys_, limit=10)["n"] == 10
    assert calls[0]["Authorization"] == "Bearer k"
    rows = read_jsonl(cfg.results_dir / "classifiers" / "valar.jsonl")
    assert rows[0]["family"] == "valar"


def test_eval_models_end_to_end(built):
    cfg = built
    llm = FakeLLM(obey_rate=0.5)
    stats = em.run(cfg, clients={"claude-haiku": llm, "openrouter-small": llm, "gemma-local": llm})
    assert {s["model"] for s in stats} == {"claude-haiku", "openrouter-small", "gemma-local"}
    raw = read_jsonl(cfg.results_dir / "models_raw.jsonl")
    settings = {r["setting"] for r in raw}
    assert settings == {"same", "cross_en_carrier", "clean"}
    assert {r["lang_condition"] for r in raw if r["setting"] == "same"} >= {"en", "ur", "roman_ur", "mixed", "native"}
    sec = [r for r in raw if r["goal"] == "SECRET_LEAK"]
    assert sec and all("The access code is" in r["system_prompt"] for r in sec)
    assert all("access code" not in r["system_prompt"] for r in raw if r["goal"] not in ("SECRET_LEAK",))
    # paired sampling: same source ids in every language
    by_lang = {}
    for r in raw:
        if r["setting"] == "same" and r["condition"] == "main" and r["model"] == "gemma-local":
            by_lang.setdefault(r["lang"], set()).add(r["source_id"])
    assert len({frozenset(v) for v in by_lang.values()}) == 1
    spend = read_json(cfg.results_dir / "spend.json")
    assert spend["total_usd"] > 0 and "fake-haiku" in spend["by_model"]
    assert "fake/gemma-4-31b-it" not in spend["by_model"] or spend["by_model"]["fake/gemma-4-31b-it"]["usd"] == 0
    summ = list(csv.DictReader(open(cfg.results_dir / "models_summary.csv")))
    assert summ and {"asr", "ci_lo", "ci_hi"} <= set(summ[0])
    # resumable: rerun makes no new calls
    n = len(llm.calls)
    em.run(cfg, clients={"claude-haiku": llm, "openrouter-small": llm, "gemma-local": llm})
    assert len(llm.calls) == n


def test_eval_models_budget_stop_and_missing_price(built):
    cfg = built
    cfg["budget"]["max_usd"] = 0.0005
    cfg["eval_models"]["models"][1]["input_usd_per_mtok"] = None
    llm = FakeLLM()
    stats = em.run(cfg, clients={"claude-haiku": llm, "openrouter-small": llm, "gemma-local": llm})
    by = {s["model"]: s for s in stats}
    assert by["claude-haiku"]["status"] == "budget_stop"
    assert by["openrouter-small"]["status"] == "skipped"
    assert by["gemma-local"]["status"] == "ok"  # free local model is unaffected
    assert read_json(cfg.results_dir / "spend.json")["total_usd"] <= 0.0005 * 0.95 + 1e-9
    assert "BUDGET STOP" in cfg.log_file.read_text()
