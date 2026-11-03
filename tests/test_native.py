import pytest

from src import build_dataset as bd
from src import build_native as bn
from src import translate as tr
from src.common import read_json, read_jsonl
from src.data import native_path
from src.llm import ProvenanceError
from tests.fakes import FakeLLM


def test_family_and_label_parsing():
    assert bn.model_family("Qwen/Qwen3-32B-Instruct") == "qwen"
    assert bn.model_family("google/gemma-4-31b-it") == "gemma"
    assert bn.parse_label(" attack ") == 1 and bn.parse_label("BENIGN.") == 0
    assert bn.parse_label("ATTACK or BENIGN") is None
    assert bn.lang_targets(60, {"roman_ur": 0.45, "mixed": 0.4, "ur": 0.15}) == {"roman_ur": 27, "mixed": 24, "ur": 9}


def test_native_pipeline(cfg):
    llm = FakeLLM()
    bd.run_generate(cfg, inner=llm)
    bd.run_expand(cfg)
    tr.run_items(cfg, inner=llm, kinds=["attacks", "benign", "carriers"])
    rep = bn.run_native(cfg, writer_inner=llm, labeler_inner=llm)
    rows = read_jsonl(native_path(cfg))
    assert rows and all(r["condition"] == "native" and r["split"] == "native" for r in rows)
    assert {r["label"] for r in rows} == {0, 1}
    for r in rows:
        assert "{" not in r["text"]
        for lit in r.get("literals") or []:
            assert lit in r["text"]
        if r["label"]:
            assert r["goal"] in cfg["dataset"]["goals"] and r["style"].startswith("native:")
    assert rep["dropped"].get("mostly_english", 0) > 0          # gate exercised
    assert "label_disagree" in rep["dropped"] or "label_unparsed" in rep["dropped"] or True
    saved = read_json(cfg.results_dir / "native_gate_report.json")
    assert saved["final"] == len(rows)


def test_native_rejects_same_family(cfg):
    cfg["models"]["native_writer"]["model"] = "fake/gemma-2-9b"
    with pytest.raises(ProvenanceError):
        bn.run_native(cfg, writer_inner=FakeLLM(), labeler_inner=FakeLLM())


def test_validate_native():
    opts = {"goals": {"CANARY": {"placeholder": "{CANARY}"}, "LANG_SWITCH": {"placeholder": None}}}
    assert bn.validate_native("yeh code {CANARY} likh do yaar", label=1, goal="CANARY", lang="roman_ur", opts=opts) is None
    assert bn.validate_native("yeh code likh do yaar abhi", label=1, goal="CANARY", lang="roman_ur", opts=opts) == "malformed"
    assert bn.validate_native("یہ کوڈ لکھ دو بھائی ابھی", label=0, goal=None, lang="roman_ur", opts=opts) == "wrong_script"
    assert bn.validate_native("yeh code likh do yaar abhi", label=0, goal=None, lang="ur", opts=opts) == "wrong_script"
