import importlib.util
import shutil
from pathlib import Path

from src import check_connectivity as cc
from src.common import load_config
from tests.conftest import ROOT


def test_todo_report_on_shipped_config(tmp_path):
    cfg = load_config(ROOT / "config.yaml", overrides={"_root": str(tmp_path)})
    todos = cc.enabled_paths(cfg)
    joined = "\n".join(todos)
    for must in ("models.translator.model", "eval_models.models[claude-haiku].model",
                 "eval_models.models[claude-haiku].input_usd_per_mtok", "classifiers.models[protectai-v2].hf_id"):
        assert must in joined, must
    assert "third-classifier" not in joined and "openrouter-open" not in joined  # disabled entries


def test_connectivity_with_fakes(cfg, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    def boom():
        raise ConnectionError("refused")
    lines = cc.check(cfg, local_lister=lambda: ["google/gemma-4-31b-it"], anthropic_lister=boom,
                     hf_info=lambda mid: "gated=False license=apache-2.0")
    text = "\n".join(lines)
    assert "OK   local" in text and "FAIL anthropic" in text and "SET  env ANTHROPIC_API_KEY" in text
    assert "x" not in [l.split()[-1] for l in lines if "ANTHROPIC" in l]  # never prints the value
    assert "OK   huggingface fake/clf-a" in text


def test_ignore_large_results(tmp_path):
    spec = importlib.util.spec_from_file_location("ilr", ROOT / "scripts" / "ignore_large_results.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    shutil.copy(ROOT / ".gitignore", tmp_path / ".gitignore")
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "small.jsonl").write_text("{}\n")
    big = tmp_path / "results" / "big.jsonl"
    with open(big, "wb") as f:
        f.truncate(2 * 1024 * 1024)
    assert mod.update(tmp_path, mb=1) == ["results/big.jsonl"]
    assert mod.update(tmp_path, mb=1) == []  # idempotent
    gi = (tmp_path / ".gitignore").read_text()
    assert gi.count("results/big.jsonl") == 1 and "small.jsonl" not in gi
