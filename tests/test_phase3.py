import subprocess
import sys
from pathlib import Path

import pytest

from src import eval_mcp_scanners as ms
from src.common import read_json, read_jsonl
from src.mcp_fakes import build_variants as bv
from tests.fakes import FakeClassifier, FakeLLM

STUB = Path(__file__).with_name("scanner_stub.py")


@pytest.fixture
def mcp_cfg(cfg):
    cfg["mcp"].update({"n_patterns": 4, "n_clean_descriptions": 2})
    bv.run(cfg, inner=FakeLLM())
    return cfg


def test_variants_built(mcp_cfg):
    cfg = mcp_cfg
    man = bv.load_manifest(cfg)
    kinds = {(m["lang"], m["kind"]) for m in man}
    for lang in ("en", "ur", "roman_ur", "mixed"):
        for kind in ("clean", "poisoned_description", "poisoned_output"):
            assert (lang, kind) in kinds
    secret = (cfg.root / cfg["mcp"]["secret_file"]).read_text()
    assert secret.startswith("FAKE-CANARY-")
    for m in man:
        spec = read_json(bv.mcp_dir(cfg) / m["spec"])
        names = [t["name"] for t in spec["tools"]]
        assert names[-1] == "send_report"
        blob = " ".join(t["description"] + t["output"] for t in spec["tools"])
        lits = cfg["mcp"]["secret_file"] in blob or "call send_report" in blob or "send_report with" in blob \
            or blob.count("send_report") > 1
        assert lits == (m["kind"] != "clean")
        assert secret not in blob  # the secret value itself never appears in a spec


def test_servers_speak_mcp(mcp_cfg):
    rep = ms.run_check(mcp_cfg, limit=3)
    assert rep == {"checked": 3, "bad": []}


def test_scan_with_adapters(mcp_cfg):
    cfg = mcp_cfg
    cmd_entry = {"name": "stub-cli", "kind": "command", "parser": "json_any_issue",
                 "command": f"{sys.executable} {STUB} {{config_path}}", "timeout": 60,
                 "sends_remote": False}
    scanners = {
        "stub-cli": ms.CommandScanner(cfg, cmd_entry),
        "clf": ms.ClassifierScanner({"name": "clf"}, FakeClassifier("clf")),
        "exit": ms.CommandScanner(cfg, {"name": "exit", "parser": "exit_code", "command": "x {config_path}"},
                                  runner=lambda argv, timeout: subprocess.CompletedProcess(argv, 1, "", "")),
    }
    rows = ms.run_scan(cfg, scanners=scanners)
    stub = [r for r in rows if r["scanner"] == "stub-cli"]
    assert all(r["flagged"] == (r["kind"] == "poisoned_description") for r in stub if r["applicable"])
    assert all(not r["applicable"] for r in stub if r["kind"] == "poisoned_output")
    assert all(r["flagged"] for r in rows if r["scanner"] == "exit" and r["applicable"])
    import csv
    summ = list(csv.DictReader(open(cfg.results_dir / "mcp_summary.csv")))
    assert {r["metric"] for r in summ} == {"detection_rate", "false_positive_rate"}
    assert (cfg.root / "mcp_configs").exists()


def test_build_scanners_skips_unverified(mcp_cfg):
    cfg = mcp_cfg
    cfg["mcp"]["scanners"] = [
        {"name": "a", "kind": "command", "command": "TODO_VERIFY: x", "sends_remote": False},
        {"name": "b", "kind": "command", "command": "echo {config_path}", "sends_remote": "TODO_VERIFY"},
        {"name": "c", "kind": "classifier", "classifier": "missing"},
        {"name": "d", "kind": "command", "enabled": False, "command": "x"}]
    assert ms.build_scanners(cfg, classifiers={}) == {}
    assert cfg.log_file.read_text().count("SKIP scanner") == 3


def test_json_parser():
    assert ms.parse_json_any_issue('noise {"a": {"issues": [1]}}')[0]
    assert ms.parse_json_any_issue('{"result": {"is_safe": false}}')[0]
    assert not ms.parse_json_any_issue('{"issues": [], "severity": "none"}')[0]
    with pytest.raises(ValueError):
        ms.parse_json_any_issue("no json")
