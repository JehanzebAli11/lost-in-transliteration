"""Whole pipeline on a tiny synthetic fixture: every stage runs with fakes and every
§7 figure and table is produced from the fake results."""
import json
import sys
from pathlib import Path

from src import (analyze, build_dataset as bd, build_native as bn, eval_classifiers as ec,
                 eval_detector as ed, eval_mcp_scanners as ms, eval_models as em, eval_valar as ev,
                 probe, train_detector as td, translate as tr)
from src.common import read_json
from src.mcp_fakes import build_variants as bv
from tests.fakes import FakeClassifier, FakeLLM
from tests.tiny_hf import make_tiny_classifier

STUB = Path(__file__).with_name("scanner_stub.py")

EXPECTED_FIGURES = [
    "fig1_detection_by_lang.png", "fig2_fpr_by_lang.png", "fig3_roc_by_classifier.png",
    "fig4_asr_by_lang.png", "fig5_asr_heatmap_style_lang.png", "fig7a_detector_tpr_vs_data.png",
    "fig7b_detector_fpr_vs_data.png", "fig7c_best_detector_vs_classifiers.png", "fig7d_heldout_style.png",
    "fig7e_native_set.png", "fig8_mcp_scanners.png", "fig9_probe.png",
]
EXPECTED_TABLES = ["table_detection_drop.csv", "table_detection_drop.md", "table_asr_gap.csv",
                   "classifiers_summary.csv", "models_summary.csv", "detector_curve.csv", "mcp_summary.csv",
                   "qualitative_failures.md", "headline_numbers.json", "spend.json", "native_gate_report.json"]


def test_full_pipeline(cfg, tmp_path, monkeypatch):
    monkeypatch.delenv("VALAR_API_URL", raising=False)
    llm = FakeLLM(obey_rate=0.4)
    # §3 data
    bd.run_generate(cfg, inner=llm)
    bd.run_expand(cfg)
    tr.run_phrases(cfg, inner=llm)
    tr.run_items(cfg, inner=llm)
    tr.run_review_sample(cfg)
    tr.run_backtranslate(cfg, inner=llm)
    bn.run_native(cfg, writer_inner=llm, labeler_inner=llm)
    # §4 Phase 1
    clfs = {"clf-a": FakeClassifier("clf-a"), "clf-b": FakeClassifier("clf-b", 0.6)}
    ec.run(cfg, classifiers=clfs)
    assert ev.run(cfg)["status"] == "skipped"
    em.run(cfg, clients={"claude-haiku": llm, "openrouter-small": llm, "gemma-local": llm})
    # §5 Phase 2
    cfg["detector"].update({"conditions": [0, 25, 100], "seeds": [1, 2], "epochs": 1, "batch_size": 8,
                            "max_length": 32, "device": "cpu", "only_base_models": ["tiny"],
                            "base_models": [{"name": "tiny", "hf_id": "TODO_VERIFY"}]})
    td.run(cfg, base_override={"tiny": str(make_tiny_classifier(tmp_path / "tiny"))})
    ed.run(cfg)
    # §6 Phase 3
    cfg["mcp"].update({"n_patterns": 3, "n_clean_descriptions": 2})
    bv.run(cfg, inner=llm)
    stub = {"name": "stub-cli", "kind": "command", "parser": "json_any_issue",
            "command": f"{sys.executable} {STUB} {{config_path}}", "sends_remote": False}
    ms.run_scan(cfg, scanners={"stub-cli": ms.CommandScanner(cfg, stub),
                               "classifier-baseline": ms.ClassifierScanner({"name": "c"}, clfs["clf-a"])})
    # §10 probe
    import numpy as np

    def extractor(c, texts):
        rng = np.random.default_rng(1)
        return {L: rng.normal(size=(len(texts), 6)) for L in (2, 4)}
    probe.run(cfg, extractor=extractor)
    # §7
    manifest = analyze.run(cfg)
    assert manifest["skipped"] == [], manifest
    for name in EXPECTED_FIGURES:
        p = cfg.figures_dir / name
        assert p.exists() and p.stat().st_size > 5000, name
    for name in EXPECTED_TABLES:
        assert (cfg.results_dir / name).exists(), name
    head = read_json(cfg.results_dir / "headline_numbers.json")
    assert head["classifier_tpr"]["values"] and head["model_asr"]["values"]
    drop = (cfg.results_dir / "table_detection_drop.md").read_text()
    assert "roman_ur" in drop and "native" in drop


def test_analyze_with_no_results_skips_cleanly(cfg):
    manifest = analyze.run(cfg)
    assert manifest["produced"] == {} and "fig1" in manifest["skipped"]
