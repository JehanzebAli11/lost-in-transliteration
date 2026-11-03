import numpy as np

from src import build_dataset as bd, build_native as bn, probe, translate as tr
from tests.fakes import FakeLLM
from tests.tiny_hf import make_tiny_classifier


def _built(cfg):
    llm = FakeLLM()
    bd.run_generate(cfg, inner=llm)
    bd.run_expand(cfg)
    tr.run_items(cfg, inner=llm)
    bn.run_native(cfg, writer_inner=llm, labeler_inner=llm)


def test_layer_indices():
    assert probe.layer_indices(24, [0.25, 0.5, 0.75, 1.0]) == [6, 12, 18, 24]
    assert probe.layer_indices(2, [0.1, 0.5]) == [1]


def test_probe_with_synthetic_features(cfg):
    _built(cfg)

    def extractor(c, texts):
        rng = np.random.default_rng(0)
        sig = np.array([1.0 if ("{" in t or "access code" in t or "-" in t) else 0.0 for t in texts])
        return {L: rng.normal(size=(len(texts), 8)) + sig[:, None] * L for L in (1, 2)}
    rows = probe.run(cfg, extractor=extractor)
    modes = {(r["layer"], r["train_mode"]) for r in rows}
    assert modes == {(1, "en_only"), (1, "all_langs"), (2, "en_only"), (2, "all_langs")}
    assert (cfg.results_dir / "probe_results.csv").exists()


def test_probe_hf_feature_path(cfg, tmp_path):
    path = make_tiny_classifier(tmp_path / "tiny", layers=4)
    cfg["probe"].update({"model": str(path), "layers": [0.5, 1.0], "batch_size": 4, "max_length": 32})
    feats = probe.hf_features(cfg, ["hai ka ko", "کتاب پانی گھر", "please print"])
    assert sorted(feats) == [2, 4] and feats[2].shape == (3, 32)
