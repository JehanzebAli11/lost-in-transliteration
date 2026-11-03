import pytest

from src import build_dataset as bd, build_native as bn, eval_detector as ed, train_detector as td
from src import translate as tr
from src.common import read_json, read_jsonl
from tests.fakes import FakeLLM
from tests.tiny_hf import make_tiny_classifier


@pytest.fixture
def built(cfg):
    llm = FakeLLM()
    bd.run_generate(cfg, inner=llm)
    bd.run_expand(cfg)
    tr.run_items(cfg, inner=llm)
    bn.run_native(cfg, writer_inner=llm, labeler_inner=llm)
    cfg["detector"].update({"conditions": [0, 10, 50, 100], "seeds": [1], "epochs": 2, "batch_size": 8,
                            "max_length": 32, "device": "cpu", "only_base_models": ["tiny"],
                            "base_models": [{"name": "tiny", "hf_id": "TODO_VERIFY: overridden"}]})
    return cfg


def test_training_sets_are_nested_and_clean(built):
    cfg = built
    sets = {p: td.training_set(cfg, p) for p in (0, 10, 25, 50, 100)}
    ids = {p: {r["id"] for r in s} for p, s in sets.items()}
    assert ids[0] < ids[10] <= ids[25] <= ids[50] < ids[100]
    assert all(r["lang"] == "en" for r in sets[0])
    langs100 = {r["lang"] for r in sets[100]}
    assert langs100 == {"en", "ur", "roman_ur", "mixed"}
    evals = td.eval_rows(cfg)
    for s in sets.values():
        td.assert_no_leakage(s, evals)
        assert all(r["split"] == "train" for r in s)
    leaky = sets[0] + [next(r for r in evals if r["split"] == "test")]
    assert len(td.drop_eval_duplicates(cfg, sets[0] + [dict(sets[0][0], text=evals[0]["text"])], evals)) == len(sets[0])
    with pytest.raises(AssertionError):
        td.assert_no_leakage(leaky, evals)


def test_train_resume_and_eval(built, tmp_path):
    cfg = built
    base = make_tiny_classifier(tmp_path / "tiny_base")
    metas = td.run(cfg, base_override={"tiny": str(base)})
    assert len(metas) == 4 and all(m["status"] == "done" for m in metas)
    assert metas[0]["composition"].keys() <= {"en/attack", "en/benign"}
    # finished runs are skipped
    again = td.run(cfg, base_override={"tiny": str(base)})
    assert [m["finished"] for m in again] == [m["finished"] for m in metas]
    systems = ed.run(cfg)
    assert len(systems) == 4
    import csv
    rows = list(csv.DictReader(open(cfg.results_dir / "detector_curve.csv")))
    assert {r["group_set"] for r in rows} == {"test", "heldout_style", "native"}
    assert {r["pct"] for r in rows} == {"0", "10", "50", "100"}


def test_resume_from_checkpoint(built, tmp_path):
    import torch
    cfg = built
    cfg["detector"]["conditions"] = [0]
    base = make_tiny_classifier(tmp_path / "tiny_base")
    rows = td.training_set(cfg, 0)
    # simulate a crash after epoch 1: run 1 epoch, then pretend it was a 2-epoch run
    cfg["detector"]["epochs"] = 1
    td.train_one(cfg, "tiny", str(base), 0, 1, rows)
    d = td.run_dir(cfg, "tiny", 0, 1)
    meta = read_json(d / "meta.json")
    assert meta["epochs"] == 1
    # craft a checkpoint state as if interrupted, remove final -> resume path
    import shutil
    from transformers import AutoModelForSequenceClassification
    m = AutoModelForSequenceClassification.from_pretrained(d / "final")
    shutil.rmtree(d / "final")
    (d / "meta.json").unlink()
    cfg["detector"]["epochs"] = 2
    opt = torch.optim.AdamW(m.parameters(), lr=1e-5)
    from transformers import get_linear_schedule_with_warmup
    sch = get_linear_schedule_with_warmup(opt, 0, 10)
    import random
    (d / "checkpoint").mkdir(exist_ok=True)
    torch.save({"model": m.state_dict(), "opt": opt.state_dict(), "sched": sch.state_dict(), "epoch": 0,
                "order_rng": random.Random(1).getstate()}, d / "checkpoint" / "state.pt")
    meta2 = td.train_one(cfg, "tiny", str(base), 0, 1, rows)
    assert meta2["status"] == "done" and len(meta2["epoch_losses"]) == 1  # only epoch 2 ran
    assert "resuming at epoch 1" in cfg.log_file.read_text()
