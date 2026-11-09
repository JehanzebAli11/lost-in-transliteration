"""RQ3 Claude Sonnet/Opus through jzapi Message Batches: offline, with a fake batch client."""
from __future__ import annotations

import csv
import json
import socket
from types import SimpleNamespace as NS

import pytest

jzapi = pytest.importorskip("jzapi")

from src import eval_models as em  # noqa: E402
from src import eval_models_claude_batch as cb  # noqa: E402
from src.common import read_json, read_jsonl, write_json  # noqa: E402
from tests.fakes import FakeLLM  # noqa: E402
from tests.test_eval_phase1 import built  # noqa: E402,F401  (fixture)

FAKE_KEY = "sk-ant-test-FAKEKEY-lit-0123456789abcdef-never-real"


class FakeBatchClient:
    """messages.batches.{create,retrieve,results}; answers with FakeLLM so the judge has work to do."""

    def __init__(self, *, empty_cids=(), statuses=None):
        self.calls = []
        self.batches = {}
        self.statuses = list(statuses or [])
        self.empty_cids = set(empty_cids)
        self.llm = FakeLLM(obey_rate=0.5)
        self.messages = NS(batches=NS(create=self._create, retrieve=self._retrieve, results=self._results),
                           count_tokens=self._count)

    def n(self, kind):
        return sum(1 for c in self.calls if c[0] == kind)

    def _count(self, **params):
        self.calls.append(("count_tokens", params))
        return NS(input_tokens=100)

    def _create(self, *, requests):
        bid = f"msgbatch_fake{len(self.batches) + 1}"
        self.calls.append(("batch_create", requests))
        self.batches[bid] = list(requests)
        return NS(id=bid, processing_status="in_progress")

    def _retrieve(self, bid):
        self.calls.append(("batch_retrieve", bid))
        ps = self.statuses.pop(0) if self.statuses else "ended"
        return NS(id=bid, processing_status=ps, request_counts=NS(processing=0, succeeded=len(self.batches[bid]),
                                                                 errored=0, canceled=0, expired=0))

    def _results(self, bid):
        self.calls.append(("batch_results", bid))
        out = []
        for r in self.batches[bid]:
            p = r["params"]
            system = "".join(b["text"] for b in p["system"])
            user = p["messages"][0]["content"]
            if r["custom_id"] in self.empty_cids:
                text, stop, otok = "", "max_tokens", p["max_tokens"]
            else:
                text = self.llm.chat(model=p["model"], system=system, user=user, temperature=None,
                                     max_tokens=p["max_tokens"]).text
                stop, otok = "end_turn", 40
            msg = NS(id="msg_fake", model=p["model"], stop_reason=stop, content=[NS(type="text", text=text)],
                     usage=NS(input_tokens=500, output_tokens=otok, cache_creation_input_tokens=0,
                              cache_read_input_tokens=0, cache_creation=None))
            out.append(NS(custom_id=r["custom_id"], result=NS(type="succeeded", message=msg)))
        return iter(out)

    def requests_of(self, model_id):
        return [r for reqs in self.batches.values() for r in reqs if r["params"]["model"] == model_id]


@pytest.fixture
def jz(built, tmp_path, monkeypatch):
    """Offline jzapi: a test budgets.yaml, a fake key in the environment, sockets refused."""
    def refuse(*a, **k):
        raise RuntimeError("network is disabled in this test")
    monkeypatch.setattr(socket.socket, "connect", refuse)
    budgets = tmp_path / "budgets.yaml"
    budgets.write_text("total_usd: 400\njobs:\n  lit_rq3:\n    cap_usd: 20\n    approved: true\n")
    monkeypatch.setenv("JZ_BUDGETS_FILE", str(budgets))
    monkeypatch.setenv("JZ_ANTHROPIC_API_KEY", FAKE_KEY)
    monkeypatch.setenv("JZ_ENV_FILE", str(tmp_path / "no.env"))
    monkeypatch.delenv("JZ_REPO_DIR", raising=False)
    jzapi.api._sessions.clear()
    return built


def run(cfg, mode, client, **kw):
    kw.setdefault("sleep", lambda s: None)
    return cb.run(cfg, mode, client=client, **kw)


def raw(cfg, model=None):
    return [r for r in read_jsonl(cfg.results_dir / "models_raw.jsonl") if model is None or r["model"] == model]


# ----------------------------------------------------------------------------- prompts and params
def test_prompts_identical_to_eval_models(jz):
    cfg = jz
    em.run(cfg, clients={"claude-haiku": FakeLLM(), "openrouter-small": FakeLLM(), "gemma-local": FakeLLM()},
           only=["claude-haiku"])
    ref = {r["key"].split("|", 1)[1]: (r["system_prompt"], r["user_prompt"]) for r in raw(cfg, "claude-haiku")}
    tasks = em.build_tasks(cfg)
    assert len(ref) == len(tasks)
    for spec in cb.model_specs(cfg):
        reqs = cb.build_requests(cfg, spec, tasks)
        assert len(reqs) == len(tasks)
        for t, r in zip(tasks, reqs):
            sys_blocks, msgs = r["params"]["system"], r["params"]["messages"]
            assert len(sys_blocks) == 1 and len(msgs) == 1 and msgs[0]["role"] == "user"
            assert (sys_blocks[0]["text"], msgs[0]["content"]) == em.build_prompt(cfg, t)
            assert (sys_blocks[0]["text"], msgs[0]["content"]) == ref[em.task_key(spec.name, t).split("|", 1)[1]]
            assert r["custom_id"] == cb.custom_id(em.task_key(spec.name, t))


def test_request_params(jz):
    cfg = jz
    t = em.build_tasks(cfg)[:3]
    son, opus = cb.model_specs(cfg)
    for spec, mid in ((son, "claude-sonnet-5-5"), (opus, "claude-opus-5-5")):
        for r in cb.build_requests(cfg, spec, t):
            p = r["params"]
            assert p["model"] == mid == jzapi.ROLES[spec.role]
            assert not {"temperature", "top_p", "top_k"} & set(p)
            assert "cache_control" not in p["system"][0]   # the shared system prompt is < 512 tokens
    assert son.max_tokens == cfg["eval_models"]["max_tokens"] == 200
    assert cb.build_requests(cfg, son, t)[0]["params"]["thinking"] == {"type": "between_tools"}
    po = cb.build_requests(cfg, opus, t)[0]["params"]
    assert po["output_config"] == {"effort": "low"} and po["max_tokens"] == 1024 and "thinking" not in po
    cids = [cb.custom_id(em.task_key("claude-opus", x)) for x in em.build_tasks(cfg)]
    assert len(set(cids)) == len(cids) and all(len(c) <= 64 and c.isalnum() for c in cids)


def test_cache_on_long_system_prompt(jz):
    cfg = jz
    cfg["eval_models"]["system_prompt"] = "You are a helpful assistant. " * 120
    r = cb.build_requests(cfg, cb.model_specs(cfg)[0], em.build_tasks(cfg)[:1])[0]
    assert r["params"]["system"][-1]["cache_control"] == {"type": "ephemeral"}


# ----------------------------------------------------------------------------- run, rows, resume
def test_rows_schema_and_scoring(jz):
    cfg = jz
    em.run(cfg, clients={"claude-haiku": FakeLLM(), "openrouter-small": FakeLLM(), "gemma-local": FakeLLM()},
           only=["claude-haiku"])
    fake = FakeBatchClient()
    run(cfg, "full", fake, sanity_gate=False)
    ref_keys = list(raw(cfg, "claude-haiku")[0].keys())
    ref_clean_keys = list(next(r for r in raw(cfg, "claude-haiku") if r["setting"] == "clean").keys())
    n = len(em.build_tasks(cfg))
    phrases = read_json(cfg.data_dir / "phrase_translations.json", default={}) or {}
    tasks = {em.task_key("x", t).split("|", 1)[1]: t for t in em.build_tasks(cfg)}
    for name, mid in (("claude-sonnet", "claude-sonnet-5-5"), ("claude-opus", "claude-opus-5-5")):
        rows = raw(cfg, name)
        assert len(rows) == n and len({r["key"] for r in rows}) == n
        for r in rows:
            assert list(r.keys()) == (ref_clean_keys if r["setting"] == "clean" else ref_keys)
            assert r["model_id"] == mid and r["temperature"] == "api_default" and r["endpoint"] == "anthropic_batch"
            t = tasks[r["key"].split("|", 1)[1]]
            if r["setting"] != "clean":
                from src.judge import judge
                assert {k: r[k] for k in judge(t["item"], r["response"], phrases)} == judge(t["item"], r["response"], phrases)
    summ = list(csv.DictReader(open(cfg.results_dir / "models_summary.csv")))
    assert {"claude-sonnet", "claude-opus", "claude-haiku"} <= {s["model"] for s in summ}
    # spend.json and usage.csv both hold the batch costs
    spend = read_json(cfg.results_dir / "spend.json")
    assert {"claude-sonnet-5-5", "claude-opus-5-5"} <= set(spend["by_model"])
    assert spend["by_model"]["claude-opus-5-5"]["calls"] == n
    usage = list(csv.DictReader(open(cfg.root / "results" / "usage.csv")))
    assert len(usage) == 2 * n and {u["job"] for u in usage} == {"lit_rq3"}
    jz_cost = sum(float(u["cost_usd"]) for u in usage)
    lit_cost = sum(spend["by_model"][m]["usd"] for m in ("claude-sonnet-5-5", "claude-opus-5-5"))
    assert abs(jz_cost - lit_cost) < 1e-4


def test_resume_never_resubmits_or_duplicates(jz):
    cfg = jz
    fake = FakeBatchClient(statuses=["in_progress", "in_progress"])
    st = run(cfg, "sanity", fake, wait=False)          # submit both, collect nothing yet
    assert fake.n("batch_create") == 2 and not raw(cfg) and {s["status"] for s in st} == {"in_progress"}
    spend0 = read_json(cfg.results_dir / "spend.json", default={"total_usd": 0})["total_usd"]
    jzapi.api._sessions.clear()
    run(cfg, "sanity", fake)                           # a rerun polls and collects
    assert fake.n("batch_create") == 2 and len(raw(cfg)) == 40
    spend1 = read_json(cfg.results_dir / "spend.json")["total_usd"]
    usage1 = (cfg.root / "results" / "usage.csv").read_text()
    assert spend1 > spend0
    run(cfg, "sanity", fake)                           # and again: nothing new anywhere
    assert fake.n("batch_create") == 2 and len(raw(cfg)) == 40
    assert read_json(cfg.results_dir / "spend.json")["total_usd"] == spend1
    assert (cfg.root / "results" / "usage.csv").read_text() == usage1
    # a lost row (crash between batches) is restored from the saved results without a new batch
    rows = read_jsonl(cfg.results_dir / "models_raw.jsonl")
    with open(cfg.results_dir / "models_raw.jsonl", "w", encoding="utf-8") as f:
        for r in rows[1:]:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    run(cfg, "sanity", fake)
    assert fake.n("batch_create") == 2 and len(raw(cfg)) == 40 and len({r["key"] for r in raw(cfg)}) == 40
    assert read_json(cfg.results_dir / "spend.json")["total_usd"] == spend1


def test_sanity_is_20_per_model_then_full_submits_the_rest(jz):
    cfg = jz
    n = len(em.build_tasks(cfg))
    fake = FakeBatchClient()
    with pytest.raises(cb.SanityGateError):
        run(cfg, "full", fake)
    assert fake.n("batch_create") == 0
    run(cfg, "sanity", fake)
    assert sorted(len(b) for b in fake.batches.values()) == [20, 20]
    first20 = {cb.custom_id(em.task_key("claude-sonnet", t)) for t in em.build_tasks(cfg)[:20]}
    assert {r["custom_id"] for r in fake.requests_of("claude-sonnet-5-5")} == first20
    run(cfg, "full", fake)
    assert sorted(len(b) for b in fake.batches.values()) == [20, 20, n - 20, n - 20]
    assert len(raw(cfg, "claude-sonnet")) == n and len(raw(cfg, "claude-opus")) == n


def test_empty_at_max_tokens_is_not_a_row_and_blocks_full(jz):
    cfg = jz
    t0 = em.build_tasks(cfg)[0]
    fake = FakeBatchClient(empty_cids={cb.custom_id(em.task_key("claude-opus", t0))})
    st = {s["model"]: s for s in run(cfg, "sanity", fake)}
    assert st["claude-opus"]["empty_at_max_tokens"] == 1 and len(raw(cfg, "claude-opus")) == 19
    with pytest.raises(cb.SanityGateError, match="empty"):
        run(cfg, "full", fake, only=["claude-opus"])
    fake.empty_cids = set()
    run(cfg, "sanity", fake, only=["claude-opus"], round_=2, opus_max_tokens=2048)
    assert len(fake.batches["msgbatch_fake3"]) == 1 and len(raw(cfg, "claude-opus")) == 20
    with pytest.raises(cb.SanityGateError, match="below"):
        run(cfg, "full", fake, only=["claude-opus"])     # default 1024 < the 2048 the sanity round used
    run(cfg, "full", fake, only=["claude-opus"], opus_max_tokens=2048)
    assert len(raw(cfg, "claude-opus")) == len(em.build_tasks(cfg))


# ----------------------------------------------------------------------------- budgets
def test_refused_by_lit_project_cap(jz):
    cfg = jz
    write_json(cfg.results_dir / "spend.json", {"max_usd": 40, "stop_fraction": 0.95, "total_usd": 37.99,
                                                "calls": 0, "cached_calls": 0, "by_model": {}, "by_stage": {}})
    fake = FakeBatchClient()
    with pytest.raises(cb.LitBudgetExceeded):
        run(cfg, "sanity", fake)
    assert fake.n("batch_create") == 0 and not (cfg.root / "results" / "jzapi_batches").exists()


def test_lit_cap_counts_submitted_unrecorded_batches(jz):
    cfg = jz
    fake = FakeBatchClient(statuses=["in_progress"] * 4)
    run(cfg, "sanity", fake, wait=False, only=["claude-opus"])
    s = jzapi.Session("lit_rq3", repo_dir=cfg.root, client=fake)
    held = cb.lit_reserved(cfg, s)
    assert held > 0
    worst_sonnet = cb.offline_worst_case(cb.build_requests(cfg, cb.model_specs(cfg)[0], em.build_tasks(cfg)[:20]))
    b = cb.Budget.from_config(cfg)
    write_json(cfg.results_dir / "spend.json", {**b.state, "total_usd": b.hard_stop_usd - held - worst_sonnet / 2})
    with pytest.raises(cb.LitBudgetExceeded):
        run(cfg, "sanity", fake, wait=False, only=["claude-sonnet"])
    assert fake.n("batch_create") == 1


def test_refused_by_jzapi_job_cap(jz, tmp_path, monkeypatch):
    cfg = jz
    b = tmp_path / "tiny.yaml"
    b.write_text("jobs:\n  lit_rq3:\n    cap_usd: 0.001\n    approved: true\n")
    monkeypatch.setenv("JZ_BUDGETS_FILE", str(b))
    fake = FakeBatchClient()
    with pytest.raises(jzapi.BudgetExceeded):
        run(cfg, "sanity", fake)
    assert fake.n("batch_create") == 0 and not raw(cfg)


def test_unapproved_job_is_refused(jz, tmp_path, monkeypatch):
    b = tmp_path / "no.yaml"
    b.write_text("jobs:\n  lit_rq3:\n    cap_usd: 20\n    approved: false\n")
    monkeypatch.setenv("JZ_BUDGETS_FILE", str(b))
    with pytest.raises(jzapi.JobNotApproved):
        run(jz, "sanity", FakeBatchClient())


# ----------------------------------------------------------------------------- estimate, key, rule 8
def test_estimate_is_offline_and_writes_nothing(jz, capsys):
    cfg = jz
    before = sorted(p for p in cfg.root.rglob("*") if p.is_file())
    est = cb.run(cfg, "estimate")                       # no client: any network use would fail
    out = capsys.readouterr().out
    assert "WORST-CASE ESTIMATE" in out and "claude-opus-5-5" in out
    assert est[0]["n_tasks_full"] == len(em.build_tasks(cfg))
    for m in ("claude-sonnet", "claude-opus"):
        assert 0 < est[0]["sanity"][m] < est[0]["full"][m]
    assert est[0]["full"]["claude-opus"] > est[0]["full"]["claude-sonnet"]
    assert sorted(p for p in cfg.root.rglob("*") if p.is_file()) == before


def test_estimate_matches_bytes_bound(jz):
    cfg = jz
    spec = cb.model_specs(cfg)[1]
    reqs = cb.build_requests(cfg, spec, em.build_tasks(cfg)[:5])
    by_hand = sum((jzapi.api.bytes_upper_bound(r["params"]) * 4.00 + 1024 * 20.00) / 1e6 * 0.5 for r in reqs)
    assert cb.offline_worst_case(reqs) == pytest.approx(by_hand)


def test_no_key_in_logs_or_files(jz, capsys):
    cfg = jz
    run(cfg, "sanity", FakeBatchClient())
    run(cfg, "estimate", None)
    out = capsys.readouterr()
    assert FAKE_KEY not in out.out + out.err
    for p in cfg.root.rglob("*"):
        if p.is_file():
            assert FAKE_KEY not in p.read_text(encoding="utf-8", errors="ignore"), p


def test_claude_rows_never_feed_training():
    from pathlib import Path
    src = Path(cb.__file__).parent
    for f in ("train_detector.py", "build_dataset.py", "translate.py", "build_native.py", "probe.py"):
        assert "models_raw" not in (src / f).read_text(encoding="utf-8")
