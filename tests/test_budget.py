import pytest

from src.budget import Budget, BudgetExceeded, DiskCache, Price, PriceMissing, estimate_tokens, price_from_cfg
from src.common import read_json
from src.llm import ChatResult, ManagedClient


class Counting:
    def __init__(self, text="ok", fail_times=0):
        self.n = 0
        self.text = text
        self.fail_times = fail_times

    def chat(self, **kw):
        self.n += 1
        if self.n <= self.fail_times:
            raise ConnectionError("transient")
        return ChatResult(text=self.text, input_tokens=1000, output_tokens=200, model=kw["model"])


def test_price_cost_and_todo():
    p = Price(1.0, 5.0)
    assert p.cost(1_000_000, 0) == pytest.approx(1.0)
    assert p.cost(0, 200_000) == pytest.approx(1.0)
    assert price_from_cfg({"input_usd_per_mtok": None, "output_usd_per_mtok": 1}) is None
    assert price_from_cfg({"input_usd_per_mtok": "TODO_VERIFY", "output_usd_per_mtok": 1}) is None
    assert price_from_cfg({}, free=True).cost(10**9, 10**9) == 0


def test_estimate_tokens_is_conservative_for_urdu():
    assert estimate_tokens("") == 0
    assert estimate_tokens("کتاب" * 10) > estimate_tokens("book" * 10)


def test_budget_hard_stop_and_persistence(tmp_path):
    path = tmp_path / "spend.json"
    b = Budget(1.0, path, 0.95)
    b.record(model="m", stage="s", input_tokens=10, output_tokens=10, cost_usd=0.90)
    b.check(0.04)  # 0.94 <= 0.95
    with pytest.raises(BudgetExceeded):
        b.check(0.06)
    # survives a restart
    b2 = Budget(1.0, path, 0.95)
    assert b2.total == pytest.approx(0.90)
    with pytest.raises(BudgetExceeded):
        b2.check(0.10)
    s = read_json(path)
    assert s["by_model"]["m"]["calls"] == 1 and s["by_stage"]["s"]["usd"] == pytest.approx(0.9)


def test_disk_cache_roundtrip(tmp_path):
    c = DiskCache(tmp_path)
    k = DiskCache.key("a", 1, {"x": [1, 2]})
    assert c.get(k) is None
    c.set(k, {"text": "hi"})
    assert c.get(k) == {"text": "hi"}
    assert DiskCache.key("a", 1, {"x": [1, 2]}) == k


def test_managed_client_caches_and_charges(tmp_path):
    inner = Counting()
    budget = Budget(40, tmp_path / "spend.json")
    mc = ManagedClient(inner, endpoint="anthropic", stage="t", price=Price(1.0, 5.0), budget=budget,
                       cache=DiskCache(tmp_path / "c"), sleep=lambda s: None)
    r1 = mc.chat(model="m", system="s", user="u", temperature=0, max_tokens=50)
    r2 = mc.chat(model="m", system="s", user="u", temperature=0, max_tokens=50)
    assert inner.n == 1 and not r1.cached and r2.cached and r2.text == "ok"
    assert budget.total == pytest.approx(1000 / 1e6 + 200 * 5 / 1e6)
    assert read_json(tmp_path / "spend.json")["cached_calls"] == 1


def test_managed_client_refuses_without_price(tmp_path):
    mc = ManagedClient(Counting(), endpoint="openrouter", stage="t", price=None,
                       budget=Budget(40, tmp_path / "s.json"), cache=None)
    with pytest.raises(PriceMissing):
        mc.chat(model="m", system=None, user="u", temperature=0, max_tokens=5)


def test_managed_client_blocks_before_calling_when_over_budget(tmp_path):
    inner = Counting()
    budget = Budget(0.001, tmp_path / "s.json")
    mc = ManagedClient(inner, endpoint="anthropic", stage="t", price=Price(1000.0, 1000.0),
                       budget=budget, cache=None)
    with pytest.raises(BudgetExceeded):
        mc.chat(model="m", system=None, user="hello " * 50, temperature=0, max_tokens=200)
    assert inner.n == 0


def test_managed_client_retries_transient(tmp_path):
    inner = Counting(fail_times=2)
    mc = ManagedClient(inner, endpoint="local", stage="t", price=Price(0, 0), budget=None,
                       cache=None, sleep=lambda s: None)
    assert mc.chat(model="m", system=None, user="u", temperature=0, max_tokens=5).text == "ok"
    assert inner.n == 3


def test_non_retryable_errors_fail_fast(tmp_path):
    class Bad:
        n = 0

        def chat(self, **kw):
            Bad.n += 1
            raise TypeError("unexpected keyword argument")

    class E400(Exception):
        status_code = 400

    class E429(Exception):
        status_code = 429
    from src.llm import is_retryable
    assert not is_retryable(E400()) and is_retryable(E429()) and is_retryable(ConnectionError())
    mc = ManagedClient(Bad(), endpoint="local", stage="t", price=Price(0, 0), budget=None, cache=None,
                       sleep=lambda s: None)
    with pytest.raises(TypeError):
        mc.chat(model="m", system=None, user="u", temperature=0, max_tokens=5)
    assert Bad.n == 1
