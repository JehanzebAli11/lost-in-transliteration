"""RQ3 at one shared max_tokens (owner decision 2026-11-09, option A): offline worst-case estimate.

The first RQ3 run used different output caps (Haiku, GPT-5.4 nano and Gemma 200 via OpenRouter,
Sonnet 400 and Opus 1024 via the Batch API), and replies hit the cap far more often in Urdu script
(REVIEW.md issue 1). The fix reruns every model whose rows were made at another cap at 1024:

* Claude Haiku / Sonnet (Opus is already at 1024): ``src.eval_models_claude_batch`` through jzapi
  (job ``lit_rq3``), base config (LIT cap $40 x 0.95);
* GPT-5.4 nano and Gemma 4 31B: ``src.eval_models --profile rq3_uniform --max-tokens 1024`` through
  OpenRouter (needs OPENROUTER_API_KEY in the environment or the repo's .env).

``python -m src.rq3_uniform estimate`` prints, with no network and no file written, the worst case
of the remaining work per model (sanity = the first 20 tasks; full = every task still to run), and
the headroom left under both caps: jzapi ``lit_rq3`` (usage.csv) and LIT's spend.json. Worst case:
Claude = jzapi's UTF-8-byte bound on prompt tokens + max_tokens of output at the 50% batch price;
OpenRouter = LIT's conservative token estimate (``budget.estimate_tokens``) + max_tokens of output at
the configured prices. Not spend.
"""
from __future__ import annotations

import argparse
from collections import Counter

from . import eval_models as em
from . import eval_models_claude_batch as cb
from .budget import Budget, PriceMissing, estimate_tokens, price_from_cfg
from .common import Config, load_config

CLAUDE = ("claude-haiku", "claude-sonnet", "claude-opus")
OPENROUTER = ("openrouter-small", "gemma")
PROFILE = "rq3_uniform"
MAX_TOKENS = cb.UNIFORM_MAX_TOKENS


def openrouter_worst_case(cfg: Config, entry: dict, tasks: list[dict], max_tokens: int) -> float:
    """What ManagedClient's pre-call budget check would sum over these tasks."""
    price = price_from_cfg(entry, free=bool(cfg["endpoints"][entry["endpoint"]].get("free", False)))
    if price is None:
        raise PriceMissing(f"{entry['name']}: no verified price in profile {PROFILE}")
    total = 0.0
    for t in tasks:
        system, user = em.build_prompt(cfg, t)
        total += price.cost(estimate_tokens(system + user), max_tokens)
    return total


def openrouter_spent(budget: Budget) -> float:
    """spend.json cost of OpenRouter models (their ids are provider/model; jzapi's Claude ids are not)."""
    return sum(float(v.get("usd", 0.0)) for k, v in budget.state.get("by_model", {}).items() if "/" in k)


def _rows_now(rows) -> str:
    c = Counter((r.get("endpoint"), r.get("max_tokens")) for r in rows)
    return ", ".join(f"{n} @{mt} {ep}" for (ep, mt), n in sorted(c.items(), key=str)) or "none"


def estimate(base_cfg: Config, or_cfg: Config, *, max_tokens: int = MAX_TOKENS, session=None) -> dict:
    """Worst case of the remaining uniform-max_tokens work, and both headrooms. Offline."""
    session = session or cb._session(base_cfg, None, "bytes")
    all_tasks = cb.select_tasks(base_cfg, "full")
    sanity_tasks = all_tasks[:cb.SANITY_N]
    lines, out = [], {"models": {}}
    specs = cb.model_specs(base_cfg, only=list(CLAUDE), haiku_max_tokens=max_tokens,
                           sonnet_max_tokens=max_tokens, opus_max_tokens=max_tokens)
    for spec in specs:
        todo = cb.todo_tasks(base_cfg, spec, all_tasks)
        todo_s = cb.todo_tasks(base_cfg, spec, sanity_tasks)
        w_s = cb.offline_worst_case(cb.build_requests(base_cfg, spec, todo_s)) if todo_s else 0.0
        w_f = cb.offline_worst_case(cb.build_requests(base_cfg, spec, todo)) if todo else 0.0
        out["models"][spec.name] = {"route": "anthropic_batch (jzapi lit_rq3)", "model_id": spec.model_id,
                                    "n_sanity": len(todo_s), "sanity_usd": w_s, "n_todo": len(todo),
                                    "full_usd": w_f,
                                    "rows_now": _rows_now(cb.model_rows(base_cfg, spec.name).values())}
    raw = em.read_jsonl(or_cfg.results_dir / "models_raw.jsonl")
    for entry in em.model_entries(or_cfg, list(OPENROUTER)):
        name = entry["name"]
        done = em.rerun_done(or_cfg, name, max_tokens)
        todo = [t for t in all_tasks if em.task_key(name, t) not in done]
        todo_s = [t for t in sanity_tasks if em.task_key(name, t) not in done]
        out["models"][name] = {"route": f"openrouter (profile {PROFILE})", "model_id": entry["model"],
                               "n_sanity": len(todo_s),
                               "sanity_usd": openrouter_worst_case(or_cfg, entry, todo_s, max_tokens),
                               "n_todo": len(todo), "full_usd": openrouter_worst_case(or_cfg, entry, todo, max_tokens),
                               "rows_now": _rows_now([r for r in raw if r.get("model") == name])}
    m = out["models"]
    claude_w = sum(m[k]["full_usd"] for k in CLAUDE if k in m)
    or_w = sum(m[k]["full_usd"] for k in OPENROUTER if k in m)
    jz = session.budget
    jz_spent, jz_res = jz.spent_so_far(), jz.reserved_batches()
    jz_head = jz.cap_usd - jz_spent - jz_res
    lit = Budget.from_config(base_cfg)
    lit_res = cb.lit_reserved(base_cfg, session)
    lit_head = lit.hard_stop_usd - lit.total - lit_res
    or_budget = Budget.from_config(or_cfg)
    or_limit = (or_cfg.get("budget") or {}).get("openrouter_key_limit_usd")
    or_spent = openrouter_spent(or_budget)
    out.update(max_tokens=max_tokens, n_tasks_per_model=len(all_tasks), claude_worst_usd=claude_w,
               openrouter_worst_usd=or_w, jzapi_cap=jz.cap_usd, jzapi_spent=jz_spent, jzapi_reserved=jz_res,
               jzapi_headroom=jz_head, jzapi_headroom_after=jz_head - claude_w, lit_cap=lit.max_usd,
               lit_hard_stop=lit.hard_stop_usd, lit_spent=lit.total, lit_reserved=lit_res, lit_headroom=lit_head,
               lit_headroom_after=lit_head - claude_w - or_w, openrouter_spent=or_spent,
               openrouter_key_limit=or_limit)
    p = lines.append
    p(f"RQ3 at one shared max_tokens = {max_tokens}: WORST-CASE ESTIMATE (offline; not spend). "
      f"{len(all_tasks)} tasks per model; sanity = first {cb.SANITY_N}.")
    p(f"{'model':17s} {'model_id':22s} {'route':36s} {'sanity':>14s} {'still to run':>18s}  rows now")
    for name, d in m.items():
        p(f"{name:17s} {d['model_id']:22s} {d['route']:36s} {d['n_sanity']:3d} ${d['sanity_usd']:8.4f} "
          f"{d['n_todo']:5d} ${d['full_usd']:9.4f}  {d['rows_now']}")
    p(f"{'total':77s} ${sum(d['sanity_usd'] for d in m.values()):8.4f} "
      f"      ${claude_w + or_w:9.4f}")
    p(f"jzapi lit_rq3 (Claude only, results/usage.csv): cap ${jz.cap_usd:.2f}, spent ${jz_spent:.4f}, reserved "
      f"${jz_res:.4f} -> headroom ${jz_head:.4f}; after the Claude worst case ${claude_w:.4f}: ${jz_head - claude_w:.4f}")
    p(f"LIT spend.json (all paid calls): ${lit.total:.4f} of ${lit.max_usd:.2f} (hard stop ${lit.hard_stop_usd:.2f}), "
      f"reserved ${lit_res:.4f} -> headroom ${lit_head:.4f}; after the worst case ${claude_w + or_w:.4f}: "
      f"${lit_head - claude_w - or_w:.4f}")
    if or_limit is not None:
        p(f"OpenRouter key (info, not enforced here): spend.json shows ${or_spent:.4f} on OpenRouter models, key "
          f"limit ${float(or_limit):.2f} (config) -> after the OpenRouter worst case ${or_w:.4f}: "
          f"${float(or_limit) - or_spent - or_w:.4f}")
    ok = jz_head - claude_w >= 0 and lit_head - claude_w - or_w >= 0
    p("Both caps hold at the worst case." if ok else "WARNING: the worst case does not fit under a cap; "
      "the runs' own guards will stop before passing it.")
    out["fits"] = ok
    print("\n".join(lines))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["estimate"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    a = ap.parse_args(argv)
    estimate(load_config(a.config, None), load_config(a.config, PROFILE), max_tokens=a.max_tokens)


if __name__ == "__main__":
    main()
