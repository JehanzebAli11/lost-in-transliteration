"""RQ3 extension: Claude Sonnet 5.5 and Opus 5.5 through the Message Batches API (PLAN §4.3).

Same tasks and prompts as ``src/eval_models.py`` (its ``build_tasks``, ``build_prompt`` and
``task_key`` are reused unchanged), one Message Batch per model through JzLab's ``jzapi``
(job ``lit_rq3``), rows appended to ``results/models_raw.jsonl`` in the same schema and scored
by the same judge. Model names in the rows: ``claude-sonnet`` and ``claude-opus``.

Rule 8 (PLAN §0): Claude outputs are evaluation only. These rows go to
``results/models_raw.jsonl``, which only ``eval_models.write_summaries`` and ``analyze`` read;
nothing here may feed a training split.

Sampling (a comparability caveat for the write-up). API version 2026-10-21 rejects
temperature/top_p/top_k and jzapi refuses them, so Sonnet and Opus run at the API default with
no temperature parameter, unlike the OpenRouter runs (Haiku 5.5, GPT-5.4 nano, Gemma) at
temperature 0. Thinking is on by default on the 5.5 models and counts toward ``max_tokens``
(the bug the OpenRouter Haiku run hit, LOG 2026-11-05), so:

* Sonnet 5.5: ``thinking: {type: "between_tools"}`` (its lowest setting; with no tools, no
  thinking) and the same ``max_tokens`` as the other models (``eval_models.max_tokens``, 200).
* Opus 5.5: thinking cannot be turned off. ``output_config.effort: "low"`` keeps it small and
  ``max_tokens`` defaults to 1024 so thinking plus a 2-3 sentence summary fits. Its visible
  reply is not capped at 200 tokens the way the others are, and it may reason first.

A reply that comes back empty because it hit ``max_tokens`` is not written as a row (it would
be scored "not followed"); it is counted in the LOG, and ``--round 2`` with a larger
``--opus-max-tokens`` reruns just the missing tasks. The full run refuses to start for a model
whose latest sanity round had any.

Budgets, both checked before any submit:
* jzapi: ``lit_rq3`` cap in JzLab/budgets.yaml ($20), from results/usage.csv plus every
  unlogged batch's worst case (jzapi's own guard, inside ``batch_submit``);
* LIT: ``results/spend.json`` total + this module's unrecorded batches + the new batch's
  worst case must stay under ``budget.max_usd x stop_fraction`` ($40 x 0.95).
Worst case = every prompt token at the input rate (UTF-8 bytes of the prompt as an offline
upper bound on tokens, or the free count_tokens endpoint x1.05 with --token-estimate count)
+ max_tokens of output, at the 50% batch price. ``--mode estimate`` prints it with no network.

Resumable: a batch whose state file exists (results/jzapi_batches/lit_rq3__<name>.json) is never
resubmitted, only polled and collected; rows already in models_raw.jsonl (by ``key``) are never
written twice; jzapi logs each batch's usage once; LIT spend is recorded once per batch
(results/rq3_claude_batches.json).

Usage (on the Mac, with jzapi installed in the venv and the key in ~/lab/JzLab/.env):
    python -m src.eval_models_claude_batch --mode estimate
    python -m src.eval_models_claude_batch --mode sanity     # first 20 tasks per model
    python -m src.eval_models_claude_batch --mode full       # the rest; waits for the batches
"""
from __future__ import annotations

import argparse
import hashlib
import statistics
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .budget import Budget, BudgetExceeded, estimate_tokens
from .common import Config, JsonlAppender, load_config, log_event, now_iso, read_json, read_jsonl, write_json
from .eval_models import build_prompt, build_tasks, task_key, write_summaries
from .judge import judge, summary_ok

JOB = "lit_rq3"
STAGE = "eval_models"            # spend.json by_stage label, shared with the other RQ3 models
ENDPOINT = "anthropic_batch"     # the "endpoint" value in models_raw.jsonl rows
SANITY_N = 20                    # same as `make eval-models-sanity` (eval_models --limit 20)
CACHE_MIN_TOKENS = 512           # below this the API does not cache a prefix on these models
LEDGER = "rq3_claude_batches.json"
TEMPERATURE_APPLIED = "api_default"
OPUS_MAX_TOKENS = 1024
OPUS_EFFORT = "low"


class LitBudgetExceeded(BudgetExceeded):
    """LIT's own project cap (spend.json) would be passed."""


class SanityGateError(RuntimeError):
    """The full run needs a clean sanity batch for the model first."""


@dataclass
class ModelSpec:
    name: str                    # model name in models_raw.jsonl
    role: str                    # jzapi role
    max_tokens: int
    extra: dict = field(default_factory=dict)   # extra Messages params (thinking, output_config)

    @property
    def model_id(self) -> str:
        import jzapi
        return jzapi.resolve(self.role)


def model_specs(cfg: Config, *, opus_max_tokens: int = OPUS_MAX_TOKENS, opus_effort: str = OPUS_EFFORT,
                only: list[str] | None = None) -> list[ModelSpec]:
    mt = int(cfg["eval_models"]["max_tokens"])
    specs = [
        ModelSpec("claude-sonnet", "sonnet", mt, {"thinking": {"type": "between_tools"}}),
        ModelSpec("claude-opus", "opus", int(opus_max_tokens), {"output_config": {"effort": opus_effort}}),
    ]
    if only:
        unknown = set(only) - {s.name for s in specs}
        if unknown:
            raise ValueError(f"unknown model(s) {sorted(unknown)}; have claude-sonnet, claude-opus")
        specs = [s for s in specs if s.name in only]
    return specs


def custom_id(key: str) -> str:
    """Batch custom_ids allow only [A-Za-z0-9_-]{1,64}; task keys contain '|' and are longer."""
    return "t" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:40]


def batch_name(model_name: str, mode: str, round_: int = 1) -> str:
    return f"rq3_{model_name}_{mode}" + (f"_r{round_}" if round_ > 1 else "")


def select_tasks(cfg: Config, mode: str) -> list[dict]:
    tasks = build_tasks(cfg)
    if not tasks:
        raise RuntimeError("no tasks: run the dataset and translation stages first")
    return tasks[:SANITY_N] if mode == "sanity" else tasks


def build_requests(cfg: Config, spec: ModelSpec, tasks: list[dict]) -> list[dict]:
    """One batch request per task, with exactly eval_models' system and user prompts."""
    import jzapi
    reqs = []
    for t in tasks:
        system, user = build_prompt(cfg, t)
        reqs.append(jzapi.request(custom_id(task_key(spec.name, t)), model=spec.role, system=system,
                                  cache_system=estimate_tokens(system) >= CACHE_MIN_TOKENS, user=user,
                                  max_tokens=spec.max_tokens, **spec.extra))
    return reqs


def offline_worst_case(reqs: list[dict]) -> float:
    """Batch worst case from the UTF-8 byte bound on prompt tokens (no network)."""
    from jzapi import api, models
    return sum(models.worst_case_usd(r["params"]["model"], prompt_tokens=api.bytes_upper_bound(r["params"]),
                                     max_tokens=int(r["params"]["max_tokens"]),
                                     cache_ttl=api.cache_ttl_of(r["params"]), batch=True) for r in reqs)


# ----------------------------------------------------------------------------- LIT-side ledger
def _ledger_path(cfg: Config) -> Path:
    return cfg.results_dir / LEDGER


def read_ledger(cfg: Config) -> dict:
    return read_json(_ledger_path(cfg), default={}) or {}


def write_ledger(cfg: Config, ledger: dict) -> None:
    write_json(_ledger_path(cfg), ledger)


def _batch_dir(session) -> Path:
    return session.batch_dir


def _state(session, name: str) -> dict | None:
    p = _batch_dir(session) / f"{JOB}__{name}.json"
    return read_json(p, default=None)


def lit_reserved(cfg: Config, session) -> float:
    """This module's batches not yet in spend.json: actual cost if jzapi has logged them, else
    their worst case."""
    ledger = read_ledger(cfg)
    total = 0.0
    d = _batch_dir(session)
    if not d.exists():
        return 0.0
    for p in sorted(d.glob(f"{JOB}__rq3_claude-*.json")):
        if p.name.endswith((".failed.json", ".results.json")):
            continue
        st = read_json(p, default=None) or {}
        name = st.get("name")
        if not name or (ledger.get(name) or {}).get("lit_spend_recorded"):
            continue
        total += float(st.get("actual_cost_usd") if st.get("status") == "logged"
                       else st.get("worst_case_usd") or 0.0)
    return total


def check_lit_cap(budget: Budget, reserved: float, worst: float, what: str) -> None:
    if budget.total + reserved + worst > budget.hard_stop_usd + 1e-12:
        raise LitBudgetExceeded(
            f"LIT cap: {what} worst case ${worst:.4f} + spent ${budget.total:.4f} + reserved ${reserved:.4f} "
            f"would pass {budget.stop_fraction:.0%} of ${budget.max_usd:.2f} (spend.json)")


def _record_lit_spend(budget: Budget, model_id: str, n_calls: int, input_tokens: int, output_tokens: int,
                      usd: float) -> None:
    """One atomic spend.json update for a whole batch (Budget.record does one per call)."""
    with budget._lock:
        s = budget.state
        s["calls"] += n_calls
        s["total_usd"] = round(s["total_usd"] + usd, 8)
        m = s["by_model"].setdefault(model_id, {"calls": 0, "input_tokens": 0, "output_tokens": 0, "usd": 0.0})
        m["calls"] += n_calls
        m["input_tokens"] += int(input_tokens)
        m["output_tokens"] += int(output_tokens)
        m["usd"] = round(m["usd"] + usd, 8)
        st = s["by_stage"].setdefault(STAGE, {"calls": 0, "usd": 0.0})
        st["calls"] += n_calls
        st["usd"] = round(st["usd"] + usd, 8)
        s["updated"] = now_iso()
        write_json(budget.spend_path, s)


# ----------------------------------------------------------------------------- rows
def make_row(cfg: Config, spec: ModelSpec, task: dict, text: str | None, usage: dict, model_id: str,
             phrases: dict) -> dict:
    """A models_raw.jsonl row with the same fields, in the same order, as eval_models.run_model."""
    it = task["item"]
    system, user = build_prompt(cfg, task)
    text = text or ""
    row = {"model": spec.name, "model_id": model_id, "endpoint": ENDPOINT,
           "key": task_key(spec.name, task), "setting": task["setting"], "item_id": it["id"],
           "source_id": it["source_id"], "lang": it["lang"],
           "lang_condition": "native" if it.get("condition") == "native" else it["lang"],
           "condition": it.get("condition", "main"), "carrier_id": task["carrier"]["carrier_id"],
           "carrier_lang": task["carrier"]["lang"], "goal": it.get("goal"), "style": it.get("style"),
           "split": it.get("split"), "position": it.get("position"),
           "system_prompt": system, "user_prompt": user, "response": text,
           "input_tokens": int(usage.get("input_tokens", 0)) + int(usage.get("cache_creation_input_tokens", 0))
           + int(usage.get("cache_read_input_tokens", 0)),
           "output_tokens": int(usage.get("output_tokens", 0)), "cached": False,
           "temperature": TEMPERATURE_APPLIED, "max_tokens": spec.max_tokens}
    if task["setting"] == "clean":
        row.update({"success": None, "summary_ok": summary_ok(text),
                    "response_script": judge({}, text)["response_script"]})
    else:
        row.update(judge(it, text, phrases))
    return row


def collect(cfg: Config, session, spec: ModelSpec, name: str, all_tasks: list[dict]) -> dict:
    """Results of an ended batch -> rows (missing keys only), LIT spend (once), stats in the ledger."""
    items = session.batch_results(name)          # jzapi logs usage.csv once per batch
    by_cid = {custom_id(task_key(spec.name, t)): t for t in all_tasks}
    raw_path = cfg.results_dir / "models_raw.jsonl"
    done = {r["key"] for r in read_jsonl(raw_path)}
    out = JsonlAppender(raw_path)
    phrases = read_json(cfg.data_dir / "phrase_translations.json", default={}) or {}
    types = Counter(it.type for it in items)
    stops = Counter(it.stop_reason for it in items if it.type == "succeeded")
    empty_at_max = sum(1 for it in items if it.type == "succeeded" and not (it.text or "").strip()
                       and it.stop_reason == "max_tokens")
    out_toks = [int(it.usage.get("output_tokens", 0)) for it in items if it.type == "succeeded"]
    new = unknown = 0
    for it in items:
        if it.type != "succeeded":
            continue
        task = by_cid.get(it.custom_id)
        if task is None:
            unknown += 1
            continue
        if task_key(spec.name, task) in done:
            continue
        if not (it.text or "").strip() and it.stop_reason == "max_tokens":
            continue  # cut off while thinking: not an observation; a later --round fills it
        out.append(make_row(cfg, spec, task, it.text, it.usage, it.model_id or spec.model_id, phrases))
        done.add(task_key(spec.name, task))
        new += 1
    ledger = read_ledger(cfg)
    entry = ledger.setdefault(name, {"model": spec.name})
    if not entry.get("lit_spend_recorded"):
        billed = [it for it in items if it.type == "succeeded"]
        usd = sum(it.cost_usd for it in billed)
        tin = sum(int(it.usage.get("input_tokens", 0)) + int(it.usage.get("cache_creation_input_tokens", 0))
                  + int(it.usage.get("cache_read_input_tokens", 0)) for it in billed)
        tout = sum(int(it.usage.get("output_tokens", 0)) for it in billed)
        _record_lit_spend(Budget.from_config(cfg), spec.model_id, len(billed), tin, tout, usd)
        entry.update(lit_spend_recorded=True, lit_spend_usd=round(usd, 6))
    stats = {"model": spec.name, "batch": name, "results": dict(types), "stop_reasons": dict(stops),
             "empty_at_max_tokens": empty_at_max, "new_rows": new, "unmatched": unknown,
             "output_tokens_median": statistics.median(out_toks) if out_toks else None,
             "output_tokens_max": max(out_toks) if out_toks else None, "max_tokens": spec.max_tokens}
    entry.update(stats=stats, collected=now_iso())
    write_ledger(cfg, ledger)
    log_event(cfg, f"{spec.name}: {stats}", stage="eval_models_claude_batch")
    return stats


# ----------------------------------------------------------------------------- run
def _session(cfg: Config, client, token_estimate: str):
    import jzapi
    return jzapi.Session(JOB, repo_dir=cfg.root, client=client, token_estimate=token_estimate)


def _sanity_gate(cfg: Config, spec: ModelSpec) -> None:
    """The latest collected sanity round of this model must have no reply left empty at max_tokens,
    and the full run must not use a smaller max_tokens than that round."""
    prefix = batch_name(spec.name, "sanity")
    rounds = {1 if n == prefix else int(n[len(prefix) + 2:]): e for n, e in read_ledger(cfg).items()
              if (n == prefix or (n.startswith(prefix + "_r") and n[len(prefix) + 2:].isdigit()))
              and e.get("stats")}
    if not rounds:
        raise SanityGateError(f"{spec.name}: run `make rq3-claude-sanity` and read its rows first "
                              f"(or pass --no-sanity-gate)")
    stats = rounds[max(rounds)]["stats"]
    if stats.get("empty_at_max_tokens"):
        raise SanityGateError(f"{spec.name}: {stats['empty_at_max_tokens']} sanity replies were empty at "
                              f"max_tokens={stats.get('max_tokens')} (no rows written for them); rerun the "
                              f"sanity batch with a larger --opus-max-tokens and --round 2 first")
    if spec.max_tokens < int(stats.get("max_tokens") or 0):
        raise SanityGateError(f"{spec.name}: full run max_tokens {spec.max_tokens} is below the sanity "
                              f"round's {stats.get('max_tokens')}; pass the same --opus-max-tokens")


def estimate(cfg: Config, specs: list[ModelSpec], *, session=None) -> dict:
    """Worst case per model for the sanity and full runs, offline. Prints a table; no network."""
    session = session or _session(cfg, None, "bytes")
    budget = Budget.from_config(cfg)
    all_tasks = select_tasks(cfg, "full")
    done = {r["key"] for r in read_jsonl(cfg.results_dir / "models_raw.jsonl")}
    out = {"sanity": {}, "full": {}, "full_remaining": {}}
    for spec in specs:
        out["sanity"][spec.name] = offline_worst_case(build_requests(cfg, spec, all_tasks[:SANITY_N]))
        out["full"][spec.name] = offline_worst_case(build_requests(cfg, spec, all_tasks))
        todo = [t for t in all_tasks if task_key(spec.name, t) not in done]
        out["full_remaining"][spec.name] = offline_worst_case(build_requests(cfg, spec, todo)) if todo else 0.0
    n_full = len(all_tasks)
    jz = session.budget
    print(f"RQ3 Claude batch: WORST-CASE ESTIMATE (offline, UTF-8 bytes as prompt tokens, max_tokens of "
          f"output, batch price x0.5). Not spend.")
    print(f"{'model':15s} {'model_id':18s} {'max_tok':>7s} {'sanity (' + str(SANITY_N) + ')':>13s} "
          f"{'full (' + str(n_full) + ')':>12s} {'still to run':>13s}")
    for spec in specs:
        print(f"{spec.name:15s} {spec.model_id:18s} {spec.max_tokens:7d} ${out['sanity'][spec.name]:12.4f} "
              f"${out['full'][spec.name]:11.4f} ${out['full_remaining'][spec.name]:12.4f}")
    tot_s, tot_f = sum(out["sanity"].values()), sum(out["full"].values())
    print(f"{'total':15s} {'':18s} {'':7s} ${tot_s:12.4f} ${tot_f:11.4f} ${sum(out['full_remaining'].values()):12.4f}")
    jz_spent, jz_res = jz.spent_so_far(), jz.reserved_batches()
    print(f"jzapi {JOB}: cap ${jz.cap_usd:.2f}, spent ${jz_spent:.4f}, reserved ${jz_res:.4f}, "
          f"headroom ${jz.cap_usd - jz_spent - jz_res:.4f}")
    print(f"LIT spend.json: ${budget.total:.4f} of ${budget.max_usd:.2f} (stop at ${budget.hard_stop_usd:.2f}); "
          f"headroom ${budget.hard_stop_usd - budget.total - lit_reserved(cfg, session):.4f}")
    out.update(n_tasks_full=n_full, jzapi_cap=jz.cap_usd, jzapi_spent=jz_spent, jzapi_reserved=jz_res,
               lit_total=budget.total, lit_hard_stop=budget.hard_stop_usd)
    return out


def run(cfg: Config, mode: str, *, client=None, only: list[str] | None = None, token_estimate: str = "bytes",
        wait: bool = True, interval: float = 60.0, round_: int = 1, sanity_gate: bool = True,
        opus_max_tokens: int = OPUS_MAX_TOKENS, opus_effort: str = OPUS_EFFORT, sleep=None) -> list[dict]:
    specs = model_specs(cfg, opus_max_tokens=opus_max_tokens, opus_effort=opus_effort, only=only)
    session = _session(cfg, client, token_estimate)
    if mode == "estimate":
        return [estimate(cfg, specs, session=session)]
    if mode not in ("sanity", "full"):
        raise ValueError("mode must be estimate, sanity or full")
    all_tasks = select_tasks(cfg, "full")
    tasks = select_tasks(cfg, mode)
    budget = Budget.from_config(cfg)
    done = {r["key"] for r in read_jsonl(cfg.results_dir / "models_raw.jsonl")}
    pending = []
    for spec in specs:
        name = batch_name(spec.name, mode, round_)
        if _state(session, name) is not None:      # never resubmit: poll and collect it
            pending.append((spec, name))
            continue
        if mode == "full" and sanity_gate:
            _sanity_gate(cfg, spec)
        todo = [t for t in tasks if task_key(spec.name, t) not in done]
        if not todo:
            log_event(cfg, f"{spec.name}: all {len(tasks)} {mode} tasks already in models_raw.jsonl",
                      stage="eval_models_claude_batch")
            continue
        reqs = build_requests(cfg, spec, todo)
        worst = offline_worst_case(reqs)
        reserved = lit_reserved(cfg, session)
        print(f"{spec.name} ({spec.model_id}): {len(reqs)} requests, worst case ${worst:.4f} "
              f"(offline estimate); jzapi {JOB} headroom ${session.budget.remaining():.4f}; LIT headroom "
              f"${budget.hard_stop_usd - budget.total - reserved:.4f}")
        check_lit_cap(budget, reserved, worst, f"batch {name}")
        st = session.batch_submit(reqs, name=name)  # jzapi checks the lit_rq3 cap before sending
        ledger = read_ledger(cfg)
        ledger.setdefault(name, {"model": spec.name}).update(
            batch_id=st.get("batch_id"), n_requests=len(reqs), submitted=now_iso(), lit_spend_recorded=False,
            worst_case_offline_usd=round(worst, 6), max_tokens=spec.max_tokens, extra=spec.extra)
        write_ledger(cfg, ledger)
        log_event(cfg, f"{spec.name}: submitted batch {name} ({st.get('batch_id')}, {len(reqs)} requests, "
                       f"worst case ${worst:.4f}); sampling at the API default (API 2026-10-21 rejects "
                       f"temperature), params {spec.extra}, max_tokens {spec.max_tokens}",
                  stage="eval_models_claude_batch")
        pending.append((spec, name))
    stats = []
    for spec, name in pending:
        kw = {"wait": wait, "interval": interval}
        if sleep is not None:
            kw["sleep"] = sleep
        st = session.batch_poll(name, **kw)
        if st.get("status") not in ("ended", "logged"):
            print(f"{spec.name}: batch {name} is {st.get('processing_status')}; rerun to collect")
            stats.append({"model": spec.name, "batch": name, "status": st.get("processing_status")})
            continue
        stats.append(collect(cfg, session, spec, name, all_tasks))
    if any("new_rows" in s for s in stats):
        write_summaries(cfg)
    return stats


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["estimate", "sanity", "full"], required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--profile", default=None)
    ap.add_argument("--only", nargs="+", help="claude-sonnet and/or claude-opus")
    ap.add_argument("--token-estimate", choices=["bytes", "count"], default="bytes",
                    help="jzapi's pre-submit worst case: bytes (offline upper bound, default) or the free "
                         "count_tokens endpoint x1.05")
    ap.add_argument("--no-wait", action="store_true", help="submit and return; rerun to poll and collect")
    ap.add_argument("--poll-interval", type=float, default=60.0)
    ap.add_argument("--round", type=int, default=1, dest="round_",
                    help="a new batch for tasks still missing (errored/expired results)")
    ap.add_argument("--opus-max-tokens", type=int, default=OPUS_MAX_TOKENS)
    ap.add_argument("--opus-effort", default=OPUS_EFFORT, choices=["low", "medium", "high", "xhigh", "max"])
    ap.add_argument("--no-sanity-gate", action="store_true")
    a = ap.parse_args(argv)
    cfg = load_config(a.config, a.profile)
    stats = run(cfg, a.mode, only=a.only, token_estimate=a.token_estimate, wait=not a.no_wait,
                interval=a.poll_interval, round_=a.round_, sanity_gate=not a.no_sanity_gate,
                opus_max_tokens=a.opus_max_tokens, opus_effort=a.opus_effort)
    if a.mode != "estimate":
        for s in stats:
            print(s)


if __name__ == "__main__":
    main()
