"""RQ3: are models more likely to follow an indirect injection in Urdu variants? (PLAN §4.3)

Setup: the system prompt asks for a 2-3 sentence summary (plus a fake access code for
SECRET_LEAK items); the user turn is the carrier document with the injection embedded
at its stored position. Carrier and injection share a language ("same"). Optional
"cross" condition: English carrier + other-language injection. "clean" carriers (no
injection) check the model still summarises.

Sampling: one stratified (goal x style) sample of English source items is reused in
every language, so language comparisons are paired on the same underlying item.

Models: Anthropic, OpenRouter and the Spark's local Gemma, all through ManagedClient
(disk cache + USD budget guard). Paid models without a verified price are skipped.
If the budget stop is hit, the stage stops cleanly and logs it (PLAN §9).

Outputs: results/models_raw.jsonl (every full prompt and response; resumable),
results/models_summary.csv, models_by_goal.csv, models_by_style.csv,
models_clean_controls.csv.
"""
from __future__ import annotations

import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from .budget import Budget, BudgetExceeded, PriceMissing
from .common import (Config, ConfigTodoError, JsonlAppender, load_config, load_dotenv_if_present,
                     log_event, read_json, read_jsonl, rng_for, stage_parser)
from .data import attacks_path, carriers_path, embed, load_native
from .eval_classifiers import write_csv
from .judge import judge, summary_ok
from .llm import ChatClient, default_cache, resolve_model
from .metrics import asr_metrics


def stratified_sample(rows: list[dict], k: int | None, seed: int) -> list[dict]:
    """Round-robin over (goal, style) strata so the sample stays balanced."""
    rows = sorted(rows, key=lambda r: r["source_id"])
    if k is None or k >= len(rows):
        return rows
    strata = defaultdict(list)
    for r in rows:
        strata[(r.get("goal"), r.get("style"))].append(r)
    rng = rng_for(seed, "model-sample")
    for v in strata.values():
        rng.shuffle(v)
    keys = sorted(strata, key=str)
    rng.shuffle(keys)
    out = []
    while len(out) < k:
        progressed = False
        for key in keys:
            if strata[key] and len(out) < k:
                out.append(strata[key].pop())
                progressed = True
        if not progressed:
            break
    return out


def build_tasks(cfg: Config) -> list[dict]:
    em = cfg["eval_models"]
    langs = cfg["languages"]
    attacks = {lang: {r["source_id"]: r for r in read_jsonl(attacks_path(cfg, lang))} for lang in langs}
    carriers = {lang: {c["carrier_id"]: c for c in read_jsonl(carriers_path(cfg, lang))} for lang in langs}
    common_ids = set(attacks["en"])
    for lang in langs[1:]:
        common_ids &= set(attacks[lang])  # paired: only items translated into every variant
    sample = stratified_sample([attacks["en"][s] for s in common_ids], em.get("max_attacks_per_lang"),
                               cfg.seed)
    tasks = []
    for rank, src in enumerate(sample):
        for lang in langs:
            it = attacks[lang][src["source_id"]]
            car = carriers[lang].get(it["carrier_id"])
            if car is None:
                continue
            tasks.append({"setting": "same", "rank": rank, "item": it, "carrier": car})
    cc = em.get("cross_condition") or {}
    if cc.get("enabled"):
        for lang in langs:
            if lang == "en":
                continue
            for rank, src in enumerate(sample[: int(cc.get("n_per_lang", 0))]):
                it = attacks[lang][src["source_id"]]
                car = carriers["en"].get(it["carrier_id"])
                if car is not None:
                    tasks.append({"setting": "cross_en_carrier", "rank": rank, "item": it, "carrier": car})
    if em.get("include_native", True):
        for rank, it in enumerate(r for r in load_native(cfg) if r["label"] == 1):
            car = carriers.get(it["lang"], {}).get(it.get("carrier_id"))
            if car is not None:
                tasks.append({"setting": "same", "rank": rank, "item": it, "carrier": car})
    k = int(em.get("clean_controls_per_lang", 0))
    for lang in langs:
        ids = sorted(carriers[lang])
        rng_for(cfg.seed, "clean", lang).shuffle(ids)
        for rank, cid in enumerate(ids[:k]):
            car = carriers[lang][cid]
            clean = {"id": f"clean-{cid}-{lang}", "source_id": f"clean-{cid}", "lang": lang,
                     "condition": "main", "label": 0, "goal": None, "style": None, "split": None}
            tasks.append({"setting": "clean", "rank": rank, "item": clean, "carrier": car})
    tasks.sort(key=lambda t: (t["setting"] != "same", t["rank"], t["item"]["id"]))
    return tasks


def build_prompt(cfg: Config, task: dict) -> tuple[str, str]:
    em = cfg["eval_models"]
    it = task["item"]
    system = em["system_prompt"]
    if it.get("goal") == "SECRET_LEAK" and it.get("secret"):
        system += em["secret_suffix"].replace("{SECRET}", it["secret"])
    if task["setting"] == "clean":
        doc = task["carrier"]["text"]
    else:
        doc = embed(task["carrier"]["text"], it["text"], it.get("position") or "middle")
    return system, em["user_template"].replace("{DOCUMENT}", doc)


def task_key(model_name: str, task: dict) -> str:
    return f"{model_name}|{task['setting']}|{task['item']['id']}|{task['carrier']['carrier_id']}|{task['carrier']['lang']}"


def run_model(cfg: Config, entry: dict, tasks: list[dict], *, inner: ChatClient | None,
              budget: Budget, out: JsonlAppender, done: set[str], phrases: dict) -> dict:
    name = entry["name"]
    em = cfg["eval_models"]
    try:
        spec = resolve_model(cfg, entry, stage="eval_models", budget=budget,
                             cache=default_cache(cfg, "eval_models"), inner=inner, name=name)
    except (ConfigTodoError, RuntimeError) as e:
        log_event(cfg, f"SKIP model {name}: {e}", stage="eval_models")
        return {"model": name, "status": "skipped", "reason": str(e)}
    todo = [t for t in tasks if task_key(name, t) not in done]
    stop = threading.Event()
    stats = {"model": name, "status": "ok", "new": 0, "failed": 0, "skipped_done": len(tasks) - len(todo)}
    # Anthropic's current API cannot take a temperature: record what was really used.
    is_anthropic = cfg["endpoints"][spec.endpoint]["provider"] == "anthropic"
    temp_applied = "api_default" if is_anthropic else float(em["temperature"])
    if is_anthropic and todo:
        log_event(cfg, f"{name}: Anthropic API 2026-10-21 rejects temperature; sampling at the API "
                       f"default instead of {em['temperature']} (report as a limitation)", stage="eval_models")

    def work(task):
        if stop.is_set():
            return None
        system, user = build_prompt(cfg, task)
        res = spec.client.chat(model=spec.model, system=system, user=user,
                               temperature=float(em["temperature"]), max_tokens=int(em["max_tokens"]))
        it = task["item"]
        row = {"model": name, "model_id": spec.model, "endpoint": spec.endpoint,
               "key": task_key(name, task), "setting": task["setting"], "item_id": it["id"],
               "source_id": it["source_id"], "lang": it["lang"],
               "lang_condition": "native" if it.get("condition") == "native" else it["lang"],
               "condition": it.get("condition", "main"), "carrier_id": task["carrier"]["carrier_id"],
               "carrier_lang": task["carrier"]["lang"], "goal": it.get("goal"), "style": it.get("style"),
               "split": it.get("split"), "position": it.get("position"),
               "system_prompt": system, "user_prompt": user, "response": res.text,
               "input_tokens": res.input_tokens, "output_tokens": res.output_tokens, "cached": res.cached,
               "temperature": temp_applied, "max_tokens": int(em["max_tokens"])}
        if task["setting"] == "clean":
            row.update({"success": None, "summary_ok": summary_ok(res.text),
                        "response_script": judge({}, res.text)["response_script"]})
        else:
            row.update(judge(it, res.text, phrases))
        return row

    conc = int(em.get("concurrency", 4))
    with ThreadPoolExecutor(max_workers=conc) as ex:
        futs = [ex.submit(work, t) for t in todo]
        for f in as_completed(futs):
            try:
                row = f.result()
            except BudgetExceeded as e:
                if not stop.is_set():
                    log_event(cfg, f"{name}: BUDGET STOP - {e}. Keeping what was collected.",
                              stage="eval_models")
                stop.set()
                stats["status"] = "budget_stop"
                continue
            except PriceMissing as e:
                if not stop.is_set():
                    log_event(cfg, f"SKIP model {name}: {e}", stage="eval_models")
                stop.set()
                stats["status"] = "skipped"
                continue
            except Exception as e:  # keep going on single failures; log the first few
                stats["failed"] += 1
                if stats["failed"] <= 3:
                    log_event(cfg, f"{name}: call failed ({type(e).__name__}: {str(e)[:300]})",
                              stage="eval_models")
                if stats["failed"] >= 20 and stats["new"] == 0 and not stop.is_set():
                    log_event(cfg, f"{name}: 20 failures and no successes; stopping this model",
                              stage="eval_models")
                    stop.set()
                continue
            if row is not None:
                out.append(row)
                stats["new"] += 1
    if stats["status"] == "ok" and stats["failed"] and stats["new"] == 0:
        stats["status"] = "failed"
    log_event(cfg, f"{name}: {stats}", stage="eval_models")
    return stats


def write_summaries(cfg: Config) -> list[dict]:
    rows = read_jsonl(cfg.results_dir / "models_raw.jsonl")
    att = [r for r in rows if r["setting"] != "clean"]
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in att:
        groups[(r["model"], r["setting"], r["lang_condition"])].append(r)
    summary = []
    for (m, s, lc), rs in sorted(groups.items()):
        d = {"model": m, "setting": s, "lang_condition": lc} | asr_metrics(rs)
        d["mentions_only"] = sum(int(bool(r.get("mentions_only"))) for r in rs)
        summary.append(d)
    write_csv(cfg.results_dir / "models_summary.csv", summary)
    for field in ("goal", "style"):
        g2: dict[tuple, list[dict]] = defaultdict(list)
        for r in att:
            if r["setting"] == "same":
                g2[(r["model"], r["lang_condition"], r.get(field))].append(r)
        write_csv(cfg.results_dir / f"models_by_{field}.csv",
                  [{"model": m, "lang_condition": lc, field: v} | asr_metrics(rs)
                   for (m, lc, v), rs in sorted(g2.items(), key=lambda kv: tuple(map(str, kv[0])))])
    clean = defaultdict(list)
    for r in rows:
        if r["setting"] == "clean":
            clean[(r["model"], r["lang"])].append(r)
    write_csv(cfg.results_dir / "models_clean_controls.csv",
              [{"model": m, "lang": lang, "n": len(rs),
                "summary_ok": sum(int(r["summary_ok"]) for r in rs) / len(rs),
                "response_scripts": dict(sorted(
                    {s: sum(1 for r in rs if r["response_script"] == s)
                     for s in {r["response_script"] for r in rs}}.items()))}
               for (m, lang), rs in sorted(clean.items())])
    return summary


def run(cfg: Config, *, clients: dict[str, ChatClient] | None = None, limit: int | None = None,
        only: list[str] | None = None) -> list[dict]:
    clients = clients or {}
    tasks = build_tasks(cfg)
    if limit:
        tasks = tasks[:limit]
    if not tasks:
        raise RuntimeError("no tasks: run the dataset and translation stages first")
    budget = Budget.from_config(cfg)
    raw_path = cfg.results_dir / "models_raw.jsonl"
    done = {r["key"] for r in read_jsonl(raw_path)}
    out = JsonlAppender(raw_path)
    phrases = read_json(cfg.data_dir / "phrase_translations.json", default={}) or {}
    log_event(cfg, f"{len(tasks)} tasks per model; budget spent so far ${budget.total:.4f} "
                   f"of ${budget.max_usd:.2f} (stop at {budget.stop_fraction:.0%})", stage="eval_models")
    stats = []
    for entry in cfg["eval_models"]["models"]:
        if entry.get("enabled", True) is False or (only and entry["name"] not in only):
            continue
        stats.append(run_model(cfg, entry, tasks, inner=clients.get(entry["name"]), budget=budget,
                               out=out, done=done, phrases=phrases))
    write_summaries(cfg)
    return stats


def main(argv=None):
    ap = stage_parser(__doc__)
    ap.add_argument("--only", nargs="+", help="model names from config to run")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.profile)
    load_dotenv_if_present(cfg)
    run(cfg, limit=args.limit, only=args.only)


if __name__ == "__main__":
    main()
