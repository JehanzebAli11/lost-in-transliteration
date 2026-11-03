"""Translate every English item into ur / roman_ur / mixed (PLAN §3.5).

Stages (``python -m src.translate <stage>``):

  phrases         translate the OVERRIDE phrase(s) once per variant -> data/phrase_translations.json
                  (the judge also accepts a translated phrase as an OVERRIDE success)
  items           attacks, benign controls and carriers -> {variant}.jsonl etc.
                  Resumable: rows already written are skipped on rerun.
  review-sample   data/human_review_sample.csv (20 random items per variant; never
                  overwrites a sheet the owner has started filling in)
  backtranslate   10% sample back to English; chrF (+ optional embedding similarity)
  all             phrases + items + review-sample

Every output is checked before it is accepted: canary / secret / fixed phrase survive
verbatim, the text is in the right script and variant, markup (HTML comments, footnotes)
survives, the length is plausible, and a LANG_SWITCH item was translated rather than
obeyed. Failures are regenerated with a new seed up to ``translation.max_retries``;
items that never pass are dropped and logged to results/translation_failures.jsonl.

Train-split items use prompts/translation/translate_train.txt and test / held-out-style
items and carriers use translate_test.txt, with different seeds (§3.5 diversity).
"""
from __future__ import annotations

import csv
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .budget import Budget, BudgetExceeded
from .common import (VARIANTS, Config, JsonlAppender, is_todo, load_config, load_dotenv_if_present,
                     log_event, read_json, read_jsonl, rng_for, stable_hash, stage_parser,
                     write_json, write_jsonl)
from .data import attacks_path, benign_path, carriers_path
from .llm import ChatClient, ModelSpec, data_model
from .prompts import literals_text, render, variant_description
from .textcheck import (clean_translation, is_french, markup_preserved, missing_literals,
                        variant_check)

KINDS = ("attacks", "benign", "carriers")


def prompt_variant(row: dict, kind: str) -> str:
    """'train' prompt only for train-split attacks/benign; everything else uses 'test'."""
    if kind == "carriers":
        return "test"
    return "train" if row.get("split") == "train" else "test"


def row_key(row: dict, kind: str) -> str:
    return row["carrier_id"] if kind == "carriers" else row["source_id"]


def check_translation(src: dict, text: str, variant: str, cfg: Config) -> list[str]:
    """All reasons a candidate translation is unacceptable (empty list = accept)."""
    reasons = []
    lits = src.get("literals") or []
    miss = missing_literals(text, lits)
    if miss:
        reasons.append(f"lost literal(s) {miss}")
    why = variant_check(text, variant)
    if why:
        reasons.append(why)
    if not markup_preserved(src["text"], text):
        reasons.append("markup lost")
    tr = cfg["translation"]
    ratio = len(text) / max(1, len(src["text"]))
    if ratio > float(tr["max_length_ratio"]) or ratio < float(tr["min_length_ratio"]):
        reasons.append(f"length ratio {ratio:.2f}")
    if src.get("goal") == "LANG_SWITCH" and is_french(text):
        reasons.append("translator obeyed LANG_SWITCH (output is French)")
    return reasons


def translate_one(model: ModelSpec, row: dict, kind: str, variant: str, cfg: Config,
                  failures: JsonlAppender | None = None) -> dict | None:
    pv = prompt_variant(row, kind)
    tmpl = f"translation/translate_{pv}.txt"
    key = row_key(row, kind)
    base_seed = int(stable_hash(cfg.seed, "tr", pv, key, variant)[:8], 16) % 1_000_000
    if pv == "test":
        base_seed += 500_000  # different seed family for test items
    max_tries = int(cfg["translation"]["max_retries"]) + 1
    prompt = render(tmpl, variant_description=variant_description(variant),
                    literals=literals_text(row.get("literals") or []), text=row["text"])
    reasons: list[str] = []
    for attempt in range(max_tries):
        res = model.chat(prompt, seed=base_seed + attempt)
        text = clean_translation(res.text)
        reasons = check_translation(row, text, variant, cfg)
        if not reasons:
            out = dict(row)
            if kind == "carriers":
                out["lang"] = variant
            else:
                out["id"] = f"{row['source_id']}-{variant}"
                out["lang"] = variant
            out["text"] = text
            out["translation"] = {"model": model.model, "prompt": pv, "seed": base_seed + attempt,
                                  "attempts": attempt + 1}
            return out
        if failures is not None:
            failures.append({"kind": kind, "key": key, "variant": variant, "attempt": attempt + 1,
                             "reasons": reasons, "output": text[:500]})
    return None


def _source_rows(cfg: Config, kind: str) -> list[dict]:
    if kind == "attacks":
        return read_jsonl(attacks_path(cfg, "en"))
    if kind == "benign":
        return read_jsonl(benign_path(cfg, "en"))
    return read_jsonl(carriers_path(cfg, "en"))


def _out_path(cfg: Config, kind: str, variant: str) -> Path:
    return {"attacks": attacks_path, "benign": benign_path, "carriers": carriers_path}[kind](cfg, variant)


def run_items(cfg: Config, *, inner: ChatClient | None = None, kinds=KINDS, variants=None,
              limit: int | None = None) -> dict:
    model = data_model(cfg, "translator", stage="translate", inner=inner,
                       budget=Budget.from_config(cfg))
    variants = variants or cfg["translation"]["variants"]
    failures = JsonlAppender(cfg.results_dir / "translation_failures.jsonl")
    report = read_json(cfg.results_dir / "translation_report.json", default={}) or {}
    conc = int(cfg["translation"].get("concurrency", 4))
    for kind in kinds:
        src = _source_rows(cfg, kind)
        if limit:
            src = src[:limit]
        if not src:
            log_event(cfg, f"no English {kind} to translate; run build_dataset first", stage="translate")
            continue
        for variant in variants:
            out_path = _out_path(cfg, kind, variant)
            done = {row_key(r, kind) for r in read_jsonl(out_path)}
            todo = [r for r in src if row_key(r, kind) not in done]
            app = JsonlAppender(out_path)
            ok = dropped = 0
            stopped = None
            with ThreadPoolExecutor(max_workers=conc) as ex:
                futs = {ex.submit(translate_one, model, r, kind, variant, cfg, failures): r for r in todo}
                for fut in as_completed(futs):
                    try:
                        out = fut.result()
                    except BudgetExceeded as e:
                        stopped = str(e)
                        continue
                    if out is None:
                        dropped += 1
                    else:
                        app.append(out)
                        ok += 1
            rows = read_jsonl(out_path)  # rewrite in stable order, deduplicated
            uniq = {row_key(r, kind): r for r in rows}
            order = {row_key(r, kind): i for i, r in enumerate(src)}
            write_jsonl(out_path, sorted(uniq.values(), key=lambda r: order.get(row_key(r, kind), 1e9)))
            attempts = Counter(r["translation"]["attempts"] for r in uniq.values())
            rep = report.setdefault(kind, {}).setdefault(variant, {})
            rep.update({"source": len(src), "translated": len(uniq),
                        "dropped_last_run": dropped,
                        "attempts_histogram": {str(k): v for k, v in sorted(attempts.items())}})
            log_event(cfg, f"{kind}/{variant}: +{ok} translated, {dropped} dropped after retries, "
                           f"{len(uniq)}/{len(src)} total" + (f"; STOPPED: {stopped}" if stopped else ""),
                      stage="translate")
            write_json(cfg.results_dir / "translation_report.json", report)
    return report


def run_phrases(cfg: Config, *, inner: ChatClient | None = None) -> dict:
    model = data_model(cfg, "translator", stage="translate", inner=inner,
                       budget=Budget.from_config(cfg))
    out = {}
    for phrase in cfg["dataset"]["override_phrases"]:
        out[phrase] = {}
        for variant in cfg["translation"]["variants"]:
            res = model.chat(render("translation/phrase.txt",
                                    variant_description=variant_description(variant), phrase=phrase),
                             seed=cfg.seed)
            out[phrase][variant] = clean_translation(res.text).strip().strip('"').strip()
    write_json(cfg.data_dir / "phrase_translations.json", out)
    log_event(cfg, f"phrase translations: {out}", stage="translate")
    return out


# ----------------------------------------------------------------------------- human review
REVIEW_COLS = ["id", "english", "translation", "variant", "natural (1-5)", "meaning_preserved (y/n)", "notes"]


def review_sheet_started(path: Path) -> bool:
    if not path.exists():
        return False
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if any((row.get(c) or "").strip() for c in REVIEW_COLS[4:]):
                return True
    return False


def run_review_sample(cfg: Config, *, force: bool = False) -> Path:
    path = cfg.data_dir / "human_review_sample.csv"
    if review_sheet_started(path):
        log_event(cfg, f"{path.name} already has owner ratings; not touching it", stage="translate")
        return path
    if path.exists() and not force:
        log_event(cfg, f"{path.name} exists (empty ratings); keeping it (use --force to resample)",
                  stage="translate")
        return path
    en = {r["source_id"]: r for r in read_jsonl(attacks_path(cfg, "en")) + read_jsonl(benign_path(cfg, "en"))}
    k = int(cfg["translation"]["review_per_variant"])
    rows = []
    for variant in cfg["translation"]["variants"]:
        pool = read_jsonl(attacks_path(cfg, variant)) + read_jsonl(benign_path(cfg, variant))
        pool.sort(key=lambda r: r["id"])
        rng = rng_for(cfg.seed, "review", variant)
        for r in rng.sample(pool, min(k, len(pool))):
            rows.append({"id": r["id"], "english": en[r["source_id"]]["text"], "translation": r["text"],
                         "variant": variant, "natural (1-5)": "", "meaning_preserved (y/n)": "", "notes": ""})
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=REVIEW_COLS)
        w.writeheader()
        w.writerows(rows)
    log_event(cfg, f"wrote {path.name}: {len(rows)} rows for the owner to rate", stage="translate")
    return path


# ----------------------------------------------------------------------------- back-translation
def chrf(hyp: str, ref: str) -> float:
    from sacrebleu.metrics import CHRF
    return float(CHRF().sentence_score(hyp, [ref]).score)


def embedding_similarity(cfg: Config, pairs: list[tuple[str, str]]) -> list[float] | None:
    model_id = cfg["translation"].get("similarity_model")
    if is_todo(model_id):
        return None
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        log_event(cfg, "sentence-transformers not installed; back-translation uses chrF only",
                  stage="translate")
        return None
    m = SentenceTransformer(model_id)
    a = m.encode([p[0] for p in pairs], normalize_embeddings=True)
    b = m.encode([p[1] for p in pairs], normalize_embeddings=True)
    return [float((x * y).sum()) for x, y in zip(a, b)]


def run_backtranslate(cfg: Config, *, inner: ChatClient | None = None) -> dict:
    model = data_model(cfg, "translator", stage="backtranslate", inner=inner,
                       budget=Budget.from_config(cfg))
    en = {r["source_id"]: r for r in read_jsonl(attacks_path(cfg, "en")) + read_jsonl(benign_path(cfg, "en"))}
    frac = float(cfg["translation"]["backtranslate_fraction"])
    rows = []
    for variant in cfg["translation"]["variants"]:
        pool = read_jsonl(attacks_path(cfg, variant)) + read_jsonl(benign_path(cfg, variant))
        pool.sort(key=lambda r: r["id"])
        rng = rng_for(cfg.seed, "backtranslate", variant)
        sample = rng.sample(pool, max(1, int(round(len(pool) * frac)))) if pool else []
        for r in sample:
            res = model.chat(render("translation/backtranslate.txt",
                                    variant_description=variant_description(variant),
                                    literals=literals_text(r.get("literals") or []), text=r["text"]),
                             seed=cfg.seed)
            back = clean_translation(res.text)
            src = en[r["source_id"]]["text"]
            rows.append({"id": r["id"], "variant": variant, "label": r["label"], "english": src,
                         "back": back, "chrf": chrf(back, src)})
    sims = embedding_similarity(cfg, [(r["english"], r["back"]) for r in rows]) if rows else None
    if sims:
        for r, s in zip(rows, sims):
            r["embed_sim"] = s
    write_jsonl(cfg.results_dir / "backtranslation.jsonl", rows)
    summary = defaultdict(dict)
    for variant in cfg["translation"]["variants"]:
        vs = [r for r in rows if r["variant"] == variant]
        if vs:
            summary[variant]["n"] = len(vs)
            summary[variant]["chrf_mean"] = sum(r["chrf"] for r in vs) / len(vs)
            if sims:
                summary[variant]["embed_sim_mean"] = sum(r["embed_sim"] for r in vs) / len(vs)
    write_json(cfg.results_dir / "backtranslation_summary.json", dict(summary))
    log_event(cfg, f"back-translation: {dict(summary)}", stage="translate")
    return dict(summary)


def main(argv=None):
    ap = stage_parser(__doc__)
    ap.add_argument("stage", choices=["phrases", "items", "review-sample", "backtranslate", "all"])
    ap.add_argument("--kinds", nargs="+", choices=KINDS, default=list(KINDS))
    ap.add_argument("--variants", nargs="+", choices=VARIANTS, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.profile)
    load_dotenv_if_present(cfg)
    if args.stage in ("phrases", "all"):
        run_phrases(cfg)
    if args.stage in ("items", "all"):
        run_items(cfg, kinds=args.kinds, variants=args.variants, limit=args.limit)
    if args.stage in ("review-sample", "all"):
        run_review_sample(cfg, force=args.force)
    if args.stage == "backtranslate":
        run_backtranslate(cfg)


if __name__ == "__main__":
    main()
