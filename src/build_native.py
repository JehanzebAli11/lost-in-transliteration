"""Natively written test set (PLAN §3.6): items that never passed through English.

A *different open-weight model family* from the translator (``models.native_writer``)
is prompted in Roman Urdu (prompts/native/*.txt) to write attacks and benign messages
directly in roman_ur / mixed / ur. No English template is ever shown to it.

Quality gate, in order (every drop is counted in results/native_gate_report.json):
  1. malformed       wrong placeholder count / stray braces / bad length
  2. wrong_script    Urdu-script item not in Urdu script, or Latin item containing it
  3. mostly_english  Latin item whose function words are mostly English
  4. duplicate       char-4gram Jaccard >= native.dup_threshold vs any main-set item
                     (all languages) or vs an already accepted native item
  5. label_disagree  the translator model labels the item blind (ATTACK/BENIGN) and
                     disagrees with the intended label (also: label_unparsed)
Then {CANARY}/{PHRASE} are filled by code and the set is sampled down to the targets.

These items are test-only: never used for training or prompt tuning.
Write-up wording: "model-written natively in Urdu variants", never "human-written".
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

from .budget import Budget
from .common import (Config, load_config, load_dotenv_if_present, log_event, read_jsonl,
                     rng_for, stage_parser, write_json, write_jsonl)
from .data import load_attacks, load_benign, load_carriers, native_path
from .llm import ChatClient, ModelSpec, ProvenanceError, data_model
from .prompts import load_yaml, render
from .textcheck import NearDupIndex, arabic_fraction, extract_json, is_mostly_english

PLACEHOLDER_RE = re.compile(r"\{[A-Z_]+\}")
FAMILIES = ["gemma", "qwen", "llama", "mistral", "mixtral", "phi", "deepseek", "aya", "command",
            "olmo", "falcon", "yi", "glm", "granite", "nemotron", "jais", "smollm", "gpt-oss"]


def model_family(model_id: str) -> str:
    low = model_id.lower()
    for fam in FAMILIES:
        if fam in low:
            return fam
    return low.split("/")[-1].split("-")[0]


def check_families(writer: ModelSpec, labeler: ModelSpec, cfg: Config) -> None:
    fw, fl = model_family(writer.model), model_family(labeler.model)
    if fw == fl and not cfg["native"].get("allow_same_family", False):
        raise ProvenanceError(f"native_writer ({writer.model}) and translator/labeler "
                              f"({labeler.model}) look like the same family ({fw}); §3.6 needs a "
                              f"different family")


def validate_native(text: str, *, label: int, goal: str | None, lang: str, opts: dict) -> str | None:
    if not isinstance(text, str) or not text.strip():
        return "malformed"
    n = len(text.split())
    if not 3 <= n <= 150:
        return "malformed"
    found = PLACEHOLDER_RE.findall(text)
    if label == 1:
        ph = opts["goals"][goal]["placeholder"]
        if ph:
            if found != [ph]:
                return "malformed"
        elif found:
            return "malformed"
    elif found or "{" in text or "}" in text:
        return "malformed"
    ar = arabic_fraction(text)
    if lang == "ur" and ar < 0.6:
        return "wrong_script"
    if lang != "ur" and ar > 0.1:
        return "wrong_script"
    return None


def parse_label(text: str) -> int | None:
    t = text.strip().upper()
    has_a, has_b = "ATTACK" in t, "BENIGN" in t
    if has_a and not has_b:
        return 1
    if has_b and not has_a:
        return 0
    return None


def lang_targets(total: int, mix: dict[str, float]) -> dict[str, int]:
    langs = list(mix)
    t = {l: int(math.floor(total * mix[l])) for l in langs}
    rem = total - sum(t.values())
    for l in sorted(langs, key=lambda l: -(total * mix[l] - t[l]))[:rem]:
        t[l] += 1
    return t


def _ask(model: ModelSpec, prompt: str, seed: int) -> list[str]:
    try:
        parsed = extract_json(model.chat(prompt, seed=seed).text)
    except ValueError:
        return []
    if isinstance(parsed, dict):
        parsed = next((v for v in parsed.values() if isinstance(v, list)), [])
    return [x.strip() for x in parsed if isinstance(x, str)] if isinstance(parsed, list) else []


def generate_candidates(cfg: Config, writer: ModelSpec, opts: dict) -> list[dict]:
    nc = cfg["native"]
    over = float(nc["overgenerate"])
    batch = int(nc["batch"])
    styles = list(opts["styles"])
    goals = list(cfg["dataset"]["goals"])
    kinds = list(opts["benign_kinds"])
    topics = opts["topics"]
    cands: list[dict] = []
    call = 0
    for label, total in ((1, int(nc["n_attacks"])), (0, int(nc["n_benign"]))):
        for lang, want in lang_targets(total, nc["lang_mix"]).items():
            target = math.ceil(want * over)
            got = 0
            i = 0
            while got < target and i < 4 * math.ceil(target / batch) + 8:
                k = min(batch, target - got)
                if label == 1:
                    style = styles[i % len(styles)]
                    goal = goals[(i // len(styles) + i) % len(goals)]
                    g = opts["goals"][goal]
                    prompt = render("native/attack.txt", n=k, lang_description=opts["lang"][lang],
                                    style_description=opts["styles"][style],
                                    goal_description=g["description"], placeholder_rule=g["rule"])
                    meta = {"style": f"native:{style}", "goal": goal}
                else:
                    kind = kinds[i % len(kinds)]
                    topic = topics[(i // len(kinds) + i) % len(topics)]
                    prompt = render("native/benign.txt", n=k, lang_description=opts["lang"][lang],
                                    kind_description=opts["benign_kinds"][kind], topic=topic)
                    meta = {"style": None, "goal": None, "kind": kind, "topic": topic}
                for text in _ask(writer, prompt, seed=cfg.seed + 7919 * label + 31 * call):
                    cands.append({"lang": lang, "label": label, "raw": text, **meta})
                    got += 1
                call += 1
                i += 1
    return cands


def run_native(cfg: Config, *, writer_inner: ChatClient | None = None,
               labeler_inner: ChatClient | None = None) -> dict:
    budget = Budget.from_config(cfg)
    writer = data_model(cfg, "native_writer", stage="native", inner=writer_inner, budget=budget)
    labeler = data_model(cfg, "native_labeler", stage="native", inner=labeler_inner, budget=budget)
    check_families(writer, labeler, cfg)
    opts = load_yaml("native/options.yaml")
    nc = cfg["native"]
    ds = cfg["dataset"]
    rng = rng_for(cfg.seed, "native")

    cands = generate_candidates(cfg, writer, opts)
    main_items = load_attacks(cfg) + load_benign(cfg)
    if not main_items:
        log_event(cfg, "main set is empty: duplicate check vs main set is vacuous", stage="native")
    main_index = NearDupIndex([r["text"] for r in main_items])
    native_index = NearDupIndex()
    dropped: Counter = Counter()
    kept: list[dict] = []
    for c in cands:
        why = validate_native(c["raw"], label=c["label"], goal=c.get("goal"), lang=c["lang"], opts=opts)
        if not why and c["lang"] != "ur" and is_mostly_english(c["raw"], max_ratio=float(nc["max_english_ratio"])):
            why = "mostly_english"
        if not why:
            filled_probe = c["raw"].replace("{CANARY}", "").replace("{PHRASE}", "")
            if main_index.max_similarity(filled_probe) >= float(nc["dup_threshold"]):
                why = "duplicate"
            elif native_index.max_similarity(filled_probe) >= float(nc["dup_threshold"]):
                why = "duplicate"
        if why:
            dropped[why] += 1
            continue
        native_index.add(c["raw"].replace("{CANARY}", "").replace("{PHRASE}", ""))
        kept.append(c)

    # Fill placeholders by code (§3.6), then blind-label with the translator model.
    used: set[str] = set()

    def fresh(words):
        while True:
            tok = f"{rng.choice(words)}-{rng.randint(1000, 9999)}"
            if tok not in used:
                used.add(tok)
                return tok

    labelled = []
    for c in kept:
        canary = fresh(ds["canary_words"]) if c.get("goal") == "CANARY" else None
        secret = fresh(ds["secret_words"]) if c.get("goal") == "SECRET_LEAK" else None
        phrase = rng.choice(ds["override_phrases"]) if c.get("goal") == "OVERRIDE" else None
        text = c["raw"]
        if canary:
            text = text.replace("{CANARY}", canary)
        if phrase:
            text = text.replace("{PHRASE}", phrase)
        pred = parse_label(labeler.chat(render("native/label.txt", text=text), seed=cfg.seed).text)
        if pred is None:
            dropped["label_unparsed"] += 1
            continue
        if pred != c["label"]:
            dropped["label_disagree"] += 1
            continue
        labelled.append({**c, "text": text, "canary": canary, "secret": secret, "phrase": phrase,
                         "literals": [x for x in (canary, phrase) if x], "labeler_label": pred})

    # Sample down to targets per (label, lang), spreading attacks across goals.
    final = []
    shortfall = {}
    for label, total in ((1, int(nc["n_attacks"])), (0, int(nc["n_benign"]))):
        for lang, want in lang_targets(total, nc["lang_mix"]).items():
            pool = [x for x in labelled if x["label"] == label and x["lang"] == lang]
            rng.shuffle(pool)
            if label == 1:
                by_goal = defaultdict(list)
                for x in pool:
                    by_goal[x["goal"]].append(x)
                pick = []
                while len(pick) < want and any(by_goal.values()):
                    for g in sorted(by_goal):
                        if by_goal[g] and len(pick) < want:
                            pick.append(by_goal[g].pop())
            else:
                pick = pool[:want]
            if len(pick) < want:
                shortfall[f"{'attack' if label else 'benign'}/{lang}"] = want - len(pick)
            final += pick

    rows = []
    counters: Counter = Counter()
    for x in final:
        lang = x["lang"]
        tag = "atk" if x["label"] else "ben"
        counters[(tag, lang)] += 1
        rid = f"nat-{tag}-{lang}-{counters[(tag, lang)]:03d}"
        carriers = sorted(load_carriers(cfg, lang))
        row = {"id": rid, "source_id": rid, "lang": lang, "condition": "native", "label": x["label"],
               "split": "native", "text": x["text"], "writer_model": writer.model,
               "labeler_label": x["labeler_label"]}
        if x["label"]:
            row.update({"style": x["style"], "goal": x["goal"], "template_id": None,
                        "canary": x["canary"], "secret": x["secret"], "phrase": x["phrase"],
                        "literals": x["literals"],
                        "carrier_id": rng.choice(carriers) if carriers else None,
                        "position": rng.choice(ds["positions"])})
        else:
            row.update({"kind": x.get("kind"), "topic": x.get("topic")})
        rows.append(row)
    write_jsonl(native_path(cfg), rows)
    report = {
        "writer_model": writer.model, "labeler_model": labeler.model,
        "generated": len(cands), "passed_text_gates": len(kept), "passed_label_gate": len(labelled),
        "dropped": dict(dropped), "final": len(rows),
        "final_by_label_lang": {f"{'attack' if k[0] == 'atk' else 'benign'}/{k[1]}": v
                                for k, v in sorted(counters.items())},
        "shortfall": shortfall,
    }
    write_json(cfg.results_dir / "native_gate_report.json", report)
    log_event(cfg, f"native set: {report['final']} items from {report['generated']} generated; "
                   f"dropped {dict(dropped)}; shortfall {shortfall or 'none'}", stage="native")
    return report


def main(argv=None):
    ap = stage_parser(__doc__)
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.profile)
    load_dotenv_if_present(cfg)
    run_native(cfg)


if __name__ == "__main__":
    main()
