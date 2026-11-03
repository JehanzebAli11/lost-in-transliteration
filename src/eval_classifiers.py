"""RQ1/RQ2: open-source prompt-injection classifiers on every item (PLAN §4.1).

Scores every attack and benign item in en / ur / roman_ur / mixed plus the native set,
with each classifier's documented default threshold. Raw per-item scores go to
results/classifiers/<name>.jsonl; summaries to results/classifiers_summary.csv,
classifiers_by_style.csv and classifiers_by_goal.csv.

Gated or unavailable models are logged and skipped (PLAN §0 rule 1). Long inputs are
scored in overlapping windows and the max window score is used.

Classifiers are injectable: ``run(cfg, classifiers={name: obj})`` where obj has
``name``, ``threshold`` and ``score(texts) -> list[float]`` (probability of injection).
"""
from __future__ import annotations

import csv
import os
from collections import defaultdict
from pathlib import Path
from typing import Protocol

from .common import (Config, is_todo, load_config, load_dotenv_if_present, log_event, read_jsonl,
                     require_verified, stable_hash, stage_parser, write_jsonl)
from .data import embed, eval_group, lang_condition, load_all_items, load_carriers
from .metrics import detection_metrics


class Classifier(Protocol):
    name: str
    threshold: float

    def score(self, texts: list[str]) -> list[float]: ...


def pick_device(pref: str = "auto") -> str:
    if pref != "auto":
        return pref
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:  # pragma: no cover
        return "cpu"


class HFClassifier:
    """A Hugging Face sequence classifier returning P(injection)."""

    def __init__(self, name: str, hf_id: str, positive_labels: list[str], threshold: float = 0.5,
                 max_length: int = 512, device: str = "auto", batch_size: int = 32,
                 token: str | None = None, stride: int = 64):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.name = name
        self.threshold = float(threshold)
        self.max_length = int(max_length)
        self.batch_size = int(batch_size)
        self.stride = max(0, min(stride, self.max_length // 4))  # must be < max_length - specials
        self.device = pick_device(device)
        self.tok = AutoTokenizer.from_pretrained(hf_id, token=token)
        self.model = AutoModelForSequenceClassification.from_pretrained(hf_id, token=token)
        self.model.to(self.device).eval()
        id2label = {int(k): str(v) for k, v in self.model.config.id2label.items()}
        wanted = {p.upper() for p in positive_labels}
        pos = [i for i, lab in id2label.items() if lab.upper() in wanted]
        if not pos:
            raise ValueError(f"{name}: none of positive_labels {positive_labels} in id2label {id2label}")
        self.pos_idx = pos
        self.id2label = id2label
        self._torch = torch

    def score(self, texts: list[str]) -> list[float]:
        torch = self._torch
        out: list[float] = [0.0] * len(texts)
        for start in range(0, len(texts), self.batch_size):
            chunk = texts[start:start + self.batch_size]
            enc = self.tok(chunk, truncation=True, max_length=self.max_length, padding=True,
                           return_overflowing_tokens=True, stride=self.stride, return_tensors="pt")
            mapping = enc.pop("overflow_to_sample_mapping").tolist()
            enc = {k: v.to(self.device) for k, v in enc.items() if k in ("input_ids", "attention_mask", "token_type_ids")}
            with torch.no_grad():
                probs = torch.softmax(self.model(**enc).logits.float(), dim=-1)
            p = probs[:, self.pos_idx].sum(-1).tolist()
            best: dict[int, float] = defaultdict(float)
            for j, s in zip(mapping, p):
                best[j] = max(best[j], s)
            for j in range(len(chunk)):
                out[start + j] = float(best[j])
        return out


def build_classifiers(cfg: Config) -> dict[str, Classifier]:
    cc = cfg["classifiers"]
    token = os.environ.get("HF_TOKEN")
    out: dict[str, Classifier] = {}
    for entry in cc["models"]:
        if entry.get("enabled", True) is False:
            continue
        name = entry["name"]
        try:
            hf_id = require_verified(entry.get("hf_id"), f"classifier {name} hf_id")
            if any(is_todo(p) for p in entry.get("positive_labels", [])):
                raise ValueError(f"classifier {name}: positive_labels not verified")
            if entry.get("gated") and not token:
                raise RuntimeError("gated model and HF_TOKEN is not set (owner must accept the licence)")
            out[name] = HFClassifier(name, hf_id, entry["positive_labels"], entry.get("threshold", 0.5),
                                     entry.get("max_length", 512), cc.get("device", "auto"),
                                     cc.get("batch_size", 32), token=token)
            log_event(cfg, f"loaded classifier {name} ({hf_id}) labels={out[name].id2label}",
                      stage="classifiers")
        except Exception as e:  # blocked -> log, skip, keep going (PLAN §0 rule 1)
            log_event(cfg, f"SKIP classifier {name}: {type(e).__name__}: {e}", stage="classifiers")
    return out


def eval_inputs(cfg: Config, items: list[dict]) -> list[tuple[dict, str, str]]:
    """(item, input_kind, text) triples: standalone always; embedded if configured."""
    triples = [(it, "standalone", it["text"]) for it in items]
    if cfg["classifiers"].get("embedded_condition"):
        carriers = {lang: load_carriers(cfg, lang) for lang in cfg["languages"]}
        for it in items:
            cid = it.get("carrier_id")
            if it["label"] == 1 and cid and cid in carriers.get(it["lang"], {}):
                triples.append((it, "embedded", embed(carriers[it["lang"]][cid]["text"], it["text"],
                                                      it.get("position", "middle"))))
            elif it["label"] == 0:
                pool = sorted(carriers.get(it["lang"], {}))
                if pool:
                    c = carriers[it["lang"]][pool[int(stable_hash(it["id"])[:8], 16) % len(pool)]]
                    triples.append((it, "embedded", embed(c["text"], it["text"], "middle")))
    return triples


def raw_row(system: str, family: str, item: dict, input_kind: str, score: float,
            threshold: float) -> dict:
    return {
        "system": system, "family": family, "item_id": item["id"], "source_id": item["source_id"],
        "lang": item["lang"], "lang_condition": lang_condition(item), "condition": item.get("condition", "main"),
        "group": eval_group(item), "label": item["label"], "split": item.get("split"),
        "style": item.get("style"), "goal": item.get("goal"), "input": input_kind,
        "score": float(score), "threshold": float(threshold), "decision": int(score >= threshold),
    }


def score_system(clf: Classifier, triples, family: str) -> list[dict]:
    texts = [t for _, _, t in triples]
    scores = clf.score(texts)
    return [raw_row(clf.name, family, it, kind, s, clf.threshold)
            for (it, kind, _), s in zip(triples, scores)]


# ----------------------------------------------------------------------------- summaries
SUMMARY_KEYS = ["system", "family", "input", "group_set", "lang_condition"]


def summarize(rows: list[dict]) -> list[dict]:
    """Metrics per system x input x lang_condition, for three item sets:
    'all' main items, 'test' (test split, comparable with Phase 2) and per-group."""
    out = []
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        base = (r["system"], r["family"], r["input"])
        lc = r["lang_condition"]
        if r["condition"] == "native":
            buckets[base + ("native", "native")].append(r)
            buckets[base + ("native", r["lang"])].append(r)
            continue
        buckets[base + ("all", lc)].append(r)
        # attacks go to their own split; benign items have only train/test, and the
        # held-out-style set is scored against the benign *test* items (PLAN §5.4)
        buckets[base + (r["split"], lc)].append(r)
        if r["label"] == 0 and r["split"] == "test":
            buckets[base + ("heldout_style", lc)].append(r)
    for key, rs in sorted(buckets.items()):
        out.append(dict(zip(SUMMARY_KEYS, key)) | detection_metrics(rs))
    return out


def summarize_by(rows: list[dict], field: str) -> list[dict]:
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        if r["label"] != 1 or r["condition"] == "native":
            continue
        buckets[(r["system"], r["input"], r["lang_condition"], r.get(field))].append(r)
    out = []
    for (system, inp, lc, val), rs in sorted(buckets.items(), key=lambda kv: tuple(map(str, kv[0]))):
        k = sum(r["decision"] for r in rs)
        out.append({"system": system, "input": inp, "lang_condition": lc, field: val,
                    "n": len(rs), "detected": k, "tpr": k / len(rs)})
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def load_detection_raw(cfg: Config) -> list[dict]:
    rows = []
    d = cfg.results_dir / "classifiers"
    if d.exists():
        for p in sorted(d.glob("*.jsonl")):
            rows += read_jsonl(p)
    return rows


def write_summaries(cfg: Config) -> list[dict]:
    rows = load_detection_raw(cfg)
    rows = [r for r in rows if r["family"] in ("classifier", "valar")]
    summ = summarize(rows)
    write_csv(cfg.results_dir / "classifiers_summary.csv", summ)
    write_csv(cfg.results_dir / "classifiers_by_style.csv", summarize_by(rows, "style"))
    write_csv(cfg.results_dir / "classifiers_by_goal.csv", summarize_by(rows, "goal"))
    return summ


def run(cfg: Config, *, classifiers: dict[str, Classifier] | None = None,
        limit: int | None = None, force: bool = False) -> list[dict]:
    items = load_all_items(cfg)
    if limit:
        items = items[:limit]
    if not items:
        raise RuntimeError("no items found; run the dataset stages first")
    classifiers = classifiers if classifiers is not None else build_classifiers(cfg)
    triples = eval_inputs(cfg, items)
    out_dir = cfg.results_dir / "classifiers"
    for name, clf in classifiers.items():
        path = out_dir / f"{name}.jsonl"
        if path.exists() and not force and len(read_jsonl(path)) == len(triples):
            log_event(cfg, f"{name}: already scored {len(triples)} inputs; skipping", stage="classifiers")
            continue
        rows = score_system(clf, triples, "classifier")
        write_jsonl(path, rows)
        log_event(cfg, f"{name}: scored {len(rows)} inputs -> {path.name}", stage="classifiers")
    summ = write_summaries(cfg)
    for s in summ:
        if s["group_set"] == "all" and s["input"] == "standalone":
            log_event(cfg, f"{s['system']} {s['lang_condition']}: TPR {s['tpr']:.3f} FPR {s['fpr']:.3f} "
                           f"AUC {s['auc']:.3f} (n={s['n_attack']}/{s['n_benign']})", stage="classifiers",
                      echo=True)
    return summ


def main(argv=None):
    ap = stage_parser(__doc__)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.profile)
    load_dotenv_if_present(cfg)
    run(cfg, limit=args.limit, force=args.force)


if __name__ == "__main__":
    main()
