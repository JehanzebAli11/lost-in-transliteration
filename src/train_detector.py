"""Phase 2 (RQ4): fine-tune a small multilingual detector with growing Urdu-variant data.

Conditions (PLAN §5.3), all with the same English train data:
  0   EN-only
  10  EN + 10% of the Urdu-variant train data (equal share of ur / roman_ur / mixed)
  25, 50, 100  likewise
Subsets are nested (the 10% set is inside the 25% set, ...) so the curve is not noisy
from resampling. Only ``train``-split items are used: never ``test``, ``heldout_style``
or native items; ``assert_no_leakage`` checks this before every run.

Training is a plain PyTorch loop that checkpoints every epoch to
``models/<base>/pct<NNN>/seed<k>/checkpoint`` and resumes from it, so a power cut loses
at most one epoch. Finished runs (``final/`` + ``meta.json``) are skipped on rerun.
Run it in the background on the Spark: ``make train`` (nohup + log file).

Weights stay in models/ (git-ignored) and are never uploaded (PLAN §0 rule 5).
"""
from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

from .common import (VARIANTS, Config, is_todo, load_config, load_dotenv_if_present, log_event,
                     now_iso, read_json, read_jsonl, require_verified, rng_for, set_global_seed,
                     stage_parser, write_json)
from .data import attacks_path, benign_path, load_native
from .textcheck import NearDupIndex

ID2LABEL = {0: "BENIGN", 1: "INJECTION"}


# ----------------------------------------------------------------------------- data
def _train_rows(cfg: Config, lang: str) -> list[dict]:
    rows = [r for r in read_jsonl(attacks_path(cfg, lang)) if r["split"] == "train"]
    rows += [r for r in read_jsonl(benign_path(cfg, lang)) if r["split"] == "train"]
    return sorted(rows, key=lambda r: r["id"])


def eval_rows(cfg: Config) -> list[dict]:
    """Everything a detector may be evaluated on (never trained on)."""
    rows = []
    for lang in cfg["languages"]:
        rows += [r for r in read_jsonl(attacks_path(cfg, lang)) if r["split"] in ("test", "heldout_style")]
        rows += [r for r in read_jsonl(benign_path(cfg, lang)) if r["split"] == "test"]
    rows += load_native(cfg)
    return rows


def variant_pool(cfg: Config) -> dict[str, list[dict]]:
    """Per-variant train rows in a fixed random order (prefixes give nested subsets)."""
    pools = {}
    for v in VARIANTS:
        rows = _train_rows(cfg, v)
        rng_for(cfg.seed, "variant-pool", v).shuffle(rows)
        pools[v] = rows
    return pools


def load_extra_english(cfg: Config, blocked_texts: list[str]) -> list[dict]:
    ex = cfg["detector"].get("extra_english_dataset") or {}
    hf_id = ex.get("hf_id")
    if not hf_id or is_todo(hf_id):
        return []
    try:
        from datasets import load_dataset
    except ImportError:
        log_event(cfg, "extra_english_dataset set but `datasets` is not installed; skipping", stage="train")
        return []
    ds = load_dataset(hf_id, split=ex.get("split", "train"))
    index = NearDupIndex(blocked_texts)
    out, dropped = [], 0
    for i, r in enumerate(ds):
        if len(out) >= int(ex.get("max_items", 2000)):
            break
        text, label = str(r[ex["text_field"]]), int(r[ex["label_field"]])
        if index.max_similarity(text) >= float(cfg["detector"]["dedup_threshold"]):
            dropped += 1
            continue
        out.append({"id": f"extra-{i:06d}", "source_id": f"extra-{i:06d}", "lang": "en",
                    "label": label, "split": "train", "text": text, "condition": "extra"})
    log_event(cfg, f"extra English data {hf_id}: kept {len(out)}, dropped {dropped} near-duplicates "
                   f"of eval items", stage="train")
    return out


def training_set(cfg: Config, pct: int, *, extra: list[dict] | None = None) -> list[dict]:
    rows = _train_rows(cfg, "en") + list(extra or [])
    if pct > 0:
        for v, pool in variant_pool(cfg).items():
            rows += pool[: int(round(len(pool) * pct / 100))]
    return rows


def assert_no_leakage(train: list[dict], evals: list[dict]) -> None:
    bad = [r["id"] for r in train if r.get("split") != "train" or r.get("condition") == "native"]
    if bad:
        raise AssertionError(f"non-train items in training data: {bad[:5]}")
    eval_src = {r["source_id"] for r in evals if r.get("condition") != "native"}
    overlap = {r["source_id"] for r in train} & eval_src
    if overlap:
        raise AssertionError(f"train/eval source overlap: {sorted(overlap)[:5]}")


def drop_eval_duplicates(cfg: Config, train: list[dict], evals: list[dict]) -> list[dict]:
    """Remove train rows whose text is identical to an evaluation text (logged)."""
    eval_texts = {r["text"].strip() for r in evals}
    keep = [r for r in train if r["text"].strip() not in eval_texts]
    if len(keep) != len(train):
        log_event(cfg, f"dropped {len(train) - len(keep)} train rows identical to eval texts: "
                       f"{[r['id'] for r in train if r['text'].strip() in eval_texts][:5]}", stage="train")
    return keep


# ----------------------------------------------------------------------------- training
def pick_device(pref: str) -> str:
    import torch
    if pref == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return pref


def run_dir(cfg: Config, base_name: str, pct: int, seed: int) -> Path:
    return cfg.models_dir / base_name / f"pct{pct:03d}" / f"seed{seed}"


def train_one(cfg: Config, base_name: str, hf_id: str, pct: int, seed: int, rows: list[dict]) -> dict:
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    dc = cfg["detector"]
    out = run_dir(cfg, base_name, pct, seed)
    meta_path = out / "meta.json"
    if (out / "final").exists() and read_json(meta_path, {}).get("status") == "done":
        return read_json(meta_path)
    out.mkdir(parents=True, exist_ok=True)
    set_global_seed(seed)
    device = pick_device(dc.get("device", "auto"))
    tok = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        hf_id, num_labels=2, id2label=ID2LABEL, label2id={v: k for k, v in ID2LABEL.items()},
        ignore_mismatched_sizes=True)
    model.to(device)
    max_len = int(dc["max_length"])
    bs = int(dc["batch_size"])
    epochs = int(dc["epochs"])

    order_rng = random.Random(seed)
    data = [(r["text"], int(r["label"])) for r in rows]

    def collate(batch):
        enc = tok([t for t, _ in batch], truncation=True, max_length=max_len, padding=True,
                  return_tensors="pt")
        enc["labels"] = torch.tensor([y for _, y in batch])
        return enc

    steps_per_epoch = math.ceil(len(data) / bs)
    total = steps_per_epoch * epochs
    opt = torch.optim.AdamW(model.parameters(), lr=float(dc["lr"]), weight_decay=float(dc["weight_decay"]))
    sched = get_linear_schedule_with_warmup(opt, int(total * float(dc["warmup_ratio"])), total)
    start_epoch = 0
    ckpt = out / "checkpoint" / "state.pt"
    if ckpt.exists():
        state = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        start_epoch = state["epoch"] + 1
        order_rng.setstate(state["order_rng"])
        log_event(cfg, f"{base_name} pct{pct} seed{seed}: resuming at epoch {start_epoch}", stage="train")
    use_bf16 = device == "cuda" and bool(dc.get("bf16", True)) and torch.cuda.is_bf16_supported()
    t0 = time.time()
    losses = []
    for epoch in range(start_epoch, epochs):
        model.train()
        order = list(range(len(data)))
        order_rng.shuffle(order)
        loader = DataLoader([data[i] for i in order], batch_size=bs, shuffle=False, collate_fn=collate)
        ep_loss = 0.0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            ep_loss += float(loss.detach())
        losses.append(ep_loss / max(1, steps_per_epoch))
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        tmp = ckpt.with_suffix(".tmp")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "epoch": epoch, "order_rng": order_rng.getstate()}, tmp)
        tmp.replace(ckpt)
        log_event(cfg, f"{base_name} pct{pct} seed{seed}: epoch {epoch + 1}/{epochs} loss {losses[-1]:.4f}",
                  stage="train")
    model.save_pretrained(out / "final")
    tok.save_pretrained(out / "final")
    comp: dict[str, int] = {}
    for r in rows:
        k = f"{r['lang']}/{'attack' if r['label'] else 'benign'}"
        comp[k] = comp.get(k, 0) + 1
    meta = {"status": "done", "base": base_name, "hf_id": hf_id, "pct": pct, "seed": seed,
            "n_train": len(rows), "composition": dict(sorted(comp.items())), "epochs": epochs,
            "epoch_losses": losses, "device": device, "bf16": use_bf16,
            "train_seconds": round(time.time() - t0, 1), "finished": now_iso()}
    write_json(meta_path, meta)
    try:
        ckpt.unlink()
    except OSError:
        pass
    with open(cfg.results_dir / "train_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(meta) + "\n")
    return meta


def base_models(cfg: Config) -> list[dict]:
    dc = cfg["detector"]
    only = dc.get("only_base_models")
    return [b for b in dc["base_models"] if not only or b["name"] in only]


def run(cfg: Config, *, base_override: dict[str, str] | None = None) -> list[dict]:
    """``base_override`` maps base name -> local model path (tests)."""
    dc = cfg["detector"]
    evals = eval_rows(cfg)
    extra = load_extra_english(cfg, [r["text"] for r in evals])
    metas = []
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    for b in base_models(cfg):
        hf_id = (base_override or {}).get(b["name"]) or require_verified(b["hf_id"], f"detector base {b['name']}")
        for pct in dc["conditions"]:
            rows = drop_eval_duplicates(cfg, training_set(cfg, int(pct), extra=extra), evals)
            assert_no_leakage(rows, evals)
            for seed in dc["seeds"]:
                log_event(cfg, f"train {b['name']} pct{pct} seed{seed}: {len(rows)} examples", stage="train")
                metas.append(train_one(cfg, b["name"], hf_id, int(pct), int(seed), rows))
    return metas


def main(argv=None):
    ap = stage_parser(__doc__)
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.profile)
    load_dotenv_if_present(cfg)
    run(cfg)


if __name__ == "__main__":
    main()
