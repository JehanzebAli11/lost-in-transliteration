"""Stretch (RQ7): linear probes on a small open model's hidden states (PLAN §10).

Only run this after Phases 1-3 and the write-up draft are done.

Items: test-split attacks + benign test items in all four languages (+ native).
They are split in half *by source_id* (so an item and its translations never sit on
both sides). Two probe setups, per layer:
  en_only     train on English items of the probe-train half; test on every language
  all_langs   train on all languages of the probe-train half; test on every language
Features: mean-pooled hidden states at ``probe.layers`` (fractions of depth).
Classifier: standardised logistic regression. Output: results/probe_results.csv.

The feature extractor is injectable (tests use synthetic features); the default loads
``probe.model`` with transformers.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Callable

import numpy as np

from .common import (Config, load_config, load_dotenv_if_present, log_event, require_verified,
                     rng_for, stage_parser)
from .data import lang_condition
from .eval_classifiers import write_csv
from .metrics import roc_auc
from .train_detector import eval_rows


def layer_indices(n_layers: int, fracs: list[float]) -> list[int]:
    return sorted({max(1, min(n_layers, int(round(f * n_layers)))) for f in fracs})


def hf_features(cfg: Config, texts: list[str]) -> dict[int, np.ndarray]:
    import torch
    from transformers import AutoModel, AutoTokenizer
    pc = cfg["probe"]
    model_id = require_verified(pc["model"], "probe.model")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModel.from_pretrained(model_id, torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32)
    model.to(device).eval()
    n_layers = model.config.num_hidden_layers
    layers = layer_indices(n_layers, pc["layers"])
    feats: dict[int, list] = defaultdict(list)
    bs = int(pc["batch_size"])
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], truncation=True, max_length=int(pc["max_length"]), padding=True,
                  return_tensors="pt").to(device)
        with torch.no_grad():
            hs = model(**enc, output_hidden_states=True).hidden_states
        mask = enc["attention_mask"].unsqueeze(-1).float()
        for L in layers:
            pooled = (hs[L].float() * mask).sum(1) / mask.sum(1).clamp(min=1)
            feats[L].append(pooled.cpu().numpy())
    return {L: np.concatenate(v) for L, v in feats.items()}


def split_by_source(items: list[dict], seed: int) -> tuple[set, set]:
    srcs = sorted({it["source_id"] for it in items})
    rng = rng_for(seed, "probe-split")
    rng.shuffle(srcs)
    half = len(srcs) // 2
    return set(srcs[:half]), set(srcs[half:])


def fit_eval(Xtr, ytr, Xte, yte) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if len(set(ytr.tolist())) < 2:
        return {"acc": float("nan"), "auc": float("nan")}
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:, 1]
    return {"acc": float(((p >= 0.5).astype(int) == yte).mean()), "auc": roc_auc(yte, p)}


def run(cfg: Config, *, extractor: Callable[[Config, list[str]], dict[int, np.ndarray]] | None = None,
        limit: int | None = None) -> list[dict]:
    items = eval_rows(cfg)
    frac = float(cfg["probe"].get("sample_fraction", 1.0))
    if frac < 1.0:
        keep, _ = split_by_source(items, cfg.seed + 1)
        srcs = sorted(keep)[: int(len({i['source_id'] for i in items}) * frac)]
        items = [i for i in items if i["source_id"] in set(srcs)]
        log_event(cfg, f"probe: using a {frac:.0%} sample ({len(items)} items)", stage="probe")
    if limit:
        items = items[:limit]
    feats = (extractor or hf_features)(cfg, [i["text"] for i in items])
    tr_src, te_src = split_by_source(items, cfg.seed)
    y = np.array([i["label"] for i in items])
    lc = np.array([lang_condition(i) for i in items])
    in_tr = np.array([i["source_id"] in tr_src for i in items])
    in_te = ~in_tr
    rows = []
    for L, X in sorted(feats.items()):
        for mode in ("en_only", "all_langs"):
            tr_mask = in_tr & ((lc == "en") if mode == "en_only" else (lc != "native"))
            for lang in sorted(set(lc.tolist())):
                te_mask = in_te & (lc == lang)
                if te_mask.sum() == 0:
                    continue
                r = fit_eval(X[tr_mask], y[tr_mask], X[te_mask], y[te_mask])
                rows.append({"layer": L, "train_mode": mode, "lang_condition": lang,
                             "n_train": int(tr_mask.sum()), "n_test": int(te_mask.sum()), **r})
    write_csv(cfg.results_dir / "probe_results.csv", rows)
    log_event(cfg, f"probe: {len(rows)} result rows over layers {sorted(feats)}", stage="probe")
    return rows


def main(argv=None):
    ap = stage_parser(__doc__)
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.profile)
    load_dotenv_if_present(cfg)
    run(cfg, limit=args.limit)


if __name__ == "__main__":
    main()
