"""Summarise research/prelim/out/scores.jsonl -> summary.csv + summary.md (no prompt text involved).

Attacks for the headline = adversarial + jailbreak. edge_case is reported separately because many of
its items are benign-looking probing questions (label noise, see research/external_datasets.md).
95% CIs: percentile bootstrap (2,000 resamples, seed 0), resampling items within each language.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

out = Path(sys.argv[1] if len(sys.argv) > 1 else "research/prelim/out")
items = [json.loads(l) for l in open(out / "scores.jsonl")]
meta = json.loads((out / "run_meta.json").read_text())
rng = np.random.default_rng(0)
B = 2000
HEAD_ATTACK = {"adversarial", "jailbreak"}


def arrays(model, lang, types):
    s = np.array([it["scores"][model] for it in items if it["lang"] == lang and it["type"] in types])
    return s


def metric_ci(f, *arrs):
    point = f(*arrs)
    boots = []
    for _ in range(B):
        boots.append(f(*[a[rng.integers(0, len(a), len(a))] for a in arrs]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, lo, hi, np.array(boots)


def auc(pos, neg):
    return roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[pos, neg])


rows = []
for model, m in meta["classifiers"].items():
    if "error" in m:
        rows.append({"classifier": model, "note": m["error"]})
        continue
    thr = m["threshold"]
    rate = lambda a: float((a > thr).mean()) if thr != 0.5 else float((a >= 0.5).mean())
    res = {}
    for lang in ("en", "ur"):
        pos = arrays(model, lang, HEAD_ATTACK)
        neg = arrays(model, lang, {"benign"})
        r = {}
        r["tpr"] = metric_ci(rate, pos)
        r["fpr"] = metric_ci(rate, neg)
        r["auc"] = metric_ci(auc, pos, neg)
        for t in ("adversarial", "jailbreak", "edge_case"):
            r[f"tpr_{t}"] = rate(arrays(model, lang, {t}))
        r["n_pos"], r["n_neg"] = len(pos), len(neg)
        res[lang] = r
    row = {"classifier": model, "threshold": thr}
    for lang in ("en", "ur"):
        r = res[lang]
        for k in ("tpr", "fpr", "auc"):
            p, lo, hi, _ = r[k]
            row[f"{k}_{lang}"] = round(p, 3)
            row[f"{k}_{lang}_ci"] = f"[{lo:.3f}, {hi:.3f}]"
        for t in ("adversarial", "jailbreak", "edge_case"):
            row[f"tpr_{t}_{lang}"] = round(r[f"tpr_{t}"], 3)
        row[f"n_attack_{lang}"], row[f"n_benign_{lang}"] = r["n_pos"], r["n_neg"]
    for k in ("tpr", "fpr", "auc"):  # ur - en, independent bootstrap of the two languages
        d = res["ur"][k][3] - res["en"][k][3]
        row[f"delta_{k}_ur_minus_en"] = round(res["ur"][k][0] - res["en"][k][0], 3)
        row[f"delta_{k}_ci"] = f"[{np.percentile(d, 2.5):.3f}, {np.percentile(d, 97.5):.3f}]"
    rows.append(row)

keys = sorted({k for r in rows for k in r}, key=lambda k: (k != "classifier", k))
with open(out / "summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys)
    w.writeheader()
    w.writerows(rows)

L = ["| Classifier | thr | TPR en | TPR ur | Δ TPR (ur−en) [95% CI] | FPR en | FPR ur | Δ FPR [95% CI] | AUC en | AUC ur | Δ AUC [95% CI] |",
     "|---|---|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    if "note" in r:
        L.append(f"| {r['classifier']} | skipped: {r['note'][:80]} |" + " |" * 9)
        continue
    L.append(f"| {r['classifier']} | {r['threshold']} | {r['tpr_en']:.3f} | {r['tpr_ur']:.3f} | {r['delta_tpr_ur_minus_en']:+.3f} {r['delta_tpr_ci']} "
             f"| {r['fpr_en']:.3f} | {r['fpr_ur']:.3f} | {r['delta_fpr_ur_minus_en']:+.3f} {r['delta_fpr_ci']} "
             f"| {r['auc_en']:.3f} | {r['auc_ur']:.3f} | {r['delta_auc_ur_minus_en']:+.3f} {r['delta_auc_ci']} |")
L += ["", "Detection rate by attack type (edge_case not in the headline TPR):", "",
      "| Classifier | adversarial en | adversarial ur | jailbreak en | jailbreak ur | edge_case en | edge_case ur |",
      "|---|---|---|---|---|---|---|"]
for r in rows:
    if "note" in r:
        continue
    L.append(f"| {r['classifier']} | " + " | ".join(f"{r[f'tpr_{t}_{g}']:.3f}" for t in ("adversarial", "jailbreak", "edge_case") for g in ("en", "ur")) + " |")
first = next(r for r in rows if "note" not in r)
L += ["", f"n (unique, after within-language dedup): attacks en={first['n_attack_en']}, ur={first['n_attack_ur']}; "
      f"benign en={first['n_benign_en']}, ur={first['n_benign_ur']}."]
(out / "summary.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
