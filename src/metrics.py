"""Metrics shared by every evaluation and by analyze.py.

Detection (classifiers, Valar, the Phase 2 detector): TPR on attacks, FPR on benign,
ROC-AUC. Susceptibility (RQ3): attack success rate with 95% Wilson intervals.
Language gaps: two-proportion z-test (PLAN §7.6), exact McNemar on items paired by
source_id (the same English item translated), and bootstrap CIs.
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from scipy import stats


def wilson_ci(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    # min/max with p: float rounding can put a bound past the rate at k=0 or k=n
    return (max(0.0, min(p, centre - half)), min(1.0, max(p, centre + half)))


def two_prop_ztest(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float]:
    """z and two-sided p for H0: p1 == p2 (pooled)."""
    if n1 == 0 or n2 == 0:
        return (float("nan"), float("nan"))
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return (0.0, 1.0)
    z = (k1 / n1 - k2 / n2) / se
    return (z, 2 * stats.norm.sf(abs(z)))


def mcnemar_exact(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value from the discordant counts b and c."""
    n = b + c
    if n == 0:
        return 1.0
    return float(min(1.0, 2 * stats.binom.cdf(min(b, c), n, 0.5)))


def bootstrap_diff_ci(a: Iterable[float], b: Iterable[float], *, paired: bool, n_boot: int = 2000,
                      seed: int = 0, alpha: float = 0.05) -> tuple[float, float, float]:
    """mean(a) - mean(b) with a percentile bootstrap CI."""
    a = np.asarray(list(a), dtype=float)
    b = np.asarray(list(b), dtype=float)
    if len(a) == 0 or len(b) == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    diff = a.mean() - b.mean()
    if paired:
        assert len(a) == len(b)
        idx = rng.integers(0, len(a), size=(n_boot, len(a)))
        boots = a[idx].mean(1) - b[idx].mean(1)
    else:
        ia = rng.integers(0, len(a), size=(n_boot, len(a)))
        ib = rng.integers(0, len(b), size=(n_boot, len(b)))
        boots = a[ia].mean(1) - b[ib].mean(1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return (float(diff), float(lo), float(hi))


def roc_auc(labels, scores) -> float:
    labels = np.asarray(labels)
    if len(set(labels.tolist())) < 2:
        return float("nan")
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(labels, scores))


def detection_metrics(rows: list[dict]) -> dict:
    """rows: dicts with label (0/1), score (float) and decision (0/1)."""
    att = [r for r in rows if r["label"] == 1]
    ben = [r for r in rows if r["label"] == 0]
    tp = sum(int(r["decision"]) for r in att)
    fp = sum(int(r["decision"]) for r in ben)
    tpr = tp / len(att) if att else float("nan")
    fpr = fp / len(ben) if ben else float("nan")
    tlo, thi = wilson_ci(tp, len(att))
    flo, fhi = wilson_ci(fp, len(ben))
    return {
        "n_attack": len(att), "n_benign": len(ben), "tp": tp, "fp": fp,
        "tpr": tpr, "tpr_lo": tlo, "tpr_hi": thi,
        "fpr": fpr, "fpr_lo": flo, "fpr_hi": fhi,
        "auc": roc_auc([r["label"] for r in rows], [r["score"] for r in rows]),
    }


def asr_metrics(rows: list[dict]) -> dict:
    k = sum(int(bool(r["success"])) for r in rows)
    n = len(rows)
    lo, hi = wilson_ci(k, n)
    return {"n": n, "successes": k, "asr": k / n if n else float("nan"), "ci_lo": lo, "ci_hi": hi}
