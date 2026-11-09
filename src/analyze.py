"""Every figure and table in PLAN §7, from files in results/ only (rule 4: no number is
invented; a figure whose inputs are missing is skipped and listed as skipped).

  fig1_detection_by_lang.png     TPR by language, grouped bars per classifier (+ Valar)
  fig2_fpr_by_lang.png           FPR on benign by language per classifier
  fig3_roc_by_classifier.png     ROC curves per classifier, one line per language
  fig4_asr_by_lang.png           attack success rate by language per model, 95% Wilson CIs
  fig5_asr_heatmap_style_lang.png ASR by attack style x language, pooled across models
  table_detection_drop.csv/.md   each variant minus English: z-test, paired McNemar, bootstrap CI
  table_asr_gap.csv/.md          the same for model ASR
  fig7a/b/c/d_*.png              Phase 2: TPR / FPR vs % Urdu-variant data, best fine-tuned
                                 vs existing classifiers (same items), held-out style
  fig8_mcp_scanners.png          Phase 3 detection / false-positive rates by language
  fig9_probe.png                 stretch probe AUC by layer (if run)
  qualitative_failures.md        examples for the "why" section (§7)
  headline_numbers.json          numbers the write-up quotes, each with its source file

Colours: one fixed colour per language everywhere (validated categorical palette,
light mode); each figure has a CSV with the same numbers (table view).
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .common import Config, load_config, log_event, read_json, read_jsonl, rng_for, stage_parser, write_json  # noqa: E402
from .eval_classifiers import load_detection_raw, write_csv  # noqa: E402
from .metrics import bootstrap_diff_ci, mcnemar_exact, two_prop_ztest, wilson_ci  # noqa: E402

LANG_ORDER = ["en", "ur", "roman_ur", "mixed", "native"]
LANG_LABEL = {"en": "English", "ur": "Urdu script", "roman_ur": "Roman Urdu",
              "mixed": "Code-switched", "native": "Native (model-written)"}
# Validated categorical palette (dataviz reference, light mode), fixed order, never cycled.
LANG_COLORS = {"en": "#2a78d6", "ur": "#eb6834", "roman_ur": "#1baf7a", "mixed": "#eda100",
               "native": "#e87ba4"}
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
MCP_KIND_LABEL = {"poisoned_description": "poisoned description", "poisoned_output": "poisoned output",
                  "clean": "clean (false positives)"}


def style_axes(ax, *, ylabel=None, xlabel=None, title=None, ylim=(0, 1.0)):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    if ylim:
        ax.set_ylim(*ylim)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2, fontsize=10)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK2, fontsize=10)
    if title:
        ax.set_title(title, color=INK, fontsize=11, loc="left")


def new_fig(w=9, h=4.5, **kw):
    fig, ax = plt.subplots(figsize=(w, h), facecolor=SURFACE, **kw)
    return fig, ax


def save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    return path


def legend(ax, **kw):
    kw.setdefault("fontsize", 9)
    return ax.legend(frameon=False, labelcolor=INK2, **kw)


def grouped_bars(ax, groups: list[str], series: list[str], values: dict, errs: dict | None = None,
                 colors: dict | None = None, labels: dict | None = None, rotation: int = 0):
    """values[(group, series)] -> float; errs -> (lo, hi)."""
    n = len(series)
    width = 0.8 / max(1, n)
    x = np.arange(len(groups))
    for j, s in enumerate(series):
        ys = [values.get((g, s), np.nan) for g in groups]
        xs = x - 0.4 + width * (j + 0.5)
        ax.bar(xs, ys, width, color=(colors or {}).get(s), edgecolor=SURFACE, linewidth=1.5,
               label=(labels or {}).get(s, s))
        if errs:
            lo = [ys[i] - errs.get((g, s), (np.nan, np.nan))[0] for i, g in enumerate(groups)]
            hi = [errs.get((g, s), (np.nan, np.nan))[1] - ys[i] for i, g in enumerate(groups)]
            # clip: CIs read back from CSV can sit 1e-16 past the rate at 0 or 1
            ax.errorbar(xs, ys, yerr=[np.clip(np.nan_to_num(lo), 0, None), np.clip(np.nan_to_num(hi), 0, None)],
                        fmt="none",
                        ecolor=INK2, elinewidth=1, capsize=2)
    for i, g in enumerate(groups):  # say so when a scanner/system has no data for this panel
        if all(np.isnan(values.get((g, s), np.nan)) for s in series):
            ax.text(x[i], 0.04, "n/a", ha="center", va="bottom", color=MUTED, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=rotation, ha="right" if rotation else "center", fontsize=9, color=INK2,
                       rotation_mode="anchor" if rotation else "default")


def read_csv(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def f(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


# ----------------------------------------------------------------------------- Phase 1 classifiers
def classifier_rows(cfg: Config) -> list[dict]:
    return [r for r in load_detection_raw(cfg) if r["family"] in ("classifier", "valar")
            and r["input"] == "standalone"]


def _rate_table(rows: list[dict], label: int, field_set: str = "all") -> tuple[dict, dict, list[str]]:
    """values[(system, lang)] = rate of decision==1 among items with this label."""
    vals, errs = {}, {}
    buckets = defaultdict(list)
    for r in rows:
        if r["label"] != label:
            continue
        lc = r["lang_condition"]
        if field_set == "all" or (field_set == "test" and (r["split"] == "test" or lc == "native")):
            buckets[(r["system"], lc)].append(r["decision"])
    for k, ds in buckets.items():
        vals[k] = sum(ds) / len(ds)
        errs[k] = wilson_ci(sum(ds), len(ds))
    systems = sorted({k[0] for k in buckets})
    return vals, errs, systems


def fig_detection(cfg: Config, rows: list[dict], label: int, name: str, title: str, ylabel: str):
    vals, errs, systems = _rate_table(rows, label)
    if not systems:
        return None
    langs = [l for l in LANG_ORDER if any((s, l) in vals for s in systems)]
    fig, ax = new_fig(max(6, 1.8 * len(systems) + 2), 4.5)
    grouped_bars(ax, systems, langs, vals, errs, LANG_COLORS, LANG_LABEL)
    style_axes(ax, ylabel=ylabel, title=title)
    legend(ax, ncol=min(5, len(langs)), loc="upper left", bbox_to_anchor=(0, -0.08))
    write_csv(cfg.results_dir / f"{name}.csv",
              [{"system": s, "lang_condition": l, "rate": vals[(s, l)], "ci_lo": errs[(s, l)][0],
                "ci_hi": errs[(s, l)][1]} for s in systems for l in langs if (s, l) in vals])
    return save(fig, cfg.figures_dir / f"{name}.png")


def fig_roc(cfg: Config, rows: list[dict]):
    from sklearn.metrics import roc_curve
    systems = sorted({r["system"] for r in rows})
    if not systems:
        return None
    n = len(systems)
    cols = min(3, n)
    nrows = math.ceil(n / cols)
    fig, axes = plt.subplots(nrows, cols, figsize=(4.2 * cols, 4 * nrows), facecolor=SURFACE, squeeze=False)
    out = []
    for i, s in enumerate(systems):
        ax = axes[i // cols][i % cols]
        for lang in LANG_ORDER:
            rs = [r for r in rows if r["system"] == s and r["lang_condition"] == lang]
            ys = [r["label"] for r in rs]
            if len(set(ys)) < 2:
                continue
            fpr, tpr, _ = roc_curve(ys, [r["score"] for r in rs])
            from sklearn.metrics import auc
            a = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=LANG_COLORS[lang], linewidth=2, label=f"{LANG_LABEL[lang]} (AUC {a:.2f})")
            out.append({"system": s, "lang_condition": lang, "auc": a, "n": len(rs)})
        ax.plot([0, 1], [0, 1], color=AXIS, linewidth=1, linestyle="--")
        style_axes(ax, xlabel="False positive rate", ylabel="True positive rate", title=s, ylim=(0, 1.01))
        legend(ax, loc="lower right", fontsize=8)
    for j in range(n, nrows * cols):
        axes[j // cols][j % cols].axis("off")
    write_csv(cfg.results_dir / "fig3_roc_auc.csv", out)
    return save(fig, cfg.figures_dir / "fig3_roc_by_classifier.png")


def gap_table(rows: list[dict], *, key: str, outcome: str, label: int | None, n_boot: int, seed: int) -> list[dict]:
    """Each variant minus English for every system (key='system') or model (key='model').

    Main variants are paired with English by source_id (McNemar + paired bootstrap).
    The native set has no English twin: unpaired z-test and bootstrap only.
    """
    out = []
    by_sys = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        if label is not None and r["label"] != label:
            continue
        by_sys[r[key]][r["lang_condition"]][r["source_id"]] = int(bool(r[outcome]))
    for s, langs in sorted(by_sys.items()):
        en = langs.get("en", {})
        if not en:
            continue
        for lang in LANG_ORDER[1:]:
            v = langs.get(lang)
            if not v:
                continue
            k1, n1 = sum(en.values()), len(en)
            k2, n2 = sum(v.values()), len(v)
            z, p = two_prop_ztest(k2, n2, k1, n1)
            row = {key: s, "variant": lang, "en_rate": k1 / n1, "variant_rate": k2 / n2,
                   "diff": k2 / n2 - k1 / n1, "z": z, "p_ztest": p, "n_en": n1, "n_variant": n2}
            common = sorted(set(en) & set(v))
            if lang != "native" and common:
                a = [v[c] for c in common]
                b = [en[c] for c in common]
                bb = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)
                cc = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)
                d, lo, hi = bootstrap_diff_ci(a, b, paired=True, n_boot=n_boot, seed=seed)
                row.update({"n_paired": len(common), "paired_diff": d, "ci_lo": lo, "ci_hi": hi,
                            "p_mcnemar": mcnemar_exact(bb, cc)})
            else:
                d, lo, hi = bootstrap_diff_ci(list(v.values()), list(en.values()), paired=False,
                                              n_boot=n_boot, seed=seed)
                row.update({"n_paired": 0, "paired_diff": float("nan"), "ci_lo": lo, "ci_hi": hi,
                            "p_mcnemar": float("nan")})
            out.append(row)
    return out


def write_md_table(path: Path, rows: list[dict], cols: list[str], title: str) -> None:
    def fmt(v):
        if isinstance(v, float):
            return "n/a" if v != v else (f"{v:.2e}" if 0 < abs(v) < 1e-3 else f"{v:.3f}")
        return str(v)
    lines = [f"# {title}", "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(r.get(c, "")) for c in cols) + " |" for r in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ----------------------------------------------------------------------------- Phase 1 models
def model_rows(cfg: Config) -> list[dict]:
    return [r for r in read_jsonl(cfg.results_dir / "models_raw.jsonl") if r["setting"] == "same"]


def fig_asr(cfg: Config, rows: list[dict]):
    if not rows:
        return None
    b = defaultdict(list)
    for r in rows:
        b[(r["model"], r["lang_condition"])].append(int(bool(r["success"])))
    vals = {k: sum(v) / len(v) for k, v in b.items()}
    errs = {k: wilson_ci(sum(v), len(v)) for k, v in b.items()}
    models = sorted({k[0] for k in b})
    langs = [l for l in LANG_ORDER if any((m, l) in vals for m in models)]
    fig, ax = new_fig(max(6, 1.8 * len(models) + 2), 4.5)
    grouped_bars(ax, models, langs, vals, errs, LANG_COLORS, LANG_LABEL)
    style_axes(ax, ylabel="Attack success rate", title="Indirect injection success by language (95% Wilson CI)")
    legend(ax, ncol=min(5, len(langs)), loc="upper left", bbox_to_anchor=(0, -0.08))
    write_csv(cfg.results_dir / "fig4_asr_by_lang.csv",
              [{"model": m, "lang_condition": l, "n": len(b[(m, l)]), "asr": vals[(m, l)],
                "ci_lo": errs[(m, l)][0], "ci_hi": errs[(m, l)][1]} for m in models for l in langs if (m, l) in vals])
    return save(fig, cfg.figures_dir / "fig4_asr_by_lang.png")


def fig_asr_heatmap(cfg: Config, rows: list[dict]):
    main = [r for r in rows if r["condition"] == "main"]
    if not main:
        return None
    b = defaultdict(list)
    for r in main:
        b[(r["style"], r["lang"])].append(int(bool(r["success"])))
    styles = sorted({k[0] for k in b})
    langs = [l for l in LANG_ORDER if l != "native" and any((s, l) in b for s in styles)]
    M = np.array([[np.mean(b[(s, l)]) if (s, l) in b else np.nan for l in langs] for s in styles])
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seqblue", SEQ_BLUE)
    fig, ax = new_fig(1.6 * len(langs) + 3, 0.6 * len(styles) + 1.8)
    im = ax.imshow(M, cmap=cmap, vmin=0, vmax=max(0.01, np.nanmax(M)), aspect="auto")
    ax.set_xticks(range(len(langs)))
    ax.set_xticklabels([LANG_LABEL[l] for l in langs], color=INK2, fontsize=9)
    ax.set_yticks(range(len(styles)))
    ax.set_yticklabels([s.replace("_", " ") for s in styles], color=INK2, fontsize=9)
    vmax = max(0.01, np.nanmax(M))
    for i in range(len(styles)):
        for j in range(len(langs)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8,
                        color="#ffffff" if M[i, j] > 0.55 * vmax else INK)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_title("Attack success rate by style x language (pooled across models)", color=INK, fontsize=11, loc="left")
    cb = fig.colorbar(im, ax=ax, fraction=0.04)
    cb.ax.tick_params(colors=INK2, labelsize=8)
    cb.outline.set_visible(False)
    write_csv(cfg.results_dir / "fig5_asr_heatmap.csv",
              [{"style": s, "lang": l, "n": len(b[(s, l)]), "asr": float(np.mean(b[(s, l)]))}
               for s in styles for l in langs if (s, l) in b])
    return save(fig, cfg.figures_dir / "fig5_asr_heatmap_style_lang.png")


# ----------------------------------------------------------------------------- Phase 2
def detector_curve(cfg: Config) -> list[dict]:
    return read_csv(cfg.results_dir / "detector_curve.csv")


def fig_detector_vs_data(cfg: Config, curve: list[dict], metric: str, name: str, title: str):
    rows = [r for r in curve if r["group_set"] == "test"]
    if not rows:
        return None
    bases = sorted({r["base"] for r in rows})
    fig, axes = plt.subplots(1, len(bases), figsize=(5 * len(bases), 4.2), facecolor=SURFACE,
                             squeeze=False, sharey=True)
    for ax, base in zip(axes[0], bases):
        for lang in LANG_ORDER[:4]:
            rs = sorted([r for r in rows if r["base"] == base and r["lang_condition"] == lang], key=lambda r: f(r["pct"]))
            if not rs:
                continue
            x = [f(r["pct"]) for r in rs]
            y = np.array([f(r[f"{metric}_mean"]) for r in rs])
            sd = np.array([f(r[f"{metric}_sd"]) for r in rs])
            ax.plot(x, y, color=LANG_COLORS[lang], linewidth=2, marker="o", markersize=6,
                    markeredgecolor=SURFACE, label=LANG_LABEL[lang])
            ax.fill_between(x, y - sd, y + sd, color=LANG_COLORS[lang], alpha=0.12, linewidth=0)
        style_axes(ax, xlabel="% of Urdu-variant train data added", title=base,
                   ylabel=title if ax is axes[0][0] else None, ylim=(0, 1.02))
        ax.set_xticks(sorted({f(r["pct"]) for r in rows}))
    legend(axes[0][0], loc="best")
    seeds = max((int(f(r.get("n_seeds", 1))) for r in rows if f(r.get("n_seeds", 1)) == f(r.get("n_seeds", 1))),
                default=1)
    band = "band = ±1 sd over seeds" if seeds > 1 else "1 seed, no seed-variance band"
    fig.suptitle(f"{title} vs training data (test split; {band})", color=INK, fontsize=11, x=0.01,
                 ha="left")
    return save(fig, cfg.figures_dir / f"{name}.png")


def _test_auc_scores(curve: list[dict]) -> dict:
    scores = defaultdict(list)
    for r in curve:
        if r["group_set"] == "test" and f(r["auc_mean"]) == f(r["auc_mean"]):
            scores[(r["base"], int(f(r["pct"])))].append(f(r["auc_mean"]))
    return scores


def best_detector(curve: list[dict]) -> tuple[str, int] | None:
    """Base model + condition with the best mean AUC on the test split across languages."""
    scores = _test_auc_scores(curve)
    if not scores:
        return None
    return max(scores, key=lambda k: (np.mean(scores[k]), k[1]))


def best_per_base(curve: list[dict]) -> list[tuple[str, int]]:
    """The same rule within each base model, so a base that fits the test templates best cannot hide
    another base that generalises better to the held-out style and native sets."""
    scores = _test_auc_scores(curve)
    best: dict[str, tuple[str, int]] = {}
    for k in scores:
        if k[0] not in best or (np.mean(scores[k]), k[1]) > (np.mean(scores[best[k[0]]]), best[k[0]][1]):
            best[k[0]] = k
    return [best[b] for b in sorted(best)]


def fig_detector_vs_classifiers(cfg: Config, curve: list[dict], clf_rows: list[dict], group_set: str,
                                name: str, title: str):
    picks = best_per_base(curve)
    if not picks:
        return None
    runs = []  # (label, rows) per picked base; seeds of one condition are pooled
    for base, pct in picks:
        rows = []
        for p in sorted((cfg.results_dir / "detector").glob("*.jsonl")):
            rs = read_jsonl(p)
            if rs and rs[0]["base"] == base and rs[0]["pct"] == pct:
                rows += rs
        if rows:
            runs.append((f"fine-tuned {base} (+{pct}%)", rows))
    if not runs:
        return None

    def in_set(r):
        if group_set == "native":
            return r["condition"] == "native"
        if r["condition"] == "native":
            return False
        return r["split"] == group_set or (r["label"] == 0 and r["split"] == "test")
    runs = [(label, [r for r in rows if in_set(r)]) for label, rows in runs]
    det_ids = {r["item_id"] for _, rows in runs for r in rows}
    # classifiers on exactly the same item ids
    c_rows = [r for r in clf_rows if r["item_id"] in det_ids]
    vals, errs = {}, {}
    rows_out = []
    for sys_name, rs in runs + sorted(
            ((s, [r for r in c_rows if r["system"] == s]) for s in {r["system"] for r in c_rows})):
        pos, neg = defaultdict(list), defaultdict(list)
        for r in rs:
            lang = r["lang_condition"] if group_set != "native" else "native"
            (pos if r["label"] == 1 else neg)[lang].append(r["decision"])
        for lang, ds in pos.items():
            vals[(sys_name, lang)] = sum(ds) / len(ds)
            errs[(sys_name, lang)] = wilson_ci(sum(ds), len(ds))
            nb = neg.get(lang, [])
            rows_out.append({"system": sys_name, "set": group_set, "lang_condition": lang, "n": len(ds),
                             "tp": sum(ds), "tpr": vals[(sys_name, lang)], "n_benign": len(nb),
                             "fp": sum(nb), "fpr": sum(nb) / len(nb) if nb else float("nan")})
    systems = [label for label, _ in runs] + sorted({r["system"] for r in c_rows})
    langs = [l for l in LANG_ORDER if any((s, l) in vals for s in systems)]
    if not langs:
        return None
    fig, ax = new_fig(max(6, 1.8 * len(systems) + 2), 4.5)
    grouped_bars(ax, systems, langs, vals, errs, LANG_COLORS, LANG_LABEL, rotation=20)
    style_axes(ax, ylabel="Detection rate (TPR)", title=title)
    legend(ax, ncol=min(5, len(langs)), loc="upper left", bbox_to_anchor=(0, -0.3))
    write_csv(cfg.results_dir / f"{name}.csv", rows_out)
    return save(fig, cfg.figures_dir / f"{name}.png")


# ----------------------------------------------------------------------------- Phase 3 / probe
def fig_mcp(cfg: Config):
    rows = read_csv(cfg.results_dir / "mcp_summary.csv")
    if not rows:
        return None
    scanners = sorted({r["scanner"] for r in rows})
    kinds = [k for k in MCP_KIND_LABEL if any(r["kind"] == k for r in rows)]
    fig, axes = plt.subplots(1, len(kinds), figsize=(6 * len(kinds) + 1, 5.2), facecolor=SURFACE,
                             squeeze=False, sharey=True)
    for ax, kind in zip(axes[0], kinds):
        vals = {(r["scanner"], r["lang"]): f(r["rate"]) for r in rows if r["kind"] == kind}
        errs = {(r["scanner"], r["lang"]): (f(r["ci_lo"]), f(r["ci_hi"])) for r in rows if r["kind"] == kind}
        langs = [l for l in LANG_ORDER if any((s, l) in vals for s in scanners)]
        grouped_bars(ax, scanners, langs, vals, errs, LANG_COLORS, LANG_LABEL, rotation=35)
        style_axes(ax, title=MCP_KIND_LABEL[kind],
                   ylabel="Flagged rate" if ax is axes[0][0] else None, ylim=(0, 1.05))
    legend(axes[0][-1], loc="lower right", bbox_to_anchor=(1, 1.1), ncol=4)
    fig.suptitle("MCP scanners: share of server variants flagged, by language", color=INK, fontsize=11,
                 x=0.01, ha="left")
    return save(fig, cfg.figures_dir / "fig8_mcp_scanners.png")


def fig_probe(cfg: Config):
    rows = read_csv(cfg.results_dir / "probe_results.csv")
    if not rows:
        return None
    modes = ["en_only", "all_langs"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor=SURFACE, sharey=True)
    for ax, mode in zip(axes, modes):
        for lang in LANG_ORDER:
            rs = sorted([r for r in rows if r["train_mode"] == mode and r["lang_condition"] == lang],
                        key=lambda r: f(r["layer"]))
            if rs:
                ax.plot([f(r["layer"]) for r in rs], [f(r["auc"]) for r in rs], color=LANG_COLORS[lang],
                        linewidth=2, marker="o", markersize=6, markeredgecolor=SURFACE, label=LANG_LABEL[lang])
        style_axes(ax, xlabel="Layer", ylabel="Probe AUC" if mode == "en_only" else None,
                   title="trained on English only" if mode == "en_only" else "trained on all languages",
                   ylim=(0.4, 1.02))
    legend(axes[0], loc="lower right")
    return save(fig, cfg.figures_dir / "fig9_probe.png")


# ----------------------------------------------------------------------------- qualitative
def qualitative(cfg: Config, clf_rows: list[dict], m_rows: list[dict]) -> Path:
    k = int(cfg["analysis"]["qualitative_per_lang"])
    rng = rng_for(cfg.seed, "qualitative")
    lines = ["# Qualitative examples", "",
             "Auto-sampled for the §7 'why' analysis. Read them, then write the patterns into the post.", ""]
    by = defaultdict(dict)
    for r in clf_rows:
        if r["label"] == 1 and r["condition"] == "main":
            by[(r["system"], r["source_id"])][r["lang"]] = r
    texts = {}
    for lang in LANG_ORDER[:4]:
        from .data import attacks_path
        for it in read_jsonl(attacks_path(cfg, lang)):
            texts[it["id"]] = it["text"]
    lines.append("## Caught in English, missed in a variant (per classifier)\n")
    for system in sorted({s for s, _ in by}):
        lines.append(f"### {system}\n")
        for lang in ("ur", "roman_ur", "mixed"):
            pairs = [(src, d) for (s, src), d in by.items() if s == system and "en" in d and lang in d
                     and d["en"]["decision"] == 1 and d[lang]["decision"] == 0]
            pairs.sort(key=lambda p: p[0])
            rng.shuffle(pairs)
            lines.append(f"**{LANG_LABEL[lang]}**: {len(pairs)} items caught in English but missed here.\n")
            for src, d in pairs[:k]:
                lines.append(f"- `{src}` ({d['en']['style']}, {d['en']['goal']}): EN score {d['en']['score']:.2f} -> "
                             f"{d[lang]['score']:.2f}")
                lines.append(f"  - EN: {texts.get(d['en']['item_id'], '')[:300]!r}")
                lines.append(f"  - {lang}: {texts.get(d[lang]['item_id'], '')[:300]!r}")
            lines.append("")
    lines.append("## Model responses that followed an injection, by language\n")
    for lang in LANG_ORDER:
        succ = sorted([r for r in m_rows if r["lang_condition"] == lang and r["success"]], key=lambda r: r["key"])
        rng.shuffle(succ)
        lines.append(f"**{LANG_LABEL[lang]}**: {len(succ)} successful injections.\n")
        for r in succ[:k]:
            lines.append(f"- {r['model']} / `{r['item_id']}` ({r['goal']}): response {r['response'][:200]!r}")
        lines.append("")
    path = cfg.results_dir / "qualitative_failures.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ----------------------------------------------------------------------------- driver
def run(cfg: Config) -> dict:
    produced, skipped = {}, []
    nb = int(cfg["analysis"]["bootstrap"])

    def do(name, fn, *a):
        try:
            p = fn(*a)
        except Exception as e:  # never let one figure stop the others; log it
            log_event(cfg, f"analyze: {name} failed: {type(e).__name__}: {e}", stage="analyze")
            p = None
        if p is None:
            skipped.append(name)
        else:
            produced[name] = str(Path(p).relative_to(cfg.root)) if str(p).startswith(str(cfg.root)) else str(p)

    clf = classifier_rows(cfg)
    do("fig1", fig_detection, cfg, clf, 1, "fig1_detection_by_lang",
       "Injection detection rate by language (95% Wilson CI)", "Detection rate (TPR)")
    do("fig2", fig_detection, cfg, clf, 0, "fig2_fpr_by_lang",
       "False positive rate on benign text by language (95% Wilson CI)", "False positive rate")
    do("fig3", fig_roc, cfg, clf)
    if clf:
        drop = gap_table(clf, key="system", outcome="decision", label=1, n_boot=nb, seed=cfg.seed)
        fpr_gap = gap_table(clf, key="system", outcome="decision", label=0, n_boot=nb, seed=cfg.seed)
        for r in drop:
            r["metric"] = "tpr"
        for r in fpr_gap:
            r["metric"] = "fpr"
        write_csv(cfg.results_dir / "table_detection_drop.csv", drop + fpr_gap)
        write_md_table(cfg.results_dir / "table_detection_drop.md", drop + fpr_gap,
                       ["system", "metric", "variant", "en_rate", "variant_rate", "diff", "ci_lo", "ci_hi",
                        "p_ztest", "p_mcnemar", "n_en", "n_variant", "n_paired"],
                       "Detection drop: variant minus English (paired by source_id where possible)")
        produced["table6"] = "results/table_detection_drop.csv"
    else:
        skipped.append("table6")
    mrows = model_rows(cfg)
    do("fig4", fig_asr, cfg, mrows)
    do("fig5", fig_asr_heatmap, cfg, mrows)
    if mrows:
        gap = gap_table(mrows, key="model", outcome="success", label=None, n_boot=nb, seed=cfg.seed)
        write_csv(cfg.results_dir / "table_asr_gap.csv", gap)
        write_md_table(cfg.results_dir / "table_asr_gap.md", gap,
                       ["model", "variant", "en_rate", "variant_rate", "diff", "ci_lo", "ci_hi", "p_ztest",
                        "p_mcnemar", "n_en", "n_variant", "n_paired"], "Attack success gap: variant minus English")
    curve = detector_curve(cfg)
    do("fig7a", fig_detector_vs_data, cfg, curve, "tpr", "fig7a_detector_tpr_vs_data", "Detection rate")
    do("fig7b", fig_detector_vs_data, cfg, curve, "fpr", "fig7b_detector_fpr_vs_data", "False positive rate")
    do("fig7c", fig_detector_vs_classifiers, cfg, curve, clf, "test", "fig7c_best_detector_vs_classifiers",
       "Best fine-tuned detector vs existing classifiers (same test items)")
    do("fig7d", fig_detector_vs_classifiers, cfg, curve, clf, "heldout_style", "fig7d_heldout_style",
       "Held-out attack style (never in training): detection by language")
    do("fig7e", fig_detector_vs_classifiers, cfg, curve, clf, "native", "fig7e_native_set",
       "Model-written native set: detection")
    do("fig8", fig_mcp, cfg)
    do("fig9", fig_probe, cfg)
    if clf or mrows:
        produced["qualitative"] = str(qualitative(cfg, clf, mrows).relative_to(cfg.root))
    headline = headline_numbers(cfg, clf, mrows, curve)
    write_json(cfg.results_dir / "headline_numbers.json", headline)
    manifest = {"produced": produced, "skipped": skipped}
    write_json(cfg.results_dir / "figures_manifest.json", manifest)
    log_event(cfg, f"analyze: produced {sorted(produced)}; skipped {skipped}", stage="analyze")
    return manifest


def headline_numbers(cfg: Config, clf: list[dict], mrows: list[dict], curve: list[dict]) -> dict:
    """Small machine-readable digest for the write-up. Each entry names its source."""
    out = {"budget": read_json(cfg.results_dir / "spend.json", {}) or {}}
    tpr = defaultdict(list)
    for r in clf:
        if r["label"] == 1:
            tpr[(r["system"], r["lang_condition"])].append(r["decision"])
    out["classifier_tpr"] = {"source": "results/classifiers/*.jsonl",
                             "values": {f"{s}|{l}": sum(v) / len(v) for (s, l), v in sorted(tpr.items())}}
    asr = defaultdict(list)
    for r in mrows:
        asr[(r["model"], r["lang_condition"])].append(int(bool(r["success"])))
    out["model_asr"] = {"source": "results/models_raw.jsonl",
                        "values": {f"{m}|{l}": sum(v) / len(v) for (m, l), v in sorted(asr.items())}}
    best = best_detector(curve)
    out["best_detector"] = {"source": "results/detector_curve.csv",
                            "base": best[0] if best else None, "pct": best[1] if best else None}
    out["native_gate"] = read_json(cfg.results_dir / "native_gate_report.json", None)
    out["backtranslation"] = read_json(cfg.results_dir / "backtranslation_summary.json", None)
    # Every table the write-up quotes, row by row with its counts, so each number has an n.
    for key, name in [("classifiers", "classifiers_summary.csv"), ("detection_gaps", "table_detection_drop.csv"),
                      ("models", "models_summary.csv"), ("asr_gaps", "table_asr_gap.csv"),
                      ("clean_controls", "models_clean_controls.csv"), ("detectors", "detector_summary.csv"),
                      ("detector_curve", "detector_curve.csv"), ("mcp", "mcp_summary.csv")]:
        out[key] = {"source": f"results/{name}",
                    "rows": [{k: _num(v) for k, v in r.items()} for r in read_csv(cfg.results_dir / name)]}
    for r in out["models"]["rows"]:
        # success that is likely only a quote of the token/phrase (judge.mentions_only) taken out
        if isinstance(r.get("n"), int) and isinstance(r.get("mentions_only"), int) and r["n"]:
            r["strict_successes"] = r["successes"] - r["mentions_only"]
            r["strict_asr"] = r["strict_successes"] / r["n"]
            r["strict_ci_lo"], r["strict_ci_hi"] = wilson_ci(r["strict_successes"], r["n"])
    return out


def _num(v):
    """CSV cell -> int, float or str (counts stay ints so they read as n)."""
    if v is None or v == "":
        return v
    try:
        return int(v)
    except ValueError:
        pass
    try:
        x = float(v)
    except ValueError:
        return v
    return x if x == x else None  # NaN -> null in JSON


def main(argv=None):
    ap = stage_parser(__doc__)
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.profile)
    run(cfg)


if __name__ == "__main__":
    main()
