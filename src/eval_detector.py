"""Phase 2 evaluation (RQ4, RQ5): score every trained detector on held-out data.

Sets (PLAN §5.4), per language:
  test           test-split attacks (held-out templates) + benign test items
  heldout_style  attacks of the never-trained style family + benign test items
  native         the §3.6 model-written native set
Rows use the classifier schema (system = ``ft:<base>:pct<NNN>:s<seed>``, family
"detector"), so analyze.py compares them with Phase 1 classifiers on exactly the same
item ids. Outputs: results/detector/<system>.jsonl, results/detector_summary.csv
(per seed) and results/detector_curve.csv (mean and sd over seeds).
"""
from __future__ import annotations

import statistics
from collections import defaultdict

from .common import Config, load_config, load_dotenv_if_present, log_event, read_json, read_jsonl, stage_parser, write_jsonl
from .eval_classifiers import HFClassifier, raw_row, summarize, write_csv
from .train_detector import base_models, eval_rows, run_dir


def trained_runs(cfg: Config) -> list[dict]:
    runs = []
    dc = cfg["detector"]
    for b in base_models(cfg):
        for pct in dc["conditions"]:
            for seed in dc["seeds"]:
                d = run_dir(cfg, b["name"], int(pct), int(seed))
                meta = read_json(d / "meta.json", {})
                if meta.get("status") == "done" and (d / "final").exists():
                    runs.append({"base": b["name"], "pct": int(pct), "seed": int(seed),
                                 "path": d / "final", "system": f"ft:{b['name']}:pct{int(pct):03d}:s{int(seed)}"})
    return runs


def parse_system(system: str) -> dict:
    _, base, pct, seed = system.split(":")
    return {"base": base, "pct": int(pct[3:]), "seed": int(seed[1:])}


def write_summaries(cfg: Config) -> None:
    rows = []
    d = cfg.results_dir / "detector"
    for p in sorted(d.glob("*.jsonl")) if d.exists() else []:
        rows += read_jsonl(p)
    # 'all' would mix test and held-out-style items here; report the PLAN §5.4 sets only
    summ = [s | parse_system(s["system"]) for s in summarize(rows) if s["group_set"] != "all"]
    write_csv(cfg.results_dir / "detector_summary.csv", summ)
    agg: dict[tuple, list[dict]] = defaultdict(list)
    for s in summ:
        agg[(s["base"], s["pct"], s["group_set"], s["lang_condition"])].append(s)
    curve = []
    for (base, pct, gs, lc), ss in sorted(agg.items()):
        row = {"base": base, "pct": pct, "group_set": gs, "lang_condition": lc, "n_seeds": len(ss)}
        for m in ("tpr", "fpr", "auc"):
            vals = [s[m] for s in ss if s[m] == s[m]]  # drop NaN
            row[f"{m}_mean"] = statistics.fmean(vals) if vals else float("nan")
            row[f"{m}_sd"] = statistics.stdev(vals) if len(vals) > 1 else 0.0
        curve.append(row)
    write_csv(cfg.results_dir / "detector_curve.csv", curve)


def run(cfg: Config, *, scorer_factory=None, force: bool = False) -> list[str]:
    items = eval_rows(cfg)
    triples = [(it, "standalone", it["text"]) for it in items]
    out = []
    factory = scorer_factory or (lambda run: HFClassifier(run["system"], str(run["path"]), ["INJECTION"],
                                                          0.5, cfg["detector"]["max_length"],
                                                          cfg["detector"].get("device", "auto")))
    runs = trained_runs(cfg)
    if not runs:
        log_event(cfg, "no finished detector runs found in models/; run `make train` first", stage="eval_detector")
    for run in runs:
        path = cfg.results_dir / "detector" / f"{run['system'].replace(':', '_')}.jsonl"
        if path.exists() and not force and len(read_jsonl(path)) == len(triples):
            continue
        clf = factory(run)
        scores = clf.score([t for _, _, t in triples])
        rows = [raw_row(run["system"], "detector", it, kind, s, 0.5) | {"base": run["base"], "pct": run["pct"],
                                                                        "seed": run["seed"]}
                for (it, kind, _), s in zip(triples, scores)]
        write_jsonl(path, rows)
        out.append(run["system"])
        log_event(cfg, f"scored {run['system']} on {len(rows)} held-out items", stage="eval_detector")
    write_summaries(cfg)
    return out


def main(argv=None):
    ap = stage_parser(__doc__)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.profile)
    load_dotenv_if_present(cfg)
    run(cfg, force=args.force)


if __name__ == "__main__":
    main()
