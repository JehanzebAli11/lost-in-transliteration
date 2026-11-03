# SUMMARY

_Status: code built and tested on fakes (2026-11-03). No experiment has run yet._
_Fill this in at the end of the run (PLAN §8.3). Every number must come from results/._

## Headline numbers
- RQ1 detection gap (best classifier, English vs Roman Urdu): _TODO from results/table_detection_drop.md_
- RQ2 false positives on benign Roman Urdu: _TODO from results/classifiers_summary.csv_
- RQ3 attack success gap: _TODO from results/table_asr_gap.md_
- RQ4 fine-tuned detector (best condition) on the native set: _TODO from results/detector_curve.csv_
- RQ5 held-out style: _TODO_
- RQ6 MCP scanners: _TODO from results/mcp_summary.csv_

## Total API spend
_TODO from results/spend.json_ (cap USD 40, hard stop at 95%).

## Skipped and why
_TODO (e.g. gated classifiers, unverified scanners, stretch probe)._

## Owner's next steps
1. Fill in `data/human_review_sample.csv` (~1 hour; 60 rows).
2. Run Valar: set `VALAR_API_URL`/`VALAR_API_KEY` (and the field paths under `valar:` in config.yaml), then `make eval-valar analyze`.
3. Decide on publishing data and the post; disclose any large scanner bypass to the vendor first.
4. Trained weights stay in `models/` (git-ignored); decide whether to share them.
