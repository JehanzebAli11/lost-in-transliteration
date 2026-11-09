# SUMMARY

_Status 2026-11-09: all planned stages have run except the optional RQ7 probe. Write-up drafted for owner review._
_Every number comes from `results/`; `results/headline_numbers.json` carries each table row by row with its n._

## Headline numbers

- **RQ1, detection gap.** Prompt Guard 2 86M: English 0.485 → Roman Urdu 0.115, paired −0.370 [−0.421, −0.321], n = 399. wolf-defender: 0.963 → 0.762, −0.201 [−0.241, −0.158]. Horizon: 0.983 → 0.902, −0.080 [−0.108, −0.053]. Natively written attacks (n = 57): wolf-defender 27/57, Prompt Guard 2 1/57, Horizon 44/57, proventra 55/57. (`results/table_detection_drop.csv`, `results/classifiers_summary.csv`)
- **RQ2, false positives.** ProtectAI v2 on benign Urdu script 0.847 vs English 0.020 (n = 300). On benign Roman Urdu: ProtectAI 0.133, proventra 0.110, and 0.000–0.003 for the others. (`results/table_detection_drop.csv`)
- **RQ3, attack success gap** (paired, 150 items). Haiku 5.5, Urdu script +0.107 [+0.033, +0.180]. GPT-5.4 nano, Roman Urdu −0.080 [−0.140, −0.020]. Sonnet 5.5, Urdu script −0.113 [−0.193, −0.027]. Opus 5.5, Urdu script −0.107 [−0.180, −0.033]. Gemma is flat. Strict (likely quotes removed) counts are in `headline_numbers.json` → `models`. (`results/table_asr_gap.csv`)
- **RQ4, fine-tuned detector** (one seed). On unseen test templates, XLM-R base trained on English only already catches 96–107 of 104–108 per variant. mDistilBERT trained on English only catches 51/104 in Urdu script, and 103/104 with 25% of the Urdu data. (`results/detector_summary.csv`)
- **RQ5, generalisation.** On the native set, XLM-R base catches 41/57 trained on English only and 52/57 with 100% of the Urdu data (1/57 false positives). On held-out-style Roman Urdu, 70/80 → 78/80. mDistilBERT catches at most 14/57 native and 28/80 held-out English: it learned the templates. (`results/detector_summary.csv`)
- **RQ6, MCP scanners** (10 servers per cell). Ramparts, mcp-guardian and Cisco mcp-scanner flag 0/10 poisoned descriptions in English; Cisco flags 2/10 in Roman Urdu and code-switched (incidental). wolf-defender on tool text flags 8/10 English and 9/10 Roman Urdu with 0/10 false alarms, but 5/10 false alarms on clean Urdu-script servers. (`results/mcp_summary.csv`)

## Total API spend

$5.71 of the $40 cap: $0.18 for data, $5.52 for model evaluation (Opus 5.5 $3.25, Sonnet 5.5 $1.54, GPT-5.4 nano $0.36, Haiku 5.5 $0.24). The $0.18 is by stage (build, translate, native, MCP), so Gemma's own RQ3 calls count under evaluation. (`results/spend.json`, updated 2026-11-09T20:43:51+05:00)

## Skipped and why

- **RQ7 linear probe (fig9).** This was the optional stretch goal (PLAN §10); not run.
- **Phase 2 reduced to 3 data levels × 1 seed** (from 5 × 3; PLAN §11 fallback) for the Mac and the deadline. The RQ4 curve therefore has no seed-variance bars. At today's speed (8 min for 6 runs on MPS), the full design is about 40 min more.
- **Valar.** Not run; no Valar results exist (optional, PLAN §0.7).
- **Human review of translations.** `data/human_review_sample.csv` has 0/60 rows rated.
- **Back-translation check.** Not run (optional in PLAN §3.5).
- **Optional extra English training set (§5.2).** Not used (`detector.extra_english_dataset.hf_id: null`).

## Owner's next steps

1. Review `writeup/post.md` and `writeup/application_blurb.md`. The blurb's last personal line is an agent draft; replace it with your own words.
2. Disclosure (PLAN §0.5) before any public post: send the Prompt Guard 2 (Meta), wolf-defender (Patronus) and ProtectAI v2 findings to the vendors.
3. Fill in `data/human_review_sample.csv` (~1 hour, 60 rows), then rerun `make analyze` if any numbers change.
4. Optional: run Valar (`VALAR_API_URL`/`VALAR_API_KEY`, then `make eval-valar analyze`), and the full 5 × 3 detector design.
5. Choose a licence. Trained weights stay in `models/` (git-ignored); decide whether to share them.
