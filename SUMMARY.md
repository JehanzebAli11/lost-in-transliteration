# SUMMARY

_Status 2026-11-09: all planned stages have run except the optional RQ7 probe. RQ3 rerun with all five models at one output cap (1,024 tokens). Write-up revised after the independent review (`writeup/REVIEW.md` → `writeup/REVISION.md`), awaiting the owner._
_Every number comes from `results/`; `results/headline_numbers.json` carries each table row by row with its n._

## Headline numbers

- **RQ1, detection gap.** Prompt Guard 2 86M: English 0.485 → Roman Urdu 0.115, paired −0.370 [−0.421, −0.321], n = 399. wolf-defender: 0.963 → 0.762, −0.201 [−0.241, −0.158]. Horizon: 0.983 → 0.902, −0.080 [−0.108, −0.053]. proventra rises about +0.13 in every variant. Natively written attacks (n = 57): wolf-defender 27/57, Prompt Guard 2 1/57, Horizon 44/57, proventra 55/57. (`results/table_detection_drop.csv`, `results/classifiers_summary.csv`)
- **RQ2, false positives.** ProtectAI v2 on benign Urdu script 0.847 vs English 0.020 (n = 300). On benign Roman Urdu: ProtectAI 0.133, proventra 0.110, and 0.000–0.003 for the others. Horizon keeps recall ≥ 0.90 and false positives ≤ 0.010 in all four translated forms. (`results/table_detection_drop.csv`)
- **RQ3, attack success** (paired, 150 items per language; exact McNemar, Holm over 15 tests). All models at max_tokens 1,024, 0 replies at the cap; Claude 5.5 models via the Anthropic Batch API at the API-default sampling (Haiku thinking disabled, Sonnet `between_tools`, Opus thinking on, effort low), GPT-5.4 nano and Gemma 4 31B via OpenRouter at temperature 0 (`results/rq3_settings.csv`, `results/rq3_truncation.csv`). Only Haiku, Urdu script, is significant: 31 → 52 of 150, +0.140 [+0.067, +0.220], p 0.0008, Holm 0.011; strict (quote-marked replies removed) 20 vs 23. Opus Urdu script −0.107, Holm 0.159. Sonnet 73 / 71 / 72 / 75 (en / ur / roman / mixed). English document + Urdu-variant attack (50 pairs per variant): no test survives Holm. Leak goal 0/196 per model. (`results/table_asr_gap.csv`, `results/rq3_mentions_relabel.csv`, `results/rq3_leak_goal.csv`)
- **RQ4, fine-tuned detector** (one seed). On unseen test templates, XLM-R base trained on English only catches 96–107 of 104–108 per variant. mDistilBERT trained on English only catches 51/104 in Urdu script, and 103/104 with 25% of the Urdu data. The pre-set rule (best mean test AUC) picks mDistilBERT +100%: 427/427 on test. (`results/detector_summary.csv`, `headline_numbers.json` best_detector)
- **RQ5, generalisation.** The pre-set pick catches 14/57 native attacks and 28/80 held-out-style English. XLM-R base (chosen after the native results) catches 41/57 native trained on English only, 37/57 at +25% and 52/57 at +100% (1/57 false positives); held-out Roman Urdu 70/80 → 78/80. Native attacks share goals and payloads with the training data. (`results/detector_summary.csv`)
- **RQ6, MCP scanners** (10 servers per cell). Ramparts, mcp-guardian and Cisco mcp-scanner flag 0/10 poisoned descriptions in English and Urdu script; Cisco flags 2/10 in Roman Urdu and code-switched (incidental). wolf-defender on tool text flags 8/10 English and 9/10 Roman Urdu with 0/10 false alarms, but 5/10 false alarms on clean Urdu-script servers. (`results/mcp_summary.csv`)
- **Translation quality.** Owner review, 20 per variant: meaning preserved 60/60; naturalness ur 4.35, roman_ur 4.40, mixed 3.40 (12/20 ≤ 3). (`results/human_review_summary.csv`)

## Total API spend

$7.56 of the $40 cap: $0.18 for data, $7.38 for model evaluation including runs later replaced (Opus 5.5 $3.25, Sonnet 5.5 $3.10 over two full runs, GPT-5.4 nano $0.54, Haiku 5.5 $0.29 over the OpenRouter and Batch runs). The $0.18 is by stage (build, translate, native, MCP), so Gemma's own RQ3 calls count under evaluation. (`results/spend.json`, updated 2026-11-09T22:06:11+05:00)

## Skipped and why

- **RQ7 linear probe (fig9).** The optional stretch goal (PLAN §10); not run.
- **Phase 2 reduced to 3 data levels × 1 seed** (from 5 × 3; PLAN §11 fallback). The RQ4 curve therefore has no seed-variance bars. At today's speed (8 min for 6 runs on MPS), the full design is about 40 min more.
- **Valar.** Not run; no Valar results exist (optional, PLAN §0.7).
- **Back-translation check.** Not run (optional in PLAN §3.5).
- **Per-language thresholds.** Not tested.
- **Optional extra English training set (§5.2).** Not used (`detector.extra_english_dataset.hf_id: null`).

## Owner's next steps

1. Review the revised `writeup/post.md` and `writeup/application_blurb.md` (changes in `writeup/REVISION.md`). The blurb's last personal line is an agent draft; replace it with your own words.
2. Disclosure (PLAN §0.5) before any public post: send the Prompt Guard 2 (Meta), wolf-defender (Patronus Protect) and ProtectAI v2 findings to the vendors.
3. Optional: the full 5 × 3 detector design (about 40 min on the Mac), Valar (`VALAR_API_URL`/`VALAR_API_KEY`, then `make eval-valar analyze`).
4. Choose a licence. Trained weights stay in `models/` (git-ignored); decide whether to share them.
