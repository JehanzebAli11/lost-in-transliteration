# Lost in Transliteration

**Do prompt-injection defences work in Urdu and Roman Urdu, and can a small model fix them?**

This is a small, reproducible study of prompt-injection defences in English, Urdu script, Roman Urdu and code-switched Urdu-English. It covers open-source injection classifiers (RQ1–2), five language models in an indirect-injection setup (RQ3), a small multilingual detector fine-tuned with a little Urdu-variant data (RQ4–5), and MCP security scanners facing tool poisoning (RQ6). [PLAN.md](PLAN.md) has the full design. The write-up draft is [writeup/post.md](writeup/post.md), revised after an independent review ([writeup/REVIEW.md](writeup/REVIEW.md); changes in [writeup/REVISION.md](writeup/REVISION.md)).

> **Status (2026-11-09):** every planned stage has run except the optional linear probe (RQ7). RQ3 was rerun with all five models at one output cap (1,024 tokens), and the owner rated 60 translations. The revised write-up awaits the owner's review. Findings about named vendors' tools will be disclosed to those vendors before any public post.

## Headline findings

Numbers are from `results/`, with n. `results/headline_numbers.json` carries every quoted table row by row.

- **Detection drops in Urdu variants for most classifiers, most in Roman Urdu.** Prompt Guard 2 (86M) catches 48.5% of English attacks and 11.5% in Roman Urdu (paired −37 points, 95% CI [−42, −32], n = 399); wolf-defender drops 20 points [−24, −16]. proventra goes the other way (+13 points in every variant, with more false positives). On 57 natively written attacks, wolf-defender catches 27 and Prompt Guard 2 catches 1. (`results/table_detection_drop.csv`, `results/classifiers_summary.csv`)
- **Over-blocking.** ProtectAI v2 flags 84.7% of benign Urdu-script text, against 2.0% in English (n = 300 each). Horizon is the one classifier with recall ≥ 0.90 and false positives ≤ 1% in all four translated forms; it catches 44/57 native attacks. (`results/table_detection_drop.csv`, `results/classifiers_summary.csv`)
- **Models.** All five at max_tokens 1,024 with no reply at the cap; Claude Haiku 5.5, Sonnet 5.5 and Opus 5.5 via the Anthropic Batch API at the API-default sampling (Haiku thinking disabled, Sonnet `thinking: between_tools`, Opus thinking on at effort low), GPT-5.4 nano and Gemma 4 31B via OpenRouter at temperature 0 (`results/rq3_settings.csv`, `results/rq3_truncation.csv`). Of 15 language-vs-English tests (exact McNemar, Holm-adjusted), one is significant: Haiku follows Urdu-script injections more often (52 vs 31 of 150 pairs, adjusted p = 0.011), and a heuristic marks most of the extra successes as quoting. Sonnet shows no language effect (73 vs 71); an earlier Sonnet "drop" was a truncation artefact. With an English document and an Urdu-variant attack (50 pairs per variant), nothing is significant. The leak goal never succeeded (0/196 per model). (`results/table_asr_gap.csv`, `results/rq3_leak_goal.csv`)
- **Fine-tuned detector.** The selection rule fixed before training picks mDistilBERT +100% Urdu data: 427/427 on test templates, 14/57 native attacks. XLM-R base, chosen afterwards, catches 52/57 native with 1 false positive (41/57 trained on English only). One seed; native attacks share payloads with the training data. (`results/headline_numbers.json`, `results/detector_summary.csv`)
- **MCP scanners.** Ramparts, mcp-guardian and Cisco mcp-scanner flag 0/10 of our subtle poisoned descriptions in English and Urdu script; Cisco flags 2/10 in Roman Urdu and code-switched through incidental keyword matches. (`results/mcp_summary.csv`)
- **Translations.** The owner (native speaker) rated 20 per variant: meaning preserved 60/60; naturalness 4.35 (Urdu script), 4.40 (Roman Urdu), 3.40 (code-switched). (`results/human_review_summary.csv`)

## Reproduce

```bash
make setup && make test        # 138 tests on fakes, no network or GPU
make connectivity && make todos
make smoke                      # ~10 items through every stage
make data                       # §3: generate, expand, translate, review sheet, native set
make phase1                     # §4: classifiers, Valar (optional), models (USD 40 cap)
make train && make eval-detector   # §5 (device auto: CUDA, then Apple MPS, then CPU)
make phase3                     # §6
make analyze                    # §7: figures/ and results/ tables, headline_numbers.json
```

`make help` lists every stage. Each stage can be resumed and caches model responses on disk, so reruns cost nothing. The final RQ3 numbers come from `make rq3-uniform-estimate`, `rq3-uniform-sanity` and `rq3-uniform` (all five models at max_tokens 1,024; Claude through Message Batches via the owner's jzapi gateway, the others through OpenRouter); replaced rows are kept in `results/rq3_claude_superseded.jsonl` and `results/rq3_superseded.jsonl`.

What actually ran where: data generation and translation used open-weight models on OpenRouter (owner-approved profile `cloud_openrouter`, $0.18); the classifier and MCP stages ran on CPU; detector training ran on a MacBook GPU (MPS, 8 minutes for 6 runs). Total API spend was $7.56, including runs later replaced (`results/spend.json`).

## Layout

```
config.yaml        models, prices, budget, seeds, sizes (TODO_VERIFY values are refused until checked)
prompts/           prompts for the open-weight models that create and translate all data
src/               one module per stage (see Makefile); src/mcp_fakes/ = Phase 3 servers
tests/             fakes + tests for every stage, incl. an end-to-end run producing all figures
data/  results/  figures/  writeup/
LOG.md             what ran, what broke, decisions;  HANDOFF.md  session handoffs
```

## Data provenance and safety

- Every data item was written or translated by an **open-weight** model: Gemma 4 31B, plus Qwen 3.5 27B for the natively written set. Claude, OpenAI and Gemini models are only ever *evaluated*; the code refuses to use them for data.
- All attack goals are benign: canary tokens, fake secrets, a fixed silly phrase, answering in French. Phase 3 ran only against local fake MCP servers holding a fake secret in `sandbox/`.
- The natively written set is "model-written natively in Urdu variants", not human-written.
- Trained detector weights stay in `models/` (git-ignored) and are not published.

## Licence

This repository is licensed under Apache-2.0 ([LICENSE](LICENSE)). Third-party models, classifiers and scanners keep their own licences. The detector bases are `FacebookAI/xlm-roberta-base` (MIT) and `distilbert/distilbert-base-multilingual-cased` (Apache-2.0), both checked 2026-11-09. Classifier licences are listed in `research/classifiers.md`.
