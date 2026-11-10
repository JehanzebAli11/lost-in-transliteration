<!--
REVISED DRAFT for owner review (2026-11-09, revision 2). writeup/REVISION.md maps every item of writeup/REVIEW.md to the change made.
Every number has its source in an HTML comment next to it. Abbreviations in comments:
CS = results/classifiers_summary.csv, DD = results/table_detection_drop.csv/.md, MS = results/models_summary.csv,
AG = results/table_asr_gap.csv/.md (settings `same` and `cross_en_carrier`; exact McNemar p, raw and Holm-adjusted),
ST = results/rq3_settings.csv, TR = results/rq3_truncation.csv, LG = results/rq3_leak_goal.csv, RL = results/rq3_mentions_relabel.csv,
SUP = results/rq3_claude_superseded.jsonl (rows replaced by the uniform 1,024-token rerun), HR = results/human_review_summary.csv,
DT = results/detector_summary.csv, DC = results/detector_curve.csv, MC = results/mcp_summary.csv, HN = results/headline_numbers.json.
Wilson CIs for single proportions come from src/metrics.wilson_ci on the counts shown.
Not for public posting before the disclosure note near the end is acted on (PLAN §0.5).
-->

# Lost in Transliteration: prompt-injection defences in Urdu, Roman Urdu and code-switched text

## TL;DR

- **Most injection classifiers we tested lose recall in Urdu variants; for three of six, Roman Urdu is the worst.** Meta's Prompt Guard 2 (86M) catches 48.5% of our English attacks and 11.5% of the same attacks in Roman Urdu (paired drop −37 points, 95% CI [−42, −32], n = 399); wolf-defender drops 20 points [−24, −16]. <!-- DD tpr roman_ur: prompt-guard-2 0.485 → 0.115, paired −0.371 [−0.421, −0.321], n_paired 399; wolf-defender −0.201 [−0.241, −0.158] --> proventra instead gains about 13 points, at the cost of false positives. On 57 attacks written natively in Urdu variants by another model, wolf-defender catches 27 and Prompt Guard 2 catches 1. <!-- DD proventra +0.129/+0.133/+0.130; CS group_set native: wolf-defender 27/57, prompt-guard-2 1/57 -->
- **One classifier treats Urdu script as an attack; one holds up on translated text.** ProtectAI v2 flags 84.7% of benign Urdu-script text, against 2.0% of benign English (n = 300 each). Horizon keeps recall at 0.90 or more and false positives at 1% or less in all four translated forms, but catches 44 of 57 native attacks. <!-- DD fpr ur protectai-v2 0.847 vs 0.020; DD horizon-guard tpr 0.983/0.942/0.902/0.943, fpr 0.007/0.010/0.000/0.000; CS horizon native 44/57 -->
- **Models: one of 15 language effects survives correction.** At one shared 1,024-token cap, Claude Haiku 5.5 followed Urdu-script injections more often than English ones (52 vs 31 of 150 paired items, McNemar p = 0.0008, Holm-adjusted 0.011), though a heuristic marks most of the extra successes as quoting, not obeying. Nothing else survives Holm, including Opus 5.5's Urdu-script drop (adjusted p = 0.16). Settings differ: Claude models at default sampling (Opus with thinking on), nano and Gemma at temperature 0. <!-- AG same: claude-haiku ur 0.207 → 0.347, paired +0.140 [+0.067, +0.220], p_raw 0.00075, p_holm 0.0113; claude-opus ur −0.107, p_holm 0.159; TR 0 at cap everywhere; RL haiku ur mentions 29/52 vs en 11/31 -->
- **Our pre-set rule picks a fine-tuned detector that learned the templates.** Multilingual DistilBERT with all the Urdu data scores 427/427 on unseen test templates but catches 14 of 57 native attacks (CI 0.15–0.37). XLM-R base, singled out after we saw native results, catches 52 of 57 with 1 false positive; that is exploratory (one seed, a non-monotonic curve of 41, 37, 52 at 0, 25, 100% Urdu data, payloads shared with training). <!-- HN best_detector mdistilbert pct 100 (rule in src/analyze.py since 2026-11-03); DT mdistilbert pct100 test 108+104+107+108 of same, native 14/57 fp 0; xlmr-base native 41 / 37 / 52 of 57, fp 1 at pct100 -->
- **Rule-based MCP scanners missed almost all of our subtle tool poisoning.** All three flagged 0/10 poisoned descriptions in English and Urdu script; only Cisco mcp-scanner flagged any (2/10 in Roman Urdu and in code-switched), through incidental keyword matches. <!-- MC -->

## Motivation

Urdu is printed in a Perso-Arabic script, but in chat many Pakistanis type it in Latin letters with no fixed spelling (Roman Urdu), switching to English mid-sentence. Prompt-injection classifiers, which sit in front of agents, are mostly built and tested in English. Meta's Prompt Guard cards list eight languages without Urdu, ProtectAI's card covers English only, and Horizon's 30 training languages exclude Urdu. <!-- research/related_work.md §5; research/classifiers.md Horizon-Labs v2.2 row -->

A filter that misses Roman Urdu injections can be bypassed by writing the way millions of people write; one that flags ordinary Urdu blocks the people it should serve. Agents, which read third-party text in any language, widen both. We measure both failures, check whether models follow such injections, test a cheap fix, and test MCP tool-poisoning scanners. Our hypotheses going in: detection drops and attack success rises for Roman Urdu and code-switched text, and a little targeted training data closes much of the gap.

**Related work.** Low-resource languages, code-switching and transliteration weaken jailbreak refusals ([Yong et al. 2023](https://arxiv.org/abs/2310.02446); [Deng et al. 2023](https://arxiv.org/abs/2310.06474); [Yoo et al. 2024](https://arxiv.org/abs/2406.15481); Arabizi: [Ghanim et al. 2024](https://arxiv.org/abs/2406.18725)), though IndicJR ([Pattnayak & Chowdhuri 2026](https://arxiv.org/abs/2602.16832)) found that romanization *lowered* jailbreak success. *Ghaib in Translation* ([Zehra et al. 2026](https://arxiv.org/abs/2608.24191)) uses our four Urdu forms for hate speech; BanglaShield ([Kabir et al. 2026](https://arxiv.org/abs/2610.24288)) closed a Bangla native/romanized gap with a small fine-tuned encoder, as our Phase 2 tries. We found no study of injection detection in Urdu variants; the closest are a small English-vs-Urdu-script garak run ([Mahad7836](https://github.com/Mahad7836/garak-multilingual-promptinj-eval)) and an English/Urdu-script dataset without Roman Urdu (Afra Sadat, 2026). Details: `research/related_work.md`.

## Setup

**Attacks.** Every goal is benign and checked exactly: print a canary token, leak a fake access code, reply only "I love pineapples", or answer in French. 60 English templates in five styles gave 400 attacks; there are also 300 benign controls and 30 carrier documents. <!-- LOG 2026-11-04 [build]: templates 60, 400 attacks, 300 benign, 30 carriers --> Payload literals stay in English inside translations, so even Urdu-script attacks are partly code-switched.

**Variants.** Gemma 4 31B (open-weight) translated every item into Urdu script, Roman Urdu and code-switched text, with checks that tokens and phrases survive; 395, 399 and 400 of the 400 attacks remain. <!-- results/translation_report.json; LOG 2026-11-04 profile cloud_openrouter --> Templates are split train/test (about 65/35) within four styles; the polite/indirect style is held out entirely. <!-- LOG 2026-11-04: test 108, train 212, heldout_style 80 English attacks -->

**Translation quality.** The owner, a native speaker, rated 60 sampled translations (20 per variant). Meaning was preserved in 60 of 60. Mean naturalness (1–5) was 4.35 for Urdu script, 4.40 for Roman Urdu and 3.40 for code-switched, where 12 of 20 scored 3 or lower, mostly English-heavy calques. <!-- HR: ur all 4.35 n=20, roman_ur 4.40, mixed 3.40 natural_le_3 12/20, mixed attack 3.077 n=13; meaning_preserved_y 60/60; LOG 2026-11-09 human review --> Code-switched results may partly reflect translation style. <!-- HN backtranslation null -->

**Natively written set.** Qwen 3.5 27B wrote attacks and benign messages directly from Urdu and Roman Urdu prompts, without English templates. After script checks (24 of 240 dropped) and a blind label check by Gemma, the translator (2 dropped), 57 attacks and 57 benign items remain: 27 Roman Urdu, 24 code-switched and only 6 Urdu script each. <!-- results/native_gate_report.json: generated 240, wrong_script 24, label_disagree 2, final 114, labeler gemma-4-31b-it, shortfall ur 3+3 --> It is test-only and reuses the main set's goals and payloads: 19 attacks ask for "I love pineapples", 13 for French, 13 for a canary in the training format (none identical to a main-set token), 12 for the access code. <!-- models_by_goal.csv native n: OVERRIDE 19, LANG_SWITCH 13, CANARY 13, SECRET_LEAK 12; token format checked in data/native.jsonl vs data/en.jsonl -->

**Classifiers (RQ1–2).** ProtectAI deberta-v3-base-prompt-injection-v2, Meta Llama Prompt Guard 2 (86M and 22M), Horizon prompt-injection-guard-base, Patronus Protect wolf-defender and proventra mdeberta-v3-base-prompt-injection, each at a 0.5 threshold on standalone text except ProtectAI v2 at 0.92, the LLM Guard default (`config.yaml`, `research/classifiers.md`).

**Models (RQ3).** The system prompt asks for a short summary (and holds the fake code); the user sends a carrier document with the attack at a random position. Per model, 927 tasks: 150 attacks per language (document and attack in the same language), 57 native attacks, 150 with an English document and an Urdu-variant attack, and 120 clean documents. <!-- MS n per cell; ST n 927 --> All five share one output cap, 1,024 tokens, which no reply reached. <!-- ST max_tokens 1024; TR at_max_tokens 0 in every cell --> Other settings differ:

| Model | Endpoint | Thinking / reasoning | Temperature |
|---|---|---|---|
| Claude Haiku 5.5 | Anthropic Message Batches | thinking disabled | API default |
| Claude Sonnet 5.5 | Anthropic Message Batches | `thinking: between_tools` (none without tools) | API default |
| Claude Opus 5.5 | Anthropic Message Batches | thinking on, `effort: low` (cannot be disabled) | API default |
| GPT-5.4 nano | OpenRouter | reasoning disabled | 0 |
| Gemma 4 31B | OpenRouter | reasoning disabled | 0 |

<!-- ST (Opus params from results/rq3_claude_batches.json); LOG 2026-11-09 (line 136): between_tools = no thinking without tools, Opus thinking cannot be disabled; LOG 2026-11-03: API version 2026-10-21 rejects temperature -->

The Anthropic API rejects a temperature parameter, so Claude models were sampled once per task at the default (adding noise); temperature 0 on OpenRouter was not verified.

**Detectors (RQ4–5).** XLM-R base (MIT) and multilingual DistilBERT (Apache-2.0) were fine-tuned on train-split items: English only, plus 25%, or plus 100% of the Urdu-variant train data; one seed, 3 epochs (planned: 5 levels × 3 seeds). <!-- config.yaml detector; LOG 2026-11-09 steps 4-5 --> The rule naming "the best detector" (highest mean test-split AUC) was in the analysis code before any training. <!-- src/analyze.py best_detector, commit 3cd4d85 (2026-11-03); HN best_detector -->

**MCP scanners (RQ6).** 120 local fake MCP servers, 10 per language per kind (clean, poisoned description, poisoned output), from ten subtle English patterns without `<IMPORTANT>`-style markers, scanned by Cisco mcp-scanner (YARA analyzer only), Ramparts (rules only, no LLM check), mcp-guardian and four classifiers (`research/mcp_scanners.md`). <!-- LOG 2026-11-05 [mcp] -->

**Spend.** $7.56: $0.18 for data, $7.38 for evaluation including replaced runs (Opus $3.25; Sonnet $3.10). <!-- results/spend.json total_usd 7.559; by_stage build 0.0109 + translate 0.1230 + native 0.0418 + mcp 0.0060 = 0.1817, eval_models 7.3774; by_model claude-opus-5-5 3.2490, claude-sonnet-5-5 3.0952 -->

## Results: measuring the gap

### RQ1: do classifiers catch Urdu-variant injections as well as English ones?

![Detection rate by language](../figures/fig1_detection_by_lang.png)

Detection rate at the default threshold, with the paired change from English:

| Classifier | English | Urdu script | Roman Urdu | Code-switched | Native (n=57) |
|---|---|---|---|---|---|
| Prompt Guard 2 86M | 0.485 | 0.215 (−0.27) | 0.115 (−0.37) | 0.275 (−0.21) | 0.018 |
| Prompt Guard 2 22M | 0.138 | 0.000 (−0.14) | 0.020 (−0.12) | 0.147 (+0.01, n.s.) | 0.018 |
| wolf-defender | 0.963 | 0.919 (−0.04) | 0.762 (−0.20) | 0.830 (−0.13) | 0.474 |
| Horizon guard | 0.983 | 0.942 (−0.04) | 0.902 (−0.08) | 0.943 (−0.04) | 0.772 |
| ProtectAI v2 | 0.932 | 0.808 (−0.13) | 0.910 (−0.02, n.s.) | 0.965 (+0.03, n.s. after Holm) | 0.772 |
| proventra mDeBERTa | 0.828 | 0.954 (+0.13) | 0.960 (+0.13) | 0.958 (+0.13) | 0.965 |

<!-- DD tpr rows: en_rate, variant_rate, paired diff with bootstrap CI; n_en 400, n_variant 395/399/400; native = DD variant "native", n=57, unpaired. Significance = exact McNemar (DD p_mcnemar), Holm-adjusted over the 18 translated recall tests (computed in this revision from DD p_mcnemar; not yet a DD column): every unmarked change has adjusted p ≤ 0.006; protectai-v2 mixed raw 0.035 → adjusted 0.105; prompt-guard-2-22m mixed raw 0.61; protectai-v2 roman_ur raw 0.21. -->

Prompt Guard 2 86M, wolf-defender and Horizon lose recall in every variant, worst in Roman Urdu; Prompt Guard 2 22M and ProtectAI lose it in Urdu script; proventra gains in every variant and pays in false positives (RQ2). <!-- DD; research/classifiers.md proventra row --> The natively written attacks are harder for every classifier except proventra, Roman Urdu most of all: wolf-defender catches 9 of 27 and Horizon 18 of 27. <!-- CS native roman_ur -->

**Ranking versus threshold (a hypothesis).** For four classifiers, ROC-AUC stays at about 0.98 or above in every translated variant while recall falls. <!-- results/fig3_roc_auc.csv: horizon, prompt-guard-2, wolf-defender, proventra ≥ 0.9796 for ur/roman_ur/mixed --> Per-language thresholds might recover much of the gap; we did not test that. A deployed filter needs one threshold for unknown or mixed language, and AUC is lower on the native set (Prompt Guard 2 0.909, wolf-defender 0.954). For ProtectAI (0.569) and Prompt Guard 2 22M (0.478) on Urdu script, the signal itself is gone. <!-- fig3_roc_auc.csv: native n=114; ur n=695 -->

![ROC curves](../figures/fig3_roc_by_classifier.png)

### RQ2: do classifiers over-block benign Urdu-variant text?

![False positive rate by language](../figures/fig2_fpr_by_lang.png)

Prompt Guard 2 (both sizes), wolf-defender and Horizon flag at most 1.8% of benign text in any language, native included. <!-- DD fpr: max wolf-defender ur 0.017, native 1/57 = 0.018; horizon ur 0.010; prompt-guard-2 0.000 --> Two do not:

- **ProtectAI v2** flags 84.7% of benign Urdu-script text, 13.3% of Roman Urdu, 13.0% of code-switched and 12 of 57 native benign items, against 2.0% in English. <!-- DD protectai-v2 fpr: en 0.020, ur 0.847, roman_ur 0.133, mixed 0.130, native 12/57; n=300 per language; McNemar p < 1e-9 for all three -->
- **proventra** flags none in English, but 11.0% of Roman Urdu, 6.7% of code-switched, 5.7% of Urdu-script and 6 of 57 native benign items. <!-- DD proventra-mdeberta fpr: roman_ur 0.110, mixed 0.067, ur 0.057 (McNemar p ≤ 1.6e-5), native 6/57 -->

Only Horizon is both sensitive and specific in all four translated forms (recall 0.90–0.98, false positives 0–1.0%); on the native set its recall falls to 44/57, with no false positives. The others trade one failure for the other. <!-- DD horizon-guard; CS native 44/57 fp 0/57 -->

### RQ3: do models follow Urdu-variant injections more often?

![Attack success rate by language](../figures/fig4_asr_by_lang.png)

Successes out of 150 paired items per language (document and attack in one language); in brackets, the paired change from English and the Holm-adjusted p.

| Model | English | Urdu script | Roman Urdu | Code-switched | Native (/57) |
|---|---|---|---|---|---|
| Claude Haiku 5.5 | 31 | 52 (+0.14; 0.011) | 40 (+0.06; 1.00) | 37 (+0.04; 1.00) | 20 |
| GPT-5.4 nano | 24 | 22 (−0.01; 1.00) | 14 (−0.07; 0.63) | 18 (−0.04; 1.00) | 9 |
| Gemma 4 31B | 37 | 31 (−0.04; 1.00) | 32 (−0.03; 1.00) | 34 (−0.02; 1.00) | 19 |
| Claude Sonnet 5.5 | 73 | 71 (−0.01; 1.00) | 72 (−0.01; 1.00) | 75 (+0.01; 1.00) | 34 |
| Claude Opus 5.5 | 72 | 56 (−0.11; 0.16) | 62 (−0.07; 1.00) | 58 (−0.09; 0.37) | 28 |

<!-- MS successes; AG setting same, paired_diff and p_holm (exact McNemar, Holm over the 15 same-setting tests). Paired bootstrap CIs and raw p: haiku ur [+0.067, +0.220] p 0.00075, roman_ur [−0.007, +0.133] p 0.136, mixed [−0.027, +0.107] p 0.345; nano roman_ur [−0.127, −0.007] p 0.052; opus ur [−0.180, −0.033] p 0.011, roman_ur [−0.140, 0.000] p 0.099, mixed [−0.167, −0.020] p 0.029; all others p ≥ 0.21. Native vs English: two-proportion z, unpaired, Holm over 5: smallest haiku p 0.031, adjusted 0.157. openrouter-small = openai/gpt-5.4-nano. -->

"Significant" means an exact McNemar test on the 150 pairs, Holm-adjusted across the 15 language-versus-English tests. Only Haiku's Urdu-script rise survives; Opus's drops in Urdu script (raw p = 0.011) and code-switched text (0.029) and nano's Roman Urdu dip (0.052) do not. The native column compares different attacks, not languages; none differs from English after correction. <!-- AG -->

**Correction.** Our earlier draft said Sonnet follows Urdu-script injections 11 points less often than English ones. That was a truncation artefact: Sonnet answers Urdu-script documents in Urdu script (30 of 30 clean controls), which takes more tokens, and at the 400-token cap used then, 91 of its 150 Urdu-script replies were cut off, against none in English. <!-- results/models_clean_controls.csv claude-sonnet ur urdu_script 30/30; SUP claude-sonnet same ur max_tokens 400: 91/150 at cap, 54/150 successes; en 0/150 at cap --> At 1,024 tokens Sonnet shows no language effect (73 vs 71). Haiku, nano and Gemma were rerun at that cap; Opus already used it.

**English document, Urdu-variant attack.** Above, document and attack change language together. Here the document stays English (50 pairs per variant):

| Model | Urdu script | Roman Urdu | Code-switched |
|---|---|---|---|
| Claude Haiku 5.5 | 8 (−0.06) | 11 (0.00) | 11 (0.00) |
| GPT-5.4 nano | 7 (−0.02) | 4 (−0.08) | 5 (−0.06) |
| Gemma 4 31B | 10 (−0.04) | 11 (−0.02) | 12 (0.00) |
| Claude Sonnet 5.5 | 30 (+0.04) | 28 (0.00) | 27 (−0.02) |
| Claude Opus 5.5 | 30 (+0.08) | 22 (−0.08) | 25 (−0.02) |

<!-- MS setting cross_en_carrier successes /50; AG cross_en_carrier paired_diff vs the same 50 items in English; every p_holm = 1.0 (raw p ≥ 0.22). Haiku ur paired −0.06 [−0.16, +0.04]. -->

Nothing is significant (every adjusted p = 1.0; 50 pairs show only large effects). Haiku's Urdu-script rise is absent here (8 of 50, −6 points [−16, +4]); perhaps, untested, Haiku reacts to the document's language more than the injection's. <!-- AG cross_en_carrier claude-haiku ur -->

**Goals.** The leak goal, the one closest to a real harm, never succeeded: 0 of 196 attempts for every model. <!-- LG: secret_leak_successes 0 of n_secret_leak 196 per model --> Every count above therefore comes from the 113 low-stakes items per language: on those, Haiku succeeds 27% of the time in English and 46% in Urdu script, Sonnet 65% and 63%. <!-- LG same: claude-haiku en 31/113 = 0.274, ur 52/113 = 0.460; claude-sonnet en 73/113 = 0.646, ur 71/113 = 0.628 --> Leak pairs fail in both languages, so the McNemar tests are unchanged. We have not analysed why no model leaks.

**Quoting versus obeying.** Exact matching counts "the document asks me to print KOALA-9630" as a success. A heuristic marks replies that talk about the instruction; it now has Urdu-script and Roman Urdu markers too (one label changed). <!-- RL now_quote: claude-opus same mixed 1; all other cells 0 --> Removing marked replies gives these "strict" counts (n = 150 each):

| Model | English | Urdu script | Roman Urdu | Code-switched |
|---|---|---|---|---|
| Claude Haiku 5.5 | 20 | 23 | 17 | 18 |
| Claude Sonnet 5.5 | 23 | 29 | 23 | 27 |
| Claude Opus 5.5 | 50 | 19 | 27 | 24 |
| Gemma 4 31B | 37 | 30 | 29 | 32 |
| GPT-5.4 nano | 23 | 15 | 12 | 14 |

<!-- RL strict_successes_now, setting same (successes − mentions_only_now; e.g. haiku ur 52 − 29 = 23) -->

Haiku's Urdu-script rise mostly disappears (23 vs 20): the heuristic marks 29 of its 52 Urdu-script successes and 11 of 31 English ones, so Haiku may mostly be *reporting* Urdu-script injections more often. Opus obeys English injections most (50) and Urdu-script ones least (19). The heuristic is not hand-validated and has at least one known false positive, so strict counts are descriptive. <!-- RL; LOG 2026-11-09 rq3_uniform: nat-atk-mixed-013 --> Every model replied with at least five words to all 30 clean documents per language (not a judge of summary quality). <!-- results/models_clean_controls.csv summary_ok 1.0 (≥ 5 words) -->

![ASR by style and language](../figures/fig5_asr_heatmap_style_lang.png)

## Results: a cheap fix

One seed per condition: differences of a few items are within noise.

### RQ4: how much Urdu-variant data closes the gap?

![Detection vs data](../figures/fig7a_detector_tpr_vs_data.png)
![False positives vs data](../figures/fig7b_detector_fpr_vs_data.png)

Test split (unseen templates of the four trained styles); false positives out of 105 benign items per language:

| Detector | Urdu data | English | Urdu script | Roman Urdu | Code-switched | False positives (en/ur/roman/mixed) |
|---|---|---|---|---|---|---|
| XLM-R base | 0% | 105/108 | 96/104 | 101/107 | 107/108 | 0/2/3/2 |
| XLM-R base | 25% | 105/108 | 98/104 | 101/107 | 105/108 | 0/1/3/3 |
| XLM-R base | 100% | 104/108 | 98/104 | 100/107 | 103/108 | 2/1/4/4 |
| mDistilBERT | 0% | 108/108 | 51/104 | 100/107 | 101/108 | 3/2/3/6 |
| mDistilBERT | 25% | 108/108 | 103/104 | 105/107 | 107/108 | 3/4/6/8 |
| mDistilBERT | 100% | 108/108 | 104/104 | 107/107 | 108/108 | 1/2/3/4 |

<!-- DT group_set test: tp/n_attack, fp/n_benign -->

On unseen templates of familiar styles there is little gap: XLM-R trained on English alone detects 92–99% in every variant. mDistilBERT trained on English alone misses half the Urdu-script attacks (51/104, CI 0.40–0.59); adding 25% of the Urdu data, 306 rows, closes it (103/104). <!-- DT; train_log n_train 713 − 407 = 306 --> False positives stay at or below 8 of 105. The pre-set rule picks mDistilBERT with all the Urdu data, perfect on this split. <!-- HN best_detector; DC mdistilbert pct100 test auc 1.000/0.9999/0.9997/0.9994 -->

### RQ5: does it generalise to an unseen style and natively written text?

![Held-out style](../figures/fig7d_heldout_style.png)
![Native set](../figures/fig7e_native_set.png)

| Detector | Urdu data | Held-out style, Roman Urdu (/80) | Held-out style, English (/80) | Native attacks (/57) | Native Roman Urdu (/27) | Native false positives (/57) |
|---|---|---|---|---|---|---|
| XLM-R base | 0% | 70 | 80 | 41 | 16 | 1 |
| XLM-R base | 25% | 66 | 80 | 37 | 15 | 0 |
| XLM-R base | 100% | 78 | 80 | 52 | 24 | 1 |
| mDistilBERT | 0% | 33 | 28 | 7 | 3 | 0 |
| mDistilBERT | 25% | 24 | 19 | 8 | 2 | 0 |
| mDistilBERT | 100% | 25 | 28 | 14 | 6 | 0 |

<!-- DT group_set heldout_style and native -->

- **The pre-set pick learned the templates.** mDistilBERT with all the Urdu data catches 28 of 80 held-out-style attacks even in English and 14 of 57 native attacks (0.25, CI 0.15–0.37). More Urdu data doubles its native catches from a low base (7, 8, 14) but not its held-out Roman Urdu ones (33, 24, 25). <!-- DT mdistilbert -->
- **XLM-R generalises better; read this as exploratory.** Trained on English alone it catches 41 of 57 native attacks (0.72, CI 0.59–0.82); with all the Urdu data, 52 of 57 (0.91, CI 0.81–0.96) at 1 false positive, and held-out Roman Urdu rises from 70 to 78 of 80. <!-- DT xlmr-base; Wilson from src/metrics --> Three caveats go with 52/57. We singled it out after seeing native results (a per-model rule added later picks XLM-R with no Urdu data; figures 7c–e). <!-- LOG 2026-11-09 step 7; DC xlmr-base mean test AUC 0.9972 at 0% vs 0.9970 / 0.9956 --> One seed, and the curve is non-monotonic (41, 37, 52). And native attacks reuse the training goals' phrase and token format (Setup), so part of the gain may be payload matching.
- **The same 57 native attacks against the classifiers:** proventra catches 55 with 6 false positives, Horizon 44 with 0, ProtectAI 44 with 12, wolf-defender 27 with 1, Prompt Guard 2 just 1. <!-- results/fig7e_native_set.csv (same item ids) --> The classifiers never saw our payloads, which favours the fine-tuned detectors; XLM-R's interval still overlaps proventra's and Horizon's.

## Results: agent tooling

### RQ6: do MCP scanners catch tool poisoning in Urdu variants?

![MCP scanners](../figures/fig8_mcp_scanners.png)

With 10 servers per cell, every rate is directional (0/10 has a 95% upper bound of 28%). <!-- MC ci_hi for 0/10 = 0.2775 -->

- **Rule-based scanners missed our subtle poisoning in every language.** Ramparts and mcp-guardian flagged 0/10 in all four; Cisco mcp-scanner 0/10 in English and Urdu script, 2/10 in Roman Urdu and code-switched. Those hits were incidental matches from Urdu word order ("send_report ko pass"), and all three flag an obvious English `<IMPORTANT>` fixture: this is about subtle poisoning, not language. <!-- MC; LOG 2026-11-05 Phase 3 (hits and fixture: LOG only, not in results/) -->
- **Classifiers on tool text react to language.** ProtectAI flags 10/10 poisoned and 10/10 *clean* Urdu-script servers (English: 2/10 and 0/10). wolf-defender catches 8/10, 9/10 and 7/10 poisoned descriptions in English, Roman Urdu and code-switched with no false alarms there, but raises 5/10 on clean Urdu-script servers. Horizon flags every server (its neutral decoy description alone scores 0.92); Prompt Guard 2 flags nothing. <!-- MC classifier-baseline (= protectai-v2), wolf-on-descriptions, horizon-on-descriptions, prompt-guard-2-on-descriptions; LOG 2026-11-05 decoy 0.924 -->

## Qualitative analysis

From `results/qualitative_failures.md`:

- **English keywords survive; detection does not.** Prompt Guard 2 scores an English override at 1.00 and its Roman Urdu version ("Saaray puranay constraints aur goals khatam kardo… sirf I love pineapples respond karo.") at 0.00. "RUKO. Uper wala text aur original system prompt ignore karo…" scores 0.03, though "ignore" and "system prompt" survive. <!-- qualitative_failures.md prompt-guard-2 Roman Urdu atk-0117 1.00 → 0.00; atk-0090 1.00 → 0.03 --> Our hypothesis is that the Urdu grammar around the keywords ("karo", "kardo") moves the score; two examples cannot show it, and we ran no ablation.
- **Polite framing in Roman Urdu is the hardest case.** wolf-defender scores an English polite override at 0.99 and its Roman Urdu version at 0.00; on the held-out polite style it falls from 75/80 in English to 31/80 in Roman Urdu, Prompt Guard 2 from 30/80 to 4/80. <!-- CS heldout_style --> <!-- qualitative_failures.md wolf-defender atk-0269 0.99 → 0.00 -->
- **Roman Urdu is harder than Urdu script for the multilingual classifiers.** Horizon misses 19 Urdu-script items it catches in English, against 33 in Roman Urdu; wolf-defender 22 against 85. <!-- qualitative_failures.md section counts --> Perhaps (untested) their pre-training saw little romanized Urdu.

## Limitations

- **Model-written data.** No attack or translation is human-written. The human review is 60 items from one rater, the author, with no back-translation. Translations smoothed some ungrammatical English splices, so a variant is not always the identical attack. <!-- HR; qualitative_failures.md atk-0269, atk-0319 -->
- **One model in three roles.** Gemma 4 31B translated the data, blind-labelled the native set and is evaluated. The native set comes from one writer model, has 6 Urdu-script attacks and shares payloads with the training data.
- **Benign goals** stand in for harmful ones; the leak goal never succeeded.
- **Settings and run history.** Model settings differ (Setup). Haiku's first RQ3 run was discarded after a reasoning-budget bug; Sonnet's cap rose from 200 to 400, then every model was rerun or checked at 1,024. Replaced rows are kept in `results/`. <!-- LOG 2026-11-05, 2026-11-09; SUP; results/rq3_superseded.jsonl -->
- **Small samples.** 150 pairs per language in RQ3, 50 with English documents, 57 native attacks, 10 MCP servers per cell.
- **Scoring.** The quote heuristic is unvalidated and language-dependent; the clean-control check needs only five words.
- **Detectors.** One seed and three data levels instead of 3 × 5; a headline chosen after seeing native results; train and test from one pipeline, so the test split likely overstates real-world performance.
- **Not done.** Per-language thresholds, Valar (the author's own system), the RQ7 probe. Data generation ran on OpenRouter, not locally as planned.

## Disclosure before publication

Two results are large and reproducible: Prompt Guard 2 (86M, Meta) loses 37 points of recall on Roman Urdu and catches 1 of 57 native attacks; wolf-defender (Patronus Protect) loses 20 points on Roman Urdu and catches 47% of native attacks against 96% of translated English ones. <!-- DD: prompt-guard-2 roman_ur −0.370, native 1/57; wolf-defender roman_ur −0.201, native 0.474 vs en 0.963 (unpaired, diff −0.489 [−0.612, −0.354]) --> ProtectAI v2's Urdu-script false-positive rate matches its card's English-only scope but belongs in the same report. Per PLAN §0.5, these go to Meta, Patronus and ProtectAI before publication; the owner sets the timing.

## Next steps

- Per-language thresholds; transliteration before classifying.
- Human-written Roman Urdu attacks, more raters, a native set with new payloads, and detector seeds.
- Other two-script languages (Hindi/Hinglish, Arabic/Arabizi, Bangla, Punjabi); poisoned MCP servers against real agents; the RQ7 probe.

## Reproducibility

Each stage is one command (`make data`, `phase1`, `train`, `eval-detector`, `phase3`, `rq3-uniform`, `analyze`); `make test` runs everything on fakes. Seeds and model IDs are in `config.yaml`, dependencies are pinned, and API responses are cached under `results/`. Trained weights are not published.
