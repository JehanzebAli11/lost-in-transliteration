# Dataset run on OpenRouter (2026-11-04, cloud session; no Spark)

`main`'s pipeline was run unchanged (`make data`) except for two things:
1. **`config.yaml`** (copy in `research/config_openrouter.yaml`):
   - Generator, translator and labeler use `google/gemma-4-31b-it`. The native writer uses `qwen/qwen3.5-27b`.
   - Both run on the `openrouter` endpoint, with reasoning off and prices added.
   - The budget cap is $3.
   - Both models are open-weight, so PLAN rule 8 holds. OpenRouter picks the host provider, so the provider is not pinned.
2. **`prompts/native/options.yaml`**, `ur` line (copy in `research/prompt_changes/`):
   - The native prompt is written in Roman Urdu, so Qwen answered in Roman Urdu even when asked for Urdu script. All 36 Urdu-script items were dropped as `wrong_script`.
   - The line now says explicitly to use Urdu letters, with an example. After the change, 12 Urdu-script items passed.
   - The `main` session should take over this line.

The owner rated all 20 smoke-test Roman Urdu items OK (`research/smoke_review.md`).

## Output (`data/`, `results/`)
| | en | ur | roman_ur | mixed |
|---|---|---|---|---|
| attacks (split: train 212 / test 108 / heldout_style 80) | 400 | 395 | 399 | 400 |
| benign | 300 | 300 | 300 | 300 |
| carriers | 30 | 30 | 30 | 30 |
| native (attack / benign) | – | 6 / 6 | 27 / 27 | 24 / 24 |

- 6 translations failed the survival check after retries; details in `results/translation_failures.jsonl`.
- The native set has 114 items, 6 short of the Urdu-script target. Drops: 24 `wrong_script`, 2 `label_disagree`.
- Total spend was **$0.18** over about 2,600 calls (`results/spend.json`).

Still open:
- The owner should rate `data/human_review_sample.csv` (60 rows).
- Run the back-translation check (`make backtranslate`).
