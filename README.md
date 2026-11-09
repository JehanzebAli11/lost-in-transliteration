# Lost in Transliteration

**Do prompt-injection defences work in Urdu and Roman Urdu, and can a small model fix them?**

This is a small, reproducible study of prompt-injection defences in English, Urdu script, Roman Urdu and code-switched Urdu-English. It covers open-source injection classifiers (RQ1–2), five language models in an indirect-injection setup (RQ3), a small multilingual detector fine-tuned with a little Urdu-variant data (RQ4–5), and MCP security scanners facing tool poisoning (RQ6). [PLAN.md](PLAN.md) has the full design. The write-up draft is [writeup/post.md](writeup/post.md).

> **Status (2026-11-09):** every planned stage has run except the optional linear probe (RQ7). The write-up is a draft awaiting the owner's review. Findings about named vendors' tools will be disclosed to those vendors before any public post.

## Headline findings

Numbers are from `results/`, with n. `results/headline_numbers.json` carries every quoted table row by row.

- **Detection drops in Urdu variants, most in Roman Urdu.** Prompt Guard 2 (86M) catches 48.5% of English attacks and 11.5% in Roman Urdu (paired −37 points, 95% CI [−42, −32], n = 399). wolf-defender drops 20 points [−24, −16]. On 57 natively written attacks, wolf-defender catches 27 and Prompt Guard 2 catches 1. (`results/table_detection_drop.csv`, `results/classifiers_summary.csv`)
- **Over-blocking.** ProtectAI v2 flags 84.7% of benign Urdu-script text, against 2.0% in English (n = 300 each). (`results/table_detection_drop.csv`)
- **Models.** Of Claude Haiku 5.5, Sonnet 5.5, Opus 5.5, GPT-5.4 nano and Gemma 4 31B, only Haiku follows injections significantly more often in a variant (Urdu script +10.7 points [+3.3, +18.0], n = 150 pairs). Sonnet and Opus follow Urdu-script injections about 11 points less often than English. (`results/table_asr_gap.csv`)
- **Cheap fix.** XLM-R base fine-tuned on 1,628 rows (407 English, 1,221 machine-translated) catches 52 of 57 natively written attacks with 1 false positive in 57. Trained on English alone, it catches 41 of 57. One seed. (`results/detector_summary.csv`)
- **MCP scanners.** Ramparts, mcp-guardian and Cisco mcp-scanner flag none of our subtle poisoned descriptions in English (0/10 each); there is no language gap because there is no detection. (`results/mcp_summary.csv`)

## Reproduce

```bash
make setup && make test        # 113 tests on fakes, no network or GPU
make connectivity && make todos
make smoke                      # ~10 items through every stage
make data                       # §3: generate, expand, translate, review sheet, native set
make phase1                     # §4: classifiers, Valar (optional), models (USD 40 cap)
make train && make eval-detector   # §5 (device auto: CUDA, then Apple MPS, then CPU)
make phase3                     # §6
make analyze                    # §7: figures/ and results/ tables, headline_numbers.json
```

`make help` lists every stage. Each stage can be resumed and caches model responses on disk, so reruns cost nothing. The Claude Sonnet/Opus RQ3 runs use `make rq3-claude-estimate`, `rq3-claude-sanity` and `rq3-claude` (Message Batches through the owner's jzapi gateway).

What actually ran where: data generation and translation used open-weight models on OpenRouter (owner-approved profile `cloud_openrouter`, $0.18); the classifier and MCP stages ran on CPU; detector training ran on a MacBook GPU (MPS, 8 minutes for 6 runs). Total API spend was $5.71 (`results/spend.json`).

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

_The owner chooses a licence before publishing._ Third-party models keep their own licences. The detector bases are `FacebookAI/xlm-roberta-base` (MIT) and `distilbert/distilbert-base-multilingual-cased` (Apache-2.0), both checked 2026-11-09. Classifier licences are listed in `research/classifiers.md`.
