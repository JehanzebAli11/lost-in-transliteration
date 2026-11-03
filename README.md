# Lost in Transliteration

**Do AI safety filters work in Urdu and Roman Urdu — and can a small model fix them?**

A small, reproducible study of prompt-injection defences in English, Urdu script, Roman
Urdu and code-switched Urdu-English: open-source injection classifiers (RQ1–2), frontier
and open models in an indirect-injection setup (RQ3), a small multilingual detector
fine-tuned with a little Urdu-variant data (RQ4–5), and MCP security scanners facing
tool poisoning (RQ6). See [PLAN.md](PLAN.md) for the full design.

> **Status (2026-11-03):** code, prompts and tests are built; no data or results exist
> yet. They are produced on the owner's DGX Spark by following [RUNBOOK.md](RUNBOOK.md).

## Headline findings
_To be written from `results/` after the run. No numbers here until then._

## Reproduce
```bash
make setup && make test        # 61 tests on fakes, no network or GPU
make connectivity && make todos
make smoke                      # ~10 items through every stage
make data                       # §3: generate, expand, translate, review sheet, native set
make phase1                     # §4: classifiers, Valar (optional), models (USD 40 cap)
make train && make eval-detector   # §5
make phase3                     # §6
make analyze                    # §7: figures/ and results/ tables
```
`make help` lists every stage. Each stage is resumable and caches model responses on
disk, so reruns cost nothing.

## Layout
```
config.yaml        models, prices, budget, seeds, sizes (TODO_VERIFY values are refused until checked)
prompts/           prompts for the open-weight models that create and translate all data
src/               one module per stage (see Makefile); src/mcp_fakes/ = Phase 3 servers
tests/             fakes + tests for every stage, incl. an end-to-end run producing all figures
data/  results/  figures/  writeup/   (filled by the run)
```

## Data provenance and safety
- Every data item is written or translated by an **open-weight** model run locally on the
  Spark (Gemma 4 31B; a second open family for the natively written set). Claude, OpenAI
  and Gemini models are only ever *evaluated*; the code refuses to use them for data.
- All attack goals are benign: canary tokens, fake secrets, a fixed silly phrase,
  answering in French. Phase 3 runs only against local fake MCP servers with a fake
  secret in `sandbox/`.
- The natively written set is "model-written natively in Urdu variants", not human-written.
- Valar, if included, is the author's own system.
- Trained detector weights are not published.

## Licence
_Owner to choose before publishing. Third-party models and datasets keep their own licences (recorded in LOG.md when verified)._
