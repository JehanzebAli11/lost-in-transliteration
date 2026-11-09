# RUNBOOK — exact order of commands on the DGX Spark

All code was built and tested in a cloud session (no GPU, no Hugging Face, no access to
the Spark's local models) on 2026-11-03. Every stage passed its tests against fakes and a
fake HTTP server speaking the OpenAI and Anthropic wire formats (`make test`, 61 tests).
**Nothing has run against a real model yet.** This runbook is for the Spark agent.

Rules that still apply (PLAN.md §0): work without stopping unless truly blocked; log
everything in `LOG.md`; commit and push to `main` often; never fabricate results; only
open-weight models create or translate data (the code enforces this: `ProvenanceError`).

Every stage reads `config.yaml`. A value that starts with `TODO_VERIFY` (or a `null`
price) makes the stage that needs it stop with `ConfigTodoError` / `PriceMissing`
instead of guessing. `make todos` lists what is left.

---

## Step 0 — environment (once)

```bash
cd ~/lab/lost-in-transliteration
git pull
python3 -m venv .venv && source .venv/bin/activate
```

**torch first, for ARM64 + CUDA.** `requirements.txt` deliberately leaves torch out. Use a
CUDA build for aarch64 that supports the Spark's GB10 GPU (the NGC PyTorch container,
or the aarch64 CUDA wheels from the PyTorch index). Tests ran with torch 2.15.0 (x86_64,
CPU). Then:

```bash
make setup                                   # pip install -r requirements.txt (pinned)
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```
Expected: a version string and `True`. If `False`, fix torch before Phase 2.

```bash
make test
```
Expected: `61 passed` (≈30 s; no network, no GPU). If anything fails here, stop and fix
it first: the failure is environmental (package versions on ARM64), not data-related.

`.env` (git-ignored, `chmod 600`) holds `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`,
`HF_TOKEN`, optionally `VALAR_API_URL`/`VALAR_API_KEY`. See `.env.example`.

## Step 1 — verify every TODO in config.yaml (do this before any data work)

```bash
make connectivity      # appends a block to LOG.md
make todos             # lists what is still unverified
```
`make connectivity` expected output: `OK local (http://localhost:8000/v1) model ids: [...]`,
`SET env ...` lines, `OK anthropic models: [...]`, `OK openrouter model count: N`, and one
`TODO hf model id not verified` line per unverified Hugging Face id.

Fill in, and note the date and source of each in LOG.md:

| config key | how to verify |
|---|---|
| `models.generator/translator/native_labeler.model` | exact id from `GET localhost:8000/v1/models` (Gemma 4 31B IT) |
| `models.native_writer` + `endpoints.native_local` | a **different open-weight family** (e.g. a Qwen instruct model): check its licence; serve it with vLLM on port 8001 if memory allows next to Gemma, otherwise point `native_local` at OpenRouter with an open-weight model id and add a price. The code refuses a writer from the translator's family. For Qwen3-style thinking models set `extra_body: {chat_template_kwargs: {enable_thinking: false}}`. |
| `eval_models.models[*].model` | current cheapest small Anthropic model id; another provider's small model on OpenRouter; Gemma id on localhost |
| `eval_models.models[*].*_usd_per_mtok` | current prices from the provider pages (USD per million tokens). Write the date checked in a comment. Paid calls are refused until set. |
| `classifiers.models[*].hf_id` and `positive_labels` | open each model card; load it once and print `model.config.id2label`; `positive_labels` must match the injection label exactly (the loader errors if none match). Prompt Guard is gated: the owner accepts the licence on huggingface.co with their own account first; otherwise it is skipped and logged. Optionally enable a third verified classifier. |
| `detector.base_models[*].hf_id` | `FacebookAI/xlm-roberta-base`, `distilbert/distilbert-base-multilingual-cased` (or `mdeberta-v3-base`): confirm id and licence |
| `probe.model` | only for the stretch goal |
| `translation.similarity_model` | optional; leave TODO to report chrF only |
| `mcp.scanners` | see Phase 3 below |

Also check, and log:
- **Gemma output shape**: send one chat request by hand; if replies start with a reasoning
  block, `strip_reasoning()` removes `<think>…</think>`; if Gemma 4 uses another format,
  extend it in `src/llm.py`.
- **vLLM honours per-request `seed`** (it does in current releases): two identical
  requests with the same seed at temperature 0.3 should match.
- **Anthropic temperature**: API version 2026-10-21 (anthropic SDK ≥ 2.0, pinned 2.4.0)
  rejects `temperature`, so Claude is sampled at the API default and every row records
  `temperature: "api_default"`. This deviates from PLAN §4.3 ("temperature 0") and must
  be stated in the write-up's limitations. (Source:
  https://platform.claude.com/docs/en/api/migration-2026-10-21)
- **Owner skim of `prompts/native/*.txt` and `prompts/native/options.yaml`**: written in
  Roman Urdu by a non-native writer. Don't block on it; log whether it happened.

Commit `config.yaml` once verified: `git commit -am "Verify model ids and prices"`.

## Step 2 — smoke run (PLAN §11: tiny end-to-end before scaling)

```bash
make smoke             # LIT_PROFILE=smoke: ~10 items through every stage into runs/smoke/
```
Writes to `runs/smoke/` (git-ignored); LOG lines go to `runs/smoke/LOG.md`. Responses
are cached in `cache/`, so repeating it is nearly free.

Expected, in order (each line also appears in `runs/smoke/LOG.md`):
1. `[build] goal payloads: {...: 2}`, `templates: 10`, `carriers: 4`, `benign: 10`,
   `expanded: 10 English attacks {...}, 10 benign, 4 carriers`
2. `[translate] phrase translations: {...}` then `attacks/ur: +N translated, M dropped after retries, N/10 total` for each kind × variant
3. `runs/smoke/data/human_review_sample.csv` written
4. `[native] native set: K items from G generated; dropped {...}; shortfall ...`
5. `[classifiers] loaded classifier ...` then a `TPR … FPR … AUC …` line per classifier × language
6. `[valar] Valar skipped: ...` (unless configured)
7. `[eval_models] N tasks per model; budget spent so far $…` then one stats line per model with `status: ok`
8. `[train] ... epoch 1/1 loss …` for pct000 and pct100, then `[eval_detector] scored ft:…`
9. `[mcp] MCP variants: …`, `MCP server check: X/X servers advertise their spec correctly`, a line per scanner
10. `[analyze] analyze: produced [...]; skipped [...]`

**Then inspect, and paste into LOG.md (KICKOFF asks for this):**
- The Roman Urdu translations of the smoke items (`runs/smoke/data/roman_ur.jsonl`, `benign_roman_ur.jsonl`) — the 20-item quality check of Gemma's Roman Urdu. If clearly poor, switch the translator to another open-weight model (never Claude/OpenAI/Gemini), log why, and rerun.
- `runs/smoke/results/translation_failures.jsonl`: the `reasons`. The script/variant
  heuristics in `src/textcheck.py` were tuned on synthetic text. If real Roman Urdu or
  code-switched translations are rejected for `reads as English` / `no Roman Urdu markers`
  more than ~10% of the time, adjust `variant_check` thresholds or `RU_MARKERS`, log
  the change, add a test with the real example, and rerun.
- `runs/smoke/results/native_gate_report.json`: drop reasons should be plausible.
- `runs/smoke/results/models_raw.jsonl`: read 5 full prompts/responses.
- `runs/smoke/results/spend.json`: cost of the smoke run; extrapolate to the full run
  (≈ 150 attacks × 4 langs + native + cross + clean ≈ 900 calls per paid model).
- `runs/smoke/figures/`: open each PNG.

Commit fixes, then `make smoke-clean` (keeps the cache).

## Step 3 — Phase 1 data (full)

```bash
make data-generate     # ~10 min. data/templates/*.{json,jsonl}
make data-expand       # instant. data/en.jsonl (400), benign_en.jsonl (300), carriers/en.jsonl (30), splits.json
```
Check `data/splits.json` counts: polite_indirect templates all `heldout_style`; each other
family ~65/35 train/test.

```bash
make phrases
nohup make translate > logs/translate.log 2>&1 &      # resumable; rerun after any crash
```
Expected: `data/{ur,roman_ur,mixed}.jsonl` (~400 each), `benign_*.jsonl` (~300 each),
`data/carriers/*.jsonl` (30 each), `results/translation_report.json`. Items that never
pass the checks are dropped and listed in `results/translation_failures.jsonl`; report the
drop count. Commit `data/` when done.

```bash
make review-sample     # data/human_review_sample.csv (60 rows) for the owner; never overwrites ratings
make native            # data/native.jsonl (~120), results/native_gate_report.json
make backtranslate     # optional (§3.5): results/backtranslation_summary.json
```

## Step 4 — Phase 1 evaluations

```bash
make eval-classifiers  # results/classifiers/*.jsonl, classifiers_summary.csv, _by_style, _by_goal
make eval-valar        # results/valar_status.json (skipped unless configured)
make eval-models-sanity   # first 20 tasks per model; read the responses and spend.json
make eval-models       # full run; resumable; stops cleanly at 95% of the $40 cap
```
`eval-models` expected: `results/models_raw.jsonl` (every prompt + response),
`models_summary.csv`, `models_by_goal.csv`, `models_by_style.csv`,
`models_clean_controls.csv`. If LOG shows `BUDGET STOP`, the stage kept what it had:
lower `eval_models.max_attacks_per_lang` only if the remaining models still need room.
Check `models_clean_controls.csv`: `summary_ok` should be near 1 for every model and language.

RQ3 extension, Claude Sonnet 5.5 + Opus 5.5 as Message Batches (jzapi job `lit_rq3`, $20 cap; needs
`pip install -e ~/lab/JzLab` in this venv and the key in `~/lab/JzLab/.env`):
```bash
make rq3-claude-estimate  # offline worst case per model, sanity and full; no network
make rq3-claude-sanity    # 20 tasks per model; waits for the batches; read the rows and LOG line
make rq3-claude           # the remaining tasks; refuses without a clean sanity run; resumable
```
Ctrl-C while waiting is safe: rerunning the same target polls the submitted batch, never resubmits.

## Step 5 — Phase 2 (start as early as the data exists; runs in the background)

```bash
make train             # nohup; logs/train.log; checkpoints per epoch; rerun resumes
make train-status
```
Default: 2 base models × 5 conditions × 3 seeds = 30 runs. On the GPU each should take
minutes. If behind schedule (PLAN §11): set `detector.conditions: [0, 25, 100]` and
`seeds: [1]`. Weights go to `models/` (git-ignored); never upload them.

```bash
make eval-detector     # results/detector/*.jsonl, detector_summary.csv, detector_curve.csv
```

## Step 6 — Phase 3 (keep it small)

```bash
make mcp-build         # data/mcp/{english,translations}.json, servers/*.json, manifest.jsonl; sandbox/fake_secret.txt
make mcp-check         # expected: "MCP server check: N/N servers advertise their spec correctly"
```
Scanners: pick 2–4 open-source MCP scanners/gateways that run locally (candidates in
`config.yaml`: `mcp-scan`, Cisco's `mcp-scanner`; both **unverified**). For each:
install it, check its licence, check whether it sends descriptions to a remote API (set
`sends_remote` to `false` or a short description; never create an account), and find the
command that scans an MCP client config file. Put that command in `command` with
`{config_path}`, set `enabled: true`. Run it once by hand on
`mcp_configs/en-poisoned_description-01.json` and on a clean one, and check that the
`json_any_issue` parser flags the first and not the second; if its JSON differs, use
`regex:<pattern>` or extend `parse_json_any_issue`. Set `scans_outputs: true` only for a
gateway that really inspects tool results.

```bash
make mcp-scan          # results/mcp_raw.jsonl, mcp_summary.csv
```
Large reproducible bypass by a vendor's tool → note in the write-up that findings will be
disclosed to the vendor before wide publication (PLAN §0.5).

## Step 7 — analysis

```bash
make analyze
```
Expected: `results/figures_manifest.json` with every figure under `produced` and an empty
`skipped` (anything skipped had missing inputs; the reason is in LOG.md). Figures in
`figures/`: fig1–fig5, fig7a–e, fig8, fig9 (probe only if run). Tables:
`results/table_detection_drop.{csv,md}`, `table_asr_gap.{csv,md}`. Read
`results/qualitative_failures.md` and write the "why" patterns (§7) into LOG.md and the post.

```bash
make ignore-large      # git-ignore any results/*.jsonl over 50 MB
git add -A && git commit -m "Results" && git push
```

## Step 8 — write-up

Fill `writeup/post.md`, `writeup/application_blurb.md`, `README.md` and `SUMMARY.md`.
Every number must come from a file in `results/` (cite the file next to each number while
drafting). `results/headline_numbers.json` collects the main ones.

## Stretch (only if everything else is done)

```bash
make probe             # results/probe_results.csv; analyze adds fig9
```

---

## What could go wrong first (cloud-session view)

1. torch on ARM64/CUDA (Step 0) — environmental, not code.
2. Real Roman Urdu tripping the variant heuristics (Step 2) — tune `src/textcheck.py`.
3. Gemma replies not being a bare JSON array — `extract_json` tolerates fences, chatter
   and `{"items": [...]}`; if Gemma wraps differently, generation shows `only k/n valid`.
4. Classifier label names — `positive_labels` must match `id2label`.
5. Scanner CLIs and output formats — completely unverified.
6. Running two vLLM models at once on 128 GB unified memory — use OpenRouter for the
   native writer (open-weight model, with a price) if needed.
