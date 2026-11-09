# One command per stage. RUNBOOK.md gives the order and the expected output of each.
# Every target honours LIT_PROFILE (e.g. `LIT_PROFILE=smoke make translate`) and
# writes to the paths of that profile; `make smoke` runs the whole chain on ~10 items.

PY      ?= python
RUN      = $(PY) -m src
LOGDIR   = logs

.PHONY: help setup test todos connectivity \
        data-generate data-expand phrases translate review-sample backtranslate native data \
        eval-classifiers eval-valar eval-models-sanity eval-models phase1 \
        rq3-claude-estimate rq3-claude-sanity rq3-claude \
        rq3-uniform-estimate rq3-uniform-sanity rq3-uniform \
        train train-fg train-status eval-detector phase2 \
        mcp-build mcp-check mcp-scan phase3 probe analyze smoke smoke-clean ignore-large all

help:
	@grep -E '^[a-z0-9-]+:.*## ' Makefile | awk -F':.*## ' '{printf "  %-20s %s\n", $$1, $$2}'

setup: ## install pinned Python deps (install torch for this machine first; see RUNBOOK step 0)
	$(PY) -m pip install -r requirements.txt

test: ## full test suite on fakes (no network, no GPU)
	$(PY) -m pytest -q tests

todos: ## list config values still marked TODO_VERIFY
	$(RUN).check_connectivity --todos

connectivity: ## reachability of local vLLM, Anthropic, OpenRouter, HF; appended to LOG.md
	$(RUN).check_connectivity

# ---------------------------------------------------------------- §3 data
data-generate: ## English payloads, wrappers, carriers, benign (local Gemma)
	$(RUN).build_dataset generate

data-expand: ## splits.json + en.jsonl + benign_en.jsonl + carriers/en.jsonl (deterministic)
	$(RUN).build_dataset expand

phrases: ## translate the OVERRIDE phrase once per variant
	$(RUN).translate phrases

translate: ## attacks, benign, carriers -> ur / roman_ur / mixed (resumable)
	$(RUN).translate items

review-sample: ## data/human_review_sample.csv for the owner (never overwrites ratings)
	$(RUN).translate review-sample

backtranslate: ## 10% back-translation sample, chrF (+ embedding similarity if configured)
	$(RUN).translate backtranslate

native: ## §3.6 natively written test set (second open-weight family)
	$(RUN).build_native

data: data-generate data-expand phrases translate review-sample native ## all §3 data stages

# ---------------------------------------------------------------- §4 Phase 1
eval-classifiers: ## RQ1/RQ2 open-source classifiers
	$(RUN).eval_classifiers

eval-valar: ## RQ1/RQ2 Valar (skips cleanly if not configured)
	$(RUN).eval_valar

eval-models-sanity: ## RQ3 on the first 20 tasks per model (check prompts and spend first)
	$(RUN).eval_models --limit 20

eval-models: ## RQ3 full run (budget-guarded, resumable)
	$(RUN).eval_models

phase1: eval-classifiers eval-valar eval-models ## all Phase 1 evaluations

# RQ3 extension: Claude Sonnet 5.5 + Opus 5.5 as Message Batches through JzLab's jzapi (job lit_rq3).
# Needs `pip install -e ~/lab/JzLab` in this venv and the key in ~/lab/JzLab/.env. Runs on the base
# config (LIT_PROFILE cleared): same tasks as the OpenRouter run, LIT cap $40 x 0.95.
CLAUDE_BATCH = LIT_PROFILE= $(RUN).eval_models_claude_batch

rq3-claude-estimate: ## RQ3 Claude batch: offline worst-case cost of the sanity and full runs (no network)
	$(CLAUDE_BATCH) --mode estimate

rq3-claude-sanity: ## RQ3 Claude batch: first 20 tasks per model; waits, then writes rows (read them!)
	$(CLAUDE_BATCH) --mode sanity

rq3-claude: ## RQ3 Claude batch: full run (resumable; refuses without a clean sanity run)
	$(CLAUDE_BATCH) --mode full

# RQ3 at one shared max_tokens = 1024 (owner decision 2026-11-09, option A; REVIEW.md issue 1).
# Claude Haiku + Sonnet rerun through the Batch API (Opus is already at 1024: nothing to do);
# GPT-5.4 nano + Gemma rerun through OpenRouter (profile rq3_uniform; needs OPENROUTER_API_KEY in
# the environment or .env). Old rows move to results/rq3_claude_superseded.jsonl (Claude) or
# results/rq3_superseded.jsonl (OpenRouter) with a reason as each new row arrives. Rounds: Sonnet
# already has sanity r1/r2 and full r1, so the uniform batches are sanity round 3 and full round 2.
RQ3_MT        = 1024
UNIFORM_BATCH = $(CLAUDE_BATCH) --only claude-haiku claude-sonnet claude-opus \
                --haiku-max-tokens $(RQ3_MT) --sonnet-max-tokens $(RQ3_MT) --opus-max-tokens $(RQ3_MT)
UNIFORM_OR    = LIT_PROFILE= $(RUN).eval_models --profile rq3_uniform --only openrouter-small gemma \
                --max-tokens $(RQ3_MT)

rq3-uniform-estimate: ## RQ3 at max_tokens 1024: offline worst case of the remaining work + both cap headrooms
	LIT_PROFILE= $(RUN).rq3_uniform estimate --max-tokens $(RQ3_MT)

rq3-uniform-sanity: ## RQ3 at 1024: first 20 tasks per model, all five (Claude batch, then OpenRouter)
	$(UNIFORM_BATCH) --mode sanity --round 3
	$(UNIFORM_OR) --limit 20

rq3-uniform: ## RQ3 at 1024: full rerun (resumable; Claude refuses without the sanity round)
	$(UNIFORM_BATCH) --mode full --round 2
	$(UNIFORM_OR)

# ---------------------------------------------------------------- §5 Phase 2
train: ## Phase 2 training in the background (nohup; checkpoints; resumes on rerun)
	@mkdir -p $(LOGDIR)
	nohup $(RUN).train_detector > $(LOGDIR)/train.log 2>&1 & echo "training PID $$! -> $(LOGDIR)/train.log"

train-fg: ## Phase 2 training in the foreground
	$(RUN).train_detector

train-status: ## tail of the training log and finished runs
	@tail -n 5 $(LOGDIR)/train.log 2>/dev/null || true
	@find models -name meta.json 2>/dev/null | wc -l | xargs echo "finished runs:"

eval-detector: ## Phase 2 evaluation on test / held-out style / native
	$(RUN).eval_detector

phase2: train-fg eval-detector

# ---------------------------------------------------------------- §6 Phase 3
mcp-build: ## generate + translate poisoning patterns, write server specs, sandbox secret
	$(RUN).mcp_fakes.build_variants

mcp-check: ## launch every fake server over stdio and verify its tools
	$(RUN).eval_mcp_scanners check

mcp-scan: ## run enabled scanners on every variant
	$(RUN).eval_mcp_scanners scan

phase3: mcp-build mcp-check mcp-scan

# ---------------------------------------------------------------- §7, §10
probe: ## stretch: linear probes on hidden states
	$(RUN).probe

analyze: ## every figure and table in §7 from results/
	$(RUN).analyze

ignore-large: ## git-ignore any results/*.jsonl over 50 MB
	$(PY) scripts/ignore_large_results.py

# ---------------------------------------------------------------- smoke
smoke: ## tiny end-to-end run of every stage into runs/smoke/ (needs verified config + local vLLM)
	LIT_PROFILE=smoke $(MAKE) data-generate data-expand phrases translate review-sample native \
	  eval-classifiers eval-valar eval-models train-fg eval-detector mcp-build mcp-check mcp-scan analyze

smoke-clean: ## delete runs/smoke (does not touch the response cache)
	rm -rf runs/smoke

all: data phase1 phase2 phase3 analyze
