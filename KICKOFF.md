# Kickoff: paste into Claude Code on the Spark

Run from `~/lab/lost-in-transliteration` with `PLAN.md` already copied into the repo.

```text
Read PLAN.md in this repo and execute it end to end, following §0 exactly.

Environment: you are on my NVIDIA DGX Spark (ARM64, CUDA GPU, 128 GB unified memory).
Secrets are in .env (never print or commit them). The repo's remote is
github.com/JehanzebAli11/lost-in-transliteration; commit and push to main as you go.

Before any data work:
1. Write .gitignore (.env, models/, caches, sandbox secrets), requirements.txt and
   config.yaml, then create LOG.md and log a connectivity check: Anthropic API,
   OpenRouter, Hugging Face, and the local teacher at http://localhost:8000/v1
   (list its model IDs). If the teacher isn't running, start Gemma 4 31B with vLLM in
   the background, log how, and continue with other setup while it loads.
2. Look up current model IDs and prices for the evaluated API models and put them in
   config.yaml with the date you checked.
3. Build budget.py with the USD 40 hard cap and the on-disk response cache before any
   paid call.
4. Do a 10-item end-to-end smoke run through every stage, including a 20-item
   translation quality check of Gemma's Roman Urdu. Show me the 20 translations in LOG.md.

Then work through the phases in priority order. Rule 8 (data provenance) is strict:
only open-weight models create or translate data; Claude and other API models are
only evaluated.

Build the natively written test set (§3.6) yourself with a second open-weight model.
Also create data/human_review_sample.csv; I may fill it in, but never block on it.

Run long jobs (translation of the full set, Phase 2 training) in the background inside
tmux or as nohup jobs that checkpoint, log their progress, and keep working on other
stages meanwhile.
```

## Owner checklist before pasting

- [ ] Create the empty repo `JehanzebAli11/lost-in-transliteration` on GitHub (no README)
- [ ] On the Spark: `cd ~/lab && git clone https://github.com/JehanzebAli11/lost-in-transliteration && cp PLAN.md lost-in-transliteration/`
- [ ] Create `.env` with `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`, `HF_TOKEN` (read-only), optionally Valar; `chmod 600 .env`
- [ ] Accept the Llama Prompt Guard licence on huggingface.co if you want it included
- [ ] Start Claude Code inside tmux (`tmux new -A -s lab`) so closing the laptop doesn't stop it
