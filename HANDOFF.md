# HANDOFF

## 2026-11-09 20:50 PKT · Phase 2 on the Mac (steps 1-6 of lit-native-mac)

**Done**
- Step 1: existing `.venv` (torch 2.15.0); `make test` passed; `torch.backends.mps.is_available()` True.
- Step 2: `pick_device("auto")` picks cuda -> mps -> cpu (one implementation in `src/eval_classifiers.py`, reused by `src/train_detector.py`; `src/eval_detector.py` gets it through `HFClassifier`). bf16 is CUDA-only (`use_bf16()`); MPS trains in fp32. Tests first: `tests/test_device.py`.
- Step 3: `detector.base_models` verified on the HF API on 2026-11-09: `FacebookAI/xlm-roberta-base` (MIT), `distilbert/distilbert-base-multilingual-cased` (Apache-2.0); both public and ungated; `TODO_VERIFY` removed.
- Step 4: `detector.conditions: [0, 25, 100]`, `seeds: [1]` (PLAN §11 fallback; Mac MPS and the deadline).
- Step 5: 6 runs trained on MPS in 8 min (20:40-20:48 PKT); `make eval-detector` scored each on 1,280 held-out items. Weights in `models/` (git-ignored, not uploaded).
- Also fixed an analyze bug that skipped fig2/fig8 on the real results (negative error-bar length at rates of 0 or 1); extended `results/headline_numbers.json` to carry every quoted table with its n.

**Verified**
- `make test`: 112 passed (82 at start; +20 device, +10 analyze edge cases).
- `models/*/pct*/seed1/meta.json`: status done, `device: mps`, `bf16: false` for all 6.
- `results/detector/ft_*_s1.jsonl` (6 files), `results/detector_summary.csv`, `results/detector_curve.csv`, `results/train_log.jsonl`.
- Smoke check before the real run: tiny classifier trained and evaluated on `auto` -> `mps` (scratch test, not committed).

**Next**
- Step 7 `make analyze` (figures, skipped list), step 8 write-up draft.
- Re-run `make analyze` once the RQ3 Sonnet/Opus full run (other worker, started 20:39 PKT) has landed in `results/models_*`.

**Open questions**
- Single seed and 3 data levels: no seed-variance bars on the RQ4 curve. If the owner wants 3 seeds × 5 levels, it is ~40 min more on this Mac (estimate from today's 8 min for 6 runs).
- The xlmr-base pct25 native result (37/57) is below pct0 (41/57); with one seed this cannot be told apart from noise.

## 2026-11-09 21:00 PKT · Analysis and write-up draft (steps 7-8 of lit-native-mac)

**Done**
- Step 7: `make analyze` produced every figure (fig1-5, fig7a-e, fig8) and table; `skipped` = [fig9] only, the RQ7 probe (optional stretch, never run). Fixed fig7c-e to compare the best condition per base model (same test-AUC rule) and made fig8 readable.
- Step 8: drafted `writeup/post.md` (2,801 words of prose), `writeup/application_blurb.md`, `README.md`, `SUMMARY.md`; `results/headline_numbers.json` carries every quoted table with n.
- RQ3 Sonnet 5.5 / Opus 5.5 full results are in the draft (no placeholder needed).

**Verified**
- `make test`: 113 passed. `results/figures_manifest.json`: skipped ["fig9"].
- Opened and checked by eye: fig2, fig4, fig7a, fig7d, fig7e, fig8 (not fig1, fig3, fig5, fig7b, fig7c).
- Every number in the post was checked against results/ (CSV rows quoted in comments); several draft errors were corrected (see LOG step 8).

**Next**
- Owner review of `writeup/` (JZ-NEED).
- The RQ3 worker still has to commit `results/models_*`, `usage.csv`, `spend.json` and its LOG lines. The analysis outputs committed here were built from those uncommitted rows. If that worker changes anything, rerun `make analyze` and update the RQ3 numbers in the post.
- Vendor disclosure before any public post (post section "Disclosure before publication").

**Open questions**
- The blurb's last personal sentence is an agent draft; the owner should write it.
- XLM-R +100% (52/57 native) is read off the curve; the pre-set selection rule picks XLM-R +0% (41/57). The post says so. The owner should decide whether that framing is right for the application.
- Optionally run the full detector design (5 levels × 3 seeds, about 40 min on this Mac) to get seed-variance bars.
