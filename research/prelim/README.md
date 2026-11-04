# Preliminary §4.1 check on third-party data (Afra Sadat EN / Urdu-script set)

Run on 2026-11-04 on a 4-core cloud CPU (about 25 min in total). No DGX Spark was needed.
**This is preliminary.** It does not replace the planned run on our own four-variant test sets.

- Data: the owner-supplied `Dataset.xlsx`, SHA-256 `b6883922…a5ad`. See `../external_datasets.md`.
  The data has no licence, so **no prompt text is committed**. `out/scores.jsonl` holds only the row index,
  a SHA-1 prefix, the labels and the scores.
- Reproduce:
  ```bash
  HF_TOKEN=... python -I research/prelim/eval_afra.py Dataset.xlsx research/prelim/out
  python research/prelim/analyze_afra.py research/prelim/out
  ```
- Preprocessing:
  - Rows were deduplicated within each language, leaving 5,043 unique items with 0 label conflicts.
  - The headline attack set is `adversarial` + `jailbreak`. `edge_case` is reported separately because of label noise.
  - The resulting n: attacks en = 1,041, ur = 823; benign en = 1,492, ur = 1,478.
- Decision rules:
  - ProtectAI flags an item at score > 0.92.
  - The other classifiers flag at P(index 1) ≥ 0.5.
  - 95% CIs come from a percentile bootstrap (2,000 resamples, seed 0).
  - Revisions are pinned in `eval_afra.py`.
- No input was truncated at 512 tokens.

## Results

Full tables: `out/summary.md` and `out/summary.csv`.

| Classifier | TPR en → ur | FPR en → ur | AUC en → ur |
|---|---|---|---|
| ProtectAI deberta-v3 v2 | 0.743 → 0.931 | **0.001 → 0.798** | 0.994 → **0.786** |
| Prompt Guard 2 86M | 0.236 → 0.179 | 0.009 → 0.000 | 0.849 → 0.950 |
| Prompt Guard 2 22M | 0.107 → 0.015 | 0.001 → 0.000 | 0.703 → 0.863 |
| Horizon-Labs guard-base | 0.917 → 0.842 | 0.001 → 0.000 | 1.000 → 0.997 |
| proventra mdeberta | 0.738 → 0.899 | 0.303 → 0.282 | 0.833 → 0.911 |
| patronus wolf-defender | 0.971 → 0.966 | 0.000 → 0.041 | 1.000 → 0.995 |

## Reading

1. **ProtectAI (English-only) treats Urdu script itself as injection.**
   - It flags 80% of benign Urdu prompts.
   - Its higher Urdu TPR is therefore an artefact of that bias, not real detection.
   - AUC drops by 0.21 [CI −0.23, −0.19]. This is the clearest failure mode, and it matches the model card's
     statement that it does not handle non-English prompts.
2. **Prompt Guard 2 recalls little on this set in either language.** Its jailbreak TPR is 0.39 (en) for 86M.
   - Most attacks here are obfuscated shell or code strings, or threat-style jailbreaks. PG2 targets explicit
     instruction overrides, so this is partly a scope mismatch.
   - At its default threshold, Urdu recall still falls: jailbreak 0.39 → 0.27 for 86M and 0.20 → 0.00 for 22M.
   - Urdu AUC is *higher*, so the drop is a calibration and threshold problem: Urdu scores are pushed towards 0.
3. **The multilingual fine-tunes look strong, but contamination is plausible.**
   - Horizon-Labs and wolf-defender both reach AUC ≥ 0.995 in both languages.
   - The set is small, templated and public (GitHub, 2026). We do not know whether it, or close templates of
     it, was in their training data.
   - Even so, Horizon's TPR falls by 0.075 [−0.106, −0.045] and wolf-defender's Urdu FPR rises to 4%.
4. **proventra mdeberta has a 30% FPR on these benign items in both languages.**
   - That makes it unusable at its default threshold.
   - Its gain in Urdu TPR is not meaningful on its own.

## Caveats

- **Urdu script only.** The set has no Roman Urdu and no code-switching, so it does not test the project's main
  hypothesis about romanisation.
- **Unknown provenance.** It is unknown how the Urdu half was produced. The text is short, templated and
  heavily duplicated.
- **Not our attack style.** The "attacks" are mostly code-payload and jailbreak strings, not the §3 benign-goal
  injections.
- The `edge_case` label is noisy, so it is not used in the headline numbers.
- These numbers are a sanity check and a motivating example for the write-up, cited as third-party data.
  They are not the RQ1 result.
