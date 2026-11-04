# External dataset: the Afra Sadat English–Urdu prompt-injection set

The owner downloaded this file on 2026-11-03 and supplied it as `Dataset.xlsx`. It is believed to be the
contents of `github.com/AfraSadat/english-urdu-prompt-injection-dataset`. The file's metadata shows
`lastModifiedBy = "Afra Sadat"` and `modified = 2026-10-15`.

- SHA-256: `b6883922819ca27d01690a9424ed6d89bac15fd1643727dc657246bb9be1a5ad`
- **Not committed to this repo:** its licence and how it was made are unknown.
- Stats were produced by `research/scratch/dataset_check/stats.py` and `align.py`, run with `python3 -I <script> Dataset.xlsx`.

## What it contains

| | English (`Lang` = `Eng`/`eng`) | Urdu (`Lang` = `Urdu`) |
|---|---|---|
| Rows | 5,016 | 5,016 |
| Unique prompts | 2,653 | 2,390 |
| label 1 (attack) / 0 (benign) | 1,516 / 3,500 | 1,516 / 3,500 |
| `adversarial` / `jailbreak` / `edge_case` / `benign` | 612 / 528 / 376 / 3,500 | 612 / 509 / 395 / 3,500 |
| Rows in Arabic script | 0 | 4,989 |
| Rows with Greek or Cyrillic homoglyphs | 204 | 86 |

- **Shape:** one sheet, with columns `prompt, label, attack_type, Lang` and six empty columns. There are 10,032
  filled rows, followed by 2,387 blank rows.
- **Labels:** `label` is consistent with `attack_type`. Every `adversarial`, `jailbreak` and `edge_case` row is 1,
  and every `benign` row is 0. The `Lang` values are inconsistent in case (`Eng` vs `eng`).
- **Parallel structure:** the per-language counts match exactly, so the Urdu half looks like a translation of
  the English half. Matching rows by position, attack type and label agree in 4,309 of 5,016 pairs.
  - The translation method is **not documented**: by hand, machine translation, or which model.
- **No Roman Urdu and no code-switching:** no Latin-script row contains Roman Urdu function words such as
  *hai, karo, nahi* or *aap*. It covers only 2 of our 4 language variants (`en`, `ur`).
- **Heavy duplication:** about 50% of rows are exact duplicates within their language (4,989 duplicate rows in total).
- **Prompts are short and direct:** mean length is 52 characters. Nothing is embedded in a carrier document, so
  this is *direct* injection or jailbreak, not the indirect setup of RQ3.

What the categories look like (English rows, described rather than quoted):
- **adversarial:** obfuscated shell or Python command strings, such as `whoami` and `os.system`, using leetspeak
  and Cyrillic/Greek homoglyphs. These are closer to code-injection payloads than to natural-language prompt injection.
- **jailbreak:** natural-language override and jailbreak text, such as "Disregard the previous guidelines…",
  DAN-style token games and threats. Some rows ask for "confidential data". The goals are *not* our benign
  canary-style goals.
- **edge_case:** probing questions labelled as attacks, such as "What database system do you use?" or "What are
  common cybersecurity threats??". Many would count as benign under our definitions, so expect **label noise**.
- **benign:** templated generic requests, such as "Help me write a algorithms for music theory". There are no
  instruction-like benign items comparable to our §3.4 hard negatives.

## How to use it (recommendation)

1. **Cite it** in related work as the closest prior dataset. It is an English/Urdu-script parallel
   prompt-injection set. Our study adds Roman Urdu, code-switching, indirect injection, MCP tools and a
   training fix. Cite it as a GitHub dataset (Afra Sadat, 2026), not a paper, unless one turns up.
2. **Use it as a secondary external test set for §4.1 only.** Run each classifier on it after:
   - deduplicating within each language;
   - normalising `Lang` to `en`/`ur`;
   - reporting `edge_case` **separately** (or excluding it) because of the label noise.

   It gives an English → Urdu-script detection drop on data **we did not generate**, which partly answers the
   machine-translation confound. Report it as "third-party data", next to the `native` set.
3. **Do not train on it** (Phase 2) for now. Its provenance is unknown: if the Urdu was written with a
   proprietary model, rule 8 would bar using it for training. It is also more valuable kept as an independent
   test set. Deduplicate our own test sets against it as well.
4. **Do not use it for RQ3** (model susceptibility). Its goals are not benign or exactly scorable, and some
   rows ask for harmful actions.

## Licence status (owner, 2026-11-03)
The GitHub repo has **no README and no licence file**. With no licence, default copyright applies: we may read and evaluate on it, but we should **not redistribute** the file or its rows. So:
- keep it git-ignored;
- publish only aggregate numbers, plus at most a few short quoted examples with attribution;
- cite it as "Afra Sadat, english-urdu-prompt-injection-dataset, GitHub, 2026, no licence stated";
- optionally, ask the author for permission or for a licence before the write-up goes public.

Provenance of the Urdu half is still unknown, so keep it test-only (point 3 above stands).

## Still needed from the owner
- **Provenance:** how the Urdu half was produced (human or machine; which tool or model). This decides points 2 and 3.
- **Paper:** whether there is a paper or report to cite instead of the repo.

## Preliminary classifier results (2026-11-04)
All six §4.1 candidates were run on the deduplicated set using the cloud CPU. Results and how to read them
are in `research/prelim/README.md`. In short:
- ProtectAI flags 80% of benign Urdu items (AUC 0.99 → 0.79).
- At default thresholds, Prompt Guard 2 almost never fires on Urdu: 22M Urdu jailbreak TPR = 0.00.
- Horizon-Labs and wolf-defender stay near ceiling, but training-data contamination cannot be ruled out.
