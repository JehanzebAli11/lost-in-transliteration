# prompts/

Prompt templates for the **open-weight** models that create and translate the data
(PLAN rule 8). They are `string.Template` files: `$name` is filled by code, and literal
`{CANARY}`, `{SECRET}`, `{PHRASE}`, `{PAYLOAD}` placeholders are passed through for the
model to copy.

None of these files contains dataset items. They describe what to write; the models on
the Spark write it.

| folder | used by | model role |
|---|---|---|
| `generation/` | `src/build_dataset.py` | `generator` (Gemma 4 31B, local) |
| `translation/` | `src/translate.py` | `translator` (Gemma 4 31B, local) |
| `native/` | `src/build_native.py` | `native_writer` (a different open family) and `native_labeler` |
| `mcp/` | `src/mcp_fakes/build_variants.py` | `generator` |

`native/*.txt` are written in Roman Urdu as PLAN §3.6 asks ("prompted in Roman Urdu /
Urdu"). They were drafted in a cloud session by a non-native writer: **the owner should
skim them before the native run** (RUNBOOK step 1).

Two translation prompts exist on purpose (§3.5): `translate_train.txt` for `train`-split
items and `translate_test.txt` for `test`, `heldout_style` and carriers, so the training
data is not worded identically to the test data.
