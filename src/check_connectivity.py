"""Pre-flight checks for the Spark (PLAN §11 'first 2h'): what is reachable and what is
still unverified. Appends a block to LOG.md; never prints secret values.

  python -m src.check_connectivity           # connectivity + TODO report
  python -m src.check_connectivity --todos   # only list config values still marked TODO_VERIFY

Checks: the local teacher (GET /v1/models on localhost:8000, lists model ids), the
native-set endpoint, Anthropic and OpenRouter (model listing; free, no tokens spent),
Hugging Face (model_info for every verified model id: gated?, licence), and which
env vars are set.
"""
from __future__ import annotations

import os
from typing import Any, Callable

from .common import Config, is_todo, load_config, load_dotenv_if_present, log_event, now_iso, stage_parser
from .llm import AnthropicClient, OpenAICompatClient


def find_todos(obj: Any, path: str = "") -> list[str]:
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k).startswith("_") or k == "profiles":
                continue
            out += find_todos(v, f"{path}.{k}" if path else str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            label = v.get("name", i) if isinstance(v, dict) else i
            out += find_todos(v, f"{path}[{label}]")
    elif isinstance(obj, str) and obj.strip().startswith("TODO_VERIFY"):
        out.append(f"{path} = {obj}")
    elif obj is None and any(path.endswith(s) for s in ("usd_per_mtok",)):
        out.append(f"{path} = null (price needed before any paid call)")
    return out


def enabled_paths(cfg: Config) -> list[str]:
    """TODOs that matter: skip entries explicitly disabled."""
    todos = find_todos(dict(cfg))
    disabled = []
    for sect, key in (("classifiers", "models"), ("eval_models", "models"), ("mcp", "scanners")):
        for e in cfg[sect][key]:
            if e.get("enabled", True) is False:
                disabled.append(f"[{e['name']}]")
    return [t for t in todos if not any(d in t for d in disabled)]


def hf_ids(cfg: Config) -> list[str]:
    ids = [e["hf_id"] for e in cfg["classifiers"]["models"] if e.get("enabled", True)]
    ids += [b["hf_id"] for b in cfg["detector"]["base_models"]]
    ids.append(cfg["probe"]["model"])
    return ids


def check(cfg: Config, *, local_lister: Callable | None = None, anthropic_lister: Callable | None = None,
          openrouter_lister: Callable | None = None, hf_info: Callable | None = None) -> list[str]:
    lines = []

    def attempt(label, fn):
        try:
            res = fn()
            lines.append(f"OK   {label}: {res}")
        except Exception as e:
            lines.append(f"FAIL {label}: {type(e).__name__}: {str(e)[:200]}")

    for name in ("local", "native_local"):
        ep = cfg["endpoints"][name]
        lister = local_lister or (lambda ep=ep: OpenAICompatClient(ep["base_url"], timeout=10).list_models())
        attempt(f"{name} ({ep['base_url']}) model ids", lambda lister=lister: lister())
    for var in ("ANTHROPIC_API_KEY", "OPENROUTER_API_KEY", "HF_TOKEN", "VALAR_API_URL", "VALAR_API_KEY"):
        lines.append(f"{'SET ' if os.environ.get(var) else 'MISS'} env {var}")
    if os.environ.get("ANTHROPIC_API_KEY") or anthropic_lister:
        fn = anthropic_lister or (lambda: [m for m in AnthropicClient(os.environ["ANTHROPIC_API_KEY"],
                                                                       timeout=20).list_models()][:20])
        attempt("anthropic models", fn)
    if os.environ.get("OPENROUTER_API_KEY") or openrouter_lister:
        fn = openrouter_lister or (lambda: len(OpenAICompatClient(cfg["endpoints"]["openrouter"]["base_url"],
                                                                  os.environ["OPENROUTER_API_KEY"],
                                                                  timeout=20).list_models()))
        attempt("openrouter model count", fn)

    def default_hf(mid):
        from huggingface_hub import model_info
        info = model_info(mid, token=os.environ.get("HF_TOKEN"))
        lic = (getattr(info, "card_data", None) or {}).get("license") if getattr(info, "card_data", None) else None
        return f"gated={getattr(info, 'gated', None)} license={lic}"
    for mid in hf_ids(cfg):
        if is_todo(mid):
            lines.append(f"TODO hf model id not verified: {mid}")
            continue
        attempt(f"huggingface {mid}", lambda mid=mid: (hf_info or default_hf)(mid))
    todos = enabled_paths(cfg)
    lines.append(f"{len(todos)} config value(s) still TODO_VERIFY (run with --todos to list)")
    return lines


def main(argv=None):
    ap = stage_parser(__doc__)
    ap.add_argument("--todos", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.profile)
    load_dotenv_if_present(cfg)
    if args.todos:
        todos = enabled_paths(cfg)
        print("\n".join(todos) if todos else "No TODO_VERIFY values left in enabled entries.")
        return
    lines = check(cfg)
    print("\n".join(lines))
    with open(cfg.log_file, "a", encoding="utf-8") as f:
        f.write(f"\n## Connectivity check {now_iso()}\n\n```\n" + "\n".join(lines) + "\n```\n\n")
    log_event(cfg, "connectivity check written above", stage="connectivity", echo=False)


if __name__ == "__main__":
    main()
