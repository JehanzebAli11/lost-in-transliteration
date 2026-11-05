"""RQ6: do MCP security scanners catch tool poisoning written in Urdu variants? (PLAN §6.3)

Stages (``python -m src.eval_mcp_scanners <stage>``):
  check   launch every fake server over stdio with the official MCP client and verify
          the advertised tools/descriptions match the spec (sanity; no scanners)
  scan    run every enabled scanner against every server variant

Scanner adapters (config.yaml ``mcp.scanners``):
  kind: command     a CLI that reads an MCP client config file. ``{config_path}`` in the
                    command is replaced by a generated mcp_configs/<variant>.json that
                    launches the fake server. Parsers: json_any_issue | exit_code |
                    regex:<pattern>. ``scans_outputs: false`` (default) means the tool
                    only reads descriptions, so poisoned_output variants are marked n/a.
  kind: classifier  a Phase 1 classifier applied to each tool description / output text
                    (a reference point, not an MCP tool).
  kind: python      injected callable(spec) -> (flagged, score, detail) (tests).

Outputs: results/mcp_raw.jsonl and results/mcp_summary.csv (scanner x lang x kind:
detection rate on poisoned servers, false-positive rate on clean ones).
Everything runs locally against fake servers; record in LOG.md if a scanner sends
descriptions to a remote API (``sends_remote``).
"""
from __future__ import annotations

import asyncio
import json
import re
import shlex
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable

from .common import (REPO_ROOT, Config, is_todo, load_config, load_dotenv_if_present, log_event,
                     read_json, stage_parser, write_json, write_jsonl)
from .eval_classifiers import write_csv
from .mcp_fakes.build_variants import load_manifest, mcp_dir
from .metrics import wilson_ci

SERVER_SCRIPT = REPO_ROOT / "src" / "mcp_fakes" / "server.py"
ISSUE_KEYS = ("issues", "findings", "vulnerabilities", "threats", "detections", "warnings", "alerts")
BAD_SEVERITY = {"low", "medium", "high", "critical", "error"}


def server_command(spec_path: Path) -> tuple[str, list[str]]:
    return sys.executable, [str(SERVER_SCRIPT), "--spec", str(spec_path.resolve())]


def write_client_config(cfg: Config, spec: dict, spec_path: Path) -> Path:
    cmd, args = server_command(spec_path)
    conf = {"mcpServers": {spec["server_name"]: {"command": cmd, "args": args}}}
    p = cfg.root / "mcp_configs" / f"{spec['variant_id']}.json"
    write_json(p, conf)
    return p


# ----------------------------------------------------------------------------- parsers
def _walk(obj):
    yield obj
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _walk(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk(v)


def parse_json_any_issue(stdout: str) -> tuple[bool, str]:
    """Generic: flagged if any issue-like list is non-empty, a bad severity appears, or
    a verdict says unsafe. Verify against each tool's real output on the Spark."""
    start = min([i for i in (stdout.find("{"), stdout.find("[")) if i != -1], default=-1)
    if start == -1:
        raise ValueError("no JSON in scanner output")
    data = json.loads(stdout[start:])
    hits = []
    for node in _walk(data):
        if not isinstance(node, dict):
            continue
        for k, v in node.items():
            kl = str(k).lower()
            if kl in ISSUE_KEYS and isinstance(v, list) and v:
                hits.append(f"{k}:{len(v)}")
            if kl == "severity" and str(v).lower() in BAD_SEVERITY:
                hits.append(f"severity:{v}")
            if kl in ("is_safe", "safe") and v is False:
                hits.append("unsafe")
            if kl in ("verdict", "status") and str(v).lower() in ("malicious", "unsafe", "vulnerable", "fail"):
                hits.append(f"{k}:{v}")
    return bool(hits), ";".join(sorted(set(hits)))[:300]


def parse_output(parser: str, proc: subprocess.CompletedProcess) -> tuple[bool, str]:
    if parser == "json_any_issue":
        return parse_json_any_issue(proc.stdout)
    if parser == "exit_code":
        return proc.returncode != 0, f"exit {proc.returncode}"
    if parser.startswith("regex:"):
        m = re.search(parser[6:], proc.stdout + proc.stderr)
        return bool(m), m.group(0)[:200] if m else ""
    raise ValueError(f"unknown parser {parser!r}")


# ----------------------------------------------------------------------------- adapters
class CommandScanner:
    def __init__(self, cfg: Config, entry: dict, runner: Callable | None = None):
        self.cfg, self.entry = cfg, entry
        self.name = entry["name"]
        self.scans_outputs = bool(entry.get("scans_outputs", False))
        self.runner = runner or (lambda argv, timeout: subprocess.run(
            argv, capture_output=True, text=True, timeout=timeout, cwd=str(cfg.root)))

    def __call__(self, spec: dict, spec_path: Path):
        conf = write_client_config(self.cfg, spec, spec_path)
        argv = shlex.split(self.entry["command"].replace("{config_path}", str(conf)))
        proc = self.runner(argv, int(self.entry.get("timeout", 180)))
        flagged, detail = parse_output(self.entry.get("parser", "json_any_issue"), proc)
        return flagged, None, detail


class ClassifierScanner:
    def __init__(self, entry: dict, clf):
        self.name = entry["name"]
        self.clf = clf
        self.scans_outputs = True

    def __call__(self, spec: dict, spec_path: Path):
        # Score everything a client would see (descriptions and outputs) on every variant, clean
        # included, so the false-positive rate is measured under the same conditions as detection.
        # Scoring outputs only on poisoned_output variants (the old behaviour) made the comparison unfair.
        descs = [t["description"] for t in spec["tools"]]
        outs = [t["output"] for t in spec["tools"]]
        scores = self.clf.score(descs + outs)
        d_best = max(scores[:len(descs)], default=0.0)
        o_best = max(scores[len(descs):], default=0.0)
        best = max(d_best, o_best)
        return best >= self.clf.threshold, best, f"max score {best:.3f} (desc {d_best:.3f}, out {o_best:.3f})"


def build_scanners(cfg: Config, *, classifiers: dict | None = None, extra: dict | None = None,
                   runner: Callable | None = None) -> dict:
    scanners = {}
    for entry in cfg["mcp"]["scanners"]:
        if entry.get("enabled", True) is False:
            continue
        name = entry["name"]
        try:
            if entry["kind"] == "command":
                if is_todo(entry.get("command")):
                    raise RuntimeError("command not verified (TODO_VERIFY)")
                if is_todo(entry.get("sends_remote")):
                    raise RuntimeError("sends_remote not verified: check whether it calls a remote API")
                scanners[name] = CommandScanner(cfg, entry, runner)
                if entry.get("sends_remote"):
                    log_event(cfg, f"{name} sends tool descriptions to a remote API "
                                   f"({entry['sends_remote']}); acceptable only because all data is fake",
                              stage="mcp")
            elif entry["kind"] == "classifier":
                if classifiers is None:
                    from .eval_classifiers import build_classifiers
                    classifiers = build_classifiers(cfg)
                clf = classifiers.get(entry["classifier"])
                if clf is None:
                    raise RuntimeError(f"classifier {entry['classifier']} unavailable")
                scanners[name] = ClassifierScanner(entry, clf)
        except Exception as e:
            log_event(cfg, f"SKIP scanner {name}: {e}", stage="mcp")
    for name, fn in (extra or {}).items():
        scanners[name] = fn
    return scanners


# ----------------------------------------------------------------------------- stages
async def _list_tools(spec_path: Path) -> list[tuple[str, str]]:
    from mcp import Client
    from mcp.client.stdio import StdioServerParameters
    cmd, args = server_command(spec_path)
    async with Client(StdioServerParameters(command=cmd, args=args)) as c:
        tl = await c.list_tools()
        return [(t.name, t.description) for t in tl.tools]


def run_check(cfg: Config, limit: int | None = None) -> dict:
    manifest = load_manifest(cfg)[: limit or None]
    bad = []
    for m in manifest:
        spec_path = mcp_dir(cfg) / m["spec"]
        spec = read_json(spec_path)
        got = asyncio.run(_list_tools(spec_path))
        want = [(t["name"], t["description"]) for t in spec["tools"]]
        if sorted(got) != sorted(want):
            bad.append(m["variant_id"])
    log_event(cfg, f"MCP server check: {len(manifest) - len(bad)}/{len(manifest)} servers advertise "
                   f"their spec correctly" + (f"; mismatched: {bad[:5]}" if bad else ""), stage="mcp")
    return {"checked": len(manifest), "bad": bad}


def run_scan(cfg: Config, *, scanners: dict | None = None, limit: int | None = None) -> list[dict]:
    manifest = load_manifest(cfg)[: limit or None]
    if not manifest:
        raise RuntimeError("no MCP variants; run `make mcp-build` first")
    scanners = scanners if scanners is not None else build_scanners(cfg)
    rows = []
    for name, scan in scanners.items():
        for m in manifest:
            spec_path = mcp_dir(cfg) / m["spec"]
            spec = read_json(spec_path)
            row = {"scanner": name, "variant_id": m["variant_id"], "lang": m["lang"], "kind": m["kind"],
                   "pattern_id": m["pattern_id"], "label": int(m["kind"] != "clean")}
            if m["kind"] == "poisoned_output" and not getattr(scan, "scans_outputs", True):
                rows.append(row | {"applicable": False, "flagged": None, "score": None, "detail": "n/a"})
                continue
            try:
                flagged, score, detail = scan(spec, spec_path)
                rows.append(row | {"applicable": True, "flagged": bool(flagged), "score": score,
                                   "detail": detail, "error": None})
            except Exception as e:
                rows.append(row | {"applicable": True, "flagged": None, "score": None, "detail": "",
                                   "error": f"{type(e).__name__}: {e}"[:300]})
        log_event(cfg, f"scanner {name}: ran on {len(manifest)} variants", stage="mcp")
    write_jsonl(cfg.results_dir / "mcp_raw.jsonl", rows)
    write_summary(cfg, rows)
    return rows


def write_summary(cfg: Config, rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for r in rows:
        if r.get("applicable") and r.get("flagged") is not None:
            groups[(r["scanner"], r["lang"], r["kind"])].append(r)
    out = []
    for (s, lang, kind), rs in sorted(groups.items()):
        k = sum(int(r["flagged"]) for r in rs)
        lo, hi = wilson_ci(k, len(rs))
        out.append({"scanner": s, "lang": lang, "kind": kind, "n": len(rs), "flagged": k,
                    "rate": k / len(rs), "ci_lo": lo, "ci_hi": hi,
                    "metric": "false_positive_rate" if kind == "clean" else "detection_rate",
                    "errors": sum(1 for r in rows if r["scanner"] == s and r["lang"] == lang
                                  and r["kind"] == kind and r.get("error"))})
    write_csv(cfg.results_dir / "mcp_summary.csv", out)
    return out


def main(argv=None):
    ap = stage_parser(__doc__)
    ap.add_argument("stage", choices=["check", "scan"])
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.profile)
    load_dotenv_if_present(cfg)
    if args.stage == "check":
        run_check(cfg, limit=args.limit)
    else:
        run_scan(cfg, limit=args.limit)


if __name__ == "__main__":
    main()
