"""Run one third-party MCP scanner on one generated client config and print a uniform verdict.

    python scripts/mcp_scan_wrap.py {cisco|ramparts|mcp-guardian} <mcp_config.json>

Prints one JSON line: {"tool": ..., "flagged": true|false, "detail": ...}. The harness entry uses
parser `regex:"flagged": true`. Verdict rules come from research/mcp_scanners.md (tested
2026-11-03 on an English fixture). Binary locations come from environment variables so the
script works on any machine:

    LIT_CISCO_BIN        mcp-scanner          (pip install cisco-ai-mcp-scanner==4.8.7)
    LIT_RAMPARTS_BIN     ramparts             (cargo install ramparts --version 0.8.10)
    LIT_RAMPARTS_RULES   <ramparts v0.8.10 clone>/rules
    LIT_MCP_GUARDIAN_BIN mcp-guardian         (npm install mcp-guardian@2.4.0)

All three run locally: no LLM or API key is set, so nothing leaves the machine.
"""
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

REMOTE_KEYS = ("MCP_SCANNER_API_KEY", "MCP_SCANNER_LLM_API_KEY", "LLM_API_KEY", "OPENAI_API_KEY",
               "ANTHROPIC_API_KEY", "CLAUDE_API_KEY", "OPENROUTER_API_KEY")


def env_local() -> dict:
    env = {k: v for k, v in os.environ.items() if k not in REMOTE_KEYS}
    return env


def server_of(config_path: str) -> tuple[str, list[str]]:
    servers = json.load(open(config_path))["mcpServers"]
    (s,) = servers.values()
    return s["command"], list(s.get("args", []))


def cisco(config_path: str) -> tuple[bool, str]:
    cmd, args = server_of(config_path)
    argv = [os.environ["LIT_CISCO_BIN"], "--analyzers", "yara", "--raw", "stdio", "--stdio-command", cmd]
    for a in args:
        argv.append(f"--stdio-arg={a}")      # "=" form: values like --spec would confuse argparse
    p = subprocess.run(argv, capture_output=True, text=True, timeout=150, env=env_local(),
                       stdin=subprocess.DEVNULL)
    out = p.stdout
    start = min([i for i in (out.find("["), out.find("{")) if i != -1], default=-1)
    if start == -1:
        raise RuntimeError(f"cisco: no JSON (exit {p.returncode}): {(p.stderr or out)[-300:]}")
    data = json.loads(out[start:])
    items = data if isinstance(data, list) else data.get("results", [data])
    unsafe = [str(i.get("tool_name", "?")) for i in items if isinstance(i, dict) and i.get("is_safe") is False]
    return bool(unsafe), "unsafe tools: " + ",".join(unsafe)


def ramparts(config_path: str) -> tuple[bool, str]:
    cmd, args = server_of(config_path)
    with tempfile.TemporaryDirectory() as home:          # fresh HOME = no rug-pull baseline
        launcher = Path(home) / "launch.sh"
        launcher.write_text("#!/bin/sh\nexec " + " ".join(shlex.quote(x) for x in [cmd, *args]) + "\n")
        launcher.chmod(0o755)
        env = env_local() | {"HOME": home, "RAMPARTS_RULES_DIR": os.environ["LIT_RAMPARTS_RULES"]}
        p = subprocess.run([os.environ["LIT_RAMPARTS_BIN"], "scan", f"stdio:/bin/sh:{launcher}",
                            "--format", "table"], capture_output=True, text=True, timeout=180,
                           env=env, stdin=subprocess.DEVNULL)
    text = p.stdout + p.stderr
    if "rules executed" not in text:
        raise RuntimeError(f"ramparts did not scan (exit {p.returncode}): {text[-300:]}")
    rules = set(re.findall(r"matched by security rule (\w+)", text)) - {"MCPToolChanged"}
    return bool(rules), "rules: " + ",".join(sorted(rules))


def guardian(config_path: str) -> tuple[bool, str]:
    p = subprocess.run([os.environ["LIT_MCP_GUARDIAN_BIN"], "--no-config", "--json", config_path],
                       capture_output=True, text=True, timeout=150, env=env_local(),
                       stdin=subprocess.DEVNULL)
    return p.returncode >= 1, f"exit {p.returncode}"


TOOLS = {"cisco": cisco, "ramparts": ramparts, "mcp-guardian": guardian}

if __name__ == "__main__":
    tool, conf = sys.argv[1], sys.argv[2]
    flagged, detail = TOOLS[tool](conf)
    print(json.dumps({"tool": tool, "flagged": bool(flagged), "detail": detail}))
