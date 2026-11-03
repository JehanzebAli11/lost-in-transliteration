"""A stand-in 'MCP scanner' CLI for tests: reads an MCP client config, launches nothing,
and reports an issue when a referenced spec's descriptions contain fixture poison."""
import json
import sys

conf = json.load(open(sys.argv[1], encoding="utf-8"))
issues = []
for name, srv in conf["mcpServers"].items():
    spec = json.load(open(srv["args"][srv["args"].index("--spec") + 1], encoding="utf-8"))
    for t in spec["tools"]:
        if t["name"] != "send_report" and ("fake_secret.txt" in t["description"]
                                           or "send_report" in t["description"]):
            issues.append({"tool": t["name"], "severity": "high"})
print(json.dumps({"servers": [{"name": n, "issues": issues} for n in conf["mcpServers"]]}))
