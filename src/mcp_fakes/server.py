"""A harmless fake MCP server built from a JSON spec (PLAN §6.2).

    python src/mcp_fakes/server.py --spec data/mcp/servers/<variant_id>.json

Self-contained on purpose (imports only the official MCP Python SDK), so scanners can
launch it from any working directory. Tools never touch the network or the file
system: ``send_report`` only returns "ok" and nothing is ever sent anywhere.

Spec keys: server_name, tools: [{name, description, output}]. A poisoned description
or poisoned output is simply text in the spec; this file contains no injection text.
"""
from __future__ import annotations

import argparse
import json
from typing import Any

try:  # official SDK v2
    from mcp.server.mcpserver import MCPServer as _Server
except ImportError:  # pragma: no cover - SDK v1
    from mcp.server.fastmcp import FastMCP as _Server


def _get_weather(output: str):
    def get_weather(city: str, notes: str = "") -> str:
        return output.replace("{city}", city)
    return get_weather


def _add_numbers(output: str):
    def add_numbers(a: float, b: float, notes: str = "") -> str:
        return output.replace("{result}", str(a + b))
    return add_numbers


def _search_notes(output: str):
    def search_notes(query: str, notes: str = "") -> str:
        return output.replace("{query}", query)
    return search_notes


def _send_report(output: str):
    def send_report(message: str) -> str:
        return output  # deliberately does nothing with the message
    return send_report


FACTORIES = {"get_weather": _get_weather, "add_numbers": _add_numbers,
             "search_notes": _search_notes, "send_report": _send_report}


def build_server(spec: dict[str, Any]):
    server = _Server(spec.get("server_name", "fake-mcp"))
    for tool in spec["tools"]:
        fn = FACTORIES[tool["name"]](tool.get("output", "ok"))
        server.add_tool(fn, name=tool["name"], description=tool["description"])
    return server


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spec", required=True)
    args = ap.parse_args(argv)
    with open(args.spec, encoding="utf-8") as f:
        spec = json.load(f)
    build_server(spec).run("stdio")


if __name__ == "__main__":
    main()
