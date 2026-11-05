#!/bin/bash
# Cisco AI Defense mcp-scanner, YARA analyzer only (fully local; no API key, no LLM).
cd "$(dirname "$0")"
unset MCP_SCANNER_API_KEY MCP_SCANNER_LLM_API_KEY
for clean in 0 1; do
  echo "### FAKE_MCP_CLEAN=$clean  mcp-scanner --analyzers yara --format by_tool stdio ..."
  timeout 120 .venv-cisco/bin/mcp-scanner --analyzers yara --format by_tool \
    stdio --stdio-command .venv-mcp/bin/python --stdio-arg fake_server.py \
    --stdio-env FAKE_MCP_CLEAN=$clean </dev/null 2>&1
  echo "exit=$?"
done
echo "### config mode, raw JSON"
timeout 120 .venv-cisco/bin/mcp-scanner --analyzers yara --raw config --config-path mcp_config.json </dev/null 2>&1 | head -60
echo "exit=$?"
