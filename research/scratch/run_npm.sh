#!/bin/bash
# npm-based scanners against the fake server. No API keys set -> local analysis only.
cd "$(dirname "$0")"
unset ANTHROPIC_API_KEY CLAUDE_API_KEY
B=npm/node_modules/.bin
for cfg in mcp_config.json mcp_config_clean.json; do
  echo "################ config=$cfg"
  echo "### mcp-shield --path $cfg"
  timeout 120 $B/mcp-shield --path "$cfg" </dev/null 2>&1; echo "exit=$?"
  echo "### mcp-guardian --no-config --json $cfg"
  timeout 120 $B/mcp-guardian --no-config --json "$cfg" </dev/null 2>&1; echo "exit=$?"
done
echo "### mcp-watch scan-local . (static source scan of this folder: fake_server.py only)"
mkdir -p /tmp/mcpwatch_target && cp fake_server.py /tmp/mcpwatch_target/
timeout 120 $B/mcp-watch scan-local /tmp/mcpwatch_target </dev/null 2>&1; echo "exit=$?"
