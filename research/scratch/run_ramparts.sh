#!/bin/bash
# Ramparts 0.8.10 (cargo install). No LLM_API_KEY set -> YARA/static checks only.
# cargo install ships no YARA rules: point RAMPARTS_RULES_DIR at rules/ from the v0.8.10 git tag.
cd "$(dirname "$0")"
unset LLM_API_KEY OPENAI_API_KEY
export RAMPARTS_RULES_DIR=$PWD/ext/ramparts-src/rules
R=.cargo-ramparts/bin/ramparts
for clean in 0 1; do
  rm -f ~/.ramparts/content-baseline.json   # reset rug-pull baseline so each run is independent
  echo "### FAKE_MCP_CLEAN=$clean  ramparts scan stdio:<python>:fake_server.py --format table"
  FAKE_MCP_CLEAN=$clean timeout 180 $R scan "stdio:$PWD/.venv-mcp/bin/python:$PWD/fake_server.py" --format table </dev/null 2>&1
  echo "exit=$?"
done
