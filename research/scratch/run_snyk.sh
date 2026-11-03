#!/bin/bash
# snyk-agent-scan (ex mcp-scan): scan needs SNYK_TOKEN (Snyk account) -> excluded; inspect is local.
cd "$(dirname "$0")"
unset SNYK_TOKEN
echo "### snyk-agent-scan inspect"; timeout 60 .venv-snyk/bin/snyk-agent-scan inspect --json mcp_config.json </dev/null; echo "exit=$?"
echo "### snyk-agent-scan scan (no SNYK_TOKEN)"; timeout 60 .venv-snyk/bin/snyk-agent-scan scan --json mcp_config.json </dev/null; echo "exit=$?"
