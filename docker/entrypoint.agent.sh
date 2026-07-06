#!/bin/bash
set -e

# Git configuration
git config --global user.name "${GIT_AUTHOR_NAME}"
git config --global user.email "${GIT_AUTHOR_EMAIL}"
git config --global --add safe.directory "${GIT_SAFE_DIRECTORY}"

# GitHub CLI auth check (gh reads GH_TOKEN automatically)
set +e
if gh auth status > /dev/null 2>&1; then
    echo "[entrypoint] gh CLI authenticated (GH_TOKEN)."
else
    echo "[entrypoint] gh CLI authentication failed. Check GH_TOKEN / GITHUB_PERSONAL_ACCESS_TOKEN."
fi
set -e

# Update Claude Code (non-blocking on failure)
set +e
echo "[entrypoint] Updating Claude Code to the latest version..."
claude update 2>/dev/null \
    && echo "[entrypoint] Claude Code update complete." \
    || echo "[entrypoint] Skipping Claude Code update (already up to date or offline)."
set -e

# Codex auth setup. For ChatGPT Pro/Plus, mount a host ~/.codex directory that
# already contains auth.json. CODEX_ACCESS_TOKEN is for supported trusted
# automation tokens, not Platform API keys.
set +e
if command -v codex > /dev/null 2>&1; then
    export CODEX_HOME="${CODEX_HOME:-/home/devuser/.codex}"
    mkdir -p "${CODEX_HOME}"

    if [ -n "${CODEX_ACCESS_TOKEN:-}" ]; then
        echo "[entrypoint] Signing in to Codex with CODEX_ACCESS_TOKEN..."
        printf '%s' "${CODEX_ACCESS_TOKEN}" \
            | timeout 60s codex login --with-access-token > /dev/null 2>&1
        if [ $? -eq 0 ]; then
            echo "[entrypoint] Codex login complete."
        else
            echo "[entrypoint] Codex login failed. Check CODEX_ACCESS_TOKEN."
        fi
    elif [ -f "${CODEX_HOME}/auth.json" ]; then
        echo "[entrypoint] Found Codex auth cache at ${CODEX_HOME}/auth.json."
    else
        echo "[entrypoint] Codex is not signed in. For Pro/Plus, mount host ~/.codex or run codex login inside the container."
    fi
else
    echo "[entrypoint] codex command not found. Skipping Codex setup."
fi
set -e

# MCP setup
set +e

mcp_add() {
    local name=$1; shift
    claude mcp remove "${name}" > /dev/null 2>&1 || true
    claude mcp add "$@" > /dev/null 2>&1
}

echo "[entrypoint] Configuring MCP servers..."

# Remove project-level MCP config, unify to global
python3 - <<'EOF'
import json, os, sys

claude_json = os.path.expanduser("~/.claude.json")
if not os.path.exists(claude_json):
    sys.exit(0)

try:
    with open(claude_json) as f:
        data = json.load(f)
except (json.JSONDecodeError, OSError):
    sys.exit(0)

changed = False
for proj in data.get("projects", {}).values():
    mcp = proj.get("mcpServers", {})
    for name in ["notion", "wandb", "github"]:
        if name in mcp:
            del mcp[name]
            changed = True

# Drop the user-scoped github MCP (unified to gh CLI); guards against
# stale registrations persisting in a mounted ~/.claude.json
if "github" in data.get("mcpServers", {}):
    del data["mcpServers"]["github"]
    changed = True

if changed:
    with open(claude_json, "w") as f:
        json.dump(data, f, indent=2)
EOF

mcp_add wandb --transport http wandb https://mcp.withwandb.com/mcp \
    --scope user \
    -H "Authorization: Bearer ${WANDB_API_KEY}"

mcp_add notion notion notion-mcp-server \
    --scope user \
    -e "NOTION_TOKEN=${NOTION_TOKEN}"

echo "[entrypoint] MCP setup complete."

set -e

exec bash
