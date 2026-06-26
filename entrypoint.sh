#!/usr/bin/env bash
set -euo pipefail

# Drone entrypoint — connects to hive with pre-provisioned credentials.
#
# All credentials are provided as env vars by `bbot-hive create-drone`.
# The drone reads them via pydantic-settings (BBOT_BEE_ prefix).
#
# Required env vars:
#   BBOT_BEE_ID               — Drone ID
#   BBOT_BEE_HIVE_URL         — WebSocket URL of the hive
#   BBOT_BEE_API_KEY          — API key for hive authentication
#
# Optional env vars:
#   BBOT_BEE_MAX_CONCURRENT_SCANS — Max concurrent scans (default: 3)
#   BBOT_BEE_LOG_LEVEL            — Log level (default: INFO)

# Validate required vars early
: "${BBOT_BEE_HIVE_URL:?BBOT_BEE_HIVE_URL is required}"
: "${BBOT_BEE_ID:?BBOT_BEE_ID is required}"
: "${BBOT_BEE_API_KEY:?BBOT_BEE_API_KEY is required}"

echo "[entrypoint] Drone ${BBOT_BEE_ID} connecting to ${BBOT_BEE_HIVE_URL}"

# In dev mode, re-sync deps to pick up volume-mounted source changes.
cd /app/bbot_bee
if [ "${UV_PROJECT_ENVIRONMENT:-}" = "/venv-dev" ]; then
    echo "[entrypoint] Dev mode — syncing dependencies..."
    uv sync --frozen
    exec uv run bbot-bee
else
    exec bbot-bee
fi
