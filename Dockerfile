# syntax=docker/dockerfile:1.7
# ---------- Stage 1: builder ----------
FROM ghcr.io/astral-sh/uv:0.7-python3.12-bookworm-slim AS builder

# git + ssh client: bbot-swarm-common is a private repo pulled over SSH.
RUN apt-get update && apt-get install -y --no-install-recommends git openssh-client \
    && rm -rf /var/lib/apt/lists/*

ENV UV_PROJECT_ENVIRONMENT=/venv
# Avoid interactive host-key prompts for github.com in the ephemeral builder.
ENV GIT_SSH_COMMAND="ssh -o StrictHostKeyChecking=accept-new"

WORKDIR /app/bbot_bee

# Phase 1: install deps only (cached until lockfile changes).
# bbot-swarm-common (private) and bbot are fetched from git; SSH agent is
# forwarded via BuildKit (`docker build --ssh default ...`).
COPY pyproject.toml uv.lock ./
RUN --mount=type=ssh uv sync --frozen --no-install-project --no-dev

# Phase 2: install project
COPY . ./
RUN --mount=type=ssh uv sync --frozen --no-dev

# ---------- Stage 2: runtime ----------
FROM python:3.12-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl jq autoconf automake libtool gcc make libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# venv has bbot-swarm-common installed (from git); the project source must be
# present for its editable install to resolve at runtime.
COPY --from=builder /venv /venv
COPY --from=builder /app/bbot_bee /app/bbot_bee

# Entrypoint
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Runs as root — bbot's ansible needs root to install module deps at runtime.
# The container itself is the security boundary.
ENV PATH="/venv/bin:$PATH"

# Pre-install all bbot module dependencies at build time so scans start instantly.
# Without this, the first scan on a fresh container spends minutes installing deps,
# and concurrent scans contend on ansible/pip/apt locks.
RUN bbot --install-all-deps

ENTRYPOINT ["/entrypoint.sh"]
