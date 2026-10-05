#!/usr/bin/env bash
#
# Deploy gate for a pydantic-agent: prove the agent we just deployed is actually
# dialable by its caller, or fail the deploy.
#
# Why this exists (and why it is NOT a CI step): reachability is a property of an
# address AND a vantage point. The addresses the gateway dials are container
# names that resolve only on the caller's Docker network (`ai_proxy`), so this
# must run ON THE HOST THAT HOSTS THE DEPLOYMENT, next to the containers - not on
# a CI runner, which is on a different network and would report every agent
# unreachable for the wrong reason. See deploy/README.md and agent/probe.py.
#
# Usage (from the directory holding docker-compose.yml):
#
#     ./scripts/deploy-gate.sh
#
# Environment:
#   AGENT_NAME          agent name to gate on        (default: data-sourcing-agent)
#   SERVICE             compose service name         (default: app)
#   WAIT_SECONDS        how long to wait for the agent to answer (default: 60)
#   LITELLM_MASTER_KEY  key that can list the fleet; if unset, the container's
#                       own registration key is used (it owns the agent it created)
#
# Exit: 0 if this agent is dialable and its served card agrees with its
# registration; non-zero otherwise, so it can gate a deploy or an auto-update.

set -euo pipefail

AGENT_NAME="${AGENT_NAME:-data-sourcing-agent}"
SERVICE="${SERVICE:-app}"
WAIT_SECONDS="${WAIT_SECONDS:-60}"

# Only pass LITELLM_MASTER_KEY through if the operator set it; otherwise the
# probe falls back to the container's own LITELLM_API_KEY / registration key.
exec_env=()
if [[ -n "${LITELLM_MASTER_KEY:-}" ]]; then
  exec_env=(-e "LITELLM_MASTER_KEY=${LITELLM_MASTER_KEY}")
fi

echo "deploy-gate: bringing up '${SERVICE}'..."
docker compose up -d

# Poll rather than assume: `up -d` returns as soon as the container is created,
# and the agent is not listening for a moment after that. A freshly-created
# container legitimately answers nothing yet; that is "not up", not "unreachable",
# so we retry until the deadline and only then call it a failure.
echo "deploy-gate: waiting up to ${WAIT_SECONDS}s for '${AGENT_NAME}' to answer..."
deadline=$((SECONDS + WAIT_SECONDS))
output=""
while true; do
  if output="$(docker compose exec -T "${exec_env[@]}" "${SERVICE}" \
        python -m agent.probe --agent "${AGENT_NAME}" 2>&1)"; then
    echo "${output}"
    echo "deploy-gate: '${AGENT_NAME}' is dialable and consistent with its registration."
    exit 0
  fi
  if ((SECONDS >= deadline)); then
    echo "${output}"
    echo "deploy-gate: '${AGENT_NAME}' did not become dialable within ${WAIT_SECONDS}s" >&2
    exit 1
  fi
  sleep 2
done
