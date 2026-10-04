#!/usr/bin/env bash
# Verify a local environment is what bootstrap-local.sh set up, and that
# the engine itself is deterministic. Refuses rather than warns on
# anything that would make a local result unreproducible.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

VENV_DIR="${VENV_DIR:-.venv}"
BIN="$VENV_DIR/bin"
LOCAL_ENV_FILE=".local-environment.json"
FAILED=0

if [ ! -x "$BIN/agentic-research" ]; then
  echo "FAIL: $BIN/agentic-research not found -- run ./scripts/bootstrap-local.sh first"
  exit 1
fi

echo "== Configuration and provider reachability =="
if ! "$BIN/agentic-research" check; then
  echo "FAIL: agentic-research check reported a blocker above"
  FAILED=1
fi

echo
echo "== Frozen-corpus engine determinism (Level 2) =="
if ! "$BIN/agentic-research" verify-reproducible; then
  echo "FAIL: engine determinism check failed"
  FAILED=1
fi

echo
echo "== Recorded-run replay (Level 1) =="
if ! "$BIN/agentic-research" replay list >/dev/null; then
  echo "FAIL: replay listing failed"
  FAILED=1
fi

echo
echo "== Ollama digest =="
if [ -n "${EXPECTED_OLLAMA_DIGEST:-}" ]; then
  if [ ! -f "$LOCAL_ENV_FILE" ]; then
    echo "FAIL: EXPECTED_OLLAMA_DIGEST is set but $LOCAL_ENV_FILE does not exist -- "
    echo "      run bootstrap-local.sh first so there is something to compare against"
    FAILED=1
  else
    OBSERVED="$(python3 -c "import json; print(json.load(open('$LOCAL_ENV_FILE')).get('ollama_digest_observed',''))")"
    if [ "$OBSERVED" != "$EXPECTED_OLLAMA_DIGEST" ]; then
      echo "FAIL: Ollama is serving digest '$OBSERVED', pinned expectation is"
      echo "      '$EXPECTED_OLLAMA_DIGEST'. A moving tag means these can differ"
      echo "      even with no action on your part -- refusing rather than"
      echo "      silently treating them as equivalent."
      FAILED=1
    else
      echo "ok: observed digest matches EXPECTED_OLLAMA_DIGEST"
    fi
  fi
else
  echo "no EXPECTED_OLLAMA_DIGEST pinned -- nothing to compare against."
  echo "This is a real limitation, not a check that passed: a moving tag"
  echo "means the model actually running can change without this script"
  echo "noticing, unless you pin EXPECTED_OLLAMA_DIGEST yourself."
  if [ -f "$LOCAL_ENV_FILE" ]; then
    OBSERVED="$(python3 -c "import json; print(json.load(open('$LOCAL_ENV_FILE')).get('ollama_digest_observed',''))" 2>/dev/null || true)"
    [ -n "$OBSERVED" ] && echo "Currently observed: $OBSERVED (export EXPECTED_OLLAMA_DIGEST=$OBSERVED to pin it)"
  fi
fi

echo
if [ "$FAILED" -ne 0 ]; then
  echo "FAILED: see the lines above."
  exit 1
fi
echo "All checks passed."
