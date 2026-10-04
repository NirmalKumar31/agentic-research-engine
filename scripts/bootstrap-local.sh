#!/usr/bin/env bash
# Set up a local development environment: a venv with the locked core
# dependencies, the frontend's dependencies, and a check of what Ollama
# is actually serving -- recorded, not assumed.
#
# What this does NOT do: pull an immutable Ollama model digest. Ollama
# pulls by tag (`ollama pull qwen3:4b`), and the tag is not pinned to one
# digest -- a later pull of the same tag can serve different weights.
# This script records the digest it actually observed after pulling, to
# an untracked file, so a later `verify-local.sh` run can tell you
# whether what is installed now is what you bootstrapped with, not
# whether it matches some digest this script invented.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

REQUIRED_PYTHON_MAJOR_MINOR="3.12"
OLLAMA_MODEL="${OLLAMA_MODEL:-qwen3:4b}"
VENV_DIR="${VENV_DIR:-.venv}"
LOCAL_ENV_FILE=".local-environment.json"

echo "== Python version =="
PY_VERSION="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
echo "found: $PY_VERSION (pyproject.toml requires >=3.11,<3.13)"

echo
echo "== Virtual environment =="
if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv "$VENV_DIR"
  echo "created $VENV_DIR"
else
  echo "reusing existing $VENV_DIR"
fi

echo
echo "== Python dependencies =="
if [ -f requirements-lock.txt ]; then
  if "$VENV_DIR/bin/pip" install --require-hashes -r requirements-lock.txt 2>/dev/null; then
    echo "installed from requirements-lock.txt (hash-verified)"
  else
    echo "requirements-lock.txt did not install on this platform -- falling back"
    echo "to the unpinned ranges in pyproject.toml. This is expected on anything"
    echo "other than linux/cp312, which is the one platform the lock targets."
    "$VENV_DIR/bin/pip" install -e ".[dev,web]"
  fi
else
  "$VENV_DIR/bin/pip" install -e ".[dev,web]"
fi
echo "Not installed: the nli-local extra (torch). Install it yourself with"
echo "  $VENV_DIR/bin/pip install -e \".[nli-local]\""
echo "if you want local semantic verification instead of a remote endpoint --"
echo "see pyproject.toml for why it is not installed by default."

echo
echo "== Frontend dependencies =="
if command -v npm >/dev/null 2>&1; then
  (cd web && npm ci)
else
  echo "npm not found -- skipping. Install Node.js to build the web interface."
fi

echo
echo "== Ollama =="
OLLAMA_VERSION="unreachable"
OLLAMA_DIGEST=""
if command -v ollama >/dev/null 2>&1; then
  if ! ollama list >/dev/null 2>&1; then
    echo "ollama is installed but not running -- start it with: ollama serve"
  else
    OLLAMA_VERSION="$(ollama --version 2>/dev/null | head -1)"
    echo "server: $OLLAMA_VERSION"
    if ollama list 2>/dev/null | grep -q "^${OLLAMA_MODEL%%:*}"; then
      echo "model $OLLAMA_MODEL already present"
    else
      echo "pulling $OLLAMA_MODEL (this can take a while)..."
      ollama pull "$OLLAMA_MODEL"
    fi
    OLLAMA_DIGEST="$(curl -s http://localhost:11434/api/tags \
      | python3 -c "
import json, sys
models = json.load(sys.stdin).get('models', [])
for m in models:
    if m.get('name') == '$OLLAMA_MODEL':
        print(m.get('digest', '')[:16])
        break
" 2>/dev/null || true)"
  fi
else
  echo "ollama not found -- install it from https://ollama.com if you want"
  echo "LLM_MODE=local to work. Cloud mode needs no local model."
fi

python3 - "$LOCAL_ENV_FILE" "$PY_VERSION" "$OLLAMA_VERSION" "$OLLAMA_MODEL" "$OLLAMA_DIGEST" <<'PY'
import json, sys
path, py_version, ollama_version, ollama_model, ollama_digest = sys.argv[1:6]
json.dump(
    {
        "python_version": py_version,
        "ollama_version": ollama_version,
        "ollama_model": ollama_model,
        "ollama_digest_observed": ollama_digest,
        "note": "This file is untracked. It records what bootstrap actually saw, "
        "not an expectation -- pin EXPECTED_OLLAMA_DIGEST yourself in .env if you "
        "want verify-local.sh to refuse a later drift.",
    },
    open(path, "w"),
    indent=2,
)
PY
echo
echo "Recorded what was actually observed to $LOCAL_ENV_FILE (untracked)."
echo
echo "Next: $VENV_DIR/bin/agentic-research check"
echo "Then: ./scripts/verify-local.sh"
