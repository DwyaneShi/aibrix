#!/usr/bin/env bash
#
# Run the AIBrix metadata service.
#
# Assumes the runtime dependencies (everything in requirements.txt) are
# already installed in the target Python — typically via:
#   pip install -r requirements.txt
# in the image build / provisioning step.
#
# This script installs the local aibrix-*.whl (with --no-deps, since deps
# are already there) and then execs aibrix_metadata.
#
# Override defaults via env:
#   PYTHON_BIN=python3.11

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

log() { printf '[run.sh %(%H:%M:%S)T] %s\n' -1 "$*"; }

log "script dir : ${SCRIPT_DIR}"
log "python bin : ${PYTHON_BIN}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  log "ERROR: ${PYTHON_BIN} not found on PATH"
  exit 1
fi
log "python ver : $("${PYTHON_BIN}" -V 2>&1)"

# Locate the bundled aibrix wheel next to this script.
WHEEL="$(ls -t "${SCRIPT_DIR}/"aibrix-*.whl 2>/dev/null | head -n 1 || true)"
if [[ -z "${WHEEL}" ]]; then
  log "ERROR: no aibrix-*.whl next to ${SCRIPT_DIR} — did build.sh run?"
  exit 1
fi
log "wheel found: $(basename "${WHEEL}")"

# Install (or upgrade) just the aibrix package — deps are expected to be
# pre-provisioned, so --no-deps avoids touching them and dodges the offline
# install pain entirely. Idempotent: skip when wheel version matches.
#
# Use importlib.metadata instead of `pip show`: pip's startup is slow and
# some pip versions silently network out to check for self-updates, which
# can hang for minutes. importlib.metadata is stdlib, instant, no network.
WHEEL_VER="$(basename "${WHEEL}" | sed -E 's/^aibrix-([^-]+)-.*$/\1/')"
log "wheel ver  : ${WHEEL_VER}"
log "checking installed aibrix version (via importlib.metadata)..."
INSTALLED_VER="$("${PYTHON_BIN}" -c \
  'import importlib.metadata as m
try: print(m.version("aibrix"))
except m.PackageNotFoundError: pass' 2>/dev/null || true)"
log "installed  : ${INSTALLED_VER:-<none>}"

if [[ "${INSTALLED_VER}" != "${WHEEL_VER}" ]]; then
  log "==> Installing $(basename "${WHEEL}") (--no-deps) into ${PYTHON_BIN}"
  ERR_LOG="$(mktemp)"
  if ! "${PYTHON_BIN}" -m pip install --no-deps --force-reinstall "${WHEEL}" \
        2> >(tee "${ERR_LOG}" >&2); then
    if grep -qi "externally-managed-environment" "${ERR_LOG}"; then
      log "==> PEP 668 detected — retrying with --break-system-packages"
      "${PYTHON_BIN}" -m pip install --break-system-packages \
        --no-deps --force-reinstall "${WHEEL}"
    else
      rm -f "${ERR_LOG}"
      log "ERROR: pip install failed (see error above)"
      exit 1
    fi
  fi
  rm -f "${ERR_LOG}"
  log "install ok"
else
  log "skip install (already at ${WHEEL_VER})"
fi


# Translate select env vars into CLI flags. The metadata app's CLI options
# don't go through pydantic-settings, so this layer bridges deploy-time env
# (k8s manifest / docker-compose) to argparse without code changes.
EXTRA_ARGS=()
case "${DRY_RUN:-0}" in
  1|true|TRUE|True|yes|YES|Yes)
    EXTRA_ARGS+=(--dry-run)
    log "DRY_RUN=${DRY_RUN} -> appending --dry-run"
    ;;
  *)
    log "DRY_RUN not set"
    ;;
esac


 # Prefer the generated console_script when its bin dir is on PATH; otherwise
 # invoke the module form so we don't depend on PATH wiring.
if command -v aibrix_metadata >/dev/null 2>&1; then
  ENTRY=("aibrix_metadata")
  log "entry point: $(command -v aibrix_metadata)"
else
  ENTRY=("${PYTHON_BIN}" -m aibrix.metadata.app)
  log "entry point: ${PYTHON_BIN} -m aibrix.metadata.app (console_script not on PATH)"
fi

log "==> exec: ${ENTRY[*]} ${EXTRA_ARGS[*]} $*"
exec "${ENTRY[@]}" "${EXTRA_ARGS[@]}" "$@"
