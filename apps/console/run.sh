#!/usr/bin/env bash
#
# Run the AIBrix Console from a built artifact directory.
#
# When this script is copied next to ./console and ./web/ (as build.sh does),
# you can simply: ./run.sh
#
# Override defaults via env:
#   HTTP_ADDR=:8080         # HTTP listener for SPA + REST/JSON
#   GRPC_ADDR=:50060        # gRPC listener
#   STATIC_FILES_DIR=...    # SPA root (defaults to ./web next to this script)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BIN="${SCRIPT_DIR}/console"
if [[ ! -x "${BIN}" ]]; then
  echo "ERROR: console binary not found at ${BIN}" >&2
  echo "       Run apps/console/build.sh first." >&2
  exit 1
fi

export STATIC_FILES_DIR="${STATIC_FILES_DIR:-${SCRIPT_DIR}/web}"
export HTTP_ADDR="${HTTP_ADDR:-:8080}"
export GRPC_ADDR="${GRPC_ADDR:-:50060}"

echo "==> Starting console"
echo "    HTTP_ADDR=${HTTP_ADDR}"
echo "    GRPC_ADDR=${GRPC_ADDR}"
echo "    STATIC_FILES_DIR=${STATIC_FILES_DIR}"

exec "${BIN}" "$@"
