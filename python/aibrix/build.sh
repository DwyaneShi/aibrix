#!/usr/bin/env bash
#
# Build the AIBrix Python package (metadata-service / runtime / batch-worker
# share the same wheel). Stages a minimal bundle under <repo>/output/metadata/.
#
# This script does NOT bundle dependency wheels. The target host is expected
# to have the deps pre-installed (typically via `pip install -r requirements.txt`
# in the provisioning step). That sidesteps every cross-platform / sdist /
# glibc / wheel-availability headache that comes with shipping a wheelhouse.
#
# Usage:
#   ./build.sh              # Build without internal dependencies
#   ./build.sh --with-internal  # Build with internal dependencies
#
# Output layout:
#   output/metadata/
#   ├── aibrix-<ver>-py3-none-any.whl   # the app wheel
#   ├── requirements.txt                # pinned runtime deps (no profiling/dev)
#   ├── build.sh                        # this script (for reference)
#   └── run.sh                          # startup script (pip install + exec)

set -euo pipefail

# Parse arguments
WITH_INTERNAL=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-internal)
      WITH_INTERNAL=true
      shift
      ;;
    *)
      echo "ERROR: Unknown argument: $1" >&2
      echo "Usage: $0 [--with-internal]" >&2
      exit 1
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUT_DIR="${REPO_ROOT}/output/metadata"

PYTHON_BIN="${PYTHON_BIN:-python3}"
POETRY_VERSION="${POETRY_VERSION:-1.8.3}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "ERROR: ${PYTHON_BIN} not found on PATH" >&2
  exit 1
fi

# Install poetry via pip --user if not present. The Dockerfile pins 1.8.3 — keep parity.
if ! command -v poetry >/dev/null 2>&1; then
  echo "==> Installing poetry ${POETRY_VERSION} (pip --user)"
  "${PYTHON_BIN}" -m pip install --user "poetry==${POETRY_VERSION}"
  USER_BIN="$("${PYTHON_BIN}" -m site --user-base)/bin"
  export PATH="${USER_BIN}:${PATH}"
fi

# poetry export lives in a plugin since 1.2; install if absent. No-op when already there.
if ! poetry self show plugins 2>/dev/null | grep -q poetry-plugin-export; then
  echo "==> Installing poetry-plugin-export"
  poetry self add poetry-plugin-export >/dev/null
fi

mkdir -p "${OUT_DIR}"

echo "==> Building aibrix wheel (excluding profiling/dev groups)"
pushd "${SCRIPT_DIR}" >/dev/null

# Construct poetry install command based on --with-internal flag
POETRY_INSTALL_ARGS="--without profiling --without dev --no-interaction --no-root"
if [[ "${WITH_INTERNAL}" == "true" ]]; then
  POETRY_INSTALL_ARGS="--with internal ${POETRY_INSTALL_ARGS}"
  echo "    (including internal dependencies)"
fi

poetry install ${POETRY_INSTALL_ARGS}
poetry build -f wheel
popd >/dev/null

# Pick the freshest wheel — dist/ may have older builds lying around.
WHEEL="$(ls -t "${SCRIPT_DIR}/dist/"aibrix-*.whl | head -n 1)"
if [[ -z "${WHEEL}" ]]; then
  echo "ERROR: no wheel produced under ${SCRIPT_DIR}/dist" >&2
  exit 1
fi
cp "${WHEEL}" "${OUT_DIR}/"

echo "==> Exporting requirements.txt (matching wheel's install_requires)"
pushd "${SCRIPT_DIR}" >/dev/null

# Construct poetry export command based on --with-internal flag
POETRY_EXPORT_ARGS="--without profiling --without dev --without-hashes -f requirements.txt -o ${OUT_DIR}/requirements.txt"
if [[ "${WITH_INTERNAL}" == "true" ]]; then
  POETRY_EXPORT_ARGS="--with internal ${POETRY_EXPORT_ARGS}"
fi

poetry export ${POETRY_EXPORT_ARGS}
popd >/dev/null

echo "==> Copying build.sh + run.sh -> ${OUT_DIR}"
cp "${SCRIPT_DIR}/build.sh" "${OUT_DIR}/build.sh"
cp "${SCRIPT_DIR}/run.sh"   "${OUT_DIR}/run.sh"
chmod +x "${OUT_DIR}/build.sh" "${OUT_DIR}/run.sh"

WHEEL_NAME="$(basename "${WHEEL}")"

cat <<EOF

✅ Metadata artifacts at: ${OUT_DIR}
   ├── ${WHEEL_NAME}
   ├── requirements.txt
   ├── build.sh
   └── run.sh

On the target host (deps installed separately, then the wheel):
  pip install -r ${OUT_DIR}/requirements.txt        # one-time, pre-provision
  ${OUT_DIR}/run.sh --port 8090 --enable-k8s-job --k8s-namespace default
EOF
