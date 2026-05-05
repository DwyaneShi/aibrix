#!/usr/bin/env bash
#
# Build the AIBrix Console (Go backend + React SPA) and stage the artifacts
# under <repo>/output/dist/. Mirrors what apps/console/Dockerfile does,
# but produces a host-runnable layout for local development.
#
# If Node.js / npm is missing, the script bootstraps it via nvm:
#   - Linux (apt): installs curl / ca-certificates / build-essential first
#   - macOS / others: assumes curl + a C toolchain are already present
# Pinned versions match what's been validated for this repo:
#   nvm v0.40.3, Node.js 24.5.0, npm 11.5.1
#
# Usage:
#   apps/console/build.sh                       # full build
#   SKIP_NPM_INSTALL=1 apps/console/build.sh    # skip `npm install`
#   SKIP_WEB=1 apps/console/build.sh            # backend only
#   SKIP_API=1 apps/console/build.sh            # SPA only
#   SKIP_NODE_BOOTSTRAP=1 apps/console/build.sh # never auto-install Node

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WEB_DIR="${SCRIPT_DIR}/web"
OUT_DIR="${REPO_ROOT}/output/dist"

NVM_VERSION="${NVM_VERSION:-v0.40.3}"
NODE_VERSION="${NODE_VERSION:-24.5.0}"
NPM_VERSION="${NPM_VERSION:-11.5.1}"


mkdir -p "${OUT_DIR}"

# Ensure node + npm are available. If missing, install via nvm using the
# pinned versions above. Safe to call repeatedly — it's a no-op when the
# binaries already exist.
ensure_node() {
  if command -v npm >/dev/null 2>&1 && command -v node >/dev/null 2>&1; then
    echo "==> Using existing node $(node -v) / npm $(npm -v)"
    return 0
  fi

  if [[ "${SKIP_NODE_BOOTSTRAP:-0}" == "1" ]]; then
    echo "ERROR: node/npm not found and SKIP_NODE_BOOTSTRAP=1" >&2
    exit 1
  fi

  echo "==> node/npm not found — bootstrapping nvm ${NVM_VERSION} + Node ${NODE_VERSION}"

  # Linux apt prerequisites. Skip silently on macOS or when apt is missing.
  if command -v apt-get >/dev/null 2>&1; then
    SUDO=""
    if [[ $EUID -ne 0 ]]; then
      if command -v sudo >/dev/null 2>&1; then SUDO="sudo"; fi
    fi
    ${SUDO} apt-get update
    ${SUDO} apt-get install -y curl ca-certificates build-essential
  fi

  export NVM_DIR="${NVM_DIR:-$HOME/.nvm}"
  # nvm.sh refuses to source if NVM_DIR doesn't exist, and the installer
  # occasionally fails to create it (e.g. when NVM_DIR was inherited from
  # a different user's shell). Pre-create it so both paths are safe.
  mkdir -p "${NVM_DIR}"
  if [[ ! -s "${NVM_DIR}/nvm.sh" ]]; then
    curl -o- "https://raw.githubusercontent.com/nvm-sh/nvm/${NVM_VERSION}/install.sh" | bash
  fi

  if [[ ! -s "${NVM_DIR}/nvm.sh" ]]; then
    echo "ERROR: nvm install finished but ${NVM_DIR}/nvm.sh is missing." >&2
    echo "       Check NVM_DIR (currently ${NVM_DIR}) — it may point at the wrong user's home." >&2
    exit 1
  fi

  # shellcheck disable=SC1091
  . "${NVM_DIR}/nvm.sh"

  nvm install "${NODE_VERSION}"
  nvm use "${NODE_VERSION}"
  nvm alias default "${NODE_VERSION}"
  npm install -g "npm@${NPM_VERSION}"

  echo "==> Installed node $(node -v) / npm $(npm -v)"
}

if [[ "${SKIP_WEB:-0}" != "1" ]]; then
  ensure_node

  echo "==> Building React SPA (apps/console/web)"
  pushd "${WEB_DIR}" >/dev/null
  if [[ "${SKIP_NPM_INSTALL:-0}" != "1" ]]; then
    npm install
  fi
  npm run build
  popd >/dev/null

  echo "==> Staging SPA -> ${OUT_DIR}/web"
  rm -rf "${OUT_DIR}/web"
  mkdir -p "${OUT_DIR}/web"
  cp -R "${WEB_DIR}/build/." "${OUT_DIR}/web/"
fi

if [[ "${SKIP_API:-0}" != "1" ]]; then
  echo "==> Building console binary (CGO_ENABLED=1, tags=nozmq)"
  # CGO_ENABLED=1 is required by gorm.io/driver/sqlite (default dev store).
  # nozmq matches the Dockerfile — console doesn't need ZeroMQ.
  pushd "${REPO_ROOT}" >/dev/null
  CGO_ENABLED=1 go build -tags nozmq -o "${OUT_DIR}/console" ./cmd/console
  popd >/dev/null
fi

# Stage build.sh + run.sh next to the artifacts so the dist directory is
# self-contained: ship the folder anywhere and run ./run.sh.
echo "==> Copying build.sh + run.sh -> ${OUT_DIR}"
cp "${SCRIPT_DIR}/build.sh" "${OUT_DIR}/build.sh"
cp "${SCRIPT_DIR}/run.sh"   "${OUT_DIR}/run.sh"

cat <<EOF

✅ Console artifacts built at: ${OUT_DIR}
   ├── console        # Go HTTP+gRPC server
   ├── web/           # React SPA (served at /)
   ├── run.sh         # startup script (uses ./console + ./web by default)
   └── build.sh       # rebuild script (for reference)

To run locally:
  ${OUT_DIR}/run.sh

Override defaults via env:
  HTTP_ADDR=:8080 GRPC_ADDR=:50060 STATIC_FILES_DIR=... ${OUT_DIR}/run.sh

Then open http://localhost:8080 in your browser.

To build the container image instead (uses this same layout internally):
  (cd apps/console/web && npm install && npm run build)
  docker build -f apps/console/Dockerfile -t aibrix/console .
EOF
