#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ---------------------------------------------------------------------------
# Environment variables
# ---------------------------------------------------------------------------
SERVICE_LOG_DIR=${SERVICE_LOG_DIR:-"/var/log/tiger"}
# vLLM Engine
MODEL_DIR=${MODEL_DIR:-"/models/Qwen3-4B"}
SERVED_MODEL_NAME=${MODEL_DIR}
KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-"auto"}
TP_SIZE=${TP_SIZE:-1}
ENGINE_PORT=${ENGINE_PORT:-"8000"}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.90}
KV_CONNECTOR_CONFIG=${KV_CONNECTOR_CONFIG:-'{"kv_connector":"AIBrixPDReuseConnector","kv_role":"kv_both","kv_connector_module_path":"aibrix_kvcache.integration.vllm.kv_connector.aibrix_pd_reuse_connector"}'}
VLLM_DECODE_COMPILATION_CONFIG=${VLLM_DECODE_COMPILATION_CONFIG:-}
ENABLE_PREFIX_CACHING=${ENABLE_PREFIX_CACHING:-0}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-"16328"}
VLLM_ENGINE_STARTUP_TIMEOUT_SECS=${VLLM_ENGINE_STARTUP_TIMEOUT_SECS:-"300"}

DECODE_COMPILATION_CONFIG_ARGS=()
if [[ -n "$VLLM_DECODE_COMPILATION_CONFIG" ]]; then
    DECODE_COMPILATION_CONFIG_ARGS=(--compilation-config "$VLLM_DECODE_COMPILATION_CONFIG")
fi

PREFIX_CACHING_ARG=""
if [[ "$ENABLE_PREFIX_CACHING" -ne 1 ]]; then
    PREFIX_CACHING_ARG="--no-enable-prefix-caching"
fi

# log timestamp suffix shared by all services
LOG_TS_SUFFIX=$(date +%Y%m%d_%H%M%S)
# exports
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-"FLASHINFER"}
export AIBRIX_KV_CACHE_OL_BLOCK_SIZE=${AIBRIX_KV_CACHE_OL_BLOCK_SIZE:-64}
export AIBRIX_KV_CACHE_OL_L1_CACHE_ENABLED=${AIBRIX_KV_CACHE_OL_L1_CACHE_ENABLED:-0}
export AIBRIX_KV_CACHE_OL_L2_CACHE_BACKEND=JBOF
export AIBRIX_KV_CACHE_OL_JBOF_KV_ADDR="192.168.1.100"
export AIBRIX_KV_CACHE_OL_JBOF_KV_NQN="nqn.2016-06.io.spdk:cnode1"
export AIBRIX_KV_CACHE_OL_JBOF_KV_CORES=8
export AIBRIX_KV_CACHE_OL_JBOF_USE_IOV_API=0

# validations
service_env_validation() {
    # check log dir
    if [ ! -d "$SERVICE_LOG_DIR" ]; then
        echo "[entrypoint] Warn: Service log directory $SERVICE_LOG_DIR does not exist, creating it"
        mkdir -p "$SERVICE_LOG_DIR"
    fi
    return 0
}

engine_env_validation() {
    # check model dir
    if [ ! -d "$MODEL_DIR" ]; then
        echo "[entrypoint] Error: Model directory $MODEL_DIR does not exist"
        return 1
    fi

    return 0
}

service_env_validation || exit 1
engine_env_validation || exit 1

# ---------------------------------------------------------------------------
# PID tracking — arrays for multiple services
# ---------------------------------------------------------------------------
SERVICE_PIDS=()     # array of PIDs
SERVICE_NAMES=()    # array of human-readable names (same index)
LOG_FOLLOWER_PIDS=()

cleanup() {
    echo "[entrypoint] Shutting down all processes..."

    # Kill all services
    for pid in "${SERVICE_PIDS[@]}"; do
        kill -TERM "$pid" 2>/dev/null || true
    done

    for pid in "${LOG_FOLLOWER_PIDS[@]}"; do
        kill -TERM "$pid" 2>/dev/null || true
    done

    # Wait up to 10s for graceful shutdown
    for i in $(seq 1 10); do
        all_dead=true
        for pid in "${SERVICE_PIDS[@]}"; do
            kill -0 "$pid" 2>/dev/null && all_dead=false
        done
        $all_dead && break
        sleep 1
    done

    # Force kill anything still alive
    for pid in "${SERVICE_PIDS[@]}"; do
        kill -9 "$pid" 2>/dev/null || true
    done

    wait 2>/dev/null || true
    echo "[entrypoint] All processes stopped. Exiting with code ${EXIT_CODE:-1}." >&2
    exit "${EXIT_CODE:-1}"
}

trap cleanup SIGINT SIGTERM EXIT

# ---------------------------------------------------------------------------
# Helper: start a service and track its PID
# ---------------------------------------------------------------------------
start_service() {
    local name="$1"
    shift
    "$@" &
    local pid=$!
    SERVICE_PIDS+=("$pid")
    SERVICE_NAMES+=("$name")
    echo "[entrypoint] Started: ${name} (PID ${pid})"
}

start_service_with_log() {
    local name="$1"
    local log_prefix="$2"
    local log_file="$3"
    shift 3
    (
        exec "$@" \
            > >(awk -v prefix="$log_prefix" '{ print prefix " " $0; fflush() }' | tee -a "$log_file") \
            2> >(awk -v prefix="$log_prefix" '{ print prefix " " $0; fflush() }' | tee -a "$log_file" >&2)
    ) &
    local pid=$!
    SERVICE_PIDS+=("$pid")
    SERVICE_NAMES+=("$name")
    echo "[entrypoint] Started: ${name} (PID ${pid})"
}

start_log_follower() {
    local log_prefix="$1"
    local log_file="$2"
    touch "$log_file"
    tail -n 0 -F "$log_file" | awk -v prefix="$log_prefix" '{ print prefix " " $0; fflush() }' &
    LOG_FOLLOWER_PIDS+=("$!")
}

check_services() {
    for i in "${!SERVICE_PIDS[@]}"; do
        local pid="${SERVICE_PIDS[$i]}"
        local name="${SERVICE_NAMES[$i]}"
        if ! kill -0 "$pid" 2>/dev/null; then
            wait "$pid" 2>/dev/null; exit_code=$?
            echo "[entrypoint] Error: ${name} (PID ${pid}) died with code ${exit_code}"
            return 1
        fi
    done
    return 0
}

wait_for_server() {
    local name="$1"
    local endpoint="$2"
    local timeout=$3
    local end=$((SECONDS + timeout))
  
    while [ $SECONDS -lt $end ]; do
        # ensure all services are running
        check_services || return 1

        local status_code
        # avoid curl trigger set -euo pipefail
        status_code=$(curl -s -o /dev/null -w "%{http_code}" "$endpoint" || true)
        if [[ "$status_code" == "200" ]]; then
            echo "[entrypoint] $name is ready!!!"
            return 0
        fi
        sleep 3
        echo "[entrypoint] Waiting for $name to be ready... (${SECONDS}/${end}), status_code: ${status_code}"
    done
  
    return 1
}

# ---------------------------------------------------------------------------
# Phase 1: Start services
# ---------------------------------------------------------------------------
# >>> ADD YOUR SERVICES HERE <<<
# Usage: start_service "Human Name" command arg1 arg2 ...
#
# In production, replace fake_service.py with your real services, e.g.:
#   start_service "vLLM Engine" python3 -m vllm.entrypoints.openai.api_server \
#       --model /models/llama --host 0.0.0.0 --port 8000
#   start_service "Main API" python3 api_server.py --port 8080
#   ...

# vLLM Engine
# cache directory used by engines
CACHE_DIR=~/.vllm_cache
rm -rf $CACHE_DIR
mkdir -p $CACHE_DIR
echo "[entrypoint] Launching engine with CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
export TORCHINDUCTOR_CACHE_DIR=$CACHE_DIR/torch_cache_0
export VLLM_CACHE_ROOT=$CACHE_DIR/vllm_cache_0
export FLASHINFER_CACHE_DIR=$CACHE_DIR/flashinfer_cache_0
start_service_with_log "vLLM Engine" "[entrypoint] [engine]" "$SERVICE_LOG_DIR/vllm_engine$LOG_TS_SUFFIX.log" python3 -m vllm.entrypoints.openai.api_server \
--port=$ENGINE_PORT \
--uvicorn-log-level=warning \
--model=$MODEL_DIR \
--served-model-name $SERVED_MODEL_NAME \
--trust-remote-code \
--disable-log-requests \
--disable-fastapi-docs \
--swap-space=0 \
$PREFIX_CACHING_ARG \
--kv-transfer-config=$KV_CONNECTOR_CONFIG \
--tensor-parallel-size=$TP_SIZE \
--gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
--kv-cache-dtype $KV_CACHE_DTYPE \
--max-model-len $MAX_MODEL_LEN \
--async-scheduling
sleep 5

unset CUDA_VISIBLE_DEVICES
unset TORCHINDUCTOR_CACHE_DIR
unset VLLM_CACHE_ROOT
unset FLASHINFER_CACHE_DIR

wait_for_server "vLLM Engine" "http://0.0.0.0:${ENGINE_PORT}/health" $VLLM_ENGINE_STARTUP_TIMEOUT_SECS || {
    echo "[entrypoint] Failed to start vLLM Engine"
    exit 1
}
echo ""

# Print summary
echo "--- Process table ---"
for i in "${!SERVICE_PIDS[@]}"; do
    echo "  PID ${SERVICE_PIDS[$i]}  ${SERVICE_NAMES[$i]}"
done
echo ""

sleep infinity
