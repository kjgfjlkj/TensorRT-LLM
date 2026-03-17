#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# nsys_profile.sh — Automated multi-host nsys profiling for TensorRT-LLM.
#
# Usage:
#   bash nsys_profile.sh [--config <path>] [--dry-run] [--mock-gpus <spec>]
#
# Options:
#   --config <path>    Path to config file (default: nsys_profile.conf in same dir)
#   --dry-run          Validate config and GPU detection logic, then exit without
#                      launching server or profiler.
#   --mock-gpus <spec> Comma-separated list of idx:mem_mb:util to simulate GPU
#                      output (implies --dry-run). Example:
#                      "0:50:0,1:0:0,2:0:0,3:0:0,4:0:0,5:0:0,6:0:0,7:0:0"

set -euo pipefail

# ---------------------------------------------------------------------------
# Global state (written by various functions, read by cleanup)
# ---------------------------------------------------------------------------
REMOTE_BG_PID=""        # PID of the local SSH process running the server
NSYS_PID=""             # nsys PID inside the container
TARGET_MACHINE=""       # Selected machine
TARGET_GPU_IDS=""       # e.g. "0,1,2,3"
REPORT_DIR=""           # Full path on the network share for this run
RUN_TIMESTAMP=""        # Timestamp used to name the run directory
RUN_ID=""               # Short run identifier used for /tmp paths
TMP_REMOTE_DIR=""       # /tmp/nsys_profile_${RUN_ID} on the remote host
MAIN_LOG=""             # Path to the main log file on the network share
DRY_RUN=false
MOCK_GPUS=""            # Non-empty when --mock-gpus is provided

# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "${MAIN_LOG:-/dev/stderr}"; }
err() { echo "[$(date '+%H:%M:%S')] ERROR: $*" >&2 | tee -a "${MAIN_LOG:-/dev/stderr}" || true; echo "[$(date '+%H:%M:%S')] ERROR: $*" >&2; }
die() { err "$*"; exit 1; }

ssh_cmd() {
    # ssh_cmd <machine> <command...>
    # Runs a command on the remote host via SSH.
    local machine="$1"; shift
    # shellcheck disable=SC2086
    ssh ${SSH_OPTS} -i "${SSH_KEY}" "${machine}" "$@"
}

docker_exec() {
    # docker_exec <machine> <command...>
    # Runs a command inside the Docker container on the remote host.
    local machine="$1"; shift
    ssh_cmd "${machine}" docker exec "${DOCKER_CONTAINER_NAME}" bash -c "$*"
}

# ---------------------------------------------------------------------------
# Cleanup (registered as EXIT/ERR/INT/TERM trap)
# ---------------------------------------------------------------------------
cleanup() {
    local exit_code=$?
    log "cleanup: starting (exit_code=${exit_code})"

    # Kill nsys and server inside the container
    if [[ -n "${TARGET_MACHINE}" ]]; then
        if [[ -n "${NSYS_PID}" ]]; then
            log "cleanup: killing nsys PID ${NSYS_PID} in container on ${TARGET_MACHINE}"
            ssh_cmd "${TARGET_MACHINE}" \
                docker exec "${DOCKER_CONTAINER_NAME}" kill "${NSYS_PID}" 2>/dev/null || true
        fi
        # Belt-and-suspenders: pkill the server command substring
        log "cleanup: pkill server command in container on ${TARGET_MACHINE}"
        ssh_cmd "${TARGET_MACHINE}" \
            docker exec "${DOCKER_CONTAINER_NAME}" \
            bash -c "pkill -f $(printf '%q' "${SERVER_CMD%% *}") 2>/dev/null || true" 2>/dev/null || true

        # Remove remote temporary directory
        if [[ -n "${TMP_REMOTE_DIR}" ]]; then
            log "cleanup: removing ${TMP_REMOTE_DIR} on ${TARGET_MACHINE}"
            ssh_cmd "${TARGET_MACHINE}" rm -rf "${TMP_REMOTE_DIR}" 2>/dev/null || true
        fi
    fi

    # Kill local background SSH process
    if [[ -n "${REMOTE_BG_PID}" ]]; then
        log "cleanup: killing local SSH background PID ${REMOTE_BG_PID}"
        kill "${REMOTE_BG_PID}" 2>/dev/null || true
    fi

    log "cleanup: done"
}

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
parse_args() {
    local script_dir
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    CONFIG_FILE="${script_dir}/nsys_profile.conf"

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --config)
                shift
                CONFIG_FILE="$1"
                ;;
            --dry-run)
                DRY_RUN=true
                ;;
            --mock-gpus)
                shift
                MOCK_GPUS="$1"
                DRY_RUN=true
                ;;
            *)
                die "Unknown argument: $1"
                ;;
        esac
        shift
    done
}

# ---------------------------------------------------------------------------
# Config loading & validation
# ---------------------------------------------------------------------------
load_config() {
    [[ -f "${CONFIG_FILE}" ]] || die "Config file not found: ${CONFIG_FILE}"
    log "Loading config from: ${CONFIG_FILE}"
    # shellcheck source=/dev/null
    source "${CONFIG_FILE}"

    # Validate required variables
    local required_vars=(
        MACHINE_TYPE NUM_GPUS TOTAL_GPUS_PER_NODE
        DOCKER_CONTAINER_NAME ENV_SETUP_SCRIPT
        SERVER_CMD SERVER_PORT SERVER_STARTUP_TIMEOUT
        CLIENT_CMD NSYS_OPTS NSYS_OUTPUT_NAME
        OUTPUT_BASE_DIR SSH_KEY SSH_OPTS
        GPU_POLL_INTERVAL GPU_FREE_MEM_MB GPU_FREE_UTIL
    )
    for v in "${required_vars[@]}"; do
        [[ -n "${!v:-}" ]] || die "Config variable '${v}' is not set"
    done

    # Validate MACHINE_LISTS
    [[ ${#MACHINE_LISTS[@]} -gt 0 ]] || die "MACHINE_LISTS is empty"
    [[ -n "${MACHINE_LISTS[${MACHINE_TYPE}]:-}" ]] || \
        die "MACHINE_TYPE '${MACHINE_TYPE}' not found in MACHINE_LISTS"

    # Validate NUM_GPUS is a power of 2
    if (( NUM_GPUS == 0 )) || (( NUM_GPUS & (NUM_GPUS - 1) )); then
        die "NUM_GPUS=${NUM_GPUS} is not a power of 2"
    fi

    # Validate NUM_GPUS <= TOTAL_GPUS_PER_NODE
    (( NUM_GPUS <= TOTAL_GPUS_PER_NODE )) || \
        die "NUM_GPUS=${NUM_GPUS} exceeds TOTAL_GPUS_PER_NODE=${TOTAL_GPUS_PER_NODE}"

    log "Config validated OK"
}

# ---------------------------------------------------------------------------
# Output directory setup
# ---------------------------------------------------------------------------
setup_output_dir() {
    RUN_TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
    RUN_ID="${RUN_TIMESTAMP}_$$"
    TMP_REMOTE_DIR="/tmp/nsys_profile_${RUN_ID}"
    REPORT_DIR="${OUTPUT_BASE_DIR}/${RUN_TIMESTAMP}_${MACHINE_TYPE}_tp${NUM_GPUS}"
    MAIN_LOG="${OUTPUT_BASE_DIR}/run_${RUN_TIMESTAMP}.log"

    log "Creating output directory: ${REPORT_DIR}"
    mkdir -p "${REPORT_DIR}" || die "Failed to create output directory: ${REPORT_DIR}"
    touch "${MAIN_LOG}"
    log "Main log: ${MAIN_LOG}"
}

# ---------------------------------------------------------------------------
# SSH reachability check
# ---------------------------------------------------------------------------
check_ssh_reachable() {
    local machine="$1"
    ssh_cmd "${machine}" true 2>/dev/null
}

# ---------------------------------------------------------------------------
# GPU availability: find an aligned free group
# ---------------------------------------------------------------------------
# Parses nvidia-smi CSV output and looks for the first aligned group of
# NUM_GPUS consecutive GPUs that are all idle (mem < GPU_FREE_MEM_MB AND
# utilization == GPU_FREE_UTIL).
#
# Returns (via echo) a comma-separated GPU ID string on success, e.g. "4,5,6,7".
# Returns exit code 1 if no suitable group is found.
find_aligned_gpu_group() {
    local nvidia_smi_output="$1"  # Multi-line: "idx, mem_used, util\n..."
    local n="${NUM_GPUS}"
    local total="${TOTAL_GPUS_PER_NODE}"

    # Parse into arrays indexed by GPU index
    declare -A gpu_mem gpu_util
    while IFS=',' read -r idx mem util; do
        # Strip whitespace
        idx="${idx// /}"
        mem="${mem// /}"
        util="${util// /}"
        [[ "${idx}" =~ ^[0-9]+$ ]] || continue
        gpu_mem["${idx}"]="${mem}"
        gpu_util["${idx}"]="${util}"
    done <<< "${nvidia_smi_output}"

    local num_groups=$(( total / n ))
    for (( g = 0; g < num_groups; g++ )); do
        local start=$(( g * n ))
        local all_free=true
        for (( i = start; i < start + n; i++ )); do
            local mem="${gpu_mem[${i}]:-99999}"
            local util="${gpu_util[${i}]:-100}"
            if (( mem >= GPU_FREE_MEM_MB )) || (( util > GPU_FREE_UTIL )); then
                all_free=false
                break
            fi
        done
        if ${all_free}; then
            # Build "start,start+1,...,start+n-1"
            local ids
            ids="$(seq -s ',' "${start}" "$(( start + n - 1 ))")"
            echo "${ids}"
            return 0
        fi
    done
    return 1
}

# ---------------------------------------------------------------------------
# Mock GPU data for --mock-gpus (dry-run testing)
# ---------------------------------------------------------------------------
build_mock_nvidia_smi() {
    # MOCK_GPUS format: "idx:mem_mb:util,idx:mem_mb:util,..."
    local output=""
    IFS=',' read -ra entries <<< "${MOCK_GPUS}"
    for entry in "${entries[@]}"; do
        IFS=':' read -r idx mem util <<< "${entry}"
        output+="${idx}, ${mem}, ${util}"$'\n'
    done
    echo "${output}"
}

# ---------------------------------------------------------------------------
# Poll all machines until a free GPU group is found
# ---------------------------------------------------------------------------
find_available_machine() {
    log "Searching for available machine (type=${MACHINE_TYPE}, gpus=${NUM_GPUS})..."

    while true; do
        # Handle mock mode
        if [[ -n "${MOCK_GPUS}" ]]; then
            local mock_output
            mock_output="$(build_mock_nvidia_smi)"
            log "dry-run: mock nvidia-smi output:"
            echo "${mock_output}" | while read -r line; do log "  ${line}"; done
            if TARGET_GPU_IDS="$(find_aligned_gpu_group "${mock_output}")"; then
                TARGET_MACHINE="mock-machine"
                log "dry-run: found aligned GPU group: ${TARGET_GPU_IDS} on ${TARGET_MACHINE}"
                return 0
            else
                log "dry-run: no free GPU group in mock data — exiting"
                exit 0
            fi
        fi

        # Real machine polling
        local machines
        read -ra machines <<< "${MACHINE_LISTS[${MACHINE_TYPE}]}"
        for machine in "${machines[@]}"; do
            log "Checking machine: ${machine}"

            if ! check_ssh_reachable "${machine}"; then
                log "  SSH unreachable, skipping"
                continue
            fi

            local smi_output
            if ! smi_output="$(ssh_cmd "${machine}" \
                nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
                --format=csv,noheader,nounits 2>/dev/null)"; then
                log "  nvidia-smi failed, skipping"
                continue
            fi

            log "  nvidia-smi output:"
            echo "${smi_output}" | while read -r line; do log "    ${line}"; done

            local gpu_ids
            if gpu_ids="$(find_aligned_gpu_group "${smi_output}")"; then
                TARGET_MACHINE="${machine}"
                TARGET_GPU_IDS="${gpu_ids}"
                log "Found available machine: ${TARGET_MACHINE}, GPU IDs: ${TARGET_GPU_IDS}"
                return 0
            else
                log "  No aligned free GPU group of size ${NUM_GPUS}"
            fi
        done

        log "No suitable machine found. Waiting ${GPU_POLL_INTERVAL}s before retrying..."
        sleep "${GPU_POLL_INTERVAL}"
    done
}

# ---------------------------------------------------------------------------
# Write helper scripts to the remote machine's /tmp
# ---------------------------------------------------------------------------
write_remote_scripts() {
    log "Writing remote helper scripts to ${TARGET_MACHINE}:${TMP_REMOTE_DIR}"
    ssh_cmd "${TARGET_MACHINE}" mkdir -p "${TMP_REMOTE_DIR}"

    # ------------------------------------------------------------------
    # start_server.sh — sourced inside the container
    # ------------------------------------------------------------------
    local start_server_script
    start_server_script="$(cat <<REMOTE_SCRIPT
#!/usr/bin/env bash
set -euo pipefail

# Source environment
if [[ -f "${ENV_SETUP_SCRIPT}" ]]; then
    source "${ENV_SETUP_SCRIPT}"
fi

export CUDA_VISIBLE_DEVICES="${TARGET_GPU_IDS}"

WORK_DIR="${TMP_REMOTE_DIR}"
mkdir -p "\${WORK_DIR}"

# Ensure network-share output directory exists (already created by setup_output_dir,
# but the remote may need to create it if not yet mounted).
mkdir -p "${REPORT_DIR}"

# Launch server under nsys (paused at start so we can start/stop precisely)
nsys profile ${NSYS_OPTS} \
    --pause-on-start=true \
    -o "${REPORT_DIR}/${NSYS_OUTPUT_NAME}" \
    bash -c "${SERVER_CMD}" \
    > "${REPORT_DIR}/server.log" 2>&1 &

echo \$! > "\${WORK_DIR}/nsys.pid"
wait  # Keep the script alive until the server exits (or cleanup kills it)
REMOTE_SCRIPT
)"

    # Write via SSH here-doc
    ssh_cmd "${TARGET_MACHINE}" \
        "cat > '${TMP_REMOTE_DIR}/start_server.sh'" <<< "${start_server_script}"
    ssh_cmd "${TARGET_MACHINE}" chmod +x "${TMP_REMOTE_DIR}/start_server.sh"

    # ------------------------------------------------------------------
    # run_client.sh — sourced inside the container
    # ------------------------------------------------------------------
    local run_client_script
    run_client_script="$(cat <<REMOTE_SCRIPT
#!/usr/bin/env bash
set -euo pipefail

if [[ -f "${ENV_SETUP_SCRIPT}" ]]; then
    source "${ENV_SETUP_SCRIPT}"
fi

mkdir -p "${REPORT_DIR}"
${CLIENT_CMD} > "${REPORT_DIR}/client.log" 2>&1
REMOTE_SCRIPT
)"

    ssh_cmd "${TARGET_MACHINE}" \
        "cat > '${TMP_REMOTE_DIR}/run_client.sh'" <<< "${run_client_script}"
    ssh_cmd "${TARGET_MACHINE}" chmod +x "${TMP_REMOTE_DIR}/run_client.sh"

    log "Remote scripts written OK"
}

# ---------------------------------------------------------------------------
# Write run metadata
# ---------------------------------------------------------------------------
write_run_info() {
    local info_file="${REPORT_DIR}/run_info.txt"
    cat > "${info_file}" <<INFO
Run info
========
Timestamp    : ${RUN_TIMESTAMP}
Machine      : ${TARGET_MACHINE}
Machine type : ${MACHINE_TYPE}
GPU IDs      : ${TARGET_GPU_IDS}
NUM_GPUS     : ${NUM_GPUS}
Container    : ${DOCKER_CONTAINER_NAME}
Server CMD   : ${SERVER_CMD}
Client CMD   : ${CLIENT_CMD}
nsys opts    : ${NSYS_OPTS}
Report dir   : ${REPORT_DIR}
INFO
    log "Run info written to ${info_file}"
}

# ---------------------------------------------------------------------------
# Start the server (background SSH → docker exec → start_server.sh)
# ---------------------------------------------------------------------------
start_server() {
    log "Starting server on ${TARGET_MACHINE} (container: ${DOCKER_CONTAINER_NAME})"

    ssh_cmd "${TARGET_MACHINE}" \
        docker exec "${DOCKER_CONTAINER_NAME}" \
        bash "${TMP_REMOTE_DIR}/start_server.sh" &
    REMOTE_BG_PID=$!
    log "Background SSH PID: ${REMOTE_BG_PID}"

    # Wait for nsys.pid to appear (nsys has forked and written its PID)
    log "Waiting for nsys PID file..."
    local elapsed=0
    local poll=2
    while true; do
        if ssh_cmd "${TARGET_MACHINE}" \
            test -f "${TMP_REMOTE_DIR}/nsys.pid" 2>/dev/null; then
            NSYS_PID="$(ssh_cmd "${TARGET_MACHINE}" cat "${TMP_REMOTE_DIR}/nsys.pid")"
            log "nsys PID: ${NSYS_PID}"
            break
        fi
        sleep "${poll}"
        elapsed=$(( elapsed + poll ))
        if (( elapsed >= SERVER_STARTUP_TIMEOUT )); then
            die "Timed out waiting for nsys PID file after ${SERVER_STARTUP_TIMEOUT}s"
        fi
    done
}

# ---------------------------------------------------------------------------
# Wait for server port to accept connections
# ---------------------------------------------------------------------------
wait_for_server_ready() {
    log "Waiting for server on port ${SERVER_PORT} (timeout ${SERVER_STARTUP_TIMEOUT}s)..."
    local elapsed=0
    local poll=5
    while true; do
        if ssh_cmd "${TARGET_MACHINE}" \
            "nc -z localhost ${SERVER_PORT}" 2>/dev/null; then
            log "Server is ready on port ${SERVER_PORT}"
            return 0
        fi
        sleep "${poll}"
        elapsed=$(( elapsed + poll ))
        if (( elapsed >= SERVER_STARTUP_TIMEOUT )); then
            die "Timed out waiting for server on port ${SERVER_PORT} after ${SERVER_STARTUP_TIMEOUT}s"
        fi
        log "  Still waiting... (${elapsed}s elapsed)"
    done
}

# ---------------------------------------------------------------------------
# Core profiling: nsys start → client → nsys stop
# ---------------------------------------------------------------------------
run_nsys_profiling() {
    log "Starting nsys collection (nsys start)"
    docker_exec "${TARGET_MACHINE}" "nsys start --pid ${NSYS_PID}"

    log "Running client command"
    docker_exec "${TARGET_MACHINE}" "bash '${TMP_REMOTE_DIR}/run_client.sh'"
    log "Client command completed"

    log "Stopping nsys collection (nsys stop)"
    docker_exec "${TARGET_MACHINE}" "nsys stop --pid ${NSYS_PID}"

    # Wait for the .nsys-rep file to appear in REPORT_DIR
    log "Waiting for nsys report file to be generated..."
    local elapsed=0
    local poll=5
    local timeout=120
    while true; do
        local rep_count
        rep_count="$(ssh_cmd "${TARGET_MACHINE}" \
            "ls '${REPORT_DIR}'/*.nsys-rep 2>/dev/null | wc -l" || echo "0")"
        if (( rep_count > 0 )); then
            log "nsys report file(s) found in ${REPORT_DIR}"
            break
        fi
        sleep "${poll}"
        elapsed=$(( elapsed + poll ))
        if (( elapsed >= timeout )); then
            die "Timed out waiting for .nsys-rep file after ${timeout}s"
        fi
        log "  Still waiting for report... (${elapsed}s)"
    done
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
main() {
    parse_args "$@"
    load_config

    # Register cleanup trap after config is loaded so SERVER_CMD etc. are available
    trap cleanup EXIT ERR INT TERM

    setup_output_dir
    log "=== nsys auto-profile run started at ${RUN_TIMESTAMP} ==="
    log "Machine type: ${MACHINE_TYPE}, GPUs requested: ${NUM_GPUS}"

    find_available_machine

    # Exit here for dry-run (mock-gpus already exits inside find_available_machine)
    if ${DRY_RUN}; then
        log "dry-run complete — target machine=${TARGET_MACHINE}, GPU IDs=${TARGET_GPU_IDS}"
        exit 0
    fi

    write_run_info
    write_remote_scripts
    start_server
    wait_for_server_ready
    run_nsys_profiling

    log "=== Profiling complete ==="
    log "Outputs:"
    ssh_cmd "${TARGET_MACHINE}" ls -lh "${REPORT_DIR}/" | while read -r line; do log "  ${line}"; done
}

main "$@"
