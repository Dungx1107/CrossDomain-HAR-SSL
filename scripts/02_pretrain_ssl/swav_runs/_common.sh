#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PY_SCRIPT="${PROJECT_ROOT}/scripts/02_pretrain_ssl/run_pretrain_swav.py"

run_swav() {
    local tag="$1"; shift
    local batch_size="$1"; shift
    local epochs="$1"; shift
    local datasets=("$@")
    echo "================================================================"
    echo "🚀 [${tag}] B=${batch_size} | Epochs=${epochs} | Datasets=${datasets[*]}"
    echo "   Output: checkpoints/ssl_pretrain/swav/<dataset>/standard_${tag}/"
    echo "================================================================"
    cd "${PROJECT_ROOT}"
    export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
    python "${PY_SCRIPT}" \
        --datasets "${datasets[@]}" \
        --backbone standard \
        --batch_size "${batch_size}" \
        --epochs "${epochs}" \
        --tag "${tag}" \
        --device cuda
    echo "✅ [${tag}] xong"
}