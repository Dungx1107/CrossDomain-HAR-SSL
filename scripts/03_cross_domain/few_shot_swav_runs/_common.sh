#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PY_SCRIPT="${PROJECT_ROOT}/scripts/03_cross_domain/run_few_shot.py"

run_few_shot() {
    local tag="$1"; shift
    local method="$1"; shift
    local backbone="$1"; shift
    local pretrain_tag="$1"; shift
    local epochs="$1"; shift
    local batch_size="$1"; shift
    local pairs=("$@")

    echo "================================================================"
    echo "🚀 [${tag}] method=${method} | backbone=${backbone} | pretrain_tag=${pretrain_tag}"
    echo "   Pairs: ${pairs[*]}"
    echo "   Checkpoint: checkpoints/ssl_pretrain/${method}/<src>/${backbone}_${pretrain_tag}/"
    echo "   Output:     outputs_evaluation/cross_k_shot/${method}/${backbone}/<src>_to_<tgt>/"
    echo "================================================================"

    cd "${PROJECT_ROOT}"
    export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

    python "${PY_SCRIPT}" \
        --pairs "${pairs[@]}" \
        --method "${method}" \
        --backbone "${backbone}" \
        --pretrain_tag "${pretrain_tag}" \
        --k_shots 10 30 50 80 100 120 150 \
        --epochs "${epochs}" \
        --seeds 42 100 2024 7 99 \
        --batch_size "${batch_size}" \
        --freeze_modes full none

    echo "✅ [${tag}] xong"
}