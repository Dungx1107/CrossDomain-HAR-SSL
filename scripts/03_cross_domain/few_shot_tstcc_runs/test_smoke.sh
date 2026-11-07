#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

echo "🧪 FEW-SHOT SMOKE TEST (TS-TCC) — checkpoint 'smoke', 1 pair, 1 k_shot, 2 epochs"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

python "${PY_SCRIPT}" \
    --pairs "uci_har:hhar_phone" \
    --method tstcc \
    --backbone standard \
    --pretrain_tag smoke \
    --k_shots 10 \
    --epochs 2 \
    --seeds 42 \
    --batch_size 16 \
    --freeze_modes full

echo "✅ Smoke test few-shot (TS-TCC) xong. Kiểm tra:"
echo "   outputs_evaluation/cross_k_shot/tstcc/standard_smoke/uci_har_to_hhar_phone/10_shot/"