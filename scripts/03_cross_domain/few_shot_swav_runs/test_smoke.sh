#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

# Test pipeline few-shot với checkpoint smoke (đã chạy 1 epoch lúc pretrain).
# Chỉ 1 pair, k_shots nhỏ, epochs 2 — chạy ~1 phút để verify code, KHÔNG phải kết quả thật.
echo "🧪 FEW-SHOT SMOKE TEST — checkpoint 'smoke', 1 pair, 1 k_shot, 2 epochs"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

python "${PY_SCRIPT}" \
    --pairs "uci_har:hhar_phone" \
    --method swav \
    --backbone standard \
    --pretrain_tag smoke \
    --k_shots 10 \
    --epochs 2 \
    --seeds 42 \
    --batch_size 16 \
    --freeze_modes full

echo "✅ Smoke test few-shot xong. Kiểm tra:"
echo "   outputs_evaluation/cross_k_shot/swav/standard/uci_har_to_hhar_phone/10_shot/"