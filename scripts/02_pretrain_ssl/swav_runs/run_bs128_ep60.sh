#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

# Chạy SwAV B=128 với 60 epochs (công bằng với TS-TCC)
run_swav "bs128_ep60" 128 60 uci_har motion_sense hhar_phone hhar_watch