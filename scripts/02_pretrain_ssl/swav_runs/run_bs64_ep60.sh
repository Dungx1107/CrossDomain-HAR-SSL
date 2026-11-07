#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

# Chạy SwAV B=64 với 60 epochs (công bằng với TS-TCC)
run_swav "bs64_ep60" 64 60 uci_har motion_sense hhar_phone hhar_watch