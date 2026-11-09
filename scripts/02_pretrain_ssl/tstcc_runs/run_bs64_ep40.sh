#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

# Chạy TS-TCC B=64 với 40 epochs (để so sánh với bản 60 epoch)
run_tstcc "bs64_ep40" 64 40 uci_har motion_sense hhar_phone hhar_watch