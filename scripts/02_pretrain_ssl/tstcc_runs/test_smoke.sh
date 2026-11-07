#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

echo "🧪 SMOKE TEST (TS-TCC) — 1 epoch trên UCI-HAR"
#run_tstcc "smoke" 64 1 uci_har
run_tstcc "64" 64 1 uci_har