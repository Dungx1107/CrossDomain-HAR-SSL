#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

echo "🧪 SMOKE TEST — 1 epoch trên UCI-HAR"
run_swav "smoke" 64 1 uci_har