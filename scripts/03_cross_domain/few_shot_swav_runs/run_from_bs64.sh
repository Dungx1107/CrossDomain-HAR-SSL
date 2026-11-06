#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

# FULL 12 cặp cross-domain với checkpoint B=64.
# k_shots: 10 30 50 80 100 120 150 | seeds: 5 | freeze_modes: full none
run_few_shot \
    "from_bs64" \
    "swav" \
    "standard" \
    "bs64" \
    40 \
    16 \
    "uci_har:motion_sense" \
    "uci_har:hhar_phone" \
    "uci_har:hhar_watch" \
    "motion_sense:uci_har" \
    "motion_sense:hhar_phone" \
    "motion_sense:hhar_watch" \
    "hhar_phone:uci_har" \
    "hhar_phone:motion_sense" \
    "hhar_phone:hhar_watch" \
    "hhar_watch:uci_har" \
    "hhar_watch:motion_sense" \
    "hhar_watch:hhar_phone"