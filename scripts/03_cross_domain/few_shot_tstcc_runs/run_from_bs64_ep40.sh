#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

# FULL 12 cặp cross-domain với checkpoint TS-TCC B=64 (40 epoch).
run_few_shot_tstcc \
    "tstcc_from_bs64_ep40" \
    "tstcc" \
    "standard" \
    "bs64_ep40" \
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