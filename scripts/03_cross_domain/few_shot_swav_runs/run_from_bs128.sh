#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

# FULL 12 cặp cross-domain với checkpoint B=128.
run_few_shot \
    "from_bs128" \
    "swav" \
    "standard" \
    "bs128" \
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