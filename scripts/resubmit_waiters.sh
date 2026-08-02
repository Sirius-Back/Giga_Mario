#!/usr/bin/env bash
set -euo pipefail
cd /home/User14
LOG=logs/resubmit_waiters.log
echo "SKIP_ALL_ADV $(date -Iseconds)" | tee -a "$LOG"
exit 0
