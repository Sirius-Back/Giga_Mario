#!/usr/bin/env bash
# Usage: notify_pipeline_done.sh <run_id> <to_email>
set -euo pipefail
RUN_ID=${1:?}
TO=${2:-dvsmutin@gmail.com}
ROOT=/home/User14
DONE=$ROOT/runs/$RUN_ID/pipeline_done.json
LOG=$ROOT/logs/${RUN_ID}_pipeline.log
SUBJ="[Cursor] $RUN_ID pipeline COMPLETED"
BODY="Run $RUN_ID finished at $(date -Is).
Done marker: $DONE
Log: $LOG
Host: $(hostname)
"
if command -v mail >/dev/null 2>&1; then
  printf '%s\n' "$BODY" | mail -s "$SUBJ" "$TO" && echo "mailed via mail" || echo "mail failed"
elif command -v sendmail >/dev/null 2>&1; then
  { echo "To: $TO"; echo "Subject: $SUBJ"; echo; echo "$BODY"; } | sendmail -t && echo "mailed via sendmail" || echo "sendmail failed"
else
  echo "NO_MTA: wrote $ROOT/logs/${RUN_ID}_email_pending.txt"
  printf '%s\n' "$SUBJ" "$BODY" > "$ROOT/logs/${RUN_ID}_email_pending.txt"
fi
