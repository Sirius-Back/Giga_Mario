# Job queue

Living log of **large** local CPU / RAM / GPU jobs.
Agents append entries on launch and mark status on completion.
Do not delete history.

Policy: `.cursor/rules/local-job-queue.mdc` — keep **≥5% RAM free**; wait on
high load using PIDs / estimated times below.

| status | meaning |
|--------|---------|
| `RUNNING` | Process launched; PID should still exist |
| `COMPLETED` | Finished successfully |
| `FAILED` | Exited non-zero or killed |
| `WAITED-OUT` | Waited past ETA; confirm PID before assuming done |

---

<!-- Append new jobs below this line -->
