# Job queue

Living log of **large** local CPU / RAM / GPU jobs.
Agents append entries on launch and mark status on completion.
Do not delete history.

Policy: `.cursor/rules/local-job-queue.mdc` — host RAM **used ≤ 95% of
MemTotal** (MemAvailable ≥ 5%); wait on high load using PIDs / estimated times
below. The 95% cap is **total RAM only** (not CPU, not RAM/CPU ratio).

| status | meaning |
|--------|---------|
| `RUNNING` | Process launched; PID should still exist |
| `COMPLETED` | Finished successfully |
| `FAILED` | Exited non-zero or killed |
| `WAITED-OUT` | Waited past ETA; confirm PID before assuming done |

---

<!-- Append new jobs below this line -->

### resume_caduceus_8_12_14 — FAILED
- **launch time:** 2026-07-30T17:37:30+03:00
- **job:** `conda run -n caduceus_env python -m src.runs.resume_caduceus_8_12_14` (batch 256, skip completed)
- **PID:** 3204734
- **estimated time:** 6h
- **resources:** GPU Caduceus train; share host with other agents (do not kill run5 etc.)
- **log:** /home/User14/logs/resume_caduceus_8_12_14.log
- **status:** FAILED (run8 skip_adv_setup: adversarial SPLIT/split.csv missing; run14 child may still be RUNNING)

### run13_7mer_legnet_split — RUNNING
- **launch time:** 2026-07-30T17:40:46+03:00
- **job:** `conda run -n legnet python -m src.runs.run13_7mer_legnet.pipeline_ready_legnet` (7-mer two-pass + RAM≤95% + progress logs)
- **PID:** 3208874 (conda parent 3208849)
- **estimated time:** 8–16h (two-pass k=7 full panel ~457k)
- **resources:** CPU-heavy split; peak RAM dominated by float32 feature matrix (~n×d); host used ≤95% MemTotal
- **log:** logs/run13_7mer_legnet_pipeline.log
- **status:** RUNNING


### run8_resume_adv — RUNNING
- **launch time:** 2026-07-30T17:41:21+03:00
- **job:** `python -m src.runs.run8_2mer_caduceus.continue_from_split skip_direct=true skip_adv_setup=true batch_size=256`
- **PID:** 3209622
- **estimated time:** 3h
- **resources:** GPU 0,1; batch 256; Caduceus adv classification
- **log:** logs/run8_resume_adv.log
- **status:** RUNNING

### run12_after_run8_waiter — RUNNING
- **launch time:** 2026-07-30T17:41:21+03:00
- **job:** wait for run8 pipeline_done.json + GPUs free, then run12 continue_from_split batch_size=256
- **PID:** 3209623
- **estimated time:** after run8 + ~3h
- **resources:** GPU 0,1 after run8; batch 256
- **log:** logs/run12_after_run8.log
- **status:** RUNNING

### run8_resume_adv — RUNNING
- **launch time:** 2026-07-30T17:44:17+03:00
- **job:** `python -m src.runs.run8_2mer_caduceus.continue_from_split skip_direct=true skip_adv_setup=true batch_size=256` (reuse FASTA/TRAIN; rebuild incomplete caduceus_input)
- **PID:** 3212552
- **estimated time:** 3h
- **resources:** GPU 0,1; batch 256
- **log:** logs/run8_resume_adv.log
- **status:** RUNNING

### run8_resume_adv — RUNNING
- **launch time:** 2026-07-30T17:45:50+03:00
- **job:** `python -m src.runs.run8_2mer_caduceus.continue_from_split skip_direct=true skip_adv_setup=true batch_size=256`
- **PID:** 3212552 (python 3212574)
- **estimated time:** 3h
- **resources:** GPU 0,1; batch 256
- **log:** logs/run8_resume_adv.log
- **status:** RUNNING

### run12_after_run8_waiter — RUNNING
- **launch time:** 2026-07-30T17:45:50+03:00
- **job:** wait run8 pipeline_done then run12 continue_from_split batch_size=256
- **PID:** 3209623
- **estimated time:** after run8 + ~3h
- **log:** logs/run12_after_run8.log
- **status:** RUNNING
