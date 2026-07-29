"""After run1 Caduceus train finishes: ZSV (mice) + train_monitor + TB sync."""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "runs" / "run1" / "direct"
SPLIT_ROOT = ROOT / "runs" / "run1"
DONE = OUT / "train_time.json"


def main() -> int:
    # Optional: wait until train writes train_time.json (max ~10 h).
    wait = "--wait" in sys.argv
    if wait:
        deadline = time.time() + 10 * 3600
        while time.time() < deadline:
            if DONE.is_file():
                break
            time.sleep(60)
        else:
            print("TIMEOUT waiting for", DONE, flush=True)
            return 1
    if not DONE.is_file():
        raise SystemExit(f"Train not finished (missing {DONE}). Pass --wait to poll.")

    zsv = [
        sys.executable,
        "-m",
        "src.pipeline.zsv_eval",
        "--model",
        "caduceus",
        "--outdir",
        str(OUT),
        "--split-root",
        str(SPLIT_ROOT),
    ]
    print("zsv:", " ".join(zsv), flush=True)
    rc = subprocess.call(zsv)
    if rc != 0:
        return rc
    mon = [
        sys.executable,
        "-m",
        "src.train_viz.train_monitor",
        "--run-dir",
        str(OUT),
    ]
    print("monitor:", " ".join(mon), flush=True)
    return subprocess.call(mon)


if __name__ == "__main__":
    raise SystemExit(main())
