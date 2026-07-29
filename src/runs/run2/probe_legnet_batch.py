"""Probe largest LegNet train_batch_size that fits on 1× V100 for 230 bp demo."""
from __future__ import annotations

import gc
import sys

import torch


def try_batch(bs: int) -> bool:
    from human_legnet.model import LegNet  # type: ignore

    torch.cuda.empty_cache()
    gc.collect()
    device = torch.device("cuda:0")
    # Match demo-ish widths; OOM probe only.
    model = LegNet(seqsize=230, use_single_channel=True).to(device)
    x = torch.randn(bs, 5, 230, device=device)  # one-hot-ish channels
    try:
        with torch.cuda.amp.autocast():
            y = model(x)
            loss = y.float().mean()
        loss.backward()
        torch.cuda.synchronize()
        return True
    except torch.cuda.OutOfMemoryError:
        return False
    finally:
        del model
        torch.cuda.empty_cache()
        gc.collect()


def main() -> int:
    if not torch.cuda.is_available():
        print("NO_CUDA")
        return 2
    candidates = [16384, 12288, 8192, 6144, 4096, 3072, 2048, 1024]
    best = 1024
    for bs in candidates:
        ok = try_batch(bs)
        print(f"batch={bs} ok={ok}", flush=True)
        if ok:
            best = bs
            break
    print(f"BEST_BATCH={best}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
