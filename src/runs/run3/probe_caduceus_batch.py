"""Probe largest Caduceus per-GPU batch that fits for ~200 bp windows (1× V100)."""
from __future__ import annotations

import gc
import os
import sys

import torch


def try_batch(bs: int, max_length: int = 208) -> bool:
    from transformers import AutoModelForSequenceClassification

    from src.caduceus import DEFAULT_MODEL

    torch.cuda.empty_cache()
    gc.collect()
    device = torch.device("cuda:0")
    model = AutoModelForSequenceClassification.from_pretrained(
        DEFAULT_MODEL,
        num_labels=1,
        trust_remote_code=True,
    ).to(device)
    input_ids = torch.randint(7, 12, (bs, max_length), device=device)
    attention_mask = torch.ones_like(input_ids)
    labels = torch.randn(bs, device=device)
    try:
        with torch.cuda.amp.autocast():
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = out.loss
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
    max_length = 208
    for tok in sys.argv[1:]:
        if tok.startswith("max_length="):
            max_length = int(tok.split("=", 1)[1])
    print(
        f"device0={torch.cuda.get_device_name(0)} "
        f"vis={os.environ.get('CUDA_VISIBLE_DEVICES')} max_length={max_length}"
    )
    # Ascending then take largest that fits (binary-ish: try high first).
    candidates = [1024, 768, 640, 512, 448, 384, 320, 256, 192, 128, 96, 64, 48, 32, 16, 8, 4]
    best = 4
    for bs in candidates:
        ok = try_batch(bs, max_length=max_length)
        print(f"batch={bs} ok={ok}", flush=True)
        if ok:
            best = bs
            break
    # Leave headroom for DDP / optimizer states / eval.
    safe = max(4, int(best * 0.75) // 8 * 8) if best >= 8 else best
    print(f"BEST_BATCH={best}", flush=True)
    print(f"SAFE_BATCH={safe}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
