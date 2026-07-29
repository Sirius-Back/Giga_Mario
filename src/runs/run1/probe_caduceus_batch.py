"""Probe largest Caduceus per-GPU batch that fits for ~200 bp windows (1× V100)."""
from __future__ import annotations

import gc
import os
import sys

import torch


def try_batch(bs: int, max_length: int = 256) -> bool:
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
    # Char-level-ish token ids; Caduceus uses DNA tokenizer — random ids OK for mem probe.
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
    # Prefer free GPU via CUDA_VISIBLE_DEVICES set by caller.
    print(f"device0={torch.cuda.get_device_name(0)} vis={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    candidates = [256, 192, 128, 96, 64, 48, 32, 16, 8, 4]
    best = 4
    for bs in candidates:
        ok = try_batch(bs)
        print(f"batch={bs} ok={ok}", flush=True)
        if ok:
            best = bs
            break
    # Leave headroom for DDP / optimizer / eval.
    safe = max(4, int(best * 0.75) // 4 * 4) if best >= 8 else best
    print(f"BEST_BATCH={best}", flush=True)
    print(f"SAFE_BATCH={safe}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
