#!/usr/bin/env python3
"""Fine-tune Caduceus for Split-1 fold membership classification on OUT_ADAPT2.

Reads caduceus_ready labels.tsv where the TPM column stores class ids 0/1/2
(split1 train/val/test). Multi-GPU via torchrun. Logs loss/accuracy each epoch.
Saves final_model/ and train_time.json. Epoch dirs are 1-indexed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Shared metrics.md helpers from @caduceus skill
_SKILL_SCRIPTS = Path(__file__).resolve().parents[1] / ".cursor" / "skills" / "caduceus" / "scripts"
if _SKILL_SCRIPTS.is_dir():
    sys.path.insert(0, str(_SKILL_SCRIPTS))
# Classification metrics (accuracy); no metrics.md regression suite for this head.


class TpmWindowDataset(Dataset):
    """caduceus_ready/{split}/labels.tsv + sequences/<id>.txt → (tokens, TPM)."""

    def __init__(self, data_root: Path, split: str, tokenizer, max_length: int):
        self.root = data_root.resolve()
        self.split = split
        labels_path = self.root / split / "labels.tsv"
        if not labels_path.is_file():
            raise FileNotFoundError(f"Missing labels: {labels_path}")
        self.samples: list[tuple[Path, float]] = []
        with labels_path.open(encoding="utf-8") as fh:
            header = fh.readline().rstrip("\n").split("\t")
            col = {name: i for i, name in enumerate(header)}
            for req in ("sample_id", "TPM"):
                if req not in col:
                    raise ValueError(f"{labels_path} missing column {req}: {header}")
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < len(header):
                    continue
                sid = parts[col["sample_id"]]
                tpm = float(parts[col["TPM"]])
                if "path" in col and parts[col["path"]]:
                    rel = parts[col["path"]]
                    seq_path = self.root / rel
                    if not seq_path.is_file():
                        # path may be relative to split dir
                        seq_path = self.root / split / Path(rel).name
                        if not seq_path.is_file():
                            seq_path = self.root / split / "sequences" / f"{sid}.txt"
                else:
                    seq_path = self.root / split / "sequences" / f"{sid}.txt"
                if not seq_path.is_file():
                    raise FileNotFoundError(f"Sequence missing for {sid}: {seq_path}")
                # TPM column holds class id (0/1/2) for predict-split1
                self.samples.append((seq_path, int(round(tpm))))
        if not self.samples:
            raise SystemExit(f"Empty dataset for split={split} under {labels_path}")
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, y = self.samples[idx]
        seq = path.read_text().strip().upper()
        enc = self.tokenizer(
            seq,
            padding=False,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(y, dtype=torch.long)
        return item


def collate_pad(batch: list[dict], pad_id: int) -> dict[str, torch.Tensor]:
    """Dynamic pad to max length in batch (left pad to match CaduceusTokenizer)."""
    max_len = max(x["input_ids"].numel() for x in batch)
    input_ids = []
    attention_mask = []
    labels = []
    for x in batch:
        ids = x["input_ids"]
        mask = x.get("attention_mask", torch.ones_like(ids))
        pad_len = max_len - ids.numel()
        if pad_len > 0:
            # CaduceusTokenizer default padding_side=left
            ids = F.pad(ids, (pad_len, 0), value=pad_id)
            mask = F.pad(mask, (pad_len, 0), value=0)
        input_ids.append(ids)
        attention_mask.append(mask)
        labels.append(x["labels"])
    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "labels": torch.stack(labels),
    }


def setup_distributed() -> tuple[int, int, int]:
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
        local = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local)
        return rank, world, local
    return 0, 1, 0


def cleanup() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


@torch.no_grad()
def evaluate_split(model, loader, device, criterion) -> dict[str, float]:
    """Full-split metrics.md suite (once per epoch; no batch-averaged correlations)."""
    model.eval()
    preds: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    total_loss = 0.0
    n = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        logits = out.logits.squeeze(-1)
        labels = batch["labels"]
        loss = criterion(logits, labels)
        bs = labels.size(0)
        total_loss += float(loss.item()) * bs
        n += bs
        preds.append(logits.detach().float().cpu())
        targets.append(labels.detach().float().cpu())
    if n == 0:
        return {
            "loss": float("nan"),
            "pearson": float("nan"),
            "spearman": float("nan"),
            "mse": float("nan"),
            "rmse": float("nan"),
            "mae": float("nan"),
            "r2": float("nan"),
            "genewise_pearson_median": float("nan"),
            "samplewise_pearson_median": float("nan"),
            "n": 0,
        }
    pred = torch.cat(preds)
    target = torch.cat(targets)
    metrics = compute_epoch_regression_metrics(pred, target, loss=total_loss / n)
    metrics["n"] = int(n)
    return metrics


def zs_slot_payload(zs_root: Path | None) -> dict | None:
    """Return None if zero-shot adapt not ready — do not invent metrics."""
    if zs_root is None:
        return None
    ready = zs_root / "caduceus_ready"
    # accept either fold layout or all/
    if not ready.is_dir():
        return None
    has_labels = any((ready / s / "labels.tsv").is_file() for s in ("all", "test", "val", "train"))
    if not has_labels:
        return None
    return {"status": "ready", "path": str(zs_root)}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Path to adapt/.../split1/caduceus_ready (train/val/test)",
    )
    p.add_argument("--runs-dir", type=Path, required=True)
    p.add_argument(
        "--model-name",
        type=str,
        default="kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16",
    )
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--eval-batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max-length", type=int, default=8192)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument(
        "--zero-shot-root",
        type=Path,
        default=None,
        help="OUT_ADAPT_ZS root; if ready, note in metrics (eval may be deferred to T-9)",
    )
    p.add_argument(
        "--eval-zero-shot",
        action="store_true",
        help="If set and ZS ready, evaluate zero-shot-validation each epoch",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rank, world, local = setup_distributed()
    device = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + rank)

    data_root = args.data_root.resolve()
    runs_dir = args.runs_dir.resolve()
    if rank == 0:
        runs_dir.mkdir(parents=True, exist_ok=True)
        for split in ("train", "val", "test"):
            labels = data_root / split / "labels.tsv"
            seqs = data_root / split / "sequences"
            if not labels.is_file() or not seqs.is_dir():
                raise SystemExit(f"Invalid caduceus_ready layout: need {labels} and {seqs}")

    if dist.is_initialized():
        dist.barrier()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 4

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        num_labels=1,
        problem_type="regression",
        ignore_mismatched_sizes=True,
    )
    model.to(device)
    if world > 1:
        model = DDP(model, device_ids=[local], output_device=local)

    def make_loader(split: str, batch_size: int, shuffle: bool, distributed: bool):
        ds = TpmWindowDataset(data_root, split, tokenizer, args.max_length)
        sampler = (
            DistributedSampler(ds, shuffle=shuffle, seed=args.seed)
            if (world > 1 and distributed)
            else None
        )
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(sampler is None and shuffle),
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=lambda b: collate_pad(b, pad_id),
        )
        return loader, ds, sampler

    train_loader, train_ds, train_sampler = make_loader(
        "train", args.batch_size, True, distributed=True
    )
    # Full-set eval on rank 0 only
    val_loader = test_loader = train_eval_loader = None
    val_ds = test_ds = None
    if rank == 0:
        train_eval_loader, _, _ = make_loader("train", args.eval_batch_size, False, False)
        val_loader, val_ds, _ = make_loader("val", args.eval_batch_size, False, False)
        test_loader, test_ds, _ = make_loader("test", args.eval_batch_size, False, False)

    zs_info = zs_slot_payload(args.zero_shot_root)
    zs_loader = None
    zs_split_used = None
    # Auto-eval ZS when adapt root is ready (task: log zero-shot-validation if ready)
    do_zs = bool(args.eval_zero_shot or (zs_info is not None and zs_info.get("status") == "ready"))
    if rank == 0 and do_zs and args.zero_shot_root:
        zs_ready = Path(args.zero_shot_root).resolve() / "caduceus_ready"
        for cand in ("all", "test", "val", "train"):
            if (zs_ready / cand / "labels.tsv").is_file():
                zs_ds = TpmWindowDataset(zs_ready, cand, tokenizer, args.max_length)
                zs_loader = DataLoader(
                    zs_ds,
                    batch_size=args.eval_batch_size,
                    shuffle=False,
                    num_workers=args.num_workers,
                    pin_memory=torch.cuda.is_available(),
                    collate_fn=lambda b: collate_pad(b, pad_id),
                )
                zs_split_used = cand
                break

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss()

    if rank == 0:
        meta = {
            "model_name": args.model_name,
            "task": "continuous_TPM_regression",
            "problem_type": "regression",
            "n_train": len(train_ds),
            "n_val": len(val_ds) if val_ds is not None else 0,
            "n_test": len(test_ds) if test_ds is not None else 0,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "world_size": world,
            "max_length": args.max_length,
            "lr": args.lr,
            "seed": args.seed,
            "data_root": str(data_root),
            "optimizer": "AdamW",
            "loss": "MSELoss",
            "target": "raw_TPM",
            "padding": "dynamic_left_pad",
            "zero_shot_adapt": zs_info,
            "zero_shot_eval_split": zs_split_used,
            "metrics": "metrics.md via metrics_logging.py",
        }
        (runs_dir / "run_config.json").write_text(json.dumps(meta, indent=2))
        # train-viz friendly: config as one JSON line then epoch lines
        with (runs_dir / "train_metrics.jsonl").open("w", encoding="utf-8") as fh:
            fh.write(json.dumps(meta) + "\n")
        print(json.dumps(meta, indent=2), flush=True)

    t0 = time.perf_counter()
    global_step = 0
    raw_model = model.module if isinstance(model, DDP) else model

    for epoch_idx in range(args.epochs):
        epoch = epoch_idx + 1  # 1-indexed deliverable dirs
        if train_sampler is not None:
            train_sampler.set_epoch(epoch_idx)
        model.train()
        running_loss = 0.0
        seen = 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            bs = batch["labels"].size(0)
            running_loss += float(loss.item()) * bs
            seen += bs
            global_step += 1

        # Rough train optimization loss (distributed mean); metrics.md suite below on full split
        opt_loss = running_loss / max(seen, 1)
        if world > 1:
            t = torch.tensor([opt_loss, float(seen)], device=device, dtype=torch.float64)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            opt_loss = (t[0] / t[1]).item()

        train_metrics = val_metrics = test_metrics = None
        zs_metrics = None
        if rank == 0:
            train_metrics = evaluate_split(raw_model, train_eval_loader, device, criterion)
            val_metrics = evaluate_split(raw_model, val_loader, device, criterion)
            test_metrics = evaluate_split(raw_model, test_loader, device, criterion)
            if zs_loader is not None:
                zs_metrics = evaluate_split(raw_model, zs_loader, device, criterion)

        if dist.is_initialized():
            dist.barrier()

        elapsed = time.perf_counter() - t0
        if rank == 0:
            assert train_metrics is not None and val_metrics is not None and test_metrics is not None
            payload: dict = {
                "epoch": epoch,
                "train": train_metrics,
                "validation": val_metrics,
                "test": test_metrics,
                "train_optim_loss": opt_loss,
                "elapsed_sec": elapsed,
                "global_step": global_step,
                "lr": args.lr,
            }
            # zero-shot-validation slot: real metrics if evaluated; else explicit deferred note
            if zs_metrics is not None:
                payload["zero-shot-validation"] = zs_metrics
            else:
                payload["zero-shot-validation"] = {
                    "status": "deferred",
                    "reason": "OUT_ADAPT_ZS not ready or --eval-zero-shot not set; reserved for T-9",
                    "path": str(args.zero_shot_root) if args.zero_shot_root else None,
                }

            epoch_dir = runs_dir / f"epoch{epoch}"
            epoch_dir.mkdir(parents=True, exist_ok=True)
            (epoch_dir / "metrics.json").write_text(json.dumps(payload, indent=2))
            log_lines = [
                format_epoch_log("train", train_metrics, epoch=epoch),
                format_epoch_log("validation", val_metrics, epoch=epoch),
                format_epoch_log("test", test_metrics, epoch=epoch),
            ]
            if zs_metrics is not None:
                log_lines.append(format_epoch_log("zero-shot-validation", zs_metrics, epoch=epoch))
            else:
                log_lines.append(
                    f"epoch={epoch} zero-shot-validation=deferred (slot reserved for T-9)"
                )
            (epoch_dir / "metrics.log").write_text("\n".join(log_lines) + "\n")
            with (runs_dir / "train_metrics.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload) + "\n")
            print(json.dumps(payload), flush=True)

        if dist.is_initialized():
            dist.barrier()

    total_time = time.perf_counter() - t0
    if rank == 0:
        final_dir = runs_dir / "final_model"
        final_dir.mkdir(parents=True, exist_ok=True)
        raw_model.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)
        (runs_dir / "train_time.json").write_text(
            json.dumps(
                {
                    "train_time_sec": total_time,
                    "train_time_min": total_time / 60.0,
                    "epochs_completed": args.epochs,
                    "global_step": global_step,
                    "world_size": world,
                    "seed": args.seed,
                },
                indent=2,
            )
        )
        print("Saved model to", final_dir, flush=True)
        print("Train time sec:", total_time, flush=True)

    cleanup()


if __name__ == "__main__":
    main()
