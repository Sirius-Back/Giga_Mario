#!/usr/bin/env python3
"""Fine-tune Caduceus on GenomicBenchmark-style trees under data/caduceus_gb/...

Multi-GPU via torchrun. Logs each epoch under runs/random/epoch{N}/.
Saves final model and train_time.json.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class GbTxtDataset(Dataset):
    def __init__(self, root: Path, split: str, label_to_id: dict[str, int], tokenizer, max_length: int):
        self.samples: list[tuple[Path, int]] = []
        split_dir = root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(split_dir)
        for label_dir in sorted(split_dir.iterdir()):
            if not label_dir.is_dir():
                continue
            if label_dir.name not in label_to_id:
                continue
            lid = label_to_id[label_dir.name]
            for p in sorted(label_dir.glob("*.txt")):
                self.samples.append((p, lid))
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        seq = path.read_text().strip().upper()
        enc = self.tokenizer(
            seq,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)
        return item


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


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    return (preds == labels).float().mean().item()


@torch.no_grad()
def evaluate(model, loader, device) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        bs = batch["labels"].size(0)
        total_loss += out.loss.item() * bs
        total_acc += accuracy(out.logits, batch["labels"]) * bs
        n += bs
    if n == 0:
        return {"loss": float("nan"), "accuracy": float("nan"), "n": 0}
    return {"loss": total_loss / n, "accuracy": total_acc / n, "n": n}


def gather_metric(value: float, device, world: int) -> float:
    if world == 1:
        return value
    t = torch.tensor([value], device=device, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return (t / world).item()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=Path("data/caduceus_gb/random_full"))
    p.add_argument("--runs-dir", type=Path, default=Path("runs/random"))
    p.add_argument("--model-name", type=str, default="kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--smoke-steps", type=int, default=0, help="If >0, run only this many train steps then exit")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rank, world, local = setup_distributed()
    device = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed + rank)

    data_root = args.data_root.resolve()
    runs_dir = args.runs_dir.resolve()
    if rank == 0:
        runs_dir.mkdir(parents=True, exist_ok=True)

    # labels from train fold
    train_dir = data_root / "train"
    labels = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    if not labels:
        raise SystemExit(f"No label dirs under {train_dir}")
    label_to_id = {n: i for i, n in enumerate(labels)}
    id_to_label = {i: n for n, i in label_to_id.items()}

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        num_labels=len(labels),
        ignore_mismatched_sizes=True,
    )
    model.to(device)
    if world > 1:
        model = DDP(model, device_ids=[local], output_device=local)

    def make_loader(split: str, shuffle: bool, distributed: bool):
        ds = GbTxtDataset(data_root, split, label_to_id, tokenizer, args.max_length)
        if len(ds) == 0:
            raise SystemExit(f"Empty dataset for split={split}")
        sampler = DistributedSampler(ds, shuffle=shuffle, seed=args.seed) if (world > 1 and distributed) else None
        return DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=(sampler is None and shuffle),
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True,
        ), ds, sampler

    train_loader, train_ds, train_sampler = make_loader("train", True, distributed=True)
    # Full-set eval on rank 0 only (avoid sharded, under-counted metrics)
    val_loader, val_ds, _ = make_loader("val", False, distributed=False)
    test_loader, test_ds, _ = make_loader("test", False, distributed=False)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    if rank == 0:
        meta = {
            "model_name": args.model_name,
            "labels": id_to_label,
            "n_train": len(train_ds),
            "n_val": len(val_ds),
            "n_test": len(test_ds),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "world_size": world,
            "max_length": args.max_length,
            "seed": args.seed,
            "data_root": str(data_root),
        }
        (runs_dir / "run_config.json").write_text(json.dumps(meta, indent=2))
        print(json.dumps(meta, indent=2))

    t0 = time.perf_counter()
    global_step = 0
    raw_model = model.module if isinstance(model, DDP) else model

    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        seen = 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            bs = batch["labels"].size(0)
            running_loss += loss.item() * bs
            running_acc += accuracy(out.logits.detach(), batch["labels"]) * bs
            seen += bs
            global_step += 1
            if args.smoke_steps and global_step >= args.smoke_steps:
                break

        train_metrics = {
            "loss": running_loss / max(seen, 1),
            "accuracy": running_acc / max(seen, 1),
            "n": seen,
        }
        # sync rough averages across ranks
        train_metrics["loss"] = gather_metric(train_metrics["loss"], device, world)
        train_metrics["accuracy"] = gather_metric(train_metrics["accuracy"], device, world)

        val_metrics = {"loss": float("nan"), "accuracy": float("nan"), "n": 0}
        test_metrics = {"loss": float("nan"), "accuracy": float("nan"), "n": 0}
        if rank == 0:
            val_metrics = evaluate(raw_model, val_loader, device)
            test_metrics = evaluate(raw_model, test_loader, device)

        elapsed = time.perf_counter() - t0
        payload = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": val_metrics,
            "test": test_metrics,
            "elapsed_sec": elapsed,
            "global_step": global_step,
        }

        if rank == 0:
            epoch_dir = runs_dir / f"epoch{epoch}"
            epoch_dir.mkdir(parents=True, exist_ok=True)
            (epoch_dir / "metrics.json").write_text(json.dumps(payload, indent=2))
            with (epoch_dir / "metrics.log").open("w") as fh:
                fh.write(
                    f"epoch={epoch} "
                    f"train_loss={train_metrics['loss']:.6f} train_acc={train_metrics['accuracy']:.4f} "
                    f"val_loss={val_metrics['loss']:.6f} val_acc={val_metrics['accuracy']:.4f} "
                    f"test_loss={test_metrics['loss']:.6f} test_acc={test_metrics['accuracy']:.4f} "
                    f"elapsed_sec={elapsed:.2f}\n"
                )
            print(payload)

        if args.smoke_steps and global_step >= args.smoke_steps:
            if rank == 0:
                print("Smoke steps reached; stopping before full epochs.")
            break

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
                    "epochs_completed": epoch + 1,
                    "global_step": global_step,
                    "world_size": world,
                },
                indent=2,
            )
        )
        print("Saved model to", final_dir)
        print("Train time sec:", total_time)

    cleanup()


if __name__ == "__main__":
    main()
