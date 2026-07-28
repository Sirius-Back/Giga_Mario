#!/usr/bin/env python3
"""Caduceus fine-tune on a splits directory (train/val/test).

Input:
  --splits-dir  directory with {train,val,test}/sequences/*.txt + labels.tsv
                (e.g. splits/random/M1 or splits/random/M2)

Output (--out, default runs/caduceus/<splits_name>/):
  logs/            epoch metrics.json / metrics.log / train_metrics.jsonl
  tensorboard/     TensorBoard event files
  final_model/     HF save_pretrained checkpoint + tokenizer

Task auto-detect from labels.tsv:
  - column ``label`` (int) → classification (M2-style)
  - else ``TPM`` → continuous regression (M1-style; metrics.md suite)

Reuse: this module is the sole @caduceus training entry — skill execs it, does not
reimplement training in-chat.

Example:
  python -m src.caduceus --splits-dir splits/random/M1 --epochs 20 --seed 42
  torchrun --nproc_per_node=4 -m src.caduceus --splits-dir splits/random/M1 ...
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# metrics.md helpers live in src/ (canonical; skills re-export if needed)
from src.metrics_logging import (
    compute_epoch_regression_metrics,
    format_epoch_log,
)

DEFAULT_MODEL = "kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16"
FOLD_NAMES = ("train", "val", "test")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
class SplitWindowDataset(Dataset):
    """{split}/labels.tsv + sequences/<id>.txt → (tokens, target)."""

    def __init__(
        self,
        data_root: Path,
        split: str,
        tokenizer,
        max_length: int,
        *,
        task: str,
        max_samples: int | None = None,
    ):
        self.root = data_root.resolve()
        self.split = split
        self.task = task
        labels_path = self.root / split / "labels.tsv"
        if not labels_path.is_file():
            raise FileNotFoundError(f"Missing labels: {labels_path}")
        self.samples: list[tuple[Path, float]] = []
        with labels_path.open(encoding="utf-8") as fh:
            header = fh.readline().rstrip("\n").split("\t")
            col = {name: i for i, name in enumerate(header)}
            if "sample_id" not in col:
                raise ValueError(f"{labels_path} missing sample_id: {header}")
            if task == "classification":
                if "label" not in col:
                    raise ValueError(f"{labels_path} missing label for classification")
                y_key = "label"
            else:
                if "TPM" not in col:
                    raise ValueError(f"{labels_path} missing TPM for regression")
                y_key = "TPM"
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < len(header):
                    continue
                sid = parts[col["sample_id"]]
                y_raw = parts[col[y_key]]
                y = float(y_raw) if task == "regression" else float(int(round(float(y_raw))))
                seq_path = self._resolve_seq(sid, parts, col)
                self.samples.append((seq_path, y))
                if max_samples is not None and len(self.samples) >= max_samples:
                    break
        if not self.samples:
            raise SystemExit(f"Empty dataset for split={split} under {labels_path}")
        self.tokenizer = tokenizer
        self.max_length = max_length

    def _resolve_seq(self, sid: str, parts: list[str], col: dict[str, int]) -> Path:
        if "path" in col and parts[col["path"]]:
            rel = parts[col["path"]]
            for cand in (
                self.root / rel,
                self.root / self.split / Path(rel).name,
                self.root / self.split / "sequences" / Path(rel).name,
                self.root / self.split / "sequences" / f"{sid}.txt",
            ):
                if cand.is_file():
                    return cand
        seq_path = self.root / self.split / "sequences" / f"{sid}.txt"
        if not seq_path.is_file():
            raise FileNotFoundError(f"Sequence missing for {sid}: {seq_path}")
        return seq_path

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
        if self.task == "classification":
            item["labels"] = torch.tensor(int(y), dtype=torch.long)
        else:
            item["labels"] = torch.tensor(y, dtype=torch.float32)
        return item


def collate_pad(batch: list[dict], pad_id: int) -> dict[str, torch.Tensor]:
    max_len = max(x["input_ids"].numel() for x in batch)
    input_ids, attention_mask, labels = [], [], []
    for x in batch:
        ids = x["input_ids"]
        mask = x.get("attention_mask", torch.ones_like(ids))
        pad_len = max_len - ids.numel()
        if pad_len > 0:
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


def detect_task(splits_dir: Path) -> str:
    labels = splits_dir / "train" / "labels.tsv"
    if not labels.is_file():
        raise FileNotFoundError(f"Need {labels}")
    header = labels.read_text(encoding="utf-8").splitlines()[0].split("\t")
    if "label" in header:
        return "classification"
    if "TPM" in header:
        return "regression"
    raise ValueError(f"Cannot detect task from columns: {header}")


def validate_splits_dir(splits_dir: Path) -> None:
    for fold in FOLD_NAMES:
        labels = splits_dir / fold / "labels.tsv"
        seqs = splits_dir / fold / "sequences"
        if not labels.is_file() or not seqs.is_dir():
            raise FileNotFoundError(f"Invalid splits layout: need {labels} and {seqs}/")


def default_out_dir(splits_dir: Path, root: Path) -> Path:
    # splits/random/M1 → runs/caduceus/random_M1
    try:
        rel = splits_dir.resolve().relative_to(root.resolve())
        name = "_".join(rel.parts[-2:]) if len(rel.parts) >= 2 else rel.name
    except ValueError:
        name = splits_dir.name
    return root / "runs" / "caduceus" / name


# ---------------------------------------------------------------------------
# Distributed / eval
# ---------------------------------------------------------------------------
def setup_distributed() -> tuple[int, int, int]:
    if "RANK" in os.environ:
        # Rank 0 may run long val/test (and capped train) eval while other ranks wait.
        dist.init_process_group(backend="nccl", timeout=timedelta(hours=4))
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
def evaluate_regression(
    model, loader, device, criterion, *, amp: bool = False
) -> dict[str, float]:
    model.eval()
    preds, targets = [], []
    total_loss, n = 0.0, 0
    use_amp = amp and device.type == "cuda"
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.cuda.amp.autocast(enabled=use_amp):
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
        return {k: float("nan") for k in (
            "loss", "pearson", "spearman", "mse", "rmse", "mae", "r2",
            "genewise_pearson_median", "samplewise_pearson_median",
        )} | {"n": 0}
    metrics = compute_epoch_regression_metrics(
        torch.cat(preds), torch.cat(targets), loss=total_loss / n
    )
    metrics["n"] = int(n)
    return metrics


@torch.no_grad()
def evaluate_classification(
    model, loader, device, *, amp: bool = False
) -> dict[str, float]:
    model.eval()
    total_loss, correct, n = 0.0, 0.0, 0
    use_amp = amp and device.type == "cuda"
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.cuda.amp.autocast(enabled=use_amp):
            out = model(**batch)
            bs = batch["labels"].size(0)
            total_loss += float(out.loss.item()) * bs
            preds = out.logits.argmax(dim=-1)
            correct += float((preds == batch["labels"]).float().sum().item())
            n += bs
    if n == 0:
        return {"loss": float("nan"), "accuracy": float("nan"), "n": 0}
    return {"loss": total_loss / n, "accuracy": correct / n, "n": int(n)}


def tb_log_split(writer, split: str, metrics: dict[str, Any], epoch: int) -> None:
    for k, v in metrics.items():
        if k == "n":
            continue
        try:
            writer.add_scalar(f"{split}/{k}", float(v), epoch)
        except (TypeError, ValueError):
            continue


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--splits-dir",
        type=Path,
        required=True,
        help="Directory with train/val/test folds (e.g. splits/random/M1)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output root (logs/, tensorboard/, final_model/). "
        "Default: runs/caduceus/<splits_name>/",
    )
    p.add_argument("--root", type=Path, default=Path("."))
    p.add_argument("--model-name", type=str, default=DEFAULT_MODEL)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--eval-batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max-length", type=int, default=8192)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument(
        "--amp",
        action="store_true",
        help="CUDA automatic mixed precision (fp16 autocast + GradScaler)",
    )
    p.add_argument("--max-samples", type=int, default=None, help="Smoke-test cap per fold")
    p.add_argument(
        "--train-eval-max-samples",
        type=int,
        default=None,
        help="Cap train-split epoch eval (default: full on 1 GPU; 8192 under DDP)",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume from out/best_model if best_meta.json exists",
    )
    p.add_argument(
        "--task",
        choices=("auto", "regression", "classification"),
        default="auto",
    )
    p.add_argument("--num-labels", type=int, default=None, help="Override class count")
    return p.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    root = args.root.resolve()
    splits_dir = args.splits_dir if args.splits_dir.is_absolute() else root / args.splits_dir
    splits_dir = splits_dir.resolve()
    validate_splits_dir(splits_dir)

    task = detect_task(splits_dir) if args.task == "auto" else args.task
    out = args.out if args.out is not None else default_out_dir(splits_dir, root)
    out = out if out.is_absolute() else root / out
    out = out.resolve()

    rank, world, local = setup_distributed()
    device = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + rank)

    train_eval_cap = args.train_eval_max_samples
    if train_eval_cap is None and world > 1:
        train_eval_cap = 8192

    logs_dir = out / "logs"
    tb_dir = out / "tensorboard"
    final_dir = out / "final_model"
    if rank == 0:
        for d in (logs_dir, tb_dir, final_dir):
            d.mkdir(parents=True, exist_ok=True)

    if dist.is_initialized():
        dist.barrier()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 4

    best_meta_path = out / "best_model" / "best_meta.json"
    start_epoch = 0
    best_val = float("inf")
    resume_dir: Path | None = None
    if args.resume and best_meta_path.is_file():
        resume_dir = out / "best_model"
        meta_resume = json.loads(best_meta_path.read_text(encoding="utf-8"))
        start_epoch = int(meta_resume.get("epoch", 0))
        best_val = float(meta_resume.get("val_loss", float("inf")))
        if rank == 0:
            print(
                json.dumps(
                    {
                        "resume": True,
                        "checkpoint": str(resume_dir),
                        "completed_epochs": start_epoch,
                        "best_val_loss": best_val,
                    },
                    indent=2,
                ),
                flush=True,
            )
    load_from = str(resume_dir) if resume_dir is not None else args.model_name

    if task == "regression":
        model = AutoModelForSequenceClassification.from_pretrained(
            load_from,
            trust_remote_code=True,
            num_labels=1,
            problem_type="regression",
            ignore_mismatched_sizes=resume_dir is None,
        )
        criterion = torch.nn.MSELoss()
        num_labels = 1
    else:
        # Infer class count from train labels if not set
        num_labels = args.num_labels
        if num_labels is None:
            ys = set()
            with (splits_dir / "train" / "labels.tsv").open(encoding="utf-8") as fh:
                header = fh.readline().rstrip("\n").split("\t")
                col = {n: i for i, n in enumerate(header)}
                for line in fh:
                    parts = line.rstrip("\n").split("\t")
                    ys.add(int(round(float(parts[col["label"]]))))
                    if args.max_samples and len(ys) >= 3 and False:
                        break
            num_labels = max(ys) + 1 if ys else 3
        model = AutoModelForSequenceClassification.from_pretrained(
            load_from,
            trust_remote_code=True,
            num_labels=num_labels,
            problem_type="single_label_classification",
            ignore_mismatched_sizes=resume_dir is None,
        )
        criterion = None

    model.to(device)
    if world > 1:
        model = DDP(model, device_ids=[local], output_device=local)

    def make_loader(
        split: str,
        batch_size: int,
        shuffle: bool,
        distributed: bool,
        *,
        max_samples: int | None = None,
    ):
        cap = args.max_samples if max_samples is None else max_samples
        ds = SplitWindowDataset(
            splits_dir,
            split,
            tokenizer,
            args.max_length,
            task=task,
            max_samples=cap,
        )
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
    val_loader = test_loader = train_eval_loader = None
    val_ds = test_ds = None
    if rank == 0:
        train_eval_loader, _, _ = make_loader(
            "train", args.eval_batch_size, False, False, max_samples=train_eval_cap
        )
        val_loader, val_ds, _ = make_loader("val", args.eval_batch_size, False, False)
        test_loader, test_ds, _ = make_loader("test", args.eval_batch_size, False, False)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    writer = None
    if rank == 0:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir=str(tb_dir))
        meta = {
            "model_name": args.model_name,
            "task": task,
            "num_labels": num_labels,
            "splits_dir": str(splits_dir),
            "out": str(out),
            "n_train": len(train_ds),
            "n_val": len(val_ds) if val_ds is not None else 0,
            "n_test": len(test_ds) if test_ds is not None else 0,
            "epochs": args.epochs,
            "start_epoch": start_epoch,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "train_eval_max_samples": train_eval_cap,
            "world_size": world,
            "max_length": args.max_length,
            "amp": bool(args.amp),
            "lr": args.lr,
            "seed": args.seed,
            "max_samples": args.max_samples,
            "resume": resume_dir is not None,
            "optimizer": "AdamW",
            "loss": "MSELoss" if task == "regression" else "CrossEntropy (model.loss)",
            "metrics": "metrics.md" if task == "regression" else "loss+accuracy",
            "tensorboard": str(tb_dir),
            "script": str(Path(__file__).resolve()),
        }
        (logs_dir / "run_config.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        (out / "run_config.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        metrics_mode = "a" if resume_dir is not None else "w"
        with (logs_dir / "train_metrics.jsonl").open(metrics_mode, encoding="utf-8") as fh:
            fh.write(json.dumps(meta) + "\n")
        print(json.dumps(meta, indent=2), flush=True)

    t0 = time.perf_counter()
    global_step = 0
    raw_model = model.module if isinstance(model, DDP) else model
    use_amp = bool(args.amp) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for epoch_idx in range(start_epoch, args.epochs):
        epoch = epoch_idx + 1
        if train_sampler is not None:
            train_sampler.set_epoch(epoch_idx)
        model.train()
        running_loss, seen = 0.0, 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optim.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                out_m = model(**batch)
                loss = out_m.loss
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()
            else:
                loss.backward()
                optim.step()
            bs = batch["labels"].size(0)
            running_loss += float(loss.item()) * bs
            seen += bs
            global_step += 1
            if rank == 0 and writer is not None and global_step % 50 == 0:
                writer.add_scalar("train/batch_loss", float(loss.item()), global_step)

        opt_loss = running_loss / max(seen, 1)
        if world > 1:
            t = torch.tensor([opt_loss, float(seen)], device=device, dtype=torch.float64)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            opt_loss = (t[0] / t[1]).item()

        train_metrics = val_metrics = test_metrics = None
        if rank == 0:
            if task == "regression":
                assert criterion is not None
                train_metrics = evaluate_regression(
                    raw_model, train_eval_loader, device, criterion, amp=use_amp
                )
                val_metrics = evaluate_regression(
                    raw_model, val_loader, device, criterion, amp=use_amp
                )
                test_metrics = evaluate_regression(
                    raw_model, test_loader, device, criterion, amp=use_amp
                )
            else:
                train_metrics = evaluate_classification(
                    raw_model, train_eval_loader, device, amp=use_amp
                )
                val_metrics = evaluate_classification(
                    raw_model, val_loader, device, amp=use_amp
                )
                test_metrics = evaluate_classification(
                    raw_model, test_loader, device, amp=use_amp
                )

        elapsed = time.perf_counter() - t0
        if rank == 0:
            assert train_metrics and val_metrics and test_metrics
            payload = {
                "epoch": epoch,
                "train": train_metrics,
                "validation": val_metrics,
                "test": test_metrics,
                "train_optim_loss": opt_loss,
                "elapsed_sec": elapsed,
                "global_step": global_step,
                "lr": args.lr,
            }
            epoch_dir = logs_dir / f"epoch{epoch}"
            epoch_dir.mkdir(parents=True, exist_ok=True)
            (epoch_dir / "metrics.json").write_text(
                json.dumps(payload, indent=2), encoding="utf-8"
            )
            if task == "regression":
                log_lines = [
                    format_epoch_log("train", train_metrics, epoch=epoch),
                    format_epoch_log("validation", val_metrics, epoch=epoch),
                    format_epoch_log("test", test_metrics, epoch=epoch),
                ]
            else:
                log_lines = [
                    f"epoch={epoch} train_loss={train_metrics['loss']:.6g} "
                    f"train_accuracy={train_metrics['accuracy']:.6g}",
                    f"epoch={epoch} validation_loss={val_metrics['loss']:.6g} "
                    f"validation_accuracy={val_metrics['accuracy']:.6g}",
                    f"epoch={epoch} test_loss={test_metrics['loss']:.6g} "
                    f"test_accuracy={test_metrics['accuracy']:.6g}",
                ]
            (epoch_dir / "metrics.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
            with (logs_dir / "train_metrics.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload) + "\n")
            with (logs_dir / "metrics.log").open("a", encoding="utf-8") as fh:
                fh.write("\n".join(log_lines) + "\n")

            if writer is not None:
                writer.add_scalar("train/optim_loss", opt_loss, epoch)
                tb_log_split(writer, "train", train_metrics, epoch)
                tb_log_split(writer, "validation", val_metrics, epoch)
                tb_log_split(writer, "test", test_metrics, epoch)
                writer.flush()

            # checkpoint best by val loss
            vloss = float(val_metrics["loss"])
            if vloss < best_val:
                best_val = vloss
                best_dir = out / "best_model"
                best_dir.mkdir(parents=True, exist_ok=True)
                raw_model.save_pretrained(best_dir)
                tokenizer.save_pretrained(best_dir)
                (best_dir / "best_meta.json").write_text(
                    json.dumps({"epoch": epoch, "val_loss": best_val}, indent=2),
                    encoding="utf-8",
                )

            print(json.dumps(payload), flush=True)

        if dist.is_initialized():
            dist.barrier()

    total_time = time.perf_counter() - t0
    if rank == 0:
        raw_model.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)
        timing = {
            "train_time_sec": total_time,
            "train_time_min": total_time / 60.0,
            "epochs_completed": args.epochs,
            "global_step": global_step,
            "world_size": world,
            "seed": args.seed,
            "best_val_loss": best_val,
        }
        (logs_dir / "train_time.json").write_text(json.dumps(timing, indent=2), encoding="utf-8")
        (out / "train_time.json").write_text(json.dumps(timing, indent=2), encoding="utf-8")
        if writer is not None:
            writer.close()
        print("Saved final_model →", final_dir, flush=True)
        print("TensorBoard →", tb_dir, flush=True)
        print("Logs →", logs_dir, flush=True)
        print("Train time sec:", total_time, flush=True)

    cleanup()
    return 0


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
