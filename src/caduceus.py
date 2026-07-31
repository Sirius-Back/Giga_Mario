#!/usr/bin/env python3
"""Caduceus fine-tune on a splits directory (train/val/test).

Input:
  --splits-dir  directory with {train,val,test}/sequences/*.txt + labels.tsv
                (e.g. splits/random/M1 or splits/random/M2)

Output (--out, default runs/caduceus/<splits_name>/):
  logs/            epoch metrics.json / metrics.log / train_metrics.jsonl
  tensorboard/     Dual TB: summary/ (SummaryWriter) + lightning/ (TensorBoardLogger)
  checkpoints/     periodic HF checkpoints every N epochs (default 10)
  best_model/      best val_loss checkpoint + best_meta.json
  final_model/     copy of best_model (selected after train)

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
import shutil
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


class ZsvPairDataset(Dataset):
    """In-memory ``(sequence, target)`` rows from universal ZSV trees."""

    def __init__(
        self,
        pairs: list[tuple[str, str, float]],
        tokenizer,
        max_length: int,
    ):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        _rid, seq, y = self.pairs[idx]
        enc = self.tokenizer(
            seq,
            padding=False,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(float(y), dtype=torch.float32)
        return item


def evaluate_zsv_root(
    *,
    model_dir: Path,
    out_json: Path,
    zsv_root: Path | None = None,
    parsed_root: Path | None = None,
    predict_root: Path | None = None,
    batch_size: int = 192,
    max_length: int = 256,
    device: int | str = 0,
    amp: bool = True,
    task: str = "regression",
) -> dict[str, Any]:
    """Universal Caduceus ZSV: PARSED|PREDICT ``zero-shot-validation`` → metrics JSON.

    Accepts either ``zsv_root`` (panel/SPLIT parent with PARSED+PREDICT) or explicit
    ``parsed_root`` + ``predict_root``. Writes the same artifact shape as LegNet ZSV
    via ``src.pipeline.zsv_eval.write_zsv_artifacts``.
    """
    from src.pipeline.zsv_eval import load_zsv_pairs, metrics_from_preds, write_zsv_artifacts

    model_dir = Path(model_dir)
    if not (model_dir / "config.json").is_file():
        raise FileNotFoundError(f"Caduceus checkpoint missing config.json: {model_dir}")

    if parsed_root is None or predict_root is None:
        if zsv_root is None:
            raise ValueError("Need zsv_root or parsed_root+predict_root")
        zsv_root = Path(zsv_root)
        parsed_root = parsed_root or (zsv_root / "PARSED")
        if not (Path(parsed_root) / "zero-shot-validation").is_dir():
            parsed_root = zsv_root / "FASTA"
        predict_root = predict_root or (zsv_root / "PREDICT")

    pairs = load_zsv_pairs(parsed_root=Path(parsed_root), predict_root=Path(predict_root))
    if task != "regression":
        raise NotImplementedError(
            f"Caduceus ZSV currently supports regression only (got task={task!r})"
        )

    if isinstance(device, int):
        torch_device = torch.device(
            f"cuda:{device}" if torch.cuda.is_available() else "cpu"
        )
    else:
        torch_device = torch.device(device)

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 4
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_dir),
        trust_remote_code=True,
        num_labels=1,
        problem_type="regression",
    )
    model.to(torch_device)
    model.eval()

    ds = ZsvPairDataset(pairs, tokenizer, max_length)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch_device.type == "cuda",
        collate_fn=lambda b: collate_pad(b, pad_id),
    )

    preds: list[float] = []
    targets: list[float] = []
    use_amp = bool(amp) and torch_device.type == "cuda"
    criterion = torch.nn.MSELoss()
    total_loss, n = 0.0, 0
    for batch in loader:
        batch = {k: v.to(torch_device) for k, v in batch.items()}
        with torch.cuda.amp.autocast(enabled=use_amp):
            out = model(**batch)
            logits = out.logits.squeeze(-1)
            labels = batch["labels"]
            loss = criterion(logits, labels)
        bs = labels.size(0)
        total_loss += float(loss.item()) * bs
        n += bs
        preds.extend(logits.detach().float().cpu().reshape(-1).tolist())
        targets.extend(labels.detach().float().cpu().reshape(-1).tolist())

    if len(preds) != len(targets) or n == 0:
        raise RuntimeError(
            f"Caduceus ZSV pred/target mismatch or empty: {len(preds)} vs {len(targets)}"
        )

    metrics = metrics_from_preds(preds, targets)
    # Align with train-time regression suite when available.
    try:
        rich = compute_epoch_regression_metrics(
            torch.tensor(preds, dtype=torch.float32),
            torch.tensor(targets, dtype=torch.float32),
            loss=total_loss / max(n, 1),
        )
        metrics = {**metrics, **{k: float(v) for k, v in rich.items()}}
        metrics["n"] = int(n)
        metrics["loss"] = float(total_loss / max(n, 1))
    except Exception:
        metrics["loss"] = float(total_loss / max(n, 1))

    return write_zsv_artifacts(
        out_json=Path(out_json),
        model="caduceus",
        checkpoint=model_dir,
        metrics=metrics,
        extra={
            "n_pairs": len(pairs),
            "max_length": int(max_length),
            "batch_size": int(batch_size),
            "amp": bool(use_amp),
            "device": str(torch_device),
        },
    )


def tb_log_split(writer, split: str, metrics: dict[str, Any], epoch: int) -> None:
    """Backward-compatible SummaryWriter-only helper."""
    from src.tb_logging import log_split_metrics

    log_split_metrics(writer, None, split, metrics, epoch)


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
        help="Output root (logs/, tensorboard/, checkpoints/, best_model/, "
        "final_model/=best). Default: runs/caduceus/<splits_name>/",
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
        default=8192,
        help="Cap train-split epoch eval (same default as --eval-max-samples; "
        "speed over full-fold precision). Pass 0 for uncapped.",
    )
    p.add_argument(
        "--eval-max-samples",
        type=int,
        default=8192,
        help="Cap val/test epoch eval sample count (default 8192; speed over "
        "full-fold precision). Pass 0 for uncapped.",
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
    p.add_argument(
        "--checkpoint-every-n-epochs",
        type=int,
        default=10,
        help="Save a periodic checkpoint under out/checkpoints/epochN every N "
        "epochs (0 disables). Best val_loss is always tracked in best_model/; "
        "final_model/ is set to best after train.",
    )
    p.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help="Stop training after this many epochs without val_loss improvement "
        "(0 disables). Best checkpoint is still promoted to final_model/.",
    )
    p.add_argument(
        "--min-epochs",
        type=int,
        default=0,
        help="Do not early-stop before this many completed epochs (0 disables).",
    )
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

    # 0 → uncapped; None should not occur with new defaults but keep DDP fallback.
    def _cap(v):
        if v is None:
            return 8192
        if int(v) <= 0:
            return None
        return int(v)

    eval_cap = _cap(args.eval_max_samples)
    train_eval_cap = _cap(args.train_eval_max_samples)
    if train_eval_cap is None and args.train_eval_max_samples is None:
        train_eval_cap = eval_cap

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
        # drop_last keeps equal batch counts across ranks (required for DDP sync).
        sampler = (
            DistributedSampler(
                ds, shuffle=shuffle, seed=args.seed, drop_last=bool(shuffle)
            )
            if (world > 1 and distributed)
            else None
        )
        # CUDA is initialized before DataLoader workers; fork + CUDA often deadlocks
        # under DDP (one rank waits on workers, the other blocks on NCCL). Prefer
        # in-process loading for multi-GPU; optional spawn if workers > 0.
        nw = int(args.num_workers)
        if world > 1 and nw > 0:
            nw = 0
        loader_kw: dict[str, Any] = {
            "batch_size": batch_size,
            "shuffle": (sampler is None and shuffle),
            "sampler": sampler,
            "num_workers": nw,
            "pin_memory": torch.cuda.is_available(),
            "collate_fn": lambda b: collate_pad(b, pad_id),
        }
        if nw > 0:
            loader_kw["persistent_workers"] = False
        loader = DataLoader(ds, **loader_kw)
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
        # eval_cap set above from --eval-max-samples
        val_loader, val_ds, _ = make_loader(
            "val", args.eval_batch_size, False, False, max_samples=eval_cap
        )
        test_loader, test_ds, _ = make_loader(
            "test", args.eval_batch_size, False, False, max_samples=eval_cap
        )

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    writer = None
    tb_logger = None
    if rank == 0:
        from src.tb_logging import (
            close_dual,
            log_scalar_pair,
            log_split_metrics,
            open_summary_writer,
            open_tensorboard_logger,
            tensorboard_root,
        )

        writer = open_summary_writer(out)
        tb_logger = open_tensorboard_logger(out)
        tb_dir = tensorboard_root(out)
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
            "eval_max_samples": eval_cap,
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
            "tensorboard_summary": str(tb_dir / "summary"),
            "tensorboard_lightning": str(tb_dir / "lightning"),
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
    patience = int(getattr(args, "early_stopping_patience", 0) or 0)
    min_epochs = int(getattr(args, "min_epochs", 0) or 0)
    epochs_since_improve = 0
    epochs_completed = start_epoch
    stopped_early = False

    for epoch_idx in range(start_epoch, args.epochs):
        epoch = epoch_idx + 1
        if train_sampler is not None:
            train_sampler.set_epoch(epoch_idx)
        model.train()
        running_loss, seen = 0.0, 0
        n_train_batches = len(train_loader)
        if rank == 0:
            print(
                json.dumps(
                    {
                        "epoch_start": epoch,
                        "train_batches_per_rank": n_train_batches,
                        "world_size": world,
                    }
                ),
                flush=True,
            )
        for step_in_epoch, batch in enumerate(train_loader, start=1):
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
            if rank == 0 and (step_in_epoch == 1 or step_in_epoch % 50 == 0):
                print(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "step": step_in_epoch,
                            "steps": n_train_batches,
                            "batch_loss": float(loss.item()),
                            "elapsed_sec": time.perf_counter() - t0,
                        }
                    ),
                    flush=True,
                )
            if rank == 0 and writer is not None and global_step % 50 == 0:
                log_scalar_pair(
                    writer, tb_logger, "train/batch_loss", float(loss.item()), global_step
                )

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
        stop_flag = torch.zeros(1, device=device, dtype=torch.int32)
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
                log_scalar_pair(writer, tb_logger, "train/optim_loss", opt_loss, epoch)
                log_split_metrics(writer, tb_logger, "train", train_metrics, epoch)
                log_split_metrics(writer, tb_logger, "validation", val_metrics, epoch)
                log_split_metrics(writer, tb_logger, "test", test_metrics, epoch)
                writer.flush()
                if tb_logger is not None:
                    tb_logger.save()

            # checkpoint best by val loss
            vloss = float(val_metrics["loss"])
            if vloss < best_val:
                best_val = vloss
                epochs_since_improve = 0
                best_dir = out / "best_model"
                best_dir.mkdir(parents=True, exist_ok=True)
                raw_model.save_pretrained(best_dir)
                tokenizer.save_pretrained(best_dir)
                (best_dir / "best_meta.json").write_text(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "metric": "val_loss",
                            "value": best_val,
                            "val_loss": best_val,
                            "selection": "min_val_loss",
                        },
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
            else:
                epochs_since_improve += 1

            # Periodic epoch checkpoints (every N epochs; also last epoch if
            # it is not already a multiple — kept only on the every-N grid).
            every_n = int(getattr(args, "checkpoint_every_n_epochs", 10) or 0)
            if every_n > 0 and epoch % every_n == 0:
                ckpt_dir = out / "checkpoints" / f"epoch{epoch}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                raw_model.save_pretrained(ckpt_dir)
                tokenizer.save_pretrained(ckpt_dir)
                (ckpt_dir / "epoch_meta.json").write_text(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "val_loss": float(val_metrics["loss"]),
                            "kind": "periodic",
                            "every_n_epochs": every_n,
                        },
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                print(f"Saved periodic checkpoint → {ckpt_dir}", flush=True)

            print(json.dumps(payload), flush=True)

            if (
                patience > 0
                and epochs_since_improve >= patience
                and (min_epochs <= 0 or epoch >= min_epochs)
            ):
                stop_flag[0] = 1
                print(
                    json.dumps(
                        {
                            "early_stopping": True,
                            "patience": patience,
                            "min_epochs": min_epochs,
                            "epochs_since_improve": epochs_since_improve,
                            "best_val_loss": best_val,
                            "stopped_at_epoch": epoch,
                        }
                    ),
                    flush=True,
                )

        if dist.is_initialized():
            dist.broadcast(stop_flag, src=0)
            dist.barrier()

        epochs_completed = epoch
        if int(stop_flag.item()) > 0:
            stopped_early = True
            break

        if dist.is_initialized():
            dist.barrier()

    total_time = time.perf_counter() - t0
    if rank == 0:
        # final_model = best validation checkpoint (not last epoch)
        best_dir = out / "best_model"
        best_meta_path = best_dir / "best_meta.json"
        if best_meta_path.is_file():
            if final_dir.exists():
                shutil.rmtree(final_dir)
            shutil.copytree(best_dir, final_dir)
            selection = json.loads(best_meta_path.read_text(encoding="utf-8"))
            selection["promoted_to_final"] = True
            (final_dir / "best_meta.json").write_text(
                json.dumps(selection, indent=2) + "\n", encoding="utf-8"
            )
            print(
                f"Selected best_model (epoch={selection.get('epoch')}, "
                f"val_loss={selection.get('val_loss')}) → final_model",
                flush=True,
            )
        else:
            raw_model.save_pretrained(final_dir)
            tokenizer.save_pretrained(final_dir)
            print(
                "WARNING: no best_model found; saved last-epoch weights as final_model",
                flush=True,
            )
        timing = {
            "train_time_sec": total_time,
            "train_time_min": total_time / 60.0,
            "epochs_completed": epochs_completed,
            "epochs_requested": args.epochs,
            "early_stopping_patience": patience,
            "stopped_early": stopped_early,
            "global_step": global_step,
            "world_size": world,
            "seed": args.seed,
            "best_val_loss": best_val,
            "checkpoint_every_n_epochs": int(
                getattr(args, "checkpoint_every_n_epochs", 10) or 0
            ),
            "final_model_source": "best_model" if best_meta_path.is_file() else "last_epoch",
        }
        (logs_dir / "train_time.json").write_text(json.dumps(timing, indent=2), encoding="utf-8")
        (out / "train_time.json").write_text(json.dumps(timing, indent=2), encoding="utf-8")
        if writer is not None or tb_logger is not None:
            from src.tb_logging import close_dual

            close_dual(writer, tb_logger)
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
