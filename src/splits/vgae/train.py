"""VGAE training loop: recon + KL + size + L_hom; early stop after ≥25 epochs.

Supports legacy objective (default) and additive ``loss_mode=homology_first``
(EMA term norm, KL anneal, Gumbel-Softmax weighted L_hom).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.pipeline.job_queue import (
    CLASS_GPU_TRAIN,
    CLASS_WAITER,
    append_queue_entry,
    can_launch_parallel,
    wait_until_launchable,
)
from src.splits.vgae.assign import (
    ROLE_ORDER,
    assignment_rows,
    role_target_fractions,
    size_constrained_assign,
    size_loss,
)
from src.splits.vgae.graph_data import PackedGraph, assert_no_homology_features, load_packed_graph
from src.splits.vgae.homology_loss import (
    EmaTermNorm,
    compute_l_hom,
    load_homology_groups,
    write_homology_sidecar,
)
from src.splits.vgae.contrastive import augment_graph_view, info_nce_pairwise
from src.splits.vgae.model import (
    ARCH_GCN,
    ClassicVGAE,
    build_vgae,
    gumbel_softmax_roles,
    gumbel_tau_schedule,
    kl_beta_schedule,
    soft_role_probs,
    uses_contrastive,
)
from src.splits.sbs.assign import assignment_rows_to_split_csv
from src.tb_logging import close_dual, log_scalar_pair, open_summary_writer, open_tensorboard_logger


# Homology-first defaults (legacy path keeps beta_kl=lambda_hom=lambda_size=1)
HOMOLOGY_FIRST_DEFAULTS: dict[str, Any] = {
    "alpha_recon": 0.3,
    "lambda_hom": 25.0,
    "lambda_size": 1.0,
    "beta_kl_max": 0.05,
    "kl_anneal_epochs": 15,
    "gumbel_tau_start": 1.0,
    "gumbel_tau_end": 0.3,
    "gumbel_anneal_epochs": 20,
    "lambda_para": 1.0,
    "lambda_ortho": 1.0,
    "ema_decay": 0.9,
    "hom_max_groups": 4096,
}

# loss_mode → sd aggregation (additive; legacy/homology_first unchanged)
LOSS_MODE_HOM_AGG: dict[str, str] = {
    "homology_first": "weighted",
    "homology_robust": "robust",
    "homology_log_balance": "log_balance",
}
HOMOLOGY_TRAIN_MODES = frozenset(LOSS_MODE_HOM_AGG.keys())


def _is_homology_train_mode(mode: str) -> bool:
    return str(mode).lower().strip() in HOMOLOGY_TRAIN_MODES


def _hom_agg_for_mode(mode: str) -> str | None:
    return LOSS_MODE_HOM_AGG.get(str(mode).lower().strip())


def _gpu_used_bytes(gpu_idx: int) -> int:
    free, total = torch.cuda.mem_get_info(int(gpu_idx))
    return int(total - free)


def _gpu_is_free(gpu_idx: int, *, max_used_mib: float = 512.0) -> bool:
    """True when device has near-empty VRAM (occupied GPUs are refused)."""
    used_mib = _gpu_used_bytes(gpu_idx) / (1024.0 * 1024.0)
    return used_mib <= float(max_used_mib)


def _pick_free_gpu(*, max_used_mib: float = 512.0) -> int | None:
    """Return a free GPU index, or None if every device is occupied.

    Does not kill processes. Prefers lowest used among free + queue-launchable.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for VGAE train (plan: 1×GPU)")
    cands: list[tuple[int, int]] = []
    for i in range(torch.cuda.device_count()):
        if not _gpu_is_free(i, max_used_mib=max_used_mib):
            continue
        ok, _reason = can_launch_parallel(
            peak_ram_gib=0.0,
            gpus=(i,),
            job_class=CLASS_GPU_TRAIN,
        )
        if not ok:
            continue
        cands.append((_gpu_used_bytes(i), i))
    if not cands:
        return None
    cands.sort()
    return int(cands[0][1])


def wait_for_free_gpu(
    *,
    peak_ram_gib: float = 12.0,
    max_used_mib: float = 512.0,
    timeout_sec: float = 6 * 3600,
    poll_sec: float = 600.0,
    label: str = "vgae_gpu_wait",
    register_waiter: bool = True,
) -> int:
    """Block until a free (near-empty VRAM) GPU is launchable. Never kill PIDs."""
    waiter_name = f"waiter_{label}"
    if register_waiter:
        append_queue_entry(
            waiter_name,
            job=f"wait_for_free_gpu:{label}",
            pid=os.getpid(),
            estimated_time=f"poll {poll_sec:.0f}s",
            status="RUNNING",
            job_class=CLASS_WAITER,
            peak_ram_gib=0.0,
            gpus=(),
            resources=f"max_used_mib={max_used_mib}",
        )
    t0 = time.monotonic()
    try:
        while True:
            gpu_idx = _pick_free_gpu(max_used_mib=max_used_mib)
            if gpu_idx is not None:
                wait_until_launchable(
                    peak_ram_gib=float(peak_ram_gib),
                    gpus=(gpu_idx,),
                    job_class=CLASS_GPU_TRAIN,
                    timeout_sec=max(60.0, float(timeout_sec) - (time.monotonic() - t0)),
                    poll_sec=min(float(poll_sec), 60.0),
                    label=f"{label}:confirm_gpu{gpu_idx}",
                )
                # Re-check VRAM after queue wait (another job may have started)
                if _gpu_is_free(gpu_idx, max_used_mib=max_used_mib):
                    ok, reason = can_launch_parallel(
                        peak_ram_gib=float(peak_ram_gib),
                        gpus=(gpu_idx,),
                        job_class=CLASS_GPU_TRAIN,
                    )
                    if ok:
                        print(
                            f"[vgae] {label}: free GPU cuda:{gpu_idx} — {reason}",
                            flush=True,
                        )
                        if register_waiter:
                            _append_status(
                                waiter_name,
                                "COMPLETED",
                                note=f"acquired cuda:{gpu_idx}",
                            )
                        return int(gpu_idx)
            elapsed = time.monotonic() - t0
            print(
                f"[vgae] {label}: no free GPU yet "
                f"(elapsed={elapsed:.0f}s); sleeping {poll_sec:.0f}s",
                flush=True,
            )
            if elapsed >= float(timeout_sec):
                if register_waiter:
                    _append_status(waiter_name, "FAILED", note="timeout waiting free GPU")
                raise TimeoutError(
                    f"[vgae] {label}: timed out waiting for a free GPU "
                    f"(max_used_mib={max_used_mib})"
                )
            time.sleep(float(poll_sec))
    except Exception:
        if register_waiter:
            _append_status(waiter_name, "FAILED", note="exception during GPU wait")
        raise


def _append_status(queue_name: str, status: str, *, note: str = "") -> None:
    """Append a status follow-up line to queue.md (do not delete history)."""
    from src.pipeline.job_queue import queue_path

    p = queue_path()
    lines = [f"\n### {queue_name} — {status}"]
    lines.append(f"- **update time:** {time.strftime('%Y-%m-%dT%H:%M:%S%z')}")
    if note:
        lines.append(f"- **note:** {note}")
    lines.append(f"- **status:** {status}")
    with p.open("a", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def _write_epoch_logs(
    out_dir: Path,
    epoch_rows: list[dict[str, Any]],
) -> None:
    logs = out_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    jsonl = logs / "train_metrics.jsonl"
    mlog = logs / "metrics.log"
    lines_j: list[str] = []
    lines_l: list[str] = []
    for rec in epoch_rows:
        ep = int(rec["epoch"])
        train_block = {
            "loss": float(rec["loss"]),
            "recon": float(rec["recon"]),
            "kl": float(rec["kl"]),
            "l_hom_soft": float(rec["l_hom_soft"]),
            "size": float(rec["size"]),
        }
        for key in (
            "recon_norm",
            "kl_norm",
            "beta_kl",
            "gumbel_tau",
            "hom_grad_share",
            "ema_recon",
            "ema_kl",
        ):
            if key in rec and rec[key] is not None:
                train_block[key] = float(rec[key])
        obj = {
            "epoch": ep,
            "train": train_block,
            "validation": {
                "loss": float(rec["l_hom_hard"]),
                "l_hom": float(rec["l_hom_hard"]),
                "mean_sd_ortho": float(rec["mean_sd_ortho"]),
                "mean_sd_para": float(rec["mean_sd_para"]),
            },
            "test": {
                "loss": float(rec["l_hom_hard"]),
                "l_hom": float(rec["l_hom_hard"]),
            },
        }
        lines_j.append(json.dumps(obj, sort_keys=True))
        lines_l.append(
            f"epoch={ep} train_loss={obj['train']['loss']:.6g} "
            f"val_l_hom={obj['validation']['l_hom']:.6g} "
            f"sd_ortho={obj['validation']['mean_sd_ortho']:.6g} "
            f"sd_para={obj['validation']['mean_sd_para']:.6g}"
        )
        ep_dir = logs / f"epoch{ep}"
        ep_dir.mkdir(parents=True, exist_ok=True)
        (ep_dir / "metrics.json").write_text(
            json.dumps(obj, indent=2) + "\n", encoding="utf-8"
        )
    jsonl.write_text("\n".join(lines_j) + ("\n" if lines_j else ""), encoding="utf-8")
    mlog.write_text("\n".join(lines_l) + ("\n" if lines_l else ""), encoding="utf-8")


def resolve_device(
    device: str | None,
    *,
    peak_ram_gib: float,
    wait_poll_sec: float,
    label: str,
    max_used_mib: float = 512.0,
    register_waiter: bool = True,
) -> tuple[str, int | None]:
    """Resolve CUDA device: wait for a free GPU when ``device`` is None."""
    if device is None:
        gpu_idx = wait_for_free_gpu(
            peak_ram_gib=float(peak_ram_gib),
            max_used_mib=float(max_used_mib),
            poll_sec=float(wait_poll_sec),
            label=label,
            register_waiter=register_waiter,
        )
        return f"cuda:{gpu_idx}", gpu_idx
    if device.startswith("cuda"):
        parts = device.split(":")
        gpu_idx = int(parts[1]) if len(parts) > 1 else 0
        if not _gpu_is_free(gpu_idx, max_used_mib=max_used_mib):
            print(
                f"[vgae] requested {device} is occupied "
                f"(used>{max_used_mib:.0f} MiB); waiting for a free GPU…",
                flush=True,
            )
            gpu_idx = wait_for_free_gpu(
                peak_ram_gib=float(peak_ram_gib),
                max_used_mib=float(max_used_mib),
                poll_sec=float(wait_poll_sec),
                label=f"{label}_rewait",
                register_waiter=register_waiter,
            )
            return f"cuda:{gpu_idx}", gpu_idx
        wait_until_launchable(
            peak_ram_gib=float(peak_ram_gib),
            gpus=(gpu_idx,),
            job_class=CLASS_GPU_TRAIN,
            timeout_sec=6 * 3600,
            poll_sec=float(wait_poll_sec),
            label=label,
        )
        return f"cuda:{gpu_idx}", gpu_idx
    return device, None


def random_split_l_hom_baseline(
    n_nodes: int,
    groups,
    *,
    ratios: tuple[float, float, float] = (3.0, 1.0, 1.0),
    seed: int = 42,
) -> dict[str, float]:
    """Hard ``L_hom`` of a size-constrained random split (seeded baseline)."""
    rng = np.random.default_rng(int(seed))
    scores = rng.normal(size=(int(n_nodes), 3))
    labels = size_constrained_assign(scores, ratios=ratios, seed=seed)
    hard = compute_l_hom(labels, groups, soft=False)
    return {
        "l_hom": float(hard["l_hom"]),
        "mean_sd_ortho": float(hard["mean_sd_ortho"]),
        "mean_sd_para": float(hard["mean_sd_para"]),
    }


def compose_objective(
    *,
    recon: torch.Tensor,
    kl: torch.Tensor,
    l_hom: torch.Tensor,
    size: torch.Tensor,
    loss_mode: str,
    epoch: int,
    beta_kl: float,
    lambda_hom: float,
    lambda_size: float,
    alpha_recon: float,
    beta_kl_max: float,
    kl_anneal_epochs: int,
    ema: EmaTermNorm | None,
) -> dict[str, Any]:
    """Compose train loss for legacy or homology_first modes (additive API)."""
    mode = str(loss_mode).lower().strip()
    if mode == "legacy":
        loss = (
            recon
            + float(beta_kl) * kl
            + float(lambda_hom) * l_hom
            + float(lambda_size) * size
        )
        return {
            "loss": loss,
            "recon_norm": None,
            "kl_norm": None,
            "beta_used": float(beta_kl),
            "ema_recon": None,
            "ema_kl": None,
            "term_recon": float(recon.detach().cpu()),
            "term_kl": float(beta_kl) * float(kl.detach().cpu()),
            "term_hom": float(lambda_hom) * float(l_hom.detach().cpu()),
            "term_size": float(lambda_size) * float(size.detach().cpu()),
        }

    if not _is_homology_train_mode(mode):
        raise ValueError(
            f"unknown loss_mode={loss_mode!r}; "
            "use legacy|homology_first|homology_robust|homology_log_balance"
        )
    if ema is None:
        raise ValueError(f"{mode} requires an EmaTermNorm instance")
    recon_n, kl_n, er, ek = ema.normalize(recon, kl)
    beta_t = kl_beta_schedule(
        int(epoch), beta_max=float(beta_kl_max), t_anneal=int(kl_anneal_epochs)
    )
    term_r = float(alpha_recon) * recon_n
    term_k = float(beta_t) * kl_n
    term_h = float(lambda_hom) * l_hom
    term_s = float(lambda_size) * size
    loss = term_r + term_k + term_h + term_s
    return {
        "loss": loss,
        "recon_norm": float(recon_n.detach().cpu()),
        "kl_norm": float(kl_n.detach().cpu()),
        "beta_used": float(beta_t),
        "ema_recon": float(er),
        "ema_kl": float(ek),
        "term_recon": float((term_r).detach().cpu()),
        "term_kl": float((term_k).detach().cpu()),
        "term_hom": float(term_h.detach().cpu()),
        "term_size": float(term_s.detach().cpu()),
    }


def run_vgae_train(
    pack: PackedGraph | Path,
    out_dir: Path,
    *,
    seed: int = 42,
    ratios: tuple[float, float, float] = (3.0, 1.0, 1.0),
    hidden_dim: int = 64,
    latent_dim: int = 32,
    lr: float = 1e-3,
    beta_kl: float = 1.0,
    lambda_hom: float = 1.0,
    lambda_size: float = 1.0,
    max_epochs: int = 200,
    min_epochs: int = 25,
    patience: int = 10,
    device: str | None = None,
    homology_table: Path | None = None,
    peak_ram_gib: float = 12.0,
    wait_poll_sec: float = 600.0,
    register_queue: bool = True,
    loss_mode: str = "legacy",
    alpha_recon: float | None = None,
    beta_kl_max: float | None = None,
    kl_anneal_epochs: int | None = None,
    gumbel_tau_start: float | None = None,
    gumbel_tau_end: float | None = None,
    gumbel_anneal_epochs: int | None = None,
    lambda_para: float | None = None,
    lambda_ortho: float | None = None,
    ema_decay: float | None = None,
    hom_max_groups: int | None = None,
    max_gpu_used_mib: float = 512.0,
    hom_agg: str | None = None,
    architecture: str = ARCH_GCN,
    gat_heads: int = 4,
    appnp_k: int = 10,
    appnp_alpha: float = 0.1,
    gcnii_layers: int = 8,
    lambda_gcl: float = 1.0,
    gcl_edge_drop: float = 0.2,
    gcl_feat_mask: float = 0.2,
    gcl_temperature: float = 0.5,
    gcl_max_nodes: int = 8192,
    early_stop_on_legacy: bool | None = None,
) -> dict[str, Any]:
    """Train classic VGAE and write ``split.csv`` + logs under ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(pack, (str, Path)):
        pack = load_packed_graph(Path(pack))
    assert_no_homology_features(pack.feature_names)
    if pack.meta.get("homology_in_encoder"):
        raise ValueError("pack meta claims homology_in_encoder=True — refused")

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))

    mode = str(loss_mode).lower().strip()
    train_agg = (
        str(hom_agg).lower().strip()
        if hom_agg is not None
        else (_hom_agg_for_mode(mode) or "mean")
    )
    # Best empirical Stage1 selection: homology_first + early-stop on legacy mean L_hom
    if early_stop_on_legacy is None:
        early_stop_on_legacy = bool(_is_homology_train_mode(mode))
    early_stop_on_legacy = bool(early_stop_on_legacy)
    use_gcl = uses_contrastive(architecture)
    hf = HOMOLOGY_FIRST_DEFAULTS
    if _is_homology_train_mode(mode):
        if alpha_recon is None:
            alpha_recon = float(hf["alpha_recon"])
        if beta_kl_max is None:
            beta_kl_max = float(hf["beta_kl_max"])
        if kl_anneal_epochs is None:
            kl_anneal_epochs = int(hf["kl_anneal_epochs"])
        if gumbel_tau_start is None:
            gumbel_tau_start = float(hf["gumbel_tau_start"])
        if gumbel_tau_end is None:
            gumbel_tau_end = float(hf["gumbel_tau_end"])
        if gumbel_anneal_epochs is None:
            gumbel_anneal_epochs = int(hf["gumbel_anneal_epochs"])
        if lambda_para is None:
            lambda_para = float(hf["lambda_para"])
        if lambda_ortho is None:
            lambda_ortho = float(hf["lambda_ortho"])
        if ema_decay is None:
            ema_decay = float(hf["ema_decay"])
        if hom_max_groups is None:
            hom_max_groups = int(hf["hom_max_groups"])
        # Homology-first default λ_hom / λ_size when caller left legacy 1.0
        if float(lambda_hom) == 1.0:
            lambda_hom = float(hf["lambda_hom"])
        if float(lambda_size) == 1.0:
            lambda_size = float(hf["lambda_size"])
    else:
        alpha_recon = float(alpha_recon if alpha_recon is not None else 1.0)
        beta_kl_max = float(beta_kl_max if beta_kl_max is not None else beta_kl)
        kl_anneal_epochs = int(kl_anneal_epochs if kl_anneal_epochs is not None else 0)
        gumbel_tau_start = float(
            gumbel_tau_start if gumbel_tau_start is not None else 1.0
        )
        gumbel_tau_end = float(gumbel_tau_end if gumbel_tau_end is not None else 1.0)
        gumbel_anneal_epochs = int(
            gumbel_anneal_epochs if gumbel_anneal_epochs is not None else 0
        )
        lambda_para = float(lambda_para if lambda_para is not None else 1.0)
        lambda_ortho = float(lambda_ortho if lambda_ortho is not None else 1.0)
        ema_decay = float(ema_decay if ema_decay is not None else 0.9)
        hom_max_groups = int(hom_max_groups if hom_max_groups is not None else 4096)

    device, gpu_idx = resolve_device(
        device,
        peak_ram_gib=float(peak_ram_gib),
        wait_poll_sec=float(wait_poll_sec),
        label=f"vgae_train:{out_dir.name}",
        max_used_mib=float(max_gpu_used_mib),
        register_waiter=register_queue,
    )

    queue_name = f"vgae_{out_dir.name}"
    if register_queue:
        append_queue_entry(
            queue_name,
            job=f"python -m src.splits.vgae --out {out_dir}",
            pid=os.getpid(),
            estimated_time="2-6h",
            job_class=CLASS_GPU_TRAIN,
            peak_ram_gib=float(peak_ram_gib),
            gpus=(gpu_idx,) if gpu_idx is not None else (),
            log=str(out_dir / "logs" / "metrics.log"),
            resources=f"device={device} n={pack.n_nodes} e={pack.n_edges} loss_mode={mode}",
        )

    try:
        groups = load_homology_groups(pack.ids, homology_table)
        write_homology_sidecar(out_dir / "pack" / "node_homology.tsv", pack.ids, groups)
        pack_out = out_dir / "pack"
        pack_out.mkdir(parents=True, exist_ok=True)
        if pack.pack_dir.resolve() != pack_out.resolve():
            import shutil

            for name in (
                "node_features.npz",
                "edges_weighted.npz",
                "ids.txt",
                "feature_meta.json",
            ):
                src = pack.pack_dir / name
                if src.is_file():
                    shutil.copy2(src, pack_out / name)

        baseline = random_split_l_hom_baseline(
            pack.n_nodes, groups, ratios=ratios, seed=int(seed)
        )
        (out_dir / "random_split_baseline.json").write_text(
            json.dumps(baseline, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"[vgae] random-split baseline L_hom={baseline['l_hom']:.6g} "
            f"(seed={seed})",
            flush=True,
        )

        dev = torch.device(device)
        x = torch.as_tensor(pack.x, dtype=torch.float32, device=dev)
        assert_no_homology_features(pack.feature_names)
        edge_index = torch.stack(
            [
                torch.as_tensor(pack.edge_u, dtype=torch.long, device=dev),
                torch.as_tensor(pack.edge_v, dtype=torch.long, device=dev),
            ],
            dim=0,
        )
        edge_weight = torch.as_tensor(pack.edge_w, dtype=torch.float32, device=dev)

        arch = str(architecture).lower().strip() or ARCH_GCN
        model = build_vgae(
            arch,
            x.size(1),
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            n_roles=3,
            gat_heads=int(gat_heads),
            appnp_k=int(appnp_k),
            appnp_alpha=float(appnp_alpha),
            gcnii_layers=int(gcnii_layers),
        ).to(dev)
        print(f"[vgae] architecture={getattr(model, 'architecture', arch)}", flush=True)
        opt = torch.optim.Adam(model.parameters(), lr=float(lr))
        target_frac = torch.as_tensor(
            role_target_fractions(ratios), dtype=torch.float32, device=dev
        )
        ema = EmaTermNorm(decay=float(ema_decay)) if _is_homology_train_mode(mode) else None

        writer = open_summary_writer(out_dir)
        tb_logger = open_tensorboard_logger(out_dir)
        ckpt_dir = out_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        log_scalar_pair(
            writer, tb_logger, "baseline/random_l_hom", baseline["l_hom"], 0
        )

        best_l_hom = float("inf")
        best_legacy_l_hom = float("inf")
        best_state: dict[str, Any] | None = None
        best_epoch = -1
        stale = 0
        epoch_rows: list[dict[str, Any]] = []

        for epoch in range(1, int(max_epochs) + 1):
            model.train()
            out = model(x, edge_index, edge_weight)
            if out["z"].size(0) != x.size(0):
                raise RuntimeError("latent/node count mismatch")

            tau = gumbel_tau_schedule(
                epoch,
                tau_start=float(gumbel_tau_start),
                tau_end=float(gumbel_tau_end),
                t_anneal=int(gumbel_anneal_epochs),
            )
            if _is_homology_train_mode(mode):
                soft_train = gumbel_softmax_roles(out["role_logits"], tau=tau, hard=False)
                soft_log = soft_role_probs(out["role_logits"])
                hom = compute_l_hom(
                    soft_train,
                    groups,
                    soft=True,
                    agg=train_agg,
                    max_groups=int(hom_max_groups),
                    subset_seed=int(seed) + int(epoch),
                    lambda_para=float(lambda_para),
                    lambda_ortho=float(lambda_ortho),
                )
                # Size constraint on softmax (stable) while L_hom uses Gumbel
                sz = size_loss(soft_log, target_frac)
                l_hom_soft = hom["l_hom"]
            else:
                soft_train = soft_role_probs(out["role_logits"])
                soft_log = soft_train
                hom = compute_l_hom(soft_train, groups, soft=True)
                l_hom_soft = hom["l_hom"]
                sz = size_loss(soft_train, target_frac)

            recon = type(model).recon_loss_neg_sample(
                out["z"], edge_index, edge_weight
            )
            kl = type(model).kl_loss(out["mu"], out["logstd"])
            compose_mode = "homology_first" if _is_homology_train_mode(mode) else mode
            composed = compose_objective(
                recon=recon,
                kl=kl,
                l_hom=l_hom_soft,
                size=sz,
                loss_mode=compose_mode,
                epoch=epoch,
                beta_kl=float(beta_kl),
                lambda_hom=float(lambda_hom),
                lambda_size=float(lambda_size),
                alpha_recon=float(alpha_recon),
                beta_kl_max=float(beta_kl_max),
                kl_anneal_epochs=int(kl_anneal_epochs),
                ema=ema,
            )
            loss = composed["loss"]
            gcl_term = None
            if use_gcl:
                g = torch.Generator()
                g.manual_seed(int(seed) + 17 * int(epoch))
                x1, ei1, ew1 = augment_graph_view(
                    x,
                    edge_index,
                    edge_weight,
                    edge_drop=float(gcl_edge_drop),
                    feat_mask=float(gcl_feat_mask),
                    generator=g,
                )
                x2, ei2, ew2 = augment_graph_view(
                    x,
                    edge_index,
                    edge_weight,
                    edge_drop=float(gcl_edge_drop),
                    feat_mask=float(gcl_feat_mask),
                    generator=g,
                )
                out1 = model(x1, ei1, ew1)
                out2 = model(x2, ei2, ew2)
                gcl_term = info_nce_pairwise(
                    out1["z"],
                    out2["z"],
                    temperature=float(gcl_temperature),
                    max_nodes=int(gcl_max_nodes),
                    generator=g,
                )
                loss = loss + float(lambda_gcl) * gcl_term
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            model.eval()
            with torch.no_grad():
                out_e = model(x, edge_index, edge_weight)
                scores = soft_role_probs(out_e["role_logits"]).cpu().numpy()
            labels = size_constrained_assign(scores, ratios=ratios, seed=seed + epoch)
            from src.splits.vgae.homology_loss import (
                evaluate_split_all_aggs,
                sd_group_balance_report,
            )
            hard = compute_l_hom(
                labels,
                groups,
                soft=False,
                agg=train_agg if _is_homology_train_mode(mode) else "mean",
                max_groups=8192,
                subset_seed=int(seed) + int(epoch),
            )
            hard_legacy = compute_l_hom(
                labels, groups, soft=False, agg="mean", max_groups=8192
            )
            balance = sd_group_balance_report(
                labels, groups, max_groups=8192, seed=int(seed) + int(epoch)
            )
            l_hom_hard = float(hard["l_hom"])
            l_hom_legacy = float(hard_legacy["l_hom"])

            mag_r = abs(float(composed["term_recon"]))
            mag_k = abs(float(composed["term_kl"]))
            mag_h = abs(float(composed["term_hom"]))
            mag_s = abs(float(composed["term_size"]))
            mag_sum = mag_r + mag_k + mag_h + mag_s + 1e-12
            hom_share = mag_h / mag_sum

            row = {
                "epoch": epoch,
                "loss": float(loss.detach().cpu()),
                "recon": float(recon.detach().cpu()),
                "kl": float(kl.detach().cpu()),
                "l_hom_soft": float(l_hom_soft.detach().cpu()),
                "size": float(sz.detach().cpu()),
                "l_hom_hard": l_hom_hard,
                "l_hom_legacy": l_hom_legacy,
                "gcl": float(gcl_term.detach().cpu()) if gcl_term is not None else None,
                "hom_agg": train_agg,
                "early_stop_on_legacy": early_stop_on_legacy,
                "mean_sd_ortho": float(hard["mean_sd_ortho"]),
                "mean_sd_para": float(hard["mean_sd_para"]),
                "legacy_mean_sd_ortho": float(hard_legacy["mean_sd_ortho"]),
                "legacy_mean_sd_para": float(hard_legacy["mean_sd_para"]),
                "sd_balance": balance,
                "recon_norm": composed["recon_norm"],
                "kl_norm": composed["kl_norm"],
                "beta_kl": composed["beta_used"],
                "gumbel_tau": float(tau) if _is_homology_train_mode(mode) else None,
                "hom_grad_share": float(hom_share),
                "ema_recon": composed["ema_recon"],
                "ema_kl": composed["ema_kl"],
            }
            epoch_rows.append(row)
            _write_epoch_logs(out_dir, epoch_rows)

            log_scalar_pair(writer, tb_logger, "train/loss", row["loss"], epoch)
            log_scalar_pair(writer, tb_logger, "train/recon", row["recon"], epoch)
            log_scalar_pair(writer, tb_logger, "train/kl", row["kl"], epoch)
            log_scalar_pair(writer, tb_logger, "train/l_hom_soft", row["l_hom_soft"], epoch)
            if row["recon_norm"] is not None:
                log_scalar_pair(
                    writer, tb_logger, "train/recon_norm", row["recon_norm"], epoch
                )
                log_scalar_pair(writer, tb_logger, "train/kl_norm", row["kl_norm"], epoch)
                log_scalar_pair(
                    writer, tb_logger, "train/hom_grad_share", row["hom_grad_share"], epoch
                )
                log_scalar_pair(writer, tb_logger, "train/beta_kl", row["beta_kl"], epoch)
                log_scalar_pair(
                    writer, tb_logger, "train/gumbel_tau", float(row["gumbel_tau"]), epoch
                )
            log_scalar_pair(writer, tb_logger, "validation/l_hom", l_hom_hard, epoch)
            log_scalar_pair(
                writer, tb_logger, "validation/l_hom_legacy", l_hom_legacy, epoch
            )
            log_scalar_pair(
                writer, tb_logger, "validation/mean_sd_ortho", row["mean_sd_ortho"], epoch
            )
            log_scalar_pair(
                writer, tb_logger, "validation/mean_sd_para", row["mean_sd_para"], epoch
            )

            # Selection metric: legacy mean under homology_first (empirically best)
            select_metric = l_hom_legacy if early_stop_on_legacy else l_hom_hard
            improved = select_metric < best_l_hom - 1e-6
            if improved:
                best_l_hom = float(select_metric)
                best_legacy_l_hom = l_hom_legacy
                best_epoch = epoch
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
                torch.save(
                    {
                        "epoch": epoch,
                        "model": best_state,
                        "l_hom": best_l_hom,
                        "l_hom_legacy": best_legacy_l_hom,
                        "hom_agg": train_agg,
                    },
                    ckpt_dir / "best.pt",
                )
                stale = 0
            else:
                if epoch >= int(min_epochs):
                    stale += 1

            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "l_hom": l_hom_hard,
                    "l_hom_legacy": l_hom_legacy,
                },
                ckpt_dir / "last.pt",
            )

            print(
                f"[vgae] epoch={epoch} loss={row['loss']:.5g} "
                f"l_hom_hard={l_hom_hard:.5g} legacy={l_hom_legacy:.5g} "
                f"sd_o={row['mean_sd_ortho']:.5g} sd_p={row['mean_sd_para']:.5g} "
                f"best={best_l_hom:.5g}@{best_epoch} stale={stale} "
                f"mode={mode} agg={train_agg} "
                f"sel={'legacy' if early_stop_on_legacy else 'train'}"
                + (f" gcl={float(gcl_term.detach().cpu()):.4g}" if gcl_term is not None else ""),
                flush=True,
            )

            if epoch >= int(min_epochs) and stale >= int(patience):
                print(
                    f"[vgae] early stop at epoch={epoch} "
                    f"(min_epochs={min_epochs}, patience={patience})",
                    flush=True,
                )
                break

        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            out_f = model(x, edge_index, edge_weight)
            scores = soft_role_probs(out_f["role_logits"]).cpu().numpy()
            z = out_f["z"].cpu().numpy()
        labels = size_constrained_assign(scores, ratios=ratios, seed=seed)
        from src.splits.vgae.homology_loss import (
            evaluate_split_all_aggs,
            sd_group_balance_report,
        )

        hard = compute_l_hom(
            labels,
            groups,
            soft=False,
            agg=train_agg if _is_homology_train_mode(mode) else "mean",
        )
        all_aggs = evaluate_split_all_aggs(labels, groups)
        balance_final = sd_group_balance_report(labels, groups, max_groups=None, seed=seed)
        rows = assignment_rows(pack.ids, labels, fold_prefix="vgae")
        split_csv = assignment_rows_to_split_csv(rows, out_dir)
        np.savez_compressed(out_dir / "latents.npz", z=z, scores=scores)
        counts = {r: labels.count(r) for r in ROLE_ORDER}
        meta = {
            "device": device,
            "seed": seed,
            "ratios": list(ratios),
            "loss_mode": mode,
            "architecture": getattr(model, "architecture", arch),
            "hom_agg": train_agg,
            "early_stop_on_legacy": early_stop_on_legacy,
            "lambda_gcl": float(lambda_gcl) if use_gcl else None,
            "best_epoch": best_epoch,
            "best_l_hom": best_l_hom,
            "best_l_hom_legacy": best_legacy_l_hom,
            "final_l_hom": float(hard["l_hom"]),
            "final_l_hom_legacy": float(all_aggs["mean"]["l_hom"]),
            "final_mean_sd_ortho": float(hard["mean_sd_ortho"]),
            "final_mean_sd_para": float(hard["mean_sd_para"]),
            "random_baseline_l_hom": baseline["l_hom"],
            "all_aggs": all_aggs,
            "sd_balance": balance_final,
            "counts": counts,
            "n_nodes": pack.n_nodes,
            "n_edges": pack.n_edges,
            "k": pack.k,
            "min_epochs": min_epochs,
            "patience": patience,
            "max_epochs": max_epochs,
            "lambda_hom": float(lambda_hom),
            "lambda_size": float(lambda_size),
            "alpha_recon": float(alpha_recon),
            "beta_kl_max": float(beta_kl_max),
            "homology_in_encoder": False,
            "split_csv": str(split_csv),
        }
        if _is_homology_train_mode(mode):
            meta["lambda_para"] = float(lambda_para)
            meta["lambda_ortho"] = float(lambda_ortho)
        (out_dir / "train_meta.json").write_text(
            json.dumps(meta, indent=2, default=str) + "\n", encoding="utf-8"
        )
        (out_dir / "legacy_eval.json").write_text(
            json.dumps(
                {
                    "best_epoch": best_epoch,
                    "train_agg": train_agg,
                    "loss_mode": mode,
                    "best_train_l_hom": best_l_hom,
                    "best_legacy_l_hom": best_legacy_l_hom,
                    "final_all_aggs": all_aggs,
                    "sd_balance": balance_final,
                    "note": (
                        "Compare best_legacy_l_hom / final_all_aggs['mean'] to "
                        "homology_first / legacy Stage1 baselines; sd_balance "
                        "shows whether mean SD is outlier-driven "
                        "(p90_over_median, top5pct_mass_frac)."
                    ),
                },
                indent=2,
                default=str,
            )
            + "\n",
            encoding="utf-8",
        )
        close_dual(writer, tb_logger)
        if register_queue:
            _append_status(
                queue_name,
                "COMPLETED",
                note=(
                    f"best_l_hom={best_l_hom:.6g} "
                    f"legacy={best_legacy_l_hom:.6g} agg={train_agg}"
                ),
            )
        return meta
    except Exception as exc:
        if register_queue:
            _append_status(queue_name, "FAILED", note=str(exc)[:500])
        raise
