"""LegNet layer embedding extraction (hooks + AGCT one-hot + RC average)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F

from src.embed import DEFAULT_LAYERS, LAYER_DIMS, SEQ_LEN
from src.embed.discover import LegNetRun
from src.embed.protocol import EmbeddingExtractor

# human_legnet utils.CODES: A=0, G=1, C=2, T=3  → channel order AGCT
_BASE_TO_IDX = {"A": 0, "G": 1, "C": 2, "T": 3, "N": -1}

DEFAULT_VENDOR = Path("software/human_legnet")


def import_legnet_vendor(vendor: Path | None = None) -> Path:
    from src.legnet_core_launcher import _patch_numpy_legacy_aliases

    _patch_numpy_legacy_aliases()
    vendor = (vendor or DEFAULT_VENDOR).resolve()
    if not vendor.is_dir():
        raise FileNotFoundError(f"human_legnet vendor missing: {vendor}")
    if str(vendor) not in sys.path:
        sys.path.insert(0, str(vendor))
    return vendor


def encode_agct(seq: str, *, seq_len: int = SEQ_LEN) -> torch.Tensor:
    """One-hot ``[4, L]`` float32; N → 0.25 on all channels (vendor Seq2Tensor)."""
    if len(seq) != seq_len:
        raise ValueError(f"seq len {len(seq)} != {seq_len}")
    x = torch.zeros(4, seq_len, dtype=torch.float32)
    for i, ch in enumerate(seq.upper()):
        idx = _BASE_TO_IDX.get(ch, -1)
        if idx < 0:
            x[:, i] = 0.25
        else:
            x[idx, i] = 1.0
    return x


def reverse_complement_onehot(x: torch.Tensor) -> torch.Tensor:
    """RC for AGCT channel order: flip channels + flip length."""
    if x.dim() == 2:
        return x.flip(0).flip(1)
    if x.dim() == 3:
        return x.flip(1).flip(2)
    raise ValueError(f"expected [4,L] or [B,4,L], got {tuple(x.shape)}")


def build_default_legnet(*, in_ch: int = 4) -> torch.nn.Module:
    """Untrained LegNet with project defaults (for unit tests)."""
    import_legnet_vendor()
    from model import LegNet

    return LegNet(
        in_ch=in_ch,
        stem_ch=64,
        stem_ks=11,
        ef_ks=9,
        ef_block_sizes=[80, 96, 112, 128],
        pool_sizes=[2, 2, 2, 2],
        resize_factor=4,
    )


def load_lit_model(
    run: LegNetRun, *, map_location: str = "cpu", vendor: Path | None = None
) -> Any:
    import_legnet_vendor(vendor)
    from trainer import LitModel
    from training_config import TrainingConfig

    cfg = TrainingConfig.from_json(run.config_json, training=False)
    return LitModel.load_from_checkpoint(
        str(run.ckpt_path), tr_cfg=cfg, map_location=map_location
    )


def _mean_max_pool(feat: torch.Tensor) -> torch.Tensor:
    """``[B, C, L]`` → ``[B, 2C]`` mean‖max."""
    return torch.cat([feat.mean(dim=-1), feat.amax(dim=-1)], dim=-1)


class LegNetLayerExtractor:
    """Hook-based multi-layer extractor implementing :class:`EmbeddingExtractor`."""

    name = "legnet"

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        device: torch.device | str = "cpu",
        layers: tuple[str, ...] = DEFAULT_LAYERS,
    ) -> None:
        self.model = model
        self.device = torch.device(device)
        self.layers = tuple(layers)
        unknown = set(self.layers) - set(LAYER_DIMS)
        if unknown:
            raise ValueError(f"unknown layers: {sorted(unknown)}")
        self.model.to(self.device)
        self.model.eval()
        self._feats: dict[str, torch.Tensor] = {}
        self._hooks: list[Any] = []
        self._install_hooks()

    def layer_dims(self) -> dict[str, int]:
        return {k: LAYER_DIMS[k] for k in self.layers}

    def close(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def __enter__(self) -> LegNetLayerExtractor:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _install_hooks(self) -> None:
        m = self.model
        need_stages = any(k.startswith("stage") for k in self.layers)
        if need_stages:

            def stage_hook(idx: int) -> Callable[..., None]:
                def _hook(_module: Any, _inp: Any, out: torch.Tensor) -> None:
                    self._feats[f"_stage{idx}"] = out.detach()

                return _hook

            for i in range(4):
                self._hooks.append(m.main[i].register_forward_hook(stage_hook(i)))

        if "pooled" in self.layers:

            def pooled_pre(_module: Any, inp: tuple[torch.Tensor, ...]) -> None:
                self._feats["_pooled"] = inp[0].detach()

            self._hooks.append(m.head[0].register_forward_pre_hook(pooled_pre))

        if "head_h" in self.layers:

            def head_h_hook(_module: Any, _inp: Any, out: torch.Tensor) -> None:
                self._feats["_head_h"] = out.detach()

            self._hooks.append(m.head[2].register_forward_hook(head_h_hook))

    def _pack(self, pred: torch.Tensor) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        if "stage0" in self.layers:
            out["stage0"] = _mean_max_pool(self._feats["_stage0"])
        if "stage1_2" in self.layers:
            a = _mean_max_pool(self._feats["_stage1"])
            b = _mean_max_pool(self._feats["_stage2"])
            out["stage1_2"] = torch.cat([a, b], dim=-1)
        if "pooled" in self.layers:
            out["pooled"] = self._feats["_pooled"]
        if "head_h" in self.layers:
            out["head_h"] = self._feats["_head_h"]
        if "pred" in self.layers:
            out["pred"] = pred.detach().reshape(-1, 1)
        return out

    @torch.no_grad()
    def forward_once(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        self._feats.clear()
        x = x.to(self.device)
        pred = self.model(x)
        return self._pack(pred)

    @torch.no_grad()
    def extract_tensor(
        self, x: torch.Tensor, *, rc_average: bool = True
    ) -> dict[str, torch.Tensor]:
        """``x``: ``[B, 4, L]``. Optionally average with RC."""
        fwd = self.forward_once(x)
        if not rc_average:
            return fwd
        rev = self.forward_once(reverse_complement_onehot(x))
        return {k: 0.5 * (fwd[k] + rev[k]) for k in fwd}

    def extract_batch(
        self, sequences: list[str], *, layers: tuple[str, ...] | None = None
    ) -> dict[str, np.ndarray]:
        if layers is not None and tuple(layers) != self.layers:
            raise ValueError("layers must match extractor construction; rebuild to change")
        if not sequences:
            return {k: np.zeros((0, LAYER_DIMS[k]), dtype=np.float32) for k in self.layers}
        x = torch.stack([encode_agct(s) for s in sequences], dim=0)
        tensors = self.extract_tensor(x, rc_average=True)
        return {k: tensors[k].float().cpu().numpy().astype(np.float32) for k in self.layers}


def pooled_manual(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Reference: adaptive_avg_pool after mapper (matches pre-head[0])."""
    with torch.no_grad():
        h = model.stem(x)
        h = model.main(h)
        h = model.mapper(h)
        h = F.adaptive_avg_pool1d(h, 1).squeeze(-1)
    return h


# Protocol structural check helper
def as_extractor(obj: LegNetLayerExtractor) -> EmbeddingExtractor:
    return obj
