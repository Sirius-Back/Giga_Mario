"""Caduceus layer embedding extraction (HF + RCPS-aware pooling)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from src.embed.protocol import EmbeddingExtractor

# RCPS hidden is 2*d_model; after fwd/rc split+avg → d_model=256
CADUCEUS_LAYER_DIMS: dict[str, int] = {
    "stage0": 512,  # HS[1] mean‖max after RCPS avg
    "stage1_2": 512,  # HS[8] mean‖max
    "pooled": 256,  # HS[-1] mean (pre-score)
    "head_h": 512,  # HS[-1] mean‖max
    "pred": 1,
}

CADUCEUS_DEFAULT_LAYERS: tuple[str, ...] = (
    "stage0",
    "stage1_2",
    "pooled",
    "head_h",
    "pred",
)

# Which hidden_states index (0=embed) for early / mid
_STAGE0_IDX = 1
_STAGE_MID_IDX = 8


def _mean_max_pool(x: torch.Tensor) -> torch.Tensor:
    """``[B, L, D]`` → ``[B, 2D]``."""
    return torch.cat([x.mean(dim=1), x.amax(dim=1)], dim=-1)


def rcps_to_strand_avg(hidden: torch.Tensor, d_model: int) -> torch.Tensor:
    """Convert RCPS ``[B, L, 2D]`` → strand-averaged ``[B, L, D]`` (Caduceus forward)."""
    if hidden.shape[-1] == d_model:
        return hidden
    if hidden.shape[-1] != 2 * d_model:
        raise ValueError(
            f"expected last dim {d_model} or {2 * d_model}, got {hidden.shape[-1]}"
        )
    fwd = hidden[..., :d_model]
    rc = torch.flip(hidden[..., d_model:], dims=[1, 2])
    return 0.5 * (fwd + rc)


def load_caduceus_model(
    model_dir: Path, *, device: str = "cuda"
) -> tuple[Any, Any, int]:
    """Load fine-tuned CaduceusForSequenceClassification + tokenizer.

    Returns ``(model, tokenizer, max_length_default)``.
    """
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_dir = Path(model_dir)
    if not (model_dir / "model.safetensors").is_file() and not (
        model_dir / "pytorch_model.bin"
    ).is_file():
        raise FileNotFoundError(f"No weights in {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir), trust_remote_code=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_dir),
        trust_remote_code=True,
        num_labels=1,
        problem_type="regression",
    )
    model.to(device)
    model.eval()
    return model, tokenizer, 256


class CaduceusLayerExtractor:
    """Multi-layer extractor implementing :class:`EmbeddingExtractor`."""

    name = "caduceus"

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        device: str = "cuda",
        max_length: int = 256,
        layers: tuple[str, ...] = CADUCEUS_DEFAULT_LAYERS,
        amp: bool = True,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = int(max_length)
        self.layers = tuple(layers)
        self.amp = bool(amp) and str(device).startswith("cuda")
        self.d_model = int(getattr(model.config, "d_model", 256))
        for k in self.layers:
            if k not in CADUCEUS_LAYER_DIMS:
                raise ValueError(f"unknown Caduceus layer {k!r}")

    def layer_dims(self) -> dict[str, int]:
        return {k: CADUCEUS_LAYER_DIMS[k] for k in self.layers}

    def __enter__(self) -> CaduceusLayerExtractor:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    @torch.inference_mode()
    def extract_batch(
        self, sequences: list[str], *, layers: tuple[str, ...] | None = None
    ) -> dict[str, np.ndarray]:
        want = tuple(layers) if layers is not None else self.layers
        if not sequences:
            return {
                k: np.zeros((0, CADUCEUS_LAYER_DIMS[k]), dtype=np.float32)
                for k in want
            }
        enc = self.tokenizer(
            sequences,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.cuda.amp.autocast(enabled=self.amp):
            out = self.model(**enc, output_hidden_states=True)
        hs = out.hidden_states  # tuple len n_layer+1
        logits = out.logits.squeeze(-1)
        if logits.ndim == 0:
            logits = logits.unsqueeze(0)

        def _avg(h: torch.Tensor) -> torch.Tensor:
            return rcps_to_strand_avg(h.float(), self.d_model)

        feats: dict[str, np.ndarray] = {}
        for k in want:
            if k == "stage0":
                x = _avg(hs[_STAGE0_IDX])
                v = _mean_max_pool(x)
            elif k == "stage1_2":
                idx = min(_STAGE_MID_IDX, len(hs) - 1)
                x = _avg(hs[idx])
                v = _mean_max_pool(x)
            elif k == "pooled":
                x = _avg(hs[-1])
                v = x.mean(dim=1)
            elif k == "head_h":
                x = _avg(hs[-1])
                v = _mean_max_pool(x)
            elif k == "pred":
                v = logits.float().reshape(-1, 1)
            else:
                raise ValueError(k)
            arr = v.detach().float().cpu().numpy().astype(np.float32)
            if arr.shape[1] != CADUCEUS_LAYER_DIMS[k]:
                raise RuntimeError(
                    f"layer {k}: got dim {arr.shape[1]} "
                    f"expected {CADUCEUS_LAYER_DIMS[k]}"
                )
            feats[k] = arr
        return feats


def as_extractor(obj: CaduceusLayerExtractor) -> EmbeddingExtractor:
    return obj
