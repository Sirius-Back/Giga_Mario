"""Thin re-export — canonical metrics live in ``src.metrics_logging``."""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[4]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.metrics_logging import (  # noqa: F401
    EPOCH_METRIC_KEYS,
    build_regression_metrics,
    compute_epoch_regression_metrics,
    format_epoch_log,
    genewise_pearson,
    samplewise_pearson,
)
