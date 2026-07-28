"""Universal GigaMario pipeline stages (I/O contracts)."""
from __future__ import annotations

from importlib import import_module

__all__ = (
    "id_gen", "id_rule", "parse_target", "adapt", "parse_data",
    "split_predict", "split_materialize", "train", "train_viz", "adversarial",
)

_MODULE_NAMES = {name: name for name in __all__}
_MODULE_NAMES["split_materialize"] = "split"


def __getattr__(name: str):
    """Lazily expose stage modules without preloading a ``python -m`` target."""
    try:
        module_name = _MODULE_NAMES[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = import_module(f"{__name__}.{module_name}")
    globals()[name] = module
    return module
