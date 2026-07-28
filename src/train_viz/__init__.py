"""Publication-quality training visualization (`python -m src.train_viz`).

Static figures: cnsplots. Interactive: Altair (HTML + Vega-Lite).
"""

from .viz import main

__all__ = ["main"]
