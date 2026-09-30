"""Streaming, flat-memory SCNet source separation on Apple silicon via Core ML."""

from .core import OVERLAP, SAMPLE_RATE, SEGMENT, SOURCES
from .model import MODEL_SHA256, MODEL_URL, ensure_model
from .pipeline import separate, separate_source

__version__ = "0.1.0"

__all__ = [
    "MODEL_SHA256",
    "MODEL_URL",
    "OVERLAP",
    "SAMPLE_RATE",
    "SEGMENT",
    "SOURCES",
    "ensure_model",
    "separate",
    "separate_source",
    "__version__",
]
