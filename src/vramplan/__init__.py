"""vramplan — estimate GPU VRAM for LLM inference."""

from .estimate import MemoryBreakdown, estimate, fits, max_context
from .specs import ModelSpec, lookup

__version__ = "0.1.0"

__all__ = [
    "ModelSpec",
    "MemoryBreakdown",
    "estimate",
    "fits",
    "max_context",
    "lookup",
    "__version__",
]
