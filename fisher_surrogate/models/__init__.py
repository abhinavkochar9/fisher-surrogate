"""Model backends.

``sklearn_models`` is always importable. ``torch_models`` requires PyTorch.
"""
from . import sklearn_models

try:  # torch is optional at import time
    from . import torch_models  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    torch_models = None

__all__ = ["sklearn_models", "torch_models"]
