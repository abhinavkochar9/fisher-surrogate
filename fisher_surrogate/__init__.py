"""fisher_surrogate -- reproducible implementation for

    Farmer, Kochar, Lee. "Beyond Pointwise Error: Mechanism-Preserving Neural
    Surrogates for Fisher-Regularized Probability Flows." Neurocomputing
    (NEUCOM-D-26-14165), under revision.
"""
from . import data, dynamics, latex, metrics

__version__ = "0.2.0"
__all__ = ["data", "dynamics", "latex", "metrics"]
