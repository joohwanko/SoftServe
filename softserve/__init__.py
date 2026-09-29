"""SoftSERVE: Secant Equation Regularized with Variational Entropy."""

from .optim import SoftServeDiag, SoftServeKron
from .secants import IntervalSecant

__all__ = ["SoftServeDiag", "SoftServeKron", "IntervalSecant"]
