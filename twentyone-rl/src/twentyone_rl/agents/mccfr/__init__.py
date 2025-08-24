"""MCCFR (Monte Carlo Counterfactual Regret Minimization) agents."""

from .deep import DeepMCCFR
from .tabular import MCCFR

__all__ = ["MCCFR", "DeepMCCFR"]
