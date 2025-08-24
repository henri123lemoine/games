"""Agents package."""

from .mccfr import MCCFR, DeepMCCFR
from .tabular_cfr import TabularCFR

__all__ = ["MCCFR", "DeepMCCFR", "TabularCFR"]
