"""Ablation study module for capacity-matched (Standard-Matched) evaluations.

Addresses Reviewer Point 5: Novelty and Causal Attribution by isolating
the inductive bias of hierarchical temporal patching from model capacity.
"""

from .models import (
    StandardMatchedTransformer,
    calculate_matched_depth,
    build_model,
)

__all__ = [
    "StandardMatchedTransformer",
    "calculate_matched_depth",
    "build_model",
]
