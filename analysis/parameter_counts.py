"""Calculate trainable and total parameter counts for reported architectures."""
from __future__ import annotations
import json
from pathlib import Path
from src.models import TimePatchTransformer


def count_parameters(model):
    return {
        "total_parameters": int(sum(p.numel() for p in model.parameters())),
        "trainable_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "encoder_parameters": int(sum(
            p.numel() for name, p in model.named_parameters()
            if not name.startswith("cls_head") and not name.startswith("ssl_head")
        )),
        "classification_head_parameters": int(sum(
            p.numel() for name, p in model.named_parameters()
            if name.startswith("cls_head")
        )),
        "projection_head_parameters": int(sum(
            p.numel() for name, p in model.named_parameters()
            if name.startswith("ssl_head")
        )),
    }


def compare_architectures(model_kwargs=None):
    model_kwargs = model_kwargs or {}
    standard = TimePatchTransformer(hierarchical=False, **model_kwargs)
    hierarchical = TimePatchTransformer(hierarchical=True, **model_kwargs)
    s = count_parameters(standard)
    h = count_parameters(hierarchical)
    return {
        "standard": s,
        "hierarchical": h,
        "hierarchical_minus_standard_trainable": (
            h["trainable_parameters"] - s["trainable_parameters"]
        ),
        "trainable_parameter_ratio_hierarchical_to_standard": (
            h["trainable_parameters"] / s["trainable_parameters"]
        ),
    }


def save_parameter_counts(result, output):
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
