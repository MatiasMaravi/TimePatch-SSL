"""Capacity-matched model definitions for causal attribution ablations.

This module provides the Standard-Matched architecture that preserves the 1D linear
patch embedding while matching the parameter capacity of the Hierarchical encoder.

All classes here are completely self-contained in this folder, ensuring that
the original codebase in `src/` remains completely unmodified.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from TimeMAE.model.TimeMAE import Encoder
    from TimeMAE.model.layers import PositionalEmbedding
except ImportError as e:
    raise ImportError("TimeMAE must be available on PYTHONPATH.") from e

from src.models import PatchEmbed1D, count_trainable_parameters


class RobustTimeMAEPatchEmbed(nn.Module):
    """Hierarchical patch embedder with safe positional embedding slicing."""
    def __init__(self, in_ch=3, patch_len=8, emb_dim=128, max_patches=5000):
        super().__init__()
        self.patch_len = patch_len
        self.input_projection = nn.Conv1d(in_ch, emb_dim, patch_len, stride=patch_len)
        self.position = PositionalEmbedding(max_patches, emb_dim)
        class Args: pass
        a = Args()
        a.d_model = emb_dim
        a.attn_heads = 4
        a.layers = 2
        a.dropout = 0.1
        a.enable_res_parameter = 1
        self.encoder = Encoder(a)

    def forward(self, x):
        T2 = (x.shape[1] // self.patch_len) * self.patch_len
        x = self.input_projection(x[:, :T2].transpose(1, 2)).transpose(1, 2).contiguous()
        # Safe slicing matches notebook line 801
        pos = self.position(x)
        x = x + pos[:, :x.size(1), :]
        return self.encoder(x)


class AblationTimePatchTransformer(nn.Module):
    """Base Transformer for the ablation study supporting standard, hierarchical, and matched modes."""
    def __init__(
        self,
        in_ch=3,
        patch_len=8,
        emb_dim=128,
        depth=4,
        heads=4,
        dropout=0.1,
        num_classes=2,
        hierarchical=False,
    ):
        super().__init__()
        self.in_ch = in_ch
        self.patch_len = patch_len
        self.emb_dim = emb_dim
        self.depth = depth
        self.hierarchical = hierarchical

        self.patch = (
            RobustTimeMAEPatchEmbed(in_ch, patch_len, emb_dim)
            if hierarchical
            else PatchEmbed1D(in_ch, patch_len, emb_dim)
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_emb = nn.Parameter(torch.zeros(1, 512, emb_dim))
        layer = nn.TransformerEncoderLayer(
            emb_dim, heads, emb_dim * 4, dropout, activation="gelu", batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, depth)
        self.norm = nn.LayerNorm(emb_dim)
        self.ssl_head = nn.Sequential(
            nn.Linear(emb_dim, emb_dim), nn.GELU(), nn.Linear(emb_dim, 64)
        )
        self.cls_head = nn.Linear(emb_dim, num_classes)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

    def forward_features(self, x):
        x = self.patch(x)
        B, N, _ = x.shape
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], 1)
        return self.norm(self.encoder(x + self.pos_emb[:, : N + 1]))[:, 0]

    def forward_ssl(self, x):
        return F.normalize(self.ssl_head(self.forward_features(x)), dim=-1)

    def forward(self, x):
        return self.cls_head(self.forward_features(x))


def calculate_matched_depth(
    base_depth: int = 4,
    emb_dim: int = 128,
    in_ch: int = 3,
    patch_len: int = 8,
    heads: int = 4,
    dropout: float = 0.1,
    num_classes: int = 2,
    search_range: int = 15,
) -> tuple[int, int, int, float]:
    """Find the global Transformer depth that matches Hierarchical parameter capacity."""
    hierarchical_model = AblationTimePatchTransformer(
        in_ch=in_ch,
        patch_len=patch_len,
        emb_dim=emb_dim,
        depth=base_depth,
        heads=heads,
        dropout=dropout,
        num_classes=num_classes,
        hierarchical=True,
    )
    h_params = count_trainable_parameters(hierarchical_model)

    best_depth = base_depth
    best_diff = float("inf")
    best_matched_params = 0

    for test_depth in range(base_depth, base_depth + search_range + 1):
        test_model = AblationTimePatchTransformer(
            in_ch=in_ch,
            patch_len=patch_len,
            emb_dim=emb_dim,
            depth=test_depth,
            heads=heads,
            dropout=dropout,
            num_classes=num_classes,
            hierarchical=False,
        )
        s_params = count_trainable_parameters(test_model)
        diff = abs(s_params - h_params)
        if diff < best_diff:
            best_diff = diff
            best_depth = test_depth
            best_matched_params = s_params

    rel_diff = ((best_matched_params - h_params) / h_params) * 100.0
    return best_depth, h_params, best_matched_params, rel_diff


class StandardMatchedTransformer(AblationTimePatchTransformer):
    """Standard TimePatch Transformer with capacity matched to the Hierarchical variant.

    Enforces hierarchical=False (direct linear patch projection) and automatically
    scales the global Transformer depth to achieve parameter parity with the
    Hierarchical counterpart under the same configuration.
    """

    def __init__(
        self,
        in_ch: int = 3,
        patch_len: int = 8,
        emb_dim: int = 128,
        depth: int = 4,
        heads: int = 4,
        dropout: float = 0.1,
        num_classes: int = 2,
        explicit_matched_depth: int | None = None,
    ):
        self.base_depth = depth
        if explicit_matched_depth is not None:
            self.matched_depth = explicit_matched_depth
        else:
            self.matched_depth, _, _, _ = calculate_matched_depth(
                base_depth=depth,
                emb_dim=emb_dim,
                in_ch=in_ch,
                patch_len=patch_len,
                heads=heads,
                dropout=dropout,
                num_classes=num_classes,
            )

        super().__init__(
            in_ch=in_ch,
            patch_len=patch_len,
            emb_dim=emb_dim,
            depth=self.matched_depth,
            heads=heads,
            dropout=dropout,
            num_classes=num_classes,
            hierarchical=False,
        )


def build_model(
    variant: str,
    in_ch: int = 3,
    patch_len: int = 8,
    emb_dim: int = 128,
    depth: int = 4,
    heads: int = 4,
    dropout: float = 0.1,
    num_classes: int = 2,
    **kwargs,
) -> nn.Module:
    """Factory function to build any of the three comparative architectures."""
    v = variant.lower().replace("-", "_")
    if v == "standard":
        return AblationTimePatchTransformer(
            in_ch=in_ch,
            patch_len=patch_len,
            emb_dim=emb_dim,
            depth=depth,
            heads=heads,
            dropout=dropout,
            num_classes=num_classes,
            hierarchical=False,
            **kwargs,
        )
    elif v == "hierarchical":
        return AblationTimePatchTransformer(
            in_ch=in_ch,
            patch_len=patch_len,
            emb_dim=emb_dim,
            depth=depth,
            heads=heads,
            dropout=dropout,
            num_classes=num_classes,
            hierarchical=True,
            **kwargs,
        )
    elif v in ("standard_matched", "matched"):
        return StandardMatchedTransformer(
            in_ch=in_ch,
            patch_len=patch_len,
            emb_dim=emb_dim,
            depth=depth,
            heads=heads,
            dropout=dropout,
            num_classes=num_classes,
            **kwargs,
        )
    else:
        raise ValueError(
            f"Unknown architecture variant: {variant}. "
            f"Must be one of: 'standard', 'hierarchical', 'standard_matched'."
        )
