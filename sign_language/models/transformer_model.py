"""
Transformer-based model for sign language recognition.

Architecture
------------
  Input: (batch, T, input_dim)  -- MediaPipe keypoint sequences
    │
    ├── Linear projection to model_dim
    │
    ├── Positional encoding (learnable or sinusoidal)
    │
    ├── N × Transformer encoder layers (multi-head self-attention + FFN)
    │
    ├── [CLS] token pooling  or  mean pooling
    │
    └── Classification head  →  (batch, num_classes)

Usage
-----
    from sign_language.models.transformer_model import SignTransformer

    model = SignTransformer(
        input_dim=258,
        model_dim=256,
        num_heads=8,
        num_layers=4,
        num_classes=100,
        dropout=0.1,
        max_seq_len=64,
    )
    # x: (B, T, 258)
    logits = model(x)   # (B, 100)
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Positional Encodings
# ---------------------------------------------------------------------------


class SinusoidalPositionalEncoding(nn.Module):
    """
    Fixed sinusoidal positional encoding (Vaswani et al., 2017).

    Args:
        model_dim: Embedding dimension.
        max_seq_len: Maximum sequence length to pre-compute.
        dropout: Dropout applied after adding the encoding.
    """

    def __init__(
        self,
        model_dim: int,
        max_seq_len: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_seq_len, model_dim)
        position = torch.arange(max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, model_dim, 2, dtype=torch.float)
            * (-math.log(10000.0) / model_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: model_dim // 2])
        pe = pe.unsqueeze(0)  # (1, max_seq_len, model_dim)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, model_dim)

        Returns:
            (B, T, model_dim)
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):
    """
    Learnable positional embedding (BERT-style).

    Args:
        model_dim: Embedding dimension.
        max_seq_len: Maximum sequence length.
        dropout: Dropout after adding.
    """

    def __init__(
        self,
        model_dim: int,
        max_seq_len: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(max_seq_len, model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, model_dim)

        Returns:
            (B, T, model_dim)
        """
        T = x.size(1)
        positions = torch.arange(T, device=x.device).unsqueeze(0)  # (1, T)
        x = x + self.embedding(positions)
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Main Model
# ---------------------------------------------------------------------------


class SignTransformer(nn.Module):
    """
    Transformer encoder classifier for sign language recognition.

    A [CLS] token is prepended to the sequence; the transformer output at
    the CLS position is used for classification.

    Args:
        input_dim: Feature dimension per timestep (e.g. 258).
        model_dim: Internal embedding dimension (must be divisible by
            ``num_heads``).
        num_heads: Number of self-attention heads.
        num_layers: Number of transformer encoder layers.
        ffn_dim: Feed-forward network hidden dimension.  Defaults to
            ``4 * model_dim``.
        num_classes: Number of output classes.
        dropout: Dropout rate throughout the model.
        max_seq_len: Maximum sequence length (for positional encoding).
        positional_encoding: ``"sinusoidal"`` (default) or ``"learnable"``.
        pooling: ``"cls"`` (default) or ``"mean"``.
    """

    def __init__(
        self,
        input_dim: int = 258,
        model_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        ffn_dim: Optional[int] = None,  # type: ignore[assignment]
        num_classes: int = 100,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        positional_encoding: str = "sinusoidal",
        pooling: str = "cls",
    ) -> None:
        super().__init__()

        if model_dim % num_heads != 0:
            raise ValueError(
                f"model_dim ({model_dim}) must be divisible by "
                f"num_heads ({num_heads})"
            )

        self.model_dim = model_dim
        self.num_classes = num_classes
        self.pooling = pooling

        ffn_dim = ffn_dim or model_dim * 4

        # Input projection
        self.input_projection = nn.Linear(input_dim, model_dim)
        self.input_norm = nn.LayerNorm(model_dim)

        # [CLS] token (used when pooling == "cls")
        if pooling == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, model_dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        else:
            self.cls_token = None

        # Positional encoding
        pe_seq_len = max_seq_len + 1  # +1 for the optional [CLS] token
        if positional_encoding == "sinusoidal":
            self.pos_enc = SinusoidalPositionalEncoding(
                model_dim, max_seq_len=pe_seq_len, dropout=dropout
            )
        elif positional_encoding == "learnable":
            self.pos_enc = LearnablePositionalEncoding(
                model_dim, max_seq_len=pe_seq_len, dropout=dropout
            )
        else:
            raise ValueError(
                f"positional_encoding must be 'sinusoidal' or 'learnable', "
                f"got '{positional_encoding}'"
            )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,  # Pre-LN for training stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Keypoint tensor of shape (B, T, input_dim).

        Returns:
            Logits of shape (B, num_classes).
        """
        B = x.size(0)

        # Project and normalize input
        x = self.input_projection(x)  # (B, T, model_dim)
        x = self.input_norm(x)

        # Prepend [CLS] token
        if self.cls_token is not None:
            cls = self.cls_token.expand(B, -1, -1)  # (B, 1, model_dim)
            x = torch.cat([cls, x], dim=1)           # (B, T+1, model_dim)

        # Add positional encoding
        x = self.pos_enc(x)

        # Transformer encoder
        x = self.transformer(x)  # (B, T[+1], model_dim)

        # Pooling
        if self.pooling == "cls":
            pooled = x[:, 0, :]   # (B, model_dim)
        else:
            pooled = x.mean(dim=1)  # (B, model_dim)

        return self.classifier(pooled)  # (B, num_classes)

    def count_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
