"""
LSTM-based model for sign language recognition.

Architecture
------------
  Input: (batch, T, input_dim)  -- MediaPipe keypoint sequences
    │
    ├── Optional layer-norm on input features
    │
    ├── Stacked bi-directional LSTM layers
    │
    ├── Temporal attention pooling
    │
    └── Classification head  →  (batch, num_classes)

Usage
-----
    from sign_language.models.lstm_model import SignLSTM

    model = SignLSTM(
        input_dim=258,
        hidden_dim=256,
        num_layers=3,
        num_classes=100,
        dropout=0.3,
        bidirectional=True,
    )
    # x: (B, T, 258)
    logits = model(x)   # (B, 100)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttention(nn.Module):
    """
    Additive temporal attention pooling.

    Compresses the time dimension (T) into a single context vector by
    computing a weighted sum over all timesteps.

    Args:
        hidden_dim: Dimensionality of the LSTM output fed into attention.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.query = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, H) LSTM output sequence.

        Returns:
            (B, H) context vector.
        """
        # (B, T, 1) → (B, T) attention weights
        scores = self.query(x).squeeze(-1)
        weights = F.softmax(scores, dim=1)  # (B, T)
        context = torch.bmm(weights.unsqueeze(1), x).squeeze(1)  # (B, H)
        return context


class SignLSTM(nn.Module):
    """
    Bidirectional LSTM classifier for sign language recognition.

    Args:
        input_dim: Feature dimension of each timestep (e.g. 258 for default
            MediaPipe keypoints without face).
        hidden_dim: Number of units in each LSTM direction.
        num_layers: Number of stacked LSTM layers.
        num_classes: Number of sign classes to predict.
        dropout: Dropout rate applied between LSTM layers and before the
            classification head.
        bidirectional: Use bidirectional LSTM (doubles effective hidden dim).
        use_attention: Use temporal attention pooling instead of taking the
            last hidden state.
    """

    def __init__(
        self,
        input_dim: int = 258,
        hidden_dim: int = 256,
        num_layers: int = 3,
        num_classes: int = 100,
        dropout: float = 0.3,
        bidirectional: bool = True,
        use_attention: bool = True,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bidirectional
        self.use_attention = use_attention

        # Input normalization
        self.input_norm = nn.LayerNorm(input_dim)

        # LSTM encoder
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        lstm_out_dim = hidden_dim * (2 if bidirectional else 1)

        # Temporal pooling
        if use_attention:
            self.attention = TemporalAttention(lstm_out_dim)
        else:
            self.attention = None

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(lstm_out_dim),
            nn.Dropout(dropout),
            nn.Linear(lstm_out_dim, lstm_out_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_out_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Keypoint tensor of shape (B, T, input_dim).

        Returns:
            Logits of shape (B, num_classes).
        """
        x = self.input_norm(x)  # (B, T, D)
        lstm_out, _ = self.lstm(x)  # (B, T, H)

        if self.attention is not None:
            pooled = self.attention(lstm_out)  # (B, H)
        else:
            # Use output at the last *actual* timestep
            pooled = lstm_out[:, -1, :]  # (B, H)

        logits = self.classifier(pooled)  # (B, C)
        return logits

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def lstm_output_dim(self) -> int:
        """Effective hidden dimension after the LSTM (accounts for bidir)."""
        return self.hidden_dim * (2 if self.bidirectional else 1)

    def count_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
