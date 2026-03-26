"""
Video-to-text model: CNN encoder + LSTM decoder with attention.

Architecture
------------
* **Encoder** – A pretrained ResNet-50 (or any torchvision backbone) processes
  each video frame independently and produces a feature vector per frame.
  The per-frame features are then aggregated across time with a single-layer
  bidirectional LSTM to produce a fixed-length context vector.

* **Decoder** – A single-layer LSTM generates one word per step.  At each step
  it attends over the per-frame encoder outputs via Bahdanau (additive)
  attention, then combines the attended context with the current word embedding
  to produce the next-word logits.

Usage example
-------------
::

    from video_to_text.model import VideoToTextModel

    model = VideoToTextModel(vocab_size=5000, embed_dim=256,
                             encoder_hidden=512, decoder_hidden=512)
    # frames: (B, T, 3, 224, 224)
    logits = model(frames, captions)   # (B, seq_len-1, vocab_size)
    words  = model.generate(frames)    # list[list[int]]
"""

import torch
import torch.nn as nn
import torchvision.models as tv_models


# ---------------------------------------------------------------------------
# Bahdanau (additive) attention
# ---------------------------------------------------------------------------
class BahdanauAttention(nn.Module):
    """Additive attention between decoder hidden state and encoder outputs."""

    def __init__(self, encoder_dim: int, decoder_dim: int, attn_dim: int = 256):
        super().__init__()
        self.W_enc = nn.Linear(encoder_dim, attn_dim, bias=False)
        self.W_dec = nn.Linear(decoder_dim, attn_dim, bias=False)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, encoder_out: torch.Tensor, decoder_hidden: torch.Tensor):
        """
        Args:
            encoder_out:    (B, T, encoder_dim)
            decoder_hidden: (B, decoder_dim)

        Returns:
            context: (B, encoder_dim)
            weights: (B, T)
        """
        energy = torch.tanh(
            self.W_enc(encoder_out) + self.W_dec(decoder_hidden).unsqueeze(1)
        )  # (B, T, attn_dim)
        scores = self.v(energy).squeeze(-1)  # (B, T)
        weights = torch.softmax(scores, dim=1)  # (B, T)
        context = (weights.unsqueeze(-1) * encoder_out).sum(dim=1)  # (B, encoder_dim)
        return context, weights


# ---------------------------------------------------------------------------
# Video encoder
# ---------------------------------------------------------------------------
class VideoEncoder(nn.Module):
    """
    Per-frame CNN + temporal BiLSTM encoder.

    Args:
        cnn_out_dim:     Output dimensionality of the CNN backbone feature map.
        hidden_dim:      Hidden size of the temporal BiLSTM.
        num_layers:      Number of BiLSTM layers.
        dropout:         Dropout probability applied between LSTM layers.
        pretrained_backbone: If ``True``, load ImageNet-pretrained weights for
                             the ResNet-50 backbone.
    """

    def __init__(
        self,
        cnn_out_dim: int = 2048,
        hidden_dim: int = 512,
        num_layers: int = 1,
        dropout: float = 0.3,
        pretrained_backbone: bool = True,
    ):
        super().__init__()

        # ---- CNN backbone (ResNet-50 without the classification head) ----
        weights = tv_models.ResNet50_Weights.IMAGENET1K_V1 if pretrained_backbone else None
        backbone = tv_models.resnet50(weights=weights)
        # Remove the average-pool and fc layers; keep everything up to layer4
        self.cnn = nn.Sequential(*list(backbone.children())[:-2])  # output: (B*T, 2048, 7, 7)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))                    # → (B*T, 2048, 1, 1)
        self.cnn_proj = nn.Linear(cnn_out_dim, hidden_dim)

        # ---- Temporal aggregation ----
        self.temporal_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_dim = hidden_dim * 2  # bidirectional

    def forward(self, frames: torch.Tensor):
        """
        Args:
            frames: (B, T, C, H, W)

        Returns:
            encoder_out:  (B, T, output_dim)  per-frame hidden states
            hidden:       (B, output_dim)     last-step hidden state (fwd + bwd)
        """
        B, T, C, H, W = frames.shape
        x = frames.view(B * T, C, H, W)
        x = self.cnn(x)          # (B*T, 2048, h, w)
        x = self.pool(x)         # (B*T, 2048, 1, 1)
        x = x.view(B * T, -1)   # (B*T, 2048)
        x = self.cnn_proj(x)     # (B*T, hidden_dim)
        x = x.view(B, T, -1)    # (B, T, hidden_dim)

        enc_out, _ = self.temporal_lstm(x)  # (B, T, hidden_dim*2)
        hidden = enc_out[:, -1, :]          # (B, hidden_dim*2)
        return enc_out, hidden


# ---------------------------------------------------------------------------
# Text decoder
# ---------------------------------------------------------------------------
class TextDecoder(nn.Module):
    """
    LSTM decoder with Bahdanau attention over encoder outputs.

    Args:
        vocab_size:   Size of the output vocabulary.
        embed_dim:    Word embedding dimensionality.
        encoder_dim:  Dimensionality of encoder output vectors (= encoder.output_dim).
        hidden_dim:   LSTM hidden size.
        dropout:      Dropout probability.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        encoder_dim: int = 1024,
        hidden_dim: int = 512,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.attention = BahdanauAttention(encoder_dim, hidden_dim)
        self.lstm = nn.LSTMCell(embed_dim + encoder_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

        # Project encoder last state → decoder initial hidden/cell
        self.h0_proj = nn.Linear(encoder_dim, hidden_dim)
        self.c0_proj = nn.Linear(encoder_dim, hidden_dim)

    def init_hidden(self, encoder_last: torch.Tensor):
        """Initialise decoder hidden/cell from encoder last hidden state."""
        h = torch.tanh(self.h0_proj(encoder_last))
        c = torch.tanh(self.c0_proj(encoder_last))
        return h, c

    def forward_step(self, word_idx, h, c, encoder_out):
        """Single decoding step.

        Args:
            word_idx:    (B,) integer tensor of current input word indices.
            h, c:        Decoder LSTM hidden and cell states, each (B, hidden_dim).
            encoder_out: (B, T, encoder_dim)

        Returns:
            logits: (B, vocab_size)
            h, c:   Updated hidden and cell states.
        """
        emb = self.dropout(self.embed(word_idx))         # (B, embed_dim)
        context, _ = self.attention(encoder_out, h)      # (B, encoder_dim)
        lstm_input = torch.cat([emb, context], dim=1)    # (B, embed_dim + encoder_dim)
        h, c = self.lstm(lstm_input, (h, c))
        logits = self.fc_out(self.dropout(h))            # (B, vocab_size)
        return logits, h, c

    def forward(self, encoder_out, encoder_last, captions):
        """
        Teacher-forced forward pass.

        Args:
            encoder_out:   (B, T, encoder_dim)
            encoder_last:  (B, encoder_dim)
            captions:      (B, seq_len)  integer token indices (includes <SOS>)

        Returns:
            logits: (B, seq_len-1, vocab_size)
        """
        B, seq_len = captions.shape
        h, c = self.init_hidden(encoder_last)

        outputs = []
        for t in range(seq_len - 1):
            logits, h, c = self.forward_step(captions[:, t], h, c, encoder_out)
            outputs.append(logits)

        return torch.stack(outputs, dim=1)  # (B, seq_len-1, vocab_size)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------
class VideoToTextModel(nn.Module):
    """
    End-to-end video captioning model.

    Args:
        vocab_size:          Size of the target vocabulary.
        embed_dim:           Word embedding dimension.
        encoder_hidden:      Hidden size of the encoder BiLSTM.
        decoder_hidden:      Hidden size of the decoder LSTM.
        attn_dim:            Attention projection dimension.
        dropout:             Dropout probability.
        pretrained_backbone: Use ImageNet-pretrained CNN backbone.
        max_gen_len:         Maximum number of tokens to generate at inference.
        sos_idx:             Index of the <SOS> token (default 1).
        eos_idx:             Index of the <EOS> token (default 2).
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        encoder_hidden: int = 512,
        decoder_hidden: int = 512,
        attn_dim: int = 256,
        dropout: float = 0.3,
        pretrained_backbone: bool = True,
        max_gen_len: int = 50,
        sos_idx: int = 1,
        eos_idx: int = 2,
    ):
        super().__init__()
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.max_gen_len = max_gen_len

        self.encoder = VideoEncoder(
            hidden_dim=encoder_hidden,
            dropout=dropout,
            pretrained_backbone=pretrained_backbone,
        )
        encoder_dim = self.encoder.output_dim  # hidden_dim * 2

        self.decoder = TextDecoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            encoder_dim=encoder_dim,
            hidden_dim=decoder_hidden,
            dropout=dropout,
        )

    def forward(self, frames: torch.Tensor, captions: torch.Tensor):
        """
        Args:
            frames:   (B, T, C, H, W)
            captions: (B, seq_len)

        Returns:
            logits: (B, seq_len-1, vocab_size)
        """
        enc_out, enc_last = self.encoder(frames)
        return self.decoder(enc_out, enc_last, captions)

    @torch.no_grad()
    def generate(self, frames: torch.Tensor) -> list[list[int]]:
        """
        Greedy inference: generate a token sequence for each video in the batch.

        Args:
            frames: (B, T, C, H, W)

        Returns:
            List of token-index lists (one per video in the batch), each
            terminated before or at ``<EOS>``.
        """
        self.eval()
        enc_out, enc_last = self.encoder(frames)
        B = frames.size(0)

        h, c = self.decoder.init_hidden(enc_last)
        word = torch.full((B,), self.sos_idx, dtype=torch.long, device=frames.device)

        sequences = [[] for _ in range(B)]
        finished = [False] * B

        for _ in range(self.max_gen_len):
            logits, h, c = self.decoder.forward_step(word, h, c, enc_out)
            word = logits.argmax(dim=-1)  # (B,)
            for i in range(B):
                if not finished[i]:
                    token = word[i].item()
                    if token == self.eos_idx:
                        finished[i] = True
                    else:
                        sequences[i].append(token)
            if all(finished):
                break

        return sequences
