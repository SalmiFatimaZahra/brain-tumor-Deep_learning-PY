"""
Training script for the video-to-text model.

Usage
-----
::

    python -m video_to_text.train \\
        --annotations data/annotations.json \\
        --output_dir  outputs/models/ \\
        --epochs      30 \\
        --batch_size  4

Annotation file format (JSON list)::

    [
        {"video": "data/videos/clip01.mp4", "caption": "a doctor examines the brain scan"},
        ...
    ]
"""

import argparse
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from .dataset import VideoTextDataset, collate_fn, PAD_TOKEN
from .model import VideoToTextModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train the video-to-text model")
    p.add_argument("--annotations", required=True, help="Path to JSON annotations file")
    p.add_argument("--output_dir", default="outputs/models/", help="Where to save checkpoints")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--num_frames", type=int, default=16, help="Frames to sample per video")
    p.add_argument("--embed_dim", type=int, default=256)
    p.add_argument("--encoder_hidden", type=int, default=512)
    p.add_argument("--decoder_hidden", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--val_split", type=float, default=0.1, help="Fraction of data for validation")
    p.add_argument("--grad_clip", type=float, default=5.0)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    p.add_argument("--no_pretrain", action="store_true", help="Do not use pretrained CNN weights")
    return p


def _save_checkpoint(state: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    print(f"  Checkpoint saved → {path}")


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(args=None):
    parser = _build_arg_parser()
    args = parser.parse_args(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ------------------------------------------------------------------ data
    print("Loading dataset …")
    dataset = VideoTextDataset(
        annotations_file=args.annotations,
        num_frames=args.num_frames,
    )
    vocab = dataset.vocab
    pad_idx = vocab.word2idx[PAD_TOKEN]

    val_size = max(1, int(len(dataset) * args.val_split))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    def make_loader(ds, shuffle):
        return DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=args.num_workers,
            collate_fn=lambda b: collate_fn(b, pad_idx=pad_idx),
            pin_memory=device.type == "cuda",
        )

    train_loader = make_loader(train_ds, shuffle=True)
    val_loader = make_loader(val_ds, shuffle=False)
    print(f"  Train samples: {train_size}  |  Val samples: {val_size}  |  Vocab size: {len(vocab)}")

    # ----------------------------------------------------------------- model
    model = VideoToTextModel(
        vocab_size=len(vocab),
        embed_dim=args.embed_dim,
        encoder_hidden=args.encoder_hidden,
        decoder_hidden=args.decoder_hidden,
        dropout=args.dropout,
        pretrained_backbone=not args.no_pretrain,
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5
    )

    start_epoch = 1
    best_val_loss = float("inf")

    if args.resume and os.path.isfile(args.resume):
        print(f"Resuming from {args.resume} …")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", best_val_loss)

    # --------------------------------------------------------------- training
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        train_loss = 0.0
        t0 = time.time()

        for frames, captions in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [train]"):
            frames = frames.to(device)
            captions = captions.to(device)

            optimizer.zero_grad()
            logits = model(frames, captions)            # (B, seq_len-1, vocab_size)
            targets = captions[:, 1:]                   # (B, seq_len-1)

            loss = criterion(
                logits.reshape(-1, logits.size(-1)),    # (B*(seq_len-1), vocab_size)
                targets.reshape(-1),                    # (B*(seq_len-1),)
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            train_loss += loss.item()

        avg_train = train_loss / len(train_loader)

        # -------------------------------------------------------- validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for frames, captions in tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [val]"):
                frames = frames.to(device)
                captions = captions.to(device)
                logits = model(frames, captions)
                targets = captions[:, 1:]
                loss = criterion(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1),
                )
                val_loss += loss.item()

        avg_val = val_loss / len(val_loader)
        scheduler.step(avg_val)
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"train_loss={avg_train:.4f}  val_loss={avg_val:.4f}  "
            f"time={elapsed:.1f}s"
        )

        # Save latest checkpoint every epoch
        _save_checkpoint(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "vocab": vocab.word2idx,
            },
            os.path.join(args.output_dir, "latest.pt"),
        )

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            _save_checkpoint(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                    "vocab": vocab.word2idx,
                },
                os.path.join(args.output_dir, "best.pt"),
            )

    print("Training complete.")


if __name__ == "__main__":
    train()
