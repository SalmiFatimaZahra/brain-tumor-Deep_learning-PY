"""
Training pipeline for sign language recognition.

Quick start
-----------
    # Train LSTM model on WLASL dataset
    python train.py \\
        --data_root data/wlasl \\
        --model lstm \\
        --num_classes 100 \\
        --epochs 50 \\
        --batch_size 32 \\
        --lr 1e-3 \\
        --output_dir outputs/models/sign_language

    # Train Transformer model
    python train.py \\
        --data_root data/wlasl \\
        --model transformer \\
        --num_classes 100 \\
        --epochs 50 \\
        --batch_size 16 \\
        --output_dir outputs/models/sign_language
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sign_language.data.dataset import WLASLDataset
from sign_language.models.lstm_model import SignLSTM
from sign_language.models.transformer_model import SignTransformer
from sign_language.utils.metrics import MetricAccumulator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------


def build_model(args: argparse.Namespace) -> nn.Module:
    """Instantiate the requested model architecture."""
    if args.model == "lstm":
        model = SignLSTM(
            input_dim=args.input_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_classes=args.num_classes,
            dropout=args.dropout,
            bidirectional=not args.no_bidirectional,
            use_attention=not args.no_attention,
        )
    elif args.model == "transformer":
        model = SignTransformer(
            input_dim=args.input_dim,
            model_dim=args.hidden_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            num_classes=args.num_classes,
            dropout=args.dropout,
            max_seq_len=args.max_frames + 1,
            positional_encoding=args.pos_encoding,
            pooling=args.pooling,
        )
    else:
        raise ValueError(f"Unknown model: {args.model}. Choose 'lstm' or 'transformer'.")

    logger.info(
        "Built %s model with %s trainable parameters",
        args.model.upper(),
        f"{model.count_parameters():,}",
    )
    return model


# ---------------------------------------------------------------------------
# Training and evaluation loops
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    scaler: torch.cuda.amp.GradScaler,
    grad_clip: float = 1.0,
) -> Dict[str, float]:
    """Run one training epoch and return metrics."""
    model.train()
    accum = MetricAccumulator(num_classes)

    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(inputs)
            loss = criterion(logits, targets)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        accum.update(logits.detach(), targets.detach(), loss.item())

        if (batch_idx + 1) % 20 == 0:
            metrics = accum.compute()
            logger.debug(
                "  step %d  loss=%.4f  acc=%.4f",
                batch_idx + 1,
                metrics.get("loss", 0.0),
                metrics.get("accuracy", 0.0),
            )

    return accum.compute()


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
) -> Dict[str, float]:
    """Evaluate the model on a data loader."""
    model.eval()
    accum = MetricAccumulator(num_classes)

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(inputs)
            loss = criterion(logits, targets)

        accum.update(logits, targets, loss.item())

    return accum.compute()


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    args: argparse.Namespace,
    path: str,
) -> None:
    """Save model checkpoint and training metadata."""
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "args": vars(args),
        },
        path,
    )


def load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    path: str,
    device: torch.device,
) -> Tuple[int, Dict[str, float]]:
    """Load a checkpoint; returns (start_epoch, best_metrics)."""
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    logger.info("Loaded checkpoint from epoch %d at %s", ckpt["epoch"], path)
    return ckpt["epoch"] + 1, ckpt.get("metrics", {})


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------


def train(args: argparse.Namespace) -> None:
    """End-to-end training pipeline."""
    # ── Device ────────────────────────────────────────────────────────────
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    logger.info("Using device: %s", device)

    # ── Output directory ───────────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Datasets & loaders ────────────────────────────────────────────────
    logger.info("Loading datasets from %s", args.data_root)
    train_dataset = WLASLDataset(
        root=args.data_root,
        split="train",
        max_frames=args.max_frames,
        use_keypoints=True,
        cache_keypoints=args.cache_keypoints,
    )
    val_dataset = WLASLDataset(
        root=args.data_root,
        split="val",
        max_frames=args.max_frames,
        use_keypoints=True,
        cache_keypoints=args.cache_keypoints,
    )

    logger.info(
        "Train samples: %d  |  Val samples: %d  |  Classes: %d",
        len(train_dataset),
        len(val_dataset),
        train_dataset.num_classes,
    )

    # Override num_classes from dataset
    if args.num_classes != train_dataset.num_classes:
        logger.warning(
            "--num_classes=%d but dataset has %d classes; using dataset value.",
            args.num_classes,
            train_dataset.num_classes,
        )
        args.num_classes = train_dataset.num_classes

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    # ── Model ─────────────────────────────────────────────────────────────
    model = build_model(args).to(device)

    # ── Optimizer & scheduler ─────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 1e-2
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    # ── Resume from checkpoint ────────────────────────────────────────────
    start_epoch = 0
    best_val_acc = 0.0
    if args.resume and os.path.isfile(args.resume):
        start_epoch, _ = load_checkpoint(model, optimizer, args.resume, device)

    # ── Save label map alongside the model ────────────────────────────────
    lm_dest = output_dir / "label_map.json"
    if not lm_dest.exists():
        with open(lm_dest, "w") as f:
            json.dump(train_dataset.label_map, f, indent=2)

    # ── Training loop ─────────────────────────────────────────────────────
    best_ckpt_path = output_dir / "best_model.pt"
    history = []

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            args.num_classes, scaler, args.grad_clip,
        )
        val_metrics = evaluate(
            model, val_loader, criterion, device, args.num_classes
        )

        scheduler.step()
        epoch_time = time.time() - t0

        logger.info(
            "Epoch %03d/%03d  [%.1fs]  "
            "train_loss=%.4f  train_acc=%.4f  "
            "val_loss=%.4f  val_acc=%.4f  val_top5=%.4f",
            epoch + 1,
            args.epochs,
            epoch_time,
            train_metrics.get("loss", 0.0),
            train_metrics.get("accuracy", 0.0),
            val_metrics.get("loss", 0.0),
            val_metrics.get("accuracy", 0.0),
            val_metrics.get("top5_accuracy", 0.0),
        )

        record = {
            "epoch": epoch + 1,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(record)

        # Save best checkpoint
        val_acc = val_metrics.get("accuracy", 0.0)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                model, optimizer, epoch, val_metrics, args, str(best_ckpt_path)
            )
            logger.info("  ✓ Saved best model (val_acc=%.4f)", best_val_acc)

        # Save periodic checkpoint
        if (epoch + 1) % args.save_every == 0:
            ckpt_path = output_dir / f"checkpoint_epoch_{epoch + 1:03d}.pt"
            save_checkpoint(model, optimizer, epoch, val_metrics, args, str(ckpt_path))

    # ── Save training history ─────────────────────────────────────────────
    history_path = output_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info("Training history saved to %s", history_path)
    logger.info("Best validation accuracy: %.4f", best_val_acc)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a sign language recognition model on WLASL.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    data = parser.add_argument_group("Data")
    data.add_argument("--data_root", required=True, help="Path to WLASL dataset root")
    data.add_argument("--max_frames", type=int, default=64, help="Temporal clip length")
    data.add_argument(
        "--cache_keypoints",
        action="store_true",
        default=True,
        help="Cache extracted keypoints to disk",
    )
    data.add_argument(
        "--no_cache_keypoints",
        dest="cache_keypoints",
        action="store_false",
    )
    data.add_argument(
        "--num_workers", type=int, default=4, help="DataLoader worker processes"
    )

    # Model
    mdl = parser.add_argument_group("Model")
    mdl.add_argument(
        "--model",
        choices=["lstm", "transformer"],
        default="lstm",
        help="Model architecture",
    )
    mdl.add_argument("--num_classes", type=int, default=100)
    mdl.add_argument("--input_dim", type=int, default=258, help="Keypoint feature dim")
    mdl.add_argument("--hidden_dim", type=int, default=256)
    mdl.add_argument("--num_layers", type=int, default=3)
    mdl.add_argument("--dropout", type=float, default=0.3)
    # LSTM-specific
    mdl.add_argument("--no_bidirectional", action="store_true")
    mdl.add_argument("--no_attention", action="store_true")
    # Transformer-specific
    mdl.add_argument("--num_heads", type=int, default=8)
    mdl.add_argument(
        "--pos_encoding",
        choices=["sinusoidal", "learnable"],
        default="sinusoidal",
    )
    mdl.add_argument("--pooling", choices=["cls", "mean"], default="cls")

    # Training
    trn = parser.add_argument_group("Training")
    trn.add_argument("--epochs", type=int, default=50)
    trn.add_argument("--batch_size", type=int, default=32)
    trn.add_argument("--lr", type=float, default=1e-3)
    trn.add_argument("--weight_decay", type=float, default=1e-4)
    trn.add_argument("--label_smoothing", type=float, default=0.1)
    trn.add_argument("--grad_clip", type=float, default=1.0)
    trn.add_argument("--device", default="", help="Force device (e.g. 'cuda:0', 'cpu')")
    trn.add_argument("--resume", default="", help="Path to checkpoint to resume from")
    trn.add_argument(
        "--output_dir",
        default="outputs/models/sign_language",
        help="Directory for checkpoints and logs",
    )
    trn.add_argument(
        "--save_every",
        type=int,
        default=10,
        help="Save periodic checkpoint every N epochs",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
