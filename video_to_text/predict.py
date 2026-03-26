"""
Inference script — generate text descriptions from video files.

Usage
-----
::

    python -m video_to_text.predict \\
        --checkpoint outputs/models/best.pt \\
        --videos     clip1.mp4 clip2.mp4 \\
        --num_frames 16

Each video path is printed alongside its predicted caption.
"""

import argparse

import torch

from .dataset import Vocabulary, extract_frames, PAD_TOKEN
from .model import VideoToTextModel


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate text captions for video files")
    p.add_argument("--checkpoint", required=True, help="Path to a saved model checkpoint (.pt)")
    p.add_argument("--videos", nargs="+", required=True, help="One or more video file paths")
    p.add_argument("--num_frames", type=int, default=16, help="Frames to sample per video")
    p.add_argument("--max_gen_len", type=int, default=50, help="Maximum caption length")
    p.add_argument("--device", default=None, help="'cuda' or 'cpu' (auto-detected by default)")
    return p


def predict(args=None):
    parser = _build_arg_parser()
    args = parser.parse_args(args)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    # ----------------------------------------------------------------- load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)

    # Rebuild vocabulary from checkpoint
    word2idx = ckpt["vocab"]
    vocab = Vocabulary()
    vocab.word2idx = word2idx
    vocab.idx2word = {int(v): k for k, v in word2idx.items()}

    vocab_size = len(vocab)
    pad_idx = vocab.word2idx.get(PAD_TOKEN, 0)

    # ----------------------------------------------------------------- rebuild model
    model = VideoToTextModel(
        vocab_size=vocab_size,
        max_gen_len=args.max_gen_len,
        pretrained_backbone=False,   # weights are loaded from checkpoint
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # ----------------------------------------------------------------- run inference
    for video_path in args.videos:
        try:
            frames = extract_frames(video_path, num_frames=args.num_frames)
        except ValueError as exc:
            print(f"{video_path}: ERROR — {exc}")
            continue

        frames = frames.unsqueeze(0).to(device)  # (1, T, C, H, W)
        token_ids = model.generate(frames)[0]
        caption = vocab.decode(token_ids)
        print(f"{video_path}: {caption}")


if __name__ == "__main__":
    predict()
