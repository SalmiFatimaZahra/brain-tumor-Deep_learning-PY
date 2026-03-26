"""
Inference script: convert sign language videos to text.

Usage
-----
    # Single video
    python infer.py \\
        --checkpoint outputs/models/sign_language/best_model.pt \\
        --label_map  outputs/models/sign_language/label_map.json \\
        --video      path/to/sign_video.mp4

    # Batch mode (directory of videos)
    python infer.py \\
        --checkpoint outputs/models/sign_language/best_model.pt \\
        --label_map  outputs/models/sign_language/label_map.json \\
        --video_dir  path/to/videos/ \\
        --output     predictions.json

    # Live webcam mode (press q to quit)
    python infer.py \\
        --checkpoint outputs/models/sign_language/best_model.pt \\
        --label_map  outputs/models/sign_language/label_map.json \\
        --webcam

Output
------
For a single video the predicted gloss is printed to stdout.
For batch mode a JSON file with one entry per video is saved.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sign_language.data.preprocessing import extract_frames
from sign_language.models.lstm_model import SignLSTM
from sign_language.models.transformer_model import SignTransformer
from sign_language.utils.pose_extraction import extract_keypoints_from_frames

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------


def load_model(
    checkpoint_path: str,
    device: torch.device,
) -> Tuple[nn.Module, dict, int]:
    """
    Load a trained model from a checkpoint.

    Args:
        checkpoint_path: Path to the ``.pt`` checkpoint file.
        device: Target device.

    Returns:
        Tuple of (model, label_map, num_classes).

    Raises:
        FileNotFoundError: If the checkpoint file is missing.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device)
    saved_args = ckpt.get("args", {})

    model_type = saved_args.get("model", "lstm")
    num_classes = saved_args.get("num_classes", 100)
    input_dim = saved_args.get("input_dim", 258)
    hidden_dim = saved_args.get("hidden_dim", 256)
    num_layers = saved_args.get("num_layers", 3)
    dropout = saved_args.get("dropout", 0.0)  # no dropout at inference

    if model_type == "lstm":
        model = SignLSTM(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=dropout,
            bidirectional=not saved_args.get("no_bidirectional", False),
            use_attention=not saved_args.get("no_attention", False),
        )
    elif model_type == "transformer":
        model = SignTransformer(
            input_dim=input_dim,
            model_dim=hidden_dim,
            num_heads=saved_args.get("num_heads", 8),
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=dropout,
            max_seq_len=saved_args.get("max_frames", 64) + 1,
            positional_encoding=saved_args.get("pos_encoding", "sinusoidal"),
            pooling=saved_args.get("pooling", "cls"),
        )
    else:
        raise ValueError(f"Unknown model type in checkpoint: {model_type}")

    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    logger.info("Loaded %s model (epoch %d) from %s", model_type.upper(),
                ckpt.get("epoch", -1) + 1, checkpoint_path)
    return model, saved_args, num_classes


# ---------------------------------------------------------------------------
# Label map helpers
# ---------------------------------------------------------------------------


def load_label_map(path: str) -> Dict[str, int]:
    """Load a JSON label map file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Label map not found: {path}")
    with open(path, "r") as f:
        return json.load(f)


def idx_to_label_map(label_map: Dict[str, int]) -> Dict[int, str]:
    """Invert a gloss→index map to index→gloss."""
    return {v: k for k, v in label_map.items()}


# ---------------------------------------------------------------------------
# Core prediction function
# ---------------------------------------------------------------------------


@torch.no_grad()
def predict_video(
    video_path: str,
    model: nn.Module,
    idx_to_label: Dict[int, str],
    max_frames: int = 64,
    device: Optional[torch.device] = None,
    top_k: int = 5,
) -> Dict:
    """
    Run inference on a single video file.

    Args:
        video_path: Path to the video.
        model: Loaded PyTorch model in eval mode.
        idx_to_label: Mapping from class index to gloss string.
        max_frames: Temporal clip length.
        device: Inference device.
        top_k: Number of top predictions to return.

    Returns:
        Dict with keys:
        - ``"video"``: video path
        - ``"prediction"``: top-1 gloss
        - ``"confidence"``: top-1 confidence in [0, 1]
        - ``"top_k"``: list of (gloss, confidence) for top-k predictions
    """
    if device is None:
        device = next(model.parameters()).device

    # Extract frames and keypoints
    frames = extract_frames(video_path, max_frames=max_frames)
    keypoints = extract_keypoints_from_frames(frames)  # (T, D)

    tensor = torch.from_numpy(keypoints).float().unsqueeze(0).to(device)  # (1, T, D)

    logits = model(tensor)  # (1, C)
    probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()  # (C,)

    top_indices = np.argsort(probs)[::-1][:top_k]
    top_predictions = [
        {"gloss": idx_to_label.get(int(i), f"<{i}>"), "confidence": float(probs[i])}
        for i in top_indices
    ]

    return {
        "video": video_path,
        "prediction": top_predictions[0]["gloss"],
        "confidence": top_predictions[0]["confidence"],
        "top_k": top_predictions,
    }


# ---------------------------------------------------------------------------
# Webcam / live mode
# ---------------------------------------------------------------------------


def run_webcam(
    model: nn.Module,
    idx_to_label: Dict[int, str],
    max_frames: int = 64,
    device: Optional[torch.device] = None,
) -> None:
    """
    Run live sign language recognition from webcam.

    Captures ``max_frames`` frames in a sliding window, runs inference,
    and overlays the prediction on the display.  Press ``q`` to quit.

    Args:
        model: Loaded PyTorch model in eval mode.
        idx_to_label: Mapping from class index to gloss string.
        max_frames: Buffer size (number of frames per prediction).
        device: Inference device.
    """
    try:
        import cv2
        import mediapipe as mp  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "opencv-python and mediapipe are required for webcam mode. "
            "Install them with: pip install opencv-python mediapipe"
        ) from exc

    from sign_language.utils.pose_extraction import PoseExtractor

    if device is None:
        device = next(model.parameters()).device

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam (device 0)")

    extractor = PoseExtractor()
    frame_buffer: List[np.ndarray] = []
    prediction_text = "Collecting frames..."

    logger.info("Webcam mode started. Press 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            keypoints = extractor.extract_frame(frame_rgb)
            frame_buffer.append(keypoints)

            # Keep only the last max_frames keypoints
            if len(frame_buffer) > max_frames:
                frame_buffer = frame_buffer[-max_frames:]

            # Run inference when buffer is full
            if len(frame_buffer) == max_frames:
                kp_array = np.stack(frame_buffer, axis=0)  # (T, D)
                tensor = (
                    torch.from_numpy(kp_array).float().unsqueeze(0).to(device)
                )
                with torch.no_grad():
                    logits = model(tensor)
                probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
                top_idx = int(np.argmax(probs))
                prediction_text = (
                    f"{idx_to_label.get(top_idx, str(top_idx))} "
                    f"({probs[top_idx]:.2%})"
                )

            # Display
            display_frame = frame.copy()
            cv2.putText(
                display_frame,
                prediction_text,
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("Sign Language Recognition", display_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        extractor.close()


# ---------------------------------------------------------------------------
# Batch inference helper
# ---------------------------------------------------------------------------


def run_batch_inference(
    video_dir: str,
    model: nn.Module,
    idx_to_label: Dict[int, str],
    max_frames: int,
    device: torch.device,
    top_k: int = 5,
    extensions: Tuple[str, ...] = (".mp4", ".avi", ".mov"),
) -> List[Dict]:
    """
    Run inference on all videos in a directory.

    Args:
        video_dir: Directory containing video files.
        model: Loaded model in eval mode.
        idx_to_label: Index-to-gloss mapping.
        max_frames: Temporal clip length.
        device: Inference device.
        top_k: Number of top predictions per video.
        extensions: File extensions to include.

    Returns:
        List of prediction dicts (one per video).
    """
    video_dir_path = Path(video_dir)
    videos = sorted(
        p for p in video_dir_path.iterdir()
        if p.suffix.lower() in extensions
    )

    if not videos:
        logger.warning("No videos found in %s with extensions %s", video_dir, extensions)
        return []

    results = []
    for i, vpath in enumerate(videos, 1):
        logger.info("[%d/%d] Processing %s", i, len(videos), vpath.name)
        try:
            result = predict_video(
                str(vpath), model, idx_to_label,
                max_frames=max_frames, device=device, top_k=top_k,
            )
            results.append(result)
        except Exception as exc:
            logger.error("  Failed to process %s: %s", vpath.name, exc)
            results.append({"video": str(vpath), "error": str(exc)})

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sign language video → text inference.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--checkpoint", required=True, help="Path to model checkpoint (.pt)"
    )
    parser.add_argument(
        "--label_map", required=True, help="Path to label_map.json"
    )

    # Input sources (mutually exclusive)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", help="Path to a single video file")
    src.add_argument("--video_dir", help="Directory of videos for batch inference")
    src.add_argument(
        "--webcam", action="store_true", help="Use live webcam input"
    )

    parser.add_argument(
        "--output", default="", help="Output JSON file (batch mode only)"
    )
    parser.add_argument(
        "--max_frames", type=int, default=64, help="Temporal clip length"
    )
    parser.add_argument("--top_k", type=int, default=5, help="Number of top predictions")
    parser.add_argument(
        "--device", default="", help="Force device (e.g. 'cuda:0', 'cpu')"
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # Load model
    model, saved_args, _ = load_model(args.checkpoint, device)
    max_frames = saved_args.get("max_frames", args.max_frames)

    # Load label map
    label_map = load_label_map(args.label_map)
    idx2label = idx_to_label_map(label_map)

    # ── Single video ──────────────────────────────────────────────────────
    if args.video:
        result = predict_video(
            args.video, model, idx2label,
            max_frames=max_frames, device=device, top_k=args.top_k,
        )
        print(f"\nPrediction: {result['prediction']}  (confidence: {result['confidence']:.2%})")
        print("Top-k predictions:")
        for entry in result["top_k"]:
            print(f"  {entry['gloss']:<30} {entry['confidence']:.4f}")

    # ── Batch mode ────────────────────────────────────────────────────────
    elif args.video_dir:
        results = run_batch_inference(
            args.video_dir, model, idx2label,
            max_frames=max_frames, device=device, top_k=args.top_k,
        )
        if args.output:
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)
            logger.info("Saved %d predictions to %s", len(results), out_path)
        else:
            for r in results:
                status = r.get("prediction", r.get("error", "?"))
                print(f"{Path(r['video']).name:<40} → {status}")

    # ── Webcam mode ───────────────────────────────────────────────────────
    elif args.webcam:
        run_webcam(model, idx2label, max_frames=max_frames, device=device)


if __name__ == "__main__":
    main()
