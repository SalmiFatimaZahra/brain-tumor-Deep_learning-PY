"""
MediaPipe-based pose and hand keypoint extraction.

Extracts landmark coordinates from video frames using Google MediaPipe's
Holistic solution, which provides:
  - 33  pose landmarks   (body)
  - 21  left-hand landmarks
  - 21  right-hand landmarks
  - 468 face landmarks   (optionally included)

The feature vector for each frame is the concatenation of
(x, y, z, visibility) for all included landmarks, giving a
fixed-length representation regardless of the number of people in frame.

Total dimensions (default – no face):
    pose:  33 * 4 = 132
    left:  21 * 3 = 63   (no visibility for hands)
    right: 21 * 3 = 63
    total:           258

Usage
-----
    import cv2
    from sign_language.utils.pose_extraction import (
        PoseExtractor,
        extract_keypoints_from_frames,
    )

    extractor = PoseExtractor()
    frames = [cv2.imread(p) for p in frame_paths]
    keypoints = extractor.extract_sequence(frames)   # (T, 258)
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Feature dimension constants
# ---------------------------------------------------------------------------

POSE_LANDMARKS = 33
HAND_LANDMARKS = 21

# (x, y, z, visibility) per pose landmark
POSE_DIM = POSE_LANDMARKS * 4  # 132

# (x, y, z) per hand landmark  (MediaPipe hands do not expose visibility)
HAND_DIM = HAND_LANDMARKS * 3  # 63

# Default total feature dimension (pose + left hand + right hand)
KEYPOINT_DIM = POSE_DIM + HAND_DIM + HAND_DIM  # 258


# ---------------------------------------------------------------------------
# PoseExtractor class
# ---------------------------------------------------------------------------


class PoseExtractor:
    """
    Wrapper around MediaPipe Holistic for sequence-level keypoint extraction.

    Args:
        include_face: If ``True`` include 468 face mesh landmarks (adds 1404
            dimensions). Defaults to ``False``.
        min_detection_confidence: Minimum detection confidence for holistic.
        min_tracking_confidence: Minimum tracking confidence for holistic.
    """

    def __init__(
        self,
        include_face: bool = False,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self.include_face = include_face
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        # Lazy import so that the module can be imported even without
        # mediapipe installed (tests can mock it).
        try:
            import mediapipe as mp  # noqa: F401

            self._mp = mp
            self._holistic = mp.solutions.holistic.Holistic(
                static_image_mode=False,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
        except ImportError as exc:
            raise ImportError(
                "mediapipe is required for pose extraction. "
                "Install it with:  pip install mediapipe"
            ) from exc

    def close(self) -> None:
        """Release the MediaPipe Holistic context."""
        if hasattr(self, "_holistic"):
            self._holistic.close()

    def __enter__(self) -> "PoseExtractor":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Per-frame extraction
    # ------------------------------------------------------------------

    def extract_frame(self, frame_rgb: np.ndarray) -> np.ndarray:
        """
        Extract keypoints from a single RGB frame.

        Args:
            frame_rgb: uint8 array of shape (H, W, 3) in RGB order.

        Returns:
            1-D float32 array of length ``KEYPOINT_DIM`` (or larger when
            ``include_face=True``).
        """
        results = self._holistic.process(frame_rgb)

        pose_kp = _extract_pose(results)
        left_kp = _extract_hand(results.left_hand_landmarks)
        right_kp = _extract_hand(results.right_hand_landmarks)

        parts = [pose_kp, left_kp, right_kp]

        if self.include_face:
            face_kp = _extract_face(results)
            parts.append(face_kp)

        return np.concatenate(parts, axis=0).astype(np.float32)

    # ------------------------------------------------------------------
    # Sequence extraction
    # ------------------------------------------------------------------

    def extract_sequence(self, frames: List[np.ndarray]) -> np.ndarray:
        """
        Extract keypoints from a list of RGB frames.

        Args:
            frames: List of uint8 arrays, each (H, W, 3) in RGB order.

        Returns:
            Float32 array of shape (T, D) where T = len(frames).
        """
        return np.stack([self.extract_frame(f) for f in frames], axis=0)


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------


def extract_keypoints_from_frames(
    frames: np.ndarray,
    include_face: bool = False,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
) -> np.ndarray:
    """
    Extract MediaPipe keypoints from a batch of RGB frames.

    This is a convenience wrapper that creates and closes a
    :class:`PoseExtractor` automatically.

    Args:
        frames: uint8 array of shape (T, H, W, 3) in RGB order.
        include_face: Whether to include face landmarks.
        min_detection_confidence: MediaPipe detection threshold.
        min_tracking_confidence: MediaPipe tracking threshold.

    Returns:
        Float32 array of shape (T, D).
    """
    with PoseExtractor(
        include_face=include_face,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    ) as extractor:
        return extractor.extract_sequence(list(frames))


# ---------------------------------------------------------------------------
# Private landmark helpers
# ---------------------------------------------------------------------------


def _extract_pose(results) -> np.ndarray:
    """Extract pose landmark array (shape 132)."""
    if results.pose_landmarks is not None:
        lm = results.pose_landmarks.landmark
        return np.array(
            [[l.x, l.y, l.z, l.visibility] for l in lm], dtype=np.float32
        ).flatten()
    return np.zeros(POSE_DIM, dtype=np.float32)


def _extract_hand(hand_landmarks) -> np.ndarray:
    """Extract one hand's landmark array (shape 63)."""
    if hand_landmarks is not None:
        lm = hand_landmarks.landmark
        return np.array(
            [[l.x, l.y, l.z] for l in lm], dtype=np.float32
        ).flatten()
    return np.zeros(HAND_DIM, dtype=np.float32)


def _extract_face(results) -> np.ndarray:
    """Extract face mesh landmark array (shape 1404)."""
    face_dim = 468 * 3
    if results.face_landmarks is not None:
        lm = results.face_landmarks.landmark
        return np.array(
            [[l.x, l.y, l.z] for l in lm], dtype=np.float32
        ).flatten()
    return np.zeros(face_dim, dtype=np.float32)
