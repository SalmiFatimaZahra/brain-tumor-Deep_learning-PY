"""
Dataset utilities for the video-to-text model.

Provides:
  - Vocabulary: builds and manages a word-level vocabulary from captions.
  - VideoTextDataset: loads (video_path, caption) pairs, extracts frames from
    each video, and returns (frames_tensor, caption_indices) pairs suitable for
    training the encoder-decoder model.
"""

import os
import json
import collections
import re

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


# ---------------------------------------------------------------------------
# Special tokens
# ---------------------------------------------------------------------------
PAD_TOKEN = "<PAD>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"
UNK_TOKEN = "<UNK>"


class Vocabulary:
    """Word-level vocabulary built from a list of caption strings."""

    def __init__(self, freq_threshold: int = 1):
        self.freq_threshold = freq_threshold
        self.word2idx = {
            PAD_TOKEN: 0,
            SOS_TOKEN: 1,
            EOS_TOKEN: 2,
            UNK_TOKEN: 3,
        }
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self._freq: collections.Counter = collections.Counter()

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.word2idx)

    # ------------------------------------------------------------------
    @staticmethod
    def tokenize(text: str) -> list[str]:
        """Lowercase and split on whitespace/punctuation."""
        return re.findall(r"\w+", text.lower())

    # ------------------------------------------------------------------
    def build(self, captions: list[str]) -> None:
        """Populate the vocabulary from a list of caption strings."""
        for caption in captions:
            self._freq.update(self.tokenize(caption))

        for word, count in self._freq.items():
            if count >= self.freq_threshold and word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

    # ------------------------------------------------------------------
    def encode(self, caption: str) -> list[int]:
        """Convert a caption string to a list of token indices."""
        tokens = self.tokenize(caption)
        return (
            [self.word2idx[SOS_TOKEN]]
            + [self.word2idx.get(t, self.word2idx[UNK_TOKEN]) for t in tokens]
            + [self.word2idx[EOS_TOKEN]]
        )

    # ------------------------------------------------------------------
    def decode(self, indices: list[int]) -> str:
        """Convert a list of token indices back to a sentence string."""
        words = []
        for idx in indices:
            word = self.idx2word.get(idx, UNK_TOKEN)
            if word in (PAD_TOKEN, SOS_TOKEN, EOS_TOKEN):
                continue
            words.append(word)
        return " ".join(words)

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"word2idx": self.word2idx, "freq_threshold": self.freq_threshold}, f)

    @classmethod
    def load(cls, path: str) -> "Vocabulary":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        vocab = cls(freq_threshold=data["freq_threshold"])
        vocab.word2idx = data["word2idx"]
        vocab.idx2word = {int(v): k for k, v in data["word2idx"].items()}
        return vocab


# ---------------------------------------------------------------------------
# Default frame transform
# ---------------------------------------------------------------------------
DEFAULT_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def extract_frames(video_path: str, num_frames: int = 16, transform=None) -> torch.Tensor:
    """
    Extract ``num_frames`` evenly-spaced frames from a video file.

    Returns a tensor of shape ``(num_frames, C, H, W)``.
    """
    if transform is None:
        transform = DEFAULT_TRANSFORM

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        raise ValueError(f"Video has no readable frames: {video_path}")

    indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=int)
    frames = []
    last_valid_frame = np.zeros((224, 224, 3), dtype=np.uint8)
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            # Repeat the last valid raw frame (or a blank frame if none yet)
            frame = last_valid_frame
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            last_valid_frame = frame
        pil_img = Image.fromarray(frame)
        frames.append(transform(pil_img))

    cap.release()
    return torch.stack(frames)  # (T, C, H, W)


# ---------------------------------------------------------------------------
# Collate helper
# ---------------------------------------------------------------------------
def collate_fn(batch, pad_idx: int = 0):
    """
    Collate a list of (frames, caption_indices) tuples into batched tensors.

    Pads captions to the length of the longest caption in the batch.
    """
    frames_list, captions_list = zip(*batch)
    frames = torch.stack(frames_list)  # (B, T, C, H, W)

    max_len = max(len(c) for c in captions_list)
    padded = torch.full((len(captions_list), max_len), pad_idx, dtype=torch.long)
    for i, cap in enumerate(captions_list):
        padded[i, : len(cap)] = torch.tensor(cap, dtype=torch.long)

    return frames, padded


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class VideoTextDataset(Dataset):
    """
    Dataset that pairs video files with text captions.

    The annotation file is a JSON file with the following structure::

        [
            {"video": "path/to/video.mp4", "caption": "a description of the video"},
            ...
        ]

    Args:
        annotations_file: Path to the JSON annotations file described above.
        vocab: A pre-built :class:`Vocabulary`.  If ``None``, a new vocabulary
               is built from all captions in the annotations file.
        num_frames: Number of frames to extract uniformly from each video.
        transform: Optional torchvision transform applied to each frame.
        freq_threshold: Minimum word frequency when building a new vocabulary.
    """

    def __init__(
        self,
        annotations_file: str,
        vocab: Vocabulary | None = None,
        num_frames: int = 16,
        transform=None,
        freq_threshold: int = 1,
    ):
        with open(annotations_file, "r", encoding="utf-8") as f:
            self.annotations = json.load(f)

        self.num_frames = num_frames
        self.transform = transform or DEFAULT_TRANSFORM

        if vocab is None:
            vocab = Vocabulary(freq_threshold=freq_threshold)
            vocab.build([ann["caption"] for ann in self.annotations])
        self.vocab = vocab

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.annotations)

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int):
        ann = self.annotations[idx]
        frames = extract_frames(ann["video"], num_frames=self.num_frames, transform=self.transform)
        caption_indices = self.vocab.encode(ann["caption"])
        return frames, caption_indices
