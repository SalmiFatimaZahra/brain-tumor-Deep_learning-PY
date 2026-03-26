"""
Evaluation metrics for sign language recognition.

Provides:
- Accuracy (top-1 and top-k)
- Precision, Recall, F1-score (macro / weighted)
- Confusion matrix helpers
- Per-epoch metric accumulator
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Basic metrics
# ---------------------------------------------------------------------------


def accuracy(
    preds: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray],
) -> float:
    """
    Compute top-1 accuracy.

    Args:
        preds: Predicted class indices, shape (N,).
        targets: Ground-truth class indices, shape (N,).

    Returns:
        Scalar accuracy in [0, 1].
    """
    preds = _to_numpy(preds).ravel()
    targets = _to_numpy(targets).ravel()
    if len(preds) == 0:
        return 0.0
    return float(np.mean(preds == targets))


def top_k_accuracy(
    logits: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray],
    k: int = 5,
) -> float:
    """
    Compute top-k accuracy.

    Args:
        logits: Raw model outputs, shape (N, C).
        targets: Ground-truth class indices, shape (N,).
        k: Number of top predictions to consider.

    Returns:
        Scalar top-k accuracy in [0, 1].
    """
    logits = _to_numpy(logits)
    targets = _to_numpy(targets).ravel()
    if len(targets) == 0:
        return 0.0
    top_k_preds = np.argsort(logits, axis=1)[:, -k:]  # (N, k) ascending
    correct = np.any(top_k_preds == targets[:, None], axis=1)
    return float(np.mean(correct))


def precision_recall_f1(
    preds: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray],
    num_classes: int,
    average: str = "macro",
) -> Dict[str, float]:
    """
    Compute precision, recall, and F1-score.

    Args:
        preds: Predicted class indices, shape (N,).
        targets: Ground-truth class indices, shape (N,).
        num_classes: Total number of classes.
        average: ``"macro"`` or ``"weighted"``.

    Returns:
        Dict with keys ``"precision"``, ``"recall"``, ``"f1"``.
    """
    preds = _to_numpy(preds).ravel()
    targets = _to_numpy(targets).ravel()

    precisions, recalls, f1s, supports = [], [], [], []
    for c in range(num_classes):
        tp = int(np.sum((preds == c) & (targets == c)))
        fp = int(np.sum((preds == c) & (targets != c)))
        fn = int(np.sum((preds != c) & (targets == c)))
        support = tp + fn

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        precisions.append(p)
        recalls.append(r)
        f1s.append(f)
        supports.append(support)

    total = sum(supports)
    if average == "macro":
        w = np.ones(num_classes) / num_classes
    elif average == "weighted":
        w = np.array(supports, dtype=float) / (total if total > 0 else 1)
    else:
        raise ValueError(f"average must be 'macro' or 'weighted', got '{average}'")

    return {
        "precision": float(np.dot(w, precisions)),
        "recall": float(np.dot(w, recalls)),
        "f1": float(np.dot(w, f1s)),
    }


def confusion_matrix(
    preds: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray],
    num_classes: int,
) -> np.ndarray:
    """
    Compute the confusion matrix.

    Args:
        preds: Predicted class indices, shape (N,).
        targets: Ground-truth class indices, shape (N,).
        num_classes: Total number of classes.

    Returns:
        Integer array of shape (num_classes, num_classes).
        ``matrix[i, j]`` = number of samples with true class i predicted as j.
    """
    preds = _to_numpy(preds).ravel()
    targets = _to_numpy(targets).ravel()
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(targets, preds):
        if 0 <= int(t) < num_classes and 0 <= int(p) < num_classes:
            cm[int(t), int(p)] += 1
    return cm


# ---------------------------------------------------------------------------
# Per-epoch metric accumulator
# ---------------------------------------------------------------------------


class MetricAccumulator:
    """
    Accumulates per-batch predictions and computes epoch-level metrics.

    Example::

        acc = MetricAccumulator(num_classes=100)
        for batch in loader:
            logits = model(batch)
            loss = criterion(logits, labels)
            acc.update(logits, labels, loss.item())

        epoch_metrics = acc.compute()
        print(epoch_metrics)
    """

    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self._preds: List[np.ndarray] = []
        self._targets: List[np.ndarray] = []
        self._logits: List[np.ndarray] = []
        self._losses: List[float] = []
        self._counts: List[int] = []

    def reset(self) -> None:
        """Clear all accumulated values."""
        self._preds.clear()
        self._targets.clear()
        self._logits.clear()
        self._losses.clear()
        self._counts.clear()

    def update(
        self,
        logits: Union[torch.Tensor, np.ndarray],
        targets: Union[torch.Tensor, np.ndarray],
        loss: Optional[float] = None,
    ) -> None:
        """
        Add a batch of predictions.

        Args:
            logits: Raw model outputs, shape (N, C).
            targets: Ground-truth class indices, shape (N,).
            loss: Optional scalar batch loss.
        """
        logits_np = _to_numpy(logits)
        targets_np = _to_numpy(targets).ravel()
        preds_np = np.argmax(logits_np, axis=1)

        self._logits.append(logits_np)
        self._preds.append(preds_np)
        self._targets.append(targets_np)
        n = len(targets_np)
        self._counts.append(n)
        if loss is not None:
            self._losses.append(loss * n)

    def compute(self) -> Dict[str, float]:
        """
        Compute epoch-level metrics from accumulated batches.

        Returns:
            Dict with keys ``"accuracy"``, ``"top5_accuracy"``,
            ``"precision"``, ``"recall"``, ``"f1"``,
            and optionally ``"loss"``.
        """
        if not self._preds:
            return {}

        all_preds = np.concatenate(self._preds)
        all_targets = np.concatenate(self._targets)
        all_logits = np.concatenate(self._logits)

        results: Dict[str, float] = {
            "accuracy": accuracy(all_preds, all_targets),
            "top5_accuracy": top_k_accuracy(all_logits, all_targets, k=5),
        }
        results.update(
            precision_recall_f1(all_preds, all_targets, self.num_classes)
        )

        if self._losses:
            total_samples = sum(self._counts)
            results["loss"] = sum(self._losses) / (total_samples or 1)

        return results


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _to_numpy(x: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)
