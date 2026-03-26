# brain-tumor-Deep_learning-PY

---

# Sign Language Recognition — WLASL Dataset

A PyTorch-based pipeline that converts sign language videos to text using
**MediaPipe** pose/hand keypoint extraction and either an **LSTM** or
**Transformer** sequence classifier.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Dataset](#dataset)
- [Usage](#usage)
  - [Training](#training)
  - [Inference](#inference)
  - [Webcam (Live)](#webcam-live)
- [Model Architectures](#model-architectures)
- [Evaluation Metrics](#evaluation-metrics)
- [Configuration Reference](#configuration-reference)

---

## Overview

| Component | Details |
|-----------|---------|
| Dataset | [WLASL](https://www.kaggle.com/datasets/risangbaskoro/wlasl-processed) (Word-Level American Sign Language) |
| Keypoints | MediaPipe Holistic — 33 pose + 21×2 hand landmarks → **258-dim** feature vector per frame |
| Models | Bidirectional LSTM with temporal attention **or** Transformer encoder with CLS pooling |
| Framework | PyTorch ≥ 2.0 |

---

## Project Structure

```
.
├── sign_language/               # Main Python package
│   ├── data/
│   │   ├── dataset.py           # WLASLDataset & KeypointDataset (PyTorch)
│   │   └── preprocessing.py     # Frame extraction, normalization helpers
│   ├── models/
│   │   ├── lstm_model.py        # Bidirectional LSTM + temporal attention
│   │   └── transformer_model.py # Transformer encoder (sinusoidal / learnable PE)
│   └── utils/
│       ├── pose_extraction.py   # MediaPipe Holistic keypoint extractor
│       └── metrics.py           # Accuracy, top-k, F1, confusion matrix
├── train.py                     # End-to-end training pipeline (CLI)
├── infer.py                     # Inference script — video / batch / webcam
├── requirements.txt
└── outputs/
    └── models/sign_language/    # Saved checkpoints & label maps (gitignored)
```

---

## Setup

```bash
# 1. Clone the repository
git clone <repo-url>
cd brain-tumor-Deep_learning-PY

# 2. Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Dataset

### Download from Kaggle

```bash
# Make sure you have a Kaggle API key at ~/.kaggle/kaggle.json
kaggle datasets download -d risangbaskoro/wlasl-processed -p data/wlasl --unzip
```

### Expected Layout

```
data/wlasl/
├── videos/                # Raw video clips (.mp4)
├── WLASL_v0.3.json        # Official metadata (train/val/test splits)
└── label_map.json         # {"hello": 0, "thank_you": 1, ...}
```

The `label_map.json` can be generated automatically from the WLASL metadata:

```python
import json

with open("data/wlasl/WLASL_v0.3.json") as f:
    data = json.load(f)

glosses = sorted({entry["gloss"] for entry in data})
label_map = {gloss: idx for idx, gloss in enumerate(glosses)}

with open("data/wlasl/label_map.json", "w") as f:
    json.dump(label_map, f, indent=2)

print(f"Created label map with {len(label_map)} classes.")
```

---

## Usage

### Training

```bash
# Train the default LSTM model (100 classes, 50 epochs)
python train.py \
    --data_root data/wlasl \
    --model lstm \
    --num_classes 100 \
    --epochs 50 \
    --batch_size 32 \
    --lr 1e-3 \
    --output_dir outputs/models/sign_language

# Train the Transformer model
python train.py \
    --data_root data/wlasl \
    --model transformer \
    --num_heads 8 \
    --num_layers 4 \
    --epochs 50 \
    --batch_size 16 \
    --output_dir outputs/models/sign_language
```

Checkpoints and training history are saved to `--output_dir`.

### Inference

```bash
# Single video -> predicted gloss
python infer.py \
    --checkpoint outputs/models/sign_language/best_model.pt \
    --label_map  outputs/models/sign_language/label_map.json \
    --video      path/to/sign_video.mp4

# Batch mode (directory of videos) -> JSON results file
python infer.py \
    --checkpoint outputs/models/sign_language/best_model.pt \
    --label_map  outputs/models/sign_language/label_map.json \
    --video_dir  path/to/videos/ \
    --output     predictions.json
```

### Webcam (Live)

```bash
python infer.py \
    --checkpoint outputs/models/sign_language/best_model.pt \
    --label_map  outputs/models/sign_language/label_map.json \
    --webcam
```

Press **`q`** to quit.

---

## Model Architectures

### LSTM (`SignLSTM`)

| Hyperparameter | Default | Description |
|---|---|---|
| `input_dim` | 258 | Keypoint feature dimension |
| `hidden_dim` | 256 | LSTM hidden size (per direction) |
| `num_layers` | 3 | Stacked LSTM layers |
| `bidirectional` | True | Use bidirectional LSTM |
| `use_attention` | True | Temporal attention pooling |
| `dropout` | 0.3 | Dropout rate |

**Architecture:**
```
Input (B, T, 258)
  -> LayerNorm
  -> Bidirectional LSTM x 3
  -> Temporal Attention pooling -> (B, 512)
  -> LayerNorm -> Dropout -> Linear(512->256) -> GELU -> Dropout -> Linear(256->C)
```

### Transformer (`SignTransformer`)

| Hyperparameter | Default | Description |
|---|---|---|
| `input_dim` | 258 | Keypoint feature dimension |
| `model_dim` | 256 | Transformer embedding dimension |
| `num_heads` | 8 | Attention heads |
| `num_layers` | 4 | Encoder layers |
| `positional_encoding` | sinusoidal | `"sinusoidal"` or `"learnable"` |
| `pooling` | cls | `"cls"` or `"mean"` |
| `dropout` | 0.1 | Dropout rate |

**Architecture:**
```
Input (B, T, 258)
  -> Linear projection -> LayerNorm
  -> Prepend [CLS] token
  -> Sinusoidal / Learnable positional encoding
  -> Transformer Encoder (Pre-LN, GELU) x 4
  -> CLS token output -> LayerNorm -> Linear -> GELU -> Dropout -> Linear(->C)
```

---

## Evaluation Metrics

The training loop reports after every epoch:

| Metric | Description |
|--------|-------------|
| `loss` | Cross-entropy with label smoothing |
| `accuracy` | Top-1 accuracy |
| `top5_accuracy` | Top-5 accuracy |
| `precision` | Macro-averaged precision |
| `recall` | Macro-averaged recall |
| `f1` | Macro-averaged F1 score |

---

## Configuration Reference

Run `python train.py --help` or `python infer.py --help` for the full list of
command-line options.