# brain-tumor-Deep_learning-PY

## Video-to-Text Model

This repository includes a deep-learning pipeline that takes **video files as
input** and generates **text descriptions** (captions) for each video.

### Architecture

| Component | Description |
|-----------|-------------|
| **CNN backbone** | Pretrained ResNet-50 — extracts a feature vector per frame |
| **Temporal encoder** | Bidirectional LSTM — aggregates frame features across time |
| **Attention** | Bahdanau (additive) attention — lets the decoder focus on relevant frames |
| **Text decoder** | LSTM with word embeddings — generates one word per step |

### Installation

```bash
pip install -r requirements.txt
```

### Dataset format

Create a JSON file (e.g. `data/annotations.json`) where each entry maps a
video file to its caption:

```json
[
  {"video": "data/videos/clip01.mp4", "caption": "a doctor examines the brain scan"},
  {"video": "data/videos/clip02.mp4", "caption": "the tumor appears on the right hemisphere"}
]
```

### Training

```bash
python -m video_to_text.train \
    --annotations data/annotations.json \
    --output_dir  outputs/models/ \
    --epochs      30 \
    --batch_size  4
```

Key options:

| Flag | Default | Description |
|------|---------|-------------|
| `--annotations` | *(required)* | Path to JSON annotations file |
| `--output_dir` | `outputs/models/` | Directory for saved checkpoints |
| `--epochs` | `30` | Number of training epochs |
| `--batch_size` | `4` | Mini-batch size |
| `--num_frames` | `16` | Frames sampled per video |
| `--lr` | `3e-4` | Initial learning rate |
| `--resume` | `None` | Path to checkpoint to resume from |

### Inference

```bash
python -m video_to_text.predict \
    --checkpoint outputs/models/best.pt \
    --videos     clip1.mp4 clip2.mp4
```

### Project structure

```
video_to_text/
  __init__.py   — public API
  dataset.py    — Vocabulary + VideoTextDataset + frame extraction
  model.py      — VideoEncoder + TextDecoder + VideoToTextModel
  train.py      — training loop with validation and checkpointing
  predict.py    — inference / caption generation
requirements.txt
outputs/
  figures/      — plots and visualisations
  models/       — saved model checkpoints (git-ignored)
```