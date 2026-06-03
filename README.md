# GPU Fusion Tracker

A GPU-accelerated fusion tracker for single-object tracking. This repository combines visual tracker outputs with a constant-velocity Kalman motion model to improve robustness during occlusion, camera motion, and tracker drift.

## Overview

This project supports both:

- `gui.py` — a PyQt6 desktop app with video preview, ROI selection, tracker algorithm switching, and confidence overlays.
- `main.py` — a command-line runner for headless tracking, video export, and CSV logging.

## Installation

```bash
cd gpu_tracker
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Dependencies

Required packages:

- `numpy>=1.24.0`
- `opencv-contrib-python>=4.8.0`
- `torch>=2.0.0`
- `torchvision>=0.15.0`
- `PyQt6>=6.5.0`
- `pytest>=7.4.0`

Optional GPU/CUDA packages:

- `opencv-contrib-python-cuda`
- CUDA-enabled PyTorch build for GPU acceleration

> Note: do not install `opencv-python` alongside `opencv-contrib-python`.

## Architecture

The tracker fuses:

- visual tracker results from OpenCV or PyTorch-based Siamese trackers
- a constant-velocity Kalman filter that predicts motion and stabilises during failures

Tracking modes:

- `tracker` — when the visual tracker provides a strong confidence signal
- `fused` — when the visual tracker is moderate and fused with motion prediction
- `motion_model` — when visual confidence is weak or the target is lost

### Fusion behavior

- `fused_bbox = 0.65 × visual_bbox + 0.35 × kf_corrected_bbox`
- source = `tracker` when visual confidence > 0.70
- source = `fused` when 0.35 ≤ visual confidence ≤ 0.70
- source = `motion_model` when visual confidence < 0.35

## Usage

### Run the GUI

```bash
cd gpu_tracker
.venv\Scripts\activate
python gui.py
```

Then open a video, draw the ROI, select the tracker algorithm, and press PLAY.

### Run headless CLI

```bash
cd gpu_tracker
.venv\Scripts\activate
python main.py --video input.mp4 --algo CSRT --output tracked.mp4 --csv tracking.csv
```

Common CLI options:

- `--video <path>` — input video file
- `--webcam <index>` — webcam device index
- `--algo <CSRT|KCF|MOSSE|MIL|DASIARMPN|SIAMRPNPP|NANOTRACK|OSTRACK|MIXFORMER>`
- `--bbox X Y W H` — initial ROI without interactive selection
- `--output <path>` — save annotated output video
- `--csv <path>` — save per-frame tracking data
- `--no-display` — disable preview window
- `--start-frame N` — start processing at frame N

## Testing

```bash
cd gpu_tracker
.venv\Scripts\activate
pytest tests/ -v
```

## Project layout

```
gpu_tracker/
├── gui.py
├── main.py
├── requirements.txt
├── README.md
├── src/
│   ├── __init__.py
│   ├── tracker.py
│   ├── pytorch_tracker.py
│   └── visualizer.py
└── tests/
    └── test_tracker.py
```

## Notes

- The system works in CPU-only mode by default.
- The GUI detects CUDA support for OpenCV and PyTorch if available.
- PyTorch trackers use backbone feature extraction and cosine similarity matching.

## License

MIT
