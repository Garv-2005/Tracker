# GPU Fusion Tracker

A GPU-accelerated fusion tracker for single-object tracking.
This repository combines visual tracker outputs with a constant-velocity Kalman motion model, template redetection, and optional PyTorch Siamese trackers to improve robustness during occlusion, camera motion, and tracker drift.

## Overview

This project contains two main entry points:

- `gui.py` — PyQt6 desktop application for interactive video loading, ROI selection, tracker selection, and live visualization.
- `main.py` — command-line tracking tool for headless batch processing, output video generation, and CSV logging.

The tracking engine is implemented in `src/tracker.py`, supported by `src/pytorch_tracker.py` for advanced Siamese models and `src/visualizer.py` for overlay rendering.

## Installation

```bash
cd gpu_tracker
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Optional GPU setup

For better performance with OpenCV or PyTorch on CUDA:

```bash
pip install opencv-contrib-python-cuda
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118
```

> Only install `opencv-contrib-python` or `opencv-contrib-python-cuda`, not both.

## Dependency list

Required packages:

- `numpy>=1.24.0`
- `opencv-contrib-python>=4.8.0`
- `torch>=2.0.0`
- `torchvision>=0.15.0`
- `PyQt6>=6.5.0`
- `pytest>=7.4.0`

Optional packages for GPU/CUDA acceleration:

- `opencv-contrib-python-cuda`
- CUDA-enabled PyTorch build

## Architecture

The tracker is built around a fusion design with three complementary subsystems:

1. Visual tracking
2. Motion prediction
3. Recovery and redetection

### High-level architecture

```
+-----------------------------+      visual bbox      +---------------------+
|  Visual tracker subsystem   | --------------------> |                     |
|  (OpenCV + PyTorch wrappers)|                       | FusionTracker       |
+-----------------------------+                       |                     |
         |                                              |  - fuse bboxes      |
         | tracker confidence                            |  - manage loss      |
         v                                              |  - update state     |
+-----------------------------+      measurement       +---------------------+
|  ConstantVelocityKalman     | <--------------------  |
|  motion model subsystem     |                       |
+-----------------------------+                       |
         |                                              |
         | predicted bbox                               |
         v                                              |
+-----------------------------+      redetection       |
|  Template / motion recovery | <--------------------  |
|  & global motion estimation |                       |
+-----------------------------+                       |
```

### Component responsibilities

- `src/tracker.py`
  - `FusionTracker` manages frame-by-frame tracking, fusion logic, loss detection, and recovery.
  - `ConstantVelocityKalman` predicts target motion using position and velocity in the state vector.
  - `_build_visual_tracker()` selects between OpenCV trackers, NanoTrack, or PyTorch Siamese wrappers.

- `src/pytorch_tracker.py`
  - Implements a lightweight Siamese tracker core for `DASIARMPN`, `SIAMRPNPP`, `OSTRACK`, and `MIXFORMER`.
  - Uses pre-trained backbone feature extractors from `torchvision`.
  - Performs crop-and-pad preprocessing, feature extraction, and response map matching.

- `src/visualizer.py`
  - Draws bounding boxes, source labels, confidence bars, and FPS overlays on frames.

### Fusion logic

The fusion strategy selects the most reliable source dynamically:

- `tracker` when visual confidence is strong (high appearance match)
- `fused` when visual confidence is moderate
- `motion_model` when visual confidence is weak or the tracker fails

A fused bounding box is computed as:

```text
fused_bbox = 0.65 × visual_bbox + 0.35 × kf_corrected_bbox
```

The tracker confidence is derived from template matching scores and rolling NCC evaluation. The motion confidence decays when the system relies solely on prediction.

### Recovery and redetection

When the tracker loses the target, the system attempts recovery using:

- optical flow on recent keypoints inside the target region
- global frame affine motion estimation to handle camera pan/zoom
- motion-compensated anchor generation
- template matching search over a large window
- reinitializing the visual tracker when a reliable candidate is found

## Execution steps

### GUI mode

1. Activate the environment:

```bash
cd gpu_tracker
.venv\Scripts\activate
```

2. Launch the GUI:

```bash
python gui.py
```

3. In the GUI:
   - Click `OPEN VIDEO FILE` or drag-and-drop a video.
   - Use the `SELECT ROI` button and draw a box around the object to track.
   - Choose the tracker algorithm from the dropdown.
   - Press `PLAY` to begin tracking.

4. During tracking:
   - use the timeline slider or frame selector to jump to a specific frame.
   - reselect ROI mid-stream by pausing and drawing a new box.
   - watch the source badge and confidence bars to understand whether the tracker, fusion, or motion model is trusted.

### CLI mode

#### Basic run

```bash
cd gpu_tracker
.venv\Scripts\activate
python main.py --video input.mp4 --algo CSRT
```

#### Export annotated video

```bash
python main.py --video input.mp4 --algo CSRT --output tracked.mp4
```

#### Export tracking data to CSV

```bash
python main.py --video input.mp4 --algo CSRT --csv tracking.csv
```

#### Start from a custom frame

```bash
python main.py --video input.mp4 --algo CSRT --start-frame 100
```

#### Use a webcam

```bash
python main.py --webcam 0 --algo CSRT
```

### CLI arguments

- `--video <path>`: input video file path.
- `--webcam <index>`: use webcam capture device.
- `--algo <...>`: visual tracker algorithm.
- `--bbox X Y W H`: initialize ROI without interactive selection.
- `--output <path>`: save annotated video result.
- `--csv <path>`: write per-frame tracking logs.
- `--no-display`: run without preview window.
- `--start-frame N`: begin processing at frame N.

### Output CSV format

The CLI CSV contains:

- `frame_id`
- `x`, `y`, `w`, `h`
- `tracker_confidence`
- `motion_confidence`
- `trusted_source`
- `proc_ms`

## Testing

Run unit tests with:

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

- The project supports CPU-only execution, and will fall back if no CUDA device is available.
- The GUI automatically detects whether OpenCV or PyTorch CUDA backends are available.
- The tracker is designed for single-object video sequences and uses both appearance and motion cues.

## Limitations

1. **Motion model logic** — The constant-velocity Kalman filter relies purely on object velocity extracted from frame coordinates. Future improvements could integrate IMU and gyroscope data from the camera to estimate global motion (pan, tilt, roll) more accurately, reducing false drift during camera movement.

2. **GPU acceleration** — OpenCV's native CUDA support is limited. Many OpenCV operations fall back to CPU execution. Further acceleration is possible by using custom CUDA wheels or re-implementing performance-critical functions (template matching, optical flow) with custom kernels for better GPU utilization.

## License

MIT
