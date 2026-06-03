# Implementation Plan — NanoTrack, Optical Flow, and GPU Acceleration

This plan outlines the integration of **NanoTrack** (lightweight Siamese tracker), **KLT Sparse Optical Flow** (for scale/rotation tracking), and key optimizations to achieve real-time performance (>60 FPS) in both the CLI and PySide GUI.

## User Review Required

> [!IMPORTANT]
> **GPU Acceleration Findings:**
> 1. **OpenCV CUDA**: The pre-built OpenCV wheel on this system (`opencv-contrib-python`) does not support CUDA backends (`getCudaEnabledDeviceCount() == 0`).
> 2. **PyTorch CUDA**: PyTorch **does** support CUDA on this system (`torch.cuda.is_available() == True`).
> 
> **Performance Solutions:**
> - **NanoTrack**: We will implement `cv2.TrackerNano` using pre-trained ONNX models (auto-downloaded on first run). Benchmarks show it runs at **~190 FPS on CPU**!
> - **GPU Trackers**: The deep Siamese trackers (`DASIARMPN` and `SIAMRPNPP`) will run on **PyTorch CUDA (GPU)**, yielding **100 - 220 FPS**.
> - **Optical Flow**: We will implement a CPU-based **KLT Sparse Optical Flow** helper. It runs at **~300+ FPS** (<1ms overhead) and tracks scale/rotation changes.
> - **GUI Bottleneck**: The GUI currently runs slow (~15 FPS) on large videos because it converts and scales full 1080p/4K frames on the main thread. We will optimize `gui.py` to downscale frames for rendering, unlocking **60+ FPS** GUI rendering.

---

## Proposed Changes

### Tracker Component

#### [MODIFY] [tracker.py](file:///c:/Users/Garv%20Sharma/Documents/gpu_fusion_tracker_v2/gpu_tracker/src/tracker.py)
- **NanoTrack Wrapper**:
  - Implement `NanoTrackerWrapper` utilizing `cv2.TrackerNano_create`.
  - Add auto-downloader logic to fetch `nanotrack_backbone_sim.onnx` and `nanotrack_head_sim.onnx` from GitHub if not found locally.
- **KLT Sparse Optical Flow**:
  - Integrate KLT keypoint tracking (`cv2.calcOpticalFlowPyrLK`) inside `FusionTracker`.
  - Sample features (`cv2.goodFeaturesToTrack`) inside the bounding box.
  - Track keypoints frame-to-frame, estimate scale/rotation/translation using `cv2.estimateAffinePartial2D` (RANSAC), and fuse the flow-based bounding box with the visual tracker and Kalman filter.
- **Visual Tracker Builder**:
  - Add `NANOTRACK` as a choice inside `_build_visual_tracker`.

---

### CLI Component

#### [MODIFY] [main.py](file:///c:/Users/Garv%20Sharma/Documents/gpu_fusion_tracker_v2/gpu_tracker/main.py)
- Add `NANOTRACK` to the `--algo` argument choices list.

---

### GUI Component

#### [MODIFY] [gui.py](file:///c:/Users/Garv%20Sharma/Documents/gpu_fusion_tracker_v2/gpu_tracker/gui.py)
- Add `NANOTRACK` to the algorithm drop-down menu in `_build_ui`.
- **Render Optimization**:
  - In `VideoCanvas.show_frame()`, if a video frame's resolution exceeds a maximum GUI window width (e.g. 960px), downscale it using `cv2.resize` *before* converting it to `QImage`/`QPixmap`. This saves significant CPU cycles and eliminates the GUI thread bottleneck, boosting playback to a smooth 60+ FPS.

---

### Testing Component

#### [MODIFY] [test_tracker.py](file:///c:/Users/Garv%20Sharma/Documents/gpu_fusion_tracker_v2/gpu_tracker/tests/test_tracker.py)
- Add test case verifying KLT optical flow update logic.
- Add test case verifying NanoTrack initialization and execution.

---

## Verification Plan

### Automated Tests
- Run `.venv\Scripts\python.exe -m pytest tests/test_tracker.py -v` to check correctness.

### Manual Verification
- Run `python main.py --video path/to/video.mp4 --algo NANOTRACK` to verify speed and re-detection.
- Launch `python gui.py` and run tracking with NANOTRACK to verify real-time GUI frame rate.
