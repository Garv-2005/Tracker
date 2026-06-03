"""
GPU-Accelerated Object Tracker with Constant Velocity Motion Model
==================================================================
Combines OpenCV CUDA-backed tracker with a Kalman Filter (constant
velocity) for smooth, occlusion-robust single-object tracking.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from pytorch_tracker import PyTorchSiameseTracker


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TrackState:
    """Output state for a single frame."""
    bbox: Tuple[int, int, int, int]          # x, y, w, h  (pixel coords)
    tracker_confidence: float                # 0-1  (visual tracker score)
    motion_confidence: float                 # 0-1  (Kalman prediction quality)
    trusted_source: str                      # "tracker" | "motion_model" | "fused"
    frame_id: int = 0
    timestamp_ms: float = 0.0


# ---------------------------------------------------------------------------
# Constant-Velocity Kalman Filter (motion model)
# ---------------------------------------------------------------------------

class ConstantVelocityKalman:
    """
    State vector: [cx, cy, w, h, vx, vy, vw, vh]
    Measures:     [cx, cy, w, h]
    """

    def __init__(self, initial_bbox: Tuple[int, int, int, int]):
        self.kf = cv2.KalmanFilter(8, 4)

        # Transition matrix (constant velocity)
        dt = 1.0
        F = np.eye(8, dtype=np.float32)
        for i in range(4):
            F[i, i + 4] = dt
        self.kf.transitionMatrix = F

        # Measurement matrix
        H = np.zeros((4, 8), dtype=np.float32)
        H[:4, :4] = np.eye(4)
        self.kf.measurementMatrix = H

        # Noise covariances
        self.kf.processNoiseCov = np.eye(8, dtype=np.float32) * 1e-2
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1e-1
        self.kf.errorCovPost = np.eye(8, dtype=np.float32)

        # Initialise state
        x, y, w, h = initial_bbox
        cx, cy = x + w / 2, y + h / 2
        self.kf.statePost = np.array(
            [cx, cy, w, h, 0, 0, 0, 0], dtype=np.float32
        ).reshape(-1, 1)

        # Track how many consecutive frames we've gone without a measurement
        self.miss_count: int = 0
        self._max_miss: int = 30          # frames before confidence bottoms out
        self._predict_only_streak: int = 0

    # ------------------------------------------------------------------
    def predict(self) -> Tuple[int, int, int, int]:
        """Predict next bounding box without a measurement update."""
        pred = self.kf.predict()
        self._predict_only_streak += 1
        return self._state_to_bbox(pred)

    def update(self, bbox: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """Correct with a fresh visual measurement."""
        x, y, w, h = bbox
        cx, cy = x + w / 2, y + h / 2
        meas = np.array([cx, cy, w, h], dtype=np.float32).reshape(-1, 1)
        corrected = self.kf.correct(meas)
        self.miss_count = 0
        self._predict_only_streak = 0
        return self._state_to_bbox(corrected)

    @property
    def confidence(self) -> float:
        """Decays the longer we rely solely on prediction."""
        decay = max(0.0, 1.0 - self._predict_only_streak / self._max_miss)
        return float(decay)

    @staticmethod
    def _state_to_bbox(state: np.ndarray) -> Tuple[int, int, int, int]:
        cx, cy, w, h = state[:4, 0]
        w, h = max(w, 1), max(h, 1)
        x = int(cx - w / 2)
        y = int(cy - h / 2)
        return (x, y, int(w), int(h))


# ---------------------------------------------------------------------------
# GPU-Accelerated Visual Tracker wrapper
# ---------------------------------------------------------------------------

class PyTorchTrackerWrapper:
    def __init__(self, model_name: str, device: str = "cpu"):
        self.tracker = PyTorchSiameseTracker(model_name=model_name, device=device)
        self.is_initialized = False

    def init(self, frame, bbox):
        self.tracker.init(frame, bbox)
        self.is_initialized = True

    def update(self, frame, predicted_center=None, min_score: float = 0.05):
        ok, bbox, _score = self.update_with_score(
            frame, predicted_center=predicted_center, min_score=min_score
        )
        return ok, bbox

    def update_with_score(self, frame, predicted_center=None, min_score: float = 0.05):
        if not self.is_initialized:
            return False, (0, 0, 0, 0), 0.0
        try:
            if predicted_center is not None:
                bbox, score = self.tracker.update(frame, predicted_center=predicted_center)
            else:
                bbox, score = self.tracker.update(frame)
            score = float(score)
            ok = score > min_score
            return ok, bbox, score
        except Exception as e:
            print(f"[PyTorchTrackerWrapper] Error: {e}")
            return False, (0, 0, 0, 0), 0.0


class NanoTrackerWrapper:
    def __init__(self):
        self.is_initialized = False
        self.tracker = None
        
        # Locate or download models
        base_dir = Path(__file__).parent.parent
        backbone_path = base_dir / "nanotrack_backbone_sim.onnx"
        head_path = base_dir / "nanotrack_head_sim.onnx"
        
        # Also check current working directory
        if not backbone_path.exists() and Path("nanotrack_backbone_sim.onnx").exists():
            backbone_path = Path("nanotrack_backbone_sim.onnx")
        if not head_path.exists() and Path("nanotrack_head_sim.onnx").exists():
            head_path = Path("nanotrack_head_sim.onnx")
            
        if not backbone_path.exists():
            print(f"[NanoTrack] Downloading backbone model to {backbone_path}...")
            url = "https://github.com/HonglinChu/SiamTrackers/raw/master/NanoTrack/models/nanotrackv2/nanotrack_backbone_sim.onnx"
            import urllib.request
            urllib.request.urlretrieve(url, str(backbone_path))
            
        if not head_path.exists():
            print(f"[NanoTrack] Downloading head model to {head_path}...")
            url = "https://github.com/HonglinChu/SiamTrackers/raw/master/NanoTrack/models/nanotrackv2/nanotrack_head_sim.onnx"
            import urllib.request
            urllib.request.urlretrieve(url, str(head_path))
            
        try:
            params = cv2.TrackerNano_Params()
            params.backbone = str(backbone_path.resolve())
            params.neckhead = str(head_path.resolve())
            
            # GPU/CUDA acceleration if OpenCV was built with CUDA DNN backend
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                try:
                    params.backend = cv2.dnn.DNN_BACKEND_CUDA
                    params.target = cv2.dnn.DNN_TARGET_CUDA
                    print("[NanoTrack] Using GPU/CUDA DNN backend for NanoTrack.")
                except Exception as ex:
                    print(f"[NanoTrack] Failed to set CUDA backend (falling back to CPU): {ex}")
            else:
                print("[NanoTrack] Using CPU DNN backend for NanoTrack.")
                
            self.tracker = cv2.TrackerNano_create(params)
        except Exception as e:
            print(f"[NanoTrack] Error creating TrackerNano: {e}")
            self.tracker = None

    def init(self, frame, bbox):
        if self.tracker is not None:
            self.tracker.init(frame, bbox)
            self.is_initialized = True

    def update(self, frame, predicted_center=None):
        if not self.is_initialized or self.tracker is None:
            return False, (0, 0, 0, 0)
        try:
            ok, bbox = self.tracker.update(frame)
            if ok:
                bbox = tuple(int(v) for v in bbox)
            return ok, bbox
        except Exception as e:
            print(f"[NanoTrack] Error during update: {e}")
            return False, (0, 0, 0, 0)


def _build_visual_tracker(algo: str = "CSRT") -> cv2.Tracker:
    """
    Try to create a CUDA-backed tracker; fall back to CPU if unavailable.
    OpenCV CUDA trackers live in opencv-contrib and require a CUDA build.
    We transparently fall back to the CPU variant so the code is always runnable.
    """
    algo_upper = algo.upper()
    if algo_upper in ("DASIARMPN", "DASIARMPM", "SIAMRPNPP", "OSTRACK", "MIXFORMER"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Tracker] Using PyTorch {algo_upper} tracker on {device}.")
        return PyTorchTrackerWrapper(algo_upper, device)

    if algo_upper == "NANOTRACK":
        print("[Tracker] Using NanoTrack tracker.")
        return NanoTrackerWrapper()

    cuda_available = cv2.cuda.getCudaEnabledDeviceCount() > 0

    if cuda_available:
        try:
            # contrib CUDA tracker (OpenCV ≥ 4.5 with CUDA contrib)
            tracker = cv2.cuda.TrackerCSRT_create()  # type: ignore[attr-defined]
            print("[Tracker] Using CUDA-accelerated CSRT tracker.")
            return tracker
        except AttributeError:
            pass

    # CPU fallback — standard contrib trackers
    builders = {
        "CSRT":  cv2.TrackerCSRT_create,
        "KCF":   cv2.TrackerKCF_create,
        "MOSSE": cv2.legacy.TrackerMOSSE_create,
        "MIL":   cv2.TrackerMIL_create,
    }
    builder = builders.get(algo_upper, cv2.TrackerCSRT_create)
    label = "CUDA-fallback→CPU" if cuda_available else "CPU"
    print(f"[Tracker] Using {label} {algo_upper} tracker.")
    return builder()


# ---------------------------------------------------------------------------
# Fusion tracker
# ---------------------------------------------------------------------------

CONFIDENCE_THRESHOLD = 0.35   # below this we distrust the visual tracker
FUSE_ALPHA           = 0.65   # weight given to visual tracker when fusing

# Re-detection while lost — motion-aware (camera pan + moving target)
REDETECT_INTERVAL          = 1
REDETECT_MAX_SEARCH_SIDE   = 480
REDETECT_SEARCH_SCALE      = 3.5   # search radius vs target size (large for pan/zoom)
REDETECT_MAX_TEMPLATES     = 2
REDETECT_SCALES            = (0.90, 1.0, 1.10)
REDETECT_MATCH_THRESH      = 0.50  # raw matchTemplate peak (before NCC verify)
REDETECT_RELOCATE_SCORE    = 0.15  # min correlation for Siamese relocate pass
GLOBAL_MOTION_MAX_CORNERS  = 150
GLOBAL_MOTION_MIN_INLIERS  = 12


class FusionTracker:
    """
    Fuses a GPU-backed visual tracker with a Kalman constant-velocity model.

    Trust logic:
      - If visual tracker succeeds and has good confidence → fuse both.
      - If visual tracker fails or confidence is low → rely on motion model only.
      - Motion model is *always* updated (predict every frame, correct when tracker fires).
    """

    def __init__(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        algo: str = "CSRT",
    ):
        self._algo = algo
        self._is_lost = False

        # Scale handling for CPU-bound OpenCV trackers to speed up frame processing
        max_dim = 640
        h, w = frame.shape[:2]
        self._scale = 1.0
        # PyTorch trackers compute on crop and don't need downscaling of the full frame
        if algo.upper() not in ("DASIARMPN", "DASIARMPM", "SIAMRPNPP") and max(h, w) > max_dim:
            self._scale = max_dim / max(h, w)

        if self._scale < 1.0:
            scaled_frame = cv2.resize(frame, (int(w * self._scale), int(h * self._scale)))
            scaled_bbox = (
                int(bbox[0] * self._scale),
                int(bbox[1] * self._scale),
                int(bbox[2] * self._scale),
                int(bbox[3] * self._scale),
            )
        else:
            scaled_frame = frame
            scaled_bbox = bbox

        self._visual = _build_visual_tracker(algo)
        self._visual.init(scaled_frame, scaled_bbox)

        self._kalman = ConstantVelocityKalman(bbox)
        self._frame_id = 0
        self._last_good_bbox = bbox

        # Cache the initial patch and setup template pool
        x, y, w, h = bbox
        initial_patch = frame[y : y + h, x : x + w]
        if initial_patch.size > 0:
            self._init_gray_patch = cv2.cvtColor(initial_patch, cv2.COLOR_BGR2GRAY)
            self._last_good_gray_patch = self._init_gray_patch.astype(np.float32)
            
            # KLT features in initial frame
            self._prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            fh_prev, fw_prev = self._prev_gray.shape[:2]
            pad = 5
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(fw_prev, x + w + pad)
            y2 = min(fh_prev, y + h + pad)
            patch = self._prev_gray[y1:y2, x1:x2]
            if patch.size > 0:
                pts = cv2.goodFeaturesToTrack(patch, maxCorners=50, qualityLevel=0.01, minDistance=5)
                if pts is not None:
                    pts[:, 0, 0] += x1
                    pts[:, 0, 1] += y1
                    self._klt_pts = pts
                else:
                    self._klt_pts = np.array([], dtype=np.float32).reshape(-1, 1, 2)
            else:
                self._klt_pts = np.array([], dtype=np.float32).reshape(-1, 1, 2)
        else:
            self._init_gray_patch = None
            self._last_good_gray_patch = None
            self._prev_gray = None
            self._klt_pts = np.array([], dtype=np.float32).reshape(-1, 1, 2)

        self._template_pool = []
        if self._init_gray_patch is not None:
            self._template_pool.append(self._init_gray_patch)
        self._max_pool_size = 5

        self._lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        # Running average of tracker response to estimate confidence
        self._response_history: list[float] = []
        self._history_len = 10
        self._lost_frame_count = 0
        self._global_affine: Optional[np.ndarray] = None
        self._global_motion_inliers = 0

    # ------------------------------------------------------------------
    @staticmethod
    def _warp_bbox_affine(
        bbox: Tuple[int, int, int, int],
        m: np.ndarray,
        frame_shape: Tuple[int, ...],
    ) -> Tuple[int, int, int, int]:
        """Apply 2x3 affine (prev -> current) to an axis-aligned bbox."""
        x, y, w, h = bbox
        corners = np.array(
            [[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32
        )
        warped = cv2.transform(corners.reshape(1, -1, 2), m).reshape(-1, 2)
        xs, ys = warped[:, 0], warped[:, 1]
        nx = int(round(float(xs.min())))
        ny = int(round(float(ys.min())))
        nw = int(round(float(xs.max() - xs.min())))
        nh = int(round(float(ys.max() - ys.min())))
        fh, fw = frame_shape[:2]
        nw, nh = max(nw, 1), max(nh, 1)
        nx = max(0, min(nx, fw - 1))
        ny = max(0, min(ny, fh - 1))
        nw = min(nw, fw - nx)
        nh = min(nh, fh - ny)
        return (nx, ny, nw, nh)

    def _estimate_global_motion(
        self,
        prev_gray: np.ndarray,
        gray: np.ndarray,
        ref_bbox: Tuple[int, int, int, int],
    ) -> Tuple[Optional[np.ndarray], int]:
        """
        Background-dominated affine from previous frame -> current.
        Excludes an expanded target ROI so camera motion dominates.
        """
        fh, fw = gray.shape[:2]
        x, y, bw, bh = ref_bbox
        pad = int(max(bw, bh) * 0.75)
        mask = np.full((fh, fw), 255, np.uint8)
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(fw, x + bw + pad)
        y2 = min(fh, y + bh + pad)
        mask[y1:y2, x1:x2] = 0

        max_side = 640
        scale = min(1.0, max_side / max(fh, fw))
        if scale < 1.0:
            sw, sh = int(fw * scale), int(fh * scale)
            prev_s = cv2.resize(prev_gray, (sw, sh), interpolation=cv2.INTER_AREA)
            curr_s = cv2.resize(gray, (sw, sh), interpolation=cv2.INTER_AREA)
            mask_s = cv2.resize(mask, (sw, sh), interpolation=cv2.INTER_NEAREST)
            inv = 1.0 / scale
        else:
            prev_s, curr_s, mask_s = prev_gray, gray, mask
            inv = 1.0

        pts = cv2.goodFeaturesToTrack(
            prev_s,
            maxCorners=GLOBAL_MOTION_MAX_CORNERS,
            qualityLevel=0.01,
            minDistance=8,
            mask=mask_s,
        )
        if pts is None or len(pts) < 8:
            pts = cv2.goodFeaturesToTrack(
                prev_s,
                maxCorners=GLOBAL_MOTION_MAX_CORNERS,
                qualityLevel=0.01,
                minDistance=8,
            )
        if pts is None or len(pts) < 8:
            return None, 0

        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_s, curr_s, pts, None, **self._lk_params
        )
        good_old = pts[status.ravel() == 1]
        good_new = next_pts[status.ravel() == 1]
        if len(good_old) < 8:
            return None, 0

        m, inliers = cv2.estimateAffinePartial2D(
            good_old,
            good_new,
            method=cv2.RANSAC,
            ransacReprojThreshold=4.0,
        )
        if m is None:
            return None, 0

        n_in = int(inliers.ravel().sum()) if inliers is not None else len(good_old)
        if n_in < GLOBAL_MOTION_MIN_INLIERS:
            return None, n_in

        if inv != 1.0:
            m[0, 2] *= inv
            m[1, 2] *= inv

        return m, n_in

    def _motion_compensated_bbox(
        self,
        frame_shape: Tuple[int, ...],
    ) -> Optional[Tuple[int, int, int, int]]:
        if self._global_affine is None:
            return None
        return self._warp_bbox_affine(self._last_good_bbox, self._global_affine, frame_shape)

    def _relocate_anchor_bbox(
        self,
        kf_pred: Tuple[int, int, int, int],
        flow_bbox: Optional[Tuple[int, int, int, int]],
        flow_ok: bool,
        frame_shape: Tuple[int, ...],
    ) -> Tuple[int, int, int, int]:
        """Fuse Kalman, KLT, and camera-motion-warped priors for search / relocate."""
        candidates: list[Tuple[int, int, int, int]] = [kf_pred]
        if flow_ok and flow_bbox is not None:
            candidates.append(flow_bbox)
        mc = self._motion_compensated_bbox(frame_shape)
        if mc is not None:
            candidates.append(mc)

        if len(candidates) == 1:
            return candidates[0]

        cx = cy = w = h = 0.0
        for bx, by, bw, bh in candidates:
            cx += bx + bw / 2.0
            cy += by + bh / 2.0
            w += bw
            h += bh
        n = len(candidates)
        w = int(round(w / n))
        h = int(round(h / n))
        cx /= n
        cy /= n
        return (int(round(cx - w / 2)), int(round(cy - h / 2)), max(w, 1), max(h, 1))

    def _search_window_for_anchor(
        self,
        anchor: Tuple[int, int, int, int],
        frame_shape: Tuple[int, ...],
    ) -> Tuple[int, int, int, int]:
        ax, ay, aw, ah = anchor
        acx = ax + aw / 2.0
        acy = ay + ah / 2.0
        sw = int(max(aw, ah) * REDETECT_SEARCH_SCALE)
        sh = sw
        # Expand with Kalman speed when target is moving
        state = self._kalman.kf.statePost
        vx, vy = float(state[4, 0]), float(state[5, 0])
        speed = float(np.hypot(vx, vy))
        grow = int(min(120, speed * (1 + self._lost_frame_count * 0.5)))
        sw += grow
        sh += grow
        sx = int(acx - sw / 2.0)
        sy = int(acy - sh / 2.0)
        fh, fw = frame_shape[:2]
        sx = max(0, min(sx, fw - 1))
        sy = max(0, min(sy, fh - 1))
        sw = max(20, min(sw, fw - sx))
        sh = max(20, min(sh, fh - sy))
        return (sx, sy, sw, sh)

    def _scaled_frame_and_bbox(
        self, frame: np.ndarray, bbox: Tuple[int, int, int, int]
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        if self._scale < 1.0:
            h, w = frame.shape[:2]
            scaled_frame = cv2.resize(
                frame, (int(w * self._scale), int(h * self._scale))
            )
            scaled_bbox = (
                int(bbox[0] * self._scale),
                int(bbox[1] * self._scale),
                int(bbox[2] * self._scale),
                int(bbox[3] * self._scale),
            )
            return scaled_frame, scaled_bbox
        return frame, bbox

    def _fullres_bbox_from_scaled(
        self, bbox: Tuple[int, int, int, int]
    ) -> Tuple[int, int, int, int]:
        if self._scale < 1.0:
            inv = 1.0 / self._scale
            return (
                int(bbox[0] * inv),
                int(bbox[1] * inv),
                int(bbox[2] * inv),
                int(bbox[3] * inv),
            )
        return tuple(int(v) for v in bbox)

    def _reinit_visual_tracker(
        self, frame: np.ndarray, bbox: Tuple[int, int, int, int]
    ) -> None:
        """Re-init visual tracker in-place (keeps compiled PyTorch backbone)."""
        scaled_frame, scaled_bbox = self._scaled_frame_and_bbox(frame, bbox)
        self._visual.init(scaled_frame, scaled_bbox)

    @staticmethod
    def _patch_to_u8(patch: np.ndarray) -> np.ndarray:
        """matchTemplate requires uint8; last-good patch may be float32."""
        if patch.dtype == np.uint8:
            return patch
        return np.clip(patch, 0, 255).astype(np.uint8)

    def _redetection_templates(self) -> list[np.ndarray]:
        """At most two grayscale templates sized from last known target."""
        templates: list[np.ndarray] = []
        if self._last_good_gray_patch is not None and self._last_good_gray_patch.size > 0:
            templates.append(self._patch_to_u8(self._last_good_gray_patch))
        if (
            self._init_gray_patch is not None
            and self._init_gray_patch.size > 0
            and len(templates) < REDETECT_MAX_TEMPLATES
        ):
            init_u8 = self._patch_to_u8(self._init_gray_patch)
            if not templates or not np.array_equal(templates[0], init_u8):
                templates.append(init_u8)
        return templates[:REDETECT_MAX_TEMPLATES]

    def _try_visual_relocate(
        self,
        scaled_frame: np.ndarray,
        anchor: Tuple[int, int, int, int],
        frame: np.ndarray,
    ) -> Tuple[Optional[Tuple[int, int, int, int]], float]:
        """Run visual tracker at motion-aware anchor (no model rebuild)."""
        acx = anchor[0] + anchor[2] / 2.0
        acy = anchor[1] + anchor[3] / 2.0

        if isinstance(self._visual, PyTorchTrackerWrapper):
            scaled_center = (acx * self._scale, acy * self._scale)
            ok, bbox_raw, _score = self._visual.update_with_score(
                scaled_frame,
                predicted_center=scaled_center,
                min_score=REDETECT_RELOCATE_SCORE,
            )
        else:
            ok, bbox_raw = self._visual.update(scaled_frame)
            _score = 1.0 if ok else 0.0

        if not ok:
            return None, 0.0

        vis_bbox = self._fullres_bbox_from_scaled(bbox_raw)
        vis_conf = self._patch_ncc_confidence(frame, vis_bbox)
        if vis_conf < CONFIDENCE_THRESHOLD:
            return None, 0.0
        return vis_bbox, vis_conf

    def _attempt_recovery_when_lost(
        self,
        frame: np.ndarray,
        scaled_frame: np.ndarray,
        kf_pred: Tuple[int, int, int, int],
        flow_bbox: Optional[Tuple[int, int, int, int]],
        flow_ok: bool,
    ) -> Tuple[Optional[Tuple[int, int, int, int]], float]:
        """
        Recover after loss: flow -> global-motion anchor -> Siamese/OpenCV -> warped template.
        """
        anchor = self._relocate_anchor_bbox(
            kf_pred, flow_bbox, flow_ok, frame.shape
        )

        if flow_ok and flow_bbox is not None:
            flow_conf = self._patch_ncc_confidence(frame, flow_bbox)
            if flow_conf >= CONFIDENCE_THRESHOLD:
                return flow_bbox, flow_conf

        mc_bbox = self._motion_compensated_bbox(frame.shape)
        if mc_bbox is not None:
            mc_conf = self._patch_ncc_confidence(frame, mc_bbox)
            if mc_conf >= CONFIDENCE_THRESHOLD:
                return mc_bbox, mc_conf

        relocate_bbox, relocate_conf = self._try_visual_relocate(
            scaled_frame, anchor, frame
        )
        if relocate_bbox is not None:
            return relocate_bbox, relocate_conf

        search_rect = self._search_window_for_anchor(anchor, frame.shape)
        score, best_bbox = self._run_redetection_search(
            frame, search_rect, anchor_bbox=anchor
        )
        if best_bbox is None:
            return None, 0.0
        patch_conf = self._patch_ncc_confidence(frame, best_bbox)
        return best_bbox, patch_conf

    # ------------------------------------------------------------------
    def _add_template_to_pool(self, patch: np.ndarray):
        if patch is None or patch.size == 0:
            return
        if len(patch.shape) == 3:
            patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        target_size = (64, 64)
        patch_resized = cv2.resize(patch, target_size)
        
        if self._init_gray_patch is not None:
            init_resized = cv2.resize(self._init_gray_patch, target_size)
            res = cv2.matchTemplate(patch_resized, init_resized, cv2.TM_CCOEFF_NORMED)
            ncc_score = float(res[0, 0])
            normalized_ncc = (ncc_score + 1.0) / 2.0
            if normalized_ncc < 0.45:
                return
        
        for existing in self._template_pool:
            ex_resized = cv2.resize(existing, target_size)
            res = cv2.matchTemplate(patch_resized, ex_resized, cv2.TM_CCOEFF_NORMED)
            ncc_score = float(res[0, 0])
            normalized_ncc = (ncc_score + 1.0) / 2.0
            if normalized_ncc > 0.90:
                return

        self._template_pool.append(patch)
        if len(self._template_pool) > self._max_pool_size:
            # Keep index 0 (initial template), pop index 1 (oldest adaptive template)
            self._template_pool.pop(1)

    # ------------------------------------------------------------------
    def _warp_template_for_motion(self, template: np.ndarray) -> np.ndarray:
        """Warp appearance template with estimated camera motion (prev -> now)."""
        if self._global_affine is None or self._global_motion_inliers < GLOBAL_MOTION_MIN_INLIERS:
            return template
        h, w = template.shape[:2]
        try:
            return cv2.warpAffine(
                template,
                self._global_affine,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
        except cv2.error:
            return template

    def _run_redetection_search(
        self,
        frame: np.ndarray,
        search_bbox: Tuple[int, int, int, int],
        anchor_bbox: Tuple[int, int, int, int],
    ) -> Tuple[float, Optional[Tuple[int, int, int, int]]]:
        """
        Template match in a large window around a motion-aware anchor.
        Templates are camera-motion-warped; no penalty toward stale image position.
        """
        sx, sy, sw, sh = search_bbox
        fh, fw = frame.shape[:2]
        sx = max(0, min(sx, fw - 1))
        sy = max(0, min(sy, fh - 1))
        sw = max(10, min(sw, fw - sx))
        sh = max(10, min(sh, fh - sy))

        templates = self._redetection_templates()
        if not templates:
            return -1.0, None

        _, _, ref_w, ref_h = anchor_bbox
        ref_w, ref_h = max(ref_w, 10), max(ref_h, 10)

        search_gray = cv2.cvtColor(
            frame[sy : sy + sh, sx : sx + sw], cv2.COLOR_BGR2GRAY
        )
        if np.std(search_gray) < 1.0:
            return -1.0, None

        inv_scale = 1.0
        max_side = max(sw, sh)
        if max_side > REDETECT_MAX_SEARCH_SIDE:
            inv_scale = max_side / REDETECT_MAX_SEARCH_SIDE
            sw_s = max(10, int(round(sw / inv_scale)))
            sh_s = max(10, int(round(sh / inv_scale)))
            search_gray = cv2.resize(
                search_gray, (sw_s, sh_s), interpolation=cv2.INTER_AREA
            )
        else:
            sw_s, sh_s = sw, sh

        acx_rel = (anchor_bbox[0] + anchor_bbox[2] / 2.0 - sx) / inv_scale
        acy_rel = (anchor_bbox[1] + anchor_bbox[3] / 2.0 - sy) / inv_scale
        norm_denom = float(max(sw_s, sh_s, 1))

        candidates: list[Tuple[float, Tuple[int, int, int, int]]] = []

        for temp in templates:
            warped = self._warp_template_for_motion(temp)
            for s in REDETECT_SCALES:
                tw = max(8, int(round(ref_w * s / inv_scale)))
                th = max(8, int(round(ref_h * s / inv_scale)))
                if tw >= sw_s or th >= sh_s:
                    continue
                templ = cv2.resize(warped, (tw, th), interpolation=cv2.INTER_AREA)
                try:
                    res = cv2.matchTemplate(
                        search_gray, templ, cv2.TM_CCOEFF_NORMED
                    )
                except cv2.error:
                    continue

                res_scan = res.copy()
                for _ in range(6):
                    _, peak, _, loc = cv2.minMaxLoc(res_scan)
                    if peak < REDETECT_MATCH_THRESH:
                        break
                    dist = float(np.hypot(loc[0] - acx_rel, loc[1] - acy_rel))
                    ranked = float(peak) - 0.18 * (dist / norm_denom)
                    match_x = int(round((loc[0] * inv_scale) + sx))
                    match_y = int(round((loc[1] * inv_scale) + sy))
                    match_w = int(round(tw * inv_scale))
                    match_h = int(round(th * inv_scale))
                    candidates.append(
                        (ranked, (match_x, match_y, match_w, match_h))
                    )
                    x0, y0 = loc[0], loc[1]
                    res_scan[
                        max(0, y0 - 3) : min(res_scan.shape[0], y0 + th + 3),
                        max(0, x0 - 3) : min(res_scan.shape[1], x0 + tw + 3),
                    ] = -1.0

        if not candidates:
            return -1.0, None

        candidates.sort(key=lambda item: item[0], reverse=True)
        for ranked, bbox in candidates:
            if self._patch_ncc_confidence(frame, bbox) >= CONFIDENCE_THRESHOLD:
                return ranked, bbox

        return candidates[0][0], None

    # ------------------------------------------------------------------
    def update(self, frame: np.ndarray) -> TrackState:
        t0 = time.perf_counter()
        self._frame_id += 1

        # 1) Kalman prediction (motion model)
        kf_pred = self._kalman.predict()
        kf_conf = self._kalman.confidence

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self._global_affine = None
        self._global_motion_inliers = 0
        if self._prev_gray is not None:
            self._global_affine, self._global_motion_inliers = self._estimate_global_motion(
                self._prev_gray, gray, self._last_good_bbox
            )

        # 1.5) Sparse Optical Flow on target region (tracks camera + object motion)
        flow_bbox = None
        flow_ok = False

        if self._prev_gray is not None and self._klt_pts is not None and len(self._klt_pts) >= 4:
            try:
                next_pts, status, err = cv2.calcOpticalFlowPyrLK(self._prev_gray, gray, self._klt_pts, None, **self._lk_params)
                good_old = self._klt_pts[status == 1]
                good_new = next_pts[status == 1]
                
                if len(good_new) >= 4:
                    M, inliers = cv2.estimateAffinePartial2D(good_old, good_new, method=cv2.RANSAC, ransacReprojThreshold=3.0)
                    if M is not None:
                        tx = M[0, 2]
                        ty = M[1, 2]
                        s = np.sqrt(M[0, 0]**2 + M[0, 1]**2)
                        
                        # Apply transformation to the last good bbox
                        lx, ly, lw, lh = self._last_good_bbox
                        flow_x = int(round(lx + tx))
                        flow_y = int(round(ly + ty))
                        flow_w = int(round(lw * s))
                        flow_h = int(round(lh * s))
                        flow_bbox = (flow_x, flow_y, flow_w, flow_h)
                        
                        # Verify the flow_bbox patch in the new frame is not flat/constant
                        fh_frame, fw_frame = gray.shape[:2]
                        fx_clamp = max(0, min(flow_x, fw_frame - 1))
                        fy_clamp = max(0, min(flow_y, fh_frame - 1))
                        fw_clamp = max(1, min(flow_w, fw_frame - fx_clamp))
                        fh_clamp = max(1, min(flow_h, fh_frame - fy_clamp))
                        flow_patch = gray[fy_clamp : fy_clamp + fh_clamp, fx_clamp : fx_clamp + fw_clamp]
                        
                        if flow_patch.size > 0 and np.std(flow_patch) >= 5.0:
                            flow_ok = True
                            # Update points
                            self._klt_pts = good_new.reshape(-1, 1, 2)
                        else:
                            flow_ok = False
            except Exception as e:
                print(f"[KLT] Error: {e}")

        # 2) Video scaling
        if self._scale < 1.0:
            h, w = frame.shape[:2]
            scaled_frame = cv2.resize(frame, (int(w * self._scale), int(h * self._scale)))
        else:
            scaled_frame = frame

        is_redetected = False
        redetected_bbox = None
        vis_conf = 0.0
        ok = False
        vis_bbox = kf_pred
        relocate_anchor = self._relocate_anchor_bbox(
            kf_pred, flow_bbox, flow_ok, frame.shape
        )

        # 3) Motion-aware recovery while lost
        if self._is_lost:
            run_search = self._lost_frame_count % REDETECT_INTERVAL == 0
            self._lost_frame_count += 1

            if run_search:
                recovered_bbox, recovered_conf = self._attempt_recovery_when_lost(
                    frame, scaled_frame, kf_pred, flow_bbox, flow_ok
                )
                if recovered_bbox is not None:
                    is_redetected = True
                    redetected_bbox = recovered_bbox
                    vis_conf = recovered_conf

        # 4) Visual tracker update (also when lost — search centered on motion anchor)
        if not is_redetected:
            if isinstance(self._visual, PyTorchTrackerWrapper):
                acx = relocate_anchor[0] + relocate_anchor[2] / 2.0
                acy = relocate_anchor[1] + relocate_anchor[3] / 2.0
                scaled_center = (acx * self._scale, acy * self._scale)
                ok, vis_bbox_raw = self._visual.update(
                    scaled_frame, predicted_center=scaled_center
                )
            else:
                ok, vis_bbox_raw = self._visual.update(scaled_frame)

            if ok:
                if self._scale < 1.0:
                    vis_bbox = (
                        int(vis_bbox_raw[0] / self._scale),
                        int(vis_bbox_raw[1] / self._scale),
                        int(vis_bbox_raw[2] / self._scale),
                        int(vis_bbox_raw[3] / self._scale),
                    )
                else:
                    vis_bbox = (
                        int(vis_bbox_raw[0]),
                        int(vis_bbox_raw[1]),
                        int(vis_bbox_raw[2]),
                        int(vis_bbox_raw[3]),
                    )
                vis_conf = self._estimate_visual_confidence(frame, vis_bbox)
            else:
                vis_conf = 0.0
                vis_bbox = relocate_anchor if self._is_lost else kf_pred
                if self._is_lost:
                    self._response_history.append(0.0)
                    if len(self._response_history) > self._history_len:
                        self._response_history.pop(0)

        # 5) State transition and Fusion decision
        if is_redetected:
            self._reinit_visual_tracker(frame, redetected_bbox)

            kf_corrected = self._kalman.update(redetected_bbox)
            self._last_good_bbox = redetected_bbox
            self._is_lost = False
            self._lost_frame_count = 0
            self._response_history = [vis_conf]

            # Cache patch
            lx, ly, lw, lh = redetected_bbox
            fh, fw = frame.shape[:2]
            lx_clamp = max(0, min(lx, fw - 1))
            ly_clamp = max(0, min(ly, fh - 1))
            lw_clamp = max(1, min(lw, fw - lx_clamp))
            lh_clamp = max(1, min(lh, fh - ly_clamp))
            new_patch = frame[ly_clamp : ly_clamp + lh_clamp, lx_clamp : lx_clamp + lw_clamp]
            if new_patch.size > 0:
                gray_patch = cv2.cvtColor(new_patch, cv2.COLOR_BGR2GRAY)
                self._add_template_to_pool(gray_patch)
                self._last_good_gray_patch = gray_patch.astype(np.float32)

            out_bbox = redetected_bbox
            source = "tracker"
            out_tracker_conf = vis_conf

        elif ok and vis_conf >= CONFIDENCE_THRESHOLD:
            if self._is_lost:
                self._reinit_visual_tracker(frame, vis_bbox)
            # Correct Kalman with fresh measurement
            kf_corrected = self._kalman.update(vis_bbox)

            # Weighted average of visual and Kalman-corrected bboxes
            alpha = FUSE_ALPHA
            fused = tuple(
                int(alpha * v + (1 - alpha) * k)
                for v, k in zip(vis_bbox, kf_corrected)
            )
            
            # Optionally fuse with KLT prediction
            if flow_ok:
                fused = tuple(
                    int(0.6 * f + 0.4 * fl)
                    for f, fl in zip(fused, flow_bbox)
                )
            
            self._last_good_bbox = fused

            # Cache the new good patch as grayscale
            lx, ly, lw, lh = fused
            fh, fw = frame.shape[:2]
            lx_clamp = max(0, min(lx, fw - 1))
            ly_clamp = max(0, min(ly, fh - 1))
            lw_clamp = max(1, min(lw, fw - lx_clamp))
            lh_clamp = max(1, min(lh, fh - ly_clamp))
            new_patch = frame[ly_clamp : ly_clamp + lh_clamp, lx_clamp : lx_clamp + lw_clamp]
            if new_patch.size > 0:
                gray_patch = cv2.cvtColor(new_patch, cv2.COLOR_BGR2GRAY)
                self._add_template_to_pool(gray_patch)
                self._last_good_gray_patch = gray_patch.astype(np.float32)

            source = "tracker" if vis_conf > 0.7 else "fused"
            out_bbox = fused
            out_tracker_conf = vis_conf
            self._is_lost = False
            self._lost_frame_count = 0

        elif flow_ok:
            was_lost = self._is_lost
            flow_conf = self._patch_ncc_confidence(frame, flow_bbox)
            kf_corrected = self._kalman.update(flow_bbox)
            fused = tuple(
                int(0.5 * fl + 0.5 * k)
                for fl, k in zip(flow_bbox, kf_corrected)
            )
            self._last_good_bbox = fused
            out_bbox = fused
            out_tracker_conf = flow_conf
            source = "fused"
            self._is_lost = False
            self._lost_frame_count = 0
            if was_lost:
                self._reinit_visual_tracker(frame, fused)
        else:
            # Motion prior: Kalman + camera-compensated anchor while lost
            self._kalman.miss_count += 1
            if self._is_lost:
                out_bbox = tuple(
                    int(0.5 * k + 0.5 * a)
                    for k, a in zip(kf_pred, relocate_anchor)
                )
            else:
                out_bbox = kf_pred
            out_tracker_conf = vis_conf
            source = "motion_model"
            self._is_lost = True
            # Decay rolling average when lost
            self._response_history.append(0.0)
            if len(self._response_history) > self._history_len:
                self._response_history.pop(0)

        out_bbox = self._clamp_bbox(out_bbox, frame.shape)
        
        # Update KLT frame
        self._prev_gray = gray

        # Re-sample KLT keypoints to prevent depletion
        if self._klt_pts is None or len(self._klt_pts) < 15 or self._frame_id % 15 == 0:
            lx, ly, lw, lh = out_bbox
            fh, fw = frame.shape[:2]
            pad = 5
            x1 = max(0, lx - pad)
            y1 = max(0, ly - pad)
            x2 = min(fw, lx + lw + pad)
            y2 = min(fh, ly + lh + pad)
            
            patch = gray[y1:y2, x1:x2]
            if patch.size > 0:
                pts = cv2.goodFeaturesToTrack(patch, maxCorners=50, qualityLevel=0.01, minDistance=5)
                if pts is not None:
                    pts[:, 0, 0] += x1
                    pts[:, 0, 1] += y1
                    self._klt_pts = pts
                else:
                    self._klt_pts = np.array([], dtype=np.float32).reshape(-1, 1, 2)
            else:
                self._klt_pts = np.array([], dtype=np.float32).reshape(-1, 1, 2)

        return TrackState(
            bbox=out_bbox,
            tracker_confidence=round(out_tracker_conf, 3),
            motion_confidence=round(kf_conf, 3),
            trusted_source=source,
            frame_id=self._frame_id,
            timestamp_ms=round((time.perf_counter() - t0) * 1000, 2),
        )

    # ------------------------------------------------------------------
    def _patch_ncc_confidence(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
    ) -> float:
        """Instant NCC vs initial + template pool (no rolling history)."""
        x, y, w, h = bbox
        fh, fw = frame.shape[:2]
        x = max(0, min(x, fw - 1))
        y = max(0, min(y, fh - 1))
        w = max(1, min(w, fw - x))
        h = max(1, min(h, fh - y))

        try:
            patch_curr = cv2.cvtColor(
                frame[y : y + h, x : x + w], cv2.COLOR_BGR2GRAY
            )
            if patch_curr.size == 0 or np.std(patch_curr) < 1.0:
                return 0.0

            target_size = (64, 64)
            pc = cv2.resize(patch_curr, target_size)
            max_score = -1.0

            if self._init_gray_patch is not None and self._init_gray_patch.size > 0:
                pr = cv2.resize(self._patch_to_u8(self._init_gray_patch), target_size)
                max_score = max(
                    max_score,
                    float(cv2.matchTemplate(pc, pr, cv2.TM_CCOEFF_NORMED)[0, 0]),
                )
            for temp in self._template_pool:
                if temp is not None and temp.size > 0:
                    pr = cv2.resize(self._patch_to_u8(temp), target_size)
                    max_score = max(
                        max_score,
                        float(cv2.matchTemplate(pc, pr, cv2.TM_CCOEFF_NORMED)[0, 0]),
                    )
            return float((max_score + 1.0) / 2.0)
        except Exception:
            return 0.0

    def _estimate_visual_confidence(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
    ) -> float:
        """Smoothed visual confidence (rolling mean of per-frame NCC)."""
        conf = self._patch_ncc_confidence(frame, bbox)
        self._response_history.append(conf)
        if len(self._response_history) > self._history_len:
            self._response_history.pop(0)
        return float(np.mean(self._response_history))

    # ------------------------------------------------------------------
    @staticmethod
    def _clamp_bbox(
        bbox: Tuple[int, int, int, int],
        shape: Tuple[int, ...]
    ) -> Tuple[int, int, int, int]:
        h_frame, w_frame = shape[:2]
        x, y, w, h = bbox
        x = max(0, min(x, w_frame - 1))
        y = max(0, min(y, h_frame - 1))
        w = max(1, min(w, w_frame - x))
        h = max(1, min(h, h_frame - y))
        return (x, y, w, h)
