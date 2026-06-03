"""
Unit tests for the GPU-Accelerated Fusion Tracker.
Run with:  pytest tests/test_tracker.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tracker import ConstantVelocityKalman, FusionTracker, TrackState
from pytorch_tracker import PyTorchSiameseTracker


# ---------------------------------------------------------------------------
# ConstantVelocityKalman tests
# ---------------------------------------------------------------------------

class TestConstantVelocityKalman:
    def test_init_and_predict(self):
        kf = ConstantVelocityKalman((100, 100, 50, 50))
        pred = kf.predict()
        assert len(pred) == 4
        assert all(isinstance(v, int) for v in pred)

    def test_update_resets_miss_streak(self):
        kf = ConstantVelocityKalman((100, 100, 50, 50))
        kf.predict()
        assert kf._predict_only_streak == 1
        kf.update((102, 100, 50, 50))
        assert kf._predict_only_streak == 0

    def test_confidence_decays_without_measurement(self):
        kf = ConstantVelocityKalman((0, 0, 30, 30))
        prev_conf = kf.confidence
        for _ in range(15):
            kf.predict()
        assert kf.confidence < prev_conf

    def test_confidence_in_range(self):
        kf = ConstantVelocityKalman((0, 0, 30, 30))
        for _ in range(50):
            kf.predict()
        assert 0.0 <= kf.confidence <= 1.0

    def test_update_corrects_towards_measurement(self):
        kf = ConstantVelocityKalman((0, 0, 50, 50))
        for _ in range(10):
            kf.predict()
        corrected = kf.update((200, 200, 50, 50))
        # After large correction, bbox should shift toward measurement
        x, y, w, h = corrected
        cx, cy = x + w / 2, y + h / 2
        assert cx > 25, "Corrected cx should move toward 225"
        assert cy > 25, "Corrected cy should move toward 225"

    def test_bbox_clamped_to_positive_size(self):
        kf = ConstantVelocityKalman((0, 0, 5, 5))
        for _ in range(100):
            kf.predict()
        x, y, w, h = kf.predict()
        assert w >= 1 and h >= 1


# ---------------------------------------------------------------------------
# FusionTracker tests
# ---------------------------------------------------------------------------

def _black_frame(h=480, w=640):
    return np.zeros((h, w, 3), dtype=np.uint8)


def _colored_frame(bbox, h=480, w=640, color=(0, 200, 100)):
    """Frame with a solid rectangle at bbox."""
    frame = _black_frame(h, w)
    x, y, bw, bh = bbox
    frame[y:y+bh, x:x+bw] = color
    return frame


class TestFusionTracker:
    def test_returns_trackstate(self):
        frame = _colored_frame((100, 100, 60, 60))
        tracker = FusionTracker(frame, (100, 100, 60, 60))
        result = tracker.update(frame)
        assert isinstance(result, TrackState)

    def test_bbox_is_4tuple(self):
        frame = _colored_frame((50, 50, 40, 40))
        tracker = FusionTracker(frame, (50, 50, 40, 40))
        state = tracker.update(frame)
        assert len(state.bbox) == 4

    def test_trusted_source_valid(self):
        frame = _colored_frame((80, 80, 50, 50))
        tracker = FusionTracker(frame, (80, 80, 50, 50))
        for _ in range(5):
            state = tracker.update(frame)
        assert state.trusted_source in ("tracker", "fused", "motion_model")

    def test_confidence_in_range(self):
        frame = _colored_frame((80, 80, 50, 50))
        tracker = FusionTracker(frame, (80, 80, 50, 50))
        state = tracker.update(frame)
        assert 0.0 <= state.tracker_confidence <= 1.0
        assert 0.0 <= state.motion_confidence <= 1.0

    def test_frame_id_increments(self):
        frame = _colored_frame((80, 80, 50, 50))
        tracker = FusionTracker(frame, (80, 80, 50, 50))
        ids = [tracker.update(frame).frame_id for _ in range(5)]
        assert ids == [1, 2, 3, 4, 5]

    def test_black_frame_falls_back_to_motion_model(self):
        """
        When the tracker can't find the object (all-black frame after init),
        the trusted source should eventually be 'motion_model'.
        """
        init_frame = _colored_frame((150, 150, 50, 50))
        tracker = FusionTracker(init_frame, (150, 150, 50, 50))
        black = _black_frame()
        source_counts: dict[str, int] = {}
        for _ in range(10):
            s = tracker.update(black)
            source_counts[s.trusted_source] = \
                source_counts.get(s.trusted_source, 0) + 1
        assert "motion_model" in source_counts or "fused" in source_counts

    def test_bbox_stays_within_frame(self):
        h, w = 240, 320
        frame = _colored_frame((5, 5, 30, 30), h=h, w=w)
        tracker = FusionTracker(frame, (5, 5, 30, 30))
        for _ in range(20):
            state = tracker.update(frame)
            x, y, bw, bh = state.bbox
            assert x >= 0 and y >= 0
            assert x + bw <= w
            assert y + bh <= h

    def test_template_pool(self):
        frame1 = _colored_frame((100, 100, 50, 50))
        tracker = FusionTracker(frame1, (100, 100, 50, 50))
        # Initial template should be in pool
        assert len(tracker._template_pool) == 1
        
        # Add templates
        for i in range(10):
            patch = frame1[100:150, 100:150]
            tracker._add_template_to_pool(patch)
            
        # The pool should be limited to self._max_pool_size (5)
        assert len(tracker._template_pool) <= 5

    def test_occlusion_recovery(self):
        target_bbox = (100, 100, 50, 50)
        frame_start = _colored_frame(target_bbox)
        tracker = FusionTracker(frame_start, target_bbox)
        
        # Occlude target (black frames)
        black = _black_frame()
        for _ in range(12):
            state = tracker.update(black)
        assert tracker._is_lost is True
        
        # Reappear target
        frame_reappear = _colored_frame((102, 102, 50, 50))
        state = tracker.update(frame_reappear)
        
        assert tracker._is_lost is False
        assert state.trusted_source in ("tracker", "fused")

    def test_nanotrack_integration(self):
        frame = _colored_frame((100, 100, 50, 50))
        tracker = FusionTracker(frame, (100, 100, 50, 50), algo="NANOTRACK")
        assert tracker._algo == "NANOTRACK"
        state = tracker.update(frame)
        assert len(state.bbox) == 4
        assert state.frame_id == 1

    def test_klt_optical_flow(self):
        frame1 = _colored_frame((100, 100, 50, 50))
        tracker = FusionTracker(frame1, (100, 100, 50, 50))
        assert tracker._klt_pts is not None
        assert len(tracker._klt_pts) > 0
        
        frame2 = _colored_frame((102, 101, 50, 50))
        state = tracker.update(frame2)
        assert len(tracker._klt_pts) > 0


# ---------------------------------------------------------------------------
# PyTorchSiameseTracker tests
# ---------------------------------------------------------------------------

class TestPyTorchSiameseTracker:
    def test_init_and_update_dasiamrpn(self):
        frame = _colored_frame((100, 100, 50, 50))
        tracker = PyTorchSiameseTracker(model_name="dasiamrpn")
        tracker.init(frame, (100, 100, 50, 50))
        bbox, score = tracker.update(frame)
        assert len(bbox) == 4
        assert isinstance(bbox[0], int)
        assert score >= 0.0

    def test_init_and_update_siamrpnpp(self):
        frame = _colored_frame((100, 100, 50, 50))
        tracker = PyTorchSiameseTracker(model_name="siamrpnpp")
        tracker.init(frame, (100, 100, 50, 50))
        bbox, score = tracker.update(frame)
        assert len(bbox) == 4
        assert isinstance(bbox[0], int)
        assert score >= 0.0

    def test_update_with_predicted_center(self):
        frame = _colored_frame((100, 100, 50, 50))
        tracker = PyTorchSiameseTracker(model_name="dasiamrpn")
        tracker.init(frame, (100, 100, 50, 50))
        bbox, score = tracker.update(frame, predicted_center=(100.0, 100.0))
        assert len(bbox) == 4
        assert score >= 0.0

    def test_init_and_update_ostrack(self):
        frame = _colored_frame((100, 100, 50, 50))
        tracker = PyTorchSiameseTracker(model_name="ostrack")
        tracker.init(frame, (100, 100, 50, 50))
        bbox, score = tracker.update(frame)
        assert len(bbox) == 4
        assert isinstance(bbox[0], int)
        assert score >= 0.0

    def test_init_and_update_mixformer(self):
        frame = _colored_frame((100, 100, 50, 50))
        tracker = PyTorchSiameseTracker(model_name="mixformer")
        tracker.init(frame, (100, 100, 50, 50))
        bbox, score = tracker.update(frame)
        assert len(bbox) == 4
        assert isinstance(bbox[0], int)
        assert score >= 0.0
