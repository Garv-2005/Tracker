"""
Visualization utilities for the GPU-accelerated tracker.
Draws bounding boxes, state text, confidence bars, and FPS.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Deque, Tuple

import cv2
import numpy as np

from tracker import TrackState


# Color palette  (BGR)
COLORS = {
    "tracker":      (0,   220,  80),   # vivid green
    "fused":        (0,   180, 255),   # orange-blue
    "motion_model": (50,   50, 255),   # red
    "ui_bg":        (10,   10,  10),
    "ui_text":      (230, 230, 230),
    "bar_bg":       (60,   60,  60),
    "bar_vis":      (0,   220,  80),
    "bar_kf":       (80,  160, 255),
}

FONT       = cv2.FONT_HERSHEY_DUPLEX
FONT_SMALL = cv2.FONT_HERSHEY_SIMPLEX


class Visualizer:
    def __init__(self, fps_window: int = 30):
        self._ts: Deque[float] = deque(maxlen=fps_window)

    # ------------------------------------------------------------------
    def draw(self, frame: np.ndarray, state: TrackState) -> np.ndarray:
        canvas = frame.copy()
        self._ts.append(time.perf_counter())

        color = COLORS.get(state.trusted_source, COLORS["fused"])
        x, y, w, h = state.bbox

        # Main bounding box
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 2)

        # Corner accents
        corner_len = max(10, min(w, h) // 5)
        self._draw_corners(canvas, x, y, w, h, color, corner_len)

        # Label chip above the box
        label = state.trusted_source.upper().replace("_", " ")
        self._draw_label(canvas, label, x, y, color)

        # Side-panel HUD
        self._draw_hud(canvas, state)

        return canvas

    # ------------------------------------------------------------------
    def _draw_corners(
        self,
        img: np.ndarray,
        x: int, y: int, w: int, h: int,
        color: Tuple[int, int, int],
        L: int,
    ) -> None:
        pts = [
            ((x, y),         (x + L, y),     (x, y + L)),
            ((x + w, y),     (x + w - L, y), (x + w, y + L)),
            ((x, y + h),     (x + L, y + h), (x, y + h - L)),
            ((x + w, y + h), (x + w - L, y + h), (x + w, y + h - L)),
        ]
        for corner, p1, p2 in pts:
            cv2.line(img, corner, p1, color, 3, cv2.LINE_AA)
            cv2.line(img, corner, p2, color, 3, cv2.LINE_AA)

    def _draw_label(
        self,
        img: np.ndarray,
        label: str,
        x: int, y: int,
        color: Tuple[int, int, int],
    ) -> None:
        (tw, th), _ = cv2.getTextSize(label, FONT_SMALL, 0.55, 1)
        pad = 4
        lx, ly = x, max(0, y - th - 2 * pad)
        cv2.rectangle(img, (lx, ly), (lx + tw + 2 * pad, ly + th + 2 * pad), color, -1)
        cv2.putText(img, label, (lx + pad, ly + th + pad), FONT_SMALL, 0.55,
                    (10, 10, 10), 1, cv2.LINE_AA)

    # ------------------------------------------------------------------
    def _draw_hud(self, img: np.ndarray, state: TrackState) -> None:
        """Draws the right-side info panel."""
        H, W = img.shape[:2]
        panel_w = 220
        x0 = W - panel_w - 8
        y0 = 8

        # Semi-transparent background
        overlay = img.copy()
        cv2.rectangle(overlay, (x0 - 6, y0 - 6),
                      (x0 + panel_w, y0 + 170), (10, 10, 10), -1)
        cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)

        fps = self._fps()
        lines = [
            (f"FPS: {fps:.1f}",          COLORS["ui_text"]),
            (f"Frame: {state.frame_id}", COLORS["ui_text"]),
            (f"Source: {state.trusted_source}", COLORS.get(state.trusted_source, COLORS["ui_text"])),
            (f"BBox: {state.bbox[0]},{state.bbox[1]}  {state.bbox[2]}x{state.bbox[3]}", COLORS["ui_text"]),
            (f"Proc: {state.timestamp_ms:.1f} ms", COLORS["ui_text"]),
        ]
        for i, (text, color) in enumerate(lines):
            cv2.putText(img, text, (x0, y0 + 22 + i * 22),
                        FONT_SMALL, 0.50, color, 1, cv2.LINE_AA)

        # Confidence bars
        bar_y = y0 + 130
        self._draw_conf_bar(img, x0, bar_y,       panel_w - 10,
                            state.tracker_confidence, "VIS", COLORS["bar_vis"])
        self._draw_conf_bar(img, x0, bar_y + 28,  panel_w - 10,
                            state.motion_confidence, " KF", COLORS["bar_kf"])

    def _draw_conf_bar(
        self,
        img: np.ndarray,
        x: int, y: int, w: int,
        conf: float,
        label: str,
        color: Tuple[int, int, int],
    ) -> None:
        cv2.putText(img, label, (x, y + 12), FONT_SMALL, 0.42,
                    COLORS["ui_text"], 1, cv2.LINE_AA)
        bx = x + 32
        bw = w - 32
        bh = 12
        cv2.rectangle(img, (bx, y), (bx + bw, y + bh), COLORS["bar_bg"], -1)
        filled = int(conf * bw)
        if filled > 0:
            cv2.rectangle(img, (bx, y), (bx + filled, y + bh), color, -1)
        cv2.rectangle(img, (bx, y), (bx + bw, y + bh), (120, 120, 120), 1)
        cv2.putText(img, f"{conf:.2f}", (bx + bw + 4, y + 10),
                    FONT_SMALL, 0.38, COLORS["ui_text"], 1, cv2.LINE_AA)

    def _fps(self) -> float:
        if len(self._ts) < 2:
            return 0.0
        return (len(self._ts) - 1) / (self._ts[-1] - self._ts[0] + 1e-9)
