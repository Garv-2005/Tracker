#!/usr/bin/env python3
"""
GPU Fusion Tracker — PyQt6 GUI
Full-featured desktop application for GPU-accelerated object tracking.
"""
# ── stdlib ──────────────────────────────────────────────────────────────────
import sys, os, time, threading, queue
from pathlib import Path

# ── third-party ─────────────────────────────────────────────────────────────
import cv2
import numpy as np
from PyQt6.QtCore import (
    Qt, QTimer, QThread, pyqtSignal, QPoint, QRect, QSize, QRectF
)
from PyQt6.QtGui import (
    QImage, QPixmap, QPainter, QPen, QBrush, QColor, QFont,
    QFontDatabase, QLinearGradient, QPainterPath, QCursor, QAction
)
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QGridLayout, QSlider, QComboBox,
    QFileDialog, QFrame, QSizePolicy, QSpacerItem, QProgressBar,
    QGroupBox, QStatusBar, QScrollArea, QToolButton, QStackedWidget,
    QSplitter, QCheckBox, QSpinBox
)

sys.path.insert(0, str(Path(__file__).parent / "src"))
from tracker import FusionTracker, TrackState

# ── colour palette ───────────────────────────────────────────────────────────
C = {
    "bg":        "#0a0c10",
    "panel":     "#111318",
    "panel2":    "#161b22",
    "border":    "#21262d",
    "border2":   "#30363d",
    "accent":    "#00d9ff",
    "accent2":   "#7c3aed",
    "green":     "#3fb950",
    "orange":    "#f0883e",
    "red":       "#f85149",
    "text":      "#e6edf3",
    "text2":     "#8b949e",
    "text3":     "#484f58",
    "tracker":   "#00d9ff",
    "fused":     "#7c3aed",
    "motion":    "#f85149",
}

STYLESHEET = f"""
QMainWindow, QWidget {{
    background: {C['bg']};
    color: {C['text']};
    font-family: 'Consolas', 'JetBrains Mono', 'Courier New', monospace;
    font-size: 13px;
}}
QLabel {{ color: {C['text']}; background: transparent; }}
QGroupBox {{
    color: {C['text2']};
    border: 1px solid {C['border']};
    border-radius: 8px;
    margin-top: 18px;
    padding: 12px 10px 10px 10px;
    font-size: 11px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px; top: 2px;
    color: {C['text3']};
    letter-spacing: 2px;
}}
QPushButton {{
    background: {C['panel2']};
    color: {C['text']};
    border: 1px solid {C['border']};
    border-radius: 6px;
    padding: 8px 16px;
    font-size: 12px;
    font-family: 'Consolas', monospace;
}}
QPushButton:hover {{
    background: {C['border']};
    border-color: {C['border2']};
    color: {C['accent']};
}}
QPushButton:pressed {{ background: {C['border2']}; }}
QPushButton:disabled {{ color: {C['text3']}; border-color: {C['border']}; }}
QPushButton#accent_btn {{
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 #006680, stop:1 #004d5e);
    border: 1px solid {C['accent']};
    color: {C['accent']};
    font-weight: bold;
    letter-spacing: 1px;
}}
QPushButton#accent_btn:hover {{
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 #008fa8, stop:1 #006677);
}}
QPushButton#danger_btn {{
    background: #2d1b1b;
    border: 1px solid {C['red']};
    color: {C['red']};
}}
QPushButton#danger_btn:hover {{ background: #3d2020; }}
QPushButton#roi_btn {{
    background: #1a1a2e;
    border: 1px solid {C['accent2']};
    color: {C['accent2']};
    font-weight: bold;
    letter-spacing: 1px;
}}
QPushButton#roi_btn:hover {{ background: #22224a; }}
QComboBox {{
    background: {C['panel2']};
    border: 1px solid {C['border']};
    border-radius: 6px;
    padding: 6px 10px;
    color: {C['text']};
}}
QComboBox:hover {{ border-color: {C['border2']}; }}
QComboBox::drop-down {{ border: none; width: 20px; }}
QComboBox::down-arrow {{ color: {C['text2']}; }}
QComboBox QAbstractItemView {{
    background: {C['panel2']};
    border: 1px solid {C['border2']};
    selection-background-color: {C['border']};
    color: {C['text']};
}}
QSlider::groove:horizontal {{
    background: {C['border']};
    height: 4px;
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: {C['accent']};
    width: 14px; height: 14px;
    margin: -5px 0;
    border-radius: 7px;
}}
QSlider::sub-page:horizontal {{ background: {C['accent']}; border-radius: 2px; }}
QProgressBar {{
    background: {C['border']};
    border: none;
    border-radius: 3px;
    height: 6px;
    text-align: center;
}}
QProgressBar::chunk {{ background: {C['accent']}; border-radius: 3px; }}
QFrame#divider {{
    background: {C['border']};
    max-height: 1px;
    border: none;
}}
QScrollBar:vertical {{
    background: {C['bg']};
    width: 8px;
    border-radius: 4px;
}}
QScrollBar::handle:vertical {{
    background: {C['border2']};
    border-radius: 4px;
    min-height: 30px;
}}
QStatusBar {{
    background: {C['panel']};
    border-top: 1px solid {C['border']};
    color: {C['text2']};
    font-size: 11px;
}}
QSpinBox {{
    background: {C['panel2']};
    border: 1px solid {C['border']};
    border-radius: 6px;
    padding: 5px 8px;
    color: {C['text']};
}}
QCheckBox {{ color: {C['text2']}; spacing: 6px; }}
QCheckBox::indicator {{
    width: 15px; height: 15px;
    border-radius: 3px;
    border: 1px solid {C['border2']};
    background: {C['panel2']};
}}
QCheckBox::indicator:checked {{
    background: {C['accent']};
    border-color: {C['accent']};
}}
"""

# ═══════════════════════════════════════════════════════════════════════════════
# Video canvas — draws frames + interactive ROI rubber-band
# ═══════════════════════════════════════════════════════════════════════════════
class VideoCanvas(QLabel):
    roi_selected = pyqtSignal(int, int, int, int)   # x,y,w,h in frame coords

    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(640, 360)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet(f"""
            background: #000;
            border: 1px solid {C['border']};
            border-radius: 8px;
        """)
        self._drawing = False
        self._roi_mode = False
        self._p1 = QPoint()
        self._p2 = QPoint()
        self._current_pixmap: QPixmap | None = None
        self._track_state: TrackState | None = None
        self._frame_w = 1
        self._frame_h = 1
        self.setCursor(QCursor(Qt.CursorShape.CrossCursor))

    def set_roi_mode(self, active: bool):
        self._roi_mode = active
        self.setCursor(QCursor(Qt.CursorShape.CrossCursor if active
                               else Qt.CursorShape.ArrowCursor))

    def show_frame(self, frame: np.ndarray, state: TrackState | None = None):
        self._frame_h, self._frame_w = frame.shape[:2]
        self._track_state = state
        
        # Calculate aspect ratio fit to canvas widget size
        widget_w = self.width()
        widget_h = self.height()
        if widget_w <= 0 or widget_h <= 0:
            widget_w, widget_h = 640, 360
            
        scale_w = widget_w / self._frame_w
        scale_h = widget_h / self._frame_h
        scale = min(scale_w, scale_h)
        
        target_w = int(self._frame_w * scale)
        target_h = int(self._frame_h * scale)
        target_w = max(16, target_w)
        target_h = max(16, target_h)
        
        # Highly optimized downscaling on CPU via OpenCV, bypassing slow Qt QPixmap scaling
        display_frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            
        rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        dh, dw, ch = rgb.shape
        # Create QImage and copy it to prevent issues with underlying numpy array memory
        qimg = QImage(rgb.data, dw, dh, ch * dw, QImage.Format.Format_RGB888).copy()
        self._current_pixmap = QPixmap.fromImage(qimg)
        self.update()

    # ── painting ──────────────────────────────────────────────────────────
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        if self._current_pixmap:
            x_off = (self.width()  - self._current_pixmap.width())  // 2
            y_off = (self.height() - self._current_pixmap.height()) // 2
            painter.drawPixmap(x_off, y_off, self._current_pixmap)

            # Draw tracking bbox
            if self._track_state:
                self._draw_track(painter, self._current_pixmap, x_off, y_off)

            # Draw rubber-band ROI
            if self._drawing and not self._p1.isNull() and not self._p2.isNull():
                rect = QRect(self._p1, self._p2).normalized()
                pen = QPen(QColor(C['accent2']), 2, Qt.PenStyle.DashLine)
                painter.setPen(pen)
                painter.setBrush(QBrush(QColor(124, 58, 237, 40)))
                painter.drawRect(rect)
        else:
            # Placeholder
            painter.fillRect(self.rect(), QColor("#000"))
            pen = QPen(QColor(C['text3']))
            painter.setPen(pen)
            painter.setFont(QFont("Consolas", 14))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                             "DROP A VIDEO FILE\nOR CLICK  [ OPEN VIDEO ]")

    def _draw_track(self, painter, scaled_pm, x_off, y_off):
        s = self._track_state
        sx = scaled_pm.width()  / self._frame_w
        sy = scaled_pm.height() / self._frame_h
        x, y, w, h = s.bbox
        rx = int(x * sx) + x_off
        ry = int(y * sy) + y_off
        rw = int(w * sx)
        rh = int(h * sy)

        color_map = {
            "tracker":      QColor(C['tracker']),
            "fused":        QColor(C['fused']),
            "motion_model": QColor(C['motion']),
        }
        col = color_map.get(s.trusted_source, QColor(C['accent']))

        # Box
        pen = QPen(col, 2)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(rx, ry, rw, rh)

        # Corner accents
        L = max(8, min(rw, rh) // 5)
        painter.setPen(QPen(col, 3))
        corners = [
            [(rx, ry), (rx+L, ry), (rx, ry+L)],
            [(rx+rw, ry), (rx+rw-L, ry), (rx+rw, ry+L)],
            [(rx, ry+rh), (rx+L, ry+rh), (rx, ry+rh-L)],
            [(rx+rw, ry+rh), (rx+rw-L, ry+rh), (rx+rw, ry+rh-L)],
        ]
        for corner, p1, p2 in corners:
            painter.drawLine(*corner, *p1)
            painter.drawLine(*corner, *p2)

        # Label chip
        label = s.trusted_source.upper().replace("_", " ")
        fm = painter.fontMetrics()
        tw = fm.horizontalAdvance(label)
        chip_x = rx
        chip_y = max(0, ry - 22)
        painter.setBrush(QBrush(col))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(chip_x, chip_y, tw + 12, 18, 4, 4)
        painter.setPen(QPen(QColor("#000")))
        painter.setFont(QFont("Consolas", 9, QFont.Weight.Bold))
        painter.drawText(chip_x + 6, chip_y + 13, label)

    # ── mouse events ─────────────────────────────────────────────────────
    def mousePressEvent(self, e):
        if self._roi_mode and e.button() == Qt.MouseButton.LeftButton:
            self._drawing = True
            self._p1 = e.position().toPoint()
            self._p2 = self._p1

    def mouseMoveEvent(self, e):
        if self._drawing:
            self._p2 = e.position().toPoint()
            self.update()

    def mouseReleaseEvent(self, e):
        if self._drawing and e.button() == Qt.MouseButton.LeftButton:
            self._drawing = False
            self._p2 = e.position().toPoint()
            self.update()
            self._emit_roi()

    def _emit_roi(self):
        if not self._current_pixmap:
            return
        scaled_w = self._current_pixmap.scaled(
            self.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        ).width()
        scaled_h = self._current_pixmap.scaled(
            self.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        ).height()
        x_off = (self.width()  - scaled_w) // 2
        y_off = (self.height() - scaled_h) // 2

        rect = QRect(self._p1, self._p2).normalized()
        # clip to image area
        img_rect = QRect(x_off, y_off, scaled_w, scaled_h)
        rect = rect.intersected(img_rect)
        if rect.width() < 5 or rect.height() < 5:
            return

        sx = self._frame_w / scaled_w
        sy = self._frame_h / scaled_h
        fx = int((rect.x() - x_off) * sx)
        fy = int((rect.y() - y_off) * sy)
        fw = int(rect.width()  * sx)
        fh = int(rect.height() * sy)
        self.roi_selected.emit(fx, fy, fw, fh)


# ═══════════════════════════════════════════════════════════════════════════════
# Stat badge widget
# ═══════════════════════════════════════════════════════════════════════════════
class StatBadge(QWidget):
    def __init__(self, label: str, value: str = "—", color: str = C['accent']):
        super().__init__()
        self._color = color
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 8, 10, 8)
        lay.setSpacing(2)

        self._val = QLabel(value)
        self._val.setFont(QFont("Consolas", 18, QFont.Weight.Bold))
        self._val.setStyleSheet(f"color: {color};")
        self._val.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._lbl = QLabel(label)
        self._lbl.setFont(QFont("Consolas", 9))
        self._lbl.setStyleSheet(f"color: {C['text3']}; letter-spacing: 1.5px;")
        self._lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)

        lay.addWidget(self._val)
        lay.addWidget(self._lbl)
        self.setStyleSheet(f"""
            background: {C['panel2']};
            border: 1px solid {C['border']};
            border-radius: 8px;
        """)

    def set_value(self, v: str):
        self._val.setText(v)


class ConfBar(QWidget):
    """Labelled confidence bar."""
    def __init__(self, label: str, color: str):
        super().__init__()
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)
        lbl = QLabel(label)
        lbl.setFixedWidth(30)
        lbl.setStyleSheet(f"color: {C['text3']}; font-size: 10px; letter-spacing:1px;")
        self._bar = QProgressBar()
        self._bar.setRange(0, 100)
        self._bar.setValue(0)
        self._bar.setTextVisible(False)
        self._bar.setFixedHeight(6)
        self._bar.setStyleSheet(f"""
            QProgressBar {{ background: {C['border']}; border:none; border-radius:3px; }}
            QProgressBar::chunk {{ background: {color}; border-radius:3px; }}
        """)
        self._pct = QLabel("0%")
        self._pct.setFixedWidth(36)
        self._pct.setStyleSheet(f"color: {color}; font-size: 11px;")
        self._pct.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        lay.addWidget(lbl)
        lay.addWidget(self._bar)
        lay.addWidget(self._pct)

    def set_value(self, v: float):
        pct = int(v * 100)
        self._bar.setValue(pct)
        self._pct.setText(f"{pct}%")


# ═══════════════════════════════════════════════════════════════════════════════
# Worker thread — runs the tracking loop off the UI thread
# ═══════════════════════════════════════════════════════════════════════════════
class TrackerWorker(QThread):
    frame_ready   = pyqtSignal(np.ndarray, object)   # frame, TrackState|None
    stats_updated = pyqtSignal(float, int, int)       # fps, frame_idx, total
    finished_sig  = pyqtSignal()
    error_sig     = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._cap:     cv2.VideoCapture | None = None
        self._tracker: FusionTracker    | None = None
        self._lock     = threading.Lock()
        self._running  = False
        self._paused   = False
        self._seek_to: int | None = None      # frame index to seek to
        self._new_roi: tuple | None = None    # pending ROI reinit
        self._algo     = "CSRT"
        self._video_path = ""
        self._total_frames = 0
        self._fps_ts: list[float] = []
        self._current_frame_idx = 0

    # ── public API (thread-safe) ──────────────────────────────────────────
    def load_video(self, path: str):
        with self._lock:
            if self._cap:
                self._cap.release()
            self._cap = cv2.VideoCapture(path)
            self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self._video_path = path
            self._tracker = None
            self._running = False
            self._paused  = False
            self._current_frame_idx = 0

    def set_algo(self, algo: str):
        self._algo = algo

    def seek(self, frame_idx: int):
        with self._lock:
            self._seek_to = frame_idx

    def init_tracker(self, frame: np.ndarray, bbox: tuple):
        with self._lock:
            self._new_roi = (frame.copy(), bbox)

    def reinit_roi(self, frame: np.ndarray, bbox: tuple):
        """Re-init tracker mid-stream."""
        with self._lock:
            self._new_roi = (frame.copy(), bbox)

    def pause(self):
        self._paused = True

    def resume(self):
        self._paused = False
        if not self.isRunning():
            self.start()

    def stop(self):
        self._running = False

    @property
    def total_frames(self): return self._total_frames
    @property
    def current_frame(self): return self._current_frame_idx

    # ── thread body ───────────────────────────────────────────────────────
    def run(self):
        self._running = True
        while self._running:
            # Handle pending seek
            with self._lock:
                if self._seek_to is not None:
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, self._seek_to)
                    self._current_frame_idx = self._seek_to
                    self._seek_to = None

            if self._paused:
                time.sleep(0.02)
                continue

            with self._lock:
                if not self._cap or not self._cap.isOpened():
                    break
                ok, frame = self._cap.read()
                if not ok:
                    self.finished_sig.emit()
                    break
                self._current_frame_idx = int(self._cap.get(cv2.CAP_PROP_POS_FRAMES))

                # Handle pending tracker init / reinit
                if self._new_roi is not None:
                    init_frame, bbox = self._new_roi
                    self._new_roi = None
                    try:
                        self._tracker = FusionTracker(init_frame, bbox, algo=self._algo)
                    except Exception as ex:
                        self.error_sig.emit(str(ex))

            # Run tracker
            state = None
            if self._tracker:
                try:
                    state = self._tracker.update(frame)
                except Exception as ex:
                    self.error_sig.emit(str(ex))
                    self._tracker = None

            # FPS estimate
            now = time.perf_counter()
            self._fps_ts.append(now)
            self._fps_ts = [t for t in self._fps_ts if now - t < 1.0]
            fps = len(self._fps_ts)

            self.frame_ready.emit(frame, state)
            self.stats_updated.emit(fps, self._current_frame_idx, self._total_frames)

            # Throttle to ~60 fps max
            time.sleep(0.005)

        self._running = False


# ═══════════════════════════════════════════════════════════════════════════════
# Side panel — controls
# ═══════════════════════════════════════════════════════════════════════════════
def _divider():
    f = QFrame()
    f.setObjectName("divider")
    f.setFixedHeight(1)
    return f

def _label(text, color=C['text2'], size=11, bold=False):
    l = QLabel(text)
    weight = "bold" if bold else "normal"
    l.setStyleSheet(f"color:{color}; font-size:{size}px; font-weight:{weight}; letter-spacing:1px;")
    return l


# ═══════════════════════════════════════════════════════════════════════════════
# Main window
# ═══════════════════════════════════════════════════════════════════════════════
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GPU Fusion Tracker")
        self.resize(1380, 820)
        self.setMinimumSize(1100, 650)
        self.setStyleSheet(STYLESHEET)

        self._worker = TrackerWorker()
        self._worker.frame_ready.connect(self._on_frame)
        self._worker.stats_updated.connect(self._on_stats)
        self._worker.finished_sig.connect(self._on_video_end)
        self._worker.error_sig.connect(self._on_error)

        self._current_frame: np.ndarray | None = None
        self._total_frames = 0
        self._roi_pending = False        # waiting for user to draw ROI
        self._seek_frame_pending = False # user wants to pick a frame first

        self._build_ui()
        self._update_controls()

    # ── UI construction ───────────────────────────────────────────────────
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── left sidebar ──────────────────────────────────────────────────
        sidebar = QWidget()
        sidebar.setFixedWidth(280)
        sidebar.setStyleSheet(f"background:{C['panel']}; border-right:1px solid {C['border']};")
        sb_lay = QVBoxLayout(sidebar)
        sb_lay.setContentsMargins(16, 20, 16, 16)
        sb_lay.setSpacing(12)

        # Logo
        logo = QLabel("⬡ FUSION\nTRACKER")
        logo.setFont(QFont("Consolas", 14, QFont.Weight.Bold))
        logo.setStyleSheet(f"color:{C['accent']}; letter-spacing:3px; line-height:1.4;")
        sb_lay.addWidget(logo)

        version = QLabel("GPU · KALMAN · CSRT  v2.0")
        version.setStyleSheet(f"color:{C['text3']}; font-size:9px; letter-spacing:2px;")
        sb_lay.addWidget(version)
        sb_lay.addWidget(_divider())

        # ── Video source ──────────────────────────────────────────────────
        sb_lay.addWidget(_label("VIDEO SOURCE", C['text3']))
        self._file_label = QLabel("No file loaded")
        self._file_label.setStyleSheet(f"color:{C['text2']}; font-size:11px; padding:6px 8px;"
                                        f"background:{C['panel2']}; border:1px solid {C['border']};"
                                        f"border-radius:5px;")
        self._file_label.setWordWrap(True)
        sb_lay.addWidget(self._file_label)
        btn_open = QPushButton("  OPEN VIDEO FILE")
        btn_open.setObjectName("accent_btn")
        btn_open.clicked.connect(self._open_video)
        sb_lay.addWidget(btn_open)

        sb_lay.addWidget(_divider())

        # ── Frame picker ──────────────────────────────────────────────────
        sb_lay.addWidget(_label("FRAME SELECTION", C['text3']))
        frame_row = QHBoxLayout()
        frame_row.setSpacing(6)
        self._frame_spin = QSpinBox()
        self._frame_spin.setRange(0, 99999)
        self._frame_spin.setPrefix("Frame: ")
        self._frame_spin.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._frame_spin.valueChanged.connect(self._on_spin_changed)
        btn_go = QPushButton("GO")
        btn_go.setFixedWidth(46)
        btn_go.clicked.connect(self._seek_to_spin)
        frame_row.addWidget(self._frame_spin)
        frame_row.addWidget(btn_go)
        sb_lay.addLayout(frame_row)

        self._timeline = QSlider(Qt.Orientation.Horizontal)
        self._timeline.setRange(0, 100)
        self._timeline.sliderMoved.connect(self._on_timeline_moved)
        self._timeline.sliderReleased.connect(self._on_timeline_released)
        sb_lay.addWidget(self._timeline)

        self._frame_label = QLabel("Frame — / —")
        self._frame_label.setStyleSheet(f"color:{C['text3']}; font-size:10px;")
        sb_lay.addWidget(self._frame_label)
        sb_lay.addWidget(_divider())

        # ── ROI ───────────────────────────────────────────────────────────
        sb_lay.addWidget(_label("REGION OF INTEREST", C['text3']))

        self._roi_hint = QLabel("Draw a box on the video frame\nto select the tracking target.")
        self._roi_hint.setStyleSheet(f"color:{C['text2']}; font-size:11px;"
                                      f"padding:8px; background:{C['panel2']};"
                                      f"border:1px solid {C['border']}; border-radius:5px;")
        self._roi_hint.setWordWrap(True)
        sb_lay.addWidget(self._roi_hint)

        self._btn_roi = QPushButton("  SELECT ROI  (draw on frame)")
        self._btn_roi.setObjectName("roi_btn")
        self._btn_roi.clicked.connect(self._activate_roi_mode)
        sb_lay.addWidget(self._btn_roi)

        self._roi_status = QLabel("No ROI set")
        self._roi_status.setStyleSheet(f"color:{C['text3']}; font-size:10px; font-style:italic;")
        sb_lay.addWidget(self._roi_status)
        sb_lay.addWidget(_divider())

        # ── Model ─────────────────────────────────────────────────────────
        sb_lay.addWidget(_label("TRACKER ALGORITHM", C['text3']))
        self._algo_combo = QComboBox()
        for algo, desc in [
            ("CSRT",  "CSRT — Accurate, moderate speed"),
            ("KCF",   "KCF  — Fast, good for rigid"),
            ("MOSSE", "MOSSE — Ultra-fast, less robust"),
            ("MIL",   "MIL  — Robust to noise"),
            ("NANOTRACK", "NanoTrack — High-speed CPU Siamese"),
            ("DASIARMPN", "DaSiamRPN (PyTorch) — Robust"),
            ("SIAMRPNPP", "SiamRPN++ (PyTorch) — Deep features"),
            ("OSTRACK", "OSTrack (PyTorch) — One-Stream Transformer"),
            ("MIXFORMER", "MixFormer (PyTorch) — Two-Stream Transformer"),
        ]:
            self._algo_combo.addItem(desc, algo)
        self._algo_combo.currentIndexChanged.connect(self._on_algo_changed)
        sb_lay.addWidget(self._algo_combo)

        gpu_row = QHBoxLayout()
        self._gpu_label = QLabel()
        self._gpu_label.setStyleSheet("font-size:10px;")
        self._set_gpu_label()
        gpu_row.addWidget(self._gpu_label)
        sb_lay.addLayout(gpu_row)
        sb_lay.addWidget(_divider())

        # ── Playback controls ────────────────────────────────────────────
        sb_lay.addWidget(_label("PLAYBACK", C['text3']))
        ctrl_row = QHBoxLayout()
        ctrl_row.setSpacing(6)
        self._btn_play = QPushButton("▶  PLAY")
        self._btn_play.setObjectName("accent_btn")
        self._btn_play.clicked.connect(self._toggle_play)
        self._btn_stop = QPushButton("■  STOP")
        self._btn_stop.setObjectName("danger_btn")
        self._btn_stop.clicked.connect(self._stop)
        ctrl_row.addWidget(self._btn_play)
        ctrl_row.addWidget(self._btn_stop)
        sb_lay.addLayout(ctrl_row)
        sb_lay.addStretch()

        # ── right side ───────────────────────────────────────────────────
        right = QWidget()
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(16, 16, 16, 16)
        right_lay.setSpacing(12)

        # ── stats row ─────────────────────────────────────────────────────
        stats_row = QHBoxLayout()
        stats_row.setSpacing(10)
        self._badge_fps    = StatBadge("FPS",    "—",  C['accent'])
        self._badge_frame  = StatBadge("FRAME",  "—",  C['text2'])
        self._badge_source = StatBadge("SOURCE", "—",  C['accent2'])
        self._badge_proc   = StatBadge("PROC ms","—",  C['green'])
        for b in [self._badge_fps, self._badge_frame, self._badge_source, self._badge_proc]:
            stats_row.addWidget(b)
        right_lay.addLayout(stats_row)

        # ── canvas ───────────────────────────────────────────────────────
        self._canvas = VideoCanvas()
        self._canvas.roi_selected.connect(self._on_roi_drawn)
        right_lay.addWidget(self._canvas, stretch=1)

        # ── confidence bars + bbox info ──────────────────────────────────
        info_row = QHBoxLayout()
        info_row.setSpacing(12)

        conf_box = QGroupBox("CONFIDENCE")
        conf_box.setFixedHeight(90)
        conf_lay = QVBoxLayout(conf_box)
        conf_lay.setSpacing(6)
        self._vis_bar = ConfBar("VIS", C['tracker'])
        self._kf_bar  = ConfBar("KF",  C['fused'])
        conf_lay.addWidget(self._vis_bar)
        conf_lay.addWidget(self._kf_bar)
        info_row.addWidget(conf_box, stretch=2)

        bbox_box = QGroupBox("BOUNDING BOX")
        bbox_box.setFixedHeight(90)
        bbox_lay = QGridLayout(bbox_box)
        bbox_lay.setSpacing(4)
        self._bbox_labels = {}
        for i, (k, label) in enumerate([("x","X"), ("y","Y"), ("w","W"), ("h","H")]):
            lbl = QLabel(f"{label}:")
            lbl.setStyleSheet(f"color:{C['text3']}; font-size:11px;")
            val = QLabel("—")
            val.setStyleSheet(f"color:{C['accent']}; font-size:13px; font-weight:bold;")
            bbox_lay.addWidget(lbl, i//2, (i%2)*2)
            bbox_lay.addWidget(val, i//2, (i%2)*2+1)
            self._bbox_labels[k] = val
        info_row.addWidget(bbox_box, stretch=1)

        source_box = QGroupBox("TRUST SOURCE")
        source_box.setFixedHeight(90)
        src_lay = QVBoxLayout(source_box)
        self._source_indicator = QLabel("WAITING")
        self._source_indicator.setFont(QFont("Consolas", 14, QFont.Weight.Bold))
        self._source_indicator.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._source_indicator.setStyleSheet(f"color:{C['text3']};")
        src_lay.addWidget(self._source_indicator)
        info_row.addWidget(source_box, stretch=1)

        right_lay.addLayout(info_row)

        # ── assemble ──────────────────────────────────────────────────────
        root.addWidget(sidebar)
        root.addWidget(right, stretch=1)

        # status bar
        self.statusBar().showMessage("Ready — open a video file to begin")

    # ── GPU label ─────────────────────────────────────────────────────────
    def _set_gpu_label(self):
        try:
            import torch
            torch_cuda = torch.cuda.is_available()
            cv2_cuda = False
            try:
                cv2_cuda = cv2.cuda.getCudaEnabledDeviceCount() > 0
            except Exception:
                pass

            if torch_cuda or cv2_cuda:
                status = []
                if cv2_cuda:
                    status.append("OpenCV CUDA")
                if torch_cuda:
                    status.append("PyTorch CUDA")
                self._gpu_label.setText(f"  ⚡ GPU detected ({' & '.join(status)})")
                self._gpu_label.setStyleSheet(f"color:{C['green']}; font-size:10px;")
            else:
                self._gpu_label.setText("  CPU mode (no CUDA)")
                self._gpu_label.setStyleSheet(f"color:{C['text3']}; font-size:10px;")
        except Exception:
            self._gpu_label.setText("  CPU mode")
            self._gpu_label.setStyleSheet(f"color:{C['text3']}; font-size:10px;")

    # ── file open ─────────────────────────────────────────────────────────
    def _open_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Video", "",
            "Video files (*.mp4 *.avi *.mov *.mkv *.webm *.m4v);;All files (*)"
        )
        if not path:
            return
        self._worker.stop()
        self._worker.wait(500)
        self._worker.load_video(path)
        self._total_frames = self._worker.total_frames
        self._frame_spin.setMaximum(max(0, self._total_frames - 1))
        self._timeline.setRange(0, max(1, self._total_frames - 1))
        name = Path(path).name
        self._file_label.setText(name)
        self._roi_status.setText("No ROI set")
        self._current_frame = None
        self._update_controls()

        # Read and show frame 0 immediately
        cap = cv2.VideoCapture(path)
        ok, frame = cap.read()
        cap.release()
        if ok:
            self._current_frame = frame
            self._canvas.show_frame(frame)
        self.statusBar().showMessage(f"Loaded: {name}  |  {self._total_frames} frames")

    # ── frame seeking ─────────────────────────────────────────────────────
    def _on_spin_changed(self, val):
        # Sync timeline to spin (without re-triggering)
        self._timeline.blockSignals(True)
        self._timeline.setValue(val)
        self._timeline.blockSignals(False)

    def _seek_to_spin(self):
        idx = self._frame_spin.value()
        self._seek_and_show(idx)

    def _on_timeline_moved(self, val):
        self._frame_spin.blockSignals(True)
        self._frame_spin.setValue(val)
        self._frame_spin.blockSignals(False)

    def _on_timeline_released(self):
        idx = self._timeline.value()
        self._seek_and_show(idx)

    def _seek_and_show(self, idx: int):
        """Seek to frame, show it, pause worker."""
        if not self._worker._cap:
            return
        was_running = self._worker.isRunning() and not self._worker._paused
        self._worker.pause()
        # Read frame directly
        cap = cv2.VideoCapture(self._worker._video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        cap.release()
        if ok:
            self._current_frame = frame
            self._canvas.show_frame(frame)
            self._worker.seek(idx)
            self._frame_label.setText(f"Frame {idx} / {self._total_frames}")
        if was_running:
            self._worker.resume()

    # ── ROI ───────────────────────────────────────────────────────────────
    def _activate_roi_mode(self):
        if not self._current_frame is not None or self._worker._video_path:
            pass
        self._roi_pending = True
        self._canvas.set_roi_mode(True)
        self._roi_status.setText("Draw ROI on the frame above →")
        self._roi_hint.setStyleSheet(
            f"color:{C['accent2']}; font-size:11px; padding:8px;"
            f"background:#1a1a2e; border:1px solid {C['accent2']}; border-radius:5px;"
        )
        self.statusBar().showMessage("Draw a bounding box on the video to select tracking target")

    def _on_roi_drawn(self, x, y, w, h):
        if not self._roi_pending:
            return
        self._roi_pending = False
        self._canvas.set_roi_mode(False)
        self._roi_hint.setStyleSheet(
            f"color:{C['text2']}; font-size:11px; padding:8px;"
            f"background:{C['panel2']}; border:1px solid {C['border']}; border-radius:5px;"
        )

        if self._current_frame is None:
            self.statusBar().showMessage("Error: no frame available for ROI init")
            return

        bbox = (x, y, w, h)
        self._roi_status.setText(f"ROI: ({x},{y}) {w}×{h}  [algo: {self._worker._algo}]")
        self._worker.init_tracker(self._current_frame, bbox)
        self.statusBar().showMessage(f"Tracker initialised at ({x},{y}) {w}×{h} — press PLAY")
        self._update_controls()

    # ── algo change ───────────────────────────────────────────────────────
    def _on_algo_changed(self, idx):
        algo = self._algo_combo.itemData(idx)
        self._worker.set_algo(algo)
        self.statusBar().showMessage(f"Algorithm changed to {algo} — re-select ROI to apply")

    # ── playback ──────────────────────────────────────────────────────────
    def _toggle_play(self):
        if self._worker.isRunning() and not self._worker._paused:
            self._worker.pause()
            self._btn_play.setText("▶  RESUME")
        else:
            if not self._worker._video_path:
                return
            self._worker.resume()
            self._btn_play.setText("⏸  PAUSE")

    def _stop(self):
        self._worker.stop()
        self._worker.wait(300)
        self._btn_play.setText("▶  PLAY")
        self.statusBar().showMessage("Stopped")

    # ── worker callbacks ──────────────────────────────────────────────────
    def _on_frame(self, frame: np.ndarray, state):
        self._current_frame = frame
        self._canvas.show_frame(frame, state)
        if state:
            src_colors = {
                "tracker":      C['tracker'],
                "fused":        C['fused'],
                "motion_model": C['motion'],
            }
            col = src_colors.get(state.trusted_source, C['text'])
            self._source_indicator.setText(state.trusted_source.upper().replace("_","\n"))
            self._source_indicator.setStyleSheet(f"color:{col}; font-size:12px; font-weight:bold;")
            self._vis_bar.set_value(state.tracker_confidence)
            self._kf_bar.set_value(state.motion_confidence)
            x, y, w, h = state.bbox
            self._bbox_labels['x'].setText(str(x))
            self._bbox_labels['y'].setText(str(y))
            self._bbox_labels['w'].setText(str(w))
            self._bbox_labels['h'].setText(str(h))
            self._badge_proc.set_value(f"{state.timestamp_ms:.1f}")

    def _on_stats(self, fps: float, frame_idx: int, total: int):
        self._badge_fps.set_value(f"{fps:.0f}")
        self._badge_frame.set_value(str(frame_idx))
        self._frame_label.setText(f"Frame {frame_idx} / {total}")
        # Sync timeline without triggering seek
        self._timeline.blockSignals(True)
        self._timeline.setValue(frame_idx)
        self._timeline.blockSignals(False)
        self._frame_spin.blockSignals(True)
        self._frame_spin.setValue(frame_idx)
        self._frame_spin.blockSignals(False)

    def _on_video_end(self):
        self._btn_play.setText("▶  PLAY")
        self.statusBar().showMessage("Video ended")

    def _on_error(self, msg: str):
        self.statusBar().showMessage(f"Error: {msg}")

    def _update_controls(self):
        has_video = bool(self._worker._video_path)
        self._btn_play.setEnabled(has_video)
        self._btn_stop.setEnabled(has_video)
        self._btn_roi.setEnabled(has_video)
        self._timeline.setEnabled(has_video)
        self._frame_spin.setEnabled(has_video)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.accept()

    def dropEvent(self, e):
        urls = e.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            if Path(path).suffix.lower() in ('.mp4','.avi','.mov','.mkv','.webm','.m4v'):
                self._worker.stop()
                self._worker.wait(300)
                self._worker.load_video(path)
                self._total_frames = self._worker.total_frames
                self._frame_spin.setMaximum(max(0, self._total_frames-1))
                self._timeline.setRange(0, max(1, self._total_frames-1))
                self._file_label.setText(Path(path).name)
                cap = cv2.VideoCapture(path)
                ok, frame = cap.read(); cap.release()
                if ok:
                    self._current_frame = frame
                    self._canvas.show_frame(frame)
                self._update_controls()
                self.statusBar().showMessage(f"Loaded: {Path(path).name}")

    def closeEvent(self, e):
        self._worker.stop()
        self._worker.wait(1000)
        e.accept()


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setApplicationName("GPU Fusion Tracker")
    win = MainWindow()
    win.setAcceptDrops(True)
    win.show()
    sys.exit(app.exec())
