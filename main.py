#!/usr/bin/env python3
"""
GPU-Accelerated Object Tracker — main entry point.

Usage
-----
  python main.py --video path/to/video.mp4
  python main.py --video path/to/video.mp4 --algo CSRT --output tracked.mp4
  python main.py --video path/to/video.mp4 --bbox 100 80 60 90   # skip GUI ROI
  python main.py --webcam 0                                        # live webcam

Controls (interactive window)
------------------------------
  SPACE / p  – pause / resume
  r          – re-select ROI on current frame
  q / ESC    – quit
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2

# Make sure src/ is on the path when running from project root
sys.path.insert(0, str(Path(__file__).parent / "src"))

from tracker import FusionTracker, TrackState
from visualizer import Visualizer


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GPU-Accelerated Fusion Object Tracker"
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--video",  type=str, help="Path to input video file.")
    src.add_argument("--webcam", type=int, help="Webcam device index (e.g. 0).")

    p.add_argument("--algo",   default="CSRT",
                   choices=["CSRT", "KCF", "MOSSE", "MIL", "DASIARMPN", "SIAMRPNPP", "NANOTRACK", "OSTRACK", "MIXFORMER"],
                   help="Visual tracker algorithm (default: CSRT).")
    p.add_argument("--output", type=str, default=None,
                   help="Save tracked video to this path (.mp4).")
    p.add_argument("--bbox",   type=int, nargs=4, metavar=("X","Y","W","H"),
                   default=None, help="Initial bounding box (skip GUI selection).")
    p.add_argument("--csv",    type=str, default=None,
                   help="Write per-frame tracking data to a CSV file.")
    p.add_argument("--no-display", action="store_true",
                   help="Disable the live preview window (useful for headless runs).")
    p.add_argument("--start-frame", type=int, default=0,
                   help="Frame index to start tracking from (default: 0).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# ROI selector
# ---------------------------------------------------------------------------

def select_roi(frame: cv2.Mat) -> Optional[Tuple[int, int, int, int]]:
    """Open OpenCV ROI selector; returns (x,y,w,h) or None on cancel."""
    win_name = "Select target — SPACE/ENTER to confirm, C to cancel"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 960, 540)
    
    print("[Info] Draw a bounding box around the target, then press SPACE or ENTER.")
    bbox = cv2.selectROI(
        win_name,
        frame, fromCenter=False, showCrosshair=True,
    )
    cv2.destroyAllWindows()
    if bbox[2] == 0 or bbox[3] == 0:
        return None
    return tuple(int(v) for v in bbox)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Tracking loop
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    # ---- Open video source ------------------------------------------------
    if args.video:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            sys.exit(f"[Error] Cannot open video: {args.video}")
    else:
        cap = cv2.VideoCapture(args.webcam)
        if not cap.isOpened():
            sys.exit(f"[Error] Cannot open webcam index {args.webcam}")

    fps_in   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # -1 for live streams

    print(f"[Info] Source: {width}×{height}  |  {fps_in:.1f} fps  |  "
          f"{n_frames if n_frames > 0 else '∞'} frames")

    # ---- Skip to start frame ----------------------------------------------
    start_frame = args.start_frame
    if start_frame > 0:
        if args.video:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            print(f"[Info] Skipped to start frame: {start_frame}")
        else:
            for _ in range(start_frame):
                cap.read()

    # ---- Read target frame ------------------------------------------------
    ok, target_frame = cap.read()
    if not ok:
        sys.exit(f"[Error] Could not read frame at index {start_frame}.")

    # ---- Bounding box -----------------------------------------------------
    if args.bbox:
        init_bbox = tuple(args.bbox)
    else:
        init_bbox = select_roi(target_frame)
        if init_bbox is None:
            sys.exit("[Info] ROI selection cancelled. Exiting.")

    print(f"[Info] Initial bbox: {init_bbox}")

    # ---- Setup ------------------------------------------------------------
    fusion   = FusionTracker(target_frame, init_bbox, algo=args.algo)
    viz      = Visualizer(fps_window=30)
    paused   = False

    # Video writer
    writer: Optional[cv2.VideoWriter] = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, fps_in, (width, height))
        print(f"[Info] Writing output to: {args.output}")

    # CSV writer
    csv_file = None
    csv_writer_obj = None
    if args.csv:
        csv_file = open(args.csv, "w", newline="")
        csv_writer_obj = csv.writer(csv_file)
        csv_writer_obj.writerow([
            "frame_id", "x", "y", "w", "h",
            "tracker_confidence", "motion_confidence",
            "trusted_source", "proc_ms",
        ])

    # Draw initial frame (not tracked yet — just the ROI)
    display_init = target_frame.copy()
    x0, y0, w0, h0 = init_bbox
    cv2.rectangle(display_init, (x0, y0), (x0+w0, y0+h0), (0, 255, 128), 2)
    cv2.putText(display_init, "Initialised", (x0, max(0, y0-8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 128), 2)

    if not args.no_display:
        cv2.namedWindow("GPU Fusion Tracker", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("GPU Fusion Tracker", 960, 540)
        cv2.imshow("GPU Fusion Tracker", display_init)
        cv2.waitKey(300)

    # ---- Main loop --------------------------------------------------------
    frame_count = 0
    t_start = time.perf_counter()

    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                print("[Info] End of video.")
                break

            frame_count += 1

            # Track
            state: TrackState = fusion.update(frame)

            # Visualise
            canvas = viz.draw(frame, state)

            if writer:
                writer.write(canvas)

            if csv_writer_obj:
                x, y, w, h = state.bbox
                csv_writer_obj.writerow([
                    state.frame_id, x, y, w, h,
                    state.tracker_confidence,
                    state.motion_confidence,
                    state.trusted_source,
                    state.timestamp_ms,
                ])

            if not args.no_display:
                cv2.imshow("GPU Fusion Tracker", canvas)

        # Key handling
        key = cv2.waitKey(1) & 0xFF if not args.no_display else 0xFF
        if key in (ord("q"), 27):        # q / ESC
            break
        elif key in (ord(" "), ord("p")):
            paused = not paused
            print(f"[Info] {'Paused' if paused else 'Resumed'}")
        elif key == ord("r") and paused:
            # Re-initialise on current frame
            new_bbox = select_roi(frame)
            if new_bbox:
                fusion = FusionTracker(frame, new_bbox, algo=args.algo)
                print(f"[Info] Re-initialised tracker at {new_bbox}")

    # ---- Cleanup ----------------------------------------------------------
    elapsed = time.perf_counter() - t_start
    print(f"\n[Info] Processed {frame_count} frames in {elapsed:.2f}s  "
          f"({frame_count/elapsed:.1f} fps average)")

    cap.release()
    if writer:
        writer.release()
        print(f"[Info] Saved: {args.output}")
    if csv_file:
        csv_file.close()
        print(f"[Info] CSV saved: {args.csv}")
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run(parse_args())
