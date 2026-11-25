"""Streamlit UI for restricted-item detection. Run: streamlit run app.py"""
from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from typing import Dict, List, Optional

import cv2
import numpy as np
import streamlit as st

from detector import Detector, VideoSource, list_available_weights
from utils import (IMAGE_EXTENSIONS, VIDEO_EXTENSIONS, AlertLogger,
                   InputValidationError, annotate_frame, bgr_to_rgb, encode_jpeg,
                   load_image_bytes, probe_video, safe_remove, save_upload_to_temp)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")

st.set_page_config(page_title="Restricted Item Detection", page_icon="🛡️", layout="wide")


# --------------------------------------------------------------------------- #
# Cached model + session state
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading model (first run may download weights)...")
def get_detector(weights_path: Optional[str], mtime: float) -> Detector:
    """Loaded once per (weights, mtime). Detector() itself never raises."""
    return Detector(weights_path)


def new_stats() -> Dict[str, float]:
    return {"frames": 0, "total": 0, "high": 0, "last_ms": 0.0, "ema_ms": 0.0}


def init_state() -> None:
    ss = st.session_state
    ss.setdefault("logger", AlertLogger())
    ss.setdefault("stats", new_stats())
    ss.setdefault("live_running", False)
    ss.setdefault("image_results", [])
    ss.setdefault("preview", None)


def clear_results() -> None:
    ss = st.session_state
    ss.logger = AlertLogger()
    ss.stats = new_stats()
    ss.image_results = []
    ss.preview = None
    ss.live_running = False


def render_metrics(ph: Dict[str, "st.delta_generator.DeltaGenerator"]) -> None:
    s = st.session_state.stats
    fps = 1000.0 / s["ema_ms"] if s["ema_ms"] > 0 else 0.0
    ph["total"].metric("Total detections", int(s["total"]))
    ph["high"].metric("High-risk alerts", int(s["high"]))
    ph["ms"].metric("Inference time", f"{s['last_ms']:.0f} ms")
    ph["fps"].metric("Inference speed", f"{fps:.1f} FPS")
    ph["frames"].metric("Frames processed", int(s["frames"]))


def process_frame(detector: Detector, frame: np.ndarray, conf: float,
                  source: str, frame_index: int):
    """Detect, annotate, update stats and log. Returns (annotated BGR frame, detections)."""
    dets, ms = detector.detect(frame, conf)
    annotated = annotate_frame(frame, dets)
    s = st.session_state.stats
    s["frames"] += 1
    s["total"] += len(dets)
    s["high"] += sum(1 for d in dets if d.risk == "HIGH")
    if ms > 0:
        s["last_ms"] = ms
        s["ema_ms"] = ms if s["ema_ms"] == 0 else 0.9 * s["ema_ms"] + 0.1 * ms
    st.session_state.logger.log(source, frame_index, dets, annotated)
    return annotated, dets


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
init_state()

with st.sidebar:
    st.header("Controls")
    source_mode = st.radio("Input source",
                           ["Upload Images", "Upload Video", "Live Webcam / RTSP"])
    conf = st.slider("Confidence threshold", 0.10, 0.95, 0.45, 0.01)
    options = list_available_weights()
    choice = st.selectbox("Model weights", list(options.keys()))
    weights_path = options[choice]
    mtime = os.path.getmtime(weights_path) if weights_path and os.path.isfile(weights_path) else 0.0
    if st.button("Reload model", use_container_width=True):
        get_detector.clear()
        st.rerun()
    stride = 1
    if source_mode != "Upload Images":
        stride = st.slider("Process every Nth frame", 1, 10, 2,
                           help="Higher values run faster but may miss brief appearances.")
    if st.button("Clear results", use_container_width=True):
        clear_results()
        st.rerun()

try:
    detector = get_detector(weights_path, mtime)
except Exception as exc:  # defensive: Detector() should never raise
    st.error(f"Model initialisation failed unexpectedly: {exc}")
    st.stop()

with st.sidebar:
    st.divider()
    st.caption(f"Active model: **{detector.model_name}**  \nDevice: **{detector.device}**")
    for level, text in detector.messages[-6:]:
        {"info": st.info, "success": st.success,
         "warning": st.warning, "error": st.error}.get(level, st.info)(text)
    if detector.ready:
        with st.expander("Model classes"):
            st.write(", ".join(detector.class_names))
    st.caption("Decision-support tool. Every alert needs human review.")


# --------------------------------------------------------------------------- #
# Dashboard
# --------------------------------------------------------------------------- #
st.title("Restricted Item Detection")
if not detector.ready:
    st.error("No model is loaded, so detection is disabled. Connect to the internet once "
             "so the default weights can download, or place a .pt file in ./weights, "
             "then click **Reload model**.")
elif detector.used_fallback:
    st.warning("Running on the default COCO model: it detects knives, scissors and bags only. "
               "Use custom weights to detect firearms, ammunition, explosives or contraband.")

cols = st.columns(5)
metric_ph = {k: c.empty() for k, c in zip(["total", "high", "ms", "fps", "frames"], cols)}
render_metrics(metric_ph)
st.divider()


def render_image_results() -> None:
    results: List[dict] = st.session_state.image_results
    if not results:
        return
    grid = st.columns(3)
    for i, r in enumerate(results):
        with grid[i % 3]:
            st.image(r["jpeg"], caption=r["caption"], use_container_width=True)


# ---- Images ---------------------------------------------------------------- #
def run_images() -> None:
    files = st.file_uploader("Upload images (single or batch)",
                             type=list(IMAGE_EXTENSIONS), accept_multiple_files=True)
    if st.button("Run detection", type="primary", disabled=not (files and detector.ready)):
        st.session_state.image_results = []
        bar = st.progress(0.0)
        for i, f in enumerate(files):
            try:
                frame = load_image_bytes(f.getvalue(), f.name)
                annotated, dets = process_frame(detector, frame, conf, f.name, 0)
                high = sum(1 for d in dets if d.risk == "HIGH")
                st.session_state.image_results.append({
                    "jpeg": encode_jpeg(annotated, max_width=1280, quality=90),
                    "caption": f"{f.name}: {len(dets)} detection(s), {high} high-risk",
                })
            except InputValidationError as exc:
                st.warning(f"Skipped: {exc}")
            except Exception as exc:
                st.warning(f"Skipped '{f.name}' after an unexpected error: {exc}")
            bar.progress((i + 1) / len(files))
            render_metrics(metric_ph)
        bar.empty()
    render_image_results()


# ---- Video ----------------------------------------------------------------- #
def run_video() -> None:
    upload = st.file_uploader("Upload a video", type=list(VIDEO_EXTENSIONS))
    preview_ph = st.empty()
    if st.session_state.preview:
        preview_ph.image(st.session_state.preview, caption="Last processed frame",
                         use_container_width=True)
    if not st.button("Process video", type="primary", disabled=not (upload and detector.ready)):
        return
    path: Optional[str] = None
    src: Optional[VideoSource] = None
    try:
        path = save_upload_to_temp(upload, upload.name)
        info = probe_video(path)
        src = VideoSource(path)
        if not src.open():
            raise InputValidationError("Video could not be opened.")
        bar = st.progress(0.0)
        idx, last_draw, last = 0, 0.0, None
        while True:
            ok, frame = src.read()
            if not ok:
                break
            idx += 1
            if idx % stride:
                continue
            last, _ = process_frame(detector, frame, conf, upload.name, idx)
            now = time.monotonic()
            if now - last_draw > 0.1:
                preview_ph.image(bgr_to_rgb(last), use_container_width=True)
                render_metrics(metric_ph)
                if info.frame_count:
                    bar.progress(min(idx / info.frame_count, 1.0))
                last_draw = now
        bar.empty()
        render_metrics(metric_ph)
        if last is not None:
            st.session_state.preview = encode_jpeg(last, max_width=1280)
        st.success(f"Finished: {idx} frames read from '{upload.name}'.")
    except InputValidationError as exc:
        st.error(str(exc))
    except Exception as exc:
        st.error(f"Video processing stopped after an unexpected error: {exc}")
    finally:
        if src is not None:
            src.release()
        safe_remove(path)


# ---- Live ------------------------------------------------------------------ #
def run_live() -> None:
    st.caption("A numeric source (0, 1, ...) opens a camera attached to the machine running "
               "this server, not your browser's camera. For remote cameras use an RTSP/HTTP URL.")
    source_text = st.text_input("Camera index or stream URL", "0",
                                placeholder="0  or  rtsp://user:pass@host:554/stream")
    max_minutes = st.number_input("Auto-stop after (minutes)", 1, 240, 30)
    c1, c2 = st.columns(2)
    if c1.button("Start", type="primary", disabled=not detector.ready, use_container_width=True):
        st.session_state.live_running = True
    if c2.button("Stop", use_container_width=True):
        st.session_state.live_running = False
    status_ph = st.empty()
    preview_ph = st.empty()
    if not st.session_state.live_running:
        if st.session_state.preview:
            preview_ph.image(st.session_state.preview, caption="Last processed frame",
                             use_container_width=True)
        return

    src = VideoSource(source_text)
    last: Optional[np.ndarray] = None
    try:
        status_ph.info("Connecting...")
        if not src.connect():
            status_ph.error(f"Could not connect to '{source_text}' ({src.last_status}). "
                            "Check the index/URL and try again.")
            st.session_state.live_running = False
            return
        started, idx, processed = time.monotonic(), 0, 0
        while st.session_state.live_running:
            if time.monotonic() - started > max_minutes * 60:
                status_ph.info("Auto-stopped after the configured time limit.")
                break
            ok, frame = src.read()
            if not ok:
                status_ph.error(f"Stream lost ({src.last_status}). Press Start to retry.")
                break
            idx += 1
            if idx % stride:
                continue
            last, _ = process_frame(detector, frame, conf, f"live:{source_text}", idx)
            processed += 1
            preview_ph.image(bgr_to_rgb(last), use_container_width=True)
            if processed % 10 == 0:
                render_metrics(metric_ph)
                status_ph.caption(f"LIVE: {src.last_status}, {processed} frames analysed")
        st.session_state.live_running = False  # only reached on normal exit
    except Exception as exc:
        status_ph.error(f"Live stream stopped after an unexpected error: {exc}")
        st.session_state.live_running = False
    finally:
        src.release()
        if last is not None:
            try:
                st.session_state.preview = encode_jpeg(last, max_width=1280)
            except Exception:
                pass
        render_metrics(metric_ph)


if source_mode == "Upload Images":
    run_images()
elif source_mode == "Upload Video":
    run_video()
else:
    run_live()


# --------------------------------------------------------------------------- #
# Alert log + snapshot gallery
# --------------------------------------------------------------------------- #
st.divider()
st.subheader("Detection log")
alog: AlertLogger = st.session_state.logger
df = alog.to_dataframe()
if df.empty:
    st.info("No detections yet. Upload a file or start a stream, then run detection.")
else:
    st.dataframe(df.tail(500).iloc[::-1], use_container_width=True, hide_index=True)
    st.download_button("Download CSV log", data=alog.to_csv_bytes(),
                       file_name=f"detections_{datetime.now():%Y%m%d_%H%M%S}.csv",
                       mime="text/csv")

st.subheader("High-risk snapshots")
snaps = alog.snapshots
if not snaps:
    st.caption("Snapshots of high-risk detections appear here (most recent 30).")
else:
    gcols = st.columns(3)
    for i, snap in enumerate(reversed(snaps)):
        with gcols[i % 3]:
            st.image(snap["jpeg"], caption=snap["caption"], use_container_width=True)
