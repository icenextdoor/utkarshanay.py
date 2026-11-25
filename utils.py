"""Validation, risk classification, frame annotation and alert logging helpers.

Nothing in this module depends on Streamlit or Ultralytics, so it can be unit
tested on its own.
"""
from __future__ import annotations

import io
import logging
import math
import os
import re
import shutil
import tempfile
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import BinaryIO, Deque, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

log = logging.getLogger("contraband.utils")

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
IMAGE_EXTENSIONS: Tuple[str, ...] = ("jpg", "jpeg", "png", "webp")
VIDEO_EXTENSIONS: Tuple[str, ...] = ("mp4", "avi")

MAX_IMAGE_BYTES = 25 * 1024 * 1024
MAX_VIDEO_BYTES = 1024 * 1024 * 1024
MAX_IMAGE_PIXELS = 60_000_000
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS

RISK_ORDER: Dict[str, int] = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
# BGR colours for OpenCV drawing.
RISK_COLORS: Dict[str, Tuple[int, int, int]] = {
    "HIGH": (0, 0, 220),
    "MEDIUM": (0, 140, 255),
    "LOW": (0, 200, 230),
}

# (category, risk, keywords). First matching rule wins, so order matters.
# Keywords of 5+ characters match as substrings of a label token; shorter
# keywords must match a whole token (avoids "ied" matching "tied").
CATEGORY_RULES: List[Tuple[str, str, Tuple[str, ...]]] = [
    ("Firearm", "HIGH", (
        "gun", "pistol", "handgun", "rifle", "firearm", "revolver", "shotgun",
        "weapon", "smg", "glock", "ak47", "ar15",
    )),
    ("Ammunition", "HIGH", (
        "ammo", "ammunition", "bullet", "cartridge", "magazine", "shell",
    )),
    ("Explosive", "HIGH", (
        "bomb", "explosive", "grenade", "dynamite", "ied", "detonator", "c4",
        "tnt",
    )),
    ("Knife", "HIGH", (
        "knife", "knives", "dagger", "machete", "blade", "dirk", "sword",
    )),
    ("Sharp Object", "MEDIUM", ("scissors", "razor", "cutter")),
    ("Contraband", "MEDIUM", (
        "contraband", "drug", "narcotic", "pill", "syringe", "cocaine",
    )),
    ("Suspicious Package", "LOW", (
        "suspicious", "package", "parcel", "backpack", "suitcase", "handbag",
        "luggage",
    )),
]

_TOKEN_SPLIT = re.compile(r"[^a-z0-9]+")


class InputValidationError(Exception):
    """Raised when an uploaded file or stream cannot be used. Message is user-safe."""


@dataclass(frozen=True)
class Detection:
    label: str
    category: str
    risk: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2 in pixels


def classify_label(label: str) -> Optional[Tuple[str, str]]:
    """Map a model class name to (category, risk). Returns None if unmapped."""
    tokens = [t for t in _TOKEN_SPLIT.split(str(label).lower()) if t]
    if not tokens:
        return None
    for category, risk, keywords in CATEGORY_RULES:
        for kw in keywords:
            for tok in tokens:
                if tok == kw or (len(kw) >= 5 and kw in tok):
                    return category, risk
    return None


# --------------------------------------------------------------------------- #
# Input validation
# --------------------------------------------------------------------------- #
def load_image_bytes(data: bytes, name: str = "image") -> np.ndarray:
    """Decode and validate image bytes. Returns a BGR uint8 array."""
    if not data:
        raise InputValidationError(f"'{name}' is empty.")
    if len(data) > MAX_IMAGE_BYTES:
        raise InputValidationError(
            f"'{name}' is larger than {MAX_IMAGE_BYTES // (1024 * 1024)} MB."
        )
    try:
        with Image.open(io.BytesIO(data)) as probe:
            probe.verify()  # detects many forms of corruption cheaply
        with Image.open(io.BytesIO(data)) as img:
            img = ImageOps.exif_transpose(img)
            arr = np.array(img.convert("RGB"))
    except Exception as exc:  # PIL raises many unrelated exception types
        raise InputValidationError(
            f"'{name}' is corrupted or not a supported image ({type(exc).__name__})."
        ) from exc
    if arr.ndim != 3 or arr.size == 0 or min(arr.shape[:2]) < 16:
        raise InputValidationError(f"'{name}' is empty or too small (min 16x16 px).")
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def safe_remove(path: Optional[str]) -> None:
    """Delete a temp file, never raising."""
    if not path:
        return
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError as exc:
        log.warning("Could not remove temp file %s: %s", path, exc)


def save_upload_to_temp(file_obj: BinaryIO, filename: str) -> str:
    """Stream an uploaded video to a temp file in 1 MB chunks. Returns the path."""
    ext = os.path.splitext(filename)[1].lower().lstrip(".")
    if ext not in VIDEO_EXTENSIONS:
        raise InputValidationError(
            f"Unsupported video type '.{ext}'. Use: {', '.join(VIDEO_EXTENSIONS)}."
        )
    fd, path = tempfile.mkstemp(prefix="upload_", suffix=f".{ext}")
    try:
        with os.fdopen(fd, "wb") as out:
            try:
                file_obj.seek(0)
            except Exception:
                pass
            shutil.copyfileobj(file_obj, out, length=1024 * 1024)
        size = os.path.getsize(path)
        if size == 0:
            raise InputValidationError(f"'{filename}' is empty.")
        if size > MAX_VIDEO_BYTES:
            raise InputValidationError(
                f"'{filename}' exceeds {MAX_VIDEO_BYTES // (1024 * 1024)} MB."
            )
    except BaseException:
        safe_remove(path)
        raise
    return path


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    frame_count: int


def probe_video(path: str) -> VideoInfo:
    """Open a video and decode its first frame to prove it is readable."""
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            raise InputValidationError("Video could not be opened (unsupported codec or corrupted).")
        ok, frame = cap.read()
        if not ok or frame is None or frame.size == 0:
            raise InputValidationError("Video has no readable frames.")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if not math.isfinite(fps) or fps <= 0:
            fps = 25.0
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        h, w = frame.shape[:2]
        return VideoInfo(width=w, height=h, fps=fps, frame_count=max(count, 0))
    finally:
        cap.release()


# --------------------------------------------------------------------------- #
# Annotation / encoding
# --------------------------------------------------------------------------- #
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def annotate_frame(frame: np.ndarray, detections: Iterable[Detection]) -> np.ndarray:
    """Return a copy of ``frame`` with boxes and labels drawn, coloured by risk."""
    out = frame.copy()
    dets = list(detections)
    h, w = out.shape[:2]
    scale = max(0.5, min(w, h) / 900.0)
    thickness = max(1, int(round(2 * scale)))
    for d in dets:
        color = RISK_COLORS.get(d.risk, (255, 255, 255))
        x1, y1, x2, y2 = d.bbox
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        text = f"{d.label} {d.confidence:.0%}"
        (tw, th), base = cv2.getTextSize(text, _FONT, 0.5 * scale, 1)
        top = max(0, y1 - th - base - 4)
        cv2.rectangle(out, (x1, top), (x1 + tw + 6, top + th + base + 4), color, -1)
        cv2.putText(out, text, (x1 + 3, top + th + 2), _FONT, 0.5 * scale,
                    (255, 255, 255), 1, cv2.LINE_AA)
    high = sum(1 for d in dets if d.risk == "HIGH")
    if high:
        banner = f"ALERT: {high} high-risk item{'s' if high != 1 else ''}"
        (tw, th), base = cv2.getTextSize(banner, _FONT, 0.7 * scale, 2)
        cv2.rectangle(out, (0, 0), (tw + 16, th + base + 12), RISK_COLORS["HIGH"], -1)
        cv2.putText(out, banner, (8, th + 6), _FONT, 0.7 * scale,
                    (255, 255, 255), 2, cv2.LINE_AA)
    return out


def encode_jpeg(frame: np.ndarray, max_width: int = 960, quality: int = 85) -> bytes:
    """Encode a BGR frame to JPEG bytes, downscaling wide frames to save memory."""
    h, w = frame.shape[:2]
    if w > max_width:
        frame = cv2.resize(frame, (max_width, max(1, int(h * max_width / w))),
                           interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("JPEG encoding failed")
    return buf.tobytes()


def bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


# --------------------------------------------------------------------------- #
# Alert logging
# --------------------------------------------------------------------------- #
class AlertLogger:
    """Bounded in-memory detection log plus a gallery of high-risk snapshots."""

    COLUMNS = ["timestamp", "source", "frame", "label", "category", "risk",
               "confidence", "x1", "y1", "x2", "y2"]

    def __init__(self, max_records: int = 50_000, max_snapshots: int = 30,
                 snapshot_interval_s: float = 1.0) -> None:
        self._records: Deque[dict] = deque(maxlen=max_records)
        self._snapshots: Deque[dict] = deque(maxlen=max_snapshots)
        self._interval = snapshot_interval_s
        self._last_snap: Dict[str, float] = {}

    def log(self, source: str, frame_index: int, detections: List[Detection],
            annotated: Optional[np.ndarray] = None) -> None:
        if not detections:
            return
        now = datetime.now().isoformat(timespec="seconds")
        for d in detections:
            x1, y1, x2, y2 = d.bbox
            self._records.append({
                "timestamp": now, "source": source, "frame": frame_index,
                "label": d.label, "category": d.category, "risk": d.risk,
                "confidence": round(float(d.confidence), 4),
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            })
        high = [d for d in detections if d.risk == "HIGH"]
        if high and annotated is not None:
            mono = time.monotonic()
            if mono - self._last_snap.get(source, -1e9) >= self._interval:
                try:
                    jpeg = encode_jpeg(annotated)
                except Exception as exc:
                    log.warning("Snapshot encoding failed: %s", exc)
                    return
                self._last_snap[source] = mono
                top = max(high, key=lambda d: d.confidence)
                self._snapshots.append({
                    "jpeg": jpeg,
                    "caption": f"{now} | {source} | frame {frame_index} | "
                               f"{top.category}: {top.label} ({top.confidence:.0%})",
                })

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(list(self._records), columns=self.COLUMNS)

    def to_csv_bytes(self) -> bytes:
        return self.to_dataframe().to_csv(index=False).encode("utf-8")

    @property
    def snapshots(self) -> List[dict]:
        return list(self._snapshots)

    def __len__(self) -> int:
        return len(self._records)
