"""Fault-tolerant YOLO inference engine and self-reconnecting video source."""
from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from utils import RISK_ORDER, Detection, classify_label

log = logging.getLogger("contraband.detector")

FALLBACK_WEIGHTS: Tuple[str, ...] = ("yolo11n.pt", "yolov8n.pt")
WEIGHTS_DIR = Path("weights")
DEFAULT_CUSTOM = WEIGHTS_DIR / "contraband.pt"
DEFAULT_LABEL = "Default fallback (yolo11n.pt, COCO)"


# --------------------------------------------------------------------------- #
# Device selection
# --------------------------------------------------------------------------- #
def select_device() -> str:
    """Return 'cuda:0', 'mps' or 'cpu', verifying the device really works."""
    try:
        import torch
    except Exception as exc:
        log.warning("PyTorch import failed (%s); using CPU.", exc)
        return "cpu"
    try:
        if torch.cuda.is_available():
            torch.zeros(1, device="cuda:0")
            return "cuda:0"
    except Exception as exc:
        log.warning("CUDA unusable (%s); trying next device.", exc)
    try:
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available() and mps.is_built():
            torch.zeros(1, device="mps")
            return "mps"
    except Exception as exc:
        log.warning("MPS unusable (%s); using CPU.", exc)
    return "cpu"


def list_available_weights() -> Dict[str, Optional[str]]:
    """Map dropdown labels to weight paths. ``None`` means the default fallback."""
    options: Dict[str, Optional[str]] = {}

    def add(path: Path) -> None:
        suffix = "" if path.is_file() else "  (missing: will fall back)"
        options.setdefault(f"{path.as_posix()}{suffix}", str(path))

    add(DEFAULT_CUSTOM)
    if WEIGHTS_DIR.is_dir():
        for p in sorted(WEIGHTS_DIR.glob("*.pt")):
            add(p)
    options[DEFAULT_LABEL] = None
    return options


# --------------------------------------------------------------------------- #
# Detector
# --------------------------------------------------------------------------- #
class Detector:
    """Loads YOLO weights with automatic fallback and never raises from detect().

    ``messages`` holds (level, text) tuples for the UI; level is one of
    'info', 'success', 'warning', 'error'.
    """

    def __init__(self, weights_path: Optional[str] = None,
                 device: Optional[str] = None, imgsz: int = 640) -> None:
        self.weights_path = weights_path
        self.imgsz = imgsz
        self.device = device or select_device()
        self.model = None
        self.model_name = "none"
        self.used_fallback = False
        self.messages: List[Tuple[str, str]] = []
        self._lock = threading.Lock()
        try:
            self._load()
        except Exception as exc:  # last line of defence: never fail construction
            self._note("error", f"Unexpected model-loading failure: {exc}")
            self.model = None

    # -- helpers ----------------------------------------------------------- #
    def _note(self, level: str, text: str) -> None:
        self.messages.append((level, text))
        getattr(log, "error" if level == "error" else "warning" if level == "warning" else "info")(text)

    @property
    def ready(self) -> bool:
        return self.model is not None

    @property
    def class_names(self) -> List[str]:
        names = getattr(self.model, "names", None) or {}
        if isinstance(names, dict):
            return [str(names[k]) for k in sorted(names)]
        return [str(n) for n in names]

    def _label(self, class_id: int) -> str:
        names = self.model.names
        try:
            return str(names[class_id])
        except (KeyError, IndexError):
            return str(class_id)

    # -- loading ----------------------------------------------------------- #
    def _load(self) -> None:
        if self.weights_path:
            if self._try_load(self.weights_path, custom=True):
                return
            self._note("warning", "Custom weights unavailable: switching to the default model. "
                                  "Default COCO weights only detect knives, scissors and bags "
                                  "(no firearms, ammunition or explosives).")
        self.used_fallback = True
        for name in FALLBACK_WEIGHTS:
            if self._try_load(name, custom=False):
                if not self.weights_path:
                    self._note("info", "Using default COCO model: detects knives, scissors and "
                                       "bags only. Add custom weights for firearms/contraband.")
                return
        self.used_fallback = False
        self._note("error", "No model could be loaded. The default weights must be downloaded "
                            "once (internet access required). Place a .pt file in ./weights "
                            "or connect to the internet, then click 'Reload model'.")

    def _try_load(self, ref: str, custom: bool) -> bool:
        try:
            if custom:
                path = Path(ref)
                if not path.is_file():
                    self._note("warning", f"Weights file not found: {ref}")
                    return False
                if path.stat().st_size < 1024:
                    self._note("warning", f"Weights file looks empty or truncated: {ref}")
                    return False
            from ultralytics import YOLO  # imported lazily so import errors are caught

            model = YOLO(ref)
            if not getattr(model, "names", None):
                raise ValueError("model has no class names")
            self._activate(model)
            self.model = model
            self.model_name = Path(ref).name
            self._note("success", f"Loaded {'custom ' if custom else 'default '}model "
                                  f"{self.model_name} on {self.device}.")
            return True
        except Exception as exc:
            self._note("warning", f"Could not load '{ref}': {type(exc).__name__}: {exc}")
            return False

    def _activate(self, model) -> None:
        """Warm up on the chosen device; drop to CPU if that device misbehaves."""
        blank = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        try:
            model.predict(blank, device=self.device, imgsz=self.imgsz, verbose=False)
        except Exception as exc:
            if self.device == "cpu":
                raise
            self._note("warning", f"Device {self.device} failed ({exc}); using CPU instead.")
            self.device = "cpu"
            model.predict(blank, device="cpu", imgsz=self.imgsz, verbose=False)

    # -- inference --------------------------------------------------------- #
    @staticmethod
    def _normalise(frame: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return None
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        if frame.ndim == 2:
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        if frame.ndim == 3 and frame.shape[2] == 4:
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        if frame.ndim == 3 and frame.shape[2] == 3:
            return frame
        return None

    def detect(self, frame: Optional[np.ndarray], conf: float) -> Tuple[List[Detection], float]:
        """Run inference. Returns (detections, milliseconds). Never raises."""
        frame = self._normalise(frame)
        if self.model is None or frame is None:
            return [], 0.0
        for attempt in range(2):
            try:
                with self._lock:
                    start = time.perf_counter()
                    results = self.model.predict(frame, conf=float(conf), imgsz=self.imgsz,
                                                 device=self.device, verbose=False)
                    elapsed = (time.perf_counter() - start) * 1000.0
                return self._parse(results, frame.shape), elapsed
            except Exception as exc:
                if "out of memory" in str(exc).lower():
                    self._free_gpu_memory()
                if attempt == 0 and self.device != "cpu":
                    self._note("warning", f"Inference failed on {self.device} "
                                          f"({type(exc).__name__}); retrying on CPU.")
                    self.device = "cpu"
                    continue
                self._note("error", f"Inference failed: {type(exc).__name__}: {exc}")
                return [], 0.0
        return [], 0.0

    @staticmethod
    def _free_gpu_memory() -> None:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _parse(self, results, shape) -> List[Detection]:
        if not results:
            return []
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)
        h, w = shape[:2]
        out: List[Detection] = []
        for (x1, y1, x2, y2), c, k in zip(xyxy, confs, classes):
            label = self._label(int(k))
            match = classify_label(label)
            if match is None:
                if self.used_fallback:
                    continue  # COCO classes like 'person' are noise for this use case
                category, risk = "Restricted Item", "MEDIUM"
            else:
                category, risk = match
            bbox = (int(max(0, min(w - 1, x1))), int(max(0, min(h - 1, y1))),
                    int(max(0, min(w - 1, x2))), int(max(0, min(h - 1, y2))))
            out.append(Detection(label=label, category=category, risk=risk,
                                 confidence=float(c), bbox=bbox))
        out.sort(key=lambda d: (RISK_ORDER.get(d.risk, 9), -d.confidence))
        return out


# --------------------------------------------------------------------------- #
# Video source with auto-reconnect
# --------------------------------------------------------------------------- #
class VideoSource:
    """cv2.VideoCapture wrapper for files, webcams (index) and RTSP/HTTP streams.

    Live sources (index or URL) reconnect with exponential backoff after
    repeated empty reads. Files simply end at EOF.
    """

    def __init__(self, source: Union[str, int], max_retries: int = 5,
                 base_delay: float = 1.0) -> None:
        text = str(source).strip()
        self.is_live = text.isdigit() or "://" in text
        self.source: Union[str, int] = int(text) if text.isdigit() else text
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.cap: Optional[cv2.VideoCapture] = None
        self.last_status = "idle"
        self.failed = False
        self._empty_reads = 0

    def open(self) -> bool:
        self.release()
        try:
            if isinstance(self.source, str) and "://" in self.source:
                os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
                cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
            else:
                cap = cv2.VideoCapture(self.source)
            if not cap.isOpened():
                cap.release()
                self.last_status = "could not open source"
                return False
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
            self.cap = cap
            self.last_status = "connected"
            return True
        except Exception as exc:
            self.last_status = f"open error: {exc}"
            return False

    def connect(self) -> bool:
        """Open once; for live sources keep retrying with backoff."""
        if self.open():
            return True
        return self._reconnect() if self.is_live else False

    def _reconnect(self) -> bool:
        for attempt in range(1, self.max_retries + 1):
            self.last_status = f"reconnecting ({attempt}/{self.max_retries})"
            time.sleep(min(self.base_delay * 2 ** (attempt - 1), 15.0))
            if self.open():
                self._empty_reads = 0
                return True
        self.failed = True
        self.last_status = "source lost"
        return False

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Return (ok, frame). ok=False means EOF (files) or permanent loss (live)."""
        while True:
            if self.cap is None:
                if not self.is_live or not self._reconnect():
                    return False, None
            try:
                ok, frame = self.cap.read()
            except Exception:
                ok, frame = False, None
            if ok and frame is not None and getattr(frame, "size", 0) > 0:
                self._empty_reads = 0
                return True, frame
            if not self.is_live:
                return False, None
            self._empty_reads += 1
            if self._empty_reads >= 3:
                self.release()
                if not self._reconnect():
                    return False, None
            else:
                time.sleep(0.05)

    def release(self) -> None:
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

    def __enter__(self) -> "VideoSource":
        return self

    def __exit__(self, *exc) -> None:
        self.release()
