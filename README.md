# Restricted Item Detection (YOLO + OpenCV + Streamlit)

Browser-based detector for firearms, ammunition, knives, explosives / suspicious
packages and contraband. Supports image batches, video files and live
webcam/RTSP streams, with a CSV log and a gallery of high-risk snapshots.

> **Decision-support only.** Detections are probabilistic. Every alert needs
> human review, and the app should not be the sole control in a real security
> process.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501. Python 3.9 to 3.12 is recommended.

- **GPU:** `pip install` pulls a default PyTorch build. For a specific CUDA build,
  install PyTorch first from https://pytorch.org/get-started/locally/. CUDA, then
  Apple MPS, then CPU are tried automatically.
- **Large uploads:** `streamlit run app.py --server.maxUploadSize 1000` (MB).
- The first run needs internet once to download `yolo11n.pt` (or `yolov8n.pt`).

## Important: what the default model can and cannot do

The fallback weights are COCO-trained. They can detect **knives, scissors,
backpacks, suitcases and handbags** (the last three are flagged as low-risk
"suspicious package" proxies). They **cannot detect firearms, ammunition,
explosives or drugs**. For those you must supply custom weights trained on a
weapons / contraband dataset.

## Using custom weights

1. Put your file at `weights/contraband.pt` (or any `weights/*.pt`).
2. Pick it in the sidebar. Class names are mapped to risk levels by keyword
   (`gun`, `pistol`, `rifle`, `ammo`, `bullet`, `grenade`, `bomb`, `knife`, ...);
   see `CATEGORY_RULES` in `utils.py` to adapt them to your labels. Unmapped
   custom classes are reported as "Restricted Item" (medium risk).
3. If the file is missing, empty or corrupted, the app shows a warning and
   loads the default model instead.

Train your own:

```bash
yolo detect train data=your_dataset/data.yaml model=yolo11s.pt epochs=100 imgsz=640
# result: runs/detect/train/weights/best.pt  ->  copy to weights/contraband.pt
```

Only load `.pt` files you trust: PyTorch weights can execute code when loaded.

## Datasets (check each licence before use)

- Roboflow Universe, weapons: https://universe.roboflow.com/search?q=gun+detection
- Kaggle, weapons: https://www.kaggle.com/search?q=weapon+detection+yolo
- X-ray baggage: SIXray https://github.com/MeioJane/SIXray,
  PIDray https://github.com/bywang2018/security-dataset,
  OPIXray https://github.com/OPIXray-author/OPIXray,
  HiXray https://github.com/DIG-Beihang/XrayDetection

## Project layout

| File | Role |
|------|------|
| `app.py` | Streamlit UI, session state, metrics, log and gallery |
| `detector.py` | Model loading with fallback, device selection, inference, auto-reconnecting `VideoSource` |
| `utils.py` | Upload validation, risk mapping, annotation, alert logger |

## Reliability behaviour

- Missing/corrupt custom weights: warning shown, default model loaded.
- No weights obtainable (offline, first run): app still starts; detection is
  disabled with instructions rather than crashing.
- Uploads are validated (decode check, size, minimum dimensions, first video
  frame); bad files are skipped with a message.
- CUDA/MPS errors or out-of-memory: retried on CPU.
- RTSP/webcam: reconnects with exponential backoff (5 attempts) before giving up.
- Model cached with `@st.cache_resource`; capture handles are released and
  temp video files deleted in `finally` blocks.

No software can promise literally 100% uptime. This design removes the common
crash paths; for production, also run under a process supervisor (systemd,
Docker `restart: always`).

## Notes

- Webcam index `0` is a camera on the **server** machine. For remote use, use an
  RTSP/HTTP URL.
- Clicking any sidebar control during a live stream restarts the loop with the
  new settings; use **Stop** to end it.
