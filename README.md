# VitalLens

Real-time vital signs monitor using a standard webcam. No contact sensors required.

The app uses **remote photoplethysmography (rPPG)** — it detects the subtle per-frame colour changes in skin caused by blood volume changes in the face, and extracts cardiovascular signals from that. Face landmark detection uses MediaPipe.

The backend is a FastAPI server that runs inference in a background thread and streams results over a WebSocket at ~1 Hz. The frontend is a React/Vite single-page app.

---

## Requirements

- Python 3.10+
- Node.js 18+
- Webcam (USB or built-in)
- Reasonably even lighting on your face

---

## Installation

### 1. Clone

```bash
git clone <repo-url>
cd VitalLens
```

### 2. Backend

Create and activate a virtual environment, then install:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Frontend

```bash
cd frontend
npm install
```

---

## Configuration

Copy the backend environment file and adjust as needed:

```bash
cp .env.example .env   # or edit .env directly
```

Key variables in `.env`:

| Variable | Default | Description |
|---|---|---|
| `RPPG_MODEL_PATH` | `weights/factorizephys.onnx` | Path to the rPPG ONNX model. Swap to `weights/vitallens_rppg.onnx` to use the EfficientPhys model instead. |
| `LIGHTING_MODEL_PATH` | `weights/vitallens_lighting.onnx` | Lighting classifier ONNX. Optional — falls back to "Good" if missing. |
| `FACE_LANDMARKER_PATH` | `weights/face_landmarker.task` | MediaPipe face landmarker task file. |
| `CAMERA_INDEX` | `0` | Webcam device index. Call `GET /cameras` to list available indices. |
| `CORS_ORIGINS` | `*` | Comma-separated allowed origins for the frontend. |
| `OPENAI_API_KEY` | — | Optional. Enables age-based HR zone display (see [HR Zones](#hr-zones)). |

The frontend reads from `frontend/.env`:

```
VITE_API_BASE=http://localhost:8000
VITE_WS_BASE=ws://localhost:8000
```

Update these if the backend runs on a different host or port.

Any change to `.env` requires a backend restart to take effect.

---

## Running

**Backend** (from repo root, venv active):

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

**Frontend** — development server:

```bash
cd frontend
npm run dev
```

Or build and serve statically:

```bash
cd frontend
npm run build
# then serve dist/ with any static file server
```

Open `http://localhost:5173` (dev) or wherever you serve the build.

---

## Reading the Dashboard

The monitor page has two columns: the camera feed and BVP waveform on the left, and the vitals panel and HR trend chart on the right.

### Warm-up period

The first reading takes approximately **30 seconds** to appear. During this time the metric cards show "—" and the BVP waveform shows "Calibrating…". This is normal — the model needs a full 160-frame clip (~5 s at 30 fps), and several clips are averaged before the first number is displayed.

### Metric cards

| Metric | Unit | Normal range | What it measures |
|---|---|---|---|
| **Heart Rate** | BPM | 60–100 | Cardiac cycle frequency, extracted from the BVP signal via Welch PSD peak detection. |
| **Breathing Rate** | br/min | 12–20 | Respiration-modulated amplitude of the BVP signal, bandpass-filtered at 0.15–0.4 Hz. |
| **HRV (RMSSD)** | ms | 20–50 | Root-mean-square of successive differences between inter-beat intervals. Higher = more heart-rate variability = generally healthier autonomic tone. Requires ~30 s of signal to stabilise. |
| **Stress Index** | 0–100 | < 30 = low | Derived directly from HRV RMSSD: `(60 - RMSSD) / 55 × 100`, clamped to [0, 100]. A score near 0 means low physiological stress; near 100 means high. This is a heuristic index, not a clinical measurement. |
| **Blink Rate** | blinks/min | 8–25 | Eye blinks per minute, counted from MediaPipe eye aspect ratio (EAR). Low blink rate can indicate intense focus or eye strain; very high rate can indicate irritation. |

### BVP waveform

The Blood Volume Pulse waveform shows the raw rPPG signal extracted by the model — the same periodic signal from which HR, HRV, and BR are derived. Each peak corresponds to one heartbeat. A clean, rhythmically spaced waveform indicates good signal quality. An irregular or flat waveform means the model is struggling, usually due to motion or poor lighting.

### HR trend chart

Shows the last two minutes of heart rate readings. Dashed reference lines mark 60 and 100 BPM (the normal resting range). The x-axis shows time elapsed since the session started.

### Heart rhythm

Displayed in the debug panel. Derived from the coefficient of variation (CV) of inter-beat intervals:
- **Regular** — CV ≤ 0.2, normal beat-to-beat variation.
- **Irregular** — CV > 0.2, unusually variable intervals. This may indicate arrhythmia but is **not a medical diagnosis**. Motion artefacts and low signal quality can also trigger this flag.

### Lighting indicator

A classifier labels each frame as **Good**, **Mixed**, or **Poor**. The majority label over the last several seconds is shown. Poor lighting degrades signal quality; if you see "Low signal" consistently, try improving ambient lighting or adding a front-facing light source. The lighting classifier requires `vitallens_lighting.onnx` — if not present, the label always shows "Good".

### HR Zones

If an `OPENAI_API_KEY` is set, the app calls GPT-4o-mini once per session to estimate your age from the first face crop, then computes personalised HR zones using `max HR = 220 − age`:

| Zone | % of max HR | Label |
|---|---|---|
| Z1 | 50–60% | Warm-up |
| Z2 | 60–70% | Fat Burn |
| Z3 | 70–80% | Aerobic |
| Z4 | 80–90% | Anaerobic |
| Z5 | 90–100% | Max |

The current zone badge appears in the header. If no API key is set, zones are not shown.

### Signal quality badge

A **"Low signal"** badge appears in the vitals panel header when the BVP signal-to-noise ratio (peak/median power in the cardiac band) drops below 3.5. When this is shown, the displayed readings are being held from the last reliable inference rather than updated live. Causes include: significant head movement, very low or uneven lighting, face partially out of frame, or partial occlusion (e.g. hand near face).

---

## Session summary

Click **End Session** to finish. The summary page shows:

- Average, min, and max HR; average BR, HRV, and Stress over the session
- A BVP signal replay (concatenated signal from the whole session)
- Lighting quality breakdown (fraction of time in Good / Mixed / Poor)
- **Export CSV** — downloads a per-second record of all readings plus the summary block

---

## Tips for best results

- Sit with your face evenly lit from the front. Avoid strong backlighting (e.g. window behind you).
- Minimise head movement — the signal is gated whenever head sway exceeds ~3% of frame width.
- The camera should be at roughly eye level, about 40–80 cm away.
- Glasses are supported but may slightly reduce EAR-based blink detection accuracy.
- Two rPPG models are bundled in `weights/`. The default (`factorizephys.onnx`) performs better in varied lighting. Switch by setting `RPPG_MODEL_PATH` in `.env`.

---

## Repository layout

```
VitalLens/
│
├── backend/                  The FastAPI server. inference.py runs the rPPG +
│   ├── main.py               lighting models in a background thread and streams
│   └── inference.py          results over WebSocket at ~1 Hz. main.py defines
│                             the REST and WebSocket routes.
│
├── frontend/                 React/Vite single-page app. Connects to the backend
│   └── src/                  WebSocket, renders live metric cards, BVP waveform,
│       ├── pages/            HR trend chart, and session summary.
│       ├── components/
│       ├── hooks/
│       └── store/
│
├── weights/                  ONNX models used at runtime by the app.
│   ├── factorizephys.onnx    Default rPPG model (FactorizePhys FSAM).
│   ├── vitallens_rppg.onnx   Alternate rPPG model (EfficientPhys/VitalLens).
│   └── face_landmarker.task  MediaPipe face landmark detector.
│
├── research/                 Everything training-related. Not needed to run the app.
│   ├── factorizephys/        Code that produced factorizephys.onnx.
│   │   ├── src/              Training scripts (phase1/2/3), dataset loaders, cache builder.
│   │   ├── notebooks/        Experiment notebooks (training runs, evals, analysis).
│   │   ├── export_onnx.py    Exports a trained .pth checkpoint to ONNX.
│   │   ├── requirements.txt  FactorizePhys-specific training deps.
│   │   └── external/         git submodule — FactorizePhys source (model architecture).
│   │       └── FactorizePhys/
│   │
│   ├── lighting/             Code that produced vitallens_lighting.onnx.
│   │   ├── model.py          MobileNetV3-based lighting classifier.
│   │   ├── dataset.py        Dataset builder from MMPD / UBFC frames.
│   │   ├── train.py          Training script.
│   │   ├── export_onnx.py    Exports trained model to ONNX.
│   │   ├── requirements.txt  Lighting training deps.
│   │   └── data/             lighting_labels_synthetic.csv (training labels).
│   │
│   ├── preprocessing/        Signal processing utilities used during training data prep.
│   ├── notebooks/            Misc inspection/setup scripts (dataset inspector, SageMaker).
│   ├── evaluate_ubfc.py      Benchmarks an rPPG ONNX against the UBFC-rPPG dataset.
│   ├── analyze_recording.py  Offline per-timestamp analysis of a saved video.
│   └── test_e2e.py           End-to-end inference test (webcam, video, or .npy clips).
│
├── requirements.txt          App-only backend deps (no PyTorch, no training libs).
├── .env.example              Template for backend configuration.
├── setup.sh                  Installs app deps via requirements.txt.
└── README.md
```

---

## Retraining the rPPG model

The rPPG model in `weights/factorizephys.onnx` was trained using [FactorizePhys](https://github.com/PhysiologicAILab/FactorizePhys) (FSAM variant). Training code and notebooks are in `research/factorizephys/`. The FactorizePhys source is included as a git submodule at `research/factorizephys/external/FactorizePhys` — initialise it before running any training script:

```bash
git submodule update --init --recursive
```

Install training dependencies:

```bash
pip install -r research/factorizephys/requirements.txt
```

Training phases run in order:

```bash
python research/factorizephys/src/train_phase1.py   # SCAMPS pre-training
python research/factorizephys/src/train_phase2.py   # multi-dataset fine-tune
python research/factorizephys/src/train_phase3.py   # temporal consistency (Our research found that phase3 / TEMPORAL CONSISTENCY does not help with better generalisation / performance on cross-datasets)
```

Checkpoints are written to `research/factorizephys/checkpoints/`. To export a checkpoint to ONNX:

```bash
python research/factorizephys/export_onnx.py \
  --checkpoint research/factorizephys/checkpoints/phase2/best.pth \
  --output weights/factorizephys.onnx
```

Training notebooks with full experiment logs are in `research/factorizephys/notebooks/`.

---

## Disclaimer

VitalLens is a research and demonstration tool. The vitals it produces are estimates derived from a consumer webcam under uncontrolled conditions. They are **not medically validated** and should not be used for clinical decisions, diagnosis, or medical monitoring.
