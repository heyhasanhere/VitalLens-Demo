# VitalLens GUI Specification

## Overview

VitalLens is a real-time vitals monitoring app using a webcam. The user sits in front of their laptop, the app detects their face, measures their BVP signal via rPPG, and displays HR, breathing rate, HRV, and stress score — all without any physical contact.

---

## Pages / Views

### 1. Home / Landing Page

**Purpose:** Entry point. Explains what the app does and lets the user start a session.

**Elements:**
- App name + tagline: "VitalLens — Contactless Vitals Monitoring"
- Brief one-paragraph explanation of how it works (webcam → face → heart rate)
- "Start Session" button → goes to Monitor page
- Small disclaimer: "For research purposes only. Not a medical device."

---

### 2. Monitor Page (Main Page)

This is the core of the app. Everything happens here.

**Layout: Two-column**

#### Left column — Live Webcam Feed
- Live webcam stream (react-webcam)
- MediaPipe face bounding box overlaid on the feed (green rectangle around detected face)
- Lighting quality badge overlaid top-right of the video:
  - 🟢 Good / 🟡 Mixed / 🔴 Poor
  - Comes from the lighting classifier
- "Recording" indicator (pulsing red dot) when session is active
- Face detection status: "Face Detected" / "No Face — please centre yourself"

#### Right column — Vitals Panel

Four metric cards, updated in real time:

| Metric | Unit | Normal Range | Source |
|--------|------|-------------|--------|
| Heart Rate | BPM | 60–100 | EfficientPhys → bvp_to_hr |
| Breathing Rate | breaths/min | 12–20 | bvp_to_br |
| HRV (RMSSD) | ms | 20–50ms | bvp_to_hrv |
| Stress Index | 0–100 | <30 = low | stress_index(hrv) |

Each card shows:
- Metric name
- Current value (large, bold)
- Small trend indicator (↑ ↓ →) compared to last reading
- Colour coding: green = normal, amber = borderline, red = out of range

#### Bottom — BVP Waveform
- Full-width scrolling line chart (Recharts)
- Shows last 10 seconds of BVP signal
- X axis: time (seconds), Y axis: normalised amplitude
- Updates every ~1 second as new clips are processed

---

### 3. Lighting Warning Banner (conditional)

If lighting classifier returns "Poor" for 3+ consecutive frames:
- Yellow banner appears at top of Monitor page
- Text: "Poor lighting detected — results may be inaccurate. Try facing a window or turning on a light."
- Dismiss button

---

### 4. Session Summary Page

Shown when user clicks "End Session".

**Elements:**
- Session duration
- Average HR, BR, HRV, Stress over the session
- Min/Max HR during session
- BVP waveform replay (static chart of full session)
- Lighting quality breakdown: % of time Good / Mixed / Poor
- "Export CSV" button (downloads session data as CSV)
- "New Session" button → back to Monitor page

---

## Backend Contract (What the Frontend Expects)

The frontend communicates with FastAPI via:
1. **WebSocket** `/ws/vitals` — streams real-time predictions
2. **REST** `POST /session/end` — returns session summary

### WebSocket Message (backend → frontend)

Sent every ~1 second (once per 30-frame clip processed):

```json
{
  "hr":           72.4,
  "br":           15.2,
  "hrv":          38.1,
  "stress":       22.0,
  "bvp":          [0.1, 0.3, -0.2, ...],
  "lighting":     "Good",
  "face_detected": true,
  "timestamp":    1713456789.123
}
```

Field definitions:
- `hr` — heart rate in BPM (float)
- `br` — breathing rate in breaths/min (float)
- `hrv` — RMSSD in milliseconds (float)
- `stress` — stress index 0–100 (float)
- `bvp` — array of 32 (or 64) normalised BVP values for waveform display
- `lighting` — one of `"Good"`, `"Mixed"`, `"Poor"`
- `face_detected` — boolean
- `timestamp` — Unix timestamp of the clip

### REST Session Summary (backend → frontend)

`POST /session/end` returns:

```json
{
  "duration_seconds": 120,
  "avg_hr":           74.1,
  "avg_br":           14.8,
  "avg_hrv":          35.2,
  "avg_stress":       25.0,
  "min_hr":           68.0,
  "max_hr":           82.0,
  "lighting_breakdown": {
    "Good":  0.72,
    "Mixed": 0.20,
    "Poor":  0.08
  },
  "bvp_series":  [...],
  "hr_series":   [...],
  "timestamps":  [...]
}
```

---

## Frontend State (Zustand Store)

```
session: {
  isActive: bool,
  startTime: timestamp,
  readings: [ { hr, br, hrv, stress, bvp, lighting, timestamp }, ... ]
}

vitals: {
  hr, br, hrv, stress     ← latest values
  bvpWindow: float[]      ← last 10s of BVP for chart
  lighting: string
  faceDetected: bool
}
```

---

## Tech Stack

- React 18 + Vite
- Tailwind CSS — styling
- Recharts — BVP waveform and session summary charts
- Zustand — global state
- react-webcam — webcam capture
- Native WebSocket API — real-time backend connection

---

## Component Tree

```
App
├── HomePage
└── MonitorPage
    ├── WebcamCapture        ← webcam feed + face box overlay + lighting badge
    ├── LightingBanner       ← conditional warning banner
    ├── VitalsPanel
    │   ├── MetricCard (HR)
    │   ├── MetricCard (BR)
    │   ├── MetricCard (HRV)
    │   └── MetricCard (Stress)
    └── BVPWaveform          ← scrolling real-time chart
└── SummaryPage
    ├── SessionStats
    ├── BVPReplay
    └── LightingBreakdown
```

---

## Mock Data (for frontend dev without backend)

While the backend is in progress, use this mock WebSocket message on a 1-second interval:

```js
const mockReading = () => ({
  hr:           60 + Math.random() * 20,
  br:           12 + Math.random() * 8,
  hrv:          20 + Math.random() * 30,
  stress:       Math.random() * 40,
  bvp:          Array.from({ length: 32 }, () => Math.sin(Math.random() * Math.PI)),
  lighting:     ["Good", "Mixed", "Poor"][Math.floor(Math.random() * 3)],
  face_detected: true,
  timestamp:    Date.now() / 1000,
})
```

This lets the entire frontend be built and tested independently before the backend exists.
