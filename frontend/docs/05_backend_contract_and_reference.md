# VitalLens Frontend — Part 5: Backend Contract & Quick Reference

## Backend Communication

The frontend communicates with a **FastAPI** backend via two channels:

### 1. WebSocket — `ws://localhost:8000/ws/vitals`

Streams real-time predictions to the frontend. One message is sent per processed clip (~every 1 second at 30fps/clip).

**Message format (JSON):**

```json
{
  "hr":           72.4,
  "br":           15.2,
  "hrv":          38.1,
  "stress":       22.0,
  "bvp":          [0.1, 0.3, -0.2, 0.4, ...],
  "lighting":     "Good",
  "face_detected": true,
  "timestamp":    1713456789.123
}
```

| Field | Type | Description |
|---|---|---|
| `hr` | `float` | Heart rate in BPM |
| `br` | `float` | Breathing rate in breaths/min |
| `hrv` | `float` | RMSSD in milliseconds |
| `stress` | `float` | Stress index, 0–100 |
| `bvp` | `float[]` | Array of 32 (or 64) normalised BVP values for the waveform |
| `lighting` | `string` | One of `"Good"`, `"Mixed"`, `"Poor"` |
| `face_detected` | `boolean` | Whether a face is currently detected |
| `timestamp` | `float` | Unix timestamp (seconds) of the clip |

**Frontend handling:**
- Parsed in `useWebSocket.js` → `ws.onmessage`
- Passed directly to `applyReading(data)` in the Zustand store
- Any field can be `null`; the store uses `??` to keep previous values

### 2. REST — `POST http://localhost:8000/session/end`

Called when the user clicks "End Session". Returns a pre-computed session summary from the backend.

**Response format (JSON):**

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
  "bvp_series":  [0.1, 0.3, -0.2, ...],
  "hr_series":   [{ "hr": 72.4, "timestamp": 1713456789 }, ...],
  "timestamps":  [1713456789, 1713456790, ...]
}
```

| Field | Type | Description |
|---|---|---|
| `duration_seconds` | `int` | Session length in seconds |
| `avg_hr/br/hrv/stress` | `float` | Session averages |
| `min_hr` / `max_hr` | `float` | Heart rate range |
| `lighting_breakdown` | `object` | Fractions (0–1) summing to ~1 |
| `bvp_series` | `float[]` | Full concatenated BVP signal for the session |
| `hr_series` | `object[]` | HR + timestamp pairs |
| `timestamps` | `float[]` | Unix timestamps of each clip |

**Frontend handling:**
- If response is `ok`: `useVitalsStore.getState().setSummary(data)` → `navigate('/summary')`
- If network error or non-OK: `endSession()` computes summary from client-side `readings[]`

---

## Running the App Locally

```bash
cd C:\UTS\4\Deep Learning\Project\VitalLens\frontend
npm install
npm run dev
```

Vite starts the dev server on `http://localhost:5173` (default). Hot module replacement (HMR) is active.

Without the backend running, the app will automatically fall back to mock data after a 2-second timeout. All three pages are fully functional in mock mode.

**Build for production:**
```bash
npm run build    # runs tsc && vite build → outputs to /dist
npm run preview  # serves the built dist locally
```

---

## Quick Reference: All Files

### Pages

| File | Route | Key Dependencies |
|---|---|---|
| `pages/HomePage.jsx` | `/` | `useVitalsStore.startSession`, `useNavigate` |
| `pages/MonitorPage.jsx` | `/monitor` | `useWebSocket`, `useVitalsStore` (isActive, wsConnected, endSession), `WebcamCapture`, `VitalsPanel`, `BVPWaveform`, `LightingBanner` |
| `pages/SummaryPage.jsx` | `/summary` | `useVitalsStore` (summary, readings, resetSession), `SessionStats`, `BVPReplay`, `LightingBreakdown` |

### Components

| File | Props | Store Keys Used |
|---|---|---|
| `WebcamCapture.jsx` | `isRecording` | `vitals.lighting`, `vitals.faceDetected` |
| `VitalsPanel.jsx` | `wsConnected` | `vitals` (all fields) |
| `MetricCard.jsx` | `id, label, metricKey, value, prevValue, unit, icon, normalRange` | None (pure display) |
| `BVPWaveform.jsx` | None | `vitals.bvpWindow` |
| `LightingBanner.jsx` | None | `consecutivePoorFrames`, `lightingBannerDismissed`, `dismissLightingBanner` |
| `SessionStats.jsx` | `summary` | None (pure display) |
| `BVPReplay.jsx` | `bvpSeries` | None (pure display) |
| `LightingBreakdown.jsx` | `breakdown` | None (pure display) |

### Hooks & Store

| File | Purpose |
|---|---|
| `hooks/useWebSocket.js` | Opens WS to backend, falls back to mock, calls `applyReading` |
| `store/useVitalsStore.js` | All global state — session lifecycle, live vitals, UI flags |

### Config Files

| File | Purpose |
|---|---|
| `index.html` | HTML shell, Inter font, SEO meta |
| `src/index.css` | Global design system, `.glass-card`, `.btn-*`, `.metric-*`, keyframes |
| `tailwind.config.js` | Color tokens, animation names, font family |
| `postcss.config.js` | Wires Tailwind v4's PostCSS plugin |
| `package.json` | Dependencies + `dev`/`build`/`preview` scripts |

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| Zustand over Context/Redux | Minimal boilerplate; selector-based subscriptions prevent unnecessary re-renders; `.getState()` works outside React (needed in async handlers) |
| Mock fallback in `useWebSocket` | Entire frontend testable without backend; zero config required |
| `isAnimationActive={false}` on charts | Prevents stuttering when streaming real-time data to Recharts at 1s intervals |
| `void el.offsetWidth` reflow trick | Cheapest way to re-trigger CSS animations without removing/re-mounting DOM nodes |
| Client-side `endSession()` computation | Ensures the Summary page always works, even if the backend REST endpoint fails |
| Fixed SVG face bounding box | Real MediaPipe coordinates would replace these; mocked so the UI feature is visually complete while the backend integration is pending |
| `useNavigate` + button triggers only | No `<Link>` elements; session state must be set before navigating, which requires imperative code |
