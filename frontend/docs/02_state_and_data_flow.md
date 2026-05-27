# VitalLens Frontend — Part 2: State Management & Data Flow

## State Architecture Overview

The entire frontend state lives in a single **Zustand store** (`src/store/useVitalsStore.js`). There is no React Context, Redux, or prop-drilling. Any component that needs state imports the hook directly and subscribes to only the slice it needs, which prevents unnecessary re-renders.

```
WebSocket / Mock Timer
        │
        ▼
   useWebSocket (hook)
        │ calls applyReading(reading)
        ▼
  useVitalsStore (Zustand)
        │
        ├── vitals (live display)
        ├── session (recordings + lifecycle)
        └── UI flags (wsConnected, lighting banner, etc.)
        │
        ▼
  React Components (subscribe via selectors)
```

---

## The Zustand Store (`useVitalsStore.js`)

### Full State Shape

```js
{
  // ── Session lifecycle ──────────────────────────────────────────────────────
  session: {
    isActive:  boolean,       // true while a session is running
    startTime: number | null, // Date.now() at session start
    readings:  Reading[],     // array of every received WebSocket message
    summary:   Summary | null // computed or received from REST on end
  },

  // ── Live vitals (updated every ~1s from WebSocket) ─────────────────────────
  vitals: {
    hr:           number | null,  // heart rate BPM
    br:           number | null,  // breathing rate br/min
    hrv:          number | null,  // RMSSD in ms
    stress:       number | null,  // stress index 0–100
    bvpWindow:    number[],       // last 320 BVP samples (~10s at 32/s)
    lighting:     string,         // 'Good' | 'Mixed' | 'Poor'
    faceDetected: boolean,
    prevHr:       number | null,  // previous reading (for trend arrows)
    prevBr:       number | null,
    prevHrv:      number | null,
    prevStress:   number | null,
  },

  // ── UI flags ───────────────────────────────────────────────────────────────
  lightingBannerDismissed: boolean,
  consecutivePoorFrames:   number,
  wsConnected:             boolean,
}
```

A **`Reading`** object (one per WebSocket message, stored in `session.readings`) has the shape:
```js
{ hr, br, hrv, stress, bvp, lighting, face_detected, timestamp }
```

---

### Actions

#### `startSession()`

Resets `session` to a clean slate, records `startTime: Date.now()`, sets `isActive: true`. Also resets `lightingBannerDismissed` and `consecutivePoorFrames`.

Called by:
- `HomePage` before navigating to `/monitor`
- `MonitorPage` as a guard if the page is navigated to directly (URL typing)
- `SummaryPage` `handleNewSession` flow

#### `endSession()`

Computes the session summary **client-side** from accumulated `readings`. This is the offline fallback when the backend REST `POST /session/end` is unavailable.

The computation:
1. Calculates `avg_hr`, `avg_br`, `avg_hrv`, `avg_stress` using a generic `avg(arr, key)` helper that filters out nulls
2. Finds `min_hr` / `max_hr` from the HR readings array
3. Computes `lighting_breakdown` as fractional percentages (e.g. `{ Good: 0.8, Mixed: 0.1, Poor: 0.1 }`)
4. Flattens all `bvp` arrays from every reading into `bvp_series` for the full-session waveform
5. Extracts `hr_series` (HR + timestamp pairs) and `timestamps`
6. Sets `session.isActive = false` and attaches the computed `summary`

#### `setSummary(summaryData)`

Overrides the summary with data received from the backend REST endpoint. Sets `isActive = false`. Called only from `MonitorPage.handleEndSession` when the `POST /session/end` call succeeds.

#### `applyReading(reading)` — The Core Data Pipeline

Called on every incoming WebSocket message (real or mock). This is the most important action:

```
Incoming reading: { hr, br, hrv, stress, bvp, lighting, face_detected, timestamp }
```

**What it does:**

1. **Updates `vitals`:**
   - Uses nullish coalescing (`??`) to keep previous values if new ones are null — prevents the display from blanking on partial readings
   - Saves previous values into `prevHr`, `prevBr`, `prevHrv`, `prevStress` (used for trend arrows in `MetricCard`)
   - Appends new BVP samples to `bvpWindow`, keeping only the last **320 samples** (`slice(-320)`)

2. **Updates `consecutivePoorFrames`:**
   - Increments if `lighting === 'Poor'`, resets to 0 otherwise
   - The `LightingBanner` watches this and appears after 3 consecutive poor frames

3. **Appends to `session.readings`:**
   - Only if `session.isActive === true`
   - The full raw reading object is pushed to the array for later summary computation

#### `dismissLightingBanner()`

Sets `lightingBannerDismissed = true`. The banner will not reappear for the remainder of the session.

#### `setWsConnected(val)`

Sets the `wsConnected` boolean. Used by `VitalsPanel` to show "Live" vs "Mock" badge.

#### `resetSession()`

Full reset to initial state — zeroes out vitals, clears readings and summary, resets all flags. Called when starting a new session from `SummaryPage` or returning home.

---

## The WebSocket Hook (`useWebSocket.js`)

A custom React hook that manages the real-time data connection. It encapsulates:
- Real WebSocket connection attempt
- Automatic mock data fallback
- Proper cleanup on unmount

### Usage

```js
// In MonitorPage — activates while the page is mounted
useWebSocket(true)

// Pass false to disconnect (e.g. when session ends)
useWebSocket(false)
```

The `enabled` parameter controls whether the hook tries to connect at all. Passing `false` cleans up any open connection and stops the mock timer.

### Connection Logic

```
On mount (enabled=true):
  1. Attempt: new WebSocket('ws://localhost:8000/ws/vitals')
  2. ws.onopen  → setWsConnected(true), stop mock
  3. ws.onerror → start mock (if not already mocking)
  4. ws.onclose → setWsConnected(false), start mock
  5. Fallback timer: if WS not OPEN after 2000ms → start mock

On unmount:
  clearTimeout(fallbackTimer)
  stopMock()
  ws.close() (if readyState < CLOSING)
```

The 2-second fallback timer prevents the UI from stalling if the WebSocket takes too long to reject.

### Mock Data Generator

```js
const mockReading = () => ({
  hr:            60 + Math.random() * 20,      // 60–80 BPM
  br:            12 + Math.random() * 8,       // 12–20 br/min
  hrv:           20 + Math.random() * 30,      // 20–50 ms
  stress:        Math.random() * 40,           // 0–40
  bvp:           Array.from({ length: 32 }, (_, i) =>
                   Math.sin((i / 32) * Math.PI * 2 + Math.random() * 0.5) * 0.8),
  lighting:      ['Good', 'Good', 'Good', 'Mixed', 'Poor'][Math.floor(Math.random() * 5)],
  face_detected: true,
  timestamp:     Date.now() / 1000,
})
```

- BVP is a noisy sine wave (32 samples per reading, simulating one ~1s clip at 32 fps)
- Lighting is weighted 60% Good / 20% Mixed / 20% Poor
- Mock fires every **1000ms** via `setInterval`

### Internal State (Refs)

The hook uses `useRef` rather than `useState` for its internal flags to avoid triggering re-renders:

| Ref | Type | Purpose |
|---|---|---|
| `wsRef` | `WebSocket \| null` | Hold the socket instance for cleanup |
| `mockTimer` | `number \| null` | Hold the `setInterval` ID |
| `isMock` | `boolean` | Track whether mock mode is active (prevents double-starting) |
