# VitalLens Frontend — Part 3: Pages

## Page Flow Diagram

```
/ (HomePage)
    │  "Start Session" button
    │  → startSession() + navigate('/monitor')
    ▼
/monitor (MonitorPage)
    │  "End Session" button
    │  → POST /session/end (or endSession() fallback)
    │  → navigate('/summary')
    ▼
/summary (SummaryPage)
    │  "New Session" → resetSession() + startSession() + navigate('/monitor')
    │  "Back to Home" → resetSession() + navigate('/')
    └──────────────────────────────────────────────────
```

---

## HomePage (`src/pages/HomePage.jsx`)

### Purpose

The landing page / marketing hero. Explains what VitalLens does and provides the single entry point into a session.

### Layout Sections

1. **Fixed Nav Bar** — Glassmorphic, `position: fixed`, `z-50`. Contains the "VL" logo badge (cyan→indigo gradient) and a "Research Preview" pill with a pulsing green dot.

2. **Hero Section** — Full viewport height `flex-col` centered. Contains:
   - Two large floating background "orbs" (`radial-gradient` divs with the `float` animation at 8s and 10s respectively, in reverse phase for visual variation)
   - A small cyan pill badge: "Contactless · Real-Time · AI-Powered"
   - The `<h1>` "VitalLens" with the `.text-gradient` class (cyan→indigo)
   - Subtitle and description paragraph
   - **"Start Session" button** (`id="start-session-btn"`) — the primary CTA
   - Research disclaimer in tiny text

3. **Preview Metric Strip** — A `grid-cols-2 md:grid-cols-4` row of four `glass-card` tiles showing the four metrics with `—` placeholders. Hovering a tile reveals a cyan border. This communicates to the user what data they'll see without showing real values yet.

4. **Feature Grid** — `grid-cols-1 md:grid-cols-2` layout with four feature tiles. Each tile has an icon, bold title, and description sentence. Title turns cyan on hover via `group-hover:text-cyan-300`.

5. **Footer** — Attribution line: "VitalLens — University of Technology Sydney · Deep Learning Research Project"

### Data Definitions (static)

```js
const features = [
  { icon: '📷', title: 'Webcam-Based', desc: '...' },
  { icon: '❤️', title: 'rPPG Technology', desc: '...' },
  { icon: '🧠', title: 'AI-Powered', desc: '...' },
  { icon: '🔒', title: 'Fully Private', desc: '...' },
]

const previewStats = [
  { label: 'Heart Rate',     unit: 'BPM',    icon: '♥' },
  { label: 'Breathing Rate', unit: 'br/min', icon: '🌬' },
  { label: 'HRV (RMSSD)',    unit: 'ms',     icon: '〰' },
  { label: 'Stress Index',   unit: '0–100',  icon: '🧘' },
]
```

### Action on "Start Session"

```js
const handleStart = () => {
  startSession()    // initialises session in Zustand
  navigate('/monitor')
}
```

---

## MonitorPage (`src/pages/MonitorPage.jsx`)

### Purpose

The core live monitoring dashboard. Renders the webcam feed and real-time vitals side by side while the WebSocket streams readings from the backend.

### Session Guard

```js
useEffect(() => {
  if (!isActive) startSession()
}, [])
```

If the user navigates directly to `/monitor` (e.g. by typing the URL), this ensures a session is started. This is a defensive measure — in normal flow `startSession()` is called by `HomePage`.

### WebSocket Activation

```js
useWebSocket(true)
```

This single line activates the WebSocket hook for the lifetime of the `MonitorPage`. When the component unmounts (navigation away), the hook's cleanup function closes the socket and stops the mock timer.

### Session Timer

A `setInterval` running every 1000ms calculates `Math.floor((Date.now() - startTime) / 1000)` and stores it in local `elapsed` state. This is formatted as `MM:SS` and displayed in the sticky header alongside the pulsing red recording dot.

```js
const formatTime = (sec) => {
  const m = String(Math.floor(sec / 60)).padStart(2, '0')
  const s = String(sec % 60).padStart(2, '0')
  return `${m}:${s}`
}
```

### End Session Flow

```js
const handleEndSession = async () => {
  try {
    const res = await fetch('http://localhost:8000/session/end', { method: 'POST' })
    if (res.ok) {
      const data = await res.json()
      useVitalsStore.getState().setSummary(data)  // use backend summary
      navigate('/summary')
      return
    }
  } catch { /* fall through */ }

  endSession()       // compute summary client-side
  navigate('/summary')
}
```

The REST call is attempted first. If the backend is unavailable (network error or non-OK response), the catch block falls through to the client-side `endSession()` computation.

> **Note:** `useVitalsStore.getState()` is used here (rather than the hook selector) because the call is inside an async function and React's rules of hooks don't apply to imperative calls.

### Layout

The page body is a `flex-col` with three areas:

1. **Sticky Header** — Contains logo, session timer with recording dot, and the "End Session" (`btn-danger`) button.

2. **Main content** — `grid grid-cols-1 lg:grid-cols-2 gap-6`:
   - **Left column:** `<WebcamCapture isRecording={isActive} />` + a tips card below it
   - **Right column:** `<VitalsPanel wsConnected={wsConnected} />`

3. **Below the grid:** `<BVPWaveform />` spanning full width

The `LightingBanner` is rendered between the header and the main grid, inside a `px-6 pt-4` wrapper. It is conditionally visible based on store state.

---

## SummaryPage (`src/pages/SummaryPage.jsx`)

### Purpose

The post-session report page. Shows aggregated statistics, a full-session BVP replay, lighting quality breakdown, and provides export / navigation options.

### Summary Guard

```js
useEffect(() => {
  if (!summary) navigate('/')
}, [summary, navigate])

if (!summary) return null
```

If `summary` is null (e.g. user refreshes the page), they're immediately redirected to home. The `return null` prevents any flash of unstyled content while the redirect fires.

### CSV Export

The `exportCSV(summary, readings)` helper (defined at module level) produces a CSV with two sections:

**Section 1 — Per-reading rows:**

| Timestamp | HR (BPM) | BR (br/min) | HRV (ms) | Stress (0-100) | Lighting |
|---|---|---|---|---|---|
| ISO8601 | 2dp float | 2dp float | 2dp float | 2dp float | string |

**Section 2 — Summary footer** (blank row separator + summary key-value rows):
- Duration (s), Avg HR, Avg BR, Avg HRV, Avg Stress, Min HR, Max HR

The file is downloaded via a programmatically created `<a>` element with a `Blob` URL. The filename is timestamped: `vitallens-session-2026-04-28T16-30-00.csv`.

### Layout

The page uses `max-w-5xl mx-auto` centered layout:

1. **Sticky Header** — Logo + "Session Summary" breadcrumb + "Export CSV" (`btn-secondary`) + "New Session" (`btn-primary`)

2. **Hero Badge** — Green "✓ Session Complete" pill + `<h1>` "Your Vitals Report"

3. **`<SessionStats summary={summary} />`** — duration + averages grid

4. **Charts Row** — `grid-cols-1 lg:grid-cols-3`:
   - Left (2/3 width): `<BVPReplay bvpSeries={summary.bvp_series} />`
   - Right (1/3 width): `<LightingBreakdown breakdown={summary.lighting_breakdown} />`

5. **Footer Actions** — "← Back to Home" + "▶ Start New Session"

6. **Disclaimer** — "For research purposes only. Not a medical device."

### Navigation Actions

| Action | Code |
|---|---|
| New Session | `resetSession()` → `startSession()` → `navigate('/monitor')` |
| Back to Home | `resetSession()` → `navigate('/')` |

`resetSession()` is always called before either navigation to ensure the Zustand store is clean for the next session.
