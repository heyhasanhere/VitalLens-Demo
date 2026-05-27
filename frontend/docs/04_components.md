# VitalLens Frontend — Part 4: Components

## Component Tree

```
App
├── HomePage                    (no sub-components — self-contained)
├── MonitorPage
│   ├── LightingBanner          ← conditional warning banner
│   ├── WebcamCapture           ← webcam + SVG overlays
│   ├── VitalsPanel
│   │   ├── MetricCard (HR)
│   │   ├── MetricCard (BR)
│   │   ├── MetricCard (HRV)
│   │   └── MetricCard (Stress)
│   └── BVPWaveform             ← real-time Recharts line chart
└── SummaryPage
    ├── SessionStats            ← averages + duration tile
    ├── BVPReplay               ← full-session Recharts line chart
    └── LightingBreakdown       ← Recharts donut chart
```

---

## `WebcamCapture.jsx`

**Props:** `{ isRecording: boolean }`

**Store subscriptions:** `vitals.lighting`, `vitals.faceDetected`

### What It Renders

A `relative` container with `aspect-video` that layers:

1. **`<Webcam>`** from `react-webcam` — `mirrored={true}`, `audio={false}`, targets 1280×720 user-facing camera, fills container with `object-cover`.

2. **Face bounding box SVG** (conditional on `faceDetected`):
   - An SVG overlay with `viewBox="0 0 100 100"` and `preserveAspectRatio="none"` so it stretches to fill the video
   - A dashed cyan rectangle (`strokeDasharray="4 2"`) at fixed coordinates (x=30, y=18, w=40, h=52) representing the face region
   - Four corner accent L-shapes (4 `<g>` elements with two `<line>` each) at each corner for a scanner aesthetic
   - All strokes are `#00d4ff` (cyan glow)
   - **Note:** The bounding box coordinates are mocked/fixed. In production, real MediaPipe face detection coordinates would replace these.

3. **Lighting quality badge** (top-right corner):
   - A pill with blurred background showing emoji + label
   - Config lookup by `lighting` string:

   | Lighting | Emoji | Color |
   |---|---|---|
   | Good | 🟢 | `rgba(16,185,129,0.85)` |
   | Mixed | 🟡 | `rgba(245,158,11,0.85)` |
   | Poor | 🔴 | `rgba(239,68,68,0.85)` |

4. **Recording indicator** (top-left, conditional on `isRecording`):
   - A black frosted pill with the `.recording-dot` animated span + "REC" in red

5. **Face detection status** (bottom center):
   - A pill that shows "✓ Face Detected" (green) or "⚠ No Face — please centre yourself" (red)

6. **Scan line overlay** (conditional on `isRecording`):
   - A subtle gradient div that oscillates via the `float` animation, giving a "scanning" visual effect

---

## `VitalsPanel.jsx`

**Props:** `{ wsConnected: boolean }`

**Store subscriptions:** `vitals` (full object)

A container component that renders the "Live Vitals" heading, the connection status badge, and a 2×2 grid of `MetricCard` components.

### Connection Status Badge

Shown next to the "Live Vitals" heading:
- **Green pill** "Live" when `wsConnected === true`
- **Amber pill** "Mock" when `wsConnected === false` (mock mode)

### Metrics Configuration

```js
const METRICS = [
  { id: 'metric-card-hr',     label: 'Heart Rate',     unit: 'BPM',    icon: '♥',  normalRange: '60–100 BPM',    storeKey: 'hr',     prevKey: 'prevHr'     },
  { id: 'metric-card-br',     label: 'Breathing Rate', unit: 'br/min', icon: '🌬', normalRange: '12–20 br/min',  storeKey: 'br',     prevKey: 'prevBr'     },
  { id: 'metric-card-hrv',    label: 'HRV (RMSSD)',    unit: 'ms',     icon: '〰', normalRange: '20–50 ms',      storeKey: 'hrv',    prevKey: 'prevHrv'    },
  { id: 'metric-card-stress', label: 'Stress Index',   unit: '/100',   icon: '🧘', normalRange: '< 30 = low',    storeKey: 'stress', prevKey: 'prevStress' },
]
```

Each `MetricCard` receives `value` and `prevValue` sourced from the corresponding `storeKey` and `prevKey` in `vitals`.

---

## `MetricCard.jsx`

**Props:** `{ id, label, metricKey, value, prevValue, unit, icon, normalRange }`

The most complex component. Handles status classification, trend detection, and value-change animation.

### Status Classification (`getStatus`)

Normal ranges are defined locally:
```js
const NORMAL = {
  hr:     [60, 100],
  br:     [12, 20],
  hrv:    [20, 50],
  stress: [0, 30],  // stress uses different logic (upper-only threshold)
}
```

Logic:
- `null` → `'neutral'`
- For `stress`: `< 30` → green, `< 60` → amber, else → red
- For others: within `[lo, hi]` → green; within a 15% margin on either side → amber; else → red

### Trend Arrow (`getTrend`)

```js
function getTrend(current, prev) {
  if (prev == null || current == null) return '→'
  const diff = current - prev
  if (Math.abs(diff) < 0.5) return '→'
  return diff > 0 ? '↑' : '↓'
}
```

A dead-band of ±0.5 prevents jitter showing spurious arrows on flat readings.

### Visual States

| Status | Border class | Value color | Trend color |
|---|---|---|---|
| green | `.metric-green` | `#10b981` | `#6ee7b7` |
| amber | `.metric-amber` | `#f59e0b` | `#fcd34d` |
| red | `.metric-red` | `#ef4444` | `#fca5a5` |
| neutral | `.metric-neutral` | `rgba(255,255,255,0.5)` | `rgba(255,255,255,0.3)` |

### Value Animation

Uses `useRef` for `numRef` (the value `<span>`) and `prevVal` (the previously rendered value). On every render where `value !== prevVal.current`:
1. Remove `number-update` class
2. Force a browser reflow via `void numRef.current.offsetWidth`
3. Add `number-update` class back

This triggers the `numberSlide` CSS animation (slide in from above) every time the value changes.

### Display

- `value.toFixed(1)` or `'—'` if null
- Normal range shown in small text at the bottom: "Normal: `{normalRange}`"

---

## `BVPWaveform.jsx`

**Store subscriptions:** `vitals.bvpWindow`

A real-time Recharts `LineChart` showing the last 10 seconds of BVP signal.

### Data Transformation

```js
const chartData = useMemo(() => {
  const slice = bvpWindow.slice(-320)  // last 320 samples
  return slice.map((v, i) => ({
    t: parseFloat(((i / slice.length) * 10).toFixed(2)),  // 0–10s x-axis
    v: parseFloat(v.toFixed(4)),
  }))
}, [bvpWindow])
```

When `bvpWindow` is empty, renders 64 flat zero points as a placeholder baseline.

### Chart Details

- **X axis:** 0–10 seconds, 6 ticks, formatted as `"Xs"`
- **Y axis:** Fixed domain `[-1.2, 1.2]`, 5 ticks
- **Line stroke:** Linear gradient `url(#bvpGradient)` — cyan at left, bright cyan center, indigo at right
- **`isAnimationActive={false}`** — critical for real-time data; Recharts' built-in transition animation would cause lag/stuttering with 320-point updates every second
- **Custom Tooltip:** Small dark pill showing the BVP value to 3dp
- **Reference line** at y=0 (faint white)

---

## `LightingBanner.jsx`

**Store subscriptions:** `consecutivePoorFrames`, `lightingBannerDismissed`, `dismissLightingBanner`

A conditional amber warning banner. Renders `null` if:
- `dismissed === true`, OR
- `consecutivePoorFrames < 3`

When visible, it slides in from above via `animate-[slideDown_0.3s_ease-out]`.

Text: "Poor lighting detected — results may be inaccurate. Try facing a window or turning on a light."

The dismiss button (`id="dismiss-lighting-banner"`) calls `dismissLightingBanner()` and the banner will not reappear for the rest of the session.

---

## `SessionStats.jsx`

**Props:** `{ summary: Summary }`

Renders a two-part layout:

1. **Duration highlight row** — Full-width `glass-card` with a timer icon, the formatted session duration as a gradient text (`Xm Ys`), and Min/Max HR on the right side (hidden below `md` breakpoint).

2. **Averages grid** — `grid-cols-2 md:grid-cols-4` with four `StatTile` sub-components:
   - Avg Heart Rate (highlighted in cyan)
   - Avg Breathing Rate
   - Avg HRV (RMSSD)
   - Avg Stress Index

`StatTile` is a local sub-component — a simple `glass-card` with label, large value, and unit.

---

## `BVPReplay.jsx`

**Props:** `{ bvpSeries: number[] }`

A static (non-updating) Recharts chart showing the full BVP signal recorded over the entire session. Used on the Summary page.

### Data Transformation

```js
return bvpSeries.map((v, i) => ({
  t: parseFloat(((i / bvpSeries.length) * (bvpSeries.length / 32)).toFixed(2)),
  v: parseFloat((+v).toFixed(4)),
}))
```

The x-axis is scaled by `bvpSeries.length / 32` to convert sample index to seconds (assuming 32 samples/s).

Returns `null` if `bvpSeries` is empty (nothing to show).

Differences from `BVPWaveform`:
- Y domain is `[-1.5, 1.5]` (slightly wider for full-session variation)
- Gradient reverses direction (indigo → cyan → cyan)
- No reference line
- Shows total sample count in the header: "`N` samples"

---

## `LightingBreakdown.jsx`

**Props:** `{ breakdown: { Good: number, Mixed: number, Poor: number } }`

A Recharts **donut chart** (PieChart with `innerRadius={50}`, `outerRadius={80}`) visualising what fraction of the session was spent in each lighting category.

### Data Preparation

```js
const data = Object.entries(breakdown)
  .filter(([, val]) => val > 0)        // skip zero-value segments
  .map(([name, val]) => ({
    name,
    value: parseFloat((val * 100).toFixed(1))  // fraction → percentage
  }))
```

### Colors

| Segment | Color |
|---|---|
| Good | `#10b981` (emerald) |
| Mixed | `#f59e0b` (amber) |
| Poor | `#ef4444` (red) |

### Custom Label

Percentage labels are rendered inside each segment via a `CustomLabel` component using trigonometry to position text at the midpoint of each arc. Segments < 5% suppress their label to avoid clutter.

### Legend

A custom vertical legend (outside the chart) shows a colored dot, the category name, and the percentage value for each segment.
