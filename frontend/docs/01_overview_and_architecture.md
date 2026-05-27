# VitalLens Frontend — Part 1: Overview & Architecture

## What Is This App?

VitalLens is a **contactless vitals monitoring** web application. The user sits in front of their webcam; the app streams camera frames to a FastAPI backend, which runs an rPPG (remote photoplethysmography) deep-learning model (EfficientPhys) and returns real-time predictions. The frontend receives those predictions over a WebSocket and renders heart rate, breathing rate, HRV, and stress index in real time.

The frontend is **fully decoupled** from the backend. When the backend is unavailable, the `useWebSocket` hook silently falls back to a built-in mock data generator, so every page and component can be developed and tested independently.

---

## Technology Stack

| Concern | Library / Tool | Version |
|---|---|---|
| UI framework | React | 18 (via Vite) |
| Build tool | Vite | ^8.0.4 |
| Routing | react-router-dom | ^7.14.1 |
| Global state | Zustand | ^5.0.12 |
| Webcam | react-webcam | ^7.2.0 |
| Charts | Recharts | ^3.8.1 |
| Styling | Tailwind CSS v4 | ^4.2.2 |
| Typography | Inter (Google Fonts) | 300–900 |

> **Note:** Tailwind CSS v4 uses the new `@import 'tailwindcss'` / `@theme` syntax. The `tailwind.config.js` extends the theme with custom color tokens and keyframes.

---

## Directory Structure

```
frontend/
├── index.html                  ← HTML shell, Inter font link, root div
├── package.json                ← dependencies & npm scripts
├── tailwind.config.js          ← theme extension (colors, keyframes, animations)
├── postcss.config.js           ← Tailwind v4 PostCSS integration
├── GUI_SPEC.md                 ← original design specification
│
└── src/
    ├── main.jsx                ← React root mount
    ├── App.jsx                 ← Router + route declarations
    ├── index.css               ← Global design system
    │
    ├── pages/
    │   ├── HomePage.jsx        ← Landing page / hero
    │   ├── MonitorPage.jsx     ← Live monitoring dashboard
    │   └── SummaryPage.jsx     ← Post-session report
    │
    ├── components/
    │   ├── WebcamCapture.jsx   ← Webcam feed + overlays
    │   ├── VitalsPanel.jsx     ← 2×2 MetricCard grid
    │   ├── MetricCard.jsx      ← Individual live metric display
    │   ├── BVPWaveform.jsx     ← Real-time scrolling BVP chart
    │   ├── LightingBanner.jsx  ← Conditional poor-lighting warning
    │   ├── SessionStats.jsx    ← Summary averages + duration
    │   ├── BVPReplay.jsx       ← Full-session BVP chart (summary page)
    │   └── LightingBreakdown.jsx ← Donut chart of lighting quality
    │
    ├── hooks/
    │   └── useWebSocket.js     ← WebSocket connection + mock fallback
    │
    └── store/
        └── useVitalsStore.js   ← Zustand global state store
```

---

## Application Entry Point

**`index.html`** is the single HTML shell. It sets the page `<title>`, `<meta name="description">` (SEO), preconnects to Google Fonts and loads **Inter** (weights 300–900), contains `<div id="root">` and a `<script type="module">` pointing to `src/main.jsx`.

**`src/main.jsx`** mounts the React tree into `#root` wrapped in `<React.StrictMode>`.

---

## Routing (`App.jsx`)

The app uses **`BrowserRouter`** with three named routes:

| Path | Component | Purpose |
|---|---|---|
| `/` | `HomePage` | Landing page |
| `/monitor` | `MonitorPage` | Live monitoring dashboard |
| `/summary` | `SummaryPage` | Post-session report |
| `*` | `<Navigate to="/" replace>` | Catch-all redirect |

Navigation is done programmatically via `useNavigate()` — every page transition is triggered by user actions (button clicks), which is appropriate for this session-gated flow.

---

## Design System (`index.css` + `tailwind.config.js`)

### Base Theme

The entire app sits on a **deep navy background** (`#0a0f1e`) with two radial gradient overlays baked into `body`:
- Cyan-teal glow at the top center (`rgba(0, 212, 255, 0.12)`)
- Indigo glow at the bottom right (`rgba(99, 102, 241, 0.08)`)

### Custom Tailwind Color Tokens

```js
colors: {
  base:  { 900: '#0a0f1e', 800: '#0f172a', 700: '#1e293b', 600: '#334155' },
  cyan:  { 400: '#22d3ee', 500: '#06b6d4', glow: '#00d4ff' },
  vital: { green: '#10b981', amber: '#f59e0b', red: '#ef4444' }
}
```

### Glassmorphism Card (`.glass-card`)

Used on every content panel:
```css
.glass-card {
  border-radius: 1rem;
  border: 1px solid rgba(255, 255, 255, 0.1);
  background: rgba(15, 23, 42, 0.6);
  backdrop-filter: blur(16px);
}
```

### Metric Glow States

Cards dynamically receive one of four border+glow classes based on vitals status:

| Class | Color | Usage |
|---|---|---|
| `.metric-green` | Emerald `#10b981` | Value within normal range |
| `.metric-amber` | Amber `#f59e0b` | Borderline (within 15% margin) |
| `.metric-red` | Red `#ef4444` | Out of range |
| `.metric-neutral` | White/10% | Value not yet received (`null`) |

### Button Variants

| Class | Style | Usage |
|---|---|---|
| `.btn-primary` | Cyan gradient + glow shadow, lifts on hover | CTA actions |
| `.btn-secondary` | Frosted glass, subtle border | Secondary actions |
| `.btn-danger` | Red gradient + red glow | End Session |

### Keyframe Animations

| Name | Effect | Used On |
|---|---|---|
| `recordingPulse` | Red dot fades/scales 1.5s loop | Recording indicator |
| `float` | Gentle ±10px vertical oscillation | Hero background orbs |
| `slideDown` | Banner enters from top | `LightingBanner` |
| `numberSlide` | Value slides in from above | Metric value updates |

The `.number-update` class is applied in `MetricCard` on every new value via a DOM reflow trick (`void el.offsetWidth`) to re-trigger the CSS animation.

### Recharts Dark Overrides

```css
.recharts-text { fill: rgba(255,255,255,0.5) !important; }
.recharts-cartesian-grid-horizontal line,
.recharts-cartesian-grid-vertical line { stroke: rgba(255,255,255,0.06) !important; }
```
