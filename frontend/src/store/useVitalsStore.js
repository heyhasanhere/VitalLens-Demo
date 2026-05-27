import { create } from 'zustand'

export const WARMUP_MS = 10_000

const useVitalsStore = create((set, get) => ({
  // ─── Session state ──────────────────────────────────────────────────────────
  session: {
    isActive: false,
    startTime: null,
    readings: [],   // array of { hr, br, hrv, stress, bvp, lighting, timestamp }
    summary: null,  // filled on session end
  },

  // ─── Live vitals ────────────────────────────────────────────────────────────
  vitals: {
    hr: null,
    br: null,
    hrv: null,
    stress: null,
    snr: null,        // BVP signal-to-noise ratio; <6 = low confidence
    bvpWindow: [],      // last ~10s of BVP values for the waveform
    lighting: 'Good',
    faceDetected: true,
    faceBbox: null,       // {x,y,w,h} fractions 0–1 from backend, or null
    prevHr: null,
    prevBr: null,
    prevHrv: null,
    prevStress: null,
    lumStd: null,
    blinkRate: null,
    prevBlinkRate: null,
    sway: null,
    rhythm: 'Unknown',
    posHr: null,
    chromHr: null,
  },

  // ─── UI state ───────────────────────────────────────────────────────────────
  lightingBannerDismissed: false,
  consecutivePoorFrames: 0,
  wsConnected: null,   // null=connecting, true=live, false=dropped
  cameraIndex: 0,
  cameraUrl: '',
  inferenceMode: localStorage.getItem('vl_inference_mode') || 'remote',  // 'local' | 'remote'
  selectedModel: localStorage.getItem('vl_model') || 'factorizephys',
  hrZones: null,   // { age: int, hr_zones: { "1": [lo,hi], ... } } — set once per session

  // ─── Actions ────────────────────────────────────────────────────────────────

  startSession: () => set(state => ({
    session: {
      ...state.session,
      isActive: true,
      startTime: Date.now(),
      readings: [],
      summary: null,
    },
    lightingBannerDismissed: false,
    consecutivePoorFrames: 0,
  })),

  endSession: () => {
    const { session, vitals } = get()
    const readings = session.readings

    // Compute summary from recorded readings (fallback if no REST response)
    if (readings.length === 0) return

    const avg = (arr, key) => {
      const vals = arr.map(r => r[key]).filter(v => v != null)
      return vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : 0
    }

    const hrVals = readings.map(r => r.hr).filter(Boolean)
    const lightingCounts = { Good: 0, Mixed: 0, Poor: 0 }
    readings.forEach(r => {
      if (lightingCounts[r.lighting] !== undefined) lightingCounts[r.lighting]++
    })
    const total = readings.length || 1
    const durationSec = session.startTime
      ? Math.round((Date.now() - session.startTime) / 1000)
      : 0

    const bvpSeries = readings.flatMap(r => r.bvp || [])
    const hrSeries = readings.map(r => ({ hr: r.hr, timestamp: r.timestamp }))

    const summary = {
      duration_seconds: durationSec,
      avg_hr: avg(readings, 'hr'),
      avg_br: avg(readings, 'br'),
      avg_hrv: avg(readings, 'hrv'),
      avg_stress: avg(readings, 'stress'),
      min_hr: hrVals.length ? Math.min(...hrVals) : 0,
      max_hr: hrVals.length ? Math.max(...hrVals) : 0,
      lighting_breakdown: {
        Good: lightingCounts.Good / total,
        Mixed: lightingCounts.Mixed / total,
        Poor: lightingCounts.Poor / total,
      },
      bvp_series: bvpSeries,
      hr_series: hrSeries,
      timestamps: readings.map(r => r.timestamp),
    }

    set(state => ({
      session: { ...state.session, isActive: false, summary },
    }))
  },

  // Override summary from REST response
  setSummary: (summaryData) => set(state => ({
    session: { ...state.session, summary: summaryData, isActive: false },
  })),

  // Called on each incoming WebSocket message
  applyReading: (reading) => set(state => {
    const {
      hr, br, hrv, stress, snr, bvp, lighting, face_detected, face_bbox,
      timestamp, lum_std, blink_rate, sway, rhythm, pos_hr, chrom_hr,
    } = reading
    const prev = state.vitals

    const newBvp     = [...(prev.bvpWindow || []), ...(bvp || [])].slice(-320)
    const poorFrames = lighting === 'Poor' ? state.consecutivePoorFrames + 1 : 0

    // Suppress rPPG-derived metrics for the first WARMUP_MS of each session.
    const warmupDone = !state.session.startTime ||
      (Date.now() - state.session.startTime >= WARMUP_MS)

    // Clear rPPG metrics when face is explicitly absent — stale values are misleading.
    const noFace = face_detected === false

    return {
      vitals: {
        hr:            noFace ? null : warmupDone ? (hr     ?? prev.hr)     : prev.hr,
        br:            noFace ? null : warmupDone ? (br     ?? prev.br)     : prev.br,
        hrv:           noFace ? null : warmupDone ? (hrv    ?? prev.hrv)    : prev.hrv,
        stress:        noFace ? null : warmupDone ? (stress ?? prev.stress) : prev.stress,
        posHr:         noFace ? null : warmupDone ? (pos_hr   ?? prev.posHr)   : prev.posHr,
        chromHr:       noFace ? null : warmupDone ? (chrom_hr ?? prev.chromHr) : prev.chromHr,
        bvpWindow:     noFace ? [] : newBvp,
        lighting:      lighting      ?? prev.lighting,
        faceDetected:  face_detected ?? prev.faceDetected,
        faceBbox:      face_bbox     ?? prev.faceBbox,
        snr:           noFace ? null : (snr ?? prev.snr),
        prevHr:        prev.hr,
        prevBr:        prev.br,
        prevHrv:       prev.hrv,
        prevStress:    prev.stress,
        lumStd:        lum_std    ?? prev.lumStd,
        blinkRate:     blink_rate ?? prev.blinkRate,
        prevBlinkRate: prev.blinkRate,
        sway:          sway   ?? prev.sway,
        rhythm:        rhythm ?? prev.rhythm,
      },
      consecutivePoorFrames: poorFrames,
      // Only record once the backend is ready AND the warmup window has passed.
      session: state.session.isActive && reading.ready !== false && warmupDone
        ? {
            ...state.session,
            readings: [
              ...state.session.readings,
              { hr, br, hrv, stress, bvp, lighting, face_detected, timestamp },
            ],
          }
        : state.session,
    }
  }),

  dismissLightingBanner: () => set({ lightingBannerDismissed: true }),
  setWsConnected: (val) => set({ wsConnected: val }),
  setCameraIndex: (idx) => set({ cameraIndex: idx }),
  setCameraUrl:   (url) => set({ cameraUrl: url }),
  setHrZones:      (data) => set({ hrZones: data }),
  setInferenceMode: (mode) => {
    localStorage.setItem('vl_inference_mode', mode)
    set({ inferenceMode: mode })
  },
  setSelectedModel: (model) => {
    localStorage.setItem('vl_model', model)
    set({ selectedModel: model })
  },

  resetSession: () => set(state => ({
    session: { isActive: false, startTime: null, readings: [], summary: null },
    vitals: {
      hr: null, br: null, hrv: null, stress: null, snr: null,
      bvpWindow: [], lighting: 'Good', faceDetected: true, faceBbox: null,
      prevHr: null, prevBr: null, prevHrv: null, prevStress: null, lumStd: null,
      blinkRate: null, prevBlinkRate: null, sway: null, rhythm: 'Unknown',
      posHr: null, chromHr: null,
    },
    lightingBannerDismissed: false,
    consecutivePoorFrames: 0,
    wsConnected: null,
    hrZones: null,
  })),
}))

export default useVitalsStore
