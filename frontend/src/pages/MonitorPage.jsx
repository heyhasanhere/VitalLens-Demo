import React, { useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { Settings } from 'lucide-react'
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid,
  ReferenceLine, Tooltip, ResponsiveContainer,
} from 'recharts'
import useVitalsStore from '../store/useVitalsStore'
import { API_BASE } from '../config'
import { useWebSocket } from '../hooks/useWebSocket'
import WebcamCapture from '../components/WebcamCapture'
import BVPWaveform from '../components/BVPWaveform'
import VitalsPanel from '../components/VitalsPanel'
import DebugPanel from '../components/DebugPanel'

// ── HR trend chart ────────────────────────────────────────────────────────────
function HRTrendChart() {
  const readings  = useVitalsStore(s => s.session.readings)
  const startTime = useVitalsStore(s => s.session.startTime)

  const data = readings
    .filter(r => r.hr > 0)
    .slice(-120)
    .map(r => ({
      t:  startTime ? Math.round((r.timestamp * 1000 - startTime) / 1000) : 0,
      hr: parseFloat((r.hr).toFixed(1)),
    }))

  const fmtTime = s => s < 60 ? `${s}s` : `${Math.floor(s / 60)}m${String(s % 60).padStart(2, '0')}s`

  return (
    <div className="rounded-xl p-3"
         style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.07)' }}>
      <p className="text-xs font-semibold text-white/30 uppercase tracking-widest mb-2">HR Trend (last 2 min)</p>
      {data.length < 3 ? (
        <div className="flex items-center justify-center h-20 text-white/20 text-xs">Calibrating…</div>
      ) : (
        <ResponsiveContainer width="100%" height={100}>
          <AreaChart data={data} margin={{ top: 4, right: 4, left: 0, bottom: 0 }}>
            <defs>
              <linearGradient id="hrGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%"  stopColor="#00d4ff" stopOpacity={0.25} />
                <stop offset="95%" stopColor="#00d4ff" stopOpacity={0}    />
              </linearGradient>
            </defs>
            <CartesianGrid vertical={false} strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
            <XAxis
              dataKey="t"
              tickFormatter={fmtTime}
              tick={{ fill: 'rgba(255,255,255,0.2)', fontSize: 9 }}
              tickLine={false} axisLine={false}
              interval="preserveStartEnd"
            />
            <YAxis
              domain={[40, 150]}
              tick={{ fill: 'rgba(255,255,255,0.2)', fontSize: 9 }}
              tickLine={false} axisLine={false} width={28}
            />
            {/* normal resting HR band markers */}
            <ReferenceLine y={60}  stroke="rgba(255,255,255,0.08)" strokeDasharray="3 2" />
            <ReferenceLine y={100} stroke="rgba(255,255,255,0.08)" strokeDasharray="3 2" />
            <Tooltip
              contentStyle={{
                background: '#0d1627', border: '1px solid rgba(255,255,255,0.1)',
                borderRadius: 8, fontSize: 11,
              }}
              labelStyle={{ color: 'rgba(255,255,255,0.4)' }}
              itemStyle={{ color: '#00d4ff' }}
              labelFormatter={fmtTime}
              formatter={v => [`${v} BPM`, 'HR']}
            />
            <Area
              type="monotone" dataKey="hr"
              stroke="#00d4ff" strokeWidth={1.5}
              fill="url(#hrGrad)" dot={false}
              isAnimationActive={false}
            />
          </AreaChart>
        </ResponsiveContainer>
      )}
    </div>
  )
}

// ── HR zone badge ─────────────────────────────────────────────────────────────
function HRZoneBadge({ hr, hrZones }) {
  if (!hrZones || !hr || hr <= 0) return null
  const zones = hrZones.hr_zones
  const zoneNum = Object.entries(zones).find(([, [lo, hi]]) => hr >= lo && hr < hi)?.[0]
  if (!zoneNum) return null
  const colors = { '1': '#10b981', '2': '#6ee7b7', '3': '#f59e0b', '4': '#f97316', '5': '#ef4444' }
  const labels = { '1': 'Warm-up', '2': 'Fat Burn', '3': 'Aerobic', '4': 'Anaerobic', '5': 'Max' }
  return (
    <span className="text-xs px-2 py-0.5 rounded-full font-semibold"
          style={{ background: `${colors[zoneNum]}22`, border: `1px solid ${colors[zoneNum]}55`, color: colors[zoneNum] }}>
      Z{zoneNum} {labels[zoneNum]}
    </span>
  )
}

// ── Main page ─────────────────────────────────────────────────────────────────
export default function MonitorPage() {
  const navigate     = useNavigate()
  const isActive     = useVitalsStore(s => s.session.isActive)
  const startTime    = useVitalsStore(s => s.session.startTime)
  const endSession   = useVitalsStore(s => s.endSession)
  const startSession = useVitalsStore(s => s.startSession)
  const wsConnected  = useVitalsStore(s => s.wsConnected)
  const faceBbox     = useVitalsStore(s => s.vitals.faceBbox)
  const hr           = useVitalsStore(s => s.vitals.hr)
  const hrZones      = useVitalsStore(s => s.hrZones)
  const setHrZones   = useVitalsStore(s => s.setHrZones)

  useEffect(() => {
    if (!isActive) startSession()
  }, []) // eslint-disable-line

  useWebSocket(true)

  // One-shot age estimation: fires once on first valid face bbox
  const ageCalledRef = React.useRef(false)
  useEffect(() => {
    if (!faceBbox || hrZones || ageCalledRef.current) return
    const video = document.querySelector('video')
    if (!video || !video.videoWidth) return
    ageCalledRef.current = true   // prevent duplicate calls while fetch is in flight
    try {
      const canvas = document.createElement('canvas')
      canvas.width = 160; canvas.height = 160
      const ctx = canvas.getContext('2d')
      const { x, y, w, h } = faceBbox
      ctx.drawImage(video,
        x * video.videoWidth,  y * video.videoHeight,
        w * video.videoWidth,  h * video.videoHeight,
        0, 0, 160, 160)
      const b64 = canvas.toDataURL('image/jpeg', 0.8).split(',')[1]
      fetch(`${API_BASE}/session/estimate-age`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: b64 }),
      })
        .then(r => r.json())
        .then(data => { if (data.hr_zones) setHrZones(data) })
        .catch(() => { ageCalledRef.current = false })  // allow retry on network error
    } catch { ageCalledRef.current = false }
  }, [faceBbox]) // eslint-disable-line

  const [showDebug, setShowDebug] = React.useState(false)
  const [elapsed, setElapsed] = React.useState(0)
  useEffect(() => {
    const t = setInterval(() => {
      if (startTime) setElapsed(Math.floor((Date.now() - startTime) / 1000))
    }, 1000)
    return () => clearInterval(t)
  }, [startTime])

  const formatTime = s => {
    const m = String(Math.floor(s / 60)).padStart(2, '0')
    const sec = String(s % 60).padStart(2, '0')
    return `${m}:${sec}`
  }

  const handleEndSession = async () => {
    const readings = useVitalsStore.getState().session.readings
    try {
      const res = await fetch(`${API_BASE}/session/end`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ readings }),
      })
      if (res.ok) {
        const data = await res.json()
        if (data.duration_seconds !== undefined) {
          useVitalsStore.getState().setSummary(data)
          navigate('/summary')
          return
        }
      }
    } catch { /* fall through */ }
    endSession()
    navigate('/summary')
  }

  return (
    <div className="min-h-screen flex flex-col" style={{ background: '#0a0f1e' }}>

      {/* ── Header ──────────────────────────────────────────────────────────── */}
      <header className="flex items-center justify-between px-3 sm:px-6 py-3"
              style={{ borderBottom: '1px solid rgba(255,255,255,0.07)' }}>
        <span className="font-semibold text-white/70 text-sm tracking-wide hidden sm:block">VitalLens</span>
        <div className="flex items-center gap-2 sm:gap-4 w-full sm:w-auto justify-between sm:justify-start">
          <HRZoneBadge hr={hr} hrZones={hrZones} />
          <span className="font-mono text-sm text-white/40">{formatTime(elapsed)}</span>
          <button
            onClick={() => setShowDebug(v => !v)}
            title="Toggle signal debug panel"
            className="text-xs px-3 py-1.5 rounded-lg transition-colors"
            style={{
              color: showDebug ? '#22d3ee' : 'rgba(255,255,255,0.4)',
              border: `1px solid ${showDebug ? 'rgba(6,182,212,0.5)' : 'rgba(255,255,255,0.12)'}`,
            }}
          >
            <Settings size={14} />
          </button>
          <button
            onClick={handleEndSession}
            className="text-xs px-4 py-1.5 rounded-lg text-white/70 border border-white/20 hover:border-white/40 transition-colors"
          >
            End Session
          </button>
        </div>
      </header>

      {/* ── Debug panel ─────────────────────────────────────────────────────── */}
      {showDebug && (
        <div className="fixed bottom-6 right-6 z-50">
          <DebugPanel onClose={() => setShowDebug(false)} />
        </div>
      )}

      {/* ── Body ────────────────────────────────────────────────────────────── */}
      <main className="flex-1 grid grid-cols-1 lg:grid-cols-2 gap-6 p-6">

        {/* Left col: camera feed + BVP waveform */}
        <div className="flex flex-col gap-4">
          <WebcamCapture isRecording={isActive} />
          <BVPWaveform autoScale />
        </div>

        {/* Right col: vitals cards + HR trend */}
        <div className="flex flex-col gap-4">
          <VitalsPanel wsConnected={wsConnected} />
          <HRTrendChart />
        </div>

      </main>
    </div>
  )
}
