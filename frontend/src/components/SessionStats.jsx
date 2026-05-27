import React from 'react'

function StatTile({ label, value, unit, highlight }) {
  return (
    <div className="glass-card p-4 flex flex-col gap-1 text-center"
         style={highlight ? { borderColor: 'rgba(6,182,212,0.4)', boxShadow: '0 0 16px rgba(6,182,212,0.1)' } : {}}>
      <span className="text-xs text-white/40 uppercase tracking-wider font-medium">{label}</span>
      <span className="text-2xl font-black tabular-nums"
            style={{ color: highlight ? '#22d3ee' : 'rgba(255,255,255,0.85)' }}>
        {value != null ? (typeof value === 'number' ? value.toFixed(1) : value) : '—'}
      </span>
      {unit && <span className="text-xs text-white/30">{unit}</span>}
    </div>
  )
}

function formatDuration(sec) {
  if (!sec) return '0:00'
  const m = Math.floor(sec / 60)
  const s = sec % 60
  return `${m}m ${s}s`
}

export default function SessionStats({ summary }) {
  if (!summary) return null
  const { duration_seconds, avg_hr, avg_br, avg_hrv, avg_stress, min_hr, max_hr } = summary

  return (
    <div>
      <h2 className="text-sm font-semibold text-white/40 uppercase tracking-widest mb-4">Session Overview</h2>

      {/* Duration highlight */}
      <div className="glass-card px-6 py-4 flex items-center justify-between mb-4"
           style={{ border: '1px solid rgba(6,182,212,0.2)' }}>
        <div className="flex items-center gap-3">
          <span className="text-2xl">⏱</span>
          <div>
            <div className="text-xs text-white/40 uppercase tracking-wider">Session Duration</div>
            <div className="text-3xl font-black text-gradient">{formatDuration(duration_seconds)}</div>
          </div>
        </div>
        <div className="hidden md:flex items-center gap-6">
          <div className="text-right">
            <div className="text-xs text-white/30">Min HR</div>
            <div className="text-lg font-bold text-white/70">{min_hr?.toFixed(0)} <span className="text-xs font-normal text-white/30">BPM</span></div>
          </div>
          <div className="text-right">
            <div className="text-xs text-white/30">Max HR</div>
            <div className="text-lg font-bold text-white/70">{max_hr?.toFixed(0)} <span className="text-xs font-normal text-white/30">BPM</span></div>
          </div>
        </div>
      </div>

      {/* Averages grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <StatTile label="Avg Heart Rate"     value={avg_hr}     unit="BPM"    highlight />
        <StatTile label="Avg Breathing Rate" value={avg_br}     unit="br/min" />
        <StatTile label="Avg HRV (RMSSD)"    value={avg_hrv}    unit="ms"     />
        <StatTile label="Avg Stress Index"   value={avg_stress} unit="/100"   />
      </div>
    </div>
  )
}
