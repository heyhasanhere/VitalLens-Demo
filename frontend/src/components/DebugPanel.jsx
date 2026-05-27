import React, { useEffect, useRef, useState } from 'react'
import { Check, RotateCcw, X } from 'lucide-react'
import useVitalsStore from '../store/useVitalsStore'
import { API_BASE as API } from '../config'

// rec = the value that works well for typical laptop/phone sessions
const PARAMS = [
  { key: 'snr_threshold',    label: 'SNR threshold',   min: 1.0,  max: 6.0,  step: 0.1,  unit: '',     rec: 2.8,  recNote: 'OBS/phone' },
  { key: 'hr_jump_thresh',   label: 'HR jump gate',    min: 5,    max: 60,   step: 1,    unit: ' BPM', rec: 25,   recNote: 'continuity filter' },
  { key: 'sway_thresh',      label: 'Head sway gate',  min: 0.01, max: 0.15, step: 0.01, unit: '',     rec: 0.03, recNote: 'fraction of frame' },
  { key: 'ear_blink_thresh', label: 'Blink EAR',       min: 0.10, max: 0.35, step: 0.01, unit: '',     rec: 0.22, recNote: 'glasses-robust' },
]

function SnrBar({ snr, threshold }) {
  const pct       = Math.min(100, (snr / 6) * 100)
  const threshPct = Math.min(100, (threshold / 6) * 100)
  const passing   = snr >= threshold
  const color     = passing ? '#10b981' : snr >= threshold * 0.8 ? '#f59e0b' : '#ef4444'
  return (
    <div className="relative h-2 rounded-full overflow-visible" style={{ background: 'rgba(255,255,255,0.08)' }}>
      <div className="h-full rounded-full transition-all duration-300" style={{ width: `${pct}%`, background: color }} />
      {/* threshold tick mark */}
      <div className="absolute top-[-2px] h-[10px] w-[2px] rounded-full"
           style={{ left: `${threshPct}%`, background: 'rgba(255,255,255,0.55)' }} />
    </div>
  )
}

export default function DebugPanel({ onClose }) {
  const snr     = useVitalsStore(s => s.vitals.snr)
  const hr      = useVitalsStore(s => s.vitals.hr)
  const posHr   = useVitalsStore(s => s.vitals.posHr)
  const chromHr = useVitalsStore(s => s.vitals.chromHr)

  const [config, setConfig] = useState(null)
  const debounceRef = useRef({})

  useEffect(() => {
    fetch(`${API}/session/config`)
      .then(r => r.json())
      .then(setConfig)
      .catch(() => {})
  }, [])

  const handleChange = (key, value) => {
    const num = parseFloat(value)
    setConfig(prev => ({ ...prev, [key]: num }))
    clearTimeout(debounceRef.current[key])
    debounceRef.current[key] = setTimeout(() => {
      fetch(`${API}/session/config`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ [key]: num }),
      }).catch(() => {})
    }, 300)
  }

  const resetToRec = (key, rec) => handleChange(key, rec)

  const snrVal    = snr ?? 0
  const threshold = config?.snr_threshold ?? 2.8
  const passing   = snrVal >= threshold

  return (
    <div className="rounded-xl p-4 flex flex-col gap-4"
         style={{ background: 'rgba(10,15,30,0.97)', border: '1px solid rgba(255,255,255,0.12)', minWidth: 290 }}>

      {/* Header */}
      <div className="flex items-center justify-between">
        <span className="text-xs font-semibold text-white/50 uppercase tracking-widest">Signal Debug</span>
        <button onClick={onClose} className="text-white/30 hover:text-white/70 leading-none"><X size={15} /></button>
      </div>

      {/* Live SNR bar + ensemble breakdown */}
      <div className="flex flex-col gap-2">
        <div className="flex items-center justify-between text-xs">
          <span className="text-white/40">BVP SNR</span>
          <span className="font-mono" style={{ color: passing ? '#10b981' : '#ef4444' }}>
            {snrVal.toFixed(2)} {passing ? '▲ pass' : '▼ hold'}
          </span>
        </div>
        <SnrBar snr={snrVal} threshold={threshold} />
        <p className="text-white/20" style={{ fontSize: 9 }}>white tick = current threshold</p>

        {/* DL / POS / CHROM HR breakdown */}
        <div className="grid grid-cols-3 gap-1 mt-1">
          {[['DL', hr], ['POS', posHr], ['CHROM', chromHr]].map(([label, val]) => (
            <div key={label} className="rounded-lg px-2 py-1.5 text-center"
                 style={{ background: 'rgba(255,255,255,0.04)' }}>
              <div className="text-white/30 text-xs">{label}</div>
              <div className="font-mono text-sm text-white/80">{val ? val.toFixed(1) : '—'}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Divider */}
      <div style={{ borderTop: '1px solid rgba(255,255,255,0.06)' }} />

      {/* Parameter sliders */}
      {config && (
        <div className="flex flex-col gap-4">
          {PARAMS.map(({ key, label, min, max, step, unit, rec, recNote }) => {
            const val     = config[key] ?? rec
            const isAtRec = Math.abs(val - rec) < step * 0.5
            const decPlaces = step < 1 ? (step < 0.05 ? 2 : 1) : 0
            return (
              <div key={key} className="flex flex-col gap-1">
                <div className="flex items-center justify-between text-xs">
                  <span className="text-white/50">{label}</span>
                  <div className="flex items-center gap-2">
                    <span className="font-mono text-white/80">{val.toFixed(decPlaces)}{unit}</span>
                    {/* reset to recommended */}
                    <button
                      onClick={() => resetToRec(key, rec)}
                      title={`Reset to recommended (${rec}${unit}) — ${recNote}`}
                      className="text-white/25 hover:text-cyan-400 transition-colors"
                      style={{ fontSize: 11 }}
                    >
                      {isAtRec ? <Check size={11} /> : <RotateCcw size={11} />}
                    </button>
                  </div>
                </div>
                <input
                  type="range" min={min} max={max} step={step}
                  value={val}
                  onChange={e => handleChange(key, e.target.value)}
                  className="w-full accent-cyan-400"
                  style={{ height: 4 }}
                />
                <div className="flex justify-between items-center" style={{ fontSize: 9, color: 'rgba(255,255,255,0.18)' }}>
                  <span>{min}</span>
                  <span style={{ color: 'rgba(6,182,212,0.5)' }}>rec {rec}{unit} ({recNote})</span>
                  <span>{max}</span>
                </div>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
