import React, { useRef, useEffect } from 'react'

// ── Normal ranges for colour coding ───────────────────────────────────────────
const NORMAL = {
  hr:        [60, 100],
  br:        [12, 20],
  hrv:       [20, 50],
  stress:    [0, 30],
  blinkRate: [8, 25],
}

function getStatus(key, value) {
  if (value == null || value <= 0) return 'neutral'
  if (!NORMAL[key]) return 'neutral'
  const [lo, hi] = NORMAL[key]
  if (key === 'stress') {
    if (value < 30) return 'green'
    if (value < 60) return 'amber'
    return 'red'
  }
  if (key === 'blinkRate') {
    if (value >= lo && value <= hi) return 'green'
    if (value < lo) return 'red'    // fatigue
    return 'amber'                  // stress/irritation
  }
  if (value >= lo && value <= hi) return 'green'
  const margin = (hi - lo) * 0.15
  if (value >= lo - margin && value <= hi + margin) return 'amber'
  return 'red'
}

function getTrend(current, prev) {
  if (prev == null || current == null) return '→'
  const diff = current - prev
  if (Math.abs(diff) < 0.5) return '→'
  return diff > 0 ? '↑' : '↓'
}

const statusStyles = {
  green:   { borderClass: 'metric-green', valueColor: '#10b981', trendColor: '#6ee7b7' },
  amber:   { borderClass: 'metric-amber', valueColor: '#f59e0b', trendColor: '#fcd34d' },
  red:     { borderClass: 'metric-red',   valueColor: '#ef4444', trendColor: '#fca5a5' },
  neutral: { borderClass: 'metric-neutral', valueColor: 'rgba(255,255,255,0.5)', trendColor: 'rgba(255,255,255,0.3)' },
  dim:     { borderClass: 'metric-neutral', valueColor: 'rgba(255,255,255,0.22)', trendColor: 'rgba(255,255,255,0.15)' },
}

// HR zone → heart color (separate from the card border status)
function hrZoneColor(hr) {
  if (hr == null) return 'rgba(255,255,255,0.25)'
  if (hr <= 80)   return '#10b981'   // green — normal resting
  if (hr <= 100)  return '#f59e0b'   // orange — elevated
  return '#ef4444'                   // red — high / tachycardia
}

const HEART_D = "M16 28.4C15.5 28 2 19.2 2 10.6 2 6.4 5.4 3 9.5 3c2.6 0 5 1.4 6.5 3.6C17.5 4.4 19.9 3 22.5 3 26.6 3 30 6.4 30 10.6 30 19.2 16.5 28 16 28.4z"

function HeartIcon({ color, beatDuration }) {
  const beatStyle = beatDuration
    ? { animation: `heartbeat ${beatDuration}s ease-in-out infinite`, transformOrigin: '16px 15px' }
    : {}

  return (
    <svg
      viewBox="0 0 32 30"
      width="22"
      height="20"
      xmlns="http://www.w3.org/2000/svg"
      style={{ display: 'block' }}
    >
      <defs>
        <radialGradient id="heartGrad" cx="40%" cy="30%" r="55%">
          <stop offset="0%" stopColor="white" stopOpacity="0.4" />
          <stop offset="100%" stopColor={color} stopOpacity="0" />
        </radialGradient>
      </defs>
      {/* Main heart — scales with heartbeat + specular highlight overlay */}
      <g style={beatStyle}>
        <path d={HEART_D} fill={color} />
        <path d={HEART_D} fill="url(#heartGrad)" />
      </g>
    </svg>
  )
}

export default function MetricCard({ id, label, metricKey, value, prevValue, unit, icon, normalRange, lowConfidence = false, rhythm = null, className = '' }) {
  const status  = getStatus(metricKey, value)
  const trend   = getTrend(value, prevValue)
  const styles  = statusStyles[status]

  const prevVal = useRef(null)
  const numRef  = useRef(null)

  // Trigger number-slide animation on value change
  useEffect(() => {
    if (value !== prevVal.current && numRef.current && value != null) {
      numRef.current.classList.remove('number-update')
      void numRef.current.offsetWidth // reflow
      numRef.current.classList.add('number-update')
      prevVal.current = value
    }
  }, [value])

  const displayValue = (value != null && value > 0) ? value.toFixed(1) : '—'

  // Heart-specific props
  const isHr        = metricKey === 'hr'
  const heartColor  = isHr ? (lowConfidence ? 'rgba(255,255,255,0.2)' : hrZoneColor(value)) : null
  const beatDuration = (isHr && !lowConfidence && value != null && value > 0)
    ? parseFloat((60 / value).toFixed(3))
    : null

  return (
    <div
      id={id}
      className={`glass-card p-5 flex flex-col gap-2 transition-all duration-500 ${styles.borderClass} ${className}`}
    >
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          {isHr ? (
            <HeartIcon color={heartColor} beatDuration={beatDuration} />
          ) : icon ? (
            React.createElement(icon, { size: 18, style: { color: 'rgba(255,255,255,0.45)', flexShrink: 0 } })
          ) : null}
          <span className="text-xs font-semibold text-white/50 uppercase tracking-wider">{label}</span>
        </div>
      </div>

      {/* Value */}
      <div className="flex items-baseline gap-2">
        <span
          ref={numRef}
          className="text-4xl font-black tabular-nums tracking-tighter transition-colors duration-500"
          style={{ color: styles.valueColor }}>
          {displayValue}
        </span>
        <span className="text-sm text-white/40 font-medium">{unit}</span>
      </div>

      {/* Normal range hint */}
      <div className="text-xs text-white/30 mt-auto">
        Normal: <span className="text-white/50">{normalRange}</span>
      </div>

    </div>
  )
}
