import React from 'react'

const COLORS = {
  Good:  '#10b981',
  Mixed: '#f59e0b',
  Poor:  '#ef4444',
}

const SIZE    = 140
const CX      = SIZE / 2
const CY      = SIZE / 2
const R_OUTER = 52
const R_INNER = 34
const MID_R   = (R_OUTER + R_INNER) / 2
const SW      = R_OUTER - R_INNER       // stroke width = ring thickness
const CIRC    = 2 * Math.PI * MID_R     // circumference at midpoint

function DonutChart({ data }) {
  const dominant = data.reduce((a, b) => a.value > b.value ? a : b)

  let cumulative = 0
  const segments = data.map(({ name, value }) => {
    const dash   = (value / 100) * CIRC
    // Start at 12 o'clock (SVG 0° = 3 o'clock, so subtract CIRC/4 to rotate back 90°)
    const offset = CIRC / 4 - cumulative
    cumulative  += dash
    return { name, value, dash, offset }
  })

  return (
    <svg width={SIZE} height={SIZE} viewBox={`0 0 ${SIZE} ${SIZE}`}>
      {/* Background ring */}
      <circle
        cx={CX} cy={CY} r={MID_R}
        fill="none"
        stroke="rgba(255,255,255,0.06)"
        strokeWidth={SW}
      />
      {/* Segments */}
      {segments.map(({ name, value, dash, offset }) => {
        const adjDash = Math.max(0, dash - (data.length > 1 ? 1.5 : 0))
        return (
          <circle
            key={name}
            cx={CX} cy={CY} r={MID_R}
            fill="none"
            stroke={COLORS[name] || '#94a3b8'}
            strokeWidth={SW}
            strokeDasharray={`${adjDash} ${Math.max(0, CIRC - adjDash)}`}
            strokeDashoffset={offset}
            strokeLinecap="butt"
          />
        )
      })}
      {/* Center: dominant % */}
      <text
        x={CX} y={CY - 5}
        textAnchor="middle" dominantBaseline="middle"
        fill="white" fontSize="20" fontWeight="800" fontFamily="ui-monospace,monospace"
      >
        {dominant.value.toFixed(0)}%
      </text>
      <text
        x={CX} y={CY + 13}
        textAnchor="middle" dominantBaseline="middle"
        fill="rgba(255,255,255,0.4)" fontSize="9" fontWeight="600"
        style={{ letterSpacing: '0.08em', textTransform: 'uppercase' }}
      >
        {dominant.name}
      </text>
    </svg>
  )
}

export default function LightingBreakdown({ breakdown }) {
  if (!breakdown) return null

  const data = Object.entries(breakdown)
    .filter(([, val]) => val > 0)
    .map(([name, val]) => ({ name, value: parseFloat((val * 100).toFixed(1)) }))

  if (data.length === 0) return null

  return (
    <div className="glass-card p-5 h-full flex flex-col">
      <h2 className="text-sm font-semibold text-white/50 uppercase tracking-widest mb-4">
        Lighting Quality Breakdown
      </h2>

      <div className="flex-1 flex flex-col items-center justify-center gap-5">
        <DonutChart data={data} />

        {/* Legend */}
        <div className="flex flex-col gap-2 w-full max-w-[160px]">
          {data.map(({ name, value }) => (
            <div key={name} className="flex items-center gap-2.5">
              <span className="w-2.5 h-2.5 rounded-full flex-shrink-0"
                    style={{ background: COLORS[name] }} />
              <span className="text-sm text-white/60 flex-1">{name}</span>
              <span className="text-sm font-bold text-white tabular-nums">{value}%</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
