import React, { useMemo } from 'react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceLine,
} from 'recharts'
import useVitalsStore from '../store/useVitalsStore'

const CustomTooltip = ({ active, payload }) => {
  if (!active || !payload?.length) return null
  return (
    <div className="px-2 py-1 rounded text-xs"
         style={{ background: 'rgba(15,23,42,0.9)', border: '1px solid rgba(6,182,212,0.3)', color: '#22d3ee' }}>
      {payload[0]?.value?.toFixed(3)}
    </div>
  )
}

export default function BVPWaveform({ autoScale = false }) {
  const bvpWindow = useVitalsStore(s => s.vitals.bvpWindow)

  const chartData = useMemo(() => {
    if (!bvpWindow || bvpWindow.length === 0) {
      return Array.from({ length: 64 }, (_, i) => ({ t: i, v: 0 }))
    }
    // Take last 320 samples and display with time axis
    const slice = bvpWindow.slice(-320)
    return slice.map((v, i) => ({
      t: parseFloat(((i / slice.length) * 10).toFixed(2)),
      v: parseFloat(v.toFixed(4)),
    }))
  }, [bvpWindow])

  return (
    <div className="glass-card p-5">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-sm font-semibold text-white/50 uppercase tracking-widest">
          BVP Waveform
        </h2>
        <span className="text-xs text-white/30">Last 10 seconds</span>
      </div>

      <ResponsiveContainer width="100%" height={autoScale ? 200 : 140}>
        <LineChart data={chartData} margin={{ top: 4, right: 8, left: -20, bottom: 4 }}>
          <defs>
            <linearGradient id="bvpGradient" x1="0" y1="0" x2="1" y2="0">
              <stop offset="0%"   stopColor="#06b6d4" stopOpacity={0.3} />
              <stop offset="50%"  stopColor="#22d3ee" stopOpacity={1}   />
              <stop offset="100%" stopColor="#818cf8" stopOpacity={0.6} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis
            dataKey="t"
            type="number"
            domain={[0, 10]}
            tickCount={6}
            tickFormatter={v => `${v}s`}
            tick={{ fontSize: 10 }}
          />
          <YAxis
            domain={autoScale ? ['auto', 'auto'] : [-1.2, 1.2]}
            tickCount={5}
            tick={{ fontSize: 10 }}
            tickFormatter={v => v.toFixed(3)}
          />
          <Tooltip content={<CustomTooltip />} />
          <ReferenceLine y={0} stroke="rgba(255,255,255,0.08)" />
          <Line
            type="monotone"
            dataKey="v"
            stroke="url(#bvpGradient)"
            strokeWidth={2}
            dot={false}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
