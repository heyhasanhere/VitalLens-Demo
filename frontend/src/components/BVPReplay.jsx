import React, { useMemo } from 'react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer,
} from 'recharts'

export default function BVPReplay({ bvpSeries }) {
  const chartData = useMemo(() => {
    if (!bvpSeries || bvpSeries.length === 0) return []
    return bvpSeries.map((v, i) => ({
      t: parseFloat(((i / bvpSeries.length) * (bvpSeries.length / 32)).toFixed(2)),
      v: parseFloat((+v).toFixed(4)),
    }))
  }, [bvpSeries])

  if (chartData.length === 0) return null

  return (
    <div className="glass-card p-5">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-sm font-semibold text-white/50 uppercase tracking-widest">Full Session BVP</h2>
        <span className="text-xs text-white/30">{chartData.length} samples</span>
      </div>
      <ResponsiveContainer width="100%" height={180}>
        <LineChart data={chartData} margin={{ top: 4, right: 8, left: -20, bottom: 4 }}>
          <defs>
            <linearGradient id="replayGradient" x1="0" y1="0" x2="1" y2="0">
              <stop offset="0%"   stopColor="#818cf8" stopOpacity={0.5} />
              <stop offset="50%"  stopColor="#22d3ee" stopOpacity={1}   />
              <stop offset="100%" stopColor="#06b6d4" stopOpacity={0.6} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis
            dataKey="t"
            type="number"
            tickFormatter={v => `${v.toFixed(0)}s`}
            tick={{ fontSize: 10 }}
          />
          <YAxis
            domain={[-1.5, 1.5]}
            tick={{ fontSize: 10 }}
            tickFormatter={v => v.toFixed(1)}
          />
          <Tooltip
            contentStyle={{ background: 'rgba(15,23,42,0.9)', border: '1px solid rgba(6,182,212,0.3)', borderRadius: 8, fontSize: 11 }}
            labelFormatter={v => `${v}s`}
            formatter={v => [v.toFixed(3), 'BVP']}
          />
          <Line
            type="monotone"
            dataKey="v"
            stroke="url(#replayGradient)"
            strokeWidth={1.5}
            dot={false}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
