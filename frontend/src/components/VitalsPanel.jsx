import React from 'react'
import { Wind, Activity, Gauge, Eye } from 'lucide-react'
import MetricCard from './MetricCard'
import useVitalsStore, { WARMUP_MS } from '../store/useVitalsStore'

const METRICS = [
  {
    id:          'metric-card-hr',
    label:       'Heart Rate',
    metricKey:   'hr',
    unit:        'BPM',
    icon:        null,  // MetricCard renders a custom animated HeartIcon for HR
    normalRange: '60–100 BPM',
    storeKey:    'hr',
    prevKey:     'prevHr',
  },
  {
    id:          'metric-card-br',
    label:       'Breathing Rate',
    metricKey:   'br',
    unit:        'br/min',
    icon:        Wind,
    normalRange: '12–20 br/min',
    storeKey:    'br',
    prevKey:     'prevBr',
  },
  {
    id:          'metric-card-hrv',
    label:       'HRV (RMSSD)',
    metricKey:   'hrv',
    unit:        'ms',
    icon:        Activity,
    normalRange: '20–50 ms',
    storeKey:    'hrv',
    prevKey:     'prevHrv',
  },
  {
    id:          'metric-card-stress',
    label:       'Stress Index',
    metricKey:   'stress',
    unit:        '/100',
    icon:        Gauge,
    normalRange: '< 30 = low',
    storeKey:    'stress',
    prevKey:     'prevStress',
  },
  {
    id:          'metric-card-blink',
    label:       'Blink Rate',
    metricKey:   'blinkRate',
    unit:        '/min',
    icon:        Eye,
    normalRange: '8–25/min',
    storeKey:    'blinkRate',
    prevKey:     'prevBlinkRate',
  },
]

const SNR_THRESHOLD = 3.5  // matches backend _BVP_SNR_THRESHOLD

export default function VitalsPanel({ wsConnected }) {
  const vitals     = useVitalsStore(s => s.vitals)
  const startTime  = useVitalsStore(s => s.session.startTime)
  const isActive   = useVitalsStore(s => s.session.isActive)

  const [warmupLeft, setWarmupLeft] = React.useState(0)
  React.useEffect(() => {
    if (!isActive || !startTime) { setWarmupLeft(0); return }
    const update = () => {
      const left = Math.max(0, Math.ceil((WARMUP_MS - (Date.now() - startTime)) / 1000))
      setWarmupLeft(left)
    }
    update()
    const t = setInterval(update, 500)
    return () => clearInterval(t)
  }, [isActive, startTime])

  // Dim all cards when SNR is below threshold — signal is noisy
  const lowConfidence = vitals.snr !== null && vitals.snr > 0 && vitals.snr < SNR_THRESHOLD

  return (
    <div className="flex flex-col gap-4 h-full">
      {/* Connection status badge */}
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h2 className="text-sm font-semibold text-white/50 uppercase tracking-widest">Live Vitals</h2>
        <div className="flex items-center gap-2">
          {warmupLeft > 0 && (
            <span className="text-xs px-2 py-0.5 rounded-full"
                  style={{ background: 'rgba(99,102,241,0.12)', border: '1px solid rgba(99,102,241,0.35)', color: '#a5b4fc' }}>
              Calibrating {warmupLeft}s
            </span>
          )}
          {lowConfidence && warmupLeft === 0 && (
            <span className="text-xs px-2 py-0.5 rounded-full"
                  style={{ background: 'rgba(239,68,68,0.12)', border: '1px solid rgba(239,68,68,0.35)', color: '#fca5a5' }}>
              Low signal
            </span>
          )}
          <div className="flex items-center gap-2 text-xs px-3 py-1 rounded-full"
               style={{
                 background: wsConnected === true ? 'rgba(16,185,129,0.12)' : wsConnected === false ? 'rgba(245,158,11,0.12)' : 'rgba(99,102,241,0.12)',
                 border: `1px solid ${wsConnected === true ? 'rgba(16,185,129,0.35)' : wsConnected === false ? 'rgba(245,158,11,0.35)' : 'rgba(99,102,241,0.35)'}`,
                 color: wsConnected === true ? '#6ee7b7' : wsConnected === false ? '#fcd34d' : '#a5b4fc',
               }}>
            <span className="w-1.5 h-1.5 rounded-full animate-pulse"
                  style={{ background: wsConnected === true ? '#10b981' : wsConnected === false ? '#f59e0b' : '#6366f1' }} />
            {wsConnected === true ? 'Live' : wsConnected === false ? 'Disconnected' : 'Connecting…'}
          </div>
        </div>
      </div>

      {/* Metric cards grid — 5 cards: 2×2 + blink rate spanning full width */}
      <div className="grid grid-cols-2 gap-3 flex-1">
        {METRICS.map((m, i) => (
          <MetricCard
            key={m.id}
            id={m.id}
            label={m.label}
            metricKey={m.metricKey}
            value={vitals[m.storeKey]}
            prevValue={vitals[m.prevKey]}
            unit={m.unit}
            icon={m.icon}
            normalRange={m.normalRange}
            lowConfidence={lowConfidence}
            className={i === METRICS.length - 1 ? 'col-span-2' : ''}
          />
        ))}
      </div>
    </div>
  )
}
