import React, { useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { Play, CheckCircle, Download } from 'lucide-react'
import useVitalsStore from '../store/useVitalsStore'
import SessionStats from '../components/SessionStats'
import BVPReplay from '../components/BVPReplay'
import LightingBreakdown from '../components/LightingBreakdown'

// CSV export helper
function exportCSV(summary, readings) {
  const rows = [
    ['Timestamp', 'HR (BPM)', 'BR (br/min)', 'HRV (ms)', 'Stress (0-100)', 'Lighting'],
    ...(readings || []).map(r => [
      r.timestamp ? new Date(r.timestamp * 1000).toISOString() : '',
      r.hr?.toFixed(2) ?? '',
      r.br?.toFixed(2) ?? '',
      r.hrv?.toFixed(2) ?? '',
      r.stress?.toFixed(2) ?? '',
      r.lighting ?? '',
    ]),
  ]

  // Add summary footer
  rows.push([])
  rows.push(['Summary'])
  rows.push(['Duration (s)', summary.duration_seconds ?? ''])
  rows.push(['Avg HR', summary.avg_hr?.toFixed(2) ?? ''])
  rows.push(['Avg BR',  summary.avg_br?.toFixed(2)  ?? ''])
  rows.push(['Avg HRV', summary.avg_hrv?.toFixed(2) ?? ''])
  rows.push(['Avg Stress', summary.avg_stress?.toFixed(2) ?? ''])
  rows.push(['Min HR', summary.min_hr?.toFixed(2) ?? ''])
  rows.push(['Max HR', summary.max_hr?.toFixed(2) ?? ''])

  const csv  = rows.map(r => r.join(',')).join('\n')
  const blob = new Blob([csv], { type: 'text/csv' })
  const url  = URL.createObjectURL(blob)
  const a    = document.createElement('a')
  a.href     = url
  a.download = `vitallens-session-${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.csv`
  a.click()
  URL.revokeObjectURL(url)
}

export default function SummaryPage() {
  const navigate     = useNavigate()
  const summary      = useVitalsStore(s => s.session.summary)
  const readings     = useVitalsStore(s => s.session.readings)
  const resetSession = useVitalsStore(s => s.resetSession)
  const startSession = useVitalsStore(s => s.startSession)

  // Guard — if no summary, send back to home
  useEffect(() => {
    if (!summary) navigate('/')
  }, [summary, navigate])

  if (!summary) return null

  const handleNewSession = () => {
    resetSession()
    startSession()
    navigate('/monitor')
  }

  const handleHome = () => {
    resetSession()
    navigate('/')
  }

  return (
    <div className="min-h-screen flex flex-col">

      {/* ── Nav ──────────────────────────────────────────────────────────── */}
      <header className="sticky top-0 z-40 flex items-center justify-between px-6 py-4"
              style={{ background: 'rgba(10,15,30,0.85)', backdropFilter: 'blur(16px)', borderBottom: '1px solid rgba(255,255,255,0.06)' }}>
        <div className="flex items-center gap-3">
          <div className="w-7 h-7 rounded-lg flex items-center justify-center flex-shrink-0"
               style={{ background: 'linear-gradient(135deg, #06b6d4, #818cf8)' }}>
            <span className="text-white font-bold text-xs">VL</span>
          </div>
          <span className="font-semibold text-white text-sm">VitalLens</span>
          <span className="text-white/20 text-sm hidden sm:block">·</span>
          <span className="text-xs text-white/40 hidden sm:block">Session Summary</span>
        </div>
        <div className="flex items-center gap-3">
          <button
            id="export-csv-btn"
            onClick={() => exportCSV(summary, readings)}
            className="btn-secondary text-xs">
            <Download size={13} className="inline mr-1" />Export CSV
          </button>
          <button
            id="new-session-btn"
            onClick={handleNewSession}
            className="btn-primary text-xs px-5 py-2.5">
            <Play size={13} className="inline mr-1" />New Session
          </button>
        </div>
      </header>

      <main className="flex-1 px-6 py-8 max-w-5xl mx-auto w-full flex flex-col gap-8">

        {/* Heading */}
        <div className="text-center">
          <div className="inline-flex items-center gap-2 mb-3 px-4 py-1.5 rounded-full text-xs font-medium"
               style={{ background: 'rgba(16,185,129,0.1)', border: '1px solid rgba(16,185,129,0.3)', color: '#6ee7b7' }}>
            <CheckCircle size={13} className="inline mr-1.5" />Session Complete
          </div>
          <h1 className="text-3xl font-black text-white mb-2">Your Vitals Report</h1>
          <p className="text-sm text-white/40">Here's a breakdown of your health metrics from this session.</p>
        </div>

        {/* Stats */}
        <SessionStats summary={summary} />

        {/* Charts row */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2">
            <BVPReplay bvpSeries={summary.bvp_series} />
          </div>
          <div>
            <LightingBreakdown breakdown={summary.lighting_breakdown} />
          </div>
        </div>

        {/* Footer actions */}
        <div className="flex items-center justify-center gap-4 pb-6">
          <button onClick={handleHome} className="btn-secondary">
            ← Back to Home
          </button>
          <button id="new-session-btn-bottom" onClick={handleNewSession} className="btn-primary">
            <Play size={13} className="inline mr-1" />Start New Session
          </button>
        </div>

      </main>
    </div>
  )
}
