import React from 'react'
import { useNavigate } from 'react-router-dom'
import { Camera, Heart, Brain, Lock, Wind, Activity, Gauge, Smartphone, Play, Cpu } from 'lucide-react'
import useVitalsStore from '../store/useVitalsStore'
import { API_BASE } from '../config'

// ── Feature tiles ─────────────────────────────────────────────────────────────
const features = [
  { icon: <Camera size={22} />,  title: 'Webcam-Based',   desc: 'Uses your existing camera — no wearables, sensors, or contact required.' },
  { icon: <Heart size={22} />,   title: 'rPPG Technology', desc: 'Detects subtle colour changes in your skin caused by blood pulsing through vessels.' },
  { icon: <Brain size={22} />,   title: 'AI-Powered',      desc: 'EfficientPhys deep-learning model delivers clinical-grade accuracy in real time.' },
  { icon: <Lock size={22} />,    title: 'Fully Private',   desc: 'Processing happens locally. No video is stored or sent anywhere.' },
]

// ── Stat preview ──────────────────────────────────────────────────────────────
const previewStats = [
  { label: 'Heart Rate',     unit: 'BPM',    icon: <Heart size={28} />    },
  { label: 'Breathing Rate', unit: 'br/min', icon: <Wind size={28} />     },
  { label: 'HRV (RMSSD)',    unit: 'ms',     icon: <Activity size={28} /> },
  { label: 'Stress Index',   unit: '0–100',  icon: <Gauge size={28} />    },
]

export default function HomePage() {
  const navigate        = useNavigate()
  const startSession    = useVitalsStore(s => s.startSession)
  const cameraIndex     = useVitalsStore(s => s.cameraIndex)
  const setCameraIndex  = useVitalsStore(s => s.setCameraIndex)  // for manual picker
  const cameraUrl        = useVitalsStore(s => s.cameraUrl)
  const setCameraUrl     = useVitalsStore(s => s.setCameraUrl)
  const selectedModel    = useVitalsStore(s => s.selectedModel)
  const setSelectedModel = useVitalsStore(s => s.setSelectedModel)
  const [cameras, setCameras] = React.useState([])

  // Display value is IP:port without the http:// prefix and /video suffix
  const phoneIpDisplay = cameraUrl
    ? cameraUrl.replace(/^https?:\/\//, '').replace(/\/video$/, '')
    : ''

  const handlePhoneIpChange = (e) => {
    const val = e.target.value.trim()
    setCameraUrl(val ? `http://${val}/video` : '')
  }

  React.useEffect(() => {
    fetch(`${API_BASE}/cameras`)
      .then(r => r.ok ? r.json() : [])
      .then(list => { if (list.length > 0) setCameras(list) })
      .catch(() => {})
  }, [])

  const handleStart = () => {
    startSession()
    navigate('/monitor')
  }

  return (
    <div className="min-h-screen flex flex-col">

      {/* ── Nav bar ─────────────────────────────────────────────────────── */}
      <nav className="fixed top-0 left-0 right-0 z-50 flex items-center justify-between px-8 py-4"
           style={{ background: 'rgba(10,15,30,0.7)', backdropFilter: 'blur(16px)', borderBottom: '1px solid rgba(255,255,255,0.06)' }}>
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg flex items-center justify-center"
               style={{ background: 'linear-gradient(135deg, #06b6d4, #818cf8)' }}>
            <span className="text-white font-bold text-sm">VL</span>
          </div>
          <span className="font-semibold text-white tracking-tight">VitalLens</span>
        </div>
        <div className="hidden md:flex items-center gap-2 text-xs text-white/40 border border-white/10 rounded-full px-3 py-1">
          <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse"></span>
          Research Preview
        </div>
      </nav>

      {/* ── Hero ────────────────────────────────────────────────────────── */}
      <section className="flex-1 flex flex-col items-center justify-center px-4 pt-24 pb-16 text-center relative overflow-hidden">

        {/* Background orbs */}
        <div className="absolute inset-0 pointer-events-none">
          <div className="absolute top-1/4 left-1/4 w-96 h-96 rounded-full opacity-10"
               style={{ background: 'radial-gradient(circle, #06b6d4, transparent)', filter: 'blur(60px)', animation: 'float 8s ease-in-out infinite' }} />
          <div className="absolute bottom-1/4 right-1/4 w-80 h-80 rounded-full opacity-8"
               style={{ background: 'radial-gradient(circle, #818cf8, transparent)', filter: 'blur(60px)', animation: 'float 10s ease-in-out infinite reverse' }} />
        </div>

        {/* Badge */}
        <div className="inline-flex items-center gap-2 mb-6 px-4 py-1.5 rounded-full text-xs font-medium"
             style={{ background: 'rgba(6,182,212,0.1)', border: '1px solid rgba(6,182,212,0.3)', color: '#22d3ee' }}>
          <span>✦</span> Contactless · Real-Time · AI-Powered
        </div>

        {/* Heading */}
        <h1 className="text-5xl md:text-7xl font-black tracking-tight mb-4 leading-tight">
          <span className="text-gradient">VitalLens</span>
        </h1>
        <p className="text-xl md:text-2xl font-light text-white/60 mb-4 tracking-wide">
          Contactless Vitals Monitoring
        </p>

        {/* Description */}
        <p className="max-w-2xl text-base text-white/50 mb-10 leading-relaxed">
          VitalLens uses your webcam and remote photoplethysmography (rPPG) to measure your heart rate,
          breathing rate, heart rate variability, and stress index - all in real time, with zero physical contact.
          Sit in front of your camera and let AI do the rest.
        </p>

        {/* Camera picker — only shown when backend is up and >1 camera found */}
        {cameras.length > 0 && (
          <div className="flex items-center gap-3 mb-3 px-4 py-3 rounded-xl"
               style={{ background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.08)' }}>
            <Camera size={13} className="text-white/40 flex-shrink-0" />
            <select
              value={cameraIndex}
              onChange={e => setCameraIndex(Number(e.target.value))}
              className="text-xs rounded-lg px-3 py-1.5 text-white/80 outline-none cursor-pointer"
              style={{ background: 'rgba(255,255,255,0.06)', border: '1px solid rgba(255,255,255,0.1)' }}
            >
              {cameras.map(cam => (
                <option key={cam.index} value={cam.index} style={{ background: '#0f172a' }}>
                  {cam.label}
                </option>
              ))}
            </select>
            <span className="text-xs text-white/25">ignored when Phone IP is set</span>
          </div>
        )}

        {/* DroidCam / phone MJPEG stream input */}
        <div className="flex items-center gap-3 mb-6 px-4 py-3 rounded-xl"
             style={{
               background: cameraUrl ? 'rgba(6,182,212,0.06)' : 'rgba(255,255,255,0.04)',
               border: `1px solid ${cameraUrl ? 'rgba(6,182,212,0.35)' : 'rgba(255,255,255,0.08)'}`,
             }}>
          <Smartphone size={13} className="text-white/40 flex-shrink-0" />
          <input
            type="text"
            placeholder="192.168.x.x:4747"
            value={phoneIpDisplay}
            onChange={handlePhoneIpChange}
            className="text-xs rounded-lg px-3 py-1.5 text-white/80 outline-none"
            style={{
              background: 'rgba(255,255,255,0.06)',
              border: `1px solid ${cameraUrl ? 'rgba(6,182,212,0.4)' : 'rgba(255,255,255,0.1)'}`,
              width: '160px',
            }}
          />
          <span className="text-xs" style={{ color: cameraUrl ? '#22d3ee' : '#ffffff40' }}>
            {cameraUrl ? `→ ${cameraUrl}` : 'DroidCam WiFi stream (leave blank for webcam)'}
          </span>
        </div>

        {/* Model selector */}
        <div className="flex items-center gap-1.5 mb-6 p-1 rounded-xl"
             style={{ background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.08)' }}>
          {[
            { model: 'factorizephys',      label: 'FactorizePhys'      },
            { model: 'factorizephys_ibvp', label: 'FactorizePhys-iBVP' },
            { model: 'efficientphys',      label: 'EfficientPhys'      },
            { model: 'physnet',            label: 'PhysNet'            },
            { model: 'physformer',         label: 'PhysFormer'         },
          ].map(({ model, label }) => {
            const active = selectedModel === model
            return (
              <button
                key={model}
                onClick={() => setSelectedModel(model)}
                className="flex items-center justify-center gap-1.5 py-2 px-3 rounded-lg text-xs font-semibold transition-all duration-200"
                style={{
                  background: active ? 'linear-gradient(135deg, rgba(129,140,248,0.2), rgba(6,182,212,0.15))' : 'transparent',
                  border: active ? '1px solid rgba(129,140,248,0.45)' : '1px solid transparent',
                  color: active ? '#c7d2fe' : 'rgba(255,255,255,0.35)',
                }}
              >
                <Cpu size={11} />{label}
              </button>
            )
          })}
        </div>

        {/* CTA */}
        <button id="start-session-btn" onClick={handleStart} className="btn-primary text-lg px-10 py-5 mb-4">
          <Play size={16} className="inline" /> Start Session
        </button>
      </section>

      {/* ── Preview metric strip ─────────────────────────────────────────── */}
      <section className="px-4 pb-12">
        <div className="max-w-4xl mx-auto grid grid-cols-2 md:grid-cols-4 gap-4">
          {previewStats.map(({ label, unit, icon }) => (
            <div key={label} className="glass-card p-5 text-center hover:border-cyan-500/30 transition-colors duration-300">
              <div className="mb-2 text-white/40 flex justify-center">{icon}</div>
              <div className="text-2xl font-bold text-white/20 mb-1">—</div>
              <div className="text-xs text-white/50 font-medium">{label}</div>
              <div className="text-xs text-cyan-400/70">{unit}</div>
            </div>
          ))}
        </div>
      </section>

      {/* ── Feature grid ────────────────────────────────────────────────── */}
      <section className="px-4 pb-20">
        <div className="max-w-4xl mx-auto">
          <h2 className="text-center text-sm font-semibold text-white/30 uppercase tracking-widest mb-8">
            How It Works
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {features.map(({ icon, title, desc }) => (
              <div key={title} className="glass-card p-6 flex items-start gap-4 group hover:border-white/20 transition-all duration-300">
                <div className="flex-shrink-0 mt-0.5 text-white/60">{icon}</div>
                <div>
                  <h3 className="text-sm font-semibold text-white mb-1 group-hover:text-cyan-300 transition-colors">{title}</h3>
                  <p className="text-xs text-white/50 leading-relaxed">{desc}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── Footer ────────────────────────────────────────────────────────── */}
      <footer className="text-center py-6 text-xs text-white/20 border-t border-white/5">
        VitalLens — University of Technology Sydney · Deep Learning Research Project
      </footer>
    </div>
  )
}
