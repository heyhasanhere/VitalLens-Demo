import React from 'react'
import { Check, AlertTriangle, RotateCw } from 'lucide-react'
import useVitalsStore from '../store/useVitalsStore'
import { API_BASE } from '../config'

function useUserMedia(videoRef, active) {
  React.useEffect(() => {
    if (!active) return
    let stream = null
    navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' }, audio: false })
      .then(s => {
        stream = s
        if (videoRef.current) {
          videoRef.current.srcObject = s
        }
      })
      .catch(err => console.error('[WebcamCapture] getUserMedia failed:', err))
    return () => {
      if (stream) stream.getTracks().forEach(t => t.stop())
    }
  }, [active]) // eslint-disable-line react-hooks/exhaustive-deps
}

// Lighting badge config
const lightingConfig = {
  Good:  { label: 'Good',  color: 'rgba(16,185,129,0.85)', dot: '#6ee7b7' },
  Mixed: { label: 'Mixed', color: 'rgba(245,158,11,0.85)',  dot: '#fcd34d' },
  Poor:  { label: 'Poor',  color: 'rgba(239,68,68,0.85)',   dot: '#fca5a5' },
}

export default function WebcamCapture({ isRecording, videoRef, inferenceMode }) {

  const lighting     = useVitalsStore(s => s.vitals.lighting)    || 'Good'
  const faceDetected = useVitalsStore(s => s.vitals.faceDetected)
  const lumStd       = useVitalsStore(s => s.vitals.lumStd)
  const faceBbox     = useVitalsStore(s => s.vitals.faceBbox)
  const cameraUrl    = useVitalsStore(s => s.cameraUrl)   // non-empty = MJPEG stream

  const isLocal = inferenceMode === 'local'
  useUserMedia(videoRef, isLocal)

  const [rotation, setRotation] = React.useState(0)
  const cycleRotation = () => setRotation(r => (r + 90) % 360)

  const badge = lightingConfig[lighting] || lightingConfig.Good

  const stabilityBadge = lumStd === null ? null
    : lumStd < 2   ? { label: `Stable σ${lumStd.toFixed(1)}`,   color: 'rgba(16,185,129,0.85)' }
    : lumStd < 5   ? { label: `Drifting σ${lumStd.toFixed(1)}`, color: 'rgba(245,158,11,0.85)' }
    :                { label: `Unstable σ${lumStd.toFixed(1)}`,  color: 'rgba(239,68,68,0.85)'  }

  // Compute bbox positions for SVG overlay.
  // Webcam component is mirrored (selfie-style) so we flip x.
  // MJPEG stream is unmirrored so we use raw coords.
  const bbox = faceBbox && faceDetected ? (() => {
    const xDisplay = cameraUrl ? faceBbox.x : 1 - (faceBbox.x + faceBbox.w)
    return {
      // SVG viewBox 0–100
      svgX: xDisplay * 100,
      svgY: faceBbox.y * 100,
      svgW: faceBbox.w * 100,
      svgH: faceBbox.h * 100,
      // CSS percentage for amplification overlay
      cssLeft:   `${xDisplay * 100}%`,
      cssTop:    `${faceBbox.y * 100}%`,
      cssWidth:  `${faceBbox.w * 100}%`,
      cssHeight: `${faceBbox.h * 100}%`,
    }
  })() : null

  // Corner accents helper for SVG
  const corners = bbox ? [
    [bbox.svgX,            bbox.svgY],
    [bbox.svgX + bbox.svgW, bbox.svgY],
    [bbox.svgX,            bbox.svgY + bbox.svgH],
    [bbox.svgX + bbox.svgW, bbox.svgY + bbox.svgH],
  ] : [
    [30, 18], [70, 18], [30, 70], [70, 70]
  ]

  return (
    <div className="relative w-full aspect-video rounded-2xl overflow-hidden"
         style={{ background: '#0a0f1e', border: '1px solid rgba(255,255,255,0.1)' }}>

      {/* ── Camera feed ─────────────────────────────────────────────────── */}
      {isLocal ? (
        <video
          ref={videoRef}
          autoPlay
          muted
          playsInline
          className="w-full h-full object-cover"
          style={{
            display: 'block',
            transform: [
              rotation ? `rotate(${rotation}deg)` : '',
              'scaleX(-1)',
            ].filter(Boolean).join(' '),
          }}
        />
      ) : (
        <img
          src={`${API_BASE}/video_feed`}
          alt="camera feed"
          className="w-full h-full object-cover"
          style={{
            display: 'block',
            transform: [
              rotation ? `rotate(${rotation}deg)` : '',
              !cameraUrl ? 'scaleX(-1)' : '',
            ].filter(Boolean).join(' ') || 'none',
          }}
        />
      )}

      {/* ── Face bounding box overlay (SVG) ─────────────────────────────── */}
      {faceDetected && (
        <svg
          className="absolute inset-0 w-full h-full pointer-events-none"
          viewBox="0 0 100 100"
          preserveAspectRatio="none"
          style={rotation ? { transform: `rotate(${rotation}deg)` } : undefined}
        >
          <rect
            x={bbox ? bbox.svgX : 30}
            y={bbox ? bbox.svgY : 18}
            width={bbox ? bbox.svgW : 40}
            height={bbox ? bbox.svgH : 52}
            fill="none"
            stroke="#00d4ff"
            strokeWidth="0.6"
            rx="1"
            strokeDasharray="4 2"
          />
          {corners.map(([cx, cy], i) => (
            <g key={i}>
              <line
                x1={cx} y1={cy}
                x2={cx + (i === 0 || i === 2 ? 4 : -4)} y2={cy}
                stroke="#00d4ff" strokeWidth="1.2" strokeLinecap="round"
              />
              <line
                x1={cx} y1={cy}
                x2={cx} y2={cy + (i < 2 ? 4 : -4)}
                stroke="#00d4ff" strokeWidth="1.2" strokeLinecap="round"
              />
            </g>
          ))}
        </svg>
      )}

      {/* ── Exposure stability badge (top-left, below REC) ──────────────── */}
      {stabilityBadge && !isRecording && (
        <div className="absolute top-3 left-3 px-2.5 py-1 rounded-full text-xs font-semibold text-white"
             style={{ background: stabilityBadge.color, backdropFilter: 'blur(8px)', boxShadow: '0 2px 12px rgba(0,0,0,0.4)' }}>
          {stabilityBadge.label}
        </div>
      )}
      {stabilityBadge && isRecording && (
        <div className="absolute top-10 left-3 px-2.5 py-1 rounded-full text-xs font-semibold text-white"
             style={{ background: stabilityBadge.color, backdropFilter: 'blur(8px)', boxShadow: '0 2px 12px rgba(0,0,0,0.4)' }}>
          {stabilityBadge.label}
        </div>
      )}

      {/* ── Lighting quality badge (top-right) ──────────────────────────── */}
      <div className="absolute top-3 right-3 flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-semibold"
           style={{
             background: badge.color,
             backdropFilter: 'blur(8px)',
             boxShadow: '0 2px 12px rgba(0,0,0,0.4)',
           }}>
        <span className="w-2 h-2 rounded-full flex-shrink-0" style={{ background: badge.dot }} />
        <span className="text-white">{badge.label}</span>
      </div>

      {/* ── Recording dot (top-left) ─────────────────────────────────────── */}
      {isRecording && (
        <div className="absolute top-3 left-3 flex items-center gap-2 px-2.5 py-1 rounded-full"
             style={{ background: 'rgba(0,0,0,0.55)', backdropFilter: 'blur(8px)' }}>
          <span className="recording-dot"></span>
          <span className="text-xs font-semibold text-red-400 tracking-wider">REC</span>
        </div>
      )}

      {/* ── Rotate button (bottom-right) ────────────────────────────────── */}
      <button
        onClick={cycleRotation}
        title="Rotate 90°"
        className="absolute bottom-3 right-3 flex items-center justify-center w-7 h-7 rounded-full"
        style={{
          background: rotation ? 'rgba(6,182,212,0.25)' : 'rgba(0,0,0,0.45)',
          border: `1px solid ${rotation ? 'rgba(6,182,212,0.6)' : 'rgba(255,255,255,0.18)'}`,
          backdropFilter: 'blur(8px)',
          color: rotation ? '#22d3ee' : 'rgba(255,255,255,0.5)',
          zIndex: 10,
        }}
      >
        <RotateCw size={13} />
      </button>

      {/* ── Face detection status (bottom) ──────────────────────────────── */}
      <div className="absolute bottom-0 left-0 right-0 flex items-center justify-center pb-3">
        <div className="px-3 py-1 rounded-full text-xs font-medium"
             style={{
               background: faceDetected
                 ? 'rgba(16,185,129,0.2)'
                 : 'rgba(239,68,68,0.2)',
               border: `1px solid ${faceDetected ? 'rgba(16,185,129,0.4)' : 'rgba(239,68,68,0.4)'}`,
               backdropFilter: 'blur(8px)',
               color: faceDetected ? '#6ee7b7' : '#fca5a5',
             }}>
          {faceDetected
            ? <><Check size={12} className="inline mr-1" />Face Detected</>
            : <><AlertTriangle size={12} className="inline mr-1" />No Face — please centre yourself</>}
        </div>
      </div>

      {/* ── Scan line animation overlay ──────────────────────────────────── */}
      {isRecording && (
        <div className="absolute inset-0 pointer-events-none"
             style={{
               background: 'linear-gradient(180deg, transparent 0%, rgba(0,212,255,0.03) 50%, transparent 100%)',
               animation: 'float 4s ease-in-out infinite',
             }} />
      )}
    </div>
  )
}
