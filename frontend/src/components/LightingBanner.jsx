import React from 'react'
import { AlertTriangle, X } from 'lucide-react'
import useVitalsStore from '../store/useVitalsStore'

export default function LightingBanner() {
  const consecutivePoor = useVitalsStore(s => s.consecutivePoorFrames)
  const dismissed       = useVitalsStore(s => s.lightingBannerDismissed)
  const dismiss         = useVitalsStore(s => s.dismissLightingBanner)

  if (dismissed || consecutivePoor < 3) return null

  return (
    <div
      id="lighting-warning-banner"
      className="flex items-center justify-between gap-4 px-5 py-3 rounded-xl text-sm font-medium animate-[slideDown_0.3s_ease-out]"
      style={{
        background: 'linear-gradient(90deg, rgba(245,158,11,0.15), rgba(234,179,8,0.1))',
        border: '1px solid rgba(245,158,11,0.4)',
        backdropFilter: 'blur(8px)',
      }}>
      <div className="flex items-center gap-3">
        <AlertTriangle size={17} className="flex-shrink-0 text-amber-400" />
        <span className="text-amber-200">
          Poor lighting detected — results may be inaccurate.{' '}
          <span className="text-amber-400/80">Try facing a window or turning on a light.</span>
        </span>
      </div>
      <button
        id="dismiss-lighting-banner"
        onClick={dismiss}
        className="flex-shrink-0 w-7 h-7 rounded-full flex items-center justify-center text-amber-300 hover:bg-amber-400/20 transition-colors"
        aria-label="Dismiss lighting warning">
        <X size={14} />
      </button>
    </div>
  )
}
