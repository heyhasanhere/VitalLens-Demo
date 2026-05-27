import { useEffect, useRef, useState } from 'react'
import useVitalsStore from '../store/useVitalsStore'
import { VitalsEngine } from '../inference/VitalsEngine.js'

const FRAME_INTERVAL_MS = 1000 / 20  // 20 fps processing

export function useLocalInference(videoRef, enabled) {
  const applyReading   = useVitalsStore(s => s.applyReading)
  const setWsConnected = useVitalsStore(s => s.setWsConnected)
  const selectedModel  = useVitalsStore(s => s.selectedModel)
  const [engineReady, setEngineReady] = useState(false)

  useEffect(() => {
    if (!enabled) return

    let active  = true
    let timerId = null
    const engine = new VitalsEngine(selectedModel)

    setWsConnected(null)  // show "connecting" indicator

    engine.init()
      .then(() => {
        if (!active) { engine.dispose(); return }
        setWsConnected(true)
        setEngineReady(true)

        timerId = setInterval(() => {
          const video = videoRef.current
          if (!video || video.readyState < 2 || !video.videoWidth) return
          engine.processFrame(video)
          applyReading(engine.getState())
        }, FRAME_INTERVAL_MS)
      })
      .catch(err => {
        console.error('[LocalInference] init failed:', err)
        if (active) setWsConnected(false)
      })

    return () => {
      active = false
      clearInterval(timerId)
      engine.dispose()
      setEngineReady(false)
    }
  }, [enabled, selectedModel])   // eslint-disable-line react-hooks/exhaustive-deps

  return { engineReady }
}
