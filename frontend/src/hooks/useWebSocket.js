import { useEffect, useRef } from 'react'
import useVitalsStore from '../store/useVitalsStore'
import { WS_BASE } from '../config'

export function useWebSocket(enabled = true) {
  const applyReading   = useVitalsStore(s => s.applyReading)
  const setWsConnected = useVitalsStore(s => s.setWsConnected)
  const cameraIndex    = useVitalsStore(s => s.cameraIndex)
  const cameraUrl      = useVitalsStore(s => s.cameraUrl)
  const selectedModel  = useVitalsStore(s => s.selectedModel)

  const wsRef = useRef(null)

  useEffect(() => {
    let active = true

    if (!enabled || cameraIndex === null) {
      if (wsRef.current) {
        wsRef.current.close()
        wsRef.current = null
      }
      return () => { active = false }
    }

    // Debounce: React StrictMode fires effects twice in dev. The 80ms delay lets
    // the first cleanup cancel before the socket opens, preventing duplicate connections.
    let ws
    const timer = setTimeout(() => {
      if (!active) return

      setWsConnected(null)

      try {
        const params = new URLSearchParams({ camera: cameraIndex })
        if (cameraUrl) params.set('camera_url', cameraUrl)
        if (selectedModel) params.set('model', selectedModel)
        ws = new WebSocket(`${WS_BASE}/ws/vitals?${params}`)
        wsRef.current = ws

        ws.onopen = () => {
          if (!active) return
          setWsConnected(true)
          console.log('[VitalLens] WebSocket connected')
        }

        ws.onmessage = (event) => {
          if (!active) return
          try {
            applyReading(JSON.parse(event.data))
          } catch (err) {
            console.warn('[VitalLens] Failed to parse WebSocket message', err)
          }
        }

        ws.onerror = () => {
          if (!active) return
          console.error('[VitalLens] WebSocket error — is the backend running?')
        }

        ws.onclose = () => {
          if (!active) return
          setWsConnected(false)
          console.info('[VitalLens] WebSocket closed')
        }
      } catch (err) {
        console.error('[VitalLens] Could not open WebSocket:', err)
      }
    }, 80)

    return () => {
      active = false
      clearTimeout(timer)
      if (ws && ws.readyState < 2) ws.close()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [enabled, cameraIndex, cameraUrl, selectedModel])
}
