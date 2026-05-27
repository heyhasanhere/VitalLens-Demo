import React from 'react'
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import HomePage    from './pages/HomePage'
import MonitorPage from './pages/MonitorPage'
import SummaryPage from './pages/SummaryPage'
import useVitalsStore from './store/useVitalsStore'
import { API_BASE } from './config'

export default function App() {
  const setCameraIndex = useVitalsStore(s => s.setCameraIndex)

  React.useEffect(() => {
    fetch(`${API_BASE}/cameras`)
      .then(r => r.ok ? r.json() : [])
      .then(list => { if (list.length) setCameraIndex(list[0].index) })
      .catch(() => setCameraIndex(0))  // backend unreachable — fall back to 0
  }, [])

  return (
    <BrowserRouter>
      <Routes>
        <Route path="/"        element={<HomePage />}    />
        <Route path="/monitor" element={<MonitorPage />} />
        <Route path="/summary" element={<SummaryPage />} />
        {/* Catch-all */}
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  )
}
