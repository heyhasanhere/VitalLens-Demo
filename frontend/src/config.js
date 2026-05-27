const _api = import.meta.env.VITE_API_BASE || 'http://localhost:8000'
export const API_BASE = _api
// Derive WS_BASE from API_BASE so only VITE_API_BASE needs to be set in production
export const WS_BASE  = import.meta.env.VITE_WS_BASE
  || _api.replace(/^https/, 'wss').replace(/^http/, 'ws')
