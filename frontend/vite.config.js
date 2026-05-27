import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import fs from 'fs'
import path from 'path'
import { fileURLToPath } from 'url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))

// Serve WASM worker files (.mjs) directly without Vite's module transformation.
// Vite rewrites import() calls inside ort.bundle.min.mjs and adds ?import to
// the worker file URLs, then tries to transform them — failing because the worker
// files contain Node.js-specific code. This plugin intercepts those requests and
// streams the raw files instead.
function wasmWorkerPassthrough() {
  const staticDirs = ['/onnx-wasm/', '/mediapipe-wasm/']
  return {
    name: 'wasm-worker-passthrough',
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        const url = req.url?.split('?')[0] ?? ''
        if (staticDirs.some(d => url.startsWith(d))) {
          const filePath = path.join(__dirname, 'public', url)
          if (fs.existsSync(filePath)) {
            const isJs = url.endsWith('.js') || url.endsWith('.mjs')
            res.setHeader('Content-Type', isJs ? 'text/javascript' : 'application/wasm')
            res.setHeader('Cache-Control', 'no-cache')
            fs.createReadStream(filePath).pipe(res)
            return
          }
        }
        next()
      })
    },
  }
}

export default defineConfig({
  plugins: [react(), wasmWorkerPassthrough()],
  optimizeDeps: {
    exclude: ['onnxruntime-web', '@mediapipe/tasks-vision'],
  },
})
