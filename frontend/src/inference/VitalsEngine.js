import {
  bvpToHr, bvpToBr, bvpToHrv, bvpToIbi, ibiToRhythm,
  bvpSnr, hrvToStress,
} from './signal.js'

const MODEL_CONFIGS = {
  factorizephys: {
    path:      '/models/factorizephys.onnx',
    inputName: 'video_clip',
    type:      'factorizephys',
    clipLen:   160,
    inputSize: 72,
    bufferLen: 160,
  },
  factorizephys_ibvp: {
    path:      '/models/factorizephys_ibvp.onnx',
    inputName: 'video_clip',
    type:      'factorizephys',
    clipLen:   160,
    inputSize: 72,
    bufferLen: 160,
  },
  efficientphys: {
    path:      '/models/vitallens_rppg.onnx',
    inputName: 'video_clip_TCWH',
    type:      'efficientphys',
    clipLen:   160,
    inputSize: 128,
    bufferLen: 161,   // needs clipLen+1 frames (161 → 160 diffs)
  },
  physnet: {
    path:      '/models/physnet.onnx',
    inputName: 'video_clip',
    type:      'factorizephys',   // same raw RGB (B,3,T,H,W) preprocessing
    clipLen:   128,
    inputSize: 72,
    bufferLen: 128,
  },
  physformer: {
    path:      '/models/physformer.onnx',
    inputName: 'video_clip',
    type:      'physformer',
    clipLen:   160,
    inputSize: 72,
    bufferLen: 160,   // collect 160, append last frame → 161 input to model
  },
}

const INFERENCE_STRIDE = 80
const BVP_BUF_LEN      = 300
const HRV_BUF_LEN      = 900
const SNR_THRESHOLD    = 2.8
const HR_JUMP_THRESH   = 25.0
const BBOX_ALPHA       = 0.85

// FactorizePhys: raw RGB (1,3,T,H,W) — channels-first
function buildTensorFactorizephys(frames, T, H, W) {
  const out = new Float32Array(3 * T * H * W)
  for (let t = 0; t < T; t++) {
    const f = frames[t]
    for (let h = 0; h < H; h++) {
      for (let w = 0; w < W; w++) {
        const pi = (h * W + w) * 3
        out[0*T*H*W + t*H*W + h*W + w] = f[pi]
        out[1*T*H*W + t*H*W + h*W + w] = f[pi+1]
        out[2*T*H*W + t*H*W + h*W + w] = f[pi+2]
      }
    }
  }
  return { data: out, shape: [1, 3, T, H, W] }
}

// EfficientPhys: DiffNorm preprocessing (1,T,3,H,W) — time-first
function buildTensorEfficientphys(frames, T, H, W) {
  // frames length = T+1 (161 frames → 160 diffs)
  const N = frames.length - 1  // should equal T=160
  const out = new Float32Array(N * 3 * H * W)
  for (let t = 0; t < N; t++) {
    const f0 = frames[t]
    const f1 = frames[t+1]
    // brightness normalize each frame
    let sum0 = 0, sum1 = 0
    const nPix = H * W
    for (let i = 0; i < nPix * 3; i++) { sum0 += f0[i]; sum1 += f1[i] }
    const mean0 = sum0 / (nPix * 3) || 1
    const mean1 = sum1 / (nPix * 3) || 1
    for (let h = 0; h < H; h++) {
      for (let w = 0; w < W; w++) {
        const pi = (h * W + w) * 3
        for (let c = 0; c < 3; c++) {
          const n0 = f0[pi+c] / mean0
          const n1 = f1[pi+c] / mean1
          const d  = (n1 - n0) / (n1 + n0 + 1e-6)
          const clamped = Math.max(-3, Math.min(3, d))
          // (t, c, h, w) → t*3*H*W + c*H*W + h*W + w
          out[t*3*H*W + c*H*W + h*W + w] = clamped
        }
      }
    }
  }
  return { data: out, shape: [1, N, 3, H, W] }
}

export class VitalsEngine {
  constructor(modelName = 'factorizephys') {
    this._cfg        = MODEL_CONFIGS[modelName] || MODEL_CONFIGS.factorizephys
    this._session    = null
    this._landmarker = null
    this._frameBuffer  = []
    this._bvpBuffer    = []
    this._hrvBuffer    = []
    this._frameCount   = 0
    this._bboxEma      = null
    this._mpTs         = 0
    this._fps          = 30
    this._fpsFrames    = 0
    this._fpsT0        = 0
    this._fpsCal       = false
    this._emaAlpha     = 0.25
    this._detCanvas    = null
    this._detCtx       = null
    this._cropCanvas   = null
    this._cropCtx      = null
    this._state = {
      hr: 0, br: 0, hrv: 0, stress: 0, snr: 0,
      pos_hr: 0, chrom_hr: 0,
      bvp: new Array(32).fill(0),
      lighting: 'Good',
      face_detected: false,
      face_bbox: null,
      timestamp: Date.now() / 1000,
      ready: false,
      lum_std: null,
      blink_rate: 0,
      sway: 0,
      rhythm: 'Unknown',
    }
  }

  async init() {
    const ort = await import('onnxruntime-web')
    ort.env.wasm.wasmPaths  = '/onnx-wasm/'
    ort.env.wasm.numThreads = 1
    this._session = await ort.InferenceSession.create(
      this._cfg.path,
      { executionProviders: ['wasm'] }
    )

    const { FaceLandmarker, FilesetResolver } = await import('@mediapipe/tasks-vision')
    const filesetResolver = await FilesetResolver.forVisionTasks('/mediapipe-wasm')
    this._landmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
      baseOptions: { modelAssetPath: '/models/face_landmarker.task', delegate: 'CPU' },
      runningMode: 'VIDEO',
      numFaces: 1,
      minFaceDetectionConfidence: 0.7,
      minFacePresenceConfidence: 0.7,
      minTrackingConfidence: 0.7,
    })

    const S = this._cfg.inputSize
    this._detCanvas = document.createElement('canvas')
    this._detCtx    = this._detCanvas.getContext('2d', { willReadFrequently: true })
    this._cropCanvas = document.createElement('canvas')
    this._cropCanvas.width  = S
    this._cropCanvas.height = S
    this._cropCtx = this._cropCanvas.getContext('2d', { willReadFrequently: true })

    this._fpsT0 = performance.now()
  }

  processFrame(videoEl) {
    if (!this._session || !this._landmarker || !videoEl) return

    const vw = videoEl.videoWidth, vh = videoEl.videoHeight
    if (!vw || !vh) return

    if (this._detCanvas.width !== vw || this._detCanvas.height !== vh) {
      this._detCanvas.width = vw; this._detCanvas.height = vh
    }
    this._detCtx.drawImage(videoEl, 0, 0, vw, vh)

    this._mpTs += 33
    const result = this._landmarker.detectForVideo(videoEl, this._mpTs)
    const hasLm  = result.faceLandmarks && result.faceLandmarks.length > 0

    let face = null, bbox = null, faceDetected = false
    const S = this._cfg.inputSize

    if (hasLm) {
      const lm = result.faceLandmarks[0]
      let x1 = Infinity, y1 = Infinity, x2 = -Infinity, y2 = -Infinity
      for (const { x, y } of lm) {
        if (x*vw < x1) x1 = x*vw; if (x*vw > x2) x2 = x*vw
        if (y*vh < y1) y1 = y*vh; if (y*vh > y2) y2 = y*vh
      }
      const padX = (x2-x1)*0.15, padY = (y2-y1)*0.15
      x1 = Math.max(0, x1-padX); y1 = Math.max(0, y1-padY)
      x2 = Math.min(vw, x2+padX); y2 = Math.min(vh, y2+padY)

      const raw = [x1, y1, x2, y2]
      if (!this._bboxEma) {
        this._bboxEma = raw
      } else {
        for (let i = 0; i < 4; i++)
          this._bboxEma[i] = BBOX_ALPHA*this._bboxEma[i] + (1-BBOX_ALPHA)*raw[i]
      }
      const [sx1,sy1,sx2,sy2] = this._bboxEma.map(Math.round)
        .map((v,i) => Math.max(0, Math.min(i<2 ? (i===0?vw:vh) : (i===2?vw:vh), v)))

      const bw = sx2-sx1, bh = sy2-sy1
      if (bw > 0 && bh > 0) {
        this._cropCtx.drawImage(this._detCanvas, sx1, sy1, bw, bh, 0, 0, S, S)
        const imgData = this._cropCtx.getImageData(0, 0, S, S)
        face = _imageDataToRGB(imgData)
        bbox = { x: sx1/vw, y: sy1/vh, w: bw/vw, h: bh/vh }
        faceDetected = true
      }
    }

    if (!face) {
      this._cropCtx.drawImage(this._detCanvas, 0, 0, vw, vh, 0, 0, S, S)
      const imgData = this._cropCtx.getImageData(0, 0, S, S)
      face = _imageDataToRGB(imgData)
    }

    if (faceDetected) {
      this._frameBuffer.push(face)
      if (this._frameBuffer.length > this._cfg.bufferLen) this._frameBuffer.shift()
      this._consecutiveNoFace = 0
    } else {
      this._consecutiveNoFace = (this._consecutiveNoFace || 0) + 1
      if (this._consecutiveNoFace >= 20) {  // ~1 s at 20 fps processing
        this._frameBuffer = []
        this._bvpBuffer   = []
        this._hrvBuffer   = []
        this._state.hr = 0; this._state.br = 0
        this._state.hrv = 0; this._state.stress = 0
        this._state.snr = 0; this._state.bvp = new Array(32).fill(0)
        this._state.ready = false
      }
    }

    this._frameCount++

    if (!this._fpsCal) {
      this._fpsFrames++
      if (this._fpsFrames >= 90) {
        this._fps = Math.round(this._fpsFrames / ((performance.now()-this._fpsT0)/1000))
        this._fpsCal = true
      }
    }

    if (faceDetected && this._frameBuffer.length === this._cfg.bufferLen && this._frameCount % INFERENCE_STRIDE === 0) {
      this._runInference().catch(console.error)
    }

    this._state.face_detected = faceDetected
    this._state.face_bbox     = bbox
    this._state.timestamp     = Date.now() / 1000
  }

  async _runInference() {
    const ort    = await import('onnxruntime-web')
    const frames = this._frameBuffer.slice()
    const { clipLen: T, inputSize: S, inputName } = this._cfg

    let tensorData, shape
    if (this._cfg.type === 'physformer') {
      // Append last frame to get T+1=161; model internally diffs → T=160
      const clip = frames.slice(-T)
      ;({ data: tensorData, shape } = buildTensorFactorizephys(
        [...clip, clip[clip.length - 1]], T + 1, S, S))
    } else if (this._cfg.type === 'efficientphys') {
      // DiffNorm: all bufferLen (T+1) frames → T diffs
      ;({ data: tensorData, shape } = buildTensorEfficientphys(frames, T, S, S))
    } else {
      // factorizephys / physnet / factorizephys_ibvp: raw RGB (B,3,T,H,W)
      const clip = frames.slice(-T)
      ;({ data: tensorData, shape } = buildTensorFactorizephys(clip, T, S, S))
    }

    const tensor = new ort.Tensor('float32', tensorData, shape)
    const output = await this._session.run({ [inputName]: tensor })
    const bvp    = Array.from(output.bvp.data)

    this._bvpBuffer.push(...bvp)
    if (this._bvpBuffer.length > BVP_BUF_LEN) this._bvpBuffer.splice(0, this._bvpBuffer.length - BVP_BUF_LEN)
    this._hrvBuffer.push(...bvp)
    if (this._hrvBuffer.length > HRV_BUF_LEN) this._hrvBuffer.splice(0, this._hrvBuffer.length - HRV_BUF_LEN)

    const fps      = this._fps
    const minSamples = fps * 4
    if (this._bvpBuffer.length < minSamples) return

    const bvpArr  = this._bvpBuffer
    const snr     = bvpSnr(bvpArr, fps)
    const liveBvp = bvpArr.slice(-32)

    this._state.snr   = Math.round(snr * 100) / 100
    this._state.bvp   = liveBvp.map(v => Math.round(v * 10000) / 10000)
    this._state.ready = true

    if (this._hrvBuffer.length >= 300) {
      const hrv    = bvpToHrv(this._hrvBuffer, fps)
      const rhythm = ibiToRhythm(bvpToIbi(this._hrvBuffer, fps))
      if (hrv > 0) {
        const cur = this._state.hrv
        const a = this._emaAlpha
        this._state.hrv    = Math.round((cur === 0 ? hrv : a*hrv + (1-a)*cur) * 10) / 10
        this._state.stress = hrvToStress(this._state.hrv)
      }
      this._state.rhythm = rhythm
    }

    if (snr < SNR_THRESHOLD) return

    const hr = bvpToHr(bvpArr, fps)
    const br = bvpToBr(bvpArr, fps)

    const a = this._emaAlpha
    const prevHr = this._state.hr
    if (prevHr > 0 && Math.abs(hr - prevHr) > HR_JUMP_THRESH) {
      // hold previous
    } else if (hr > 0) {
      this._state.hr = Math.round((prevHr === 0 ? hr : a*hr + (1-a)*prevHr) * 10) / 10
    }
    if (br > 0) {
      const prevBr = this._state.br
      this._state.br = Math.round((prevBr === 0 ? br : a*br + (1-a)*prevBr) * 10) / 10
    }
  }

  getState() {
    return { ...this._state }
  }

  dispose() {
    if (this._session)    this._session.release?.()
    if (this._landmarker) this._landmarker.close()
    this._session    = null
    this._landmarker = null
  }
}

function _imageDataToRGB(imageData) {
  const { data, width, height } = imageData
  const out = new Float32Array(width * height * 3)
  for (let i = 0; i < width * height; i++) {
    out[i*3+0] = data[i*4+0] / 255
    out[i*3+1] = data[i*4+1] / 255
    out[i*3+2] = data[i*4+2] / 255
  }
  return out
}
