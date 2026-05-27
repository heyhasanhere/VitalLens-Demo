// ============================================================
// Complex number helpers
// ============================================================
const cadd  = (a, b) => [a[0]+b[0], a[1]+b[1]]
const csub  = (a, b) => [a[0]-b[0], a[1]-b[1]]
const cmul  = (a, b) => [a[0]*b[0]-a[1]*b[1], a[0]*b[1]+a[1]*b[0]]
const cdiv  = (a, b) => { const d=b[0]*b[0]+b[1]*b[1]; return [(a[0]*b[0]+a[1]*b[1])/d,(a[1]*b[0]-a[0]*b[1])/d] }
const csqrt = ([r,i]) => { const m=Math.sqrt(r*r+i*i),g=Math.atan2(i,r),s=Math.sqrt(m); return [s*Math.cos(g/2),s*Math.sin(g/2)] }
const cabs  = ([r,i]) => Math.sqrt(r*r+i*i)

function _bilinear(s, K) {
  return cdiv([K+s[0], s[1]], [K-s[0], -s[1]])
}

// ============================================================
// Butterworth bandpass SOS design
// Equivalent to scipy.signal.butter(N,[lo,hi],btype='band',fs=fs,output='sos')
// ============================================================
export function butterBandpassSOS(N, lowHz, highHz, fsHz) {
  const K  = 2 * fsHz
  const wl = K * Math.tan(Math.PI * lowHz  / fsHz)
  const wh = K * Math.tan(Math.PI * highHz / fsHz)
  const bw = wh - wl
  const w0sq = wl * wh

  // Nth-order Butterworth LP prototype poles: e^(j*π*(2k+N-1)/(2N)) for k=1..N
  const allLp = []
  for (let k = 1; k <= N; k++) {
    const theta = Math.PI * (2*k + N - 1) / (2*N)
    allLp.push([Math.cos(theta), Math.sin(theta)])
  }

  // Separate: real LP poles vs positive-imaginary LP poles (one per conj pair)
  const realLp = allLp.filter(([, pi]) => Math.abs(pi) < 1e-8)
  const posLp  = allLp.filter(([, pi]) => pi > 1e-8)

  const sos = []

  // Real LP pole → 1 analog BP conjugate pair → 1 SOS
  for (const [pr] of realLp) {
    const sr   = pr * bw
    const disc = sr * sr - 4 * w0sq
    let za, zb
    if (disc >= 0) {
      const sd = Math.sqrt(disc)
      za = _bilinear([(sr+sd)/2, 0], K)
      zb = _bilinear([(sr-sd)/2, 0], K)
    } else {
      const sd = Math.sqrt(-disc)
      za = _bilinear([sr/2,  sd/2], K)
      zb = _bilinear([sr/2, -sd/2], K)
    }
    // za = conj(zb) → sumZ real, prodZ = |za|²
    const sumZ  = za[0] + zb[0]
    const prodZ = za[0]*zb[0] - za[1]*zb[1]
    sos.push([1, 0, -1, 1, -sumZ, prodZ])
  }

  // Complex LP pole pair → 2 SOS
  for (const pLp of posLp) {
    const pbw  = cmul(pLp, [bw, 0])
    const disc = csub(cmul(pbw, pbw), [4*w0sq, 0])
    const sqD  = csqrt(disc)
    const s1   = [(pbw[0]+sqD[0])/2, (pbw[1]+sqD[1])/2]
    const s2   = [(pbw[0]-sqD[0])/2, (pbw[1]-sqD[1])/2]
    const z1   = _bilinear(s1, K)
    const z2   = _bilinear(s2, K)
    // SOS for (z1, conj(z1))
    sos.push([1, 0, -1, 1, -2*z1[0], z1[0]*z1[0]+z1[1]*z1[1]])
    // SOS for (z2, conj(z2))
    sos.push([1, 0, -1, 1, -2*z2[0], z2[0]*z2[0]+z2[1]*z2[1]])
  }

  // Normalize gain at passband center so |H(e^jwc)| = 1
  const wc    = Math.PI * (lowHz + highHz) / fsHz
  const zinv  = [Math.cos(wc), -Math.sin(wc)]  // z^-1 on unit circle
  const zinv2 = cmul(zinv, zinv)
  let H = [1, 0]
  for (const [b0, b1, b2, , a1, a2] of sos) {
    const num = cadd([b0,0], cadd(cmul([b1,0], zinv), cmul([b2,0], zinv2)))
    const den = cadd([1, 0], cadd(cmul([a1,0], zinv), cmul([a2,0], zinv2)))
    H = cmul(H, cdiv(num, den))
  }
  const gain = cabs(H)
  if (gain > 1e-12) { sos[0][0]/=gain; sos[0][1]/=gain; sos[0][2]/=gain }

  return sos
}

// ============================================================
// SOS filter application (direct form II transposed)
// ============================================================
export function sosFilt(sos, signal) {
  const out = new Float64Array(signal.length)
  const zi  = sos.map(() => [0.0, 0.0])
  for (let n = 0; n < signal.length; n++) {
    let x = signal[n]
    for (let s = 0; s < sos.length; s++) {
      const [b0, b1, b2, , a1, a2] = sos[s]
      const [d0, d1] = zi[s]
      const y = b0*x + d0
      zi[s][0] = b1*x - a1*y + d1
      zi[s][1] = b2*x - a2*y
      x = y
    }
    out[n] = x
  }
  return out
}

// Zero-phase filter: forward pass + reverse + backward pass + reverse
export function sosFiltFilt(sos, signal) {
  if (signal.length < 4) return signal instanceof Float64Array ? signal : new Float64Array(signal)
  const fwd = sosFilt(sos, signal)
  const rev = sosFilt(sos, fwd.slice().reverse())
  return rev.reverse()
}

// ============================================================
// In-place radix-2 FFT (Cooley-Tukey)
// ============================================================
function fft(re, im) {
  const N = re.length
  // Bit-reversal
  for (let i = 1, j = 0; i < N; i++) {
    let bit = N >> 1
    for (; j & bit; bit >>= 1) j ^= bit
    j ^= bit
    if (i < j) {
      ;[re[i], re[j]] = [re[j], re[i]]
      ;[im[i], im[j]] = [im[j], im[i]]
    }
  }
  // Butterfly
  for (let len = 2; len <= N; len <<= 1) {
    const ang = -2 * Math.PI / len
    const wR = Math.cos(ang), wI = Math.sin(ang)
    for (let i = 0; i < N; i += len) {
      let cR = 1, cI = 0
      for (let j = 0; j < len >> 1; j++) {
        const uR = re[i+j], uI = im[i+j]
        const vR = re[i+j+(len>>1)]*cR - im[i+j+(len>>1)]*cI
        const vI = re[i+j+(len>>1)]*cI + im[i+j+(len>>1)]*cR
        re[i+j] = uR+vR; im[i+j] = uI+vI
        re[i+j+(len>>1)] = uR-vR; im[i+j+(len>>1)] = uI-vI
        const nR = cR*wR - cI*wI; cI = cR*wI + cI*wR; cR = nR
      }
    }
  }
}

// ============================================================
// Welch power spectral density
// Mirrors scipy.signal.welch with hann window, 50% overlap
// ============================================================
export function welch(signal, fs, nperseg = 150, nfft = 2048) {
  nperseg = Math.min(nperseg, signal.length)
  const step = Math.max(1, nperseg >> 1)  // 50% overlap
  const nSegs = Math.max(1, Math.floor((signal.length - nperseg) / step) + 1)
  const nBins = (nfft >> 1) + 1

  // Precompute Hann window + its power
  const win = new Float64Array(nperseg)
  let winPow = 0
  for (let i = 0; i < nperseg; i++) {
    win[i] = 0.5 * (1 - Math.cos(2 * Math.PI * i / (nperseg - 1)))
    winPow += win[i] * win[i]
  }

  const psdAcc = new Float64Array(nBins)

  for (let seg = 0; seg < nSegs; seg++) {
    const start = seg * step
    const re = new Float64Array(nfft)
    const im = new Float64Array(nfft)
    for (let i = 0; i < nperseg && start+i < signal.length; i++) {
      re[i] = signal[start+i] * win[i]
    }
    fft(re, im)
    for (let k = 0; k < nBins; k++) psdAcc[k] += re[k]*re[k] + im[k]*im[k]
  }

  const scale = 1.0 / (fs * winPow * nSegs)
  const freqs = new Float64Array(nBins)
  const psd   = new Float64Array(nBins)
  for (let k = 0; k < nBins; k++) {
    freqs[k] = k * fs / nfft
    psd[k]   = psdAcc[k] * scale * (k > 0 && k < nBins-1 ? 2 : 1)  // one-sided scaling
  }
  return { freqs, psd }
}

// Dominant frequency in [loHz, hiHz] with parabolic interpolation
export function peakHz(freqs, psd, loHz, hiHz) {
  let bestK = -1, bestP = -Infinity
  for (let k = 0; k < freqs.length; k++) {
    if (freqs[k] >= loHz && freqs[k] <= hiHz && psd[k] > bestP) {
      bestP = psd[k]; bestK = k
    }
  }
  if (bestK < 1 || bestK >= freqs.length - 1) return bestK >= 0 ? freqs[bestK] : 0
  const a = psd[bestK-1], b = psd[bestK], g = psd[bestK+1]
  const denom = a - 2*b + g
  if (Math.abs(denom) < 1e-12) return freqs[bestK]
  const p = 0.5 * (a - g) / denom
  return freqs[bestK] + p * (freqs[1] - freqs[0])
}

// ============================================================
// Find local maxima with minimum inter-peak distance
// ============================================================
export function findPeaks(signal, minDist) {
  const peaks = []
  for (let i = 1; i < signal.length - 1; i++) {
    if (signal[i] > signal[i-1] && signal[i] > signal[i+1]) {
      if (peaks.length === 0 || i - peaks[peaks.length-1] >= minDist) {
        peaks.push(i)
      } else if (signal[i] > signal[peaks[peaks.length-1]]) {
        peaks[peaks.length-1] = i  // replace if closer peak is higher
      }
    }
  }
  return peaks
}

// ============================================================
// Vitals computation
// ============================================================

// Cached SOS designs — keyed by `${loHz}-${hiHz}-${fps}`
const _sosCache = {}
function _getSOS(lo, hi, fps) {
  const key = `${lo}-${hi}-${fps}`
  if (!_sosCache[key]) _sosCache[key] = butterBandpassSOS(3, lo, hi, fps)
  return _sosCache[key]
}

export function bvpToHr(bvp, fps = 30) {
  if (bvp.length < 32) return 0
  const arr = new Float64Array(bvp)
  const sos = _getSOS(0.67, 3.0, fps)
  const filtered = sosFiltFilt(sos, arr)
  const { freqs, psd } = welch(filtered, fps, Math.min(filtered.length, 150), 2048)
  let fHz = peakHz(freqs, psd, 0.67, 3.0)
  if (fHz === 0) return 0

  // Harmonic suppression: if dominant > 1.3 Hz, check if f/2 has ≥15% power → use half
  if (fHz > 1.3) {
    let halfPow = 0, peakPow = 0
    for (let k = 0; k < freqs.length; k++) {
      if (Math.abs(freqs[k] - fHz/2) < (freqs[1]-freqs[0])) halfPow = psd[k]
      if (Math.abs(freqs[k] - fHz)   < (freqs[1]-freqs[0])) peakPow = psd[k]
    }
    if (freqs.some(f => f >= 0.67) && halfPow >= 0.15 * peakPow) {
      const subHz = peakHz(freqs, psd, 0.67, fHz * 0.6)
      if (subHz > 0) fHz = subHz
    }
  }
  return fHz * 60
}

export function bvpToBr(bvp, fps = 30) {
  if (bvp.length < 64) return 0
  const arr = new Float64Array(bvp)
  const sos = _getSOS(0.15, 0.4, fps)
  const filtered = sosFiltFilt(sos, arr)
  const { freqs, psd } = welch(filtered, fps, Math.min(filtered.length, 150), 2048)
  const fHz = peakHz(freqs, psd, 0.15, 0.4)
  return fHz * 60
}

export function bvpToHrv(bvp, fps = 30) {
  if (bvp.length < 60) return 0
  const hrWelch = bvpToHr(bvp, fps)
  if (hrWelch <= 0) return 0
  const expIbi = 60000 / hrWelch
  const ibiLo  = Math.max(300,  expIbi * 0.75)
  const ibiHi  = Math.min(1500, expIbi * 1.25)

  const arr = new Float64Array(bvp)
  const sos = _getSOS(0.67, 3.0, fps)
  const filtered = sosFiltFilt(sos, arr)
  const minDist = Math.max(Math.floor(fps * 0.4), 5)

  const peaksP = findPeaks(filtered, minDist)
  const peaksN = findPeaks(filtered.map(v => -v), minDist)

  function rmssd(peaks) {
    if (peaks.length < 3) return 0
    const ibi = []
    for (let i = 1; i < peaks.length; i++) {
      const ms = (peaks[i] - peaks[i-1]) / fps * 1000
      if (ms >= ibiLo && ms <= ibiHi) ibi.push(ms)
    }
    if (ibi.length < 3) return 0
    let sumSq = 0
    for (let i = 1; i < ibi.length; i++) sumSq += (ibi[i]-ibi[i-1])**2
    const r = Math.sqrt(sumSq / (ibi.length - 1))
    return (r >= 5 && r <= 150) ? Math.round(r * 10) / 10 : 0
  }

  const hp = rmssd(peaksP), hn = rmssd(peaksN)
  if (hp > 0 && hn > 0) return peaksP.length >= peaksN.length ? hp : hn
  return hp || hn
}

export function bvpToIbi(bvp, fps = 30) {
  if (bvp.length < 60) return null
  const arr = new Float64Array(bvp)
  const sos = _getSOS(0.67, 3.0, fps)
  const filtered = sosFiltFilt(sos, arr)
  const minDist = Math.max(Math.floor(fps * 0.4), 5)
  const peaks = findPeaks(filtered, minDist)
  if (peaks.length < 5) return null
  const ibi = []
  for (let i = 1; i < peaks.length; i++) {
    const ms = (peaks[i]-peaks[i-1]) / fps * 1000
    if (ms >= 300 && ms <= 1500) ibi.push(ms)
  }
  return ibi.length >= 4 ? ibi : null
}

export function ibiToRhythm(ibi) {
  if (!ibi || ibi.length < 4) return 'Unknown'
  const mean = ibi.reduce((a,b)=>a+b,0) / ibi.length
  const std  = Math.sqrt(ibi.reduce((a,v)=>a+(v-mean)**2,0) / ibi.length)
  return (std / (mean + 1e-6)) > 0.2 ? 'Irregular' : 'Regular'
}

export function bvpSnr(bvp, fps = 30) {
  if (bvp.length < 64) return 0
  const arr = new Float64Array(bvp)
  const sos = _getSOS(0.67, 3.0, fps)
  const filtered = sosFiltFilt(sos, arr)
  const { freqs, psd } = welch(filtered, fps, Math.min(filtered.length, 150), 2048)
  const band = psd.filter((_, k) => freqs[k] >= 0.67 && freqs[k] <= 3.0)
  if (band.length === 0) return 0
  const maxP = Math.max(...band)
  const sorted = [...band].sort((a,b)=>a-b)
  const median = sorted[Math.floor(sorted.length/2)]
  return maxP / (median + 1e-9)
}

export function hrvToStress(hrv) {
  if (hrv <= 0) return 0
  const clamped = Math.max(5, Math.min(60, hrv))
  return Math.round(Math.max(0, Math.min(100, (60 - clamped) / 55 * 100)) * 10) / 10
}
