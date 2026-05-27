"""
VitalsEngine — real-time rPPG + lighting inference engine.

Runs in a background thread; the FastAPI WebSocket handler reads state
every ~1s and sends it to the frontend.

Signal pipeline per WebSocket session:
  1. Capture frame from webcam (background thread, ~30fps)
  2. Lighting ONNX: classify frame → "Good" / "Mixed" / "Poor"
  3. Haar cascade: detect + crop face
  4. Preprocessing: CLAHE → white balance → gamma (if poor/mixed)
  5. Accumulate in frame buffer (deque of 161 frames)
  6. Every INFERENCE_STRIDE frames: run rPPG ONNX → BVP[160]
  7. Accumulate BVP into rolling buffer (10 s)
  8. Recompute HR / BR / HRV / Stress from rolling buffer
"""
from __future__ import annotations

import threading
import time
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision
import numpy as np
import onnxruntime as ort
from scipy.signal import butter, filtfilt, find_peaks, welch

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

import os
from dotenv import load_dotenv
load_dotenv()

_BASE = Path(__file__).parent.parent   # VitalLens/
_W    = _BASE / "weights"

RPPG_PATH            = Path(os.environ.get("RPPG_MODEL_PATH",
                            str(_W / "factorizephys.onnx")))
LIGHTING_PATH        = Path(os.environ.get("LIGHTING_MODEL_PATH",
                            str(_W / "vitallens_lighting.onnx")))
FACE_LANDMARKER_PATH = Path(os.environ.get("FACE_LANDMARKER_PATH",
                            str(_W / "face_landmarker.task")))

MODEL_PATHS: dict[str, Path] = {
    'factorizephys':      _W / 'factorizephys.onnx',
    'factorizephys_ibvp': _W / 'factorizephys_ibvp.onnx',
    'efficientphys':      _W / 'vitallens_rppg.onnx',
    'physnet':            _W / 'physnet.onnx',
    'physformer':         _W / 'physformer.onnx',
}

DEFAULT_CAMERA_INDEX = int(os.environ.get("CAMERA_INDEX", "0"))
_USE_V4L2            = os.environ.get("USE_V4L2", "0").strip() == "1"

CLIP_LEN         = 160
INPUT_SIZE       = 128
INFERENCE_STRIDE = 160      # run rPPG on fully non-overlapping clips ≈ 5.3 s at 30 fps
BVP_BUFFER_LEN   = 300       # 10 s rolling BVP history for HR/BR/HRV
LIGHTING_LABELS  = ["Good", "Poor", "Mixed"]

# ImageNet normalisation for lighting classifier input
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_BBOX_ALPHA        = 0.85  # EMA momentum on previous bbox (high = smooth, slow to shift)

# Runtime-adjustable signal-processing parameters — can be tuned live via PATCH /session/config
_runtime_config: dict = {
    'snr_threshold':   2.8,    # BVP SNR gate; below this holds previous reading
    'hr_jump_thresh':  25.0,   # BPM; continuity filter — reject jump larger than this
    'sway_thresh':     0.03,   # fraction of frame; head motion gate
    'ear_blink_thresh': 0.22,  # EAR below this counts as a blink frame
}
_config_lock = threading.Lock()

def get_rppg_config() -> dict:
    with _config_lock:
        return dict(_runtime_config)

def update_rppg_config(updates: dict) -> dict:
    _allowed = {'snr_threshold', 'hr_jump_thresh', 'sway_thresh', 'ear_blink_thresh'}
    with _config_lock:
        for k, v in updates.items():
            if k in _allowed:
                _runtime_config[k] = float(v)
    return get_rppg_config()

# MediaPipe FaceMesh landmark indices
_MP_LEFT_EYE_OUTER  = 33
_MP_RIGHT_EYE_OUTER = 263
_MP_NOSE_TIP        = 1
_STAB_ANCHORS       = [_MP_LEFT_EYE_OUTER, _MP_RIGHT_EYE_OUTER, _MP_NOSE_TIP]

# EAR (Eye Aspect Ratio) landmark indices — (P1, P2, P3, P4, P5, P6)
_MP_LEFT_EAR_PTS  = (33, 160, 158, 133, 153, 144)
_MP_RIGHT_EAR_PTS = (362, 385, 387, 263, 373, 380)
_EAR_BLINK_THRESH  = 0.22   # below this = blink frame (raised slightly for glasses)
_EAR_BLINK_MIN_FRAMES = 2  # must be below threshold for this many frames to count as a real blink


# ---------------------------------------------------------------------------
# Signal processing helpers
# ---------------------------------------------------------------------------

def _bandpass(sig: np.ndarray, low: float, high: float, fps: float) -> np.ndarray:
    nyq = fps / 2.0
    lo  = max(low  / nyq, 1e-4)
    hi  = min(high / nyq, 0.9999)
    if lo >= hi:
        return sig
    try:
        b, a = butter(3, [lo, hi], btype="band")
        return filtfilt(b, a, sig)
    except Exception:
        return sig


def _peak_hz(freqs: np.ndarray, psd: np.ndarray, band: np.ndarray) -> float:
    """Dominant frequency in band with parabolic sub-bin interpolation."""
    band_psd = psd[band]
    global_indices = np.where(band)[0]
    k = int(global_indices[band_psd.argmax()])
    if 0 < k < len(freqs) - 1:
        a, b, g = psd[k - 1], psd[k], psd[k + 1]
        denom = a - 2 * b + g
        if denom != 0:
            p = 0.5 * (a - g) / denom
            return float(freqs[k] + p * (freqs[1] - freqs[0]))
    return float(freqs[k])


def _pos_hr(rgb_trace: np.ndarray, fps: float) -> float:
    """
    POS algorithm (Wang et al. 2017).
    rgb_trace: (T, 3) float32, R-G-B order, raw face-ROI means.
    """
    if len(rgb_trace) < 32:
        return 0.0
    mu  = rgb_trace.mean(axis=0) + 1e-6
    Cn  = rgb_trace / mu                               # temporal-mean normalised
    S1  = Cn[:, 0] - Cn[:, 1]                          # R - G
    S2  = Cn[:, 0] + Cn[:, 1] - 2.0 * Cn[:, 2]        # R + G - 2B
    alpha = np.std(S1) / (np.std(S2) + 1e-9)
    H   = S1 + alpha * S2
    H_f = _bandpass(H, 0.67, 3.0, fps)
    freqs, psd = welch(H_f, fs=fps, nperseg=min(len(H_f), 150), nfft=2048)
    band = (freqs >= 0.67) & (freqs <= 3.0)
    return _peak_hz(freqs, psd, band) * 60.0 if band.any() else 0.0


def _chrom_hr(rgb_trace: np.ndarray, fps: float) -> float:
    """
    CHROM algorithm (de Haan & Jeanne 2013).
    rgb_trace: (T, 3) float32, R-G-B order, raw face-ROI means.
    """
    if len(rgb_trace) < 32:
        return 0.0
    mu  = rgb_trace.mean(axis=0) + 1e-6
    Rn, Gn, Bn = rgb_trace[:, 0] / mu[0], rgb_trace[:, 1] / mu[1], rgb_trace[:, 2] / mu[2]
    Xc  = 3.0 * Rn - 2.0 * Gn
    Yc  = 1.5 * Rn + Gn - 1.5 * Bn
    alpha = np.std(Xc) / (np.std(Yc) + 1e-9)
    S   = Xc - alpha * Yc
    S_f = _bandpass(S, 0.67, 3.0, fps)
    freqs, psd = welch(S_f, fs=fps, nperseg=min(len(S_f), 150), nfft=2048)
    band = (freqs >= 0.67) & (freqs <= 3.0)
    return _peak_hz(freqs, psd, band) * 60.0 if band.any() else 0.0


def bvp_to_hr(bvp: np.ndarray, fps: float = 30.0) -> float:
    if len(bvp) < 32:
        return 0.0
    filtered   = _bandpass(bvp, 0.67, 3.0, fps)
    freqs, psd = welch(filtered, fs=fps, nperseg=min(len(filtered), 150), nfft=2048)
    band       = (freqs >= 0.67) & (freqs <= 3.0)
    if not band.any():
        return 0.0
    f_hz = _peak_hz(freqs, psd, band)
    # Harmonic suppression: if dominant peak > 1.3 Hz (78 BPM), check whether
    # f/2 has ≥15% of dominant power — if so, the dominant is likely the 2nd harmonic.
    # Threshold lowered from 35%→15% to catch laptop ISP and OBS re-encoded video cases
    # where the 2nd harmonic dominates the PSD strongly.
    if f_hz > 1.3:
        half_idx = int(np.argmin(np.abs(freqs - f_hz / 2.0)))
        peak_idx = int(np.argmin(np.abs(freqs - f_hz)))
        if freqs[half_idx] >= 0.67 and psd[half_idx] >= 0.15 * psd[peak_idx]:
            sub_band = (freqs >= 0.67) & (freqs <= f_hz * 0.6)
            if sub_band.any():
                f_hz = _peak_hz(freqs, psd, sub_band)
    return f_hz * 60.0


def bvp_to_br(bvp: np.ndarray, fps: float = 30.0) -> float:
    """Breathing rate via Welch PSD in the 0.15–0.4 Hz band (9–24 br/min)."""
    if len(bvp) < 64:
        return 0.0
    filtered   = _bandpass(bvp, 0.15, 0.4, fps)
    freqs, psd = welch(filtered, fs=fps, nperseg=min(len(filtered), 150), nfft=2048)
    band       = (freqs >= 0.15) & (freqs <= 0.4)
    if not band.any():
        return 0.0
    return _peak_hz(freqs, psd, band) * 60.0


def bvp_to_hrv(bvp: np.ndarray, fps: float = 30.0) -> float:
    """RMSSD in ms. Anchors IBI range to Welch HR to reject noise peaks."""
    if len(bvp) < 60:
        return 0.0
    hr_welch = bvp_to_hr(bvp, fps)
    if hr_welch <= 0:
        return 0.0
    expected_ibi = 60000.0 / hr_welch
    ibi_lo = max(300.0,  expected_ibi * 0.75)
    ibi_hi = min(1500.0, expected_ibi * 1.25)

    filtered  = _bandpass(bvp, 0.67, 3.0, fps)
    min_dist  = max(int(fps * 0.4), 5)

    # Try both signal polarities — EfficientPhys BVP can be inverted
    peaks_p, _ = find_peaks( filtered, distance=min_dist)
    peaks_n, _ = find_peaks(-filtered, distance=min_dist)

    def _rmssd(peaks):
        if len(peaks) < 3:
            return 0.0
        ibi = np.diff(peaks) / fps * 1000.0
        ibi = ibi[(ibi >= ibi_lo) & (ibi <= ibi_hi)]
        if len(ibi) < 3:
            return 0.0
        r = float(np.sqrt(np.mean(np.diff(ibi) ** 2)))
        return round(r, 1) if 5.0 <= r <= 150.0 else 0.0

    hrv_p = _rmssd(peaks_p)
    hrv_n = _rmssd(peaks_n)
    if hrv_p > 0 and hrv_n > 0:
        # prefer polarity with more valid peaks
        return hrv_p if len(peaks_p) >= len(peaks_n) else hrv_n
    return hrv_p or hrv_n


def hrv_to_stress(hrv_ms: float) -> float:
    """Map HRV → stress score 0–100. High HRV = low stress."""
    if hrv_ms <= 0:
        return 0.0
    hrv_clamped = max(5.0, min(60.0, hrv_ms))
    stress = (60.0 - hrv_clamped) / 55.0 * 100.0
    return round(max(0.0, min(100.0, stress)), 1)


# ---------------------------------------------------------------------------
# AF detection helpers
# ---------------------------------------------------------------------------

def bvp_to_ibi(bvp: np.ndarray, fps: float = 30.0) -> np.ndarray | None:
    """Extract inter-beat intervals (ms) from BVP. Returns None if < 5 peaks."""
    if len(bvp) < 60:
        return None
    filtered = _bandpass(bvp, 0.67, 3.0, fps)
    min_dist = max(int(fps * 0.4), 5)
    peaks, _ = find_peaks(filtered, distance=min_dist)
    if len(peaks) < 5:
        return None
    ibi_ms = np.diff(peaks) / fps * 1000.0
    valid  = ibi_ms[(ibi_ms >= 300) & (ibi_ms <= 1500)]
    return valid if len(valid) >= 4 else None


def ibi_to_rhythm(ibi_ms: np.ndarray | None) -> str:
    """CV > 0.2 → 'Irregular' (possible AF). NOT a medical diagnosis."""
    if ibi_ms is None or len(ibi_ms) < 4:
        return "Unknown"
    cv = float(np.std(ibi_ms) / (np.mean(ibi_ms) + 1e-6))
    return "Irregular" if cv > 0.2 else "Regular"


# ---------------------------------------------------------------------------
# Blink / EAR helpers
# ---------------------------------------------------------------------------

def _compute_ear(landmarks: np.ndarray) -> float:
    """Mean Eye Aspect Ratio across both eyes. landmarks: (468, 2) pixel coords."""
    def _ear_single(idx_tuple):
        pts = landmarks[list(idx_tuple)]   # (6, 2)
        v1 = float(np.linalg.norm(pts[1] - pts[5]))
        v2 = float(np.linalg.norm(pts[2] - pts[4]))
        h  = float(np.linalg.norm(pts[0] - pts[3]))
        return (v1 + v2) / (2.0 * h + 1e-6)
    return (_ear_single(_MP_LEFT_EAR_PTS) + _ear_single(_MP_RIGHT_EAR_PTS)) / 2.0


# ---------------------------------------------------------------------------
# Affine stabilisation helper
# ---------------------------------------------------------------------------

def _stabilise_clip(
    frames: list[np.ndarray],
    landmarks: list[np.ndarray | None],
) -> list[np.ndarray]:
    """
    Warp each frame to align its eye/nose anchor triangle with frame 0's.
    Uses similarity transform (rotation + scale + translation, no shear).
    Frames with missing landmarks are passed through unchanged.
    """
    ref_pts = None
    for lm in landmarks:
        if lm is not None:
            ref_pts = lm[_STAB_ANCHORS].astype(np.float32)
            break
    if ref_pts is None:
        return frames

    h, w = frames[0].shape[:2]
    stabilised = []
    for frame, lm in zip(frames, landmarks):
        if lm is None:
            stabilised.append(frame)
            continue
        src_pts = lm[_STAB_ANCHORS].astype(np.float32)
        M, _ = cv2.estimateAffinePartial2D(src_pts, ref_pts)
        if M is None:
            stabilised.append(frame)
        else:
            warped = cv2.warpAffine(
                frame, M, (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT,
            )
            stabilised.append(warped)
    return stabilised


# ---------------------------------------------------------------------------
# Frame preprocessing helpers
# ---------------------------------------------------------------------------

def _apply_clahe(frame: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab   = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    return cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]), cv2.COLOR_LAB2BGR)


def _apply_white_balance(frame: np.ndarray) -> np.ndarray:
    f        = frame.astype(np.float32)
    means    = f.mean(axis=(0, 1))          # (B, G, R)
    overall  = means.mean()
    for c in range(3):
        if means[c] > 0:
            f[:, :, c] *= overall / means[c]
    return np.clip(f, 0, 255).astype(np.uint8)


def _apply_gamma(frame: np.ndarray) -> np.ndarray:
    lum   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean()
    gamma = 0.5 if lum < 40 else 0.7 if lum < 80 else 0.85 if lum < 120 else 1.0
    if gamma == 1.0:
        return frame
    lut = np.array([min(255, int((i / 255.0) ** gamma * 255))
                    for i in range(256)], dtype=np.uint8)
    return cv2.LUT(frame, lut)


def preprocess_for_rppg(frames: list[np.ndarray], brightness_norm: bool = True,
                        input_size: int = INPUT_SIZE) -> np.ndarray:
    """DiffNorm preprocessing for EfficientPhys/VitalLens. Input: 161 frames → (1, 160, 3, H, W)."""
    resized = np.array(
        [cv2.resize(f, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
         for f in frames],
        dtype=np.float32,
    )
    if brightness_norm:
        lum     = resized.mean(axis=(1, 2, 3), keepdims=True)
        resized = resized / (lum + 1e-6)
    f_next = resized[1:]
    f_curr = resized[:-1]
    diff   = (f_next - f_curr) / (f_next + f_curr + 1e-6)
    diff   = np.clip(diff, -3.0, 3.0)
    return diff.transpose(0, 3, 1, 2)[np.newaxis]   # (1, T, 3, H, W)


def preprocess_for_factorizephys(frames: list[np.ndarray], input_size: int = 72) -> np.ndarray:
    """Raw RGB [0,1] for FactorizePhys/PhysNet. Input: T frames → (1, 3, T, H, W)."""
    resized = np.array(
        [cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), (input_size, input_size),
                    interpolation=cv2.INTER_LINEAR)
         for f in frames],
        dtype=np.float32,
    ) / 255.0                                        # (T, H, W, 3) in [0, 1]
    return resized.transpose(3, 0, 1, 2)[np.newaxis]  # (1, 3, T, H, W)


def preprocess_for_physformer(frames: list[np.ndarray], input_size: int = 72) -> np.ndarray:
    """Raw RGB [0,1] for PhysFormer. Input: 160 frames → (1, 3, 161, H, W).
    Appends last frame; model internally diffs to produce 160-frame BVP."""
    resized = np.array(
        [cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), (input_size, input_size),
                    interpolation=cv2.INTER_LINEAR)
         for f in frames],
        dtype=np.float32,
    ) / 255.0                                             # (T, H, W, 3) in [0, 1]
    with_extra = np.concatenate([resized, resized[-1:]], axis=0)   # (T+1, H, W, 3)
    return with_extra.transpose(3, 0, 1, 2)[np.newaxis]            # (1, 3, T+1, H, W)


def preprocess_for_lighting(frame: np.ndarray) -> np.ndarray:
    """ImageNet-normalised RGB frame → (1, 3, 224, 224) float32."""
    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
    norm    = (resized.astype(np.float32) / 255.0 - _IMAGENET_MEAN) / _IMAGENET_STD
    return norm.transpose(2, 0, 1)[np.newaxis]       # (1, 3, 224, 224)


# ---------------------------------------------------------------------------
# VitalsEngine
# ---------------------------------------------------------------------------

class VitalsEngine:
    """
    Manages the full capture → inference → signal pipeline.

    Designed to run in a background thread. The FastAPI WebSocket handler
    calls `get_message()` every ~1s to read the latest state.
    """

    def __init__(
        self,
        camera_index:    int   = 0,
        camera_url:      str | None = None,   # MJPEG stream URL — takes precedence over camera_index
        brightness_norm: bool  = True,
        target_fps:      float = 30.0,
        lock_exposure:   float | None = None,  # e.g. -6.0; None = leave auto
        rppg_path:       Path | None = None,   # override ONNX model; defaults to RPPG_PATH
    ):
        self.camera_index    = camera_index
        self.camera_url      = camera_url
        self.brightness_norm = brightness_norm
        self.target_fps      = target_fps
        self.lock_exposure   = lock_exposure
        self._rppg_path      = rppg_path or RPPG_PATH

        # ONNX sessions
        self._rppg_sess        = None
        self._lighting_sess    = None
        self._rppg_input       = None
        self._light_input      = None
        # Set by _load_models after ONNX shape inspection
        self._model_type       = 'efficientphys'
        self._model_clip_len   = CLIP_LEN
        self._model_input_size = INPUT_SIZE

        _fl_options = mp_vision.FaceLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(
                model_asset_path=str(FACE_LANDMARKER_PATH)
            ),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.3,
            min_face_presence_confidence=0.3,
            min_tracking_confidence=0.3,
        )
        self._face_landmarker = mp_vision.FaceLandmarker.create_from_options(_fl_options)
        self._mp_timestamp_ms = 0   # monotonically increasing, required by VIDEO mode

        # FPS calibration — measured over first 90 processed frames
        self._fps_calibrated  = False
        self._fps_frame_count = 0
        self._fps_t0          = 0.0

        # Rolling buffers
        self._frame_buf    : deque = deque(maxlen=CLIP_LEN + 1)
        self._landmark_buf : deque = deque(maxlen=CLIP_LEN + 1)  # (468,2) or None per frame
        self._bvp_buf      : deque = deque(maxlen=BVP_BUFFER_LEN)
        self._hrv_buf      : deque = deque(maxlen=900)   # 30s @ 30fps — sufficient for RMSSD (~30 IBIs)
        self._rgb_buf      : deque = deque(maxlen=BVP_BUFFER_LEN)
        self._light_buf    : deque = deque(maxlen=30)
        self._temporal     : deque = deque(maxlen=3)
        self._lum_buf      : deque = deque(maxlen=30)
        self._ear_buf      : deque = deque(maxlen=1800)  # 60s EAR history for blink rate
        self._nose_tip_buf : deque = deque(maxlen=300)   # 10s nose-tip positions for sway

        # Face bbox EMA state — pixel coords (x1, y1, x2, y2), None before first detection
        self._bbox_ema   : np.ndarray | None = None

        # Latest JPEG frame for MJPEG stream endpoint
        self._jpeg_lock  = threading.Lock()
        self._latest_jpeg: bytes | None = None

        # Shared state (read by get_message, written by capture thread)
        self._lock        = threading.Lock()
        self._hr          = 0.0
        self._br          = 0.0
        self._hrv         = 0.0
        self._stress      = 0.0
        self._snr         = 0.0
        self._pos_hr      = 0.0
        self._chrom_hr    = 0.0
        self._hr_raw      = 0.0
        self._br_raw      = 0.0
        self._ema_alpha   = 0.25
        self._bvp_window  : list = [0.0] * 32
        self._lighting    = "Good"
        self._face_detected = False
        self._face_bbox   = None
        self._frame_count = 0
        self._ready       = False
        self._blink_timestamps   : list = []  # time.time() of each confirmed blink
        self._blink_frames_below = 0  # consecutive frames below EAR threshold
        self._prev_was_blink     = False
        self._blink_rate         = 0.0   # blinks/min
        self._sway            = 0.0   # normalised motion magnitude
        self._rhythm          = "Unknown"  # "Regular" / "Irregular" / "Unknown"
        self._consecutive_no_face = 0

        # Thread control
        self._running  = False
        self._thread   : threading.Thread | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        self._load_models()
        self._running = True
        self._thread  = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)

    @property
    def is_running(self) -> bool:
        return self._running

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_models(self) -> None:
        providers = ["CPUExecutionProvider"]
        if not self._rppg_path.exists():
            raise FileNotFoundError(f"rPPG ONNX not found: {self._rppg_path}")
        self._rppg_sess  = ort.InferenceSession(str(self._rppg_path), providers=providers)
        self._rppg_input = self._rppg_sess.get_inputs()[0].name

        # Auto-detect model type from ONNX input shape:
        #   EfficientPhys/VitalLens: (B, T, C, H, W) → dim[1] is T (≠ 3)
        #   FactorizePhys / PhysNet: (B, C, T, H, W) → dim[1] = 3, dim[2] = T (even)
        #   PhysFormer:              (B, C, T+1, H, W) → dim[1] = 3, dim[2] = T+1 (odd, e.g. 161)
        inp_shape = self._rppg_sess.get_inputs()[0].shape
        if inp_shape[1] != 3:
            self._model_type       = 'efficientphys'
            self._model_clip_len   = inp_shape[1]   # T dimension
            self._model_input_size = inp_shape[3]
            self._frame_buf    = deque(maxlen=self._model_clip_len + 1)
            self._landmark_buf = deque(maxlen=self._model_clip_len + 1)
        elif inp_shape[2] % 2 == 1:
            # Odd T dimension → PhysFormer (T+1 frames; model diffs internally)
            self._model_type       = 'physformer'
            self._model_clip_len   = inp_shape[2] - 1   # true clip length (e.g. 160)
            self._model_input_size = inp_shape[3]
            self._frame_buf    = deque(maxlen=self._model_clip_len)
            self._landmark_buf = deque(maxlen=self._model_clip_len)
        else:
            # FactorizePhys or PhysNet: exact T frames, raw RGB
            self._model_type       = 'factorizephys'
            self._model_clip_len   = inp_shape[2]
            self._model_input_size = inp_shape[3]
            self._frame_buf    = deque(maxlen=self._model_clip_len)
            self._landmark_buf = deque(maxlen=self._model_clip_len)

        print(f"Model: {self._model_type}  clip={self._model_clip_len}  size={self._model_input_size}  path={self._rppg_path.name}")

        if LIGHTING_PATH.exists():
            self._lighting_sess  = ort.InferenceSession(str(LIGHTING_PATH), providers=providers)
            self._light_input    = self._lighting_sess.get_inputs()[0].name
        else:
            print(f"Warning: lighting ONNX not found at {LIGHTING_PATH} — defaulting to 'Good'")

    # ------------------------------------------------------------------
    # Capture loop (background thread)
    # ------------------------------------------------------------------

    def _capture_loop(self) -> None:
        max_retries = int(os.environ.get("CAMERA_OPEN_RETRIES", "5"))
        retry_delay = float(os.environ.get("CAMERA_OPEN_RETRY_DELAY", "1.0"))

        if self.camera_url:
            src_desc = self.camera_url
        else:
            src_desc = f"camera {self.camera_index}"

        cap = None
        for attempt in range(max_retries):
            if not self._running:
                return  # stop() called during retry — exit cleanly

            if self.camera_url:
                cap = cv2.VideoCapture(self.camera_url)
            else:
                backend = cv2.CAP_V4L2 if _USE_V4L2 else cv2.CAP_ANY
                cap = cv2.VideoCapture(self.camera_index, backend)

            if cap.isOpened():
                break

            cap.release()
            cap = None
            if attempt < max_retries - 1:
                print(f"Cannot open {src_desc} — retrying in {retry_delay}s "
                      f"({attempt + 1}/{max_retries})")
                # Sleep in small increments so stop() is noticed quickly
                deadline = time.time() + retry_delay
                while time.time() < deadline:
                    if not self._running:
                        return
                    time.sleep(0.1)

        if cap is None or not cap.isOpened():
            self._running = False
            raise RuntimeError(f"Cannot open {src_desc} after {max_retries} attempts.")

        # Exposure lock only applies to local DirectShow cameras, not URL streams.
        if self.lock_exposure is not None and not self.camera_url:
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            cap.set(cv2.CAP_PROP_EXPOSURE, float(self.lock_exposure))
            actual = cap.get(cv2.CAP_PROP_EXPOSURE)
            print(f"Exposure lock requested={self.lock_exposure}, camera reports={actual:.1f}")

        raw_fps    = cap.get(cv2.CAP_PROP_FPS) or self.target_fps
        stride     = max(1, round(raw_fps / self.target_fps))
        frame_idx  = 0
        self._fps_t0 = time.time()

        print(f"{src_desc} @ {raw_fps:.0f} fps (stride={stride}) — calibrating actual fps…")

        while self._running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            # Re-assert exposure lock every 150 frames (~5 s) — local cameras only.
            if self.lock_exposure is not None and not self.camera_url and frame_idx % 150 == 0 and frame_idx > 0:
                cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
                cap.set(cv2.CAP_PROP_EXPOSURE, float(self.lock_exposure))

            if frame_idx % stride == 0:
                try:
                    self._process_frame(frame)
                except Exception as exc:
                    print(f"[VitalsEngine] _process_frame error (frame {frame_idx}): {exc}")

                # Measure actual fps over first 90 processed frames (~3 s at 30 fps)
                if not self._fps_calibrated:
                    self._fps_frame_count += 1
                    if self._fps_frame_count >= 90:
                        elapsed = time.time() - self._fps_t0
                        self.target_fps = round(self._fps_frame_count / elapsed, 2)
                        self._fps_calibrated = True
                        print(f"Actual fps measured: {self.target_fps:.1f} Hz")

            frame_idx += 1

        cap.release()

    # ------------------------------------------------------------------
    # Per-frame processing
    # ------------------------------------------------------------------

    def _process_frame(self, frame: np.ndarray) -> None:
        # Encode frame for MJPEG stream (downscale to 640-wide to save bandwidth)
        h, w = frame.shape[:2]
        preview = cv2.resize(frame, (640, int(h * 640 / w))) if w > 640 else frame
        ok, buf = cv2.imencode('.jpg', preview, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok:
            with self._jpeg_lock:
                self._latest_jpeg = buf.tobytes()

        # Track raw mean luminance — measures ISP drift
        self._lum_buf.append(float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean()))

        # 1. Lighting classification (raw frame)
        lighting = self._run_lighting(frame)

        # 2. Face crop + MediaPipe landmarks
        face, detected, bbox, landmarks = self._crop_face_mp(frame)

        # 3. Buffer face crop and landmarks for rPPG
        self._frame_buf.append(face)
        self._landmark_buf.append(landmarks)
        self._light_buf.append(lighting)
        self._frame_count += 1

        # 4. Per-frame blink detection (EAR) and nose-tip sway tracking
        if landmarks is not None:
            h_f, w_f = frame.shape[:2]
            ear = _compute_ear(landmarks)
            self._ear_buf.append(ear)
            is_blink = ear < _runtime_config['ear_blink_thresh']
            if is_blink:
                self._blink_frames_below += 1
            else:
                # Count as a blink only if EAR was below threshold for ≥2 consecutive frames
                # (filters single-frame noise from glasses / low-res landmarks)
                if self._prev_was_blink and self._blink_frames_below >= _EAR_BLINK_MIN_FRAMES:
                    self._blink_timestamps.append(time.time())
                self._blink_frames_below = 0
            self._prev_was_blink = is_blink

            nose = landmarks[_MP_NOSE_TIP]
            self._nose_tip_buf.append([nose[0] / w_f, nose[1] / h_f])

        # 5. Accumulate raw mean RGB for POS/CHROM
        if detected and self._bbox_ema is not None:
            h_f, w_f = frame.shape[:2]
            sx1, sy1, sx2, sy2 = np.clip(
                self._bbox_ema.astype(int), [0, 0, 0, 0], [w_f - 1, h_f - 1, w_f, h_f]
            )
            raw_crop = frame[sy1:sy2, sx1:sx2]
            if raw_crop.size > 0:
                bgr_mean = raw_crop.reshape(-1, 3).mean(axis=0)
                self._rgb_buf.append(bgr_mean[[2, 1, 0]])  # BGR → RGB

        # 6. Run rPPG inference aligned to model clip boundaries (once buffer is full)
        required = self._model_clip_len + 1 if self._model_type == 'efficientphys' else self._model_clip_len
        if (detected and
                len(self._frame_buf) == required and
                self._frame_count % self._model_clip_len == 0):
            self._run_rppg()

        # 7. Update shared state
        majority = self._majority_vote(self._light_buf)
        with self._lock:
            self._lighting      = majority
            self._face_detected = detected
            self._face_bbox     = bbox

        # 8. Reset metrics and clear signal buffers after sustained face absence
        if not detected:
            self._consecutive_no_face += 1
            if self._consecutive_no_face >= 30:  # ~1 s at 30 fps
                self._bvp_buf.clear()
                self._hrv_buf.clear()
                self._rgb_buf.clear()
                self._frame_buf.clear()
                with self._lock:
                    self._hr = self._br = self._hrv = self._stress = 0.0
                    self._snr = self._pos_hr = self._chrom_hr = 0.0
                    self._bvp_window = [0.0] * 32
                    self._ready = False
        else:
            self._consecutive_no_face = 0

    def _run_lighting(self, frame: np.ndarray) -> str:
        if self._lighting_sess is None:
            return "Good"
        inp  = preprocess_for_lighting(frame)
        out  = self._lighting_sess.run(None, {self._light_input: inp})[0][0]  # (3,)
        return LIGHTING_LABELS[int(np.argmax(out))]

    def _crop_face_mp(
        self, frame: np.ndarray
    ) -> tuple[np.ndarray, bool, dict | None, np.ndarray | None]:
        """
        MediaPipe FaceLandmarker face crop with EMA-smoothed bbox.
        Returns (crop, detected, bbox_frac|None, landmarks (478,2)|None).
        """
        h, w = frame.shape[:2]
        rgb      = cv2.cvtColor(frame if frame.shape[2] == 3 else frame[:, :, :3], cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self._mp_timestamp_ms += 33   # ~30fps; must be strictly increasing
        result = self._face_landmarker.detect_for_video(mp_image, self._mp_timestamp_ms)

        if not result.face_landmarks:
            return frame, False, None, None

        lm  = result.face_landmarks[0]   # list of NormalizedLandmark
        pts = np.array([[l.x * w, l.y * h] for l in lm], dtype=np.float32)  # (478, 2)

        x1 = int(pts[:, 0].min())
        y1 = int(pts[:, 1].min())
        x2 = int(pts[:, 0].max())
        y2 = int(pts[:, 1].max())
        pad_x = int((x2 - x1) * 0.15)
        pad_y = int((y2 - y1) * 0.15)
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)

        raw = np.array([x1, y1, x2, y2], dtype=np.float32)
        if self._bbox_ema is None:
            self._bbox_ema = raw
        else:
            self._bbox_ema = _BBOX_ALPHA * self._bbox_ema + (1 - _BBOX_ALPHA) * raw
        sx1, sy1, sx2, sy2 = self._bbox_ema.astype(int)
        sx1, sy1 = max(0, sx1), max(0, sy1)
        sx2, sy2 = min(w, sx2), min(h, sy2)

        crop = frame[sy1:sy2, sx1:sx2]
        bbox = {
            "x": round(sx1 / w, 4), "y": round(sy1 / h, 4),
            "w": round((sx2 - sx1) / w, 4), "h": round((sy2 - sy1) / h, 4),
        }
        return (crop if crop.size > 0 else frame), True, bbox, pts

    @staticmethod
    def _bvp_snr(bvp_arr: np.ndarray, fps: float) -> float:
        """Ratio of cardiac-band Welch peak to median noise floor. Higher = cleaner."""
        if len(bvp_arr) < 64:
            return 0.0
        filtered   = _bandpass(bvp_arr, 0.67, 3.0, fps)
        freqs, psd = welch(filtered, fs=fps, nperseg=min(len(filtered), 150), nfft=2048)
        band       = (freqs >= 0.67) & (freqs <= 3.0)
        if not band.any():
            return 0.0
        return float(psd[band].max() / (np.median(psd[band]) + 1e-9))

    def _run_rppg(self) -> None:
        frames    = list(self._frame_buf)
        landmarks = list(self._landmark_buf)

        if self._model_type == 'factorizephys':
            clip = preprocess_for_factorizephys(frames, self._model_input_size)
        elif self._model_type == 'physformer':
            clip = preprocess_for_physformer(frames, self._model_input_size)
        else:
            frames = _stabilise_clip(frames, landmarks)
            clip   = preprocess_for_rppg(frames, self.brightness_norm, self._model_input_size)

        bvp = self._rppg_sess.run(None, {self._rppg_input: clip})[0][0]  # (T,) or (T-1,)

        self._bvp_buf.extend(bvp.tolist())
        self._hrv_buf.extend(bvp.tolist())

        # Head sway — compute before SNR gate so it's always fresh
        if len(self._nose_tip_buf) >= 30:
            pts  = np.array(self._nose_tip_buf)
            sway = float(np.std(pts, axis=0).mean())
            with self._lock:
                self._sway = round(sway, 4)

        # Blink rate — rolling 5-second window × 12, updated every inference cycle (~5s)
        now = time.time()
        self._blink_timestamps = [t for t in self._blink_timestamps if t > now - 5.0]
        with self._lock:
            self._blink_rate = round(len(self._blink_timestamps) * 12, 1)

        # Warm-up: hold frontend in "calibrating" state until 4 s of BVP data
        min_samples = int(self.target_fps * 4)
        if len(self._bvp_buf) < min_samples:
            print(f"rPPG: calibrating ({len(self._bvp_buf)}/{min_samples} samples)")
            return

        bvp_arr     = np.array(self._bvp_buf)
        live_window = list(self._bvp_buf)[-32:] if len(self._bvp_buf) >= 32 else [0.0] * 32

        snr = self._bvp_snr(bvp_arr, self.target_fps)
        with self._lock:
            self._snr        = round(snr, 2)
            self._bvp_window = [round(float(v), 4) for v in live_window]
            self._ready      = True

        # HRV / rhythm computed from rolling buffer regardless of SNR (validity gate in bvp_to_hrv)
        hrv_arr = np.array(self._hrv_buf)
        hrv     = bvp_to_hrv(hrv_arr, self.target_fps) if len(self._hrv_buf) >= 900 else 0.0
        rhythm  = ibi_to_rhythm(
            bvp_to_ibi(hrv_arr, self.target_fps) if len(self._hrv_buf) >= 900 else None
        )
        if hrv > 0 or rhythm != "Unknown":
            with self._lock:
                a = self._ema_alpha
                if hrv > 0:
                    self._hrv    = round(hrv if self._hrv == 0.0 else a * hrv + (1 - a) * self._hrv, 1)
                    self._stress = round(hrv_to_stress(self._hrv), 1)
                self._rhythm = rhythm

        if snr < _runtime_config['snr_threshold']:
            print(f"rPPG: SNR={snr:.2f} below threshold — holding previous reading")
            return

        # Sway gate — high motion contaminates DiffNorm even after stabilisation
        if self._sway > _runtime_config['sway_thresh']:
            print(f"rPPG: sway={self._sway:.3f} — motion artefact, holding previous reading")
            return

        hr = bvp_to_hr(bvp_arr, self.target_fps)
        br = bvp_to_br(bvp_arr, self.target_fps)

        rgb_arr    = np.array(self._rgb_buf) if len(self._rgb_buf) >= 32 else None
        pos_hr_v   = _pos_hr(rgb_arr,   self.target_fps) if rgb_arr is not None else 0.0
        chrom_hr_v = _chrom_hr(rgb_arr, self.target_fps) if rgb_arr is not None else 0.0

        # If both classical methods strongly disagree with DL (>30 BPM each),
        # DL likely landed on a harmonic — use median of all three as anchor instead.
        classical = [v for v in [pos_hr_v, chrom_hr_v] if v > 0]
        if (len(classical) == 2
                and all(abs(v - hr) > 30 for v in classical)):
            hr = float(np.median([hr] + classical))

        candidates = [hr]
        if pos_hr_v > 0 and abs(pos_hr_v - hr) <= 5.0:
            candidates.append(pos_hr_v)
        if chrom_hr_v > 0 and abs(chrom_hr_v - hr) <= 5.0:
            candidates.append(chrom_hr_v)
        hr_final = float(np.median(candidates))

        with self._lock:
            if self._hr > 0 and abs(hr_final - self._hr) > _runtime_config['hr_jump_thresh']:
                hr_final = self._hr
            a = self._ema_alpha
            self._hr     = round(hr_final if self._hr == 0.0  else a * hr_final + (1 - a) * self._hr, 1)
            self._br     = round(br if self._br == 0.0        else a * br  + (1 - a) * self._br,  1)
            self._pos_hr   = round(pos_hr_v, 1)
            self._chrom_hr = round(chrom_hr_v, 1)
            print(
                f"rPPG: SNR={snr:.1f} sway={self._sway:.3f} HR={self._hr} "
                f"(DL={hr:.1f} POS={pos_hr_v:.1f} CHROM={chrom_hr_v:.1f} "
                f"ensemble={len(candidates)}) BR={self._br} HRV={self._hrv} "
                f"blinks={self._blink_rate:.1f}/min rhythm={self._rhythm}"
            )

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    @staticmethod
    def enumerate_cameras(max_index: int = 5) -> list[dict]:
        backend = cv2.CAP_V4L2 if _USE_V4L2 else cv2.CAP_ANY
        found   = []
        for i in range(max_index):
            cap = cv2.VideoCapture(i, backend)
            if not cap.isOpened():
                cap.release()
                continue
            # USB cameras often return empty frames for the first few reads while
            # the sensor initialises — skip up to 5 frames before giving up.
            ok = False
            for _ in range(5):
                ok, _ = cap.read()
                if ok:
                    break
            cap.release()
            if ok:
                found.append({"index": i, "label": f"Camera {i}"})
        return found

    @staticmethod
    def _majority_vote(buf: deque) -> str:
        if not buf:
            return "Good"
        counts = {}
        for label in buf:
            counts[label] = counts.get(label, 0) + 1
        return max(counts, key=counts.get)

    def get_jpeg(self) -> bytes | None:
        with self._jpeg_lock:
            return self._latest_jpeg

    def get_message(self) -> dict:
        """Thread-safe snapshot of current vitals — called by WebSocket handler."""
        lum_buf = list(self._lum_buf)
        lum_std = round(float(np.std(lum_buf)), 2) if len(lum_buf) >= 10 else None
        with self._lock:
            face = self._face_detected
            return {
                "hr":            self._hr    if face else None,
                "br":            self._br    if face else None,
                "hrv":           self._hrv   if face else None,
                "stress":        self._stress if face else None,
                "snr":           self._snr   if face else None,
                "pos_hr":        self._pos_hr   if face else None,
                "chrom_hr":      self._chrom_hr if face else None,
                "bvp":           list(self._bvp_window) if face else [],
                "lighting":      self._lighting,
                "face_detected": face,
                "face_bbox":     self._face_bbox,
                "timestamp":     time.time(),
                "ready":         self._ready and face,
                "lum_std":       lum_std,
                "blink_rate":    self._blink_rate,
                "sway":          self._sway,
                "rhythm":        self._rhythm,
            }
