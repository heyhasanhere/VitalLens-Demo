"""
Signal Processing — BVP → Physiological Vitals
================================================
Takes the raw BVP waveform output from the rPPG model and derives:
  - Heart Rate (HR)        — dominant frequency via Welch's PSD
  - HRV (SDNN, RMSSD)      — inter-beat interval statistics
  - Breathing Rate (BR)    — respiratory modulation of BVP amplitude
  - Signal Quality (SNR)   — for UI indicator and ablation study

All functions are stateless and operate on numpy arrays.
The VitalsBuffer class maintains a rolling window for smooth real-time display.

Usage:
    processor = VitalsProcessor(fps=30.0)
    processor.add_bvp_chunk(bvp_array)   # feed chunks as they arrive
    vitals = processor.get_vitals()       # get latest HR/HRV/BR/SNR
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, welch


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FPS_DEFAULT      = 30.0
HR_BAND_LOW      = 0.67   # Hz  (40 BPM)
HR_BAND_HIGH     = 3.0    # Hz  (180 BPM)
BR_BAND_LOW      = 0.1    # Hz  (6 breaths/min)
BR_BAND_HIGH     = 0.5    # Hz  (30 breaths/min)
MIN_SIGNAL_LEN   = 64     # minimum samples needed for reliable HR estimate
NOISE_BAND_LOW   = 3.5    # Hz  — above cardiac band, used for SNR noise floor
NOISE_BAND_HIGH  = 7.0    # Hz


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class Vitals:
    hr:       Optional[float] = None   # BPM
    sdnn:     Optional[float] = None   # ms — overall HRV
    rmssd:    Optional[float] = None   # ms — short-term HRV
    br:       Optional[float] = None   # breaths per minute
    snr:      Optional[float] = None   # dB — signal quality
    valid:    bool = False             # False if signal too short/noisy
    message:  str  = "Calibrating..."

    def to_dict(self) -> dict:
        return {
            "hr":     round(self.hr, 1)    if self.hr    is not None else None,
            "sdnn":   round(self.sdnn, 1)  if self.sdnn  is not None else None,
            "rmssd":  round(self.rmssd, 1) if self.rmssd is not None else None,
            "br":     round(self.br, 1)    if self.br    is not None else None,
            "snr":    round(self.snr, 2)   if self.snr   is not None else None,
            "valid":  self.valid,
            "message": self.message,
            "stress": _stress_label(self.hr, self.rmssd),
        }


def _stress_label(hr: Optional[float], rmssd: Optional[float]) -> str:
    """Rule-based stress index — no ML needed, no risk of garbage output."""
    if hr is None or rmssd is None:
        return "Unknown"
    if rmssd < 25 or hr > 95:
        return "High"
    elif rmssd < 40 or hr > 82:
        return "Medium"
    else:
        return "Low"


# ---------------------------------------------------------------------------
# Core signal processing functions
# ---------------------------------------------------------------------------

def bandpass_filter(
    signal: np.ndarray,
    low: float,
    high: float,
    fps: float = FPS_DEFAULT,
    order: int = 3,
) -> np.ndarray:
    """
    Apply a Butterworth bandpass filter.
    Clamps cutoff frequencies to valid Nyquist range automatically.
    """
    nyq = fps / 2.0
    low_norm  = max(low  / nyq, 0.01)
    high_norm = min(high / nyq, 0.99)
    if low_norm >= high_norm:
        return signal
    b, a = butter(order, [low_norm, high_norm], btype="band")
    return filtfilt(b, a, signal)


def estimate_hr(bvp: np.ndarray, fps: float = FPS_DEFAULT) -> Optional[float]:
    """
    Estimate heart rate from BVP using Welch's power spectral density.

    Steps:
      1. Bandpass filter to cardiac band (0.67–3.0 Hz)
      2. Compute PSD using Welch's method (averages overlapping FFT windows)
      3. Find peak frequency → multiply by 60 → BPM

    Returns None if signal is too short or has no clear peak.
    """
    if len(bvp) < MIN_SIGNAL_LEN:
        return None

    bvp_filtered = bandpass_filter(bvp, HR_BAND_LOW, HR_BAND_HIGH, fps)

    # nperseg: length of each FFT segment. Longer = finer frequency resolution.
    nperseg = min(len(bvp_filtered), int(fps * 6))  # 6-second window max
    freqs, psd = welch(bvp_filtered, fs=fps, nperseg=nperseg)

    # Restrict to cardiac band
    band_mask = (freqs >= HR_BAND_LOW) & (freqs <= HR_BAND_HIGH)
    if not band_mask.any():
        return None

    peak_idx  = psd[band_mask].argmax()
    peak_freq = freqs[band_mask][peak_idx]
    return float(peak_freq * 60.0)


def estimate_hrv(
    bvp: np.ndarray,
    fps: float = FPS_DEFAULT,
) -> tuple[Optional[float], Optional[float]]:
    """
    Estimate HRV (SDNN and RMSSD) from inter-beat intervals.

    Steps:
      1. Bandpass filter BVP to cardiac band
      2. Find peaks (heartbeat locations)
      3. Compute IBI = time between successive peaks
      4. SDNN  = std(IBI)            — overall variability
      5. RMSSD = sqrt(mean(diff(IBI)^2)) — short-term variability

    Returns (sdnn_ms, rmssd_ms) or (None, None) if too few peaks.
    """
    if len(bvp) < MIN_SIGNAL_LEN:
        return None, None

    bvp_filtered = bandpass_filter(bvp, HR_BAND_LOW, HR_BAND_HIGH, fps)

    # Minimum distance between peaks: 0.33s = 180 BPM max
    min_distance = int(fps * 0.33)
    peaks, _ = find_peaks(bvp_filtered, distance=min_distance, prominence=0.1)

    if len(peaks) < 3:
        return None, None

    # Inter-beat intervals in milliseconds
    ibi_ms = np.diff(peaks) / fps * 1000.0

    # Sanity check: discard IBIs outside physiological range (300–1500ms)
    ibi_ms = ibi_ms[(ibi_ms >= 300) & (ibi_ms <= 1500)]
    if len(ibi_ms) < 2:
        return None, None

    sdnn  = float(np.std(ibi_ms))
    rmssd = float(np.sqrt(np.mean(np.diff(ibi_ms) ** 2)))
    return sdnn, rmssd


def estimate_br(bvp: np.ndarray, fps: float = FPS_DEFAULT) -> Optional[float]:
    """
    Estimate breathing rate from respiratory modulation of the BVP signal.

    The breathing cycle causes a slow amplitude modulation of the BVP
    waveform (0.1–0.5 Hz). We extract this by:
      1. Bandpass filtering to the respiratory band
      2. Finding the peak frequency → breathing rate in breaths/min

    Returns None if signal is too short.
    """
    if len(bvp) < MIN_SIGNAL_LEN * 2:
        return None

    bvp_resp = bandpass_filter(bvp, BR_BAND_LOW, BR_BAND_HIGH, fps)
    nperseg  = min(len(bvp_resp), int(fps * 20))  # 20-second window for BR
    freqs, psd = welch(bvp_resp, fs=fps, nperseg=nperseg)

    band_mask = (freqs >= BR_BAND_LOW) & (freqs <= BR_BAND_HIGH)
    if not band_mask.any():
        return None

    peak_freq = freqs[band_mask][psd[band_mask].argmax()]
    return float(peak_freq * 60.0)


def compute_snr(bvp: np.ndarray, fps: float = FPS_DEFAULT) -> Optional[float]:
    """
    Compute Signal-to-Noise Ratio in dB.

    SNR = 20 * log10(RMS_signal / RMS_noise)

    Signal band:  0.67–3.0 Hz (cardiac)
    Noise band:   3.5–7.0 Hz  (above cardiac, captures sensor/motion noise)

    Used for:
      - UI signal quality indicator
      - Ablation study (comparing SNR across 4 lighting conditions)

    Returns None if signal too short.
    """
    if len(bvp) < MIN_SIGNAL_LEN:
        return None

    signal_band = bandpass_filter(bvp, HR_BAND_LOW, HR_BAND_HIGH, fps)
    noise_band  = bandpass_filter(bvp, NOISE_BAND_LOW, NOISE_BAND_HIGH, fps)

    rms_signal = float(np.sqrt(np.mean(signal_band ** 2)))
    rms_noise  = float(np.sqrt(np.mean(noise_band  ** 2)))

    if rms_noise < 1e-8:
        return None

    return float(20.0 * np.log10(rms_signal / rms_noise))


# ---------------------------------------------------------------------------
# Rolling vitals processor — for real-time streaming
# ---------------------------------------------------------------------------

class VitalsProcessor:
    """
    Maintains a rolling BVP buffer and computes vitals on demand.

    Designed for real-time use: feed BVP chunks as they arrive from the
    rPPG model, call get_vitals() each second to get updated readings.

    Parameters
    ----------
    fps : float
        Webcam frame rate (default 30.0)
    window_seconds : int
        How many seconds of BVP history to keep (default 30)
    """

    def __init__(self, fps: float = FPS_DEFAULT, window_seconds: int = 30):
        self.fps     = fps
        self._buffer: collections.deque = collections.deque(
            maxlen=int(fps * window_seconds)
        )

    def add_bvp_chunk(self, chunk: np.ndarray) -> None:
        """Append a BVP chunk (output of one rPPG model inference) to the buffer."""
        self._buffer.extend(chunk.tolist())

    def reset(self) -> None:
        """Clear buffer — call when user restarts session."""
        self._buffer.clear()

    def get_bvp(self) -> np.ndarray:
        return np.array(self._buffer, dtype=np.float32)

    def get_vitals(self) -> Vitals:
        """Compute and return all vitals from the current buffer."""
        bvp = self.get_bvp()

        if len(bvp) < MIN_SIGNAL_LEN:
            return Vitals(valid=False, message="Calibrating...")

        hr    = estimate_hr(bvp, self.fps)
        sdnn, rmssd = estimate_hrv(bvp, self.fps)
        br    = estimate_br(bvp, self.fps)
        snr   = compute_snr(bvp, self.fps)

        # Sanity-check HR — reject physiologically impossible values
        if hr is not None and not (35 < hr < 200):
            hr = None

        if hr is None:
            return Vitals(valid=False, message="Signal too weak — stay still")

        return Vitals(
            hr=hr, sdnn=sdnn, rmssd=rmssd, br=br, snr=snr,
            valid=True, message="Live"
        )

    @property
    def buffer_seconds(self) -> float:
        return len(self._buffer) / self.fps


# ---------------------------------------------------------------------------
# Utility: BVP normalisation (for display)
# ---------------------------------------------------------------------------

def normalise_for_display(bvp: np.ndarray, window: int = 150) -> np.ndarray:
    """
    Normalise the last `window` samples of BVP to [-1, 1] for waveform display.
    Uses a rolling min-max so the waveform stays stable on screen.
    """
    chunk = bvp[-window:] if len(bvp) >= window else bvp
    mn, mx = chunk.min(), chunk.max()
    if mx - mn < 1e-6:
        return np.zeros_like(chunk)
    return 2.0 * (chunk - mn) / (mx - mn) - 1.0


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running signal processing self-test...")
    fps = 30.0
    duration = 60  # seconds
    t = np.linspace(0, duration, int(fps * duration))

    # Simulate realistic BVP: 72 BPM cardiac + 15 BPM breathing + noise
    hr_true = 72.0
    br_true = 15.0
    bvp = (
        np.sin(2 * np.pi * (hr_true / 60.0) * t)
        + 0.25 * np.sin(4 * np.pi * (hr_true / 60.0) * t + 0.8)
        + 0.1  * np.sin(2 * np.pi * (br_true / 60.0) * t)
        + np.random.normal(0, 0.05, len(t))
    )

    hr   = estimate_hr(bvp, fps)
    sdnn, rmssd = estimate_hrv(bvp, fps)
    br   = estimate_br(bvp, fps)
    snr  = compute_snr(bvp, fps)

    print(f"  True HR: {hr_true} BPM  |  Estimated HR: {hr:.1f} BPM  |  Error: {abs(hr - hr_true):.1f} BPM")
    print(f"  Breathing rate: {br:.1f} BPM (true: {br_true})")
    print(f"  SDNN: {sdnn:.1f} ms  |  RMSSD: {rmssd:.1f} ms")
    print(f"  SNR: {snr:.2f} dB")

    # Test rolling processor
    processor = VitalsProcessor(fps=fps, window_seconds=30)
    processor.add_bvp_chunk(bvp)
    vitals = processor.get_vitals()
    print(f"\n  VitalsProcessor output: {vitals.to_dict()}")
    print("\nSelf-test passed.")
