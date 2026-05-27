"""
VitalLens Preprocessing Pipeline
=================================
Prepares webcam frames for rPPG signal extraction.

Pipeline steps (per frame):
  1. CLAHE          — contrast enhancement in LAB space (luminance channel only)
  2. White Balance  — gray-world assumption correction
  3. Gamma          — adaptive power-law correction for dark frames
  4. Temporal Denoise — average across 3-frame buffer
  5. Face ROI Crop  — OpenCV Haar cascade → forehead + cheek patches

Usage:
    pipeline = PreprocessingPipeline()
    result = pipeline.process(frame, lighting_class="poor")
    # result.corrected_frame  — preprocessed BGR frame
    # result.roi_patches      — {"forehead": ..., "left_cheek": ..., "right_cheek": ...}
    # result.face_bbox        — (x, y, w, h) or None if no face detected
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

_face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


@dataclass
class PipelineResult:
    corrected_frame: np.ndarray          # preprocessed BGR frame (full size)
    roi_patches: Dict[str, np.ndarray]   # cropped face region patches
    face_bbox: Optional[Tuple[int, int, int, int]]  # (x, y, w, h)
    lighting_class: str                  # "good" | "poor" | "mixed"
    gamma_applied: float                 # actual gamma used (1.0 = no change)
    face_detected: bool


class PreprocessingPipeline:
    """
    Stateful preprocessing pipeline — holds a frame buffer for temporal denoising.

    Parameters
    ----------
    buffer_size : int
        Number of frames to average for temporal denoising (default 3).
    clahe_clip_limit : float
        CLAHE clip limit — higher = more contrast enhancement (default 2.0).
    clahe_tile_grid : tuple
        CLAHE tile grid size (default (8, 8)).
    target_patch_size : tuple
        Output size for ROI patches fed to the rPPG model (default (72, 72)).
    """

    def __init__(
        self,
        buffer_size: int = 3,
        clahe_clip_limit: float = 2.0,
        clahe_tile_grid: Tuple[int, int] = (8, 8),
        target_patch_size: Tuple[int, int] = (72, 72),
    ):
        self.buffer_size = buffer_size
        self.target_patch_size = target_patch_size
        self._clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_tile_grid)
        self._frame_buffer: collections.deque = collections.deque(maxlen=buffer_size)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, frame: np.ndarray, lighting_class: str = "good") -> PipelineResult:
        """
        Process a single BGR frame.

        Parameters
        ----------
        frame : np.ndarray
            Input BGR frame from webcam (H x W x 3, uint8).
        lighting_class : str
            Output of the lighting classifier: "good", "poor", or "mixed".
            Gamma correction is only applied on "poor" or "mixed".
        """
        lighting_class = lighting_class.lower().strip()

        frame = self._apply_clahe(frame)
        frame = self._apply_white_balance(frame)

        gamma_value = 1.0
        if lighting_class in ("poor", "mixed"):
            frame, gamma_value = self._apply_gamma(frame)

        frame = self._apply_temporal_denoise(frame)

        face_bbox, roi_patches, face_detected = self._extract_face_roi(frame)

        return PipelineResult(
            corrected_frame=frame,
            roi_patches=roi_patches,
            face_bbox=face_bbox,
            lighting_class=lighting_class,
            gamma_applied=gamma_value,
            face_detected=face_detected,
        )

    def reset(self) -> None:
        """Clear the temporal buffer. Call when switching subjects or after a pause."""
        self._frame_buffer.clear()

    # ------------------------------------------------------------------
    # Step 1: CLAHE
    # ------------------------------------------------------------------

    def _apply_clahe(self, frame: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        l_enhanced = self._clahe.apply(l_channel)
        return cv2.cvtColor(cv2.merge([l_enhanced, a_channel, b_channel]), cv2.COLOR_LAB2BGR)

    # ------------------------------------------------------------------
    # Step 2: White Balance
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_white_balance(frame: np.ndarray) -> np.ndarray:
        f = frame.astype(np.float32)
        mb, mg, mr = f[:, :, 0].mean(), f[:, :, 1].mean(), f[:, :, 2].mean()
        overall = (mb + mg + mr) / 3.0
        if mb > 0: f[:, :, 0] *= overall / mb
        if mg > 0: f[:, :, 1] *= overall / mg
        if mr > 0: f[:, :, 2] *= overall / mr
        return np.clip(f, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------
    # Step 3: Gamma Correction
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_gamma(frame: np.ndarray) -> Tuple[np.ndarray, float]:
        lum = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean()
        gamma = 0.5 if lum < 40 else 0.7 if lum < 80 else 0.85 if lum < 120 else 1.0
        if gamma == 1.0:
            return frame, gamma
        lut = np.array([min(255, int((i / 255.0) ** gamma * 255)) for i in range(256)], dtype=np.uint8)
        return cv2.LUT(frame, lut), gamma

    # ------------------------------------------------------------------
    # Step 4: Temporal Denoising
    # ------------------------------------------------------------------

    def _apply_temporal_denoise(self, frame: np.ndarray) -> np.ndarray:
        self._frame_buffer.append(frame.astype(np.float32))
        if len(self._frame_buffer) < self.buffer_size:
            return frame
        return np.stack(list(self._frame_buffer)).mean(axis=0).astype(np.uint8)

    # ------------------------------------------------------------------
    # Step 5: Face ROI Extraction
    # ------------------------------------------------------------------

    def _extract_face_roi(
        self,
        frame: np.ndarray,
    ) -> Tuple[Optional[Tuple[int, int, int, int]], Dict[str, np.ndarray], bool]:
        h, w = frame.shape[:2]
        target = self.target_patch_size
        empty = {
            "forehead":    np.zeros((*target, 3), dtype=np.uint8),
            "left_cheek":  np.zeros((*target, 3), dtype=np.uint8),
            "right_cheek": np.zeros((*target, 3), dtype=np.uint8),
        }

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = _face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )

        if len(faces) == 0:
            return None, empty, False

        x, y, fw, fh = faces[0]

        def _crop(y1, y2, x1, x2):
            y1, y2 = max(0, y1), min(h, y2)
            x1, x2 = max(0, x1), min(w, x2)
            patch = frame[y1:y2, x1:x2]
            if patch.size == 0:
                patch = frame[y:y + fh, x:x + fw]
            return cv2.resize(patch, target, interpolation=cv2.INTER_LINEAR)

        roi_patches = {
            "forehead":    _crop(y,                  y + int(fh * 0.35), x + int(fw * 0.1),  x + int(fw * 0.9)),
            "left_cheek":  _crop(y + int(fh * 0.4),  y + int(fh * 0.7),  x,                  x + int(fw * 0.35)),
            "right_cheek": _crop(y + int(fh * 0.4),  y + int(fh * 0.7),  x + int(fw * 0.65), x + fw),
        }

        return (x, y, fw, fh), roi_patches, True
