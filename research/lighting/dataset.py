"""
Lighting Classifier Dataset
============================
Builds a labelled dataset of Good / Poor / Mixed lighting frames
for training MobileNetV3-Small.

Supports two data sources:
  1. MMPD (.mat files) — preferred, has built-in lighting metadata labels
     LED-high / Nature(bright) → Good
     LED-low                   → Poor
     Incandescent              → Mixed

  2. Synthetic fallback (UBFC-rPPG frames + augmentation)
     Use while waiting for MMPD access.
     UBFC frames → Good
     UBFC + heavy darkening → Poor
     UBFC + moderate darkening + gradient → Mixed

Usage:
    # --- Option A: MMPD (when you have access) ---
    builder = MMPDBuilder("data/mmpd")
    builder.build(save_cache="data/lighting_labels.csv")

    # --- Option B: Synthetic fallback from UBFC ---
    builder = SyntheticBuilder("data/ubfc_rppg")
    builder.build(save_cache="data/lighting_labels.csv")

    # --- Train ---
    dataset = LightingDataset(cache_csv="data/lighting_labels.csv", augment=True)
"""

from __future__ import annotations

import csv
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

try:
    import scipy.io as sio
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


LABEL_MAP = {"Good": 0, "Poor": 1, "Mixed": 2}
LABEL_NAMES = ["Good", "Poor", "Mixed"]

# MMPD lighting condition → our label
MMPD_LIGHT_MAP = {
    "LED-high":     "Good",
    "LED-low":      "Poor",
    "Incandescent": "Mixed",
    "Nature":       "Good",   # will refine with luminance check below
}


# ---------------------------------------------------------------------------
# Option A: MMPD Builder (.mat files with built-in lighting metadata)
# ---------------------------------------------------------------------------

class MMPDBuilder:
    """
    Extracts labelled frames from MMPD .mat files.

    Each .mat file contains:
      - video   : [T, W, H, C] uint8 rendered frames at 320x240 (or 80x60 mini)
      - GT_ppg  : [T] float32 PPG signal
      - metadata: dict with light condition (LED-high/low, Incandescent, Nature)

    The lighting label comes directly from the MMPD metadata — no guessing needed.
    """

    def __init__(self, mmpd_root: str | Path, frame_stride: int = 15):
        if not SCIPY_AVAILABLE:
            raise ImportError("scipy is required for MMPD .mat files: pip install scipy")
        self.root = Path(mmpd_root)
        self.frame_stride = frame_stride

    def build(
        self,
        max_per_class: int = 4000,
        save_cache: str | Path = "data/lighting_labels.csv",
    ) -> List[dict]:
        print("Building lighting dataset from MMPD...")
        all_samples: List[dict] = []

        subject_dirs = sorted([d for d in self.root.iterdir() if d.is_dir()])
        if not subject_dirs:
            raise FileNotFoundError(f"No subject directories in {self.root}")

        for subj_dir in subject_dirs:
            mat_files = sorted(subj_dir.glob("*.mat"))
            for mat_path in mat_files:
                samples = self._process_mat(mat_path)
                all_samples.extend(samples)

        return self._balance_and_save(all_samples, max_per_class, save_cache)

    def _process_mat(self, mat_path: Path) -> List[dict]:
        try:
            mat = sio.loadmat(str(mat_path))
        except Exception as e:
            print(f"  Warning: could not load {mat_path.name}: {e}")
            return []

        video = mat.get("video")  # [T, W, H, C] or [T, H, W, C]
        if video is None:
            return []

        # Determine lighting label from metadata
        light_label = self._get_light_label(mat, mat_path)

        samples = []
        n_frames = video.shape[0]
        for t in range(0, n_frames, self.frame_stride):
            frame = video[t]  # (H, W, C) or similar
            if frame.ndim == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.shape[-1] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Refine "Nature" label with actual luminance
            if light_label == "Good":
                gray_mean = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean()
                if gray_mean < 60:
                    light_label = "Poor"
                elif gray_mean < 100:
                    light_label = "Mixed"

            samples.append({
                "path": f"{mat_path}::{t}",
                "label": light_label,
                "mean_lum": round(float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean()), 2),
                "std_lum": round(float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).std()), 2),
            })
        return samples

    @staticmethod
    def _get_light_label(mat: dict, mat_path: Path) -> str:
        """Extract lighting condition from MMPD metadata."""
        # Try metadata field first
        for key in ["light", "lighting", "light_condition"]:
            val = mat.get(key)
            if val is not None:
                val_str = str(val).strip()
                for mmpd_key, our_label in MMPD_LIGHT_MAP.items():
                    if mmpd_key.lower() in val_str.lower():
                        return our_label

        # Fall back to filename — MMPD files often encode condition in name
        name = mat_path.stem.lower()
        if "high" in name:
            return "Good"
        elif "low" in name:
            return "Poor"
        elif "incandescent" in name:
            return "Mixed"

        return "Mixed"  # safe default

    @staticmethod
    def _balance_and_save(
        all_samples: List[dict],
        max_per_class: int,
        save_cache: str | Path,
    ) -> List[dict]:
        by_class: Dict[str, List] = {k: [] for k in LABEL_NAMES}
        for s in all_samples:
            by_class[s["label"]].append(s)

        balanced = []
        for label, samples in by_class.items():
            random.shuffle(samples)
            taken = samples[:max_per_class]
            balanced.extend(taken)
            print(f"  '{label}': {len(taken)} samples (available: {len(samples)})")

        random.shuffle(balanced)
        save_path = Path(save_cache)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["path", "label", "mean_lum", "std_lum"])
            writer.writeheader()
            writer.writerows(balanced)

        print(f"Saved {len(balanced)} samples to {save_path}")
        return balanced


# ---------------------------------------------------------------------------
# Option B: Synthetic Builder (fallback while waiting for MMPD)
# ---------------------------------------------------------------------------

class SyntheticBuilder:
    """
    Creates Good / Poor / Mixed lighting examples from UBFC-rPPG frames
    using synthetic augmentation.

    Good  → original UBFC frame (lab-controlled, well-lit)
    Poor  → original × dark factor (0.15–0.35) + optional harsh shadow gradient
    Mixed → original × moderate factor (0.45–0.70) + slight colour cast

    This is valid because the lighting classifier learns luminance and contrast
    statistics — these augmentations faithfully reproduce those statistics.
    """

    def __init__(self, ubfc_root: str | Path, frame_stride: int = 20):
        self.root = Path(ubfc_root)
        self.frame_stride = frame_stride

    def build(
        self,
        max_per_class: int = 3000,
        save_cache: str | Path = "data/lighting_labels_synthetic.csv",
    ) -> List[dict]:
        print("Building synthetic lighting dataset from UBFC-rPPG...")

        # Extract frames from all UBFC subjects
        ubfc_frames = self._extract_ubfc_frames()
        if not ubfc_frames:
            raise FileNotFoundError(f"No UBFC frames found in {self.root}")

        print(f"  Extracted {len(ubfc_frames)} source frames from UBFC-rPPG")

        # Save frames to disk so the dataset can re-load them
        frames_dir = Path(save_cache).parent / "synthetic_frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        samples = []
        n = min(len(ubfc_frames), max_per_class)

        for i, frame in enumerate(ubfc_frames[:n]):
            # Good — save original
            good_path = frames_dir / f"good_{i:05d}.jpg"
            cv2.imwrite(str(good_path), frame)
            samples.append({"path": str(good_path), "label": "Good",
                            "mean_lum": round(float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean()), 2),
                            "std_lum": round(float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).std()), 2)})

            # Poor — heavy darkening + optional shadow
            poor_frame = self._make_poor(frame)
            poor_path = frames_dir / f"poor_{i:05d}.jpg"
            cv2.imwrite(str(poor_path), poor_frame)
            samples.append({"path": str(poor_path), "label": "Poor",
                            "mean_lum": round(float(cv2.cvtColor(poor_frame, cv2.COLOR_BGR2GRAY).mean()), 2),
                            "std_lum": round(float(cv2.cvtColor(poor_frame, cv2.COLOR_BGR2GRAY).std()), 2)})

            # Mixed — moderate darkening + colour cast
            mixed_frame = self._make_mixed(frame)
            mixed_path = frames_dir / f"mixed_{i:05d}.jpg"
            cv2.imwrite(str(mixed_path), mixed_frame)
            samples.append({"path": str(mixed_path), "label": "Mixed",
                            "mean_lum": round(float(cv2.cvtColor(mixed_frame, cv2.COLOR_BGR2GRAY).mean()), 2),
                            "std_lum": round(float(cv2.cvtColor(mixed_frame, cv2.COLOR_BGR2GRAY).std()), 2)})

        save_path = Path(save_cache)
        with open(save_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["path", "label", "mean_lum", "std_lum"])
            writer.writeheader()
            writer.writerows(samples)

        counts = Counter(s["label"] for s in samples)
        print(f"Saved {len(samples)} synthetic samples — {dict(counts)}")
        print(f"Frames saved to: {frames_dir}")
        return samples

    def _extract_ubfc_frames(self) -> List[np.ndarray]:
        """Extract one frame every frame_stride from all UBFC videos."""
        frames = []
        for subj_dir in sorted(self.root.iterdir()):
            if not subj_dir.is_dir():
                continue
            vid_candidates = sorted(subj_dir.glob("*.avi"))
            if not vid_candidates:
                continue
            vid_path = vid_candidates[0]
            cap = cv2.VideoCapture(str(vid_path))
            idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if idx % self.frame_stride == 0:
                    frames.append(frame)
                idx += 1
            cap.release()
        return frames

    @staticmethod
    def _make_poor(frame: np.ndarray) -> np.ndarray:
        """Heavy darkening + optional one-sided shadow gradient."""
        dark_factor = random.uniform(0.15, 0.35)
        result = (frame.astype(np.float32) * dark_factor)

        # 50% chance of adding a harsh shadow on one side
        if random.random() < 0.5:
            h, w = frame.shape[:2]
            shadow_side = random.choice(["left", "right", "top"])
            mask = np.ones((h, w), dtype=np.float32)
            if shadow_side == "left":
                mask[:, : w // 2] *= random.uniform(0.1, 0.3)
            elif shadow_side == "right":
                mask[:, w // 2 :] *= random.uniform(0.1, 0.3)
            else:
                mask[: h // 2, :] *= random.uniform(0.1, 0.3)
            result *= mask[:, :, np.newaxis]

        return np.clip(result, 0, 255).astype(np.uint8)

    @staticmethod
    def _make_mixed(frame: np.ndarray) -> np.ndarray:
        """Moderate darkening + warm or cool colour cast."""
        factor = random.uniform(0.45, 0.70)
        result = frame.astype(np.float32) * factor

        # Warm (yellowish) or cool (bluish) cast
        cast_type = random.choice(["warm", "cool", "none"])
        if cast_type == "warm":
            result[:, :, 2] = np.clip(result[:, :, 2] * 1.2, 0, 255)  # boost red
            result[:, :, 0] = np.clip(result[:, :, 0] * 0.85, 0, 255)  # reduce blue
        elif cast_type == "cool":
            result[:, :, 0] = np.clip(result[:, :, 0] * 1.2, 0, 255)  # boost blue
            result[:, :, 2] = np.clip(result[:, :, 2] * 0.85, 0, 255)  # reduce red

        return np.clip(result, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# PyTorch Dataset (shared by both builders)
# ---------------------------------------------------------------------------

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

TRAIN_TRANSFORMS = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

VAL_TRANSFORMS = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


class LightingDataset(Dataset):
    """
    PyTorch Dataset for the lighting classifier.
    Works with any CSV cache produced by MMPDBuilder or SyntheticBuilder.
    """

    def __init__(self, cache_csv: str | Path, augment: bool = False):
        self.transform = TRAIN_TRANSFORMS if augment else VAL_TRANSFORMS
        self.samples: List[dict] = []

        with open(cache_csv, newline="") as f:
            for row in csv.DictReader(f):
                self.samples.append(row)

        counts = Counter(s["label"] for s in self.samples)
        print(f"LightingDataset loaded: {len(self.samples)} samples — {dict(counts)}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        sample = self.samples[idx]
        label_idx = LABEL_MAP[sample["label"]]
        frame = self._load_frame(sample["path"])
        if frame is None:
            frame = np.zeros((224, 224, 3), dtype=np.uint8)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.transform(frame_rgb), label_idx

    @staticmethod
    def _load_frame(path_str: str) -> Optional[np.ndarray]:
        """Load from image path or mat_path::frame_idx reference."""
        if "::" in path_str:
            mat_path, t_str = path_str.rsplit("::", 1)
            if not SCIPY_AVAILABLE:
                return None
            try:
                mat = sio.loadmat(mat_path)
                video = mat.get("video")
                if video is None:
                    return None
                frame = video[int(t_str)]
                if frame.shape[-1] == 3:
                    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                return frame
            except Exception:
                return None
        return cv2.imread(path_str)
