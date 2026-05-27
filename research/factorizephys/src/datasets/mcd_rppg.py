import json
import random
from pathlib import Path

import av
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from scipy.signal import butter, filtfilt

ROOT     = Path('/mnt/sata-ssd/rppg_project/rppg_dataset/MCD-rPPG')
DB       = ROOT / 'db.csv'
VID_DIR  = ROOT / 'video'
SYNC_DIR = ROOT / 'ppg_sync'

CAMERAS  = ['FullHDwebcam', 'USBVideo', 'IriunWebcam']

HR_MIN   = 40
HR_MAX   = 200
PPG_FLAT_STD  = 5.0
PPG_CLIP_FRAC = 0.20
MIN_FRAMES    = 160


def _load_ppg_sync(pid, cam, step):
    path = SYNC_DIR / f'{pid}_{cam}_{step}.txt'
    if not path.exists():
        return None
    return np.array([int(l.split()[0]) for l in open(path)], dtype=np.float32)


def _is_good_ppg(ppg):
    if ppg is None or len(ppg) < MIN_FRAMES:
        return False
    if ppg.std() < PPG_FLAT_STD:
        return False
    clipped = (np.mean(ppg == 0) + np.mean(ppg == 255))
    if clipped > PPG_CLIP_FRAC:
        return False
    return True


def _bandpass(signal, fs=30.0, lo=0.6, hi=4.0):
    nyq = fs / 2.0
    b, a = butter(4, [lo / nyq, hi / nyq], btype='bandpass')
    return filtfilt(b, a, signal.astype(np.float64)).astype(np.float32)


def _read_frames_av(path, target_size=72):
    """Decode all frames from an AVI via PyAV, resize to target_size x target_size."""
    container = av.open(str(path))
    frames = []
    for frame in container.decode(video=0):
        img = frame.to_ndarray(format='rgb24')
        # Simple resize via numpy slicing after converting to PIL is slow;
        # we use av's built-in resampler for speed.
        frames.append(img)
    container.close()
    if not frames:
        return None
    # Resize via PIL (imported lazily to avoid top-level dependency)
    from PIL import Image
    out = []
    for f in frames:
        img = Image.fromarray(f).resize((target_size, target_size), Image.BILINEAR)
        out.append(np.array(img))
    return np.stack(out)  # [T, H, W, 3] uint8


class MCDrPPGDataset(Dataset):
    """
    MCD-rPPG clip dataset for FactorizePhys training.

    Each item is:
        frames : FloatTensor [3, clip_len+1, H, W]  (normalised 0-1)
        ppg    : FloatTensor [clip_len]              (bandpass-filtered, z-scored)
        hr     : float                               (scalar HR in bpm from pulse column)

    Args:
        split_json : path to mcd_split.json  (produced by notebook 08)
        split      : 'train' or 'held_out'
        cameras    : list of cameras to include (default: all 3)
        steps      : list of steps to include  (default: ['before', 'after'])
        clip_len   : frames per clip (default 160, matching FactorizePhys input T-1)
        stride     : stride between clips within a video (default 160, non-overlapping)
        target_size: spatial resize (default 72)
        face_detector: callable(frame_rgb_hwc) -> (x1,y1,x2,y2) or None.
                        If None, full frame is used.
        transform  : optional video augmentation callable applied to [T,H,W,3] uint8 array
        seed       : random seed for clip sampling reproducibility
    """

    def __init__(
        self,
        split_json,
        split='train',
        cameras=None,
        steps=None,
        clip_len=160,
        stride=160,
        target_size=72,
        face_detector=None,
        transform=None,
        seed=42,
    ):
        self.clip_len      = clip_len
        self.stride        = stride
        self.target_size   = target_size
        self.face_detector = face_detector
        self.transform     = transform
        self.cameras       = cameras or CAMERAS
        self.steps         = steps or ['before', 'after']

        with open(split_json) as f:
            split_data = json.load(f)
        subject_ids = set(split_data[split])

        df = pd.read_csv(DB)
        df = df[df['patient_id'].isin(subject_ids)]
        df = df[df['camera'].isin(self.cameras)]
        df = df[df['step'].isin(self.steps)]
        df = df[(df['pulse'] >= HR_MIN) & (df['pulse'] <= HR_MAX)]

        self.clips = []  # list of (video_path, ppg_array, hr, start_frame)
        for _, row in df.iterrows():
            pid, cam, step = row.patient_id, row.camera, row.step
            ppg = _load_ppg_sync(pid, cam, step)
            if not _is_good_ppg(ppg):
                continue
            vid = VID_DIR / f'{pid}_{cam}_{step}.avi'
            if not vid.exists():
                continue
            n_frames = len(ppg)
            starts = list(range(0, n_frames - clip_len, stride))
            if not starts:
                continue
            hr = float(row.pulse)
            for s in starts:
                self.clips.append((vid, ppg, hr, s))

        rng = random.Random(seed)
        rng.shuffle(self.clips)
        print(f'MCDrPPGDataset [{split}]: {len(self.clips)} clips from {len(df)} recordings')

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        vid_path, ppg_full, hr, start = self.clips[idx]

        # Load clip frames
        container = av.open(str(vid_path))
        target_frames = list(range(start, start + self.clip_len + 1))
        target_set = set(target_frames)
        frames = {}
        for i, frame in enumerate(container.decode(video=0)):
            if i in target_set:
                frames[i] = frame.to_ndarray(format='rgb24')
            if i > target_frames[-1]:
                break
        container.close()

        imgs = np.stack([frames[i] for i in target_frames if i in frames])

        # Face crop (optional)
        if self.face_detector is not None and len(imgs) > 0:
            box = self.face_detector(imgs[0])
            if box is not None:
                x1, y1, x2, y2 = box
                imgs = imgs[:, y1:y2, x1:x2, :]

        # Resize
        from PIL import Image
        resized = np.stack([
            np.array(Image.fromarray(f).resize(
                (self.target_size, self.target_size), Image.BILINEAR))
            for f in imgs
        ])  # [T+1, H, W, 3]

        if len(resized) < self.clip_len + 1:
            pad = np.zeros((self.clip_len + 1 - len(resized), self.target_size, self.target_size, 3), dtype=np.uint8)
            resized = np.concatenate([resized, pad], axis=0)

        # Augment
        if self.transform is not None:
            resized = self.transform(resized)

        # To tensor [3, T+1, H, W]
        frames_t = torch.from_numpy(resized.astype(np.float32) / 255.0).permute(3, 0, 1, 2)

        # PPG GT for clip
        ppg_clip = ppg_full[start : start + self.clip_len].copy()
        ppg_clip = _bandpass(ppg_clip, fs=30.0)
        ppg_clip = (ppg_clip - ppg_clip.mean()) / (ppg_clip.std() + 1e-8)
        ppg_t = torch.from_numpy(ppg_clip)

        return frames_t, ppg_t, hr
