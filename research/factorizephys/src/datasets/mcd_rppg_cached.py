"""
MCDrPPGCachedDataset — HDF5-backed MCD-rPPG dataset for Phase 2 Run 2.

Cache protocol: FullHDwebcam only, YOLO5Face crop → 72×72, float16 LZF.
Cache dir: rppg_dataset/MCD-rPPG/mcd_cache_frontal_72/

__getitem__ slices a clip from the HDF5 file — ~5ms vs ~500ms for on-the-fly AVI decode.
"""
import json
import random
from pathlib import Path
from collections import OrderedDict

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from scipy.signal import butter, filtfilt

ROOT      = Path('/mnt/sata-ssd/rppg_project/rppg_dataset/MCD-rPPG')
CACHE_DIR = ROOT / 'mcd_cache_frontal_72'
DB        = ROOT / 'db.csv'

CAMERA        = 'FullHDwebcam'
HR_MIN        = 40
HR_MAX        = 160    # observed max in MCD is 153; 160 gives 7 bpm buffer
PPG_FLAT_STD  = 5.0    # catches only std=0 dead recordings (gap to next value is ~37)
MIN_FRAMES    = 160


def _is_good_ppg(ppg):
    if ppg is None or len(ppg) < MIN_FRAMES:
        return False
    return float(ppg.std()) >= PPG_FLAT_STD


def _bandpass(signal, fs=30.0, lo=0.6, hi=4.0):
    nyq = fs / 2.0
    b, a = butter(4, [lo / nyq, hi / nyq], btype='bandpass')
    return filtfilt(b, a, signal.astype(np.float64)).astype(np.float32)


class MCDrPPGCachedDataset(Dataset):
    """
    Fast HDF5-cached MCD-rPPG dataset. FullHDwebcam frontal only.

    Each item returns {'frames': FloatTensor [3, clip_len+1, H, W],
                       'ppg':    FloatTensor [clip_len],
                       'gt_hr':  float}

    transform receives a [3, clip_len+1, H, W] float32 tensor and returns same shape.
    """
    MAX_OPEN = 32

    def __init__(
        self,
        split_json,
        split='train',
        steps=None,
        clip_len=160,
        stride=160,
        transform=None,
        seed=42,
    ):
        self.clip_len  = clip_len
        self.stride    = stride
        self.transform = transform
        self.steps     = steps or ['before', 'after']

        with open(split_json) as f:
            split_data = json.load(f)
        subject_ids = set(split_data[split])

        df = pd.read_csv(DB)
        df = df[df['patient_id'].isin(subject_ids)]
        df = df[df['camera'] == CAMERA]
        df = df[df['step'].isin(self.steps)]
        df = df[(df['pulse'] >= HR_MIN) & (df['pulse'] <= HR_MAX)]

        self.clips = []
        missing = 0
        for _, row in df.iterrows():
            pid, step = row.patient_id, row.step
            h5_path = CACHE_DIR / f'{pid}_{CAMERA}_{step}.h5'
            if not h5_path.exists():
                missing += 1
                continue
            hr = float(row.pulse)

            try:
                with h5py.File(str(h5_path), 'r') as hf:
                    n_frames = int(hf.attrs.get('n_frames', hf['frames'].shape[1]))
                    ppg_raw  = hf['ppg'][:]
                    if not _is_good_ppg(ppg_raw):
                        continue
            except Exception:
                continue

            for s in range(0, n_frames - clip_len, stride):
                self.clips.append((h5_path, hr, s))

        rng = random.Random(seed)
        rng.shuffle(self.clips)

        if missing > 0:
            print(f'MCDrPPGCachedDataset [{split}]: {missing} h5 files missing')
        print(f'MCDrPPGCachedDataset [{split}]: {len(self.clips)} clips '
              f'from {len(df) - missing} recordings')

    def __len__(self):
        return len(self.clips)

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('_h5c', None)
        return state

    def _get_h5(self, path_str):
        if not hasattr(self, '_h5c'):
            self._h5c = OrderedDict()
        if path_str in self._h5c:
            self._h5c.move_to_end(path_str)
            return self._h5c[path_str]
        if len(self._h5c) >= self.MAX_OPEN:
            _, oldest = self._h5c.popitem(last=False)
            try: oldest.close()
            except: pass
        f = h5py.File(path_str, 'r')
        self._h5c[path_str] = f
        return f

    def __getitem__(self, idx):
        h5_path, hr, start = self.clips[idx]
        path_str = str(h5_path)

        hf  = self._get_h5(path_str)
        end = start + self.clip_len + 1

        frames_np = hf['frames'][:, start:end, :, :].astype(np.float32)
        frames_t  = torch.from_numpy(frames_np)  # [3, clip_len+1, H, W]

        ppg_raw  = hf['ppg'][start: start + self.clip_len].astype(np.float32)
        ppg_bp   = _bandpass(ppg_raw, fs=30.0)
        ppg_norm = (ppg_bp - ppg_bp.mean()) / (ppg_bp.std() + 1e-8)
        ppg_t    = torch.from_numpy(ppg_norm)

        if self.transform is not None:
            frames_t = self.transform(frames_t)

        return {'frames': frames_t, 'ppg': ppg_t, 'gt_hr': hr}
