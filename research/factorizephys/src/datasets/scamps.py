from pathlib import Path
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import h5py


class SCAMPSDataset(Dataset):
    """
    Streams face-video clips and PPG labels from SCAMPS data.

    Reads from a pre-built 64×64 cache (fast path) if cache_dir is set and the
    subject file exists there, otherwise falls back to the original .mat HDF5 files.

    Cache format (per subject .h5):
      frames : (3, 600, 64, 64)  float32  — resized to img_size at build time
      ppg    : (600,)            float32  — raw PPG waveform

    Original .mat format:
      Xsub  : (3, 240, 240, 600) float64  — face-cropped video in [0, 1]
      d_ppg : (600, 1)           float64  — raw PPG waveform

    Returns per item:
      frames    : (3, T, H, W) float32 [0, 1]
      ppg       : (T,)         float32  z-normalised (model target)
      ppg_raw   : (T,)         float32  original scale (HR extraction)
      subject_id: str
    """
    TOTAL_FRAMES = 600
    MAX_OPEN     = 32    # LRU cap on open h5py handles per worker

    def __init__(self, subjects, video_dir, clip_len=128, img_size=64,
                 clips_per_subj=4, augment=False, seed=42, cache_dir=None):
        self.video_dir  = Path(video_dir)
        self.cache_dir  = Path(cache_dir) if cache_dir else None
        self.clip_len   = clip_len
        self.img_size   = img_size
        self.augment    = augment
        self.max_start  = max(1, self.TOTAL_FRAMES - clip_len)

        rng = np.random.RandomState(seed)
        self.clips = []

        for subj in subjects:
            sid  = subj['subject_id']
            path = self.video_dir / f'{sid}.mat'
            if not path.exists():
                continue

            n = min(clips_per_subj, self.max_start)

            if augment:
                starts = rng.choice(self.max_start, size=n, replace=False).tolist()
            else:
                if n == 1:
                    starts = [self.max_start // 2]
                else:
                    starts = [int(i * self.max_start / (n - 1)) for i in range(n)]

            for s in starts:
                self.clips.append({'subject_id': sid, 'start': int(s)})

        split_label = 'Train' if augment else 'Val/Test'
        print(f'  {split_label:10s}: {len(self.clips):5d} clips '
              f'from {len(subjects)} subjects ({clips_per_subj} clips/subj)')

    def __len__(self):
        return len(self.clips)

    def __getstate__(self):
        # Called when pickling to send this object to a worker process.
        # h5py file handles cannot be pickled, so drop the cache.
        # Each worker recreates it lazily on its first __getitem__ call.
        state = self.__dict__.copy()
        state.pop('_h5_cache', None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def _get_h5(self, path_str):
        if not hasattr(self, '_h5_cache'):
            self._h5_cache = OrderedDict()
        if path_str in self._h5_cache:
            self._h5_cache.move_to_end(path_str)
            return self._h5_cache[path_str]
        if len(self._h5_cache) >= self.MAX_OPEN:
            _, oldest = self._h5_cache.popitem(last=False)
            try:
                oldest.close()
            except Exception:
                pass
        f = h5py.File(path_str, 'r')
        self._h5_cache[path_str] = f
        return f

    def _use_cache(self, sid):
        """Return cache h5 path if it exists, else None."""
        if self.cache_dir is None:
            return None
        p = self.cache_dir / f'{sid}.h5'
        return str(p) if p.exists() else None

    def __getitem__(self, idx):
        info  = self.clips[idx]
        sid   = info['subject_id']
        start = info['start']

        if self.augment:
            jitter = np.random.randint(-16, 17)
            start  = int(np.clip(start + jitter, 0, self.max_start))

        end = start + self.clip_len

        cache_path = self._use_cache(sid)
        if cache_path is not None:
            # Fast path: cache stores (3, 600, H, W), chunk=(3,128,H,W)
            # Supports float32 (v1) and float16+LZF (v2) — cast to float32 on load
            f = self._get_h5(cache_path)
            frames_np = f['frames'][:, start:end, :, :]   # (3, T, H, W)
            frames_t  = torch.from_numpy(np.ascontiguousarray(frames_np).astype('float32'))
            if self.img_size != 64:
                frames_t = F.interpolate(
                    frames_t.permute(1, 0, 2, 3),
                    size=(self.img_size, self.img_size),
                    mode='bilinear', align_corners=False,
                ).permute(1, 0, 2, 3).contiguous()
            ppg_raw = f['ppg'][start:end]
        else:
            # Slow path: original .mat HDF5 (3, 240, 240, 600) float64
            f = self._get_h5(str(self.video_dir / f'{sid}.mat'))
            raw      = f['Xsub'][:, :, :, start:end]          # (3, 240, 240, T)
            frames   = np.ascontiguousarray(raw.astype('float32').transpose(3, 0, 1, 2))
            frames_t = torch.from_numpy(frames)                # (T, 3, 240, 240)
            if self.img_size != 240:
                frames_t = F.interpolate(
                    frames_t, size=(self.img_size, self.img_size),
                    mode='bilinear', align_corners=False,
                )
            frames_t = frames_t.permute(1, 0, 2, 3).contiguous()  # (3, T, H, W)
            ppg_raw  = f['d_ppg'][start:end, 0].astype('float32')

        if self.augment:
            if np.random.random() > 0.5:
                frames_t = frames_t.flip(-1)
            alpha = 1.0 + (np.random.random() - 0.5) * 0.10
            beta  =       (np.random.random() - 0.5) * 0.04
            frames_t = (frames_t * alpha + beta).clamp_(0.0, 1.0)

        ppg_raw  = np.asarray(ppg_raw, dtype='float32')
        ppg_norm = (ppg_raw - ppg_raw.mean()) / (ppg_raw.std() + 1e-8)

        return {
            'frames':     frames_t,
            'ppg':        torch.from_numpy(ppg_norm.copy()),
            'ppg_raw':    torch.from_numpy(ppg_raw.copy()),
            'subject_id': sid,
        }
