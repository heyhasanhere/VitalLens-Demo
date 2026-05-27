"""
Build SCAMPS 72x72 float16+LZF cache with YOLO5Face face detection.
Paper protocol: detect every 30 frames, squarify+1.5x expand, INTER_AREA resize.
One YOLO5Face model is loaded once per worker process via _init_worker().
"""
import sys
import types
import time
from pathlib import Path

import numpy as np
import cv2
import h5py

_FP_ROOT = str(Path(__file__).parent.parent / 'external' / 'FactorizePhys')
_y5f_model = None


def _init_worker():
    global _y5f_model
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    for pkg in ['dataset', 'dataset.data_loader']:
        if pkg not in sys.modules:
            m = types.ModuleType(pkg)
            m.__path__ = [_FP_ROOT + '/' + pkg.replace('.', '/')]
            m.__package__ = pkg
            sys.modules[pkg] = m
    if _FP_ROOT not in sys.path:
        sys.path.insert(0, _FP_ROOT)
    from dataset.data_loader.face_detector.YOLO5Face import YOLO5Face
    _y5f_model = YOLO5Face(backend='Y5F', device='cpu')


def _get_bbox(frame_rgb_uint8, img_h, img_w, coef=1.5):
    res = _y5f_model.detect_face(frame_rgb_uint8)
    if res is None:
        return [0, 0, img_w, img_h]
    x1, y1, x2, y2 = res
    w, h = x2 - x1, y2 - y1
    sq = max(w, h)
    cx, cy = x1 + w // 2, y1 + h // 2
    new_x = cx - sq // 2
    new_y = cy - sq // 2
    ex = max(0, int(new_x - (coef - 1.0) / 2 * sq))
    ey = max(0, int(new_y - (coef - 1.0) / 2 * sq))
    ew = int(coef * sq)
    eh = int(coef * sq)
    return [ex, ey, ew, eh]


def build_subject_cache_72_y5f(args):
    mat_path, cache_path = args
    mat_path   = Path(mat_path)
    cache_path = Path(cache_path)
    img_size     = 72
    detect_every = 30

    t0 = time.perf_counter()

    with h5py.File(mat_path, 'r') as f:
        raw = f['Xsub'][:].astype('float32')      # (3, 240, 240, 600) [0,1]
        ppg = f['d_ppg'][:, 0].astype('float32')  # (600,)

    # (3, 240, 240, 600) → (600, 240, 240, 3) float32 [0,1]
    frames_hwc = raw.transpose(3, 1, 2, 0)
    T, img_h, img_w, _ = frames_hwc.shape

    # uint8 copy for YOLO5Face (expects RGB uint8)
    frames_u8 = (frames_hwc * 255).clip(0, 255).astype(np.uint8)

    # detect face every 30 frames
    bboxes = {}
    for i in range(0, T, detect_every):
        bboxes[i] = _get_bbox(frames_u8[i], img_h, img_w)

    # crop + INTER_AREA resize each frame
    out = np.empty((T, img_size, img_size, 3), dtype=np.float32)
    for i in range(T):
        anchor = (i // detect_every) * detect_every
        x, y, w, h = bboxes[anchor]
        crop = frames_hwc[i, max(y, 0):min(y + h, img_h), max(x, 0):min(x + w, img_w), :]
        out[i] = cv2.resize(crop, (img_size, img_size), interpolation=cv2.INTER_AREA)

    # (T, H, W, 3) → (3, T, H, W) float16
    frames_out = out.transpose(3, 0, 1, 2).astype('float16')

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(cache_path, 'w') as f:
        f.create_dataset(
            'frames', data=frames_out,
            chunks=(3, 128, 72, 72),
            compression='lzf',
            dtype='float16',
        )
        f.create_dataset('ppg', data=ppg, dtype='float32')

    return mat_path.stem, time.perf_counter() - t0
