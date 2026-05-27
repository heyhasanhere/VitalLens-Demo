"""
Build HDF5 cache for MCD-rPPG — frontal (FullHDwebcam) only, YOLO5Face crop → 72×72.

Protocol:
  - Camera: FullHDwebcam only (frontal, view='front')
  - Split: train (500 subjects) only for training cache
  - Face detection: YOLO5Face on first frame, squarify + 1.5× expand, propagated to all frames
  - Resize: INTER_AREA → 72×72
  - Format: [3, T, H, W] float16 / 255, LZF compressed HDF5

Usage (training cache):
  /home/dex/rppg_venv/bin/python src/cache/mcd_cache_builder.py --split train
Usage (eval cache, held_out 100 subjects):
  /home/dex/rppg_venv/bin/python src/cache/mcd_cache_builder.py --split held_out

Output:
  train   → rppg_dataset/MCD-rPPG/mcd_cache_frontal_72/
  held_out → eval_caches/mcd_eval_cache/ (one .h5 per video, then merged to .pt by eval script)
"""
import sys, os, json, time, types, argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import threading
import numpy as np
import av
import cv2
import h5py
import torch

_y5f_lock = threading.Lock()

PROJECT_ROOT = Path('/mnt/sata-ssd/rppg_project')
MCD_ROOT     = PROJECT_ROOT / 'rppg_dataset' / 'MCD-rPPG'
VID_DIR      = MCD_ROOT / 'video'
SYNC_DIR     = MCD_ROOT / 'ppg_sync'
FP_ROOT      = PROJECT_ROOT / 'external' / 'FactorizePhys'

TRAIN_CACHE_DIR = MCD_ROOT / 'mcd_cache_frontal_72'
EVAL_CACHE_DIR  = PROJECT_ROOT / 'eval_caches' / 'mcd_eval_cache'

TARGET_SIZE   = 72
CAMERA        = 'FullHDwebcam'
FACE_COEF     = 1.5       # squarify + expand factor for face crop
FACE_CONF     = 0.6       # YOLO5Face default confidence threshold
PPG_FLAT_STD  = 5.0       # from Step 0: catches only std=0 dead recordings
NUM_WORKERS   = 4         # thread workers (I/O bound; YOLO5Face is the bottleneck)


def _setup_yolo():
    """Load YOLO5Face on GPU. Call once per process."""
    for pkg in ['dataset', 'dataset.data_loader']:
        if pkg not in sys.modules:
            m = types.ModuleType(pkg)
            m.__path__ = [str(FP_ROOT / pkg.replace('.', '/'))]
            m.__package__ = pkg
            sys.modules[pkg] = m
    if str(FP_ROOT) not in sys.path:
        sys.path.insert(0, str(FP_ROOT))
    from dataset.data_loader.face_detector.YOLO5Face import YOLO5Face
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    y5f = YOLO5Face(backend='Y5F', device=device)
    return y5f


def _get_bbox(y5f, frame_rgb_uint8):
    """Run YOLO5Face and return squarified+expanded crop box (x, y, w, h) or None."""
    res = y5f.detect_face(frame_rgb_uint8)
    if res is None:
        return None
    x1, y1, x2, y2 = res
    w, h  = x2 - x1, y2 - y1
    sq    = max(w, h)
    cx    = x1 + w // 2
    cy    = y1 + h // 2
    ex    = max(0, cx - int(FACE_COEF * sq // 2))
    ey    = max(0, cy - int(FACE_COEF * sq // 2))
    ew    = int(FACE_COEF * sq)
    eh    = int(FACE_COEF * sq)
    return ex, ey, ew, eh


def _load_ppg(pid, step):
    path = SYNC_DIR / f'{pid}_{CAMERA}_{step}.txt'
    if not path.exists():
        return None
    return np.array([int(l.split()[0]) for l in open(path)], dtype=np.float32)


def _is_good_ppg(ppg, min_frames=160):
    if ppg is None or len(ppg) < min_frames:
        return False
    return ppg.std() >= PPG_FLAT_STD


def process_video(y5f, vid_path, out_path, pid, step):
    """Process one video: detect face, crop all frames, write HDF5."""
    if out_path.exists():
        return str(vid_path.name), 'skip', 0

    ppg = _load_ppg(pid, step)
    if not _is_good_ppg(ppg):
        return str(vid_path.name), 'bad_ppg', 0

    try:
        container = av.open(str(vid_path))
        raw_frames = []
        for f in container.decode(video=0):
            raw_frames.append(f.to_ndarray(format='rgb24'))
        container.close()
    except Exception as e:
        return str(vid_path.name), f'decode_error: {e}', 0

    if not raw_frames:
        return str(vid_path.name), 'empty', 0

    # YOLO5Face on first frame; propagate box to all frames
    # Lock ensures multiple threads don't call the shared model simultaneously
    with _y5f_lock:
        bbox = _get_bbox(y5f, raw_frames[0])
    if bbox is not None:
        ex, ey, ew, eh = bbox
        H, W = raw_frames[0].shape[:2]
        x2c, y2c = min(ex + ew, W), min(ey + eh, H)
        crop = lambda f: f[ey:y2c, ex:x2c]
    else:
        crop = lambda f: f  # fallback: full frame

    frames_out = []
    for f in raw_frames:
        cropped = crop(f)
        if cropped.size == 0:
            cropped = f
        resized = cv2.resize(cropped, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_AREA)
        frames_out.append(resized)

    arr = np.stack(frames_out).transpose(3, 0, 1, 2).astype(np.float16) / 255.0  # [3,T,H,W]
    n_frames = arr.shape[1]

    # Align PPG to frame count
    if len(ppg) > n_frames:
        ppg = ppg[:n_frames]
    elif len(ppg) < n_frames:
        ppg = np.pad(ppg, (0, n_frames - len(ppg)), mode='edge')

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(str(out_path), 'w') as hf:
        hf.create_dataset('frames', data=arr, compression='lzf')
        hf.create_dataset('ppg',    data=ppg, compression='lzf')
        hf.attrs['pid']      = pid
        hf.attrs['cam']      = CAMERA
        hf.attrs['step']     = step
        hf.attrs['n_frames'] = n_frames
        hf.attrs['face_detected'] = (bbox is not None)

    return str(vid_path.name), 'ok', n_frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=['train', 'held_out'], default='train',
                        help='Which split to cache (train=500 subjects, held_out=100 subjects)')
    args = parser.parse_args()

    cache_dir = TRAIN_CACHE_DIR if args.split == 'train' else EVAL_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    with open(PROJECT_ROOT / 'checkpoints' / 'mcd_split.json') as f:
        split = json.load(f)
    subject_ids = split[args.split]

    steps = ['before', 'after']
    tasks = []
    for pid in sorted(subject_ids):
        for step in steps:
            vid_path = VID_DIR / f'{pid}_{CAMERA}_{step}.avi'
            if not vid_path.exists():
                continue
            out_path = cache_dir / f'{pid}_{CAMERA}_{step}.h5'
            tasks.append((vid_path, out_path, pid, step))

    print(f'Split: {args.split} ({len(subject_ids)} subjects)')
    print(f'Videos to process: {len(tasks)}')
    print(f'Cache dir: {cache_dir}')
    print(f'Workers: {NUM_WORKERS} (threads, shared YOLO5Face GPU)')
    print()

    y5f = _setup_yolo()
    print(f'YOLO5Face loaded on {y5f.device}')
    print()

    t0 = time.time()
    done, skipped, bad_ppg, errors = 0, 0, 0, 0

    # ThreadPoolExecutor: YOLO5Face is GPU-bound; threads share one GPU model safely
    # since YOLO5Face.detect_face() is stateless and torch.no_grad is used internally.
    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as ex:
        futs = {ex.submit(process_video, y5f, *t): t for t in tasks}
        for i, fut in enumerate(as_completed(futs), 1):
            name, status, n = fut.result()
            if status == 'skip':
                skipped += 1
            elif status == 'ok':
                done += 1
            elif status == 'bad_ppg':
                bad_ppg += 1
            else:
                errors += 1
                print(f'  FAIL: {name}: {status}')

            if i % 50 == 0 or i == len(tasks):
                elapsed = time.time() - t0
                rate    = i / elapsed
                eta     = (len(tasks) - i) / rate if rate > 0 else 0
                print(f'[{i:4d}/{len(tasks)}] ok={done} skip={skipped} '
                      f'bad_ppg={bad_ppg} err={errors}  '
                      f'rate={rate:.1f}/s  ETA={eta/60:.0f}min')

    elapsed = time.time() - t0
    print(f'\nDone in {elapsed/60:.1f} min')
    print(f'  cached={done}  skipped={skipped}  bad_ppg={bad_ppg}  errors={errors}')

    total_bytes = sum(os.path.getsize(f) for f in cache_dir.glob('*.h5'))
    print(f'Cache size: {total_bytes/1e9:.1f} GB')


if __name__ == '__main__':
    main()
