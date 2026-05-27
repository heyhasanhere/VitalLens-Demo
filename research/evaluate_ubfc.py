"""
Full UBFC-rPPG DATASET_2 evaluation.

Produces per-subject HR MAE, BVP Pearson r, and subject-level Pearson r / r²
across all 42 subjects — the standard benchmark reported in rPPG papers.

Memory-safe: uses a rolling deque of 161 pre-cropped 128×128 frames (~8 MB peak).
Never loads a full video into RAM.

Usage (run from VitalLens project root):
  python evaluate_ubfc.py
  python evaluate_ubfc.py --dataset-path "C:/UTS/4/Deep Learning/Project/UBFC_Dataset/DATASET_2"
  python evaluate_ubfc.py --onnx-path models/weights/vitallens_rppg.onnx
  python evaluate_ubfc.py --save-csv results_ubfc.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from scipy.signal import butter, filtfilt, welch
from scipy.stats import pearsonr

_REPO_ROOT      = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = None
DEFAULT_ONNX    = _REPO_ROOT / "weights" / "vitallens_rppg.onnx"

CLIP_LEN   = 160
STRIDE     = 80
INPUT_SIZE = 128
TARGET_FPS = 30.0


# ---------------------------------------------------------------------------
# BVP → HR  (same Welch params as backend/inference.py)
# ---------------------------------------------------------------------------

def bvp_to_hr(bvp: np.ndarray, fps: float) -> float:
    if len(bvp) < 64:
        return 0.0
    nyq = fps / 2.0
    low, high = 0.67 / nyq, min(3.0 / nyq, 0.99)
    try:
        b, a  = butter(3, [low, high], btype="band")
        bvp_f = filtfilt(b, a, bvp)
    except Exception:
        bvp_f = bvp
    freqs, psd = welch(bvp_f, fs=fps, nperseg=min(len(bvp_f), 150), nfft=2048)
    band = (freqs >= 0.67) & (freqs <= 3.0)
    if not band.any():
        return 0.0
    band_psd  = psd[band]
    band_freq = freqs[band]
    peak_idx  = band_psd.argmax()
    # Parabolic sub-bin interpolation
    if 0 < peak_idx < len(band_psd) - 1:
        alpha, beta, gamma = band_psd[peak_idx - 1], band_psd[peak_idx], band_psd[peak_idx + 1]
        denom = alpha - 2 * beta + gamma
        if abs(denom) > 1e-10:
            df    = freqs[1] - freqs[0] if len(freqs) > 1 else 0.0
            shift = 0.5 * (alpha - gamma) / denom * df
            return float((band_freq[peak_idx] + shift) * 60.0)
    return float(band_freq[peak_idx] * 60.0)


# ---------------------------------------------------------------------------
# DiffNorm on pre-resized 128×128 frames stored in deque
# ---------------------------------------------------------------------------

def diffnorm_from_deque(buf: deque) -> np.ndarray:
    """
    buf: deque of CLIP_LEN+1 float32 (128, 128, 3) frames, already brightness-normalised.
    Returns float32 (1, CLIP_LEN, 3, 128, 128).
    """
    arr    = np.array(buf, dtype=np.float32)   # (161, 128, 128, 3)
    f_next = arr[1:]
    f_curr = arr[:-1]
    diff   = (f_next - f_curr) / (f_next + f_curr + 1e-6)
    diff   = np.clip(diff, -3.0, 3.0)
    diff   = diff.transpose(0, 3, 1, 2)        # (160, 3, 128, 128)
    return diff[np.newaxis]                    # (1, 160, 3, 128, 128)


# ---------------------------------------------------------------------------
# Per-subject streaming evaluation
# ---------------------------------------------------------------------------

def evaluate_subject(
    subject_dir: Path,
    sess: ort.InferenceSession,
    verbose: bool = False,
) -> dict:
    vid_path = subject_dir / "vid.avi"
    gt_path  = subject_dir / "ground_truth.txt"

    if not vid_path.exists() or not gt_path.exists():
        return {"subject": subject_dir.name, "error": "missing files"}

    gt           = np.loadtxt(gt_path)   # (3, N_frames)
    gt_bvp       = gt[0]
    gt_hr_series = gt[1]

    cap = cv2.VideoCapture(str(vid_path))
    if not cap.isOpened():
        return {"subject": subject_dir.name, "error": "cannot open video"}

    raw_fps   = cap.get(cv2.CAP_PROP_FPS) or TARGET_FPS
    stride    = max(1, round(raw_fps / TARGET_FPS))
    eff_fps   = raw_fps / stride

    # Haar detector — detect face in first valid frame, then lock bbox
    haar_path  = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector   = cv2.CascadeClassifier(haar_path)
    locked_box = None   # (x1, y1, x2, y2) set once

    # Rolling buffer: holds CLIP_LEN+1 brightness-normalised 128×128 float32 frames
    # Memory: 161 × 128 × 128 × 3 × 4 bytes ≈ 31 MB peak per subject
    buf = deque(maxlen=CLIP_LEN + 1)

    bvp_segments: list[np.ndarray] = []
    frames_kept  = 0   # frames added to buf so far
    next_clip_at = CLIP_LEN + 1   # buf size at which to fire next clip

    raw_frame_idx = 0  # raw video frame counter (before stride subsampling)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if raw_frame_idx % stride != 0:
            raw_frame_idx += 1
            continue

        raw_frame_idx += 1

        # Lock face bbox on first frame
        if locked_box is None and not detector.empty():
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
            )
            if len(faces) > 0:
                fx, fy, fw, fh = faces[0]
                px, py = int(fw * 0.3), int(fh * 0.3)
                h, w   = frame.shape[:2]
                locked_box = (
                    max(0, fx - px), max(0, fy - py),
                    min(w, fx + fw + px), min(h, fy + fh + py),
                )

        # Crop to face if detected, else use full frame
        if locked_box is not None:
            x1, y1, x2, y2 = locked_box
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                frame = crop

        # Resize to 128×128 and brightness-normalise before buffering
        small = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE),
                           interpolation=cv2.INTER_LINEAR).astype(np.float32)
        lum   = small.mean() + 1e-6
        small = small / lum

        buf.append(small)
        frames_kept += 1

        # Fire a clip whenever the buffer is full and we've advanced STRIDE from last clip
        if frames_kept == next_clip_at:
            clip_in = diffnorm_from_deque(buf)
            bvp     = sess.run(None, {sess.get_inputs()[0].name: clip_in})[0][0]
            bvp_segments.append(bvp)
            next_clip_at += STRIDE   # next clip fires STRIDE frames later

    cap.release()

    if not bvp_segments:
        return {"subject": subject_dir.name, "error": f"no clips (frames_kept={frames_kept})"}

    # Concatenate BVP
    bvp_concat = np.concatenate(
        [bvp_segments[j][:STRIDE] for j in range(len(bvp_segments) - 1)]
        + [bvp_segments[-1]]
    )

    pred_hr = bvp_to_hr(bvp_concat, fps=eff_fps)

    # GT HR: mean of per-frame HR series, subsampled to eff_fps
    gt_hr_sub = gt_hr_series[::stride]
    gt_hr     = float(np.mean(gt_hr_sub[:frames_kept]))

    # BVP signal-level Pearson r — align GT BVP to concat BVP length
    gt_bvp_sub = gt_bvp[::stride]
    n_align    = min(len(gt_bvp_sub) - 1, len(bvp_concat))
    r_bvp      = float("nan")
    if n_align >= 10:
        r_bvp, _ = pearsonr(bvp_concat[:n_align], gt_bvp_sub[1:n_align + 1])

    mae = abs(pred_hr - gt_hr)

    if verbose:
        print(f"  {subject_dir.name:12s}  pred={pred_hr:6.1f}  gt={gt_hr:6.1f}  "
              f"MAE={mae:5.1f}  r_bvp={r_bvp:+.3f}  clips={len(bvp_segments)}")

    return {
        "subject":  subject_dir.name,
        "pred_hr":  pred_hr,
        "gt_hr":    gt_hr,
        "mae":      mae,
        "r_bvp":    r_bvp,
        "n_clips":  len(bvp_segments),
        "n_frames": frames_kept,
        "fps":      eff_fps,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", required=True, help="Path to UBFC-rPPG DATASET_2 directory")
    parser.add_argument("--onnx-path",    default=str(DEFAULT_ONNX))
    parser.add_argument("--save-csv",     default=None, help="Save per-subject results to CSV")
    parser.add_argument("--quiet",        action="store_true")
    args = parser.parse_args()

    onnx_path    = Path(args.onnx_path)
    dataset_path = Path(args.dataset_path)

    if not onnx_path.exists():
        sys.exit(f"ONNX not found: {onnx_path}\n  Run from VitalLens project root.")
    if not dataset_path.exists():
        sys.exit(f"Dataset not found: {dataset_path}")

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    print(f"Model  : {onnx_path}")
    print(f"Dataset: {dataset_path}")

    subject_dirs = sorted([d for d in dataset_path.iterdir() if d.is_dir()])
    print(f"Found {len(subject_dirs)} subjects — streaming eval, ~8 MB RAM per subject\n")

    results = []
    for i, sd in enumerate(subject_dirs, 1):
        print(f"[{i:2d}/{len(subject_dirs)}] {sd.name}", end="  ", flush=True)
        r = evaluate_subject(sd, sess, verbose=False)
        results.append(r)
        if "error" in r:
            print(f"ERROR: {r['error']}")
        else:
            print(f"pred={r['pred_hr']:.1f}  gt={r['gt_hr']:.1f}  MAE={r['mae']:.1f}  r={r['r_bvp']:.3f}")

    valid  = [r for r in results if "error" not in r]
    errors = [r for r in results if "error" in r]

    if errors:
        print(f"\nSkipped: {[e['subject'] for e in errors]}")

    if not valid:
        sys.exit("No valid subjects.")

    maes      = np.array([r["mae"]     for r in valid])
    r_bvps    = np.array([r["r_bvp"]   for r in valid if not np.isnan(r["r_bvp"])])
    pred_hrs  = np.array([r["pred_hr"] for r in valid])
    gt_hrs    = np.array([r["gt_hr"]   for r in valid])

    r_subj, p_subj = pearsonr(pred_hrs, gt_hrs)
    r2_subj        = r_subj ** 2

    print(f"\n{'='*52}")
    print(f"  Subjects evaluated  : {len(valid)} / {len(subject_dirs)}")
    print(f"  HR MAE              : {maes.mean():.2f} ± {maes.std():.2f} BPM")
    print(f"  HR MAE median       : {np.median(maes):.2f} BPM")
    print(f"  BVP Pearson r       : {r_bvps.mean():.3f} ± {r_bvps.std():.3f}  (signal-level, mean)")
    print(f"  Subject Pearson r   : {r_subj:.3f}  (pred vs GT HR, n={len(valid)})")
    print(f"  Subject Pearson r²  : {r2_subj:.3f}")
    print(f"  p-value             : {p_subj:.4f}")
    print(f"{'='*52}")

    if args.save_csv:
        csv_path = Path(args.save_csv)
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["subject","pred_hr","gt_hr","mae","r_bvp","n_clips","n_frames","fps"]
            )
            writer.writeheader()
            writer.writerows(valid)
        print(f"Results saved → {csv_path}")


if __name__ == "__main__":
    main()
