"""
End-to-end inference test for VitalLens rPPG model.

Usage:
  python test_e2e.py --sanity                        # random-input sanity check (no camera)
  python test_e2e.py --oximeter 82                   # webcam, face crop on, 3 clips, median HR
  python test_e2e.py --n-clips 5 --oximeter 82       # more clips = more stable estimate
  python test_e2e.py --no-face-crop --oximeter 82    # disable face crop for comparison
  python test_e2e.py --video path.mp4 --oximeter 82  # from video file
  python test_e2e.py --npy subject_01_clips.npy      # dataset npy (paired _bvps.npy)
  python test_e2e.py --npy subject_01_clips.npy --all-clips
  python test_e2e.py --video path.mp4 --save-viz     # also write viz_output/diffnorm.mp4 + amplified.mp4
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from scipy.signal import butter, filtfilt, welch

_REPO_ROOT = Path(__file__).resolve().parent.parent
ONNX_PATH  = _REPO_ROOT / "weights" / "vitallens_rppg.onnx"  # overrideable via --onnx-path
CLIP_LEN   = 160
STRIDE     = 80     # same overlap used in training extraction
FPS        = 30.0
INPUT_SIZE = 128


# ---------------------------------------------------------------------------
# BVP → HR  (identical to train.py)
# ---------------------------------------------------------------------------

def bvp_to_hr(bvp: np.ndarray, fps: float = FPS) -> float:
    if len(bvp) < 10:
        return 0.0
    nyq = fps / 2.0
    low, high = 0.67 / nyq, min(3.0 / nyq, 0.99)
    try:
        b, a  = butter(3, [low, high], btype="band")
        bvp_f = filtfilt(b, a, bvp)
    except Exception:
        bvp_f = bvp
    freqs, psd = welch(bvp_f, fs=fps, nperseg=min(len(bvp_f), 1024))
    band = (freqs >= 0.67) & (freqs <= 3.0)
    if not band.any():
        return 0.0
    return float(freqs[band][psd[band].argmax()] * 60.0)


# ---------------------------------------------------------------------------
# Face crop via OpenCV Haar cascade (bundled with OpenCV, no extra deps)
# ---------------------------------------------------------------------------

def make_face_detector():
    path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    det  = cv2.CascadeClassifier(path)
    if det.empty():
        print("Warning: Haar cascade not found — face crop disabled")
        return None
    return det


def crop_face(frame: np.ndarray, detector, padding: float = 0.3) -> np.ndarray:
    """Crop to first detected face + padding. Returns full frame if no face found."""
    h, w = frame.shape[:2]
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        return frame
    fx, fy, fw, fh = faces[0]
    px = int(fw * padding)
    py = int(fh * padding)
    x1, y1 = max(0, fx - px), max(0, fy - py)
    x2, y2 = min(w, fx + fw + px), min(h, fy + fh + py)
    crop = frame[y1:y2, x1:x2]
    return crop if crop.size > 0 else frame


# ---------------------------------------------------------------------------
# DiffNorm preprocessing  (matches s3_preprocess.py exactly)
# ---------------------------------------------------------------------------

def preprocess_frames(raw_frames: list[np.ndarray], brightness_norm: bool = True) -> np.ndarray:
    """
    raw_frames      : CLIP_LEN+1 BGR uint8 frames (H×W×3), already cropped if desired
    brightness_norm : divide each frame by its mean luminance before DiffNorm — cancels
                      auto-exposure drift which is 100-1000x larger than the cardiac signal
    returns         : float32 (1, CLIP_LEN, 3, 128, 128)
    """
    resized = np.array(
        [cv2.resize(f, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
         for f in raw_frames],
        dtype=np.float32,
    )
    if brightness_norm:
        # Per-frame mean luminance — shape (T+1, 1, 1, 1) for broadcasting
        lum     = resized.mean(axis=(1, 2, 3), keepdims=True)
        resized = resized / (lum + 1e-6)

    f_next = resized[1:]
    f_curr = resized[:-1]
    diff   = (f_next - f_curr) / (f_next + f_curr + 1e-6)
    diff   = diff.transpose(0, 3, 1, 2)
    return diff[np.newaxis]


# ---------------------------------------------------------------------------
# ONNX helpers
# ---------------------------------------------------------------------------

def load_session(path: Path) -> ort.InferenceSession:
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    print(f"Loaded ONNX : {path}")
    print(f"  Input     : {sess.get_inputs()[0].name}  {sess.get_inputs()[0].shape}")
    print(f"  Output    : {sess.get_outputs()[0].name} {sess.get_outputs()[0].shape}")
    return sess


def run_inference(sess: ort.InferenceSession, clip: np.ndarray) -> np.ndarray:
    return sess.run(None, {sess.get_inputs()[0].name: clip})[0][0]  # (160,)


# ---------------------------------------------------------------------------
# Capture helpers
# ---------------------------------------------------------------------------

TARGET_FPS = 30.0  # model was trained on 30 fps


def collect_frames(source, n_frames: int, detector=None, start_sec: float = 0.0) -> tuple[list[np.ndarray], float]:
    """
    source    : 0 for webcam, or path string for video file
    detector  : Haar CascadeClassifier or None
    start_sec : seek to this timestamp before collecting (video files only)
    Subsamples to TARGET_FPS if source fps differs (e.g. 60fps phone video → every 2nd frame).
    Returns list of (optionally face-cropped) frames and effective FPS.
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        sys.exit(f"Cannot open source: {source}")
    raw_fps   = cap.get(cv2.CAP_PROP_FPS) or TARGET_FPS
    is_webcam = (source == 0)

    if start_sec > 0 and not is_webcam:
        start_frame = int(start_sec * raw_fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        print(f"Seeking     : frame {start_frame} ({start_sec:.1f}s @ {raw_fps:.0f} fps)")

    # Subsample stride so effective fps ≈ TARGET_FPS
    stride = max(1, round(raw_fps / TARGET_FPS))
    eff_fps = raw_fps / stride

    if stride > 1:
        print(f"Source FPS  : {raw_fps:.1f} → subsampling every {stride} frames → {eff_fps:.1f} FPS")
    elif is_webcam:
        print(f"Webcam FPS  : {raw_fps:.1f}")

    if is_webcam:
        print(f"Hold still for ~{n_frames / eff_fps:.1f} s  (press Q to abort) ...")

    # Detect face once in the first frame and lock that bbox for the entire clip.
    # Re-running Haar every frame shifts the crop box by a few pixels each time;
    # after resize to 320px that looks like constant zoom/drift and completely
    # swamps the cardiac signal in EVM (motion >> colour change).
    locked_box = None   # (x1, y1, x2, y2) pixel coords, set on first detection

    frames    = []
    frame_idx = 0
    while len(frames) < n_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % stride == 0:
            if detector is not None:
                if locked_box is None:
                    h, w = frame.shape[:2]
                    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = detector.detectMultiScale(
                        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
                    )
                    if len(faces) > 0:
                        fx, fy, fw, fh = faces[0]
                        px, py = int(fw * 0.3), int(fh * 0.3)
                        locked_box = (
                            max(0, fx - px), max(0, fy - py),
                            min(w, fx + fw + px), min(h, fy + fh + py),
                        )
                        print(f"Face locked : bbox {locked_box}")
                if locked_box is not None:
                    x1, y1, x2, y2 = locked_box
                    crop = frame[y1:y2, x1:x2]
                    frame = crop if crop.size > 0 else frame
            frames.append(frame)
        frame_idx += 1
        if is_webcam:
            cv2.imshow("VitalLens — collecting (Q to abort)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    return frames, eff_fps


# ---------------------------------------------------------------------------
# NPY mode — paired {subject}_clips.npy + {subject}_bvps.npy (exact training format)
# ---------------------------------------------------------------------------

def _load_npy_pair(clips_path: str) -> tuple[np.ndarray, np.ndarray]:
    clips_path = Path(clips_path)
    bvps_path  = clips_path.parent / clips_path.name.replace("_clips.npy", "_bvps.npy")
    if not bvps_path.exists():
        sys.exit(f"BVP file not found alongside clips: expected {bvps_path}")
    clips = np.load(clips_path, mmap_mode='r')  # (N, 160, 3, 128, 128)
    bvps  = np.load(bvps_path,  mmap_mode='r')  # (N, 160)
    return clips, bvps


def run_npy(
    sess: ort.InferenceSession,
    clips_path: str,
    clip_idx: int,
    all_clips: bool = False,
) -> None:
    clips, bvps = _load_npy_pair(clips_path)
    n_clips = len(clips)
    print(f"NPY loaded  : {clips_path}  ({n_clips} clips, shape {clips.shape})")

    from scipy.stats import pearsonr

    if all_clips:
        maes, rs = [], []
        for i in range(n_clips):
            clip_in  = clips[i][np.newaxis].astype(np.float32)
            bvp_pred = run_inference(sess, clip_in)
            bvp_gt   = bvps[i]
            hr_pred  = bvp_to_hr(bvp_pred)
            hr_gt    = bvp_to_hr(bvp_gt)
            r, _     = pearsonr(bvp_pred, bvp_gt)
            maes.append(abs(hr_pred - hr_gt))
            rs.append(r)
            print(f"  clip {i:3d}: pred={hr_pred:6.1f}  gt={hr_gt:6.1f}  MAE={maes[-1]:5.1f}  r={r:.3f}")

        print(f"\n{'='*44}")
        print(f"  Mean HR MAE  : {np.mean(maes):.1f} ± {np.std(maes):.1f} BPM")
        print(f"  Mean Pearson : {np.mean(rs):.3f}")
        print(f"{'='*44}")
        return

    if clip_idx >= n_clips:
        sys.exit(f"--clip-idx {clip_idx} out of range (file has {n_clips} clips)")

    clip_in  = clips[clip_idx][np.newaxis].astype(np.float32)
    bvp_gt   = np.array(bvps[clip_idx])

    t0       = time.perf_counter()
    bvp_pred = run_inference(sess, clip_in)
    dt       = time.perf_counter() - t0

    hr_pred = bvp_to_hr(bvp_pred)
    hr_gt   = bvp_to_hr(bvp_gt)
    r, _    = pearsonr(bvp_pred, bvp_gt)

    print(f"\n{'='*44}")
    print(f"  Clip index      : {clip_idx} / {n_clips - 1}")
    print(f"  HR predicted    : {hr_pred:.1f} BPM")
    print(f"  HR ground truth : {hr_gt:.1f} BPM")
    print(f"  HR MAE          : {abs(hr_pred - hr_gt):.1f} BPM")
    print(f"  BVP Pearson r   : {r:.3f}")
    print(f"  Inference       : {dt*1000:.1f} ms")
    print(f"{'='*44}")

    _save_bvp_plot(bvp_pred, bvp_gt, hr_pred, hr_gt, Path("bvp_output_npy.png"))
    print("BVP plot saved → bvp_output_npy.png")


# ---------------------------------------------------------------------------
# Visualization helpers  (--save-viz)
# ---------------------------------------------------------------------------

_VIZ_SIZE  = 320   # output frame size (both axes) for viz videos
_STRIP_H   = 80    # pixel height of BVP waveform strip at bottom


def _save_diffnorm_video(raw_frames: list[np.ndarray], fps: float, out: Path) -> None:
    """
    Write a video of DiffNorm temporal-difference frames — exactly what the model sees,
    rescaled from [-1, 1] to [0, 255].  Grey = no change; bright/dark = colour shift.
    """
    s = _VIZ_SIZE
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (s, s))
    for t in range(len(raw_frames) - 1):
        a    = cv2.resize(raw_frames[t],   (s, s)).astype(np.float32)
        b    = cv2.resize(raw_frames[t+1], (s, s)).astype(np.float32)
        diff = (b - a) / (b + a + 1e-6)                       # DiffNorm ∈ [-1, 1]
        vis  = np.clip((diff + 1.0) / 2.0 * 255, 0, 255).astype(np.uint8)
        cv2.putText(vis, f"DiffNorm  t={t}", (6, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
        writer.write(vis)
    writer.release()


def _save_amplified_video(
    raw_frames:  list[np.ndarray],
    bvp_concat:  np.ndarray,
    fps:         float,
    hr:          float,
    out:         Path,
    amplify:     float = 30.0,
) -> None:
    """
    Side-by-side video: original face (left) | EVM green-channel amplified (right).
    BVP waveform strip at the bottom with a moving cursor.

    EVM pipeline:
      1. Extract green channel across all frames
      2. Bandpass filter pixel-by-pixel to cardiac band (0.67–3.0 Hz)
      3. Scale up by `amplify` and add back to the original green channel
    """
    T   = len(raw_frames)
    s   = _VIZ_SIZE
    sh  = _STRIP_H
    out_w = s * 2

    writer = cv2.VideoWriter(
        str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, s + sh)
    )

    # Resize all frames once, extract green channel for EVM
    frames_arr = np.array(
        [cv2.resize(f, (s, s)) for f in raw_frames], dtype=np.float32
    )                                           # (T, s, s, 3)  BGR
    green      = frames_arr[:, :, :, 1]         # (T, s, s)

    nyq = fps / 2.0
    lo, hi = 0.67 / nyq, min(3.0 / nyq, 0.99)
    b_c, a_c = butter(3, [lo, hi], btype="band")

    green_flat = green.reshape(T, -1)           # (T, s*s)
    try:
        filt_flat = filtfilt(b_c, a_c, green_flat, axis=0)
    except Exception:
        filt_flat = green_flat - green_flat.mean(axis=0, keepdims=True)
    filtered = filt_flat.reshape(T, s, s)       # (T, s, s)

    # Normalise BVP waveform for the strip overlay
    bvp_n = bvp_concat - bvp_concat.mean()
    if bvp_n.std() > 0:
        bvp_n = bvp_n / bvp_n.std()
    n_bvp = len(bvp_n)

    for t in range(T):
        orig      = frames_arr[t].astype(np.uint8)

        amp_frame = frames_arr[t].copy()
        amp_frame[:, :, 1] = np.clip(amp_frame[:, :, 1] + amplify * filtered[t], 0, 255)
        amp_frame = amp_frame.astype(np.uint8)

        cv2.putText(orig,      "Original",       (6, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(amp_frame, f"EVM  x{amplify:.0f}",  (6, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 255, 120), 1, cv2.LINE_AA)

        side = np.concatenate([orig, amp_frame], axis=1)   # (s, 2s, 3)

        # BVP waveform strip
        strip = np.zeros((sh, out_w, 3), dtype=np.uint8)
        if n_bvp > 1:
            xs = (np.arange(n_bvp) / (n_bvp - 1) * (out_w - 1)).astype(int)
            ys = np.clip(
                (sh // 2 - bvp_n * (sh // 2 - 10)).astype(int), 0, sh - 1
            )
            pts = np.stack([xs, ys], axis=1).reshape(-1, 1, 2)
            cv2.polylines(strip, [pts], False, (100, 200, 255), 1, cv2.LINE_AA)
            cursor_t   = min(t, n_bvp - 1)
            cx, cy     = int(xs[cursor_t]), int(ys[cursor_t])
            cv2.line(strip,   (cx, 0), (cx, sh), (70, 70, 70), 1)
            cv2.circle(strip, (cx, cy), 4, (60, 255, 160), -1, cv2.LINE_AA)
        cv2.putText(strip, f"BVP   HR = {hr:.1f} BPM", (6, sh - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170, 170, 170), 1, cv2.LINE_AA)

        writer.write(np.concatenate([side, strip], axis=0))

    writer.release()


# ---------------------------------------------------------------------------
# Plot helper
# ---------------------------------------------------------------------------

def _save_bvp_plot(
    bvp_pred: np.ndarray,
    bvp_gt: np.ndarray | None,
    hr_pred: float,
    hr_gt: float | None,
    out: Path,
    fps: float = FPS,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.arange(len(bvp_pred)) / fps
    fig, axes = plt.subplots(1 if bvp_gt is None else 2, 1,
                             figsize=(10, 3 if bvp_gt is None else 5), sharex=True)

    if bvp_gt is None:
        axes = [axes]

    axes[0].plot(t, bvp_pred, color="steelblue")
    axes[0].set_ylabel("BVP predicted")
    title = f"Predicted HR: {hr_pred:.1f} BPM"
    if hr_gt is not None:
        title += f"   |   Ground truth HR: {hr_gt:.1f} BPM   |   MAE: {abs(hr_pred-hr_gt):.1f} BPM"
    axes[0].set_title(title)

    if bvp_gt is not None:
        axes[1].plot(t, bvp_gt, color="darkorange")
        axes[1].set_ylabel("BVP ground truth")
        axes[1].set_xlabel("Time (s)")
    else:
        axes[0].set_xlabel("Time (s)")

    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def sanity_check(sess: ort.InferenceSession) -> None:
    print("\n--- Sanity check (random input) ---")
    dummy = np.random.randn(1, CLIP_LEN, 3, INPUT_SIZE, INPUT_SIZE).astype(np.float32)
    bvp   = run_inference(sess, dummy)
    print(f"BVP range  : [{bvp.min():.3f}, {bvp.max():.3f}]")
    print(f"HR (noise, meaningless): {bvp_to_hr(bvp):.1f} BPM")
    print("ONNX runtime ✓")


def run_webcam_or_video(
    sess: ort.InferenceSession,
    source,           # 0 = webcam, str = video path
    n_clips: int,
    use_face_crop: bool,
    oximeter: float | None,
    brightness_norm: bool = True,
    save_viz: bool = False,
    amplify: float = 30.0,
    output_tag: str = "output",
    start_sec: float = 0.0,
) -> None:
    detector = make_face_detector() if use_face_crop else None
    if detector is None and use_face_crop:
        print("Face crop unavailable — running without it")

    # total raw frames needed: last clip ends at (n_clips-1)*STRIDE + CLIP_LEN + 1
    n_raw = (n_clips - 1) * STRIDE + CLIP_LEN + 1
    fps_label = "Video" if source != 0 else "Webcam"

    frames, fps = collect_frames(source, n_raw, detector=detector, start_sec=start_sec)

    if len(frames) < CLIP_LEN + 1:
        sys.exit(f"Need at least {CLIP_LEN + 1} frames, only got {len(frames)}")

    # Slice overlapping clips, run inference, collect BVP outputs
    bvp_segments = []
    t0 = time.perf_counter()

    actual_clips = min(n_clips, (len(frames) - CLIP_LEN - 1) // STRIDE + 1)
    for i in range(actual_clips):
        start = i * STRIDE
        window = frames[start : start + CLIP_LEN + 1]
        clip   = preprocess_frames(window, brightness_norm=brightness_norm)
        bvp    = run_inference(sess, clip)
        bvp_segments.append(bvp)

    dt = time.perf_counter() - t0

    # Per-clip HR (coarse, 11.25 BPM bins)
    hr_per_clip = [bvp_to_hr(b, fps=fps) for b in bvp_segments]
    hr_median   = float(np.median(hr_per_clip))
    hr_std      = float(np.std(hr_per_clip))

    # Concatenated BVP → finer Welch resolution
    # Each clip contributes STRIDE new frames; last clip contributes CLIP_LEN
    bvp_concat  = np.concatenate(
        [bvp_segments[i][:STRIDE] for i in range(len(bvp_segments) - 1)]
        + [bvp_segments[-1]]
    )
    hr_concat   = bvp_to_hr(bvp_concat, fps=fps)
    bin_width   = fps / len(bvp_concat) * 60

    print(f"\n{'='*50}")
    print(f"  Clips run      : {actual_clips}  ({fps_label} @ {fps:.1f} FPS)")
    print(f"  Face crop      : {'on' if use_face_crop and detector is not None else 'off'}")
    print(f"  Brightness norm: {'on' if brightness_norm else 'off'}")
    print(f"  HR per clip    : {[f'{h:.1f}' for h in hr_per_clip]}  (bin={11.25:.2f} BPM)")
    print(f"  HR median      : {hr_median:.1f} BPM  (σ={hr_std:.1f})")
    print(f"  HR concatenated: {hr_concat:.1f} BPM  (bin={bin_width:.2f} BPM ← finer)")
    if oximeter is not None:
        print(f"  HR oximeter    : {oximeter:.1f} BPM")
        print(f"  MAE (median)   : {abs(hr_median  - oximeter):.1f} BPM")
        print(f"  MAE (concat)   : {abs(hr_concat  - oximeter):.1f} BPM")
    print(f"  Inference time : {dt*1000:.0f} ms")
    print(f"{'='*50}")

    plot_path = Path(f"bvp_{output_tag}.png")
    _save_bvp_plot(bvp_concat, None, hr_concat, oximeter, plot_path, fps=fps)
    print(f"BVP plot (concatenated) saved → {plot_path}")

    if save_viz:
        out_dir = Path("viz_output")
        out_dir.mkdir(exist_ok=True)
        dn_path  = out_dir / "diffnorm.mp4"
        amp_path = out_dir / "amplified.mp4"
        print("\nSaving visualizations ...")
        _save_diffnorm_video(frames, fps, dn_path)
        print(f"  DiffNorm video  → {dn_path}")
        _save_amplified_video(frames, bvp_concat, fps, hr_concat, amp_path, amplify=amplify)
        print(f"  EVM amplified   → {amp_path}  (x{amplify:.0f} green channel)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",         help="Path to a video file (default: webcam)")
    parser.add_argument("--camera-index", type=int, default=0,
                        help="Webcam device index (0=default, 1=OBS virtual camera)")
    parser.add_argument("--npy",           help="Path to {subject}_clips.npy")
    parser.add_argument("--clip-idx",      type=int, default=0)
    parser.add_argument("--all-clips",     action="store_true")
    parser.add_argument("--n-clips",       type=int, default=3,
                        help="Overlapping clips to run and median (default: 3, ~10.7 s)")
    parser.add_argument("--no-face-crop",       action="store_true", help="Disable Haar face crop")
    parser.add_argument("--no-brightness-norm", action="store_true",
                        help="Disable per-frame luminance normalisation (A/B test vs AE)")
    parser.add_argument("--oximeter",           type=float, help="Finger oximeter HR (BPM)")
    parser.add_argument("--save-viz",           action="store_true",
                        help="Write viz_output/diffnorm.mp4 and viz_output/amplified.mp4")
    parser.add_argument("--amplify",            type=float, default=30.0,
                        help="EVM green-channel amplification factor (default: 30)")
    parser.add_argument("--start-sec",            type=float, default=0.0,
                        help="Seek to this timestamp (seconds) before collecting frames (video only)")
    parser.add_argument("--onnx-path",           default=None,
                        help="Override default ONNX path (default: models/weights/vitallens_rppg.onnx)")
    parser.add_argument("--sanity",             action="store_true")
    args = parser.parse_args()

    onnx_path = Path(args.onnx_path) if args.onnx_path else ONNX_PATH
    if not onnx_path.exists():
        sys.exit(f"ONNX not found at {onnx_path} — run from VitalLens project root")

    sess = load_session(onnx_path)

    if args.sanity:
        sanity_check(sess)
        return

    if args.npy:
        run_npy(sess, args.npy, args.clip_idx, all_clips=args.all_clips)
        return

    source = args.video if args.video else args.camera_index
    run_webcam_or_video(
        sess,
        source          = source,
        n_clips         = args.n_clips,
        use_face_crop   = not args.no_face_crop,
        oximeter        = args.oximeter,
        brightness_norm = not args.no_brightness_norm,
        save_viz        = args.save_viz,
        amplify         = args.amplify,
        output_tag      = onnx_path.stem,
        start_sec       = args.start_sec,
    )


if __name__ == "__main__":
    main()
