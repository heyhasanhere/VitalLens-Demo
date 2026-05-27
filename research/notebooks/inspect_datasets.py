"""
Dataset Inspector
=================
Run this as soon as your datasets are downloaded to verify structure,
frame counts, and ground truth availability.

Usage:
    python notebooks/inspect_datasets.py
"""

import sys
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


def inspect_ubfc(root: Path):
    print("\n" + "="*50)
    print(f"UBFC-rPPG  →  {root}")
    print("="*50)

    if not root.exists():
        print("  ✗ Directory not found. Check your data path.")
        return

    subject_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    print(f"  Subjects found: {len(subject_dirs)}")

    hr_values = []

    for subj in subject_dirs:
        vid_candidates = sorted(subj.glob("*.avi"))
        gt_candidates  = [subj / n for n in ("ground_truth.txt", "gtdump.xmp") if (subj / n).exists()]

        vid_ok = bool(vid_candidates)
        gt_ok  = bool(gt_candidates)

        if not vid_ok or not gt_ok:
            print(f"  {subj.name}: MISSING {'video' if not vid_ok else 'GT'}")
            continue

        vid = vid_candidates[0]
        gt  = gt_candidates[0]

        cap = cv2.VideoCapture(str(vid))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        delimiter = "," if gt.suffix == ".xmp" else None
        gt_data   = np.loadtxt(str(gt), delimiter=delimiter)
        # gtdump.xmp: cols = [timestamp, HR, SpO2, BVP]
        # ground_truth.txt: cols = [timestamp, BVP, HR]
        if gt.suffix == ".xmp":
            hr_col = gt_data[:, 1]
        else:
            hr_col = gt_data[:, 2] if gt_data.ndim > 1 and gt_data.shape[1] > 2 else []
        mean_hr = float(np.mean(hr_col)) if len(hr_col) > 0 else 0

        hr_values.append(mean_hr)
        print(f"  {subj.name}: {n_frames} frames @ {fps:.1f}fps, "
              f"{w}x{h}, GT={gt.name}, mean HR={mean_hr:.1f} BPM")

    print(f"\n  Summary: {len(subject_dirs)} subjects inspected")
    if hr_values:
        print(f"  HR range: {min(hr_values):.1f}–{max(hr_values):.1f} BPM")


def inspect_pure(root: Path):
    print("\n" + "="*50)
    print(f"PURE  →  {root}")
    print("="*50)

    if not root.exists():
        print("  ✗ Directory not found. Check your data path.")
        return

    import re, json
    session_dirs = sorted([
        d for d in root.iterdir()
        if d.is_dir() and re.match(r"\d{2}-\d{2}", d.name)
    ])
    print(f"  Sessions found: {len(session_dirs)}")

    for sess in session_dirs[:6]:
        pngs = list(sess.glob("*.png"))
        jsons = list(sess.glob("*.json"))
        print(f"  {sess.name}: {len(pngs)} frames, "
              f"{'GT ✓' if jsons else 'GT ✗'}")

    if len(session_dirs) > 6:
        print(f"  ... and {len(session_dirs) - 6} more sessions")


def inspect_mmpd(root: Path):
    print("\n" + "="*50)
    print(f"MMPD  →  {root}")
    print("="*50)

    if not root.exists():
        print("  ✗ Directory not found. Check your data path.")
        return

    # MMPD structure varies — just show what's there
    all_items = list(root.iterdir())
    dirs = [i for i in all_items if i.is_dir()]
    files = [i for i in all_items if i.is_file()]

    print(f"  Subdirectories: {len(dirs)}")
    for d in sorted(dirs)[:5]:
        contents = list(d.iterdir())
        print(f"    {d.name}/  ({len(contents)} items)")
    if len(dirs) > 5:
        print(f"    ... and {len(dirs) - 5} more")

    if files:
        print(f"  Files at root: {[f.name for f in files[:5]]}")


def main():
    project_root = Path(__file__).parent.parent.parent  # C:/UTS/4/Deep Learning/Project

    inspect_ubfc(project_root / "UBFC_Dataset" / "DATASET_2")
    inspect_pure(project_root / "PURE_Dataset")
    inspect_mmpd(project_root / "MMPD_Dataset")

    print("\n" + "="*50)
    print("Done. If structure looks wrong, adjust paths in")
    print("models/rppg/dataset.py and models/lighting/dataset.py")
    print("="*50)


if __name__ == "__main__":
    main()
