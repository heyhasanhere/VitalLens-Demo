"""
Test the preprocessing pipeline on a single image or webcam snapshot.

Usage:
    # Test on a saved image
    python preprocessing/test_pipeline.py --image path/to/face.jpg

    # Test on a live webcam snapshot (captures one frame then exits)
    python preprocessing/test_pipeline.py --webcam

    # Test all lighting_class modes
    python preprocessing/test_pipeline.py --image face.jpg --all-modes
"""

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

# Add project root to path so imports work from any directory
sys.path.insert(0, str(Path(__file__).parent.parent))
from preprocessing.pipeline import PreprocessingPipeline


def capture_webcam_frame() -> np.ndarray:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam.")
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Failed to capture frame.")
    return frame


def visualise_pipeline(frame: np.ndarray, lighting_class: str = "poor") -> None:
    """Show before/after grid for each pipeline step."""
    pipeline = PreprocessingPipeline()

    # Run each step individually to visualise intermediate results
    steps = {}
    steps["0. Original"] = frame.copy()

    after_clahe = pipeline._apply_clahe(frame.copy())
    steps["1. CLAHE"] = after_clahe

    after_wb = pipeline._apply_white_balance(after_clahe.copy())
    steps["2. White Balance"] = after_wb

    after_gamma, gamma_val = pipeline._apply_gamma(after_wb.copy())
    steps[f"3. Gamma (γ={gamma_val:.2f})"] = after_gamma

    # Temporal denoise needs 3 frames — simulate by processing 3 identical frames
    pipeline.reset()
    for _ in range(3):
        after_denoise = pipeline._apply_temporal_denoise(after_gamma.copy())
    steps["4. Temporal Denoise"] = after_denoise

    # Full pipeline result with face ROI
    pipeline.reset()
    result = pipeline.process(frame.copy(), lighting_class=lighting_class)
    steps["5. Full Pipeline"] = result.corrected_frame

    n = len(steps)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 5))
    fig.suptitle(
        f"Preprocessing Pipeline — lighting_class='{lighting_class}' | "
        f"Face detected: {result.face_detected}",
        fontsize=13,
    )

    for ax, (title, img) in zip(axes, steps.items()):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(rgb)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    # Draw face bbox on the final panel if detected
    if result.face_detected and result.face_bbox is not None:
        x, y, w, h = result.face_bbox
        ax = axes[-1]
        rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor="lime", facecolor="none")
        ax.add_patch(rect)

    plt.tight_layout()
    plt.savefig("preprocessing_test_output.png", dpi=150, bbox_inches="tight")
    print("Saved: preprocessing_test_output.png")
    plt.show()

    # Also show ROI patches if face was detected
    if result.face_detected:
        _show_roi_patches(result.roi_patches)


def _show_roi_patches(roi_patches: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(9, 4))
    fig.suptitle("Extracted Face ROI Patches (72×72px)", fontsize=12)
    for ax, (name, patch) in zip(axes, roi_patches.items()):
        rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
        ax.imshow(rgb)
        ax.set_title(name.replace("_", " ").title())
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("roi_patches_output.png", dpi=150, bbox_inches="tight")
    print("Saved: roi_patches_output.png")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Test VitalLens preprocessing pipeline")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=str, help="Path to input image")
    group.add_argument("--webcam", action="store_true", help="Capture from webcam")
    parser.add_argument(
        "--lighting",
        choices=["good", "poor", "mixed"],
        default="poor",
        help="Lighting class to simulate (default: poor)",
    )
    parser.add_argument(
        "--all-modes",
        action="store_true",
        help="Run all lighting modes and compare",
    )
    args = parser.parse_args()

    if args.webcam:
        print("Capturing webcam frame...")
        frame = capture_webcam_frame()
        print(f"Captured frame: {frame.shape}")
    else:
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"Error: could not load image '{args.image}'")
            sys.exit(1)
        print(f"Loaded image: {frame.shape}")

    if args.all_modes:
        for mode in ["good", "poor", "mixed"]:
            print(f"\n--- Lighting mode: {mode} ---")
            visualise_pipeline(frame.copy(), lighting_class=mode)
    else:
        visualise_pipeline(frame, lighting_class=args.lighting)


if __name__ == "__main__":
    main()
