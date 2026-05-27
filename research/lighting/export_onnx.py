"""
Export lighting classifier to ONNX.
Run this on SageMaker where lighting_best.pt lives:

  python export_lighting_onnx.py
  # then download models/weights/vitallens_lighting.onnx to local machine
"""
from pathlib import Path
from models.lighting.model import LightingClassifier, export_onnx

ckpt_path = Path("models/weights/lighting_best.pt")
out_path  = Path("models/weights/vitallens_lighting.onnx")

if not ckpt_path.exists():
    raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

model = LightingClassifier.load(ckpt_path)
export_onnx(model, output_path=out_path)
print(f"\nDownload this file to your local machine:\n  {out_path}")
