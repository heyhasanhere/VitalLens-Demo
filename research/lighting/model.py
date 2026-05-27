"""
Lighting Condition Classifier
==============================
MobileNetV3-Small fine-tuned for 3-class lighting classification:
  0 = Good   (bright, even illumination)
  1 = Poor   (dark or harsh shadows)
  2 = Mixed  (moderate / uneven)

Fine-tuning strategy:
  Phase 1 (epochs 1-5):  backbone frozen, only classifier head trains
  Phase 2 (epochs 6-15): top 3 backbone layers unfrozen, full fine-tuning

This two-phase approach prevents the pretrained ImageNet features from being
destroyed early in training when the randomly-initialised head produces large
gradients.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


LABEL_NAMES = ["Good", "Poor", "Mixed"]
NUM_CLASSES  = len(LABEL_NAMES)


class LightingClassifier(nn.Module):
    """
    MobileNetV3-Small with a custom classification head for lighting detection.

    Parameters
    ----------
    pretrained : bool
        Load ImageNet-pretrained weights (default True). Set False for unit tests.
    dropout : float
        Dropout rate in the classifier head (default 0.3).
    """

    def __init__(self, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()

        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        base    = mobilenet_v3_small(weights=weights)

        # Keep the full convolutional backbone
        self.features = base.features    # 16 InvertedResidual blocks
        self.avgpool  = base.avgpool     # AdaptiveAvgPool2d(1)

        # Replace ImageNet head (1000 classes) with our 3-class head
        in_features = base.classifier[0].in_features  # 576
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, NUM_CLASSES),
        )

        self._phase = 1
        self.freeze_backbone()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor [B, 3, 224, 224] float32, ImageNet-normalised

        Returns
        -------
        logits : Tensor [B, 3]
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (predicted_class_indices [B], confidence_scores [B])."""
        self.train(False)
        logits     = self(x)
        probs      = torch.softmax(logits, dim=1)
        confidence, predicted = probs.max(dim=1)
        self.train(True)
        return predicted, confidence

    @torch.no_grad()
    def predict_label(self, x: torch.Tensor) -> Tuple[str, float]:
        """
        Single-frame convenience wrapper.
        x : [1, 3, 224, 224]
        Returns (label_string, confidence_float).
        """
        predicted, confidence = self.predict(x)
        return LABEL_NAMES[predicted.item()], confidence.item()

    # ------------------------------------------------------------------
    # Fine-tuning phase management
    # ------------------------------------------------------------------

    def freeze_backbone(self) -> None:
        """Phase 1: freeze all backbone params, train the head only."""
        for param in self.features.parameters():
            param.requires_grad = False
        for param in self.classifier.parameters():
            param.requires_grad = True
        self._phase = 1
        print("LightingClassifier: Phase 1 — backbone frozen, training head only")

    def unfreeze_top_layers(self, n_layers: int = 3) -> None:
        """
        Phase 2: unfreeze the last n_layers of the MobileNetV3 backbone.

        MobileNetV3-Small has 16 feature blocks (0-15).
        Unfreezing the last 3 exposes high-level feature extractors
        while keeping low-level edge/texture detectors frozen.
        """
        for param in self.features.parameters():
            param.requires_grad = False

        for block in list(self.features.children())[-n_layers:]:
            for param in block.parameters():
                param.requires_grad = True

        self._phase = 2
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.parameters())
        print(
            f"LightingClassifier: Phase 2 — top {n_layers} layers unfrozen "
            f"({trainable:,} / {total:,} params trainable)"
        )

    def trainable_params(self):
        return [p for p in self.parameters() if p.requires_grad]

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.state_dict(),
            "phase":       self._phase,
            "label_names": LABEL_NAMES,
        }, path)
        print(f"Saved → {path}")

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "LightingClassifier":
        ckpt  = torch.load(path, map_location=device, weights_only=True)
        model = cls(pretrained=False)
        model.load_state_dict(ckpt["state_dict"])
        model._phase = ckpt.get("phase", 2)
        model.train(False)
        return model


# ---------------------------------------------------------------------------
# ONNX Export
# ---------------------------------------------------------------------------

def export_onnx(
    model: LightingClassifier,
    output_path: str | Path = "backend/models/vitallens_lighting.onnx",
    device: str = "cpu",
) -> None:
    """
    Export to ONNX for CPU inference via ONNX Runtime.

    Input shape:  [B, 3, 224, 224]  (ImageNet-normalised RGB)
    Output shape: [B, 3]            (logits → softmax for probabilities)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.train(False).to(device)
    dummy = torch.randn(1, 3, 224, 224, device=device)

    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        input_names=["frame"],
        output_names=["logits"],
        dynamic_axes={"frame": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    print(f"Exported ONNX → {output_path}")

    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(
            str(output_path), providers=["CPUExecutionProvider"]
        )
        out = sess.run(None, {"frame": dummy.numpy()})
        print(f"ONNX validation passed — output shape: {out[0].shape}")
    except ImportError:
        print("onnxruntime not installed — skipping validation")


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model = LightingClassifier(pretrained=False)
    dummy = torch.randn(4, 3, 224, 224)

    out = model(dummy)
    print(f"Phase 1 output: {out.shape}")
    print(f"Trainable params: {sum(p.numel() for p in model.trainable_params()):,}")

    model.unfreeze_top_layers(3)
    out = model(dummy)
    print(f"Phase 2 output: {out.shape}")
    print(f"Trainable params: {sum(p.numel() for p in model.trainable_params()):,}")

    label, conf = model.predict_label(dummy[:1])
    print(f"Predicted: {label} ({conf:.3f})")
