"""
Export PhysFormer (ViT_ST_ST_Compact3_TDC_gra_sharp) to ONNX.

Input:  (1, 3, 161, 72, 72)  raw RGB frames in [0, 1]  (C, T+1, H, W)
        T+1=161 because the model internally applies torch.diff on the time axis.
        The preprocessing appends the last frame to 160 → 161.
Output: (1, 160)              BVP signal

Default checkpoint: external/FactorizePhys/final_model_release/UBFC-rPPG_PhysFormer.pth

Usage:
    python export_onnx.py
    python export_onnx.py --checkpoint /path/to/PhysFormer.pth --output physformer.onnx
"""
import sys
import argparse
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).parent
# Try the local submodule first; fall back to the sibling rppg_project checkout.
_candidates = [
    PROJECT_ROOT.parent / 'factorizephys' / 'external' / 'FactorizePhys',
    Path('/mnt/sata-ssd/rppg_project/external/FactorizePhys'),
]
FP_ROOT = next((p for p in _candidates if (p / 'neural_methods').exists()), _candidates[0])
sys.path.insert(0, str(FP_ROOT))

from neural_methods.model.PhysFormer import ViT_ST_ST_Compact3_TDC_gra_sharp

_FRAMES     = 160
_HEIGHT     = 72
_WIDTH      = 72
_PATCH_SIZE = 4
_DIM        = 96
_FF_DIM     = 144
_NUM_HEADS  = 4
_NUM_LAYERS = 12
_THETA      = 0.7
_GRA_SHARP  = 2.0


class PhysFormerWrapper(nn.Module):
    """Strips multi-output tuple and hardcodes gra_sharp=2.0 for ONNX export."""
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rppg, _, _, _ = self.model(x, _GRA_SHARP)
        return rppg   # (B, T)


def build_model(ckpt_path: Path) -> nn.Module:
    model = ViT_ST_ST_Compact3_TDC_gra_sharp(
        in_channels=3,
        image_size=(_FRAMES, _HEIGHT, _WIDTH),
        patches=(_PATCH_SIZE,) * 3,
        dim=_DIM,
        ff_dim=_FF_DIM,
        num_heads=_NUM_HEADS,
        num_layers=_NUM_LAYERS,
        dropout_rate=0.0,
        theta=_THETA,
    )
    raw = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
    sd  = OrderedDict((k.replace('module.', ''), v) for k, v in raw.items())
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"  Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"  Unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
    return model


def export(ckpt_path: Path, output_path: Path, opset: int = 17) -> None:
    print(f"Checkpoint : {ckpt_path}")
    print(f"Output     : {output_path}")

    model   = build_model(ckpt_path)
    model.eval()
    wrapper = PhysFormerWrapper(model)
    wrapper.eval()

    # T+1=161 frames: model internally does torch.diff to get T=160 diffs
    dummy = torch.zeros(1, 3, _FRAMES + 1, _HEIGHT, _WIDTH)

    with torch.no_grad():
        pt_out = wrapper(dummy)
    print(f"PyTorch output shape : {pt_out.shape}")   # expect (1, 160)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # torch.diff is not ONNX-native — patch with slice-subtract (same as FactorizePhys export)
    _orig_diff = torch.diff
    def _onnx_diff(input, n=1, dim=-1, prepend=None, append=None):
        if n == 1 and prepend is None and append is None:
            slc_a = [slice(None)] * input.dim()
            slc_b = [slice(None)] * input.dim()
            slc_a[dim] = slice(1, None)
            slc_b[dim] = slice(None, -1)
            return input[tuple(slc_a)] - input[tuple(slc_b)]
        return _orig_diff(input, n=n, dim=dim, prepend=prepend, append=append)
    torch.diff = _onnx_diff

    try:
        torch.onnx.export(
            wrapper,
            dummy,
            str(output_path),
            input_names=['video_clip'],
            output_names=['bvp'],
            dynamic_axes={'video_clip': {0: 'batch'}, 'bvp': {0: 'batch'}},
            opset_version=opset,
            do_constant_folding=True,
        )
    finally:
        torch.diff = _orig_diff

    print(f"Exported ONNX → {output_path}  ({output_path.stat().st_size / 1024:.1f} KB)")

    try:
        import onnxruntime as ort
        import numpy as np
        sess    = ort.InferenceSession(str(output_path), providers=['CPUExecutionProvider'])
        ort_out = sess.run(None, {'video_clip': dummy.numpy()})[0]
        print(f"ONNX output shape    : {ort_out.shape}")
        max_diff = float(np.abs(ort_out - pt_out.numpy()).max())
        print(f"Max |ONNX - PyTorch| : {max_diff:.2e}")
        if max_diff < 5e-3:
            print("VALIDATION PASSED")
        else:
            print(f"WARNING: diff {max_diff:.2e} exceeds 5e-3")
    except ImportError:
        print("onnxruntime not installed — skipping validation")


def main() -> None:
    default_ckpt = FP_ROOT / 'final_model_release' / 'UBFC-rPPG_PhysFormer.pth'
    default_out  = PROJECT_ROOT.parent.parent / 'weights' / 'physformer.onnx'

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default=str(default_ckpt))
    parser.add_argument('--output',     default=str(default_out))
    parser.add_argument('--opset',      type=int, default=17)
    args = parser.parse_args()

    export(
        ckpt_path   = Path(args.checkpoint),
        output_path = Path(args.output),
        opset       = args.opset,
    )


if __name__ == '__main__':
    main()
