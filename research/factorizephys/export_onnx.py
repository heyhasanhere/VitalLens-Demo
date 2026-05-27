"""
Export phase2/best.pth (FactorizePhys+FSAM) to ONNX for VitalLens integration.

Input:  (1, 3, 160, 72, 72)  raw RGB frames in [0, 1]  (C, T, H, W)
Output: (1, 159)              BVP signal  (T-1, due to internal torch.diff)

Usage:
    python export_onnx.py
    python export_onnx.py --checkpoint checkpoints/phase2/best.pth --output checkpoints/phase2/factorizephys.onnx
"""
import sys
import argparse
from pathlib import Path
from collections import OrderedDict

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).parent
FP_ROOT = PROJECT_ROOT / 'external' / 'FactorizePhys'
sys.path.insert(0, str(FP_ROOT))

from neural_methods.model.FactorizePhys.FactorizePhys import FactorizePhys, model_config


def make_fp_cfg() -> dict:
    cfg = dict(model_config)   # start from defaults
    cfg['MD_FSAM']      = True
    cfg['MD_TYPE']      = 'NMF'
    cfg['MD_R']         = 1
    cfg['MD_S']         = 1
    cfg['MD_STEPS']     = 4
    cfg['MD_RESIDUAL']  = True
    cfg['MD_INFERENCE'] = True
    # RAND_INIT=True is the default — bases are fresh torch.ones() each forward
    return cfg


class FPWrapper(nn.Module):
    """Strip multi-output tuple; return only rPPG signal."""
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        return out[0]   # (B, T-1)


def build_model(ckpt_path: Path) -> nn.Module:
    cfg = make_fp_cfg()
    model = FactorizePhys(
        frames=160,
        md_config=cfg,
        device=torch.device('cpu'),
        in_channels=3,
    )
    raw = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
    sd  = OrderedDict((k.replace('module.', ''), v) for k, v in raw.items())
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"  Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"  Unexpected keys: {unexpected}")
    return model


def export(ckpt_path: Path, output_path: Path, opset: int = 17) -> None:
    print(f"Checkpoint : {ckpt_path}")
    print(f"Output     : {output_path}")

    model = build_model(ckpt_path)
    model.eval()

    wrapper = FPWrapper(model)
    wrapper.eval()

    dummy = torch.zeros(1, 3, 160, 72, 72)

    # Verify PyTorch output before export
    with torch.no_grad():
        pt_out = wrapper(dummy)
    print(f"PyTorch output shape : {pt_out.shape}")   # expect (1, 159)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # torch.diff is not supported by ONNX — patch it to slice-subtract
    # which is equivalent for n=1, no prepend/append (the only usage in FactorizePhys).
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
            dynamic_axes={
                'video_clip': {0: 'batch'},
                'bvp':        {0: 'batch'},
            },
            opset_version=opset,
            do_constant_folding=True,
        )
    finally:
        torch.diff = _orig_diff
    print(f"Exported ONNX → {output_path}  ({output_path.stat().st_size / 1024:.1f} KB)")

    # Validate with onnxruntime
    try:
        import onnxruntime as ort
        sess    = ort.InferenceSession(str(output_path), providers=['CPUExecutionProvider'])
        ort_out = sess.run(None, {'video_clip': dummy.numpy()})[0]
        print(f"ONNX output shape    : {ort_out.shape}")

        import numpy as np
        max_diff = float(np.abs(ort_out - pt_out.numpy()).max())
        print(f"Max |ONNX - PyTorch| : {max_diff:.2e}")

        # InstanceNorm float32 accumulation can produce ~2e-3 diff — acceptable
        # for rPPG (FFT peak detection is robust to small numerical differences)
        if max_diff < 5e-3:
            print("VALIDATION PASSED")
        else:
            print(f"WARNING: diff {max_diff:.2e} exceeds 5e-3 — check model")
    except ImportError:
        print("onnxruntime not installed — skipping validation")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='checkpoints/phase2/best.pth')
    parser.add_argument('--output',     default='checkpoints/phase2/factorizephys.onnx')
    parser.add_argument('--opset',      type=int, default=17)
    args = parser.parse_args()

    export(
        ckpt_path   = PROJECT_ROOT / args.checkpoint,
        output_path = PROJECT_ROOT / args.output,
        opset       = args.opset,
    )


if __name__ == '__main__':
    main()
