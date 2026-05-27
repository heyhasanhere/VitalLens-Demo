"""
Phase 3 — Temporal Consistency Fine-Tuning.
Launch: /home/dex/rppg_venv/bin/torchrun --nproc_per_node=2 src/train_phase3.py
Config: checkpoints/phase3/config.json

Objective: reduce HR jump between consecutive overlapping sliding windows.
Each sample returns two consecutive overlapping 160-frame clips (stride=30).
Temporal consistency loss: NegPearson between predictions in the 130-frame overlap region.
No SCAMPS — pure MCD frontal only.
"""
import sys, os, json, time, math, warnings
from pathlib import Path
from collections import OrderedDict

import numpy as np
import pandas as pd
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from scipy.signal import butter, filtfilt
from tqdm.auto import tqdm

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path('/mnt/sata-ssd/rppg_project')
FP_ROOT      = PROJECT_ROOT / 'external' / 'FactorizePhys'
sys.path.insert(0, str(FP_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

# ── Config ────────────────────────────────────────────────────────────────────
CFG_PATH = PROJECT_ROOT / 'checkpoints' / 'phase3' / 'config.json'
with open(CFG_PATH) as f:
    CFG = json.load(f)

UBFC_CACHE      = Path(CFG['ubfc_cache'])
RPPG10_CACHE    = Path(CFG['rppg10_cache'])
MCD_EVAL_CACHE  = Path(CFG['mcd_eval_cache'])
MCD_HELD_H5_DIR = Path(CFG['mcd_held_h5_dir'])
MCD_CACHE_DIR   = Path(CFG['mcd_cache_dir'])
MCD_SPLIT_JSON  = Path(CFG['mcd_split_json'])
START_CKPT      = Path(CFG['start_ckpt'])
BEST_CKPT       = Path(CFG['best_ckpt'])
LAST_CKPT       = Path(CFG['last_ckpt'])
METRICS_JSON    = Path(CFG['metrics_json'])

EPOCHS          = CFG['epochs']
BATCH_SIZE      = CFG['batch_size']
CLIP_LEN        = CFG['clip_len']       # 160 frames per window
WINDOW_STRIDE   = CFG['window_stride']  # 30 frames between consecutive windows
IMG_SIZE        = CFG['img_size']
LR_MAX          = CFG['lr_max']
LR_MIN          = CFG['lr_min']
WEIGHT_DECAY    = CFG['weight_decay']
GRAD_CLIP       = CFG['grad_clip']
GRAD_ACCUM_STEPS = CFG['grad_accum_steps']
LOSS_CAP        = CFG['loss_cap']
LAMBDA_TC       = CFG['lambda_tc']
SNR_THRESHOLD   = CFG['snr_threshold']
NUM_WORKERS     = CFG['num_workers']
SEED            = CFG['seed']
EARLY_STOP_PATIENCE = CFG.get('early_stop_patience', 8)
CLEARML_TASK_ID = CFG.get('clearml_task_id')
SCAMPS_CACHE    = Path(CFG['scamps_cache_dir']) if CFG.get('scamps_cache_dir') else None
SCAMPS_CSV      = Path(CFG['scamps_split_csv']) if CFG.get('scamps_split_csv') else None
SCAMPS_CPX      = CFG.get('scamps_clips_per_subj', 2)
SCAMPS_FREQ     = CFG.get('scamps_freq', 9)
FP_PURE_CKPT    = Path(CFG['fp_pure_ckpt']) if CFG.get('fp_pure_ckpt') else None
LAMBDA_HR       = CFG.get('lambda_hr', 0.5)
LAMBDA_KD       = CFG.get('lambda_kd', 0.3)
LAMBDA_SNR      = CFG.get('lambda_snr', 0.1)

EPOCH_CKPT_DIR = Path(CFG['best_ckpt']).parent / 'epochs'
EPOCH_CKPT_DIR.mkdir(parents=True, exist_ok=True)

# Overlap region length: frames shared between consecutive windows
OVERLAP = CLIP_LEN - WINDOW_STRIDE  # 130 frames


# ── Augmentation ──────────────────────────────────────────────────────────────
class VideoAugmentFast:
    """Same augmentations as Phase 2, applied to the full extended clip."""
    def __init__(self):
        self.p_noise   = 0.35
        self.p_sat     = 0.45
        self.p_gamma   = 0.45
        self.p_geom    = 0.30
        self.p_erase   = 0.25
        self.p_roi     = 0.20
        self.p_subface = 0.30

    def __call__(self, x):
        if np.random.random() < self.p_noise:   x = self._noise(x)
        if np.random.random() < self.p_sat:     x = self._sat(x)
        if np.random.random() < self.p_gamma:   x = self._gamma(x)
        if np.random.random() < self.p_geom:    x = self._geom(x)
        if np.random.random() < self.p_erase:   x = self._erase(x)
        if np.random.random() < self.p_roi:     x = self._roi(x)
        if np.random.random() < self.p_subface: x = self._subface(x)
        return x

    def _noise(self, x):
        return (x + torch.randn_like(x) * np.random.uniform(0.003, 0.02)).clamp_(0, 1)

    def _sat(self, x):
        s      = np.random.uniform(0.7, 1.5)
        scales = torch.tensor(
            [np.random.uniform(0.90, 1.10),
             np.random.uniform(0.95, 1.05),
             np.random.uniform(0.90, 1.10)],
            dtype=x.dtype).view(3, 1, 1, 1)
        gray = 0.299*x[0:1] + 0.587*x[1:2] + 0.114*x[2:3]
        return ((gray + s * (x - gray)) * scales).clamp_(0, 1)

    def _gamma(self, x):
        g    = np.random.uniform(0.7, 1.4)
        bias = np.random.uniform(-0.05, 0.05)
        return (x.clamp(1e-6, 1.0) ** g + bias).clamp_(0, 1)

    def _geom(self, x):
        C, T, H, W = x.shape
        ang   = np.random.uniform(-6, 6)
        tx    = np.random.uniform(-4, 4) / W * 2
        ty    = np.random.uniform(-4, 4) / H * 2
        cos_a = math.cos(math.radians(ang))
        sin_a = math.sin(math.radians(ang))
        theta = torch.tensor(
            [[cos_a, -sin_a, tx], [sin_a, cos_a, ty]], dtype=x.dtype
        ).unsqueeze(0)
        imgs = x.permute(1, 0, 2, 3)
        grid = F.affine_grid(theta.expand(T, -1, -1), imgs.shape, align_corners=False)
        out  = F.grid_sample(imgs, grid, mode='bilinear',
                             padding_mode='reflection', align_corners=False)
        return out.permute(1, 0, 2, 3).clamp_(0, 1)

    def _erase(self, x):
        C, T, H, W = x.shape
        frac = np.random.uniform(0.05, 0.20)
        rh, rw = int(H * frac**0.5), int(W * frac**0.5)
        y0 = np.random.randint(0, max(1, H - rh))
        x0 = np.random.randint(0, max(1, W - rw))
        out = x.clone()
        out[:, :, y0:y0+rh, x0:x0+rw] = 0.0
        return out

    def _roi(self, x):
        C, T, H, W = x.shape
        frac = np.random.uniform(0.55, 0.65)
        rh, rw = int(H * frac), int(W * frac)
        y0, x0 = (0, 0) if np.random.random() < 0.5 else (0, W - rw)
        patch  = x[:, :, y0:y0+rh, x0:x0+rw]
        p4d    = patch.reshape(C*T, 1, rh, rw)
        resized = F.interpolate(p4d, size=(H, W), mode='bilinear', align_corners=False)
        return resized.reshape(C, T, H, W).clamp_(0, 1)

    def _subface(self, x):
        """Simulate cheek/forehead-only ROI crops (as in rPPG10 evaluation)."""
        C, T, H, W = x.shape
        mode = np.random.choice(['cheek_right', 'cheek_left', 'forehead'])
        if mode == 'cheek_right':
            crop = x[:, :, H//3:,    W//2:  ]
        elif mode == 'cheek_left':
            crop = x[:, :, H//3:,    :W//2  ]
        else:
            crop = x[:, :, :H//3,    W//4:3*W//4]
        c, t, ch, cw = crop.shape
        resized = F.interpolate(crop.reshape(c*t, 1, ch, cw),
                                size=(H, W), mode='bilinear', align_corners=False)
        return resized.reshape(C, T, H, W).clamp_(0, 1)


# ── SCAMPS dataset (regularization anchor) ────────────────────────────────────
class SCAMPSCacheDataset(Dataset):
    MAX_OPEN = 24

    def __init__(self, subject_ids, cache_dir, clip_len=160, clips_per_subj=2, seed=42):
        self.cache_dir = Path(cache_dir)
        self.clip_len  = clip_len
        self.max_start = 600 - clip_len - 1

        rng = np.random.RandomState(seed)
        self.clips = []
        for sid in subject_ids:
            if not (self.cache_dir / f'{sid}.h5').exists():
                continue
            starts = rng.choice(self.max_start, size=min(clips_per_subj, self.max_start),
                                replace=False).tolist()
            for s in starts:
                self.clips.append({'sid': sid, 'start': int(s)})

    def __len__(self): return len(self.clips)

    def __getstate__(self):
        s = self.__dict__.copy(); s.pop('_h5c', None); return s

    def _get_h5(self, path):
        if not hasattr(self, '_h5c'): self._h5c = OrderedDict()
        if path in self._h5c:
            self._h5c.move_to_end(path); return self._h5c[path]
        if len(self._h5c) >= self.MAX_OPEN:
            _, old = self._h5c.popitem(last=False)
            try: old.close()
            except: pass
        f = h5py.File(path, 'r'); self._h5c[path] = f; return f

    def __getitem__(self, idx):
        info  = self.clips[idx]
        sid, start = info['sid'], info['start']
        jitter = np.random.randint(-16, 17)
        start  = int(np.clip(start + jitter, 0, self.max_start))
        path   = str(self.cache_dir / f'{sid}.h5')
        f      = self._get_h5(path)
        end    = start + self.clip_len + 1
        frames = np.ascontiguousarray(f['frames'][:, start:end, :, :].astype('float32'))
        ppg    = np.ascontiguousarray(f['ppg'][start:start+self.clip_len].astype('float32'))
        ppg_n  = (ppg - ppg.mean()) / (ppg.std() + 1e-8)
        return {'frames': torch.from_numpy(frames), 'ppg': torch.from_numpy(ppg_n), 'gt_hr': -1.0}


# ── MCD temporal-pair dataset ─────────────────────────────────────────────────
def _bandpass(signal, fs=30.0, lo=0.6, hi=4.0):
    nyq = fs / 2.0
    b, a = butter(4, [lo/nyq, hi/nyq], btype='bandpass')
    return filtfilt(b, a, signal.astype(np.float64)).astype(np.float32)


class MCDTemporalPairDataset(Dataset):
    """
    Returns pairs of consecutive overlapping 160-frame windows from MCD frontal cache.
    Min recording length: CLIP_LEN + WINDOW_STRIDE + 1 = 191 frames.
    Each item: (frames_w1 [3,161,72,72], frames_w2 [3,161,72,72],
                ppg_w1 [160], ppg_w2 [160], gt_hr float)
    """
    MAX_OPEN = 32
    MIN_FRAMES = CLIP_LEN + WINDOW_STRIDE + 1  # 191

    def __init__(self, cache_dir, split_json, split='train', steps=None,
                 stride=CLIP_LEN, transform=None, seed=42):
        self.cache_dir = Path(cache_dir)
        self.transform = transform
        self.steps     = steps or ['before', 'after']
        self.clip_len  = CLIP_LEN
        self.w_stride  = WINDOW_STRIDE

        with open(split_json) as f:
            subj_ids = set(json.load(f)[split])

        db = pd.read_csv(self.cache_dir.parent / 'db.csv')
        db = db[(db['patient_id'].isin(subj_ids)) & (db['camera'] == 'FullHDwebcam')]
        db = db[db['step'].isin(self.steps)]
        db = db[(db['pulse'] >= 40) & (db['pulse'] <= 160)]

        self.clips = []
        missing = 0
        for _, row in db.iterrows():
            pid, step = row.patient_id, row.step
            h5_path = self.cache_dir / f'{pid}_FullHDwebcam_{step}.h5'
            if not h5_path.exists():
                missing += 1
                continue
            hr = float(row.pulse)
            try:
                with h5py.File(str(h5_path), 'r') as hf:
                    n_frames = int(hf.attrs.get('n_frames', hf['frames'].shape[1]))
                    ppg_raw  = hf['ppg'][:]
                    if ppg_raw.std() < 5.0 or n_frames < self.MIN_FRAMES:
                        continue
            except Exception:
                continue

            # Sample starting positions for window-pairs with given stride
            for s in range(0, n_frames - self.MIN_FRAMES + 1, stride):
                self.clips.append((h5_path, hr, s))

        rng = __import__('random').Random(seed)
        rng.shuffle(self.clips)
        if missing > 0:
            print(f'MCDTemporalPairDataset [{split}]: {missing} h5 files missing')
        print(f'MCDTemporalPairDataset [{split}]: {len(self.clips)} pairs '
              f'from {len(db) - missing} recordings')

    def __len__(self): return len(self.clips)

    def __getstate__(self):
        s = self.__dict__.copy(); s.pop('_h5c', None); return s

    def _get_h5(self, path_str):
        if not hasattr(self, '_h5c'): self._h5c = OrderedDict()
        if path_str in self._h5c:
            self._h5c.move_to_end(path_str); return self._h5c[path_str]
        if len(self._h5c) >= self.MAX_OPEN:
            _, old = self._h5c.popitem(last=False)
            try: old.close()
            except: pass
        f = h5py.File(path_str, 'r'); self._h5c[path_str] = f; return f

    def __getitem__(self, idx):
        h5_path, hr, start = self.clips[idx]

        hf = self._get_h5(str(h5_path))
        total_needed = self.clip_len + self.w_stride + 1  # 191 frames

        # Load extended clip
        frames_np = hf['frames'][:, start:start+total_needed, :, :].astype(np.float32)
        ppg_raw   = hf['ppg'][start:start+total_needed].astype(np.float32)

        frames_t = torch.from_numpy(frames_np)  # [3, 191, 72, 72]

        if self.transform is not None:
            frames_t = self.transform(frames_t)

        # Split into two overlapping windows
        w1 = frames_t[:, 0:self.clip_len+1, :, :]          # [3, 161, 72, 72]
        w2 = frames_t[:, self.w_stride:self.w_stride+self.clip_len+1, :, :]  # [3, 161, 72, 72]

        # PPG ground truth for each window
        ppg1_raw = ppg_raw[0:self.clip_len]
        ppg2_raw = ppg_raw[self.w_stride:self.w_stride+self.clip_len]
        ppg1_bp  = _bandpass(ppg1_raw)
        ppg2_bp  = _bandpass(ppg2_raw)
        ppg1_n   = (ppg1_bp - ppg1_bp.mean()) / (ppg1_bp.std() + 1e-8)
        ppg2_n   = (ppg2_bp - ppg2_bp.mean()) / (ppg2_bp.std() + 1e-8)

        return {
            'w1':    w1,
            'w2':    w2,
            'ppg1':  torch.from_numpy(ppg1_n),
            'ppg2':  torch.from_numpy(ppg2_n),
            'gt_hr': hr,
        }


# ── Model and losses ──────────────────────────────────────────────────────────
from yacs.config import CfgNode as CN
from neural_methods.model.FactorizePhys.FactorizePhys import FactorizePhys


def make_fp_cfg():
    cfg = CN()
    cfg.CHANNELS     = 3
    cfg.FRAME_NUM    = 160
    cfg.MD_FSAM      = True
    cfg.MD_TYPE      = 'NMF'
    cfg.MD_R         = 1
    cfg.MD_S         = 1
    cfg.MD_STEPS     = 4
    cfg.MD_RESIDUAL  = True
    cfg.MD_INFERENCE = True
    return cfg


def build_model(ckpt_path, device):
    model = FactorizePhys(frames=160, md_config=make_fp_cfg(), device=device, in_channels=3)
    raw   = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
    sd    = OrderedDict((k.replace('module.', ''), v) for k, v in raw.items())
    model.load_state_dict(sd, strict=False)
    return model.to(device)


def neg_pearson_per_clip(pred, label, eps=1e-8):
    pc  = pred  - pred.mean(dim=1, keepdim=True)
    lc  = label - label.mean(dim=1, keepdim=True)
    num = (pc * lc).sum(dim=1)
    den = torch.sqrt((pc**2).sum(dim=1) * (lc**2).sum(dim=1) + eps)
    return (1 - num / den).clamp(max=LOSS_CAP)


def freq_loss(pred, label, fps=30.0, lo=0.6, hi=4.0):
    pred, label = pred.float(), label.float()
    freqs = torch.fft.rfftfreq(pred.shape[-1], d=1.0/fps, device=pred.device)
    mask  = (freqs >= lo) & (freqs <= hi)
    pf    = torch.abs(torch.fft.rfft(pred,  dim=-1))[:, mask]
    lf    = torch.abs(torch.fft.rfft(label, dim=-1))[:, mask]
    return F.mse_loss(pf, lf)


def tc_loss(pred1, pred2, overlap=OVERLAP, eps=1e-8):
    """
    Temporal consistency loss: NegPearson on the overlap region.
    pred1: [B, CLIP_LEN] — predictions for window 1
    pred2: [B, CLIP_LEN] — predictions for window 2
    Overlap region: pred1[:, WINDOW_STRIDE:] vs pred2[:, :OVERLAP]
    """
    p1 = pred1[:, WINDOW_STRIDE:]   # [B, OVERLAP]
    p2 = pred2[:, :overlap]         # [B, OVERLAP]
    p1c = p1 - p1.mean(dim=1, keepdim=True)
    p2c = p2 - p2.mean(dim=1, keepdim=True)
    num = (p1c * p2c).sum(dim=1)
    den = torch.sqrt((p1c**2).sum(dim=1) * (p2c**2).sum(dim=1) + eps)
    return (1 - num / den).mean()


def soft_hr_loss(pred, gt_hr_bpm, fps=30.0, T=50.0):
    """Differentiable HR MAE via soft-argmax on FFT peak. Normalized to ~[0,1] scale."""
    freqs_bpm = torch.fft.rfftfreq(pred.shape[-1], d=1.0/fps).to(pred.device) * 60.0
    mask = (freqs_bpm >= 40.0) & (freqs_bpm <= 200.0)
    fft_mag = torch.abs(torch.fft.rfft(pred.float(), dim=-1))[:, mask]
    weights = F.softmax(fft_mag * T, dim=-1)
    pred_hr = (weights * freqs_bpm[mask]).sum(dim=-1)
    gt = gt_hr_bpm.to(pred.device).float()
    valid = (gt >= 40.0) & (gt <= 200.0)
    if not valid.any():
        return pred.sum() * 0.0
    return F.l1_loss(pred_hr[valid], gt[valid]) / 60.0  # normalize to ~[0,1]


def snr_loss(pred, fps=30.0):
    """Maximize fraction of prediction power in physiological HR band (0.67–3.33 Hz)."""
    freqs = torch.fft.rfftfreq(pred.shape[-1], d=1.0/fps).to(pred.device)
    fft_pow = torch.abs(torch.fft.rfft(pred.float(), dim=-1)) ** 2
    hr_mask = (freqs >= 0.67) & (freqs <= 3.33)
    ratio = fft_pow[:, hr_mask].sum(-1) / (fft_pow.sum(-1) + 1e-8)
    return -ratio.mean()


def composite_score(ubfc, mcd, r10):
    return 0.3*(ubfc/1.23) + 0.3*(mcd/12.36) + 0.4*(r10/13.89)


def extract_hr_fft(ppg, fps=30.0, lo=0.67, hi=3.33):
    freqs = np.fft.rfftfreq(len(ppg), d=1.0/fps)
    fft   = np.abs(np.fft.rfft(ppg))
    mask  = (freqs >= lo) & (freqs <= hi)
    if not mask.any(): return float('nan'), 0.0
    peak_idx = np.argmax(fft[mask])
    peak_pow = fft[mask][peak_idx]**2
    other    = np.delete(fft[mask], peak_idx)
    bg       = (other**2).mean() if len(other) > 0 else 1e-8
    snr      = peak_pow / (bg + 1e-8)
    hr       = freqs[mask][peak_idx] * 60.0
    return hr, snr


# ── Per-clip MAE eval (composite score) ───────────────────────────────────────
@torch.no_grad()
def eval_cache(model, clips, device, batch=8, gt_mode='scalar'):
    model.eval()
    maes = []
    for i in range(0, len(clips), batch):
        b      = clips[i: i+batch]
        frames = torch.stack([c['frames'] for c in b]).to(device)
        out    = model(frames)
        preds  = out[0].float().cpu().numpy()
        for j, clip in enumerate(b):
            pred_hr, snr = extract_hr_fft(preds[j])
            if snr < SNR_THRESHOLD or not (40 <= pred_hr <= 200):
                continue
            if gt_mode == 'fft':
                gt_hr, _ = extract_hr_fft(clip['ppg'].numpy())
            else:
                gt_hr = clip['gt_hr']
            if math.isnan(gt_hr) or not (40 <= gt_hr <= 200):
                continue
            maes.append(abs(pred_hr - gt_hr))
    return float(np.mean(maes)) if maes else float('nan')


# ── Sliding-window HR consistency eval ────────────────────────────────────────
@torch.no_grad()
def eval_consistency(model, h5_dir, device, clip_len=160, stride=30, fps=30.0,
                     max_windows=20):
    """
    For each held-out recording: slide a clip_len window with stride steps,
    run inference on each window, extract HR via FFT, compute std(HR).
    Reads per-window from HDF5 to avoid loading full recordings into RAM.
    max_windows: cap per recording to limit eval time.
    Returns: mean_std, median_std, pct_valid_windows
    """
    model.eval()
    h5_files = sorted(Path(h5_dir).glob('*.h5'))
    all_stds = []
    total_windows, valid_windows = 0, 0

    for h5_path in h5_files:
        try:
            with h5py.File(str(h5_path), 'r') as hf:
                n_frames = int(hf.attrs.get('n_frames', hf['frames'].shape[1]))
                if n_frames < clip_len + 1:
                    continue
                starts = list(range(0, n_frames - clip_len, stride))[:max_windows]
                hr_list = []
                for s in starts:
                    clip_np = hf['frames'][:, s:s+clip_len+1, :, :].astype(np.float32)
                    clip = torch.from_numpy(clip_np).unsqueeze(0).to(device)
                    total_windows += 1
                    out  = model(clip)
                    pred = out[0].float().squeeze(0).cpu().numpy()
                    hr, snr = extract_hr_fft(pred, fps=fps)
                    if snr >= SNR_THRESHOLD and 40 <= hr <= 200:
                        hr_list.append(hr)
                        valid_windows += 1
        except Exception:
            continue

        if len(hr_list) >= 2:
            all_stds.append(float(np.std(hr_list)))

    mean_std   = float(np.mean(all_stds))   if all_stds else float('nan')
    median_std = float(np.median(all_stds)) if all_stds else float('nan')
    pct_valid  = 100.0 * valid_windows / total_windows if total_windows > 0 else 0.0
    return mean_std, median_std, pct_valid


# ── DDP main ──────────────────────────────────────────────────────────────────
def main():
    dist.init_process_group(backend='nccl')
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    device     = torch.device(f'cuda:{rank}')
    torch.manual_seed(SEED + rank)
    torch.backends.cuda.matmul.allow_tf32 = True

    aug = VideoAugmentFast()

    # SCAMPS regularization dataset
    scamps_loader = None
    scamps_iter   = None
    if SCAMPS_CACHE and SCAMPS_CSV:
        df_split = pd.read_csv(str(SCAMPS_CSV), names=['subject_id', 'url', 'split'], skiprows=1)
        scamps_ids = df_split[df_split['split'] == 'Train']['subject_id'].tolist()
        scamps_ds  = SCAMPSCacheDataset(scamps_ids, SCAMPS_CACHE,
                                        clip_len=CLIP_LEN, clips_per_subj=SCAMPS_CPX, seed=SEED)
        scamps_loader = DataLoader(scamps_ds, batch_size=BATCH_SIZE, shuffle=True,
                                   num_workers=2, drop_last=True, persistent_workers=True)
        scamps_iter = iter(scamps_loader)
        if rank == 0:
            print(f'SCAMPS clips      : {len(scamps_ds)}  (injected every {SCAMPS_FREQ} MCD steps)')

    # FP_PURE weight anchor: store frozen reference parameters for L2 regularization
    pure_params = {}
    pure_norm_sq = torch.tensor(0.0, device=device)
    if FP_PURE_CKPT and FP_PURE_CKPT.exists():
        raw = torch.load(str(FP_PURE_CKPT), map_location=device, weights_only=False)
        for k, v in raw.items():
            clean_k = k.replace('module.', '')
            pure_params[clean_k] = v.to(device).detach()
        pure_norm_sq = sum(p.pow(2).sum() for p in pure_params.values())
        if rank == 0:
            n_matched = len(pure_params)
            print(f'L2 weight anchor (FP_PURE): {n_matched} params  λ_kd={LAMBDA_KD}')

    mcd_ds = MCDTemporalPairDataset(
        cache_dir  = MCD_CACHE_DIR,
        split_json = str(MCD_SPLIT_JSON),
        split      = 'train',
        steps      = ['before', 'after'],
        stride     = CLIP_LEN,   # non-overlapping pair starts
        transform  = aug,
        seed       = SEED,
    )

    model = build_model(START_CKPT, device)
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)

    # Scheduler steps
    sampler_tmp   = DistributedSampler(mcd_ds, num_replicas=world_size, rank=rank,
                                       shuffle=True, seed=SEED)
    batches_per_ep = math.ceil(
        len(DataLoader(mcd_ds, batch_size=BATCH_SIZE, sampler=sampler_tmp,
                       drop_last=True)) / GRAD_ACCUM_STEPS
    )
    total_steps = batches_per_ep * EPOCHS

    optimizer = AdamW(model.parameters(), lr=LR_MAX, weight_decay=WEIGHT_DECAY)
    scheduler = OneCycleLR(
        optimizer, max_lr=LR_MAX, total_steps=total_steps,
        pct_start=0.1, div_factor=10,
        final_div_factor=LR_MAX / LR_MIN,
    )

    if rank == 0:
        print(f'MCD train pairs  : {len(mcd_ds)}')
        print(f'Opt steps/epoch  : {batches_per_ep} | total: {total_steps}')
        print(f'Lambda TC        : {LAMBDA_TC}')

    # ── Rank-0 eval ───────────────────────────────────────────────────────────
    if rank == 0:
        ubfc_clips   = torch.load(str(UBFC_CACHE),   weights_only=False)
        rppg10_clips = torch.load(str(RPPG10_CACHE), weights_only=False)
        mcd_clips    = torch.load(str(MCD_EVAL_CACHE), weights_only=False)
        print(f'UBFC clips  : {len(ubfc_clips)}')
        print(f'rPPG10 clips: {len(rppg10_clips)}')
        print(f'MCD clips   : {len(mcd_clips)}')

        ev         = model.module
        s0_ubfc    = eval_cache(ev, ubfc_clips,   device, gt_mode='fft')
        s0_mcd     = eval_cache(ev, mcd_clips,    device, gt_mode='scalar')
        s0_rppg10  = eval_cache(ev, rppg10_clips, device, gt_mode='scalar')
        s0_score   = composite_score(s0_ubfc, s0_mcd, s0_rppg10)
        s0_tc_mean, s0_tc_med, s0_tc_pct = eval_consistency(ev, MCD_HELD_H5_DIR, device)

        print(f'Start — UBFC={s0_ubfc:.3f}  MCD={s0_mcd:.3f}  '
              f'rPPG10={s0_rppg10:.3f}  score={s0_score:.4f}')
        print(f'Start — HR consistency: mean_std={s0_tc_mean:.2f}  '
              f'median_std={s0_tc_med:.2f}  valid={s0_tc_pct:.1f}%')

        best_score = s0_score
        no_improve = 0
        metrics_hist = [{'epoch': 0, 'train_loss': None,
                         'ubfc_mae': round(s0_ubfc, 3),
                         'mcd_mae':  round(s0_mcd,  3),
                         'rppg10_mae': round(s0_rppg10, 3),
                         'score': round(s0_score, 5),
                         'tc_mean_std': round(s0_tc_mean, 3),
                         'tc_median_std': round(s0_tc_med, 3),
                         'no_improve': 0}]
        stop_flag = [False]

    dist.barrier()

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(EPOCHS):
        sampler = DistributedSampler(mcd_ds, num_replicas=world_size, rank=rank,
                                     shuffle=True, seed=SEED + epoch)
        loader  = DataLoader(
            mcd_ds, batch_size=BATCH_SIZE, sampler=sampler,
            num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
            persistent_workers=(NUM_WORKERS > 0),
        )
        sampler.set_epoch(epoch)
        model.train()

        if rank == 0:
            print(f'\nEpoch {epoch+1}/{EPOCHS}')

        total_loss, total_tc, total_kd, n_steps, opt_steps = 0.0, 0.0, 0.0, 0, 0
        pbar = tqdm(loader, desc=f'E{epoch+1}', disable=(rank != 0))

        optimizer.zero_grad(set_to_none=True)

        for step_idx, batch in enumerate(pbar, 1):
            w1     = batch['w1'].to(device, non_blocking=True)   # [B, 3, 161, 72, 72]
            w2     = batch['w2'].to(device, non_blocking=True)
            ppg1   = batch['ppg1'].to(device, non_blocking=True) # [B, 160]
            ppg2   = batch['ppg2'].to(device, non_blocking=True)
            gt_hr  = batch['gt_hr'].float().to(device, non_blocking=True)
            B      = w1.shape[0]

            # Per-clip channel normalization (training only)
            w1 = w1 - w1.mean(dim=(-2, -1), keepdim=True)
            w2 = w2 - w2.mean(dim=(-2, -1), keepdim=True)

            # Single forward pass on both windows (2B samples)
            combined = torch.cat([w1, w2], dim=0)   # [2B, 3, 161, 72, 72]
            out      = model(combined)
            pred_all = out[0].float()               # [2B, 160]
            pred1    = pred_all[:B]
            pred2    = pred_all[B:]

            # Normalize for NegPearson
            pn1 = (pred1 - pred1.mean(1, keepdim=True)) / (pred1.std(1, keepdim=True) + 1e-8)
            pn2 = (pred2 - pred2.mean(1, keepdim=True)) / (pred2.std(1, keepdim=True) + 1e-8)
            ln1 = (ppg1  - ppg1.mean(1,  keepdim=True)) / (ppg1.std(1,  keepdim=True) + 1e-8)
            ln2 = (ppg2  - ppg2.mean(1,  keepdim=True)) / (ppg2.std(1,  keepdim=True) + 1e-8)

            # NegPearson waveform loss
            loss_np = (neg_pearson_per_clip(pn1, ln1).mean() +
                       neg_pearson_per_clip(pn2, ln2).mean()) * 0.5

            # Differentiable HR loss (directly optimizes HR MAE)
            loss_hr = (soft_hr_loss(pred1, gt_hr) + soft_hr_loss(pred2, gt_hr)) * 0.5

            # Temporal consistency loss
            loss_tc = tc_loss(pn1, pn2)

            # HR-band SNR plausibility loss (no labels needed)
            loss_snr = (snr_loss(pred1) + snr_loss(pred2)) * 0.5

            # L2 weight regularization toward FP_PURE parameter space (no forward pass)
            loss_kd = torch.zeros(1, device=device)
            if pure_params:
                diff_sq = sum(
                    (p - pure_params[n]).pow(2).sum()
                    for n, p in model.module.named_parameters()
                    if n in pure_params
                )
                loss_kd = diff_sq / (pure_norm_sq + 1e-8)

            loss = (loss_np
                    + LAMBDA_HR  * loss_hr
                    + LAMBDA_TC  * loss_tc
                    + LAMBDA_SNR * loss_snr
                    + LAMBDA_KD  * loss_kd) / GRAD_ACCUM_STEPS

            loss.backward()
            total_loss += loss_np.item()
            total_tc   += loss_tc.item()
            total_kd   += loss_kd.item()

            # SCAMPS regularization: NegPearson on SCAMPS clip every SCAMPS_FREQ steps
            if scamps_iter is not None and step_idx % SCAMPS_FREQ == 0:
                try:
                    sb = next(scamps_iter)
                except StopIteration:
                    scamps_iter = iter(scamps_loader)
                    sb = next(scamps_iter)
                sf = sb['frames'].to(device, non_blocking=True)
                sp = sb['ppg'].to(device, non_blocking=True)
                sf = sf - sf.mean(dim=(-2, -1), keepdim=True)
                s_out  = model(sf)
                s_pred = s_out[0].float()
                s_pn   = (s_pred - s_pred.mean(1, keepdim=True)) / (s_pred.std(1, keepdim=True) + 1e-8)
                s_ln   = (sp     - sp.mean(1,     keepdim=True)) / (sp.std(1,     keepdim=True) + 1e-8)
                s_loss = neg_pearson_per_clip(s_pn, s_ln).mean() / GRAD_ACCUM_STEPS
                s_loss.backward()
            n_steps    += 1

            if step_idx % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                opt_steps += 1
                if rank == 0:
                    pbar.set_postfix({
                        'np':  f'{total_loss/n_steps:.3f}',
                        'tc':  f'{total_tc/n_steps:.3f}',
                        'kd':  f'{total_kd/n_steps:.3f}',
                    })

        # Flush remaining gradients
        if n_steps % GRAD_ACCUM_STEPS:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step(); scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        avg_loss = total_loss / max(n_steps, 1)
        avg_tc   = total_tc   / max(n_steps, 1)
        avg_kd   = total_kd   / max(n_steps, 1)
        loss_t   = torch.tensor(avg_loss, device=device)
        dist.all_reduce(loss_t, op=dist.ReduceOp.AVG)
        avg_loss = loss_t.item()
        dist.barrier()

        # ── Per-epoch eval (rank 0) ───────────────────────────────────────────
        if rank == 0:
            model.eval()
            ev        = model.module
            ubfc_mae  = eval_cache(ev, ubfc_clips,   device, gt_mode='fft')
            mcd_mae   = eval_cache(ev, mcd_clips,    device, gt_mode='scalar')
            r10_mae   = eval_cache(ev, rppg10_clips, device, gt_mode='scalar')
            score     = composite_score(ubfc_mae, mcd_mae, r10_mae)
            tc_mean, tc_med, tc_pct = eval_consistency(ev, MCD_HELD_H5_DIR, device)

            epoch_ckpt = EPOCH_CKPT_DIR / f'epoch_{epoch+1:02d}.pth'
            torch.save(model.module.state_dict(), str(epoch_ckpt))
            torch.save(model.module.state_dict(), str(LAST_CKPT))

            mark = ''
            if score < best_score:
                best_score = score
                no_improve = 0
                torch.save(model.module.state_dict(), str(BEST_CKPT))
                mark = '  ★'
            else:
                no_improve += 1

            row = {'epoch': epoch+1, 'train_np': round(avg_loss, 5),
                   'train_tc': round(avg_tc, 5), 'train_kd': round(avg_kd, 5),
                   'ubfc_mae': round(ubfc_mae, 3), 'mcd_mae': round(mcd_mae, 3),
                   'rppg10_mae': round(r10_mae, 3), 'score': round(score, 5),
                   'tc_mean_std': round(tc_mean, 3), 'tc_median_std': round(tc_med, 3),
                   'tc_pct_valid': round(tc_pct, 1), 'no_improve': no_improve}
            metrics_hist.append(row)
            json.dump(metrics_hist, open(str(METRICS_JSON), 'w'), indent=2)

            print(f'Epoch {epoch+1:2d} | np={avg_loss:.4f}  tc={avg_tc:.4f}  kd={avg_kd:.4f} | '
                  f'UBFC={ubfc_mae:.3f}  MCD={mcd_mae:.3f}  rPPG10={r10_mae:.3f} | '
                  f'score={score:.4f}{mark}')
            print(f'         HR consistency: mean_std={tc_mean:.2f}  '
                  f'median_std={tc_med:.2f}  valid={tc_pct:.1f}%'
                  + (f'  [patience {no_improve}/{EARLY_STOP_PATIENCE}]' if no_improve else ''))

            if CLEARML_TASK_ID:
                try:
                    from clearml import Task
                    t  = Task.get_task(task_id=CLEARML_TASK_ID)
                    lg = t.get_logger()
                    lg.report_scalar('Loss', 'train',   value=avg_loss,  iteration=epoch+1)
                    lg.report_scalar('Loss', 'tc',      value=avg_tc,    iteration=epoch+1)
                    lg.report_scalar('MAE',  'UBFC',    value=ubfc_mae,  iteration=epoch+1)
                    lg.report_scalar('MAE',  'MCD',     value=mcd_mae,   iteration=epoch+1)
                    lg.report_scalar('MAE',  'rPPG10',  value=r10_mae,   iteration=epoch+1)
                    lg.report_scalar('Score','composite',value=score,     iteration=epoch+1)
                    lg.report_scalar('TC',   'mean_std', value=tc_mean,  iteration=epoch+1)
                except Exception:
                    pass

            if no_improve >= EARLY_STOP_PATIENCE:
                print(f'Early stopping: {EARLY_STOP_PATIENCE} epochs without improvement.')
                stop_flag[0] = True

        stop_t = torch.tensor(1 if (rank == 0 and stop_flag[0]) else 0, device=device)
        dist.all_reduce(stop_t, op=dist.ReduceOp.MAX)
        dist.barrier()
        if stop_t.item():
            if rank == 0: print('All ranks stopping.')
            break

    dist.destroy_process_group()


if __name__ == '__main__':
    main()
