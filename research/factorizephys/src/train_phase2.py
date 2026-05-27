"""
Phase 2 Run 2 — MCD-rPPG Real-World Fine-Tuning.
Launch: /home/dex/rppg_venv/bin/torchrun --nproc_per_node=2 src/train_phase2.py
Config: checkpoints/phase2/config.json

Run 2 fixes vs Run 1:
  - Frontal-only MCD cache (YOLO5Face crops, same distribution as eval)
  - Soft SCAMPS curriculum: linear phaseout over curriculum_end_epoch epochs
  - Gradient accumulation ×4 (effective batch 64)
  - Per-clip channel normalization before loss
  - Per-clip NegPearson loss cap (clips with loss > loss_cap are clamped)
  - SNR filter in eval (excludes clips with no dominant FFT peak)
  - ROI subregion augmentation in VideoAugmentFast
  - lr_max=3e-5 (down from 1e-4)
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
from torch.utils.data import Dataset, DataLoader, DistributedSampler, ConcatDataset, Subset
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
CFG_PATH = PROJECT_ROOT / 'checkpoints' / 'phase2' / 'config.json'
with open(CFG_PATH) as f:
    CFG = json.load(f)

UBFC_CACHE       = Path(CFG['ubfc_cache'])
RPPG10_CACHE     = Path(CFG['rppg10_cache'])
MCD_HELD_H5_DIR  = Path(CFG['mcd_held_h5_dir'])   # held-out .h5 files dir
MCD_EVAL_CACHE   = Path(CFG['mcd_eval_cache'])     # .pt eval cache (built at startup)
START_CKPT       = Path(CFG['start_ckpt'])
BEST_CKPT        = Path(CFG['best_ckpt'])
LAST_CKPT        = Path(CFG['last_ckpt'])
METRICS_JSON     = Path(CFG['metrics_json'])
MCD_SPLIT_JSON   = Path(CFG['mcd_split_json'])
SCAMPS_CACHE     = Path(CFG['scamps_cache_dir'])
SCAMPS_CSV       = Path(CFG['scamps_split_csv'])

EPOCHS               = CFG['epochs']
CURRICULUM_END_EPOCH = CFG['curriculum_end_epoch']  # SCAMPS fraction → 0 by this epoch
GRAD_ACCUM_STEPS     = CFG['grad_accum_steps']       # 4 → effective batch = 8×2×4 = 64
BATCH_SIZE           = CFG['batch_size']
CLIP_LEN             = CFG['clip_len']
IMG_SIZE             = CFG['img_size']
CLIPS_PER_SUBJ       = CFG['scamps_clips_per_subj']
LR_MAX               = CFG['lr_max']
LR_MIN               = CFG['lr_min']
WEIGHT_DECAY         = CFG['weight_decay']
GRAD_CLIP            = CFG['grad_clip']
LOSS_CAP             = CFG['loss_cap']
SNR_THRESHOLD        = CFG['snr_threshold']
NUM_WORKERS          = CFG['num_workers']
SEED                 = CFG['seed']
EARLY_STOP_PATIENCE  = CFG.get('early_stop_patience', 8)
CLEARML_TASK_ID      = CFG.get('clearml_task_id')

EPOCH_CKPT_DIR = Path(CFG['best_ckpt']).parent / 'epochs'
EPOCH_CKPT_DIR.mkdir(parents=True, exist_ok=True)


# ── Augmentation ──────────────────────────────────────────────────────────────
class VideoAugmentFast:
    """
    Fast tensor-only augmentations. All ops apply a single random parameter
    across all T+1 frames to preserve temporal consistency.
    """
    def __init__(self):
        self.p_noise = 0.40
        self.p_sat   = 0.50
        self.p_gamma = 0.50
        self.p_geom  = 0.35
        self.p_erase = 0.30
        self.p_roi   = 0.25   # ROI subregion crop (cheek/forehead patch)

    def __call__(self, x):
        if np.random.random() < self.p_noise: x = self._noise(x)
        if np.random.random() < self.p_sat:   x = self._sat(x)
        if np.random.random() < self.p_gamma: x = self._gamma(x)
        if np.random.random() < self.p_geom:  x = self._geom(x)
        if np.random.random() < self.p_erase: x = self._erase(x)
        if np.random.random() < self.p_roi:   x = self._roi(x)
        return x

    def _noise(self, x):
        sigma = np.random.uniform(0.003, 0.02)
        return (x + torch.randn_like(x) * sigma).clamp_(0, 1)

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
        rh   = int(H * frac**0.5)
        rw   = int(W * frac**0.5)
        y0   = np.random.randint(0, max(1, H - rh))
        x0   = np.random.randint(0, max(1, W - rw))
        out  = x.clone()
        out[:, :, y0:y0+rh, x0:x0+rw] = 0.0
        return out

    def _roi(self, x):
        # Crop to a cheek or forehead subregion (55–65% of frame) and resize back.
        # Teaches model to extract rPPG from partial face patches.
        C, T, H, W = x.shape
        frac = np.random.uniform(0.55, 0.65)
        rh, rw = int(H * frac), int(W * frac)
        # Pick upper-left (forehead/cheek) or upper-right (other cheek)
        if np.random.random() < 0.5:
            y0, x0 = 0, 0
        else:
            y0, x0 = 0, W - rw
        patch = x[:, :, y0:y0+rh, x0:x0+rw]  # [C, T, rh, rw]
        # Resize back to H×W using bilinear
        patch_4d = patch.reshape(C*T, 1, rh, rw)
        resized  = F.interpolate(patch_4d, size=(H, W), mode='bilinear', align_corners=False)
        return resized.reshape(C, T, H, W).clamp_(0, 1)


# ── SCAMPS dataset ────────────────────────────────────────────────────────────
class SCAMPSCacheDataset(Dataset):
    MAX_OPEN = 24

    def __init__(self, subject_ids, cache_dir, clip_len=160, clips_per_subj=4, seed=42):
        self.cache_dir = Path(cache_dir)
        self.clip_len  = clip_len
        self.max_start = 600 - clip_len - 1

        rng = np.random.RandomState(seed)
        self.clips = []
        for sid in subject_ids:
            if not (self.cache_dir / f'{sid}.h5').exists():
                continue
            n      = min(clips_per_subj, self.max_start)
            starts = rng.choice(self.max_start, size=n, replace=False).tolist()
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
        sid   = info['sid']
        start = info['start']
        # Small temporal jitter for augmentation
        jitter = np.random.randint(-16, 17)
        start  = int(np.clip(start + jitter, 0, self.max_start))

        path    = str(self.cache_dir / f'{sid}.h5')
        f       = self._get_h5(path)
        end     = start + self.clip_len + 1
        frames  = np.ascontiguousarray(f['frames'][:, start:end, :, :].astype('float32'))
        ppg_raw = np.ascontiguousarray(f['ppg'][start:start+self.clip_len].astype('float32'))
        ppg_norm = (ppg_raw - ppg_raw.mean()) / (ppg_raw.std() + 1e-8)
        return {'frames': torch.from_numpy(frames), 'ppg': torch.from_numpy(ppg_norm), 'gt_hr': -1.0}


# ── MCD dataset ───────────────────────────────────────────────────────────────
from datasets.mcd_rppg_cached import MCDrPPGCachedDataset


# ── Model & losses ────────────────────────────────────────────────────────────
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
    """Per-clip NegPearson loss, shape [B]. Clamp to LOSS_CAP before averaging."""
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


def composite_score(ubfc, mcd, r10):
    return 0.3*(ubfc/1.23) + 0.3*(mcd/12.36) + 0.4*(r10/13.89)


def _bandpass(signal, fs=30.0, lo=0.6, hi=4.0):
    nyq = fs / 2.0
    b, a = butter(4, [lo/nyq, hi/nyq], btype='bandpass')
    return filtfilt(b, a, signal.astype(np.float64)).astype(np.float32)


# ── Build MCD eval clips from held-out .h5 files ─────────────────────────────
def build_mcd_eval_clips(h5_dir, split_json, clip_len=160):
    """Load held-out MCD .h5 files into eval clip list. Saved to .pt for reuse."""
    h5_dir = Path(h5_dir)
    with open(split_json) as f:
        held_ids = set(json.load(f)['held_out'])

    db = pd.read_csv(Path('/mnt/sata-ssd/rppg_project/rppg_dataset/MCD-rPPG/db.csv'))
    db = db[(db['patient_id'].isin(held_ids)) & (db['camera'] == 'FullHDwebcam')]

    clips = []
    for _, row in db.iterrows():
        pid, step = row.patient_id, row.step
        h5_path = h5_dir / f'{pid}_FullHDwebcam_{step}.h5'
        if not h5_path.exists():
            continue
        try:
            with h5py.File(str(h5_path), 'r') as hf:
                n_frames = int(hf.attrs.get('n_frames', hf['frames'].shape[1]))
                if n_frames < clip_len:
                    continue
                ppg_raw = hf['ppg'][:clip_len].astype(np.float32)
                ppg_bp  = _bandpass(ppg_raw)
                ppg_t   = torch.from_numpy(ppg_bp)
                frames  = hf['frames'][:, :clip_len+1, :, :].astype(np.float32)
                frames_t = torch.from_numpy(frames)
        except Exception:
            continue
        clips.append({'frames': frames_t, 'ppg': ppg_t, 'gt_hr': float(row.pulse)})

    return clips


# ── Evaluation — per-clip MAE with SNR filter ─────────────────────────────────
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

            # SNR filter: skip clips with no dominant HR peak
            if snr < SNR_THRESHOLD:
                continue
            if not (40 <= pred_hr <= 200):
                continue

            if gt_mode == 'fft':
                gt_hr, _ = extract_hr_fft(clip['ppg'].numpy())
            else:
                gt_hr = clip['gt_hr']

            if isinstance(gt_hr, float) and math.isnan(gt_hr):
                continue
            if not (40 <= gt_hr <= 200):
                continue
            maes.append(abs(pred_hr - gt_hr))

    return float(np.mean(maes)) if maes else float('nan')


# ── DDP main ──────────────────────────────────────────────────────────────────
def main():
    dist.init_process_group(backend='nccl')
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    device     = torch.device(f'cuda:{rank}')
    torch.manual_seed(SEED + rank)
    torch.backends.cuda.matmul.allow_tf32 = True

    # ── MCD dataset (frontal-only, augmented) ─────────────────────────────────
    aug = VideoAugmentFast()
    mcd_ds = MCDrPPGCachedDataset(
        split_json=str(MCD_SPLIT_JSON),
        split='train',
        steps=['before', 'after'],
        clip_len=CLIP_LEN,
        stride=CLIP_LEN,
        transform=aug,
        seed=SEED,
    )

    # ── SCAMPS dataset ────────────────────────────────────────────────────────
    df_split = pd.read_csv(str(SCAMPS_CSV), names=['subject_id', 'url', 'split'], skiprows=1)
    scamps_train_ids = df_split[df_split['split'] == 'Train']['subject_id'].tolist()
    scamps_ds = SCAMPSCacheDataset(
        scamps_train_ids, SCAMPS_CACHE,
        clip_len=CLIP_LEN, clips_per_subj=CLIPS_PER_SUBJ, seed=SEED,
    )

    if rank == 0:
        print(f'MCD train clips   : {len(mcd_ds)}')
        print(f'SCAMPS clips      : {len(scamps_ds)}')
        print(f'Soft curriculum   : linear phaseout over {CURRICULUM_END_EPOCH} epochs')
        print(f'Grad accum steps  : {GRAD_ACCUM_STEPS} → eff. batch = '
              f'{BATCH_SIZE * world_size * GRAD_ACCUM_STEPS}')

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(START_CKPT, device)
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)

    # ── Scheduler: total optimizer steps (after accumulation) ─────────────────
    mcd_sampler_tmp  = DistributedSampler(mcd_ds, num_replicas=world_size,
                                          rank=rank, shuffle=True, seed=SEED)
    batches_per_ep   = math.ceil(
        len(DataLoader(mcd_ds, batch_size=BATCH_SIZE, sampler=mcd_sampler_tmp,
                       drop_last=True)) / GRAD_ACCUM_STEPS
    )
    total_steps = batches_per_ep * EPOCHS

    optimizer = AdamW(model.parameters(), lr=LR_MAX, weight_decay=WEIGHT_DECAY)
    scheduler = OneCycleLR(
        optimizer, max_lr=LR_MAX, total_steps=total_steps,
        pct_start=0.1, div_factor=25,
        final_div_factor=LR_MAX / LR_MIN,
    )

    if rank == 0:
        print(f'Optimizer steps/epoch: {batches_per_ep} | total: {total_steps}')

    # ── Rank-0 eval data ──────────────────────────────────────────────────────
    if rank == 0:
        ubfc_clips   = torch.load(str(UBFC_CACHE),   weights_only=False)
        rppg10_clips = torch.load(str(RPPG10_CACHE), weights_only=False)

        # Build MCD eval clips from held-out .h5 files (or load cached .pt)
        if MCD_EVAL_CACHE.exists():
            mcd_clips = torch.load(str(MCD_EVAL_CACHE), weights_only=False)
            print(f'Loaded MCD eval cache: {len(mcd_clips)} clips')
        else:
            print('Building MCD eval clips from held-out h5 files...')
            mcd_clips = build_mcd_eval_clips(MCD_HELD_H5_DIR, str(MCD_SPLIT_JSON), CLIP_LEN)
            MCD_EVAL_CACHE.parent.mkdir(parents=True, exist_ok=True)
            torch.save(mcd_clips, str(MCD_EVAL_CACHE))
            print(f'Built and saved MCD eval cache: {len(mcd_clips)} clips')

        print('Evaluating starting checkpoint...')
        ev         = model.module
        s0_ubfc    = eval_cache(ev, ubfc_clips,   device, gt_mode='fft')
        s0_mcd     = eval_cache(ev, mcd_clips,    device, gt_mode='scalar')
        s0_rppg10  = eval_cache(ev, rppg10_clips, device, gt_mode='scalar')
        s0_score   = composite_score(s0_ubfc, s0_mcd, s0_rppg10)
        print(f'Start — UBFC={s0_ubfc:.3f}  MCD={s0_mcd:.3f}  '
              f'rPPG10={s0_rppg10:.3f}  score={s0_score:.4f}')

        best_score   = s0_score
        no_improve   = 0
        metrics_hist = [{'epoch': 0, 'train_loss': None,
                         'ubfc_mae': round(s0_ubfc, 3),
                         'mcd_mae': round(s0_mcd, 3),
                         'rppg10_mae': round(s0_rppg10, 3),
                         'score': round(s0_score, 5),
                         'no_improve': 0}]
        stop_flag = [False]

    dist.barrier()

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(EPOCHS):
        # Soft curriculum: linear SCAMPS phaseout
        scamps_frac = max(0.0, 1.0 - epoch / CURRICULUM_END_EPOCH)
        n_scamps    = int(scamps_frac * len(scamps_ds))

        if n_scamps > 0:
            rng_idx  = np.random.RandomState(SEED + epoch)
            scamps_idx = rng_idx.choice(len(scamps_ds), size=n_scamps, replace=False).tolist()
            train_ds = ConcatDataset([mcd_ds, Subset(scamps_ds, scamps_idx)])
        else:
            train_ds = mcd_ds

        sampler = DistributedSampler(train_ds, num_replicas=world_size,
                                     rank=rank, shuffle=True, seed=SEED + epoch)
        loader  = DataLoader(
            train_ds, batch_size=BATCH_SIZE, sampler=sampler,
            num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
            persistent_workers=(NUM_WORKERS > 0),
        )
        sampler.set_epoch(epoch)
        model.train()

        if rank == 0:
            print(f'\nEpoch {epoch+1}/{EPOCHS} — '
                  f'MCD {len(mcd_ds)} + SCAMPS {n_scamps} '
                  f'(frac={scamps_frac:.2f}) = {len(train_ds)} clips')

        total_loss, n_steps, opt_steps = 0.0, 0, 0
        pbar = tqdm(loader, desc=f'E{epoch+1}', disable=(rank != 0))

        optimizer.zero_grad(set_to_none=True)

        for step_idx, batch in enumerate(pbar, 1):
            frames = batch['frames'].to(device, non_blocking=True)  # [B, 3, T+1, H, W]
            labels = batch['ppg'].to(device, non_blocking=True)     # [B, T]

            # Per-clip channel normalization: subtract spatial mean per channel per frame
            frames = frames - frames.mean(dim=(-2, -1), keepdim=True)

            out  = model(frames)
            pred = out[0].float()   # [B, T]
            lbl  = labels.float()

            # Normalize for NegPearson
            pn = (pred - pred.mean(1, keepdim=True)) / (pred.std(1, keepdim=True) + 1e-8)
            ln = (lbl  - lbl.mean(1, keepdim=True))  / (lbl.std(1, keepdim=True)  + 1e-8)

            # Per-clip loss cap then mean
            loss_np = neg_pearson_per_clip(pn, ln).mean()
            loss_fl = freq_loss(pn, ln)
            loss    = (loss_np + 0.2 * loss_fl) / GRAD_ACCUM_STEPS

            loss.backward()
            total_loss += loss.item() * GRAD_ACCUM_STEPS
            n_steps    += 1

            if step_idx % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                opt_steps += 1
                if rank == 0:
                    pbar.set_postfix({'loss': f'{total_loss/n_steps:.4f}'})

        # Flush remaining gradients if batch count not divisible by accum steps
        remaining = n_steps % GRAD_ACCUM_STEPS
        if remaining:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        avg_loss = total_loss / max(n_steps, 1)
        loss_t   = torch.tensor(avg_loss, device=device)
        dist.all_reduce(loss_t, op=dist.ReduceOp.AVG)
        avg_loss = loss_t.item()
        dist.barrier()

        # ── Per-epoch eval (rank 0 only) ──────────────────────────────────────
        if rank == 0:
            model.eval()
            ev       = model.module
            ubfc_mae = eval_cache(ev, ubfc_clips,   device, gt_mode='fft')
            mcd_mae  = eval_cache(ev, mcd_clips,    device, gt_mode='scalar')
            r10_mae  = eval_cache(ev, rppg10_clips, device, gt_mode='scalar')
            score    = composite_score(ubfc_mae, mcd_mae, r10_mae)

            # Per-epoch checkpoint
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

            row = {'epoch': epoch+1, 'train_loss': round(avg_loss, 5),
                   'ubfc_mae': round(ubfc_mae, 3), 'mcd_mae': round(mcd_mae, 3),
                   'rppg10_mae': round(r10_mae, 3), 'score': round(score, 5),
                   'no_improve': no_improve}
            metrics_hist.append(row)
            json.dump(metrics_hist, open(str(METRICS_JSON), 'w'), indent=2)

            print(f'Epoch {epoch+1:2d} | loss={avg_loss:.4f} | '
                  f'UBFC={ubfc_mae:.3f}  MCD={mcd_mae:.3f}  rPPG10={r10_mae:.3f} | '
                  f'score={score:.4f}{mark}'
                  + (f'  [patience {no_improve}/{EARLY_STOP_PATIENCE}]' if no_improve else ''))

            if no_improve >= EARLY_STOP_PATIENCE:
                print(f'Early stopping: {EARLY_STOP_PATIENCE} epochs without improvement.')
                stop_flag[0] = True

            if CLEARML_TASK_ID:
                try:
                    from clearml import Task
                    t  = Task.get_task(task_id=CLEARML_TASK_ID)
                    lg = t.get_logger()
                    for name, val in [('train', avg_loss)]:
                        lg.report_scalar('Loss', name, value=val, iteration=epoch+1)
                    for name, val in [('UBFC', ubfc_mae), ('MCD', mcd_mae), ('rPPG10', r10_mae)]:
                        lg.report_scalar('MAE', name, value=val, iteration=epoch+1)
                    lg.report_scalar('Score', 'composite', value=score, iteration=epoch+1)
                except Exception:
                    pass

        # Broadcast early-stop to all ranks
        stop_t = torch.tensor(1 if (rank == 0 and stop_flag[0]) else 0, device=device)
        dist.all_reduce(stop_t, op=dist.ReduceOp.MAX)
        dist.barrier()
        if stop_t.item():
            if rank == 0: print('All ranks stopping.')
            break

    dist.destroy_process_group()


if __name__ == '__main__':
    main()
