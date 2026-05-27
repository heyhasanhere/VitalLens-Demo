"""
Phase 1 — SCAMPS Augmented Adaptation training script.
Launch: torchrun --nproc_per_node=2 src/train_phase1.py
Config loaded from: checkpoints/phase1/config.json
"""
import sys, os, json, time, math, warnings
from pathlib import Path
from collections import OrderedDict, defaultdict

import numpy as np
import pandas as pd
import h5py
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
# AMP disabled: FSAM NMF uses 1e-6 eps that underflows to 0 in float16 → NaN
from scipy.signal import butter, filtfilt, find_peaks
from tqdm.auto import tqdm

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path('/mnt/sata-ssd/rppg_project')
FP_ROOT      = PROJECT_ROOT / 'external' / 'FactorizePhys'
sys.path.insert(0, str(FP_ROOT))

# ── Config ────────────────────────────────────────────────────────────────────
CFG_PATH = PROJECT_ROOT / 'checkpoints' / 'phase1' / 'config.json'
with open(CFG_PATH) as f:
    CFG = json.load(f)

CACHE_DIR    = Path(CFG['cache_dir'])
UBFC_CACHE   = Path(CFG['ubfc_cache'])
RPPG10_CACHE = Path(CFG['rppg10_cache'])
MCD_CACHE    = Path(CFG['mcd_cache'])
START_CKPT   = Path(CFG['start_ckpt'])
BEST_CKPT    = Path(CFG['best_ckpt'])
LAST_CKPT    = Path(CFG['last_ckpt'])
METRICS_JSON = Path(CFG['metrics_json'])

EPOCHS         = CFG['epochs']
BATCH_SIZE     = CFG['batch_size']
CLIP_LEN       = CFG['clip_len']
IMG_SIZE       = CFG['img_size']
CLIPS_PER_SUBJ = CFG['clips_per_subj']
LR_MAX         = CFG['lr_max']
LR_MIN         = CFG['lr_min']
WEIGHT_DECAY   = CFG['weight_decay']
GRAD_CLIP      = CFG['grad_clip']
NUM_WORKERS    = CFG['num_workers']
SEED           = CFG['seed']
TRAIN_IDS      = CFG['train_ids']

CLEARML_TASK_ID = CFG.get('clearml_task_id')

# ── Dataset ───────────────────────────────────────────────────────────────────
class SCAMPSCacheDataset(Dataset):
    MAX_OPEN = 24

    def __init__(self, subject_ids, cache_dir, clip_len=160, clips_per_subj=4,
                 augment=False, seed=42, transform=None):
        self.cache_dir    = Path(cache_dir)
        self.clip_len     = clip_len
        self.augment      = augment
        self.transform    = transform
        self.max_start    = 600 - clip_len - 1  # stay in bounds for diff

        rng = np.random.RandomState(seed)
        self.clips = []
        for sid in subject_ids:
            if not (self.cache_dir / f'{sid}.h5').exists():
                continue
            n = min(clips_per_subj, self.max_start)
            if augment:
                starts = rng.choice(self.max_start, size=n, replace=False).tolist()
            else:
                starts = [int(i * self.max_start / max(n - 1, 1)) for i in range(n)]
            for s in starts:
                self.clips.append({'sid': sid, 'start': int(s)})

    def __len__(self):
        return len(self.clips)

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('_h5c', None)
        return state

    def _get_h5(self, path):
        if not hasattr(self, '_h5c'):
            self._h5c = OrderedDict()
        if path in self._h5c:
            self._h5c.move_to_end(path)
            return self._h5c[path]
        if len(self._h5c) >= self.MAX_OPEN:
            _, oldest = self._h5c.popitem(last=False)
            try: oldest.close()
            except: pass
        f = h5py.File(path, 'r')
        self._h5c[path] = f
        return f

    def __getitem__(self, idx):
        info  = self.clips[idx]
        sid   = info['sid']
        start = info['start']

        if self.augment:
            jitter = np.random.randint(-16, 17)
            start  = int(np.clip(start + jitter, 0, self.max_start))

        path    = str(self.cache_dir / f'{sid}.h5')
        f       = self._get_h5(path)
        end     = start + self.clip_len + 1
        frames  = np.ascontiguousarray(f['frames'][:, start:end, :, :].astype('float32'))
        ppg_raw = np.ascontiguousarray(f['ppg'][start:start + self.clip_len].astype('float32'))

        frames_t = torch.from_numpy(frames)
        if self.transform is not None:
            frames_t = self.transform(frames_t)

        ppg_norm = (ppg_raw - ppg_raw.mean()) / (ppg_raw.std() + 1e-8)
        return {'frames': frames_t, 'ppg': torch.from_numpy(ppg_norm)}


# ── Augmentation ──────────────────────────────────────────────────────────────
class VideoAugment:
    def __init__(self):
        self.p_jpeg  = 0.30
        self.p_noise = 0.40
        self.p_sat   = 0.50
        self.p_blur  = 0.25
        self.p_gamma = 0.50
        self.p_geom  = 0.40
        self.p_erase = 0.30

    def __call__(self, x):
        if np.random.random() < self.p_jpeg:  x = self._jpeg(x)
        if np.random.random() < self.p_noise: x = self._noise(x)
        if np.random.random() < self.p_sat:   x = self._sat(x)
        if np.random.random() < self.p_blur:  x = self._blur(x)
        if np.random.random() < self.p_gamma: x = self._gamma(x)
        if np.random.random() < self.p_geom:  x = self._geom(x)
        if np.random.random() < self.p_erase: x = self._erase(x)
        return x

    def _jpeg(self, x):
        q = np.random.randint(35, 75)
        out = x.clone()
        for t in range(x.shape[1]):
            fr = (x[:, t].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            fr = cv2.cvtColor(fr, cv2.COLOR_RGB2BGR)
            _, enc = cv2.imencode('.jpg', fr, [cv2.IMWRITE_JPEG_QUALITY, q])
            fr = cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            out[:, t] = torch.from_numpy(fr.astype(np.float32).transpose(2, 0, 1) / 255.0)
        return out.clamp_(0, 1)

    def _noise(self, x):
        sigma = np.random.uniform(0.005, 0.03)
        scale = np.random.uniform(30, 100)
        return (torch.poisson(x * scale) / scale + torch.randn_like(x) * sigma).clamp_(0, 1)

    def _sat(self, x):
        s = np.random.uniform(0.7, 1.5)
        r, g, b = [np.random.uniform(0.9, 1.1), np.random.uniform(0.95, 1.05),
                   np.random.uniform(0.9, 1.1)]
        scales = torch.tensor([r, g, b], dtype=x.dtype).view(3, 1, 1, 1)
        gray   = 0.299 * x[0:1] + 0.587 * x[1:2] + 0.114 * x[2:3]
        return ((gray + s * (x - gray)) * scales).clamp_(0, 1)

    def _blur(self, x):
        k = np.random.choice([3, 5, 7])
        ang = np.random.uniform(0, 180)
        kern = np.zeros((k, k), np.float32)
        cx = cy = k // 2
        ca, sa = math.cos(math.radians(ang)), math.sin(math.radians(ang))
        for i in range(-(k // 2), k // 2 + 1):
            rx = int(round(cx + i * ca)); ry = int(round(cy + i * sa))
            if 0 <= rx < k and 0 <= ry < k: kern[rx, ry] = 1.0
        if kern.sum() == 0: kern[cx, cy] = 1.0
        kern /= kern.sum()
        out = x.clone()
        for t in range(x.shape[1]):
            fr = (x[:, t].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            out[:, t] = torch.from_numpy(
                cv2.filter2D(fr, -1, kern).astype(np.float32).transpose(2, 0, 1) / 255.0)
        return out.clamp_(0, 1)

    def _gamma(self, x):
        g = np.random.uniform(0.7, 1.4)
        return (x.clamp(1e-6, 1) ** g + np.random.uniform(-0.05, 0.05)).clamp_(0, 1)

    def _geom(self, x):
        _, T, H, W = x.shape
        ang = np.random.uniform(-8, 8)
        M = cv2.getRotationMatrix2D((W / 2, H / 2), ang, 1.0)
        M[0, 2] += np.random.uniform(-4, 4); M[1, 2] += np.random.uniform(-4, 4)
        out = x.clone()
        for t in range(T):
            fr = (x[:, t].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            out[:, t] = torch.from_numpy(
                cv2.warpAffine(fr, M, (W, H), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT_101
                               ).astype(np.float32).transpose(2, 0, 1) / 255.0)
        return out.clamp_(0, 1)

    def _erase(self, x):
        _, T, H, W = x.shape
        frac = np.random.uniform(0.05, 0.20)
        rh, rw = int(H * frac ** 0.5), int(W * frac ** 0.5)
        y0 = np.random.randint(0, max(1, H - rh))
        x0 = np.random.randint(0, max(1, W - rw))
        out = x.clone()
        out[:, :, y0:y0 + rh, x0:x0 + rw] = 0.0
        return out


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


def neg_pearson(pred, label):
    eps    = 1e-8
    pc     = pred  - pred.mean(dim=1, keepdim=True)
    lc     = label - label.mean(dim=1, keepdim=True)
    num    = (pc * lc).sum(dim=1)
    den    = torch.sqrt((pc ** 2).sum(dim=1) * (lc ** 2).sum(dim=1) + eps)
    return (1 - num / den).mean()


def freq_loss(pred, label, fps=30.0, lo=0.6, hi=4.0):
    # cast to float32: cuFFT doesn't support non-power-of-2 sizes in float16
    pred, label = pred.float(), label.float()
    freqs = torch.fft.rfftfreq(pred.shape[-1], d=1.0 / fps, device=pred.device)
    mask  = (freqs >= lo) & (freqs <= hi)
    pf    = torch.abs(torch.fft.rfft(pred,  dim=-1))[:, mask]
    lf    = torch.abs(torch.fft.rfft(label, dim=-1))[:, mask]
    return F.mse_loss(pf, lf)


def extract_hr_fft(ppg, fps=30.0, lo=0.6, hi=4.0):
    freqs = np.fft.rfftfreq(len(ppg), d=1.0 / fps)
    fft   = np.abs(np.fft.rfft(ppg))
    mask  = (freqs >= lo) & (freqs <= hi)
    if not mask.any(): return float('nan')
    return freqs[mask][np.argmax(fft[mask])] * 60.0


def composite_score(ubfc, mcd, r10):
    return 0.3 * (ubfc / 1.23) + 0.3 * (mcd / 12.36) + 0.4 * (r10 / 13.89)


# ── Evaluation ────────────────────────────────────────────────────────────────
@torch.no_grad()
@torch.no_grad()
def eval_cache(model, clips, device, batch=8, gt_mode='fft'):
    model.eval()
    subj_pred = defaultdict(list)
    subj_gt   = {}

    for i in range(0, len(clips), batch):
        b      = clips[i: i + batch]
        frames = torch.stack([c['frames'] for c in b]).to(device)
        out    = model(frames)
        preds  = out[0].float().cpu().numpy()

        for j, clip in enumerate(b):
            hr = extract_hr_fft(preds[j])
            if not (40 <= hr <= 200): continue
            subj = clip['subj']
            subj_pred[subj].append(hr)
            if subj not in subj_gt:
                subj_gt[subj] = (clip['gt_hr'] if gt_mode == 'scalar'
                                 else extract_hr_fft(clip['ppg'].numpy()))

    maes = []
    for subj, preds in subj_pred.items():
        gt = subj_gt.get(subj)
        if gt is None or (isinstance(gt, float) and math.isnan(gt)): continue
        pm = float(np.mean(preds))
        if 40 <= pm <= 200: maes.append(abs(pm - gt))
    return float(np.mean(maes)) if maes else float('nan')


# ── DDP main ──────────────────────────────────────────────────────────────────
def main():
    dist.init_process_group(backend='nccl')
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    device     = torch.device(f'cuda:{rank}')
    torch.manual_seed(SEED + rank)

    # ── Dataset + loader ──────────────────────────────────────────────────────
    train_ds = SCAMPSCacheDataset(
        TRAIN_IDS, CACHE_DIR, clip_len=CLIP_LEN,
        clips_per_subj=CLIPS_PER_SUBJ, augment=True, seed=SEED + rank,
        transform=VideoAugment(),
    )
    sampler = DistributedSampler(train_ds, num_replicas=world_size,
                                 rank=rank, shuffle=True, seed=SEED)
    loader  = DataLoader(
        train_ds, batch_size=BATCH_SIZE, sampler=sampler,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
        persistent_workers=NUM_WORKERS > 0,
    )

    if rank == 0:
        print(f'Train clips: {len(train_ds)} | Steps/epoch: {len(loader)} | '
              f'Batch: {BATCH_SIZE}×{world_size}={BATCH_SIZE*world_size}')

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(START_CKPT, device)
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)

    # ── Optimiser ─────────────────────────────────────────────────────────────
    total_steps = len(loader) * EPOCHS
    optimizer   = AdamW(model.parameters(), lr=LR_MAX, weight_decay=WEIGHT_DECAY)
    scheduler   = OneCycleLR(
        optimizer, max_lr=LR_MAX, total_steps=total_steps,
        pct_start=0.1, div_factor=25,
        final_div_factor=LR_MAX / LR_MIN,
    )
    torch.backends.cuda.matmul.allow_tf32 = True  # TF32 for speed on Ampere

    # ── Rank-0 eval data ──────────────────────────────────────────────────────
    if rank == 0:
        ubfc_clips   = torch.load(str(UBFC_CACHE),   weights_only=False)
        rppg10_clips = torch.load(str(RPPG10_CACHE), weights_only=False)
        mcd_clips    = torch.load(str(MCD_CACHE),     weights_only=False)

        print('Evaluating starting checkpoint...')
        model.eval()
        ev = model.module
        s0_ubfc   = eval_cache(ev, ubfc_clips,   device, gt_mode='fft')
        s0_mcd    = eval_cache(ev, mcd_clips,    device, gt_mode='scalar')
        s0_rppg10 = eval_cache(ev, rppg10_clips, device, gt_mode='scalar')
        s0_score  = composite_score(s0_ubfc, s0_mcd, s0_rppg10)
        print(f'Start — UBFC={s0_ubfc:.2f} MCD={s0_mcd:.2f} rPPG10={s0_rppg10:.2f} '
              f'score={s0_score:.4f}')

        best_score   = float('inf')
        metrics_hist = []

    dist.barrier()

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(EPOCHS):
        sampler.set_epoch(epoch)
        model.train()
        total_loss, n_steps = 0.0, 0

        pbar = tqdm(loader, desc=f'Epoch {epoch+1}/{EPOCHS}', disable=(rank != 0))
        for batch in pbar:
            frames = batch['frames'].to(device, non_blocking=True)
            labels = batch['ppg'].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            out  = model(frames)
            pred = out[0]
            lbl  = labels
            pn   = (pred - pred.mean(1, keepdim=True)) / (pred.std(1, keepdim=True) + 1e-8)
            ln   = (lbl  - lbl.mean(1, keepdim=True))  / (lbl.std(1, keepdim=True)  + 1e-8)
            loss = neg_pearson(pn, ln) + 0.2 * freq_loss(pn, ln)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            n_steps    += 1
            if rank == 0:
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

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

            mark  = ''
            if score < best_score:
                best_score = score
                torch.save(model.module.state_dict(), str(BEST_CKPT))
                mark = '  ★'
            torch.save(model.module.state_dict(), str(LAST_CKPT))

            row = {'epoch': epoch+1, 'train_loss': round(avg_loss, 5),
                   'ubfc_mae': round(ubfc_mae, 3), 'mcd_mae': round(mcd_mae, 3),
                   'rppg10_mae': round(r10_mae, 3), 'score': round(score, 5)}
            metrics_hist.append(row)
            json.dump(metrics_hist, open(str(METRICS_JSON), 'w'), indent=2)

            print(f'Epoch {epoch+1:2d}/{EPOCHS} | loss={avg_loss:.4f} | '
                  f'UBFC={ubfc_mae:.2f} MCD={mcd_mae:.2f} rPPG10={r10_mae:.2f} | '
                  f'score={score:.4f}{mark}')

            # ClearML logging
            if CLEARML_TASK_ID:
                try:
                    from clearml import Task
                    t = Task.get_task(task_id=CLEARML_TASK_ID)
                    lg = t.get_logger()
                    lg.report_scalar('Loss',  'train',     value=avg_loss,  iteration=epoch+1)
                    lg.report_scalar('MAE',   'UBFC',      value=ubfc_mae,  iteration=epoch+1)
                    lg.report_scalar('MAE',   'MCD',       value=mcd_mae,   iteration=epoch+1)
                    lg.report_scalar('MAE',   'rPPG-10',   value=r10_mae,   iteration=epoch+1)
                    lg.report_scalar('Score', 'composite', value=score,     iteration=epoch+1)
                except Exception:
                    pass

        dist.barrier()

    dist.destroy_process_group()


if __name__ == '__main__':
    main()
