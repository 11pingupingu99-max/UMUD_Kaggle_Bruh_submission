"""R71T fascicle model F71: R22 architecture (ConvNeXt-Tiny U-Net, 3 outputs: fascicle logit + axial cos2t, sin2t) trained
at higher resolution on tubes around the annotated fascicle traces, with a skeleton-recall term (continuity -> longer
fragments) and the axial orientation of every annotated line. Same folds as R22 (from R22 OOF.csv), so out-of-fold
fragments of R22 and F71 are compared on the same images. Resumable with a session budget.
SPDX-License-Identifier: GPL-3.0-or-later
"""
from pathlib import Path
import gc
import hashlib
import json
import math
import os
import tarfile
import time
import numpy as np
import pandas as pd
import cv2
import torch
from torch.nn import functional as F
from scipy import ndimage as ndi
from r45_core import read_gray, read_binary, letterbox_image, letterbox_geometry, restore_map, atomic_json, sha256, fit_line
from r45_models import MuscleNet, atomic_torch

F71_VERSION = 'R71T_FASC_v1.0.0'
F71_CONFIG = dict(side=1024, epochs=20, batch=1, accumulation=4, seed=20261025, encoder_lr=3e-5, decoder_lr=3e-4,
                  weight_decay=1e-4, warmup_epochs=1, orientation_weight=.15, recall_weight=.5, pos_weight=3.,
                  tube_threshold=.2, band_px=3, rotation_degrees=8., scale_range=(.88, 1.12), shift_frac=.04,
                  ckpt_every=2, workers=2)
IMG_EXT = {'.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp'}
NO_ANGLE = 65535


def digest(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, default=str).encode()).hexdigest()


def list_pairs(image_dir, mask_dir):
    image_dir, mask_dir = Path(image_dir), Path(mask_dir)
    imgs = {p.stem: p for p in image_dir.iterdir() if p.suffix.lower() in IMG_EXT}
    masks = {p.stem: p for p in mask_dir.iterdir() if p.suffix.lower() in IMG_EXT}
    common = sorted(set(imgs) & set(masks))
    if not common:
        raise FileNotFoundError(f'No image/mask pairs in {image_dir} / {mask_dir}')
    frame = pd.DataFrame({'name': common, 'file_name': [imgs[k].name for k in common],
                          'image': [str(imgs[k]) for k in common], 'mask': [str(masks[k]) for k in common]})
    return frame, {'images': len(imgs), 'masks': len(masks), 'pairs': len(common)}


def folds_from_r22(frame, r22_oof_csv):
    """Same samples and folds as R22 (R22 OOF.csv: file_name -> fold, group_id). Pairs R22 did not use are dropped."""
    oof = pd.read_csv(r22_oof_csv)[['file_name', 'fold', 'group_id']].drop_duplicates('file_name')
    f = frame.merge(oof, on='file_name', how='inner').sort_values('name').reset_index(drop=True)
    return f, {'r22_samples': int(len(oof)), 'matched': int(len(f)), 'dropped_pairs': int(len(frame) - len(f))}


# ----------------------------------------------------------------------------- cache (letterboxed image + targets)

def make_targets(gray_shape, mask, side, tube_thr=.2, band_px=3):
    """Soft trace map at `side` (mask -> image frame -> letterbox, area averaging), and an angle map (degrees in [0,180),
    uint16 x100) on a band of `band_px` around the traces: each pixel takes the axial angle of the nearest trace
    (one straight-line fit per connected trace)."""
    nh, nw, y, x = letterbox_geometry(gray_shape, side)
    m = np.asarray(mask, np.float32)
    soft = np.zeros((side, side), np.float32)
    soft[y:y + nh, x:x + nw] = cv2.resize(m, (nw, nh), interpolation=cv2.INTER_AREA if (m.shape[0] > nh or m.shape[1] > nw) else cv2.INTER_LINEAR)
    line = soft >= tube_thr
    ang = np.full((side, side), NO_ANGLE, np.uint16)
    n_lines = 0
    if line.any():
        lab, n = ndi.label(line, structure=np.ones((3, 3)))
        comp_ang = np.full(n + 1, np.nan)
        for i, sl in enumerate(ndi.find_objects(lab), 1):
            if sl is None:
                continue
            yy, xx = np.nonzero(lab[sl] == i)
            if len(xx) < 4:
                continue
            z = fit_line(np.c_[xx + sl[1].start, yy + sl[0].start])
            if z:
                comp_ang[i] = z['angle'] % 180.; n_lines += 1
        d, (iy, ix) = ndi.distance_transform_edt(~line, return_indices=True)
        near = lab[iy, ix]
        a = comp_ang[near]
        ok = (d <= band_px) & np.isfinite(a)
        ang[ok] = np.rint(a[ok] * 100).astype(np.uint16) % 18000
    valid = np.zeros((side, side), np.uint8); valid[y:y + nh, x:x + nw] = 1
    return np.clip(soft, 0, 1), ang, valid, n_lines


def build_cache(frame, cache_dir, side, cfg, log=print, log_every=100):
    cache_dir = Path(cache_dir); cache_dir.mkdir(parents=True, exist_ok=True)
    rows = []; t0 = time.time(); made = 0
    for i, r in enumerate(frame.itertuples(index=False)):
        dest = cache_dir / f'{r.name}.npz'; meta_path = cache_dir / f'{r.name}.json'
        if dest.exists() and meta_path.exists():
            try:
                rows.append(json.loads(meta_path.read_text())); continue
            except Exception:
                pass
        gray = read_gray(r.image); mask = read_binary(r.mask)
        square, _ = letterbox_image(gray, side)
        soft, ang, valid, n_lines = make_targets(gray.shape, mask, side, cfg['tube_threshold'], cfg['band_px'])
        tmp = dest.with_suffix('.tmp.npz')
        np.savez_compressed(tmp, image=np.rint(square * 255).astype(np.uint8), line=np.rint(soft * 255).astype(np.uint8),
                            ang=ang, valid=valid)
        os.replace(tmp, dest)
        meta = dict(name=r.name, h=int(gray.shape[0]), w=int(gray.shape[1]), n_lines=int(n_lines),
                    mask_mismatch=bool(mask.shape != gray.shape), line_frac=float((soft >= cfg['tube_threshold']).mean()))
        meta_path.write_text(json.dumps(meta)); rows.append(meta); made += 1
        if made % log_every == 0:
            log(f'cache {i + 1}/{len(frame)} ({made} nowych) {time.time() - t0:.0f}s')
    return pd.DataFrame(rows)


def sync_cache(frame, cache_dir, drive_tar, side, cfg, log=print):
    """Local cache <- one tar on Drive (fast) <- training images (slow, first time only). Tar written once, complete."""
    cache_dir = Path(cache_dir); cache_dir.mkdir(parents=True, exist_ok=True); drive_tar = Path(drive_tar)
    need = set(frame.name)
    have = {p.stem for p in cache_dir.glob('*.npz')}
    if need - have and drive_tar.exists():
        t0 = time.time()
        with tarfile.open(drive_tar) as tf:
            try:
                tf.extractall(cache_dir, filter='data')
            except TypeError:
                tf.extractall(cache_dir)
        log(f'cache rozpakowany z Drive: {drive_tar.name} ({time.time() - t0:.0f} s)')
    idx = build_cache(frame, cache_dir, side, cfg, log)
    if not drive_tar.exists():
        t0 = time.time(); tmp = drive_tar.with_suffix('.tmp')
        with tarfile.open(tmp, 'w') as tf:
            for p in sorted(cache_dir.iterdir()):
                if p.suffix in ('.npz', '.json') and p.stem in need:
                    tf.add(p, arcname=p.name)
        os.replace(tmp, drive_tar)
        log(f'cache zapisany na Drive: {drive_tar.name} ({drive_tar.stat().st_size / 1e6:.0f} MB, {time.time() - t0:.0f} s)')
    return idx


def load_sample(cache_dir, name):
    with np.load(Path(cache_dir) / f'{name}.npz', allow_pickle=False) as z:
        x = z['image'].astype(np.float32) / 255.; L = z['line'].astype(np.float32) / 255.
        ang = z['ang']; v = z['valid'].astype(np.uint8)
    return x, L, ang, v


# ----------------------------------------------------------------------------- augmentation (angles transformed with the image)

def transform_angles(theta_deg, A):
    """Axial angles (deg) of directions d=(cos t, sin t) mapped by the 2x2 linear part A of the image warp."""
    t = np.radians(theta_deg)
    dx = A[0, 0] * np.cos(t) + A[0, 1] * np.sin(t); dy = A[1, 0] * np.cos(t) + A[1, 1] * np.sin(t)
    return np.degrees(np.arctan2(dy, dx)) % 180.


def augment(x, L, ang, v, rng, cfg):
    side = x.shape[0]
    has = ang != NO_ANGLE
    theta = np.where(has, ang.astype(np.float32) / 100., 0.).astype(np.float32)
    if rng.random() < .5:
        x = x[:, ::-1]; L = L[:, ::-1]; v = v[:, ::-1]; has = has[:, ::-1]; theta = (180. - theta[:, ::-1]) % 180.
    a = float(rng.uniform(-cfg['rotation_degrees'], cfg['rotation_degrees']))
    sc = float(rng.uniform(*cfg['scale_range']))
    tx, ty = (rng.uniform(-cfg['shift_frac'], cfg['shift_frac'], 2) * side).tolist()
    M = cv2.getRotationMatrix2D(((side - 1) / 2, (side - 1) / 2), a, sc); M[0, 2] += tx; M[1, 2] += ty
    warp = lambda z, interp: cv2.warpAffine(np.ascontiguousarray(z, dtype=np.float32), M, (side, side), flags=interp,
                                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    x = warp(x, cv2.INTER_LINEAR); L = warp(L, cv2.INTER_LINEAR)
    v = (warp(v, cv2.INTER_NEAREST) > .5).astype(np.uint8)
    has = warp(has.astype(np.float32), cv2.INTER_NEAREST) > .5
    theta = transform_angles(warp(theta, cv2.INTER_NEAREST), M[:, :2])
    x = np.clip(x, 0, 1) ** float(rng.uniform(.8, 1.25))
    x = np.clip(x * float(rng.uniform(.8, 1.2)) + float(rng.uniform(-.04, .04)), 0, 1)
    if rng.random() < .25:
        x = cv2.GaussianBlur(x, (0, 0), float(rng.uniform(.5, 1.2)))
    if rng.random() < .3:
        x = np.clip(x + rng.normal(0, rng.uniform(.005, .02), x.shape).astype(np.float32), 0, 1)
    return x * v, np.clip(L, 0, 1) * v, np.where(has & (v > 0), theta, np.nan).astype(np.float32), v


def targets(L, theta, v, cfg):
    """Tube (trace >= threshold, dilated by 1 px) for segmentation and recall; orientation on the band."""
    tube = cv2.dilate((L >= cfg['tube_threshold']).astype(np.uint8), np.ones((3, 3), np.uint8)) * v
    band = np.isfinite(theta).astype(np.float32)
    t2 = np.radians(2 * np.nan_to_num(theta))
    return tube.astype(np.float32), np.cos(t2).astype(np.float32) * band, np.sin(t2).astype(np.float32) * band, band


class FascDataset(torch.utils.data.Dataset):
    def __init__(self, frame, cache_dir, cfg, train=False, seed=0):
        self.frame = frame.reset_index(drop=True); self.cache = Path(cache_dir); self.cfg = cfg
        self.train = train; self.seed = seed; self.epoch = 0

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, i):
        x, L, ang, v = load_sample(self.cache, self.frame.iloc[i]['name'])
        if self.train:
            rng = np.random.default_rng([self.seed, self.epoch, i, int(torch.initial_seed() % (2 ** 31))])
            x, L, theta, v = augment(x, L, ang, v, rng, self.cfg)
        else:
            theta = np.where(ang != NO_ANGLE, ang.astype(np.float32) / 100., np.nan).astype(np.float32)
        tube, c2, s2, band = targets(L, theta, v, self.cfg)
        to = lambda a: torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)[None])
        return to(x), to(tube), to(v), torch.cat([to(c2), to(s2)]), to(band)


def fasc_loss(out, tube, v, cs, band, cfg, pos_weight):
    out = out.float(); tube = tube.float(); v = v.float()
    logit = out[:, :1]
    bce = F.binary_cross_entropy_with_logits(logit, tube, reduction='none', pos_weight=logit.new_tensor(pos_weight))
    bce = (bce * v).sum() / v.sum().clamp_min(1.)
    p = torch.sigmoid(logit) * v
    dims = (1, 2, 3)
    dice = 1 - ((2 * (p * tube).sum(dims) + 1) / (p.sum(dims) + tube.sum(dims) + 1)).mean()
    recall = 1 - ((p * tube).sum(dims) + 1) / (tube.sum(dims) + 1)
    recall = recall.mean()
    o = F.normalize(out[:, 1:3], dim=1, eps=1e-6)
    orient = (((o - cs) ** 2).sum(1, keepdim=True) * band).sum() / band.sum().clamp_min(1.)
    seg = bce + dice + cfg['recall_weight'] * recall
    total = seg + cfg['orientation_weight'] * orient
    return total, dict(bce=float(bce.detach()), dice=float(dice.detach()), recall=float(recall.detach()), orient=float(orient.detach()))


# ----------------------------------------------------------------------------- model, precision, training

def new_f71(pretrained_encoder=None):
    model = MuscleNet(3)
    if pretrained_encoder is not None:
        state = torch.load(pretrained_encoder, map_location='cpu', weights_only=True)
        model.encoder.load_state_dict(state['encoder'], strict=True)
    return model


def load_f71(path, device='cpu'):
    state = torch.load(path, map_location='cpu', weights_only=True)
    if state.get('version') != F71_VERSION:
        raise ValueError(f'Not an F71 checkpoint: {path}')
    model = MuscleNet(3)
    model.load_state_dict({k: v.float() for k, v in state['model'].items()}, strict=True)
    return model.to(device).eval(), {k: state.get(k) for k in ('fold', 'epoch', 'identity', 'cfg')}


def precision_mode(device):
    """bf16 on GPUs with native bf16 (sm >= 80: A100, L4, H100); T4/V100 -> fp16 + GradScaler."""
    if device != 'cuda':
        return None, False
    if torch.cuda.get_device_capability()[0] >= 8 and torch.cuda.is_bf16_supported():
        return torch.bfloat16, False
    return torch.float16, True


class SessionBudgetReached(Exception):
    pass


def train_fold(frame, cache_dir, out_dir, pretrained, cfg, fold, device='cuda', log=print, deadline=None, after_train=None):
    out = Path(out_dir) / f'fold_{fold}'; out.mkdir(parents=True, exist_ok=True)
    train = frame[frame.fold != fold].reset_index(drop=True); held = frame[frame.fold == fold].reset_index(drop=True)
    if not len(train) or not len(held):
        raise ValueError('Empty fold')
    science = {k: v for k, v in cfg.items() if not k.startswith('_') and k != 'workers'}
    identity = digest({'version': F71_VERSION, 'cfg': science, 'fold': fold, 'train': train.name.tolist(), 'held': held.name.tolist()})
    done = out / 'DONE.json'
    if done.exists():
        d = json.loads(done.read_text())
        if d['identity'] != identity:
            raise RuntimeError(f'fold {fold}: ukończony z inną konfiguracją; użyj nowego RUN_NAME')
        log(f'fold {fold}: już gotowy'); return d
    final = out / 'final.pt'
    batch, accumulation = cfg['_batch'], cfg['_accumulation']
    dtype, use_scaler = precision_mode(device)
    torch.manual_seed(cfg['seed'] + fold); np.random.seed(cfg['seed'] + fold)
    model = new_f71(pretrained).to(device)
    enc = list(model.encoder.parameters()); enc_ids = {id(p) for p in enc}
    dec = [p for p in model.parameters() if id(p) not in enc_ids]
    opt = torch.optim.AdamW([{'params': enc, 'lr': cfg['encoder_lr']}, {'params': dec, 'lr': cfg['decoder_lr']}],
                            weight_decay=cfg['weight_decay'])
    E = cfg['epochs']; wu = cfg['warmup_epochs']
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda e: min(1., (e + 1) / max(wu, 1)) * (.05 + .95 * .5 * (1 + math.cos(math.pi * min(e, E) / E))))
    scaler = torch.amp.GradScaler('cuda', enabled=use_scaler)
    start = 0; history = []
    last = out / 'last.pt'
    if final.exists() and not done.exists():
        st = torch.load(final, map_location='cpu', weights_only=True)
        if st.get('identity') == identity:            # trening skończony, przerwane po nim (OOF / eksport)
            model.load_state_dict({k: v.float() for k, v in st['model'].items()}); start = E
            log(f'fold {fold}: model gotowy, dokańczam ocenę OOF')
        del st
    if start < E and last.exists():
        st = torch.load(last, map_location='cpu', weights_only=True)
        if st['identity'] != identity:
            raise RuntimeError(f'fold {fold}: checkpoint z inną konfiguracją')
        model.load_state_dict(st['model']); opt.load_state_dict(st['optimizer']); sched.load_state_dict(st['scheduler'])
        if use_scaler and st.get('scaler'):
            scaler.load_state_dict(st['scaler'])
        for vals in opt.state.values():
            for k, val in vals.items():
                if torch.is_tensor(val):
                    vals[k] = val.to(device)
        start = int(st['epoch']); history = st['history']
        log(f'fold {fold}: wznowiony po epoce {start}')
        del st
    ds = FascDataset(train, cache_dir, cfg, train=True, seed=cfg['seed'] + 1000 * fold)

    def save_last(epochs_done):
        atomic_torch(last, {'version': F71_VERSION, 'identity': identity, 'model': model.state_dict(),
                            'optimizer': opt.state_dict(), 'scheduler': sched.state_dict(),
                            'scaler': scaler.state_dict() if use_scaler else None, 'epoch': epochs_done, 'history': history})
    saved_at = start
    for epoch in range(start, E):
        if deadline is not None and time.time() > deadline:
            if saved_at != epoch:
                save_last(epoch)
            raise SessionBudgetReached(f'fold {fold}: limit czasu sesji po epoce {epoch}; checkpoint zapisany')
        ds.epoch = epoch
        g = torch.Generator().manual_seed(cfg['seed'] + 1000 * fold + epoch)
        loader = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True, generator=g, drop_last=len(ds) > batch,
                                             num_workers=cfg['workers'] if device == 'cuda' else 0,
                                             pin_memory=device == 'cuda', persistent_workers=False)
        model.train(); opt.zero_grad(set_to_none=True); t0 = time.time(); sums = {}; seen = 0; skipped = 0
        n_steps = len(loader)
        for step, (x, tube, v, cs, band) in enumerate(loader):
            x, tube, v, cs, band = [z.to(device, non_blocking=True) for z in (x, tube, v, cs, band)]
            with torch.amp.autocast('cuda' if device == 'cuda' else 'cpu', dtype=dtype or torch.float32, enabled=dtype is not None):
                out_ = model(x)
            loss, parts = fasc_loss(out_, tube, v, cs, band, cfg, cfg['pos_weight'])
            if not torch.isfinite(loss):
                skipped += 1; opt.zero_grad(set_to_none=True); continue
            scaler.scale(loss / accumulation).backward()
            if (step + 1) % accumulation == 0 or step + 1 == n_steps:
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            for k, val in {'loss': float(loss.detach()), **parts}.items():
                sums[k] = sums.get(k, 0.) + val * len(x)
            seen += len(x)
            if (step + 1) % 200 == 0:
                log(f'fold {fold} epoka {epoch + 1}/{E} krok {step + 1}/{n_steps} strata {sums["loss"] / seen:.4f}')
        sched.step()
        row = dict(epoch=int(epoch + 1), **{k: float(val / max(seen, 1)) for k, val in sums.items()},
                   seconds=float(time.time() - t0), skipped=int(skipped), lr_enc=float(opt.param_groups[0]['lr']))
        history.append(row); pd.DataFrame(history).to_csv(out / 'history.csv', index=False)
        log(f'fold {fold} epoka {epoch + 1}/{E}: strata {row["loss"]:.4f} (bce {row.get("bce", 0):.4f}, dice {row.get("dice", 0):.4f}, '
            f'recall {row.get("recall", 0):.4f}, kierunek {row.get("orient", 0):.4f}) {row["seconds"]:.0f}s')
        if (epoch + 1) % cfg['ckpt_every'] == 0 and epoch + 1 < E:
            save_last(epoch + 1); saved_at = epoch + 1
    if start < E or not final.exists():
        atomic_torch(final, {'version': F71_VERSION, 'identity': identity, 'fold': fold, 'epoch': E, 'cfg': science,
                             'model': {k: v.detach().half().cpu() if v.is_floating_point() else v.cpu() for k, v in model.state_dict().items()}})
    model.eval()
    metrics = evaluate_held(held, cache_dir, cfg, [model], device)
    metrics.to_csv(out / 'oof_metrics.csv', index=False)
    extra = after_train(fold, [model], held) if after_train is not None else {}
    d = {'identity': identity, 'fold': fold, 'epochs': E, 'n_train': len(train), 'n_held': len(held), 'final_sha256': sha256(final),
         'oof_summary': {k: float(metrics[k].median()) for k in ('dice', 'recall', 'precision') if k in metrics}, **(extra or {})}
    atomic_json(done, d)
    if last.exists():
        last.unlink()
    del model, opt; gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()
    return d


@torch.inference_mode()
def predict_maps(models, square, device='cpu', tta=True):
    """Same contract as r45_models.predict_r22 (mean probability, mean unit axial vector, flip TTA)."""
    from r45_models import predict_r22
    return predict_r22(models, square, device, tta)


def evaluate_held(held, cache_dir, cfg, models, device):
    """Pixel metrics at training resolution on held-out images: Dice / recall / precision of p>=0.5 against the tube."""
    rows = []
    for r in held.itertuples(index=False):
        x, L, ang, v = load_sample(cache_dir, r.name)
        p, _, _ = predict_maps(models, x, device, True)
        tube = cv2.dilate((L >= cfg['tube_threshold']).astype(np.uint8), np.ones((3, 3), np.uint8)) * v
        pr = (p >= .5) & (v > 0)
        tp = float((pr & (tube > 0)).sum())
        near = cv2.dilate(tube, np.ones((5, 5), np.uint8)) > 0
        rows.append(dict(name=r.name, fold=int(r.fold), dice=(2 * tp + 1) / (pr.sum() + tube.sum() + 1),
                         recall=(tp + 1) / (tube.sum() + 1), precision=(float((pr & near).sum()) + 1) / (pr.sum() + 1)))
    return pd.DataFrame(rows)
