"""R45T aponeurosis v2: data cache, augmentation, training with resume, OOF edge metrics.
SPDX-License-Identifier: GPL-3.0-or-later
"""
from pathlib import Path
import gc
import hashlib
import json
import math
import os
import shutil
import time
import numpy as np
import pandas as pd
import cv2
import torch
from torch.nn import functional as F
from r45_core import (read_gray, read_binary, letterbox_image, letterbox_soft, restore_map, atomic_json, sha256)
from r45_models import (APO2_VERSION, new_apo2, load_apo2, predict_apo2, atomic_torch)
from r45_decode import apo_pair_subpixel

APO_CONFIG = dict(side=768, epochs=16, batch=2, accumulation=2, seed=20261002, group_size=10, n_folds=5,
                  encoder_lr=6e-5, decoder_lr=6e-4, weight_decay=1e-4, warmup_epochs=1,
                  rotation_degrees=6., scale_range=(.88, 1.12), shift_frac=.04, edge_weight=3.,
                  dice_weight=.5, pos_weight=1., ckpt_every=2, workers=2)
IMG_EXT = {'.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp'}


def digest(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, default=str).encode()).hexdigest()


def list_pairs(image_dir, mask_dir):
    image_dir, mask_dir = Path(image_dir), Path(mask_dir)
    imgs = {p.stem: p for p in image_dir.iterdir() if p.suffix.lower() in IMG_EXT}
    masks = {p.stem: p for p in mask_dir.iterdir() if p.suffix.lower() in IMG_EXT}
    common = sorted(set(imgs) & set(masks))
    if not common:
        raise FileNotFoundError(f'No image/mask pairs in {image_dir} / {mask_dir}')
    frame = pd.DataFrame({'name': common, 'image': [str(imgs[k]) for k in common], 'mask': [str(masks[k]) for k in common]})
    report = {'images': len(imgs), 'masks': len(masks), 'pairs': len(common),
              'images_without_mask': sorted(set(imgs) - set(masks))[:20], 'masks_without_image': sorted(set(masks) - set(imgs))[:20]}
    return frame, report


def assign_folds(frame, group_size=10, n_folds=5, seed=20261002):
    """Contiguous blocks of sorted file names approximate sequences; whole blocks go to one fold."""
    f = frame.sort_values('name').reset_index(drop=True).copy()
    f['group_id'] = np.arange(len(f)) // group_size
    groups = f.group_id.unique()
    rng = np.random.default_rng(seed); perm = rng.permutation(groups)
    fold_of = {g: i % n_folds for i, g in enumerate(perm)}
    f['fold'] = f.group_id.map(fold_of).astype(int)
    return f


def build_cache(frame, cache_dir, side=768, log_every=100):
    cache_dir = Path(cache_dir); cache_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    t0 = time.time()
    for i, r in enumerate(frame.itertuples(index=False)):
        dest = cache_dir / f'{r.name}.npz'
        meta_path = cache_dir / f'{r.name}.json'
        if dest.exists() and meta_path.exists():
            try:
                rows.append(json.loads(meta_path.read_text())); continue
            except Exception:
                pass
        img = read_gray(r.image); m = read_binary(r.mask)
        mismatch = m.shape != img.shape
        if mismatch:
            m = (cv2.resize(m.astype(np.float32), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR) >= .5).astype(np.uint8)
        square, geom = letterbox_image(img, side)
        soft, valid = letterbox_soft(m, side)
        np.savez_compressed(dest.with_suffix('.tmp.npz'), image=np.rint(square * 255).astype(np.uint8),
                            target=np.rint(soft * 255).astype(np.uint8), valid=valid)
        os.replace(dest.with_suffix('.tmp.npz'), dest)
        meta = dict(name=r.name, h=int(img.shape[0]), w=int(img.shape[1]), fg_frac=float(m.mean()),
                    shape_mismatch=bool(mismatch), dtype=str(img.dtype))
        meta_path.write_text(json.dumps(meta)); rows.append(meta)
        if (i + 1) % log_every == 0:
            print(f'cache {i + 1}/{len(frame)}  {time.time() - t0:.0f}s', flush=True)
    return pd.DataFrame(rows)


def sync_cache(frame, cache_dir, drive_tar, side=768, log=print):
    """Local cache <- one tar on Drive (fast) <- source images (slow, first session only).
    The tar is written once, after the cache is complete."""
    import tarfile
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
    idx = build_cache(frame, cache_dir, side)
    if not drive_tar.exists():
        t0 = time.time(); tmp = drive_tar.with_suffix('.tmp')
        with tarfile.open(tmp, 'w') as tf:
            for p in sorted(cache_dir.iterdir()):
                if p.suffix in ('.npz', '.json') and p.stem in need:
                    tf.add(p, arcname=p.name)
        os.replace(tmp, drive_tar)
        log(f'cache zapisany na Drive: {drive_tar.name} ({drive_tar.stat().st_size / 1e6:.0f} MB, {time.time() - t0:.0f} s)')
    return idx


def _edge_weight(t, valid, k):
    b = (t >= .5).astype(np.uint8)
    grad = cv2.dilate(b, np.ones((3, 3), np.uint8)) - cv2.erode(b, np.ones((3, 3), np.uint8))
    grad = cv2.dilate(grad, np.ones((5, 5), np.uint8))
    return (1. + k * grad.astype(np.float32)) * valid


def augment(x, t, v, rng, cfg):
    """x, t in [0,1] float32, v uint8. Geometry identical for image and soft target."""
    side = x.shape[0]
    if rng.random() < .5:
        x = x[:, ::-1]; t = t[:, ::-1]; v = v[:, ::-1]
    ang = float(rng.uniform(-cfg['rotation_degrees'], cfg['rotation_degrees']))
    sc = float(rng.uniform(*cfg['scale_range']))
    tx, ty = (rng.uniform(-cfg['shift_frac'], cfg['shift_frac'], 2) * side).tolist()
    M = cv2.getRotationMatrix2D(((side - 1) / 2, (side - 1) / 2), ang, sc); M[0, 2] += tx; M[1, 2] += ty
    warp = lambda a, interp: cv2.warpAffine(np.ascontiguousarray(a, dtype=np.float32), M, (side, side), flags=interp,
                                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    x = warp(x, cv2.INTER_LINEAR); t = warp(t, cv2.INTER_LINEAR); v = (warp(v, cv2.INTER_NEAREST) > .5).astype(np.uint8)
    x = np.clip(x, 0, 1) ** float(rng.uniform(.8, 1.25))
    x = np.clip(x * float(rng.uniform(.8, 1.2)) + float(rng.uniform(-.04, .04)), 0, 1)
    if rng.random() < .25:
        x = cv2.GaussianBlur(x, (0, 0), float(rng.uniform(.5, 1.2)))
    if rng.random() < .3:
        x = np.clip(x + rng.normal(0, rng.uniform(.005, .02), x.shape).astype(np.float32), 0, 1)
    x = x * v
    return x, np.clip(t, 0, 1) * v, v


class ApoDataset(torch.utils.data.Dataset):
    def __init__(self, frame, cache_dir, cfg, train=False, seed=0):
        self.frame = frame.reset_index(drop=True); self.cache = Path(cache_dir); self.cfg = cfg
        self.train = train; self.seed = seed; self.epoch = 0

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, i):
        name = self.frame.iloc[i]['name']
        with np.load(self.cache / f'{name}.npz', allow_pickle=False) as z:
            x = z['image'].astype(np.float32) / 255.; t = z['target'].astype(np.float32) / 255.; v = z['valid'].astype(np.uint8)
        if self.train:
            rng = np.random.default_rng([self.seed, self.epoch, i, int(torch.initial_seed() % (2 ** 31))])
            x, t, v = augment(x, t, v, rng, self.cfg)
        e = _edge_weight(t, v.astype(np.float32), self.cfg['edge_weight'])
        to = lambda a: torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)[None])
        return to(x), to(t), to(v), to(e)


def apo_loss(logit, t, v, e, pos_weight, dice_weight=.5):
    logit = logit.float(); t = t.float(); v = v.float(); e = e.float()
    bce = F.binary_cross_entropy_with_logits(logit, t, reduction='none', pos_weight=logit.new_tensor(pos_weight))
    bce = (bce * e).sum() / (e * v).sum().clamp_min(1.)
    p = torch.sigmoid(logit) * v
    dims = (1, 2, 3)
    dice = 1 - ((2 * (p * t).sum(dims) + 1) / (p.sum(dims) + (t * v).sum(dims) + 1)).mean()
    return (1 - dice_weight) * bce + dice_weight * dice, bce.detach(), dice.detach()


def precision_mode(device):
    """bf16 only on GPUs with native bf16 (compute capability >= 8: A100, L4, H100).
    T4/V100 (sm_75/sm_70): is_bf16_supported() can report True through emulation, but cuDNN has no
    bf16 convolution engine there ('FIND was unable to find an engine') -> fp16 + GradScaler."""
    if device != 'cuda':
        return None, False
    if torch.cuda.get_device_capability()[0] >= 8 and torch.cuda.is_bf16_supported():
        return torch.bfloat16, False
    return torch.float16, True


def pick_batch(pretrained, cfg, device):
    """One real forward/backward at the configured batch; fall back to batch 1 if memory is tight."""
    if device != 'cuda':
        return min(cfg['batch'], 2), cfg['accumulation']
    dtype, _ = precision_mode(device)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    model = new_apo2(pretrained).to(device).train()
    s = cfg['side']; b = cfg['batch']
    try:
        x = torch.rand(b, 1, s, s, device=device); t = (torch.rand(b, 1, s, s, device=device) > .9).float()
        v = torch.ones_like(t); e = torch.ones_like(t)
        with torch.amp.autocast('cuda', dtype=dtype):
            out = model(x); loss, _, _ = apo_loss(out, t, v, e, 3.)
        loss.backward(); torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() / torch.cuda.get_device_properties(0).total_memory
        ok = peak < .6
    except torch.cuda.OutOfMemoryError:
        ok = False; peak = 1.
    del model; gc.collect(); torch.cuda.empty_cache()
    if ok:
        return b, cfg['accumulation']
    return 1, cfg['batch'] * cfg['accumulation']


class SessionBudgetReached(Exception):
    pass


def train_fold(frame, cache_dir, out_dir, pretrained, cfg, fold, device='cuda', log=print, deadline=None):
    out = Path(out_dir) / f'fold_{fold}'; out.mkdir(parents=True, exist_ok=True)
    train = frame[frame.fold != fold].reset_index(drop=True); held = frame[frame.fold == fold].reset_index(drop=True)
    if not len(train) or not len(held):
        raise ValueError('Empty fold')
    if set(train.group_id) & set(held.group_id):
        raise ValueError('Group leak across folds')
    science = {k: v for k, v in cfg.items() if not k.startswith('_') and k != 'workers'}
    identity = digest({'version': APO2_VERSION, 'cfg': science, 'fold': fold, 'train': train.name.tolist(),
                       'held': held.name.tolist()})
    done = out / 'DONE.json'
    if done.exists():
        d = json.loads(done.read_text())
        if d['identity'] != identity:
            raise RuntimeError(f'fold {fold}: finished with a different configuration; use a new RUN_NAME')
        log(f'fold {fold}: already complete'); return d
    batch, accumulation = cfg['_batch'], cfg['_accumulation']
    dtype, use_scaler = precision_mode(device)
    torch.manual_seed(cfg['seed'] + fold); np.random.seed(cfg['seed'] + fold)
    model = new_apo2(pretrained).to(device)
    enc = list(model.encoder.parameters()); enc_ids = {id(p) for p in enc}
    dec = [p for p in model.parameters() if id(p) not in enc_ids]
    opt = torch.optim.AdamW([{'params': enc, 'lr': cfg['encoder_lr']}, {'params': dec, 'lr': cfg['decoder_lr']}],
                            weight_decay=cfg['weight_decay'])
    E = cfg['epochs']; wu = cfg['warmup_epochs']
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda e: min(1., (e + 1) / max(wu, 1)) * (.05 + .95 * .5 * (1 + math.cos(math.pi * min(e, E) / E))))
    scaler = torch.amp.GradScaler('cuda', enabled=use_scaler)
    pos = tot = 0.
    probe = ApoDataset(train, cache_dir, cfg, train=False)
    for i in np.linspace(0, len(probe) - 1, min(96, len(probe))).astype(int):
        _, t, v, _ = probe[i]; pos += float(t.sum()); tot += float(v.sum())
    # pos_weight != 1 moves the 0.5 crossing of the predicted probability away from the true edge
    # (optimum p = w*q/(w*q + 1 - q)), which would bias the inner-edge MT. Default: unweighted BCE + Dice.
    pos_weight = float(cfg.get('pos_weight') or np.clip(math.sqrt((tot - pos) / max(pos, 1.)), 1., 10.))
    start = 0; history = []
    last = out / 'last.pt'
    if last.exists():
        st = torch.load(last, map_location='cpu', weights_only=True)
        if st['identity'] != identity:
            raise RuntimeError(f'fold {fold}: checkpoint from a different configuration')
        model.load_state_dict(st['model']); opt.load_state_dict(st['optimizer']); sched.load_state_dict(st['scheduler'])
        if use_scaler and st.get('scaler'):
            scaler.load_state_dict(st['scaler'])
        for vals in opt.state.values():
            for k, val in vals.items():
                if torch.is_tensor(val): vals[k] = val.to(device)
        start = int(st['epoch']); history = st['history']; pos_weight = st['pos_weight']
        log(f'fold {fold}: resumed after epoch {start}')
        del st
    ds = ApoDataset(train, cache_dir, cfg, train=True, seed=cfg['seed'] + 1000 * fold)
    def save_last(epochs_done):
        atomic_torch(last, {'version': APO2_VERSION, 'identity': identity, 'model': model.state_dict(),
                            'optimizer': opt.state_dict(), 'scheduler': sched.state_dict(),
                            'scaler': scaler.state_dict() if use_scaler else None, 'epoch': epochs_done,
                            'history': history, 'pos_weight': pos_weight})
    saved_at = start
    for epoch in range(start, E):
        if deadline is not None and time.time() > deadline:
            if saved_at != epoch:
                save_last(epoch)
            raise SessionBudgetReached(f'fold {fold}: session budget reached after epoch {epoch}; checkpoint saved')
        ds.epoch = epoch
        g = torch.Generator().manual_seed(cfg['seed'] + 1000 * fold + epoch)
        loader = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True, generator=g, drop_last=len(ds) > batch,
                                             num_workers=cfg['workers'] if device == 'cuda' else 0,
                                             pin_memory=device == 'cuda', persistent_workers=False)
        model.train(); opt.zero_grad(set_to_none=True); t0 = time.time(); sums = np.zeros(3); seen = 0; skipped = 0
        n_steps = len(loader)
        for step, (x, t, v, e) in enumerate(loader):
            x, t, v, e = [z.to(device, non_blocking=True) for z in (x, t, v, e)]
            with torch.amp.autocast('cuda' if device == 'cuda' else 'cpu', dtype=dtype or torch.float32, enabled=dtype is not None):
                logit = model(x)
            loss, bce, dice = apo_loss(logit, t, v, e, pos_weight, cfg['dice_weight'])
            if not torch.isfinite(loss):
                skipped += 1; opt.zero_grad(set_to_none=True); continue
            scaler.scale(loss / accumulation).backward()
            if (step + 1) % accumulation == 0 or step + 1 == n_steps:
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            sums += np.array([float(loss.detach()), float(bce), float(dice)]) * len(x); seen += len(x)
            if (step + 1) % 100 == 0:
                log(f'fold {fold} epoch {epoch + 1}/{E} step {step + 1}/{n_steps} loss {sums[0] / seen:.4f}')
        sched.step()
        row = dict(epoch=int(epoch + 1), loss=float(sums[0] / max(seen, 1)), bce=float(sums[1] / max(seen, 1)),
                   dice_loss=float(sums[2] / max(seen, 1)), seconds=float(time.time() - t0), skipped=int(skipped),
                   lr_enc=float(opt.param_groups[0]['lr']), lr_dec=float(opt.param_groups[1]['lr']))
        history.append(row); pd.DataFrame(history).to_csv(out / 'history.csv', index=False)
        log(f'fold {fold} epoch {epoch + 1}/{E}: loss {row["loss"]:.4f} (bce {row["bce"]:.4f}, dice {row["dice_loss"]:.4f}) {row["seconds"]:.0f}s')
        if (epoch + 1) % cfg['ckpt_every'] == 0 and epoch + 1 < E:
            save_last(epoch + 1); saved_at = epoch + 1
    final = out / 'final.pt'
    atomic_torch(final, {'version': APO2_VERSION, 'identity': identity, 'fold': fold, 'epoch': E, 'cfg': cfg,
                         'model': {k: v.detach().half().cpu() if v.is_floating_point() else v.cpu() for k, v in model.state_dict().items()}})
    model.eval()
    oof = evaluate_oof(held, cfg, [model], device)
    oof.to_csv(out / 'oof.csv', index=False)
    d = {'identity': identity, 'fold': fold, 'epochs': E, 'n_train': len(train), 'n_held': len(held),
         'final_sha256': sha256(final), 'oof_summary': summarize_oof(oof)}
    atomic_json(done, d)
    if last.exists():
        last.unlink()
    del model, opt; gc.collect()
    if device == 'cuda': torch.cuda.empty_cache()
    return d


def evaluate_oof(held, cfg, models, device):
    """Native-resolution comparison with the GT mask: inner-edge MT, center MT, deep apo angle, Dice."""
    rows = []
    for r in held.itertuples(index=False):
        img = read_gray(r.image); gt = read_binary(r.mask)
        if gt.shape != img.shape:
            gt = (cv2.resize(gt.astype(np.float32), (img.shape[1], img.shape[0])) >= .5).astype(np.uint8)
        square, _ = letterbox_image(img, cfg['side'])
        prob = restore_map(predict_apo2(models, square, device, tta=True), img.shape)
        pr = prob >= .5
        dice = float((2 * (pr & (gt > 0)).sum() + 1) / (pr.sum() + (gt > 0).sum() + 1))
        a = apo_pair_subpixel(gt.astype(np.float32)); b = apo_pair_subpixel(prob)
        row = dict(name=r.name, fold=int(r.fold), h=img.shape[0], w=img.shape[1], dice=dice, gt_pair=a is not None, pred_pair=b is not None)
        if a is not None and b is not None:
            same = abs(a['m']['upper_mid_y'] - b['m']['upper_mid_y']) < .05 * img.shape[0] and \
                abs(a['m']['lower_mid_y'] - b['m']['lower_mid_y']) < .05 * img.shape[0]
            row.update(same_pair=same, gt_mt_inner_px=a['m']['mt_inner_mean'], pred_mt_inner_px=b['m']['mt_inner_mean'],
                       gt_mt_center_px=a['m']['mt_center_mean'], pred_mt_center_px=b['m']['mt_center_mean'],
                       gt_deep_deg=a['m']['deep_top_deg'], pred_deep_deg=b['m']['deep_top_deg'],
                       gt_sup_deg=a['m']['sup_bottom_deg'], pred_sup_deg=b['m']['sup_bottom_deg'])
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_oof(oof):
    s = {'images': int(len(oof)), 'dice_median': float(oof.dice.median()) if len(oof) else None}
    if 'same_pair' in oof and oof.same_pair.notna().any():
        ok = oof[oof.same_pair == True]  # noqa: E712
        s['same_pair'] = int(len(ok)); s['pair_mismatch'] = int((oof.same_pair == False).sum())  # noqa: E712
        if len(ok):
            rel = (ok.pred_mt_inner_px - ok.gt_mt_inner_px) / ok.gt_mt_inner_px
            s.update(mt_inner_rel_err_median_abs=float(rel.abs().median()), mt_inner_rel_bias_median=float(rel.median()),
                     mt_inner_px_err_median_abs=float((ok.pred_mt_inner_px - ok.gt_mt_inner_px).abs().median()),
                     mt_center_px_err_median_abs=float((ok.pred_mt_center_px - ok.gt_mt_center_px).abs().median()),
                     deep_deg_err_median_abs=float((ok.pred_deep_deg - ok.gt_deep_deg).abs().median()),
                     sup_deg_err_median_abs=float((ok.pred_sup_deg - ok.gt_sup_deg).abs().median()))
    return s
