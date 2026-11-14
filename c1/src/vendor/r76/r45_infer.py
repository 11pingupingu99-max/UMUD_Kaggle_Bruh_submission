"""R45T inference stages (resumable, per-image JSONL cache) and OSF / reproduction reports.
SPDX-License-Identifier: GPL-3.0-or-later
"""
from pathlib import Path
import json
import os
import time
import numpy as np
import pandas as pd
from r45_core import read_gray, letterbox_image, restore_map, clean_json, sha256
from r45_models import predict_r22
from r45_decode import decode_image


def load_manifest(csv_path, project_root, check_sha=True):
    m = pd.read_csv(csv_path)
    m['path'] = [str(Path(project_root) / p) for p in m.relative_path]
    missing = [p for p in m.path if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError(f'{len(missing)} missing images, e.g. {missing[:3]}')
    if check_sha:
        bad = [r.image_id for r in m.itertuples() if sha256(r.path) != r.image_sha256]
        if bad:
            raise ValueError(f'{len(bad)} images differ from the manifest, e.g. {bad[:5]}')
    if m.image_id.duplicated().any():
        raise ValueError('Duplicate image ids')
    return m


def _read_cache(path, sig):
    done = {}
    if Path(path).exists():
        for line in Path(path).read_text().splitlines():
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get('_sig') == sig:
                done[rec['image_id']] = rec
    return done


def run_stage(name, manifest, out_dir, r22_models, apo_sources, signature, side=768, device='cpu',
              tta=True, with_streamlines=True, log=print):
    """apo_sources: {name: f(path, gray, square) -> native probability}. Resumable per image."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    sig = json.dumps(signature, sort_keys=True, default=str)
    cache = out_dir / f'{name}.jsonl'
    done = _read_cache(cache, sig)
    todo = [r for r in manifest.itertuples(index=False) if r.image_id not in done]
    log(f'{name}: {len(done)} cached, {len(todo)} to compute')
    t0 = time.time()
    if cache.exists() and cache.stat().st_size and not cache.read_bytes().endswith(b'\n'):
        with open(cache, 'a') as fh:
            fh.write('\n')  # a previous session died mid-line
    with open(cache, 'a') as fh:
        for k, r in enumerate(todo):
            t1 = time.time()
            gray = read_gray(r.path)
            square, _ = letterbox_image(gray, side)
            p, c, s = predict_r22(r22_models, square, device, tta)
            p, c, s = (restore_map(z, gray.shape) for z in (p, c, s))
            probs = {key: f(r.path, gray, square) for key, f in apo_sources.items()}
            try:
                feats = decode_image(p, c, s, probs, r.px_per_mm, with_streamlines)
                status = 'OK'
            except Exception as e:  # noqa: BLE001
                feats = {}; status = f'ERROR:{type(e).__name__}:{e}'[:300]
            rec = {'image_id': r.image_id, '_sig': sig, 'h': int(gray.shape[0]), 'w': int(gray.shape[1]),
                   'px_per_mm': float(r.px_per_mm), 'decode_status': status, **feats, 'seconds': time.time() - t1}
            fh.write(json.dumps(clean_json(rec)) + '\n'); fh.flush()
            done[r.image_id] = rec
            if (k + 1) % 20 == 0 or k + 1 == len(todo):
                el = time.time() - t0
                log(f'{name}: {k + 1}/{len(todo)}  {el / 60:.1f} min, ~{el / (k + 1) * (len(todo) - k - 1) / 60:.1f} min left')
    frame = pd.DataFrame([done[i] for i in manifest.image_id if i in done]).drop(columns=['_sig'])
    frame.to_csv(out_dir / f'{name}.csv', index=False)
    return frame


def osf_truth(path):
    d = pd.read_csv(path)
    out = pd.DataFrame({'image_id': d.ImageID})
    for t in ('MT', 'FL', 'PA'):
        out[f'gt_{t.lower()}'] = d[[f'R{i}_{t}' for i in range(1, 8)]].mean(1)
    return out


def target_of(col):
    c = col.lower()
    if c.startswith('gt_') or c.endswith(('_n', '_w', '_frac', '_resid_px', '_deg')) or '_r_' in c or c.endswith('seconds'):
        return None
    if '_or_r_' in c or '_or_w_' in c or 'or_theta' in c:
        return None
    if '_pa_' in c or c.startswith('pa_') or c.endswith('_pa_chord'):
        return 'pa'
    if ('_fl_' in c or '_st_chord_mm' in c or '_st_arc_mm' in c) and 'iqr' not in c:
        return 'fl'
    if '_mt_' in c and c.endswith('_mm'):
        return 'mt'
    return None


def osf_table(frame, truth):
    x = frame.merge(truth, on='image_id', how='inner')
    min_n = 5 if len(x) >= 10 else 2
    rows = []
    for col in frame.columns:
        t = target_of(col)
        if t is None:
            continue
        v = pd.to_numeric(x[col], errors='coerce'); g = x[f'gt_{t}']
        ok = np.isfinite(v) & np.isfinite(g)
        if ok.sum() < min_n:
            continue
        e = (v - g)[ok]
        rows.append(dict(column=col, target=t, n=int(ok.sum()), mae=float(e.abs().mean()), bias_median=float(e.median()),
                         mae_debiased=float((e - e.median()).abs().mean()),
                         spearman=float(pd.Series(v[ok]).rank().corr(pd.Series(g[ok]).rank()))))
    cols = ['column', 'target', 'n', 'mae', 'bias_median', 'mae_debiased', 'spearman']
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows)[cols].sort_values(['target', 'mae_debiased']).reset_index(drop=True)


def repro_table(frame, saved, prefix='r12b_L_'):
    keys = {'pa_r17_median': .05, 'pa_local_support_baseline': .05, 'pa_local_median': .05, 'fl_mm': .5, 'mt_mm': .03}
    x = frame.merge(saved, on='image_id', how='inner', suffixes=('', '_saved'))
    rows = []
    for k, tol in keys.items():
        a = pd.to_numeric(x.get(prefix + k), errors='coerce'); b = pd.to_numeric(x.get(k), errors='coerce')
        if a is None or b is None:
            continue
        both = np.isfinite(a) & np.isfinite(b)
        d = (a - b)[both].abs()
        rows.append(dict(field=k, compared=int(both.sum()), only_new=int((np.isfinite(a) & ~np.isfinite(b)).sum()),
                         only_saved=int((~np.isfinite(a) & np.isfinite(b)).sum()),
                         median_abs_diff=float(d.median()) if len(d) else None,
                         p90_abs_diff=float(d.quantile(.9)) if len(d) else None,
                         within_tol=float((d <= tol).mean()) if len(d) else None, tol=tol))
    return pd.DataFrame(rows)
