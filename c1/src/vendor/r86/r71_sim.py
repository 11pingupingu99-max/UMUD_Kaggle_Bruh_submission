"""R71T stages S / E: per-image fascicle maps -> fragments (line fit + orientation-field angle + confidence) at several
thresholds, aponeurosis pairs (R12B and v2), ground-truth fascicle lines (training set). Resumable JSONL cache (one line
per image, signature hash), compact CSV.gz exports (< 10 MB each) with plain line parameters.
SPDX-License-Identifier: GPL-3.0-or-later
"""
from pathlib import Path
import hashlib
import json
import time
import numpy as np
import pandas as pd
from r45_core import read_gray, letterbox_image, restore_map, clean_json
from r71_geom import frag_lines, gt_lines, apo_pair_record, FRAG_FIELDS, GT_FIELDS

THRESHOLDS = (.4, .5, .6)


def sig_hash(signature):
    return hashlib.sha256(json.dumps(signature, sort_keys=True, default=str).encode()).hexdigest()[:24]


def _read_cache(path, sh):
    done = {}
    p = Path(path)
    if p.exists():
        for line in p.read_text().splitlines():
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get('_sig') == sh:
                done[rec['image_id']] = rec
    return done


def run_images(name, rows, out_dir, fasc_fn, apo_fns, signature, side, thresholds=THRESHOLDS, log=print, with_gt=False):
    """rows: DataFrame with image_id, path (+ mask, fold for the training set).
    fasc_fn(row, gray, square) -> (p, c, s) native maps; apo_fns: {cfg: f(row, gray, square) -> native probability}
    (empty dict = no aponeuroses, e.g. stage E reuses the pairs of stage S)."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    sh = sig_hash(signature)
    (out_dir / f'{name}.signature.json').write_text(json.dumps(signature, sort_keys=True, default=str, indent=1))
    cache = out_dir / f'{name}.jsonl'
    done = _read_cache(cache, sh)
    todo = [r for r in rows.itertuples(index=False) if r.image_id not in done]
    log(f'{name}: {len(done)} gotowe, {len(todo)} do policzenia')
    if cache.exists() and cache.stat().st_size and not cache.read_bytes().endswith(b'\n'):
        with open(cache, 'a') as fh:
            fh.write('\n')
    t0 = time.time()
    with open(cache, 'a') as fh:
        for k, r in enumerate(todo):
            t1 = time.time()
            rec = {'image_id': r.image_id, '_sig': sh}
            try:
                gray = read_gray(r.path)
                square, _ = letterbox_image(gray, side)
                h, w = gray.shape
                rec.update(h=int(h), w=int(w))
                p, c, s = fasc_fn(r, gray, square)
                rec['frags'] = {f'{thr:.2f}': frag_lines(p, c, s, thr, w) for thr in thresholds}
                rec['apo'] = {cfg: apo_pair_record(f(r, gray, square)) for cfg, f in apo_fns.items()}
                if with_gt:
                    g, info = gt_lines(r.mask, gray.shape)
                    rec['gt'] = g; rec.update(info)
                rec['status'] = 'OK'
            except Exception as e:  # noqa: BLE001 - one image never stops the run
                rec['status'] = f'ERROR:{type(e).__name__}:{e}'[:300]
            rec['seconds'] = round(time.time() - t1, 3)
            fh.write(json.dumps(clean_json(rec)) + '\n'); fh.flush()
            done[r.image_id] = rec
            if (k + 1) % 50 == 0 or k + 1 == len(todo):
                el = time.time() - t0
                log(f'{name}: {k + 1}/{len(todo)}  {el / 60:.1f} min, zostało ok. {el / (k + 1) * (len(todo) - k - 1) / 60:.1f} min')
    return [done[i] for i in rows.image_id if i in done]


def export(name, recs, out_dir, model_tag, extra_cols=None):
    """Compact exports: {name}_frags_{model}_t{40,50,60}.csv.gz, {name}_apo.csv.gz, {name}_gt.csv.gz, {name}_images.csv."""
    out_dir = Path(out_dir)
    fr, ap, gt, im = [], [], [], []
    for r in recs:
        base = {'image_id': r['image_id']}
        im.append({**base, 'h': r.get('h'), 'w': r.get('w'), 'status': r.get('status'), 'seconds': r.get('seconds'),
                   **{k: r.get(k) for k in ('mask_h', 'mask_w', 'mask_fg')}, **({k: r.get(k) for k in extra_cols} if extra_cols else {})})
        for thr, frs in (r.get('frags') or {}).items():
            for j, f in enumerate(frs):
                fr.append({**base, 'thr': float(thr), 'frag_id': j, **{q: f.get(q) for q in FRAG_FIELDS}})
        for cfg, rec in (r.get('apo') or {}).items():
            if rec is not None:
                ap.append({**base, 'cfg': cfg, **rec})
        for j, g in enumerate(r.get('gt') or []):
            gt.append({**base, 'gt_id': j, **{q: g.get(q) for q in GT_FIELDS}})
    files = {}
    def save(df, fname):
        if len(df):
            p = out_dir / fname
            df.to_csv(p, index=False, float_format='%.7g', compression='gzip')
            files[fname] = round(p.stat().st_size / 1e6, 2)
    frd = pd.DataFrame(fr)
    if len(frd):
        for thr, g in frd.groupby('thr'):                    # jeden plik na prog (kazdy < 10 MB)
            save(g, f'{name}_frags_{model_tag}_t{int(round(thr * 100))}.csv.gz')
    save(pd.DataFrame(ap), f'{name}_apo.csv.gz')
    save(pd.DataFrame(gt), f'{name}_gt.csv.gz')
    pd.DataFrame(im).to_csv(out_dir / f'{name}_images.csv', index=False); files[f'{name}_images.csv'] = None
    return files
