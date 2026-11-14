"""R89T: test-time variants of the EXISTING fascicle models (no training) + an untrained line detector, exported as fascicle
fragments in the R71T format, for the test set and (out-of-fold) for the training set / protocol simulator.
  ms    - multi-scale TTA: mean of native maps at 0.875 / 1 / 1.125 x the model's input side (each with flip TTA);
  clahe - contrast-limited adaptive histogram equalisation of the input (device-contrast normalisation), single scale;
  base  - single scale, control (must reproduce the existing fragments);
  mix   - map-level ensemble of the two model families (mean of the R22 and F71 maps, then one fragment extraction);
  lsd   - OpenCV line segment detector on a smoothed image (independent of every trained model).
SPDX-License-Identifier: GPL-3.0-or-later
"""
import json
import math
import time
from pathlib import Path
import numpy as np
import cv2
import r45_core
import r45_models
import r71_geom
import r71_sim


def sides(base, scales):
    return [int(round(base * s / 32.)) * 32 for s in scales]


def clahe01(a):
    """Native image -> CLAHE on the 0..255 normalised image (same normalisation as letterbox_image)."""
    g = np.clip(r45_core.normalize(a) * 255., 0, 255).astype(np.uint8)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)


def clahe_square(x, hw):
    """CLAHE of the image region of a letterboxed 0..1 square (training cache), padding stays 0: tiles cover the image
    as in clahe01 on the native image."""
    x = np.asarray(x, np.float32); nh, nw, y, xo = r45_core.letterbox_geometry(hw, x.shape[0])
    z = np.zeros_like(x)
    c8 = np.clip(x[y:y + nh, xo:xo + nw] * 255., 0, 255).astype(np.uint8)
    z[y:y + nh, xo:xo + nw] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(c8).astype(np.float32) / 255.
    return z


def mix(a, b):
    """Map-level ensemble of two model families: mean probability and mean axial vector (as predict_r22 over models)."""
    return tuple((u + v) / 2. for u, v in zip(a, b))


def run_multi(name, rows, out_dir, maps_fn, tags, signature, thresholds=(0.6,), log=print):
    """One pass over the images for several variants. maps_fn(row, gray) -> {tag: (p, c, s)} native maps.
    Resumable per image (JSONL cache with a signature hash, as r71_sim.run_images). Returns {tag: records} in the
    r71_sim.export format."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    sh = r71_sim.sig_hash(signature)
    (out_dir / f'{name}.signature.json').write_text(json.dumps(signature, sort_keys=True, default=str, indent=1))
    cache = out_dir / f'{name}.jsonl'
    done = r71_sim._read_cache(cache, sh)
    todo = [r for r in rows.itertuples(index=False) if r.image_id not in done]
    log(f'{name}: {len(done)} gotowe, {len(todo)} do policzenia')
    if cache.exists() and cache.stat().st_size and not cache.read_bytes().endswith(b'\n'):
        with open(cache, 'a') as fh:
            fh.write('\n')
    t0 = time.time()
    with open(cache, 'a') as fh:
        for k, r in enumerate(todo):
            t1 = time.time(); rec = {'image_id': r.image_id, '_sig': sh}
            try:
                gray = r45_core.read_gray(r.path); h, w = gray.shape; rec.update(h=int(h), w=int(w))
                maps = maps_fn(r, gray)
                rec['frags'] = {t: {f'{thr:.2f}': r71_geom.frag_lines(*maps[t], thr, w) for thr in thresholds} for t in tags}
                rec['status'] = 'OK'
            except Exception as e:  # noqa: BLE001 - one image never stops the run
                rec['status'] = f'ERROR:{type(e).__name__}:{e}'[:300]
            rec['seconds'] = round(time.time() - t1, 3)
            fh.write(json.dumps(r45_core.clean_json(rec)) + '\n'); fh.flush()
            done[r.image_id] = rec
            if (k + 1) % 50 == 0 or k + 1 == len(todo):
                el = time.time() - t0
                log(f'{name}: {k + 1}/{len(todo)}  {el / 60:.1f} min, zostało ok. {el / (k + 1) * (len(todo) - k - 1) / 60:.1f} min')
    recs = [done[i] for i in rows.image_id if i in done]
    return {t: [{**{q: r.get(q) for q in ('image_id', 'h', 'w', 'status', 'seconds')}, 'frags': (r.get('frags') or {}).get(t)}
                for r in recs] for t in tags}


def native_maps(models, gray, side_list, device, square_fn=None):
    """Mean (over sides) of native-resolution maps p, c, s from r45_models.predict_r22 (flip TTA inside)."""
    acc = None
    for side in side_list:
        sq = square_fn(side) if square_fn is not None else r45_core.letterbox_image(gray, side)[0]
        out = r45_models.predict_r22(models, sq, device, True)
        nat = [r45_core.restore_map(z, gray.shape if square_fn is None else square_fn.hw) for z in out]
        acc = nat if acc is None else [a + b for a, b in zip(acc, nat)]
    return tuple(a / len(side_list) for a in acc)


class CacheSquare:
    """Squares at any side from a cached letterboxed square (training cache): resize keeps the letterbox geometry."""
    def __init__(self, x, hw):
        self.x, self.hw = x.astype(np.float32), tuple(hw)

    def __call__(self, side):
        if side == self.x.shape[0]:
            return self.x
        return cv2.resize(self.x, (side, side), interpolation=cv2.INTER_AREA if side < self.x.shape[0] else cv2.INTER_LINEAR)


def lsd_lines(gray_u8, w, sigma=None, max_abs_angle=60.):
    """Line segments (OpenCV LSD) as R71T fragments; angle convention as angle_fit (image coordinates, y down)."""
    sigma = sigma or max(1.0, w / 500.)
    g = cv2.GaussianBlur(gray_u8, (0, 0), sigmaX=sigma)
    det = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    found = det.detect(g)[0]
    out = []
    if found is None:
        return out
    min_len = r71_geom.min_fragment_length(w)
    for x1, y1, x2, y2 in found.reshape(-1, 4).astype(float):
        if x2 < x1:
            x1, y1, x2, y2 = x2, y2, x1, y1
        L = math.hypot(x2 - x1, y2 - y1)
        if L < min_len or x2 - x1 < 1:
            continue
        a = (y2 - y1) / (x2 - x1); ang = math.degrees(math.atan(a))
        if abs(ang) > max_abs_angle:
            continue
        out.append(dict(x0=float(x1), x1=float(x2), a=float(a), b=float(y1 - a * x1), angle_fit=float(ang), length=float(L),
                        residual=0.0, n=int(L), angle_or=float(ang), or_R=1.0, p_mean=1.0, p_q10=1.0))
    return out


def to_native(frags, hw, side):
    """Fragments found on a letterboxed square -> native coordinates."""
    nh, nw, y, x = r45_core.letterbox_geometry(hw, side); r = nw / hw[1]
    out = []
    for f in frags:
        x0 = (f['x0'] - x) / r; x1 = (f['x1'] - x) / r
        yc0 = (f['a'] * f['x0'] + f['b'] - y) / r
        b = yc0 - f['a'] * x0
        out.append(dict(f, x0=x0, x1=x1, b=b, length=f['length'] / r))
    return out
