"""UMUD R45T core: IO, letterbox, R17/R20 geometry. SPDX-License-Identifier: GPL-3.0-or-later.

Geometry functions are verbatim copies of R17 (umud_core) and R20 (r20_geometry), so that
decoders computed here are directly comparable with the decoders already used on the
leaderboard (pa_r17_median, pa_local_support_baseline, pa_local_median, fl_mm).
"""
from pathlib import Path
import hashlib
import json
import math
import os
import numpy as np
import cv2
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

# ----------------------------------------------------------------------------- IO

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def atomic_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, default=str, allow_nan=False)); os.replace(tmp, path)


def clean_json(x):
    """Replace non-finite floats by None (JSON has no NaN)."""
    if isinstance(x, dict):
        return {k: clean_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean_json(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return float(x) if np.isfinite(x) else None
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def read_gray(path):
    a = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if a is None:
        try:
            from PIL import Image
            a = np.array(Image.open(path))
        except Exception:
            a = None
    if a is None:
        raise ValueError(f'Unreadable image: {path}')
    if a.ndim == 3:
        a = cv2.cvtColor(a[:, :, :3], cv2.COLOR_BGR2GRAY)
    if a.ndim != 2 or not np.isfinite(a).all():
        raise ValueError(f'Invalid image: {path}')
    return a


def normalize(a):
    if a.dtype == np.uint8:
        return a.astype(np.float32) / 255.
    x = a.astype(np.float32); lo, hi = np.percentile(x, [.5, 99.5])
    return np.clip((x - lo) / max(float(hi - lo), 1e-6), 0, 1)


def read_binary(path):
    a = read_gray(path)
    if a.max() <= 1:
        return (a > 0).astype(np.uint8)
    return (a > float(a.max()) * .5).astype(np.uint8)


def letterbox_geometry(hw, side):
    h, w = hw; r = min(side / h, side / w); nh = max(1, round(h * r)); nw = max(1, round(w * r))
    return nh, nw, (side - nh) // 2, (side - nw) // 2


def letterbox_image(a, side):
    nh, nw, y, x = letterbox_geometry(a.shape, side)
    z = np.zeros((side, side), np.float32)
    z[y:y + nh, x:x + nw] = cv2.resize(normalize(a), (nw, nh),
                                       interpolation=cv2.INTER_AREA if max(a.shape) > side else cv2.INTER_LINEAR)
    return z, (nh, nw, y, x)


def letterbox_soft(mask01, side):
    """Soft (area-averaged) letterboxed mask in [0,1]; keeps sub-pixel edge information."""
    nh, nw, y, x = letterbox_geometry(mask01.shape, side)
    z = np.zeros((side, side), np.float32)
    m = np.asarray(mask01, np.float32)
    z[y:y + nh, x:x + nw] = cv2.resize(m, (nw, nh),
                                       interpolation=cv2.INTER_AREA if max(m.shape) > side else cv2.INTER_LINEAR)
    valid = np.zeros((side, side), np.uint8); valid[y:y + nh, x:x + nw] = 1
    return np.clip(z, 0, 1), valid


def restore_map(p, hw):
    """Letterboxed square map -> native (h, w), bilinear."""
    nh, nw, y, x = letterbox_geometry(hw, p.shape[-1])
    return cv2.resize(np.ascontiguousarray(p[y:y + nh, x:x + nw], dtype=np.float32), (hw[1], hw[0]),
                      interpolation=cv2.INTER_LINEAR)

# ----------------------------------------------------------------------------- R17 geometry (verbatim)

def axial_diff(a, b):
    return np.abs((np.asarray(a) - np.asarray(b) + 90) % 180 - 90)


def fit_line(points):
    p = np.asarray(points, np.float64)
    if len(p) < 3: return None
    c = p.mean(0); _, s, v = np.linalg.svd(p - c, full_matrices=False); d = v[0]
    if d[0] < 0: d = -d
    if abs(d[0]) < 1e-8: return None
    a = float(d[1] / d[0]); b = float(c[1] - a * c[0]); t = (p - c) @ d
    return dict(a=a, b=b, angle=math.degrees(math.atan2(d[1], d[0])),
                length=float(np.ptp(t)), residual=float(np.sqrt(np.mean(((p - c) @ v[1]) ** 2))),
                x0=float(p[:, 0].min()), x1=float(p[:, 0].max()), n=len(p))


def fasc_lines(mask, min_length=15):
    lab, n = ndi.label(skeletonize(np.asarray(mask) > 0), structure=np.ones((3, 3)))
    out = []
    for i, sl in enumerate(ndi.find_objects(lab), 1):
        if sl is None: continue
        y, x = np.nonzero(lab[sl] == i)
        if len(x) < max(6, min_length * .5): continue
        z = fit_line(np.c_[x + sl[1].start, y + sl[0].start])
        if z and z['length'] >= min_length and z['residual'] < max(3, .05 * z['length']): out.append(z)
    return out


def aponeurosis_components(mask):
    """R17 candidate components, with their label id kept (needed for sub-pixel edges)."""
    mask = np.asarray(mask) > 0; h, w = mask.shape
    lab, n = ndi.label(mask, structure=np.ones((3, 3))); components = []
    for i, sl in enumerate(ndi.find_objects(lab), 1):
        if sl is None: continue
        yy, xx = np.nonzero(lab[sl] == i); yy = yy + sl[0].start; xx = xx + sl[1].start
        if len(xx) < max(20, .0002 * h * w) or np.ptp(xx) < .2 * w: continue
        cols = np.unique(xx); center = []; top = []; bottom = []
        for x in cols:
            ys = yy[xx == x]; center.append(np.mean(ys)); top.append(np.min(ys)); bottom.append(np.max(ys))
        center_line = fit_line(np.c_[cols, center])
        if not center_line or abs(center_line['angle']) > 45: continue
        components.append(dict(center=center_line, top=fit_line(np.c_[cols, top]), bottom=fit_line(np.c_[cols, bottom]),
                               width=float(np.ptp(xx)), area=int(len(xx)), mid=float(np.mean(yy)), label=i))
    return lab, components


def aponeurosis_lines(mask, return_labels=False):
    """Exactly the R17 pair selection; optionally also the label image and label ids."""
    mask = np.asarray(mask) > 0; h, w = mask.shape
    lab, components = aponeurosis_components(mask)
    pairs = []
    for i, u in enumerate(components):
        for l in components[i + 1:]:
            if u['mid'] > l['mid']: u0, l0 = l, u
            else: u0, l0 = u, l
            x0 = max(u0['center']['x0'], l0['center']['x0']); x1 = min(u0['center']['x1'], l0['center']['x1'])
            if x1 - x0 < .2 * w: continue
            xs = np.linspace(x0, x1, 31); a = u0['bottom']; b = l0['top']
            gap = (b['a'] * xs + b['b']) - (a['a'] * xs + a['b'])
            if gap.min() <= .025 * h or gap.max() >= .85 * h: continue
            quality = (x1 - x0) / (1 + u0['center']['residual'] + l0['center']['residual'])
            pairs.append((quality, u0, l0, x0, x1))
    if not pairs:
        return (None, lab) if return_labels else None
    pairs.sort(key=lambda z: z[0], reverse=True)
    q, u, l, x0, x1 = pairs[0]
    out = dict(upper=u, lower=l, x0=x0, x1=x1, candidate_pairs=len(pairs), quality=q)
    return (out, lab) if return_labels else out


def measure_geometry(apo_prob, fasc_prob, px_per_mm, apo_thr=.5, fasc_thr=.5):
    if not np.isfinite(px_per_mm) or px_per_mm <= 0: raise ValueError('Invalid px_per_mm')
    if apo_prob.shape != fasc_prob.shape: raise ValueError('Probability maps must share native coordinates')
    h, w = apo_prob.shape; pair = aponeurosis_lines(apo_prob >= apo_thr)
    out = dict(pa_deg=None, fl_mm=None, mt_mm=None, mt_center_mm=None, band_correction_mm=None,
               n_fasc=0, apo_pairs=0, extrapolation_ratio=None, status='NO_APO_PAIR')
    if pair is None: return out
    u, l = pair['upper'], pair['lower']; xs = np.linspace(pair['x0'], pair['x1'], 101)
    def yy(z): return z['a'] * xs + z['b']
    inner_gap = yy(l['top']) - yy(u['bottom']); center_gap = yy(l['center']) - yy(u['center'])
    out.update(mt_mm=float(np.mean(inner_gap) / px_per_mm), mt_center_mm=float(np.mean(center_gap) / px_per_mm),
               band_correction_mm=float(np.mean(center_gap - inner_gap) / px_per_mm), apo_pairs=pair['candidate_pairs'], status='NO_FASCICLE')
    arr = []
    for f in fasc_lines(fasc_prob >= fasc_thr, min_length=max(12, .025 * w)):
        endpoints = []
        for a in (u['bottom'], l['top']):
            den = f['a'] - a['a']
            if abs(den) < 1e-6: break
            x = (a['b'] - f['b']) / den; y = f['a'] * x + f['b']; endpoints.append((x, y))
        if len(endpoints) != 2: continue
        p, q = np.asarray(endpoints)
        if not (-w <= p[0] <= 2 * w and -w <= q[0] <= 2 * w): continue
        xm = (f['x0'] + f['x1']) / 2; ym = f['a'] * xm + f['b']
        if not (u['bottom']['a'] * xm + u['bottom']['b'] <= ym <= l['top']['a'] * xm + l['top']['b']): continue
        pa = float(axial_diff(f['angle'], l['top']['angle']))
        length = float(np.linalg.norm(p - q)); fl = length / px_per_mm
        if not (1 < pa < 85 and 0 < fl < 500): continue
        arr.append((pa, fl, length / max(f['length'], 1)))
    if arr:
        med = np.median(np.asarray(arr), axis=0)
        out.update(pa_deg=float(med[0]), fl_mm=float(med[1]), n_fasc=len(arr), extrapolation_ratio=float(med[2]), status='OK')
    return out

# ----------------------------------------------------------------------------- R20 decoders (verbatim)

R20_CONFIG = {'apo_threshold': .5, 'fasc_threshold': .65,
              'local_radius_width_fraction': .05, 'local_min_radius_px': 12,
              'local_required_span_fraction': .8, 'orientation_sigma_deg': 10.0}
R20_VARIANTS = ('pa_r17_median', 'pa_center_median', 'pa_length_weighted',
                'pa_coherent_weighted', 'pa_local_median')


def weighted_median(values, weights):
    v, w = np.asarray(values, float), np.asarray(weights, float)
    ok = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not ok.any(): return None
    v, w = v[ok], w[ok]
    order = np.argsort(v, kind='stable'); v, w = v[order], w[order]
    return float(v[np.searchsorted(np.cumsum(w), w.sum() / 2, side='left')])


def axial_mean(angles, weights):
    a, w = np.radians(2 * np.asarray(angles)), np.asarray(weights)
    z = np.sum(w * np.exp(1j * a)) / np.sum(w)
    return float(np.degrees(np.angle(z)) / 2), float(abs(z))


def lower_contour(mask, selected):
    lab, _ = ndi.label(mask > 0, structure=np.ones((3, 3)))
    matches = []
    for i, sl in enumerate(ndi.find_objects(lab), 1):
        if sl is None: continue
        yy, xx = np.nonzero(lab[sl] == i)
        if len(xx) != selected['area']: continue
        yy, xx = yy + sl[0].start, xx + sl[1].start
        if abs(float(yy.mean()) - selected['mid']) > 1e-8: continue
        if abs(float(np.ptp(xx)) - selected['width']) > 1e-8: continue
        cols = np.unique(xx)
        top = np.array([yy[xx == x].min() for x in cols], float)
        center = np.array([yy[xx == x].mean() for x in cols], float)
        matches.append((cols.astype(float), top, center))
    if len(matches) != 1:
        raise RuntimeError('Ambiguous lower aponeurosis component identity')
    return matches[0]


def local_reference(fasc, contour, global_qx, width):
    xs, ys, _ = contour
    d = fasc['a'] * xs + fasc['b'] - ys
    roots = list(xs[np.abs(d) < 1e-9])
    for i in np.flatnonzero((d[:-1] * d[1:] < 0) & (np.diff(xs) <= 1.01)):
        roots.append(float(xs[i] - d[i] * (xs[i + 1] - xs[i]) / (d[i + 1] - d[i])))
    if not roots:
        return {'local_status': 'NO_OBSERVED_INTERSECTION', 'local_pa': None}
    qx = min(roots, key=lambda x: abs(x - global_qx))
    radius = max(R20_CONFIG['local_min_radius_px'], R20_CONFIG['local_radius_width_fraction'] * width)
    take = np.abs(xs - qx) <= radius
    lx, ly = xs[take], ys[take]
    fraction = R20_CONFIG['local_required_span_fraction']
    if (len(lx) < 8 or qx - lx.min() < fraction * radius or lx.max() - qx < fraction * radius
            or np.max(np.diff(lx)) > 1.01):
        return {'local_status': 'INSUFFICIENT_LOCAL_SUPPORT', 'local_pa': None,
                'local_intersection_x': float(qx)}
    fit = fit_line(np.c_[lx, ly])
    if fit is None:
        return {'local_status': 'NO_LOCAL_FIT', 'local_pa': None}
    return {'local_status': 'OK', 'local_pa': float(axial_diff(fasc['angle'], fit['angle'])),
            'local_reference_angle': fit['angle'], 'local_residual_px': fit['residual'],
            'local_intersection_x': float(qx), 'local_intersection_y': float(fasc['a'] * qx + fasc['b']),
            'local_root_count': len(roots), 'local_radius_px': float(radius)}


def diagnose(apo_mask, fasc_mask, px_per_mm):
    """R20 diagnose (verbatim logic): accepted R17 fragments + local tangent decoders."""
    apo_mask, fasc_mask = np.asarray(apo_mask, bool), np.asarray(fasc_mask, bool)
    base = measure_geometry(apo_mask, fasc_mask, px_per_mm)
    h, w = apo_mask.shape
    pair = aponeurosis_lines(apo_mask)
    row = {**base, **{v: None for v in R20_VARIANTS}, 'pa_local_support_baseline': None,
           'local_coverage': 0.0, 'n_local': 0, 'orientation_concentration': None,
           'pa_iqr_deg': None, 'exact_mask_baseline': True}
    fragments = []
    if pair is None: return row, fragments
    u, l = pair['upper'], pair['lower']
    contour = lower_contour(apo_mask, l)
    row.update(apo_inner_angle=l['top']['angle'], apo_center_angle=l['center']['angle'])
    for f in fasc_lines(fasc_mask, min_length=max(12, .025 * w)):
        endpoints = []
        for a in (u['bottom'], l['top']):
            den = f['a'] - a['a']
            if abs(den) < 1e-6: break
            x = (a['b'] - f['b']) / den
            endpoints.append((x, f['a'] * x + f['b']))
        if len(endpoints) != 2: continue
        p, q = np.asarray(endpoints)
        if not (-w <= p[0] <= 2 * w and -w <= q[0] <= 2 * w): continue
        xm = (f['x0'] + f['x1']) / 2; ym = f['a'] * xm + f['b']
        if not u['bottom']['a'] * xm + u['bottom']['b'] <= ym <= l['top']['a'] * xm + l['top']['b']: continue
        pa = float(axial_diff(f['angle'], l['top']['angle']))
        length = float(np.linalg.norm(p - q)); fl = length / px_per_mm
        if not (1 < pa < 85 and 0 < fl < 500): continue
        frag = dict(fragment_id=len(fragments), angle_deg=f['angle'], pa_deg=pa,
                    pa_center=float(axial_diff(f['angle'], l['center']['angle'])),
                    visible_length_px=f['length'], fit_residual_px=f['residual'],
                    x0=f['x0'], x1=f['x1'], a=f['a'], b=f['b'],
                    fl_mm=fl, extrapolation_ratio=length / max(f['length'], 1),
                    upper_x=float(p[0]), upper_y=float(p[1]),
                    lower_x=float(q[0]), lower_y=float(q[1]),
                    deep_intersection_inside_image=bool(0 <= q[0] < w and 0 <= q[1] < h))
        frag.update(local_reference(f, contour, q[0], w))
        fragments.append(frag)
    if not fragments:
        return row, fragments
    pa = np.array([f['pa_deg'] for f in fragments])
    weights = np.array([f['visible_length_px'] for f in fragments])
    mean_angle, concentration = axial_mean([f['angle_deg'] for f in fragments], weights)
    differences = axial_diff([f['angle_deg'] for f in fragments], mean_angle)
    coherent_weights = weights * np.exp(-.5 * (differences / R20_CONFIG['orientation_sigma_deg']) ** 2)
    local = [f for f in fragments if f['local_pa'] is not None]
    row.update(pa_r17_median=base['pa_deg'],
               pa_center_median=float(np.median([f['pa_center'] for f in fragments])),
               pa_length_weighted=weighted_median(pa, weights),
               pa_coherent_weighted=weighted_median(pa, coherent_weights),
               orientation_concentration=concentration,
               pa_iqr_deg=float(np.percentile(pa, 75) - np.percentile(pa, 25)),
               n_local=len(local), local_coverage=len(local) / len(fragments))
    if local:
        row.update(pa_local_median=float(np.median([f['local_pa'] for f in local])),
                   pa_local_support_baseline=float(np.median([f['pa_deg'] for f in local])))
    return row, fragments
