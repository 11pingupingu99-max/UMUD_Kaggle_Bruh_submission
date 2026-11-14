"""Frozen-mask PA diagnostics. SPDX-License-Identifier: GPL-3.0-or-later."""
import hashlib
import json
import math
import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from umud_core import (aponeurosis_lines, axial_diff, fasc_lines, fit_line,
                       measure_geometry, atomic_json)

VERSION = 'R20_v1.0.0'
CONFIG = {'apo_threshold': .5, 'fasc_threshold': .65,
          'local_radius_width_fraction': .05, 'local_min_radius_px': 12,
          'local_required_span_fraction': .8, 'orientation_sigma_deg': 10.0}
VARIANTS = ('pa_r17_median', 'pa_center_median', 'pa_length_weighted',
            'pa_coherent_weighted', 'pa_local_median')

def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, allow_nan=False).encode()).hexdigest()

def finite_or_none(x):
    return float(x) if x is not None and np.isfinite(x) else None

def weighted_median(values, weights):
    v, w = np.asarray(values, float), np.asarray(weights, float)
    ok = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not ok.any(): return None
    v, w = v[ok], w[ok]
    order = np.argsort(v, kind='stable'); v, w = v[order], w[order]
    return float(v[np.searchsorted(np.cumsum(w), w.sum()/2, side='left')])

def axial_mean(angles, weights):
    a, w = np.radians(2*np.asarray(angles)), np.asarray(weights)
    z = np.sum(w*np.exp(1j*a))/np.sum(w)
    return float(np.degrees(np.angle(z))/2), float(abs(z))

def lower_contour(mask, selected):
    """Recover exactly the connected component selected by original R17 geometry."""
    lab, _ = ndi.label(mask > 0, structure=np.ones((3, 3)))
    matches = []
    for i, sl in enumerate(ndi.find_objects(lab), 1):
        if sl is None: continue
        yy, xx = np.nonzero(lab[sl] == i)
        if len(xx) != selected['area']: continue
        yy, xx = yy+sl[0].start, xx+sl[1].start
        if abs(float(yy.mean())-selected['mid']) > 1e-8: continue
        if abs(float(np.ptp(xx))-selected['width']) > 1e-8: continue
        cols = np.unique(xx)
        top = np.array([yy[xx == x].min() for x in cols], float)
        center = np.array([yy[xx == x].mean() for x in cols], float)
        matches.append((cols.astype(float), top, center))
    if len(matches) != 1:
        raise RuntimeError('Ambiguous lower aponeurosis component identity')
    return matches[0]

def local_reference(fasc, contour, global_qx, width):
    """Observed contour intersection, then PCA tangent in a supported local window.
    Never extrapolate a contour or a tangent beyond observed columns.
    Multiple intersections are resolved by distance to the original global crossing.
    """
    xs, ys, _ = contour
    d = fasc['a']*xs+fasc['b']-ys
    roots = list(xs[np.abs(d) < 1e-9])
    for i in np.flatnonzero((d[:-1]*d[1:] < 0) & (np.diff(xs) <= 1.01)):
        roots.append(float(xs[i]-d[i]*(xs[i+1]-xs[i])/(d[i+1]-d[i])))
    if not roots:
        return {'local_status': 'NO_OBSERVED_INTERSECTION', 'local_pa': None}
    qx = min(roots, key=lambda x: abs(x-global_qx))
    radius = max(CONFIG['local_min_radius_px'], CONFIG['local_radius_width_fraction']*width)
    take = np.abs(xs-qx) <= radius
    lx, ly = xs[take], ys[take]
    fraction = CONFIG['local_required_span_fraction']
    if (len(lx) < 8 or qx-lx.min() < fraction*radius or lx.max()-qx < fraction*radius
            or np.max(np.diff(lx)) > 1.01):
        return {'local_status': 'INSUFFICIENT_LOCAL_SUPPORT', 'local_pa': None,
                'local_intersection_x': float(qx)}
    fit = fit_line(np.c_[lx, ly])
    if fit is None:
        return {'local_status': 'NO_LOCAL_FIT', 'local_pa': None}
    return {'local_status': 'OK', 'local_pa': float(axial_diff(fasc['angle'], fit['angle'])),
            'local_reference_angle': fit['angle'], 'local_residual_px': fit['residual'],
            'local_intersection_x': float(qx), 'local_intersection_y': float(fasc['a']*qx+fasc['b']),
            'local_root_count': len(roots), 'local_radius_px': float(radius)}

def diagnose(apo_mask, fasc_mask, px_per_mm):
    """Exactly reproduce accepted R17 fragments before changing one definition."""
    apo_mask, fasc_mask = np.asarray(apo_mask, bool), np.asarray(fasc_mask, bool)
    base = measure_geometry(apo_mask, fasc_mask, px_per_mm)
    h, w = apo_mask.shape
    pair = aponeurosis_lines(apo_mask)
    row = {**base, **{v: None for v in VARIANTS}, 'pa_local_support_baseline': None,
           'local_coverage': 0.0, 'n_local': 0, 'orientation_concentration': None,
           'pa_iqr_deg': None, 'exact_mask_baseline': True}
    fragments = []
    if pair is None: return row, fragments
    u, l = pair['upper'], pair['lower']
    contour = lower_contour(apo_mask, l)
    row.update(apo_inner_angle=l['top']['angle'], apo_center_angle=l['center']['angle'])
    for f in fasc_lines(fasc_mask, min_length=max(12, .025*w)):
        endpoints = []
        for a in (u['bottom'], l['top']):
            den = f['a']-a['a']
            if abs(den) < 1e-6: break
            x = (a['b']-f['b'])/den
            endpoints.append((x, f['a']*x+f['b']))
        if len(endpoints) != 2: continue
        p, q = np.asarray(endpoints)
        if not (-w <= p[0] <= 2*w and -w <= q[0] <= 2*w): continue
        xm = (f['x0']+f['x1'])/2; ym = f['a']*xm+f['b']
        if not u['bottom']['a']*xm+u['bottom']['b'] <= ym <= l['top']['a']*xm+l['top']['b']: continue
        pa = float(axial_diff(f['angle'], l['top']['angle']))
        length = float(np.linalg.norm(p-q)); fl = length/px_per_mm
        if not (1 < pa < 85 and 0 < fl < 500): continue
        frag = dict(fragment_id=len(fragments), angle_deg=f['angle'], pa_deg=pa,
                    pa_center=float(axial_diff(f['angle'], l['center']['angle'])),
                    visible_length_px=f['length'], fit_residual_px=f['residual'],
                    x0=f['x0'], x1=f['x1'], a=f['a'], b=f['b'],
                    fl_mm=fl, extrapolation_ratio=length/max(f['length'], 1),
                    upper_x=float(p[0]), upper_y=float(p[1]),
                    lower_x=float(q[0]), lower_y=float(q[1]),
                    deep_intersection_inside_image=bool(0 <= q[0] < w and 0 <= q[1] < h))
        frag.update(local_reference(f, contour, q[0], w))
        fragments.append(frag)
    if not fragments:
        assert base['n_fasc'] == 0
        return row, fragments
    pa = np.array([f['pa_deg'] for f in fragments])
    weights = np.array([f['visible_length_px'] for f in fragments])
    assert len(fragments) == base['n_fasc']
    for field in ('pa_deg', 'fl_mm', 'extrapolation_ratio'):
        assert abs(float(np.median([f[field] for f in fragments]))-base[field]) < 1e-10, field
    mean_angle, concentration = axial_mean([f['angle_deg'] for f in fragments], weights)
    differences = axial_diff([f['angle_deg'] for f in fragments], mean_angle)
    coherent_weights = weights*np.exp(-.5*(differences/CONFIG['orientation_sigma_deg'])**2)
    local = [f for f in fragments if f['local_pa'] is not None]
    row.update(pa_r17_median=base['pa_deg'],
               pa_center_median=float(np.median([f['pa_center'] for f in fragments])),
               pa_length_weighted=weighted_median(pa, weights),
               pa_coherent_weighted=weighted_median(pa, coherent_weights),
               orientation_concentration=concentration,
               pa_iqr_deg=float(np.percentile(pa, 75)-np.percentile(pa, 25)),
               n_local=len(local), local_coverage=len(local)/len(fragments))
    if local:
        row.update(pa_local_median=float(np.median([f['local_pa'] for f in local])),
                   pa_local_support_baseline=float(np.median([f['pa_deg'] for f in local])))
    return row, fragments

def check_previous(frame, previous):
    if frame.image_id.duplicated().any() or previous.image_id.duplicated().any():
        raise ValueError('Duplicate baseline IDs')
    if set(frame.image_id) != set(previous.image_id): raise ValueError('Baseline ID mismatch')
    x = frame.merge(previous, on='image_id', suffixes=('_now', '_old'), validate='one_to_one')
    report = {'n': len(x), 'same_status': bool((x.status_now == x.status_old).all()),
              'same_fragment_count': bool((x.n_fasc_now == x.n_fasc_old).all()), 'fields': {}}
    for col, tol in [('pa_deg', .03), ('fl_mm', .1), ('mt_mm', .03)]:
        a, b = x[col+'_now'].to_numpy(float), x[col+'_old'].to_numpy(float)
        valid = np.isfinite(a) & np.isfinite(b)
        same = bool(np.array_equal(np.isfinite(a), np.isfinite(b)))
        error = float(np.max(np.abs(a[valid]-b[valid]))) if valid.any() else None
        report['fields'][col] = {'max_abs_delta': error, 'tolerance': tol, 'same_coverage': same,
                                'pass': same and (error is None or error <= tol)}
    report['pass'] = report['same_status'] and report['same_fragment_count'] and all(z['pass'] for z in report['fields'].values())
    return report

def external_comparison(predictions, truth):
    if set(predictions.image_id) != set(truth.image_id): raise ValueError('OSF ID mismatch')
    x = predictions.merge(truth[['image_id', 'pa_gt']], on='image_id', validate='one_to_one')
    rows = []
    all_common = np.isfinite(x[['pa_gt', *VARIANTS]].astype(float)).all(axis=1)
    for variant in VARIANTS:
        # For local tangent report both the global baseline and SAME FRAGMENTS baseline.
        refs = ['pa_r17_median'] if variant != 'pa_local_median' else ['pa_r17_median', 'pa_local_support_baseline']
        for ref in refs:
            for support, ok in [('pairwise', np.isfinite(x[['pa_gt', variant, ref]].astype(float)).all(axis=1)),
                                ('all_variants_common', all_common & x[ref].notna())]:
                a = np.abs(x.loc[ok, variant]-x.loc[ok, 'pa_gt'])
                b = np.abs(x.loc[ok, ref]-x.loc[ok, 'pa_gt'])
                rows.append(dict(variant=variant, reference=ref, support=support, n=int(ok.sum()), total=len(x),
                                 candidate_available=int(x[variant].notna().sum()),
                                 candidate_MAE=finite_or_none(a.mean()), reference_MAE=finite_or_none(b.mean()),
                                 paired_delta_MAE=finite_or_none((a-b).mean()),
                                 improved=int((a < b).sum()), worsened=int((a > b).sum()),
                                 role='REUSED_OSF_DIAGNOSTIC_ONLY'))
    return pd.DataFrame(rows)

def mask_digest(arrays):
    h = hashlib.sha256()
    for name in sorted(arrays):
        a = np.asarray(arrays[name], dtype=bool)
        h.update(name.encode()); h.update(str(a.shape).encode()); h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()

def save_masks(path, signature, **arrays):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {'signature': signature, 'sha256': mask_digest(arrays)}
    temp = path.with_suffix('.tmp.npz')
    with temp.open('wb') as handle:
        np.savez_compressed(handle, metadata=np.array(json.dumps(metadata, sort_keys=True)),
                            **{k: np.asarray(v, bool) for k, v in arrays.items()})
    os.replace(temp, path)

def load_masks(path, signature):
    with np.load(path, allow_pickle=False) as z:
        metadata = json.loads(str(z['metadata']))
        arrays = {k: z[k].copy() for k in z.files if k != 'metadata'}
    if metadata['signature'] != signature: raise ValueError('Stale mask cache signature')
    if metadata['sha256'] != mask_digest(arrays): raise ValueError('Corrupt mask cache')
    return arrays

def selftest():
    import cv2
    h, w = 400, 800
    ap = np.zeros((h, w), np.uint8); fa = np.zeros_like(ap)
    ap[78:83, 20:780] = 1; ap[238:243, 20:780] = 1
    cv2.line(fa, (150, 60), (550, 291), 1, 2)
    z, fs = diagnose(ap, fa, 10.)
    assert abs(z['pa_r17_median']-30) < .7 and z['n_local'] == 1
    assert abs(z['pa_local_median']-z['pa_r17_median']) < .1
    mirrored, _ = diagnose(ap[:, ::-1], fa[:, ::-1], 10.)
    scaled, _ = diagnose(ap, fa, 20.)
    resized, _ = diagnose(cv2.resize(ap, (1600, 800), interpolation=cv2.INTER_NEAREST),
                          cv2.resize(fa, (1600, 800), interpolation=cv2.INTER_NEAREST), 20.)
    for name in VARIANTS:
        assert abs(mirrored[name]-z[name]) < .3, name
        assert abs(scaled[name]-z[name]) < 1e-9, name
        assert abs(resized[name]-z[name]) < .4, name
    # Short fragment whose deep intersection lies outside the observed apo support.
    short = np.zeros_like(ap); cv2.line(short, (280, 130), (420, 158), 1, 1)
    outside, fr = diagnose(ap, short, 10.)
    assert outside['n_fasc'] > 0 and outside['n_local'] == 0
    assert all(f['local_status'] == 'NO_OBSERVED_INTERSECTION' for f in fr)
    empty, _ = diagnose(np.zeros_like(ap), fa, 10.)
    assert empty['status'] == 'NO_APO_PAIR' and empty['pa_local_median'] is None
    # Analytic curved contour; tangent differs from the global reference.
    xs = np.arange(800, dtype=float); ys = 200+.0004*(xs-400)**2
    slope = .5; qx = 600.; fy = 200+.0004*(qx-400)**2
    local = local_reference({'a': slope, 'b': fy-slope*qx, 'angle': math.degrees(math.atan(slope))},
                            (xs, ys, ys+2), qx, 800)
    expected = abs(math.degrees(math.atan(slope))-math.degrees(math.atan(.0008*(qx-400))))
    assert local['local_status'] == 'OK' and abs(local['local_pa']-expected) < .1
    return {'pass': True, 'straight_PA': z['pa_r17_median'], 'curved_local_PA': local['local_pa'],
            'checks': ['baseline_identity', 'mirror', 'units', 'isotropic_resize', 'unsupported_intersection', 'missing_apo', 'analytic_curve']}
