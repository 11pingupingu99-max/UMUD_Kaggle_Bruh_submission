"""R71T geometry: fascicle fragments with orientation-field angle and confidence, ground-truth fascicle lines from the
training masks, aponeurosis pair geometry. Everything is exported as plain line parameters (y = a*x + b in native pixel
coordinates of the image), so that every protocol rule (FL, PA, extrapolation, selection) can be recomputed offline.
SPDX-License-Identifier: GPL-3.0-or-later
"""
import math
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize
from r45_core import fit_line, aponeurosis_lines, read_binary

NAN = float('nan')
FRAG_FIELDS = ('x0', 'x1', 'a', 'b', 'angle_fit', 'length', 'residual', 'n', 'angle_or', 'or_R', 'p_mean', 'p_q10')
GT_FIELDS = ('x0', 'x1', 'a', 'b', 'angle', 'length', 'residual', 'n')


def min_fragment_length(w):
    """Same minimum fragment length as r45_core.diagnose / measure_geometry."""
    return max(12, .025 * w)


def frag_lines(p, c, s, thr, w=None):
    """Exactly r45_core.fasc_lines(p >= thr, min_length=max(12, .025 w)) - same skeleton, components, line fits and
    filters, in the same order - plus, on the skeleton pixels of each fragment: probability-weighted axial mean of the
    R22 orientation field (angle_or, resultant length or_R) and the fascicle probability (mean, 10th percentile)."""
    p = np.asarray(p, np.float32)
    w = p.shape[1] if w is None else w
    ml = min_fragment_length(w)
    lab, _ = ndi.label(skeletonize(p >= thr), structure=np.ones((3, 3)))
    out = []
    for i, sl in enumerate(ndi.find_objects(lab), 1):
        if sl is None:
            continue
        y, x = np.nonzero(lab[sl] == i)
        if len(x) < max(6, ml * .5):
            continue
        y = y + sl[0].start; x = x + sl[1].start
        z = fit_line(np.c_[x, y])
        if not (z and z['length'] >= ml and z['residual'] < max(3, .05 * z['length'])):
            continue
        pw = p[y, x].astype(np.float64)
        rec = dict(x0=z['x0'], x1=z['x1'], a=z['a'], b=z['b'], angle_fit=z['angle'], length=z['length'],
                   residual=z['residual'], n=int(z['n']), p_mean=float(pw.mean()), p_q10=float(np.quantile(pw, .1)))
        if c is not None and s is not None:
            C = float((pw * c[y, x]).sum()); S = float((pw * s[y, x]).sum()); W = float(pw.sum())
            rec['angle_or'] = .5 * math.degrees(math.atan2(S, C)) if W > 0 else NAN
            rec['or_R'] = math.hypot(C, S) / W if W > 0 else NAN
        else:
            rec['angle_or'] = NAN; rec['or_R'] = NAN
        out.append(rec)
    return out


def gt_lines(mask, image_shape, min_pixels=8):
    """Annotated fascicle lines (thin traces in the training masks) as straight-line fits in the IMAGE frame.
    Masks whose size differs from the image are mapped by scaling pixel centres (x+0.5)*sx-0.5, (y+0.5)*sy-0.5, which
    is what resizing the mask to the image does (same as the R45T cache)."""
    m = read_binary(mask) if not isinstance(mask, np.ndarray) else (np.asarray(mask) > 0).astype(np.uint8)
    h, w = image_shape
    sy, sx = h / m.shape[0], w / m.shape[1]
    lab, _ = ndi.label(skeletonize(m > 0), structure=np.ones((3, 3)))
    out = []
    for i, sl in enumerate(ndi.find_objects(lab), 1):
        if sl is None:
            continue
        y, x = np.nonzero(lab[sl] == i)
        if len(x) < min_pixels:
            continue
        X = (x + sl[1].start + .5) * sx - .5; Y = (y + sl[0].start + .5) * sy - .5
        z = fit_line(np.c_[X, Y])
        if not z:
            continue
        out.append(dict(x0=z['x0'], x1=z['x1'], a=z['a'], b=z['b'], angle=z['angle'], length=z['length'],
                        residual=z['residual'], n=int(z['n'])))
    return out, {'mask_h': int(m.shape[0]), 'mask_w': int(m.shape[1]), 'mask_fg': float(m.mean())}


def apo_pair_record(prob, thr=.5):
    """Aponeurosis pair exactly as r45_core.aponeurosis_lines (R17 selection) -> flat line parameters."""
    pair = aponeurosis_lines(np.asarray(prob) >= thr)
    if pair is None:
        return None
    rec = dict(x0=float(pair['x0']), x1=float(pair['x1']), quality=float(pair['quality']), n_pairs=int(pair['candidate_pairs']))
    for nm in ('upper', 'lower'):
        comp = pair[nm]
        for edge in ('top', 'center', 'bottom'):
            ln = comp[edge]
            rec[f'{nm[0]}_{edge}_a'] = None if ln is None else float(ln['a'])
            rec[f'{nm[0]}_{edge}_b'] = None if ln is None else float(ln['b'])
        rec[f'{nm[0]}_x0'] = float(comp['center']['x0']); rec[f'{nm[0]}_x1'] = float(comp['center']['x1'])
        rec[f'{nm[0]}_area'] = int(comp['area'])
    return rec


# ----------------------------------------------------------------------------- protocol (offline and in tests)

def axial_diff(a, b):
    return np.abs((np.asarray(a, float) - np.asarray(b, float) + 90) % 180 - 90)


def protocol_fragments(frags, apo, w, h, angle_key='angle_fit'):
    """Fragments -> protocol quantities with an aponeurosis pair (same acceptance rules as r45_core.diagnose):
    FL (px, inner edges), extrapolation ratio, PA vs the deep aponeurosis top edge, both-inside flag, depth fraction.
    angle_key: 'angle_fit' (line fit, as R63T) or 'angle_or' (orientation field; line through the fragment centre)."""
    if apo is None:
        return []
    au, bu = apo['u_bottom_a'], apo['u_bottom_b']; al, bl = apo['l_top_a'], apo['l_top_b']
    deep = math.degrees(math.atan(al))
    out = []
    for f in frags:
        xm = .5 * (f['x0'] + f['x1']); ym = f['a'] * xm + f['b']
        ang = f[angle_key]
        if not np.isfinite(ang):
            continue
        af = math.tan(math.radians(ang)); bf = ym - af * xm
        ends = []
        for a2, b2 in ((au, bu), (al, bl)):
            den = af - a2
            if abs(den) < 1e-6:
                break
            x = (b2 - bf) / den; ends.append((x, af * x + bf))
        if len(ends) != 2:
            continue
        pp, qq = np.asarray(ends)
        if not (-w <= pp[0] <= 2 * w and -w <= qq[0] <= 2 * w):
            continue
        if not (au * xm + bu <= ym <= al * xm + bl):
            continue
        pa = float(axial_diff(ang, deep))
        length = float(np.linalg.norm(pp - qq))
        if not (1 < pa < 85 and 0 < length < 500 * 30):
            continue
        yu = au * xm + bu; yl = al * xm + bl
        out.append(dict(f, fl_px=length, ratio=length / max(f['length'], 1), pa=pa,
                        both_inside=bool(0 <= pp[0] < w and 0 <= pp[1] < h and 0 <= qq[0] < w and 0 <= qq[1] < h),
                        depth_frac=(ym - yu) / (yl - yu) if yl - yu > 1 else NAN, xm_frac=xm / w))
    return out


def lex_select(pf, k, coherent_deg=8.0, order='ratio'):
    """K least-extrapolated coherent fragments (R63T rule): |PA - median PA| <= 8 deg, sort by ratio then length."""
    if not pf:
        return []
    med = float(np.median([f['pa'] for f in pf]))
    coh = [f for f in pf if abs(f['pa'] - med) <= coherent_deg] or pf
    key = (lambda f: (f['ratio'], -f['length'])) if order == 'ratio' else (lambda f: (f['fl_px'] - f['fl_px'] / f['ratio'], -f['length']))
    return sorted(coh, key=key)[:k]
