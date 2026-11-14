"""R45T decoders. SPDX-License-Identifier: GPL-3.0-or-later.

1) Aponeurosis pair with SUB-PIXEL edges from a probability map (same pair selection as R17).
2) PA from the dense axial orientation field of R22 (weighted by fascicle probability).
3) FL by streamlines through the smoothed orientation field (curvature-aware, straight
   extrapolation outside the image), chord and arc length.
4) R17/R20 line decoders on any aponeurosis mask (pa_r17_median, local support, local median).
All angles are in pixel space (aspect correction is applied later, per source, on Kaggle).
"""
import math
import numpy as np
from scipy import ndimage as ndi
from r45_core import aponeurosis_lines, axial_diff, diagnose

NAN = float('nan')


def _robust_poly(x, y, deg=1, passes=2):
    x = np.asarray(x, float); y = np.asarray(y, float)
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < deg + 3:
        return None, keep
    for _ in range(passes + 1):
        coef = np.polyfit(x[keep], y[keep], deg)
        r = y - np.polyval(coef, x)
        mad = 1.4826 * np.median(np.abs(r[keep] - np.median(r[keep])))
        new = keep & (np.abs(r) <= max(1.0, 3 * mad))
        if new.sum() < deg + 3 or new.sum() == keep.sum():
            break
        keep = new
    coef = np.polyfit(x[keep], y[keep], deg)
    return coef, keep


def _first_last(mask_cols):
    """mask_cols: (h, n) bool. first/last True row per column (-1 if none)."""
    h = mask_cols.shape[0]
    has = mask_cols.any(0)
    first = np.where(has, mask_cols.argmax(0), -1)
    last = np.where(has, h - 1 - mask_cols[::-1].argmax(0), -1)
    return first, last, has


def _cross_down(prob, r, cols, thr):
    """Sub-pixel crossing below row r (prob[r]>=thr, prob[r+1]<thr)."""
    h = prob.shape[0]
    y = r.astype(float) + .5
    ok = (r >= 0) & (r + 1 < h)
    a = prob[np.clip(r, 0, h - 1), cols]; b = prob[np.clip(r + 1, 0, h - 1), cols]
    den = a - b
    good = ok & (den > 1e-6)
    y[good] = r[good] + np.clip((a[good] - thr) / den[good], 0, 1)
    return y


def _cross_up(prob, r, cols, thr):
    """Sub-pixel crossing above row r (prob[r]>=thr, prob[r-1]<thr)."""
    h = prob.shape[0]
    y = r.astype(float) - .5
    ok = (r - 1 >= 0) & (r < h)
    a = prob[np.clip(r - 1, 0, h - 1), cols]; b = prob[np.clip(r, 0, h - 1), cols]
    den = b - a
    good = ok & (den > 1e-6)
    y[good] = (r[good] - 1) + np.clip((thr - a[good]) / den[good], 0, 1)
    return y


def apo_pair_subpixel(prob, thr=.5):
    """Returns dict with fitted edge lines (pixel coords, y = a*x + b) and MT measures in px, or None."""
    prob = np.asarray(prob, np.float32); h, w = prob.shape
    pair, lab = aponeurosis_lines(prob >= thr, return_labels=True)
    if pair is None:
        return None
    u, l = pair['upper'], pair['lower']
    x0 = max(0, int(math.ceil(pair['x0']))); x1 = min(w - 1, int(math.floor(pair['x1'])))
    if x1 - x0 < 10:
        return None
    cols = np.arange(x0, x1 + 1)
    U = lab[:, cols] == u['label']; L = lab[:, cols] == l['label']
    ut, ub, hu = _first_last(U); lt, lb, hl = _first_last(L)
    ok = hu & hl & (lt > ub)
    if ok.sum() < 10:
        return None
    c = cols[ok]
    y_ub = _cross_down(prob, ub[ok], c, thr); y_ut = _cross_up(prob, ut[ok], c, thr)
    y_lt = _cross_up(prob, lt[ok], c, thr); y_lb = _cross_down(prob, lb[ok], c, thr)
    rows = np.arange(h, dtype=float)[:, None]
    pu = np.where(U[:, ok], prob[:, c], 0.); pl = np.where(L[:, ok], prob[:, c], 0.)
    y_uc = (rows * pu).sum(0) / np.maximum(pu.sum(0), 1e-6)
    y_lc = (rows * pl).sum(0) / np.maximum(pl.sum(0), 1e-6)
    fits = {}
    for name, y in (('ub', y_ub), ('ut', y_ut), ('lt', y_lt), ('lb', y_lb), ('uc', y_uc), ('lc', y_lc)):
        coef, keep = _robust_poly(c, y, 1)
        if coef is None:
            return None
        fits[name] = (float(coef[0]), float(coef[1]))
    q, _ = _robust_poly(c, y_lt, 2)
    xa, xb = float(c.min()), float(c.max())
    xs = np.linspace(xa, xb, 101)
    line = lambda k, x: fits[k][0] * x + fits[k][1]
    inner = line('lt', xs) - line('ub', xs); center = line('lc', xs) - line('uc', xs)
    xm = .5 * (xa + xb)
    raw_gap = y_lt - y_ub
    am = .5 * (fits['ub'][0] + fits['lt'][0])
    m = dict(
        mt_inner_mean=float(inner.mean()),
        mt_inner_mid=float(line('lt', xm) - line('ub', xm)),
        mt_inner_c=float(line('lt', w / 2) - line('ub', w / 2)) if xa <= w / 2 <= xb else NAN,
        mt_inner_rawmed=float(np.median(raw_gap)),
        mt_inner_perp=float(inner.mean() * math.cos(math.atan(am))),
        mt_center_mean=float(center.mean()),
        band_corr=float(center.mean() - inner.mean()),
        band_u=float(np.median(y_ub - y_ut)), band_l=float(np.median(y_lb - y_lt)),
        deep_top_deg=math.degrees(math.atan(fits['lt'][0])), deep_center_deg=math.degrees(math.atan(fits['lc'][0])),
        sup_bottom_deg=math.degrees(math.atan(fits['ub'][0])), sup_center_deg=math.degrees(math.atan(fits['uc'][0])),
        overlap_frac=float((xb - xa) / w), n_cols=int(ok.sum()), candidate_pairs=int(pair['candidate_pairs']),
        upper_mid_y=float(line('uc', xm)), lower_mid_y=float(line('lc', xm)),
        deep_resid_px=float(np.sqrt(np.mean((y_lt - line('lt', c)) ** 2))),
        sup_resid_px=float(np.sqrt(np.mean((y_ub - line('ub', c)) ** 2))),
    )
    return dict(fits=fits, quad_lt=None if q is None else tuple(float(v) for v in q), xa=xa, xb=xb,
                shape=(h, w), m=m, r17=pair)


def _region(sub, shape, margin_frac=.06, margin_px=2., border_frac=.02):
    """Muscle zone between the inner aponeurosis edges, away from the bands and from the left/right image
    borders (network border artefacts)."""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    a_u, b_u = sub['fits']['ub']; a_l, b_l = sub['fits']['lt']
    yu = a_u * xx + b_u; yl = a_l * xx + b_l; gap = yl - yu
    marg = margin_frac * gap + margin_px
    bx = max(4., border_frac * w)
    M = (yy > yu + marg) & (yy < yl - marg) & (gap > .025 * h) & (xx >= bx) & (xx <= w - 1 - bx)
    frac = np.where(gap > 1, (yy - yu) / np.maximum(gap, 1), -1)
    return M, frac, xx, yy


def orientation_decoders(p, c, s, sub, min_weight=50.):
    """PA from the probability-weighted axial mean of the R22 orientation field in muscle zones."""
    out = {}
    if sub is None:
        return out
    M, frac, xx, yy = _region(sub, p.shape)
    a_l = sub['fits']['lt'][0]
    th_apo = math.degrees(math.atan(a_l))
    # gamma=2 suppresses the (unsupervised) background orientation between fascicles
    zones = (('all', M, 1), ('p2', M, 2), ('thr', M & (p >= .5), 1), ('ins', M & (frac >= .6), 2),
             ('mid', M & (frac >= .3) & (frac <= .7), 2), ('sup', M & (frac <= .4), 2))
    for name, sel, gamma in zones:
        wgt = np.where(sel, p ** gamma, 0.).astype(np.float64)
        W = float(wgt.sum())
        if W < min_weight:
            out[f'pa_or_{name}'] = NAN; out[f'or_R_{name}'] = NAN; out[f'or_W_{name}'] = W
            continue
        C = float((wgt * c).sum()); S = float((wgt * s).sum())
        th = .5 * math.degrees(math.atan2(S, C))
        out[f'pa_or_{name}'] = float(axial_diff(th, th_apo)); out[f'or_R_{name}'] = math.hypot(C, S) / W
        out[f'or_W_{name}'] = W; out[f'or_theta_{name}'] = th
        if name == 'ins' and sub['quad_lt'] is not None:
            xbar = float((wgt * xx).sum() / W)
            q2, q1, _ = sub['quad_lt']
            th_loc = math.degrees(math.atan(2 * q2 * xbar + q1))
            out['pa_or_ins_tan'] = float(axial_diff(th, th_loc))
    return out


def _dir_field(p, c, s, M, sigma):
    wgt = (p * p * M).astype(np.float32)
    Cs = ndi.gaussian_filter(wgt * c, sigma); Ss = ndi.gaussian_filter(wgt * s, sigma)
    Ws = ndi.gaussian_filter(wgt, sigma)
    coh = np.hypot(Cs, Ss) / np.maximum(Ws, 1e-6)
    return Cs, Ss, Ws, coh


def _sample(field, pts):
    return ndi.map_coordinates(field, [pts[:, 1], pts[:, 0]], order=1, mode='nearest')


def streamline_fl(p, c, s, sub, step=2.0, max_seeds=240, sigma=None, max_len_frac=4.0):
    """Trace fascicle streamlines from seeds inside the muscle up to the superficial and down to the deep
    aponeurosis inner edges. Leaving the image or an incoherent field -> straight extrapolation with the mean
    direction of the last steps. Returns per-image medians in px."""
    out = {}
    if sub is None:
        return out
    h, w = p.shape
    sigma = sigma or max(6., .02 * w)
    M, frac, xx, yy = _region(sub, (h, w), margin_frac=.04, margin_px=2.)
    Cs, Ss, Ws, coh = _dir_field(p, c, s, M, sigma)
    wmax = float(Ws.max()) if Ws.size else 0.
    if wmax <= 0:
        return out
    bx = max(6., 1.5 * sigma)
    valid = (Ws > .05 * wmax) & (coh > .3) & (xx >= bx) & (xx <= w - 1 - bx)
    M = M & valid
    cand = M & (p >= .5) & (frac >= .2) & (frac <= .8) & valid
    ys, xs = np.nonzero(cand)
    if len(xs) < 5:
        cand = M & (p >= .3) & valid; ys, xs = np.nonzero(cand)
    if len(xs) < 5:
        return out
    rng = np.random.default_rng(12345)
    if len(xs) > max_seeds:
        pick = np.sort(rng.choice(len(xs), max_seeds, replace=False)); xs, ys = xs[pick], ys[pick]
    seeds = np.c_[xs, ys].astype(np.float64)
    a_u, b_u = sub['fits']['ub']; a_l, b_l = sub['fits']['lt']
    gap_mid = (a_l * w / 2 + b_l) - (a_u * w / 2 + b_u)
    max_steps = int(max_len_frac * max(h, w) / step)

    def direction(pts):
        C = _sample(Cs, pts); S = _sample(Ss, pts)
        th = .5 * np.arctan2(S, C)
        return np.c_[np.cos(th), np.sin(th)]

    def trace(sign):
        pos = seeds.copy(); n = len(pos)
        prev = np.tile(np.array([0., float(sign)]), (n, 1))
        hist = np.zeros((n, 2)); arc = np.zeros(n); extr = np.zeros(n)
        active = np.ones(n, bool); end = pos.copy()
        for _ in range(max_steps):
            if not active.any():
                break
            idx = np.flatnonzero(active); P = pos[idx]
            d = direction(P); flip = (d * prev[idx]).sum(1) < 0; d[flip] *= -1
            mid = P + .5 * step * d
            d2 = direction(mid); flip2 = (d2 * d).sum(1) < 0; d2[flip2] *= -1
            newp = P + step * d2
            # leaving the muscle zone through the target boundary -> straight to the apo inner edge
            x, y = newp[:, 0], newp[:, 1]
            if sign > 0:
                y_apo = a_l * x + b_l; reached = y >= y_apo - (.04 * (y_apo - (a_u * x + b_u)) + 2.)
            else:
                y_apo = a_u * x + b_u; reached = y <= y_apo + (.04 * ((a_l * x + b_l) - y_apo) + 2.)
            outside = (x < 0) | (x > w - 1) | (y < 0) | (y > h - 1)
            inval = np.zeros(len(idx), bool)
            inside = ~outside
            if inside.any():
                ii = np.flatnonzero(inside)
                xi = np.clip(np.rint(x[ii]).astype(int), 0, w - 1); yi = np.clip(np.rint(y[ii]).astype(int), 0, h - 1)
                inval[ii] = ~valid[yi, xi]
            hist[idx] = .95 * hist[idx] + .05 * d2  # ~40 px memory: extrapolation direction
            arc[idx] += step
            pos[idx] = newp; prev[idx] = d2
            stop = reached | outside | inval
            if stop.any():
                sidx = idx[stop]
                dd = hist[sidx]; nn = np.linalg.norm(dd, axis=1, keepdims=True); dd = dd / np.maximum(nn, 1e-9)
                dd[(dd[:, 1] * sign) <= 1e-3] = np.array([0., float(sign)])  # never travel away from the target
                P0 = pos[sidx]
                a, b = (a_l, b_l) if sign > 0 else (a_u, b_u)
                den = dd[:, 1] - a * dd[:, 0]
                t = (a * P0[:, 0] + b - P0[:, 1]) / np.where(np.abs(den) > 1e-6, den, np.nan)
                t = np.where(np.isfinite(t) & (t > -3 * step), np.maximum(t, 0), np.nan)
                end[sidx] = P0 + t[:, None] * dd
                arc[sidx] += np.nan_to_num(t, nan=0.)
                extr[sidx] = np.where(reached[stop], 0., np.nan_to_num(t, nan=0.))
                bad = ~np.isfinite(t)
                arc[sidx[bad]] = np.nan
                active[sidx] = False
        arc[active] = np.nan; end[active] = np.nan
        return end, arc, extr

    e_dn, arc_dn, ex_dn = trace(+1); e_up, arc_up, ex_up = trace(-1)
    chord = np.linalg.norm(e_dn - e_up, axis=1); arc = arc_dn + arc_up
    ok = np.isfinite(chord) & np.isfinite(arc) & (chord > .5 * gap_mid)
    if ok.sum() < 3:
        out['st_n'] = int(ok.sum()); return out
    ch = chord[ok]; ar = arc[ok]; ex = (ex_dn + ex_up)[ok]
    out.update(st_chord_px=float(np.median(ch)), st_arc_px=float(np.median(ar)),
               st_chord_iqr_px=float(np.percentile(ch, 75) - np.percentile(ch, 25)),
               st_extr_frac=float(np.median(ex / np.maximum(ar, 1e-6))), st_n=int(ok.sum()),
               st_sigma_px=float(sigma))
    # straight-line angle of each streamline chord relative to the deep apo line
    vec = (e_dn - e_up)[ok]; th = np.degrees(np.arctan2(vec[:, 1], vec[:, 0]))
    out['st_pa_chord'] = float(np.median(axial_diff(th, math.degrees(math.atan(a_l)))))
    return out


LINE_KEYS = ('pa_r17_median', 'pa_local_support_baseline', 'pa_local_median', 'pa_center_median',
             'pa_length_weighted', 'pa_coherent_weighted', 'fl_mm', 'mt_mm', 'mt_center_mm', 'band_correction_mm',
             'n_fasc', 'n_local', 'status', 'apo_inner_angle', 'extrapolation_ratio', 'orientation_concentration', 'pa_iqr_deg')


def line_decoders(apo_prob, fasc_prob, px_per_mm, apo_thr=.5, fasc_thr=.5, fascicle_lines=None):
    try:
        row, _ = diagnose(np.asarray(apo_prob) >= apo_thr, np.asarray(fasc_prob) >= fasc_thr, float(px_per_mm), fascicle_lines=fascicle_lines)
    except Exception as e:  # noqa: BLE001 - one image must never stop the run
        return {'status': 'ERROR:' + type(e).__name__}
    return {k: row.get(k) for k in LINE_KEYS}


def decode_image(p, c, s, apo_probs, px_per_mm, with_streamlines=True, fascicle_lines=None):
    """apo_probs: {name: native probability map}. Returns a flat dict of features (mm and degrees)."""
    out = {}
    ppm = float(px_per_mm)
    h, w = p.shape
    W = float(p.sum())
    if W > 0:
        C = float((p * c).sum()); S = float((p * s).sum())
        out['img_theta_p'] = .5 * math.degrees(math.atan2(S, C)); out['img_R_p'] = math.hypot(C, S) / W
    for name, prob in apo_probs.items():
        if prob is None:
            continue
        if prob.shape != p.shape:
            raise ValueError(f'{name}: aponeurosis map {prob.shape} != fascicle map {p.shape}')
        sub = apo_pair_subpixel(prob)
        out[f'{name}_pair'] = sub is not None
        if sub is not None:
            for k, v in sub['m'].items():
                if k.startswith(('mt_', 'band_')):
                    out[f'{name}_{k}_mm'] = v / ppm
                else:
                    out[f'{name}_{k}'] = v
            out[f'{name}_upper_mid_y_frac'] = sub['m']['upper_mid_y'] / h
            out[f'{name}_lower_mid_y_frac'] = sub['m']['lower_mid_y'] / h
            for k, v in orientation_decoders(p, c, s, sub).items():
                out[f'{name}_{k}'] = v
            if with_streamlines:
                for k, v in streamline_fl(p, c, s, sub).items():
                    out[f'{name}_{k[:-3]}_mm' if k.endswith('_px') else f'{name}_{k}'] = (v / ppm if k.endswith('_px') else v)
        for k, v in line_decoders(prob, p, ppm, fascicle_lines=fascicle_lines).items():
            out[f'{name}_L_{k}'] = v
    return out
