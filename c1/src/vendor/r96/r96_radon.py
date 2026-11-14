"""R96T: texture orientation of the muscle from the Radon transform (no model, no training).

Every image is cut into overlapping square tiles (side 1/6 of the shorter image side, stride 1/2 tile). For each tile and each
line angle phi in -60..60 deg (2 deg step) we compute the band-limited variance of the Radon projection along phi (integration
lines parallel to phi). Oriented texture (fascicles) keeps its stripes only in the projection along its own direction, so the
profile peaks at the texture angle. The projection variance is computed with the Fourier slice theorem: the 1-D spectrum of the
projection along phi is the 2-D spectrum of the tile on the central slice perpendicular to phi, so one FFT per tile gives the
whole profile (tile mean removed, radial Hann window, zero padding x3, spatial wavelengths 4..48 native pixels).

Angle convention as `angle_fit` of the fascicle fragments: atan(dy/dx) in image coordinates (x right, y down).
Images enter as the letterboxed 0..1 square of the training cache (side 1024, uint8-quantised), so training (cache) and test
(native image) tiles are made the same way. Tiles are exported with their centre and side in NATIVE pixels; the choice of
tiles (muscle between the aponeuroses) and the aggregation are done offline, with the aponeurosis pair of each image.
SPDX-License-Identifier: GPL-3.0-or-later
"""
import base64
import numpy as np
import cv2
from scipy import ndimage as ndi
import r45_core

ANGLES = np.arange(-60., 60.001, 2.)
WORK = 96                       # tile resampled to WORK x WORK before the FFT
PAD = 3                         # zero padding of the FFT (finer spectrum grid: angle error < 0.3 deg on test stripes)
BAND_NATIVE = (4., 48.)         # spatial wavelengths kept (native pixels)
TILE_FRAC, STRIDE_FRAC = 1 / 6., .5
ROW_FIELDS = ('cx', 'cy', 'side', 'mean', 'std', 'energy', 'peak_rel')


def radial_window(n):
    yy, xx = np.mgrid[:n, :n].astype(np.float64)
    c = (n - 1) / 2.
    r = np.hypot(yy - c, xx - c) / (n / 2.)
    return np.where(r < 1, .5 * (1 + np.cos(np.pi * r)), 0.).astype(np.float32)


def slice_coords(n, k_lo, k_hi, angles=ANGLES):
    """(row, col) positions in the fftshifted n x n spectrum of the central slice perpendicular to each line angle.
    Line direction d = (cos phi, sin phi) (x, y; y down) -> slice along the normal (-sin phi, cos phi); k in cycles/pixel."""
    ks = np.arange(k_lo, k_hi + 1e-9, .5 / n)
    phi = np.radians(np.asarray(angles, float))[:, None]
    fx = -np.sin(phi) * ks[None]
    fy = np.cos(phi) * ks[None]
    return np.stack([fy * n + n // 2, fx * n + n // 2]), len(ks)


def tile_profile(patch, win, coords, pad=PAD):
    """Band-limited Radon projection variance of one tile for every angle of `coords` (unnormalised). `coords` must be made
    for the padded size: slice_coords(work * pad, ...)."""
    z = np.asarray(patch, np.float32)
    n = z.shape[0]
    Z = np.zeros((n * pad, n * pad), np.float32)
    Z[:n, :n] = (z - z.mean()) * win
    P = np.abs(np.fft.fftshift(np.fft.fft2(Z))) ** 2
    v = ndi.map_coordinates(P, coords.reshape(2, -1), order=1, mode='nearest').reshape(coords.shape[1:])
    return v.sum(1)


def radon_tiles(square, hw, tile_frac=TILE_FRAC, stride_frac=STRIDE_FRAC, work=WORK, band_native=BAND_NATIVE, angles=ANGLES):
    """square: letterboxed 0..1 image of a native image of size hw = (h, w).
    Returns rows (n_tiles x 7 float32: cx, cy, side [native px], mean, std [0..1], energy, peak_rel) and profiles
    (n_tiles x n_angles, uint8 = round(255 * profile / max)); energy = profile mean / number of spectrum samples,
    peak_rel = max / mean, so profile / mean = prof_u8 / 255 * peak_rel."""
    nh, nw, y0, x0 = r45_core.letterbox_geometry(hw, square.shape[0])
    img = np.asarray(square, np.float32)[y0:y0 + nh, x0:x0 + nw]
    r = nw / float(hw[1])                                        # square pixels per native pixel
    T = max(16, int(round(tile_frac * min(nh, nw))))
    st = max(4, int(round(T * stride_frac)))
    f = work / float(T)                                          # work pixels per square pixel
    lam = np.asarray(band_native, float) * r * f                 # kept wavelengths in work pixels
    k_lo, k_hi = 1. / lam[1], min(1. / lam[0], .45)
    coords, nk = slice_coords(work * PAD, k_lo, k_hi, angles)
    win = radial_window(work)
    ny, nx = (nh - T) // st + 1, (nw - T) // st + 1
    oy, ox = (nh - T - (ny - 1) * st) // 2, (nw - T - (nx - 1) * st) // 2
    rows, profs = [], []
    for iy in range(ny):
        for ix in range(nx):
            ty, tx = oy + iy * st, ox + ix * st
            tile = img[ty:ty + T, tx:tx + T]
            patch = cv2.resize(tile, (work, work), interpolation=cv2.INTER_AREA if T > work else cv2.INTER_LINEAR)
            pr = tile_profile(patch, win, coords)
            m, mx = float(pr.mean()), float(pr.max())
            rows.append(((tx + T / 2.) / r - .5, (ty + T / 2.) / r - .5, T / r, float(tile.mean()), float(tile.std()), m / nk,
                         mx / m if m > 0 else 0.))
            profs.append(np.rint(255. * pr / mx) if mx > 0 else np.zeros_like(pr))
    return (np.asarray(rows, np.float32).reshape(-1, len(ROW_FIELDS)),
            np.asarray(profs, np.float64).clip(0, 255).astype(np.uint8).reshape(-1, len(angles)))


def encode(rows, profs):
    return {'n': int(len(rows)), 'rows': base64.b64encode(np.ascontiguousarray(rows, np.float32).tobytes()).decode(),
            'prof': base64.b64encode(np.ascontiguousarray(profs, np.uint8).tobytes()).decode()}


def decode(d, n_angles=len(ANGLES)):
    rows = np.frombuffer(base64.b64decode(d['rows']), np.float32).reshape(-1, len(ROW_FIELDS))
    profs = np.frombuffer(base64.b64decode(d['prof']), np.uint8).reshape(-1, n_angles)
    return rows, profs


def relative(rows, profs):
    """uint8 profiles -> profile / mean (float)."""
    return profs.astype(np.float64) / 255. * rows[:, ROW_FIELDS.index('peak_rel')][:, None]


def peak(profile, angles=ANGLES):
    """Peak angle (parabolic interpolation, deg) and prominence (peak / median - 1) of an aggregated profile."""
    p = np.asarray(profile, float)
    i = int(np.argmax(p))
    off = 0.
    if 0 < i < len(p) - 1:
        den = p[i - 1] - 2 * p[i] + p[i + 1]
        off = .5 * (p[i - 1] - p[i + 1]) / den if den < 0 else 0.
    step = float(angles[1] - angles[0])
    med = float(np.median(p))
    return float(angles[i] + off * step), (float(p[i]) / med - 1.) if med > 0 else float('nan'), bool(0 < i < len(p) - 1)


def export(name, recs, out_dir):
    """recs: [{'image_id', 'radon': encode(...)}] -> {name}_radon.npz (tiles of all images, image index per tile)."""
    ids, idx, R, Pf = [], [], [], []
    for r in recs:
        d = r.get('radon')
        if not d:
            continue
        rows, profs = decode(d)
        idx.append(np.full(len(rows), len(ids), np.int32)); ids.append(r['image_id']); R.append(rows); Pf.append(profs)
    if not ids:
        return None
    p = out_dir / f'{name}_radon.npz'
    np.savez_compressed(p, image_id=np.asarray(ids), tile_image=np.concatenate(idx), rows=np.concatenate(R),
                        prof=np.concatenate(Pf), angles=ANGLES.astype(np.float32), row_fields=np.asarray(ROW_FIELDS),
                        band_native=np.asarray(BAND_NATIVE, np.float32), work=np.int32(WORK))
    return p
