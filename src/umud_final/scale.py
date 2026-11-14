"""UMUD final pipeline - physical scale (px per cm) read from the on-screen rulers.

Six screen layouts occur in the challenge test set. For each one the scale is read from the image itself:

* ``1200_right_ticks`` (Siemens Acuson, 1200x800 screenshot): tick column right of the B-mode image. The ruler
  runs from the 0 mark to the full imaging depth; ``px/cm = span / depth``, where the depth (multiple of 0.5 cm)
  follows from the spacing of the long (1 cm) ticks.
* ``png_variable`` (Philips Lumify, PNG): ticks every 0.5 cm on the left edge, long ticks at whole centimetres;
  ``px/cm`` = distance between the 0 cm and the 1 cm tick.
* ``1200_left_ticks`` (Telemed, 1200x800), ``644x1088`` (Telemed crop), ``853_gray`` and ``512_fixed`` (raw Telemed
  frames): fixed device layouts. The ruler is measured (left ruler 0-50 mm, or bottom ticks 1 cm apart) and the
  layout's calibration constant is used when the measurement agrees with it within ``FIXED_TOL``; otherwise the
  measured value is returned and flagged.

Licence: GPL-3.0-or-later.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import numpy as np

FIXED_PX_PER_CM = {"1200_left_ticks": 148.2, "644x1088": 126.1, "853_gray": 166.8, "512_fixed": 77.8}
FIXED_TOL = 0.01          # relative agreement required between the measured ruler and the layout constant


@dataclass
class ScaleResult:
    layout: str
    px_per_cm: float
    measured_px_per_cm: float
    method: str
    n_ticks: int
    ok: bool
    note: str = ""

    def as_dict(self):
        return asdict(self)


def to_gray(img) -> np.ndarray:
    """PIL image or array -> 2-D uint8 array (first channel / luminance)."""
    a = np.asarray(img)
    if a.ndim == 3:
        a = a[..., :3].mean(2) if a.shape[2] >= 3 else a[..., 0]
    if a.dtype != np.uint8:
        a = a.astype(float)
        hi = np.percentile(a, 99.9) or 1.0
        a = np.clip(a / hi * 255.0, 0, 255)
    return a.astype(np.uint8)


def detect_layout(a: np.ndarray, is_png: bool) -> str:
    """Screen layout from image size and the position of the ruler."""
    h, w = a.shape
    if is_png:
        return "png_variable"
    if (h, w) == (644, 1088):
        return "644x1088"
    if h == 853:
        return "853_gray"
    if h in (512, 513) and w <= 470:
        return "512_fixed"
    if (h, w) == (800, 1200):
        # Telemed screenshots have a grey side panel with a vertical axis line on the left; Siemens has black margins
        left = a[100:700, 0:100]
        axis = (left > 120).sum(0).max()
        return "1200_left_ticks" if (np.median(a[420:600, 5:40]) >= 20 or axis > 500) else "1200_right_ticks"
    return "unknown"


def _runs(on: np.ndarray):
    """Start/end indices of consecutive True runs."""
    out, i, n = [], 0, len(on)
    while i < n:
        if on[i]:
            j = i
            while j + 1 < n and on[j + 1]:
                j += 1
            out.append((i, j))
            i = j + 1
        else:
            i += 1
    return out


def ticks_vertical(a: np.ndarray, x0: int, x1: int, y0: int, y1: int, thr: float, min_len: int = 1):
    """Horizontal tick marks stacked along a vertical ruler (columns x0..x1). Returns [(y_center, length)]."""
    sub = a[y0:y1, x0:x1].astype(float)
    m = sub > thr
    run = m.sum(1)
    res = []
    for i, j in _runs(run >= min_len):
        wts = run[i:j + 1].astype(float)
        res.append((y0 + float((np.arange(i, j + 1) * wts).sum() / wts.sum()), int(run[i:j + 1].max())))
    return res


def ticks_horizontal(a: np.ndarray, y0: int, y1: int, x0: int, x1: int, thr: float, min_len: int = 1):
    """Vertical tick marks along a horizontal ruler (rows y0..y1). Returns [(x_center, length)]."""
    res = ticks_vertical(a.T, y0, y1, x0, x1, thr, min_len)
    return res


def _regular_subset(pos: np.ndarray, step_guess: float | None = None, tol: float = 0.06):
    """Longest chain of ticks with (nearly) constant spacing - removes markers and anatomy touching the ruler."""
    pos = np.sort(np.asarray(pos, float))
    if len(pos) < 3:
        return pos
    best = pos[:0]
    diffs = np.diff(pos)
    cands = [step_guess] if step_guess else list(np.unique(np.round(diffs, 0)))
    for step in cands:
        if step is None or step < 5:
            continue
        for s in range(len(pos)):
            chain = [pos[s]]
            for p in pos[s + 1:]:
                k = (p - chain[-1]) / step
                if abs(k - round(k)) <= tol and round(k) >= 1 and round(k) <= 2:
                    if round(k) == 1:
                        chain.append(p)
                    else:            # one tick hidden (e.g. focus marker) - accept the gap
                        chain.append(p)
            if len(chain) > len(best):
                best = np.array(chain)
    return best


def lattice_fit(pos, step_guess: float, tol: float = 0.04):
    """Ticks consistent with a regular lattice of spacing ~step_guess; returns (matched positions, fitted step).
    The guess only selects which marks belong to the ruler; the returned step is measured from the image."""
    pos = np.sort(np.asarray(pos, float))
    best, best_k = pos[:0], np.array([], int)
    for anchor in pos:
        k = np.round((pos - anchor) / step_guess)
        res = pos - (anchor + k * step_guess)
        m = np.abs(res) <= tol * step_guess
        # one tick per lattice index
        kk, idx = np.unique(k[m], return_index=True)
        if len(kk) > len(best):
            best, best_k = pos[m][idx], kk.astype(int)
    if len(best) < 3:
        return best, float("nan")
    step = float(np.polyfit(best_k, best, 1)[0])
    return best, step


def read_scale(img, is_png: bool) -> ScaleResult:
    a = to_gray(img)
    lay = detect_layout(a, is_png)
    h, w = a.shape
    if lay == "1200_right_ticks":
        # tick column: columns right of the image with short periodic bright marks (skip the grey bar further right)
        cols = (a[60:h - 30, 1080:1185] > 50).sum(0)
        xs = np.where((cols >= 8) & (cols <= 60))[0]
        if len(xs) == 0:
            return ScaleResult(lay, float("nan"), float("nan"), "right_ticks", 0, False, "no tick column")
        x0, x1 = 1080 + int(xs.min()), 1080 + int(xs.max()) + 1
        t = ticks_vertical(a, x0, x1, 40, h - 20, thr=50, min_len=3)
        pos = np.array([p for p, _ in t]); lens = np.array([l for _, l in t])
        if len(pos) < 4:
            return ScaleResult(lay, float("nan"), float("nan"), "right_ticks", len(pos), False, "too few ticks")
        minor = float(np.median(np.diff(pos)))
        span = float(pos[-1] - pos[0])
        n_int = int(round(span / minor))
        short = float(np.min(lens))          # minor tick length (long ticks are >= 2 px longer)
        longs = pos[lens >= short + 2]
        if len(longs) >= 3:
            r = float(np.median(np.diff(longs))) / minor
        else:
            r = 5.0
        per_cm = 2 if r < 3.5 else 5          # minor ticks every 0.5 cm (deep settings) or every 0.2 cm (shallow)
        depth = n_int / per_cm
        px = span / depth
        ok = bool(depth >= 1 and abs(span / n_int - minor) / minor < 0.05)
        return ScaleResult(lay, px, px, "right_ticks_span_over_depth", len(pos), ok,
                           f"span={span:.1f} depth={depth} minor={minor:.2f} ticks_per_cm={per_cm}")
    if lay == "png_variable":
        t = ticks_vertical(a, 0, 12, 0, h, thr=100, min_len=1)
        pos = np.array([p for p, _ in t]); lens = np.array([l for _, l in t])
        longs = pos[lens >= 5]
        if len(longs) < 2:
            return ScaleResult(lay, float("nan"), float("nan"), "png_long_ticks", len(pos), False, "too few long ticks")
        d = np.diff(longs)
        step = float(np.median(d))
        chain = _regular_subset(longs, step_guess=step, tol=0.05)
        px = float(chain[1] - chain[0]) if len(chain) >= 2 else step
        return ScaleResult(lay, px, step, "png_first_cm_interval", len(pos), True, f"long_ticks={len(longs)}")
    if lay in ("1200_left_ticks", "644x1088"):
        # vertical axis line at the left of the grey panel; ticks to its right every 10 mm, 0..50 mm
        prof = (a[:, :120] > 120).sum(0)
        xa = int(np.argmax(prof))
        t = ticks_vertical(a, xa + 1, xa + 6, 0, h, thr=120, min_len=2)
        chain, step = lattice_fit([p for p, _ in t], FIXED_PX_PER_CM[lay])
        measured = float((chain[-1] - chain[0]) / round((chain[-1] - chain[0]) / step)) if len(chain) >= 3 else float("nan")
    elif lay in ("853_gray", "512_fixed"):
        # bottom border line with ticks 1 cm apart just above it
        rows = (a > 150).sum(1)
        yb = int(np.argmax(rows[h - 15:])) + h - 15
        t = ticks_horizontal(a, yb - 5, yb, 0, w, thr=150, min_len=3)
        chain, step = lattice_fit([p for p, _ in t], FIXED_PX_PER_CM[lay])
        measured = float((chain[-1] - chain[0]) / round((chain[-1] - chain[0]) / step)) if len(chain) >= 3 else float("nan")
    else:
        return ScaleResult(lay, float("nan"), float("nan"), "unknown_layout", 0, False, f"size {w}x{h}")
    const = FIXED_PX_PER_CM[lay]
    agree = np.isfinite(measured) and abs(measured / const - 1) <= FIXED_TOL
    px = const if agree else measured
    return ScaleResult(lay, px, measured, "fixed_layout_verified" if agree else "measured_unverified",
                       len(chain), bool(agree), f"const={const}")
