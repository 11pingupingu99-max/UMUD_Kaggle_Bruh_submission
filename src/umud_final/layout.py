"""Image source (device layout), group, sub-group and video blocks - all from the images themselves.

Groups used by the token pipeline: sie (Siemens), telr (Telemed 1200x800), teln (Telemed crop 1088x644),
gry (raw Telemed frames: 853 px high -> gry_a, ~465x513 -> gry_b), png (Philips Lumify).
Licence: GPL-3.0-or-later."""
from __future__ import annotations

import numpy as np

LAYOUT_TO_GROUP = {"1200_right_ticks": "sie", "1200_left_ticks": "telr", "644x1088": "teln",
                   "853_gray": "gry", "512_fixed": "gry", "png_variable": "png"}
LAYOUT_TO_GRYSUB = {"853_gray": "gry_a", "512_fixed": "gry_b"}


def group_of(layout: str) -> str:
    return LAYOUT_TO_GROUP.get(layout, "?")


def grysub_of(layout: str) -> str:
    return LAYOUT_TO_GRYSUB.get(layout, "")


def _thumb(a: np.ndarray, size=(96, 64)) -> np.ndarray:
    """Central B-mode region, block-averaged to a small thumbnail, zero mean / unit norm."""
    h, w = a.shape
    c = a[int(.12 * h):int(.88 * h), int(.15 * w):int(.85 * w)].astype(float)
    ys = np.linspace(0, c.shape[0], size[1] + 1).astype(int); xs = np.linspace(0, c.shape[1], size[0] + 1).astype(int)
    t = np.array([[c[ys[i]:ys[i + 1], xs[j]:xs[j + 1]].mean() for j in range(size[0])] for i in range(size[1])])
    t -= t.mean(); n = np.linalg.norm(t)
    return t / n if n > 0 else t


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    ta, tb = _thumb(a), _thumb(b)
    return float((ta * tb).sum()) if ta.shape == tb.shape else -1.0


def blocks_from_images(numbers, layouts, loader, thr: float = 0.90, min_len: int = 3):
    """Video blocks: consecutive file numbers with the same layout whose neighbouring frames are near-identical
    (normalised cross-correlation of thumbnails >= thr). Returns block id per image (-1 = no block) and the NCC list.
    `loader(k)` returns the 2-D grey image of the k-th entry."""
    numbers = np.asarray(numbers); order = np.argsort(numbers, kind="stable")
    bid = np.full(len(numbers), -1, int); cur = 0; run = [order[0]]; sims = {}
    cache = {}
    def img(k):
        if k not in cache:
            cache.clear(); cache[k] = loader(k)
        return cache[k]
    prev = img(order[0])
    for a, b in zip(order[:-1], order[1:]):
        cur_img = loader(b)
        same = numbers[b] == numbers[a] + 1 and layouts[b] == layouts[a]
        s = ncc(prev, cur_img) if same else -1.0
        sims[(int(a), int(b))] = s
        if same and s >= thr:
            run.append(b)
        else:
            if len(run) >= min_len:
                bid[run] = cur; cur += 1
            run = [b]
        prev = cur_img
    if len(run) >= min_len:
        bid[run] = cur
    return bid, sims
