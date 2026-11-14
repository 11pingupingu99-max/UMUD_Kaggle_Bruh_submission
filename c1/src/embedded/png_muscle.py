"""Odczyt podpisu mięśnia na zrzutach PNG (Philips Lumify): dopasowanie wzorców słów gm/GM/vl/VL/rf/RF.
Kod przeniesiony bez zmian z notebooka (R44 -> R94); szablony zapisane raz jako tablice (nie zależą od plików testu).
SPDX-License-Identifier: GPL-3.0-or-later"""
import numpy as np

from scipy import ndimage as ndi
from PIL import Image
def block(a):
    cm = a[250:550].mean(0); xs = np.where(cm > 12)[0]; xs = xs[(xs > 150) & (xs < 1050)]
    x0, x1 = (int(xs.min()), int(xs.max())) if len(xs) else (300, 900)
    rm = a[:, x0 + 5:x1 - 5].mean(1); ys = np.where(rm > 8)[0]; ys = ys[(ys > 80) & (ys < 790)]
    y1 = int(ys.max()) if len(ys) else 720
    return x0, x1, y1

def words(a, thr=150):
    x0, x1, y1 = block(a)
    ya, yb = max(0, y1 - 115), min(a.shape[0], y1 + 35); xa, xb = max(0, x0 - 15), min(a.shape[1], x0 + 340)
    s = a[ya:yb, xa:xb]
    b = s > thr
    lab, n = ndi.label(b)
    comps = []
    for i, sl in enumerate(ndi.find_objects(lab)):
        h = sl[0].stop - sl[0].start; w = sl[1].stop - sl[1].start; area = int((lab[sl] == i + 1).sum())
        if 5 <= h <= 30 and w <= 45 and area >= 6:
            comps.append([sl[1].start, sl[1].stop, sl[0].start, sl[0].stop])
    comps.sort()
    ws = []
    for c in comps:
        if ws and c[0] - ws[-1][1] <= 5 and abs((c[2] + c[3]) / 2 - (ws[-1][2] + ws[-1][3]) / 2) < 12:
            w = ws[-1]; ws[-1] = [w[0], max(w[1], c[1]), min(w[2], c[2]), max(w[3], c[3])]
        else:
            ws.append(c)
    # tylko slowa wygladajace na tekst: wysokosc 10-26 px
    ws = [w for w in ws if 10 <= w[3] - w[2] <= 26 and w[1] - w[0] >= 8]
    return s, ws

def norm_word(s, w, H=24, W=64):
    x0, x1, y0, y1 = w
    crop = s[max(0, y0 - 2):y1 + 2, max(0, x0 - 2):x1 + 2]
    im = Image.fromarray(np.clip(crop, 0, 255).astype(np.uint8)).resize((max(4, int(crop.shape[1] * H / crop.shape[0])), H))
    z = np.zeros((H, W)); arr = np.asarray(im, float)[:, :W]; z[:, :arr.shape[1]] = arr
    z = z - z.mean(); n = np.linalg.norm(z)
    return z / n if n > 0 else z, (x1 - x0) / max(1, (y1 - y0))

def ncc(t, z):
    return float((t * z).sum())

TEMPLATE_SRC = {"gm": ("IMG_00275.png", 0), "GM": ("IMG_00265.png", 1), "vl": ("IMG_00252.png", 0),
                "VL": ("IMG_00290.png", 0), "rf": ("IMG_00256.png", 0), "RF": ("IMG_00291.png", 1)}
MUSCLE = {"gm": "gm", "GM": "gm", "vl": "vl", "VL": "vl", "rf": "rf", "RF": "rf"}

def build_templates(load):
    T = {}
    for k, (f, idx) in TEMPLATE_SRC.items():
        a = load(f); s, ws = words(a)
        T[k] = norm_word(s, ws[idx])[0]
    return T

def classify(a, T, thr=0.58, margin=0.08):
    s, ws = words(a)
    best = {}
    for w in ws:
        z, asp = norm_word(s, w)
        if asp > 2.2:          # "rechts" / "recht"
            continue
        for k, t in T.items():
            sc = ncc(t, z); m = MUSCLE[k]
            best[m] = max(best.get(m, -1), sc)
    if not best:
        return "?", best
    order = sorted(best.items(), key=lambda kv: -kv[1])
    top, sc = order[0]
    second = order[1][1] if len(order) > 1 else -1
    if sc < thr or sc - second < margin:
        return "?", best
    return top, best
