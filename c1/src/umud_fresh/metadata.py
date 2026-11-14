"""Read scale and device layout from every supplied image; no image-ID registry."""
from __future__ import annotations
import base64
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from embedded import scale, png_muscle
from umud_final.layout import group_of, grysub_of
from .common import sha256


def read_images(image_dir, manifest=None, templates=None):
    root = Path(image_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Image directory: {root}")
    if manifest:
        frame = pd.read_csv(manifest, dtype={"image_id": str})
        if "image_id" not in frame:
            raise ValueError("Manifest requires image_id; optional: path, px_per_mm, frame_index, clip_id")
        if "path" not in frame:
            frame["path"] = frame.image_id
    else:
        ps = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in (".tif", ".tiff", ".png"))
        frame = pd.DataFrame({"image_id": [p.name for p in ps], "path": [str(p) for p in ps]})
    if len(frame) == 0 or frame.image_id.duplicated().any():
        raise ValueError("Empty image collection or duplicate image_id; supply an explicit manifest for duplicate names")
    T = None
    if templates:
        d = json.loads(Path(templates).read_text())
        raw = base64.b64decode(d["b64"])
        import hashlib
        if d.get("sha256") and hashlib.sha256(raw).hexdigest() != d["sha256"]:
            raise ValueError("Corrupt fixed glyph templates")
        arr = np.frombuffer(raw, np.float64).reshape(d["shape"])
        T = dict(zip(d["keys"], arr))
    rows = []
    for j, r in enumerate(frame.to_dict("records")):
        p = Path(str(r["path"])); p = p if p.is_absolute() else root / p
        with Image.open(p) as im:
            a = scale.to_gray(im); a_ocr = np.asarray(im.convert("L")); h, w = a.shape
            is_png = im.format == "PNG"
            result = scale.read_scale(im, is_png=is_png)
        d = result.as_dict(); layout = d["layout"]; group = group_of(layout)
        supplied_group = r.get("group")
        if supplied_group is not None and pd.notna(supplied_group):
            if supplied_group not in ("sie", "telr", "teln", "gry", "png"):
                raise ValueError(f"Unsupported manifest group {supplied_group}")
            if group != "?" and group != supplied_group:
                raise ValueError(f"Manifest group disagrees with image layout: {r['image_id']}")
            group = str(supplied_group)
        supplied = r.get("px_per_mm")
        if supplied is not None and pd.notna(supplied):
            ppm = float(supplied); scale_source = "explicit_manifest"
        elif d["ok"] and np.isfinite(d["px_per_cm"]) and d["px_per_cm"] > 0:
            ppm = float(d["px_per_cm"]) / 10.; scale_source = "image_ruler"
        else:
            raise ValueError(f"Unreadable scale for {r['image_id']}: {d}. Supply independently known px_per_mm in manifest.")
        if not np.isfinite(ppm) or ppm <= 0 or group == "?":
            raise ValueError(f"Unsupported layout or scale for {r['image_id']}; got {layout}, {ppm}")
        muscle = "?"
        if is_png:
            if T is None:
                raise ValueError("PNG input requires the packaged fixed glyph templates")
            muscle = png_muscle.classify(a_ocr.astype(float), T)[0]
        n = r.get("frame_index"); sequence_key = str(p.parent)
        if n is None or pd.isna(n):
            m = re.fullmatch(r"(.*?)(\d+)", p.stem)
            n = int(m.group(2)) if m else None
            sequence_key += "/" + m.group(1) if m else "/unordered"
        clip = r.get("clip_id")
        rows.append({"image_id": str(r["image_id"]), "path": str(p.resolve()), "image_sha256": sha256(p),
                     "h": h, "w": w, "group": group, "grysub": grysub_of(layout), "muscle": muscle,
                     "is_png": is_png, "frame_index": n, "sequence_key": sequence_key,
                     "clip_id": None if pd.isna(clip) else clip,
                     "px_per_mm": ppm, "scale_source": scale_source, **{f"ruler_{k}": v for k, v in d.items()}})
    return pd.DataFrame(rows)


def sequence_blocks(base, meta):
    """R96 neighbouring-frame rule, using manifest ordering when supplied. No fixed five-frame ID lists.

    Explicit clip_id takes precedence. Otherwise consecutive frame_index, same PNG/TIFF domain,
    |delta MT|<=0.10mm and |delta PA|<=3deg, minimum three frames. Unordered UUIDs remain independent.
    """
    n = len(base); bid = np.full(n, -1, int); cur = 0
    m = meta.set_index("image_id").loc[base.image_id].reset_index()
    explicit = m.clip_id.notna()
    for clip, g in m[explicit].groupby("clip_id", sort=False):
        ix = g.index.to_numpy()
        if len(ix) >= 3:
            bid[ix] = cur; cur += 1
    ix = m.index[~explicit & m.frame_index.notna()].to_numpy()
    ix = ix[np.argsort(m.loc[ix, "frame_index"].to_numpy(), kind="stable")]
    run = []
    def close(run):
        nonlocal cur
        if len(run) >= 3:
            bid[run] = cur; cur += 1
    for b in ix:
        if run:
            a = run[-1]
            same_key = "sequence_key" not in m or m.sequence_key.iloc[b] == m.sequence_key.iloc[a]
            same = (same_key and m.frame_index.iloc[b] == m.frame_index.iloc[a] + 1 and m.is_png.iloc[b] == m.is_png.iloc[a]
                    and abs(base.mt_mm.iloc[b] - base.mt_mm.iloc[a]) <= .10
                    and abs(base.pa_deg.iloc[b] - base.pa_deg.iloc[a]) <= 3.)
            if not same:
                close(run); run = []
        run.append(int(b))
    close(run)
    return bid
