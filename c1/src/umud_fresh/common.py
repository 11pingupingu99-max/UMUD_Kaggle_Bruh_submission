"""Content addressed cache and recovered source selection."""
from __future__ import annotations
import hashlib
import importlib
import json
import os
import sys
from pathlib import Path
import numpy as np

SRC = Path(__file__).resolve().parents[1]


def activate_vendor():
    # One canonical R45 core for all late decoders; historical R20/R17 kept exact.
    roots = [SRC / "vendor" / s for s in ("r76", "r93", "r50", "r86", "r96")]
    for p in reversed(roots):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    for name in ("r45_core", "r45_models", "r45_decode", "umud_core", "umud_model", "r71_train"):
        if name in sys.modules:
            p = Path(sys.modules[name].__file__).resolve()
            if not p.is_relative_to(SRC / "vendor"):
                raise RuntimeError(f"Conflicting imported module {name}: {p}. Restart the runtime.")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [clean(v) for v in x]
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return float(x) if np.isfinite(x) else None
    if isinstance(x, Path):
        return str(x)
    return x


def digest(x):
    return hashlib.sha256(json.dumps(clean(x), sort_keys=True, allow_nan=False).encode()).hexdigest()


def atomic_json(path, obj):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    q = p.with_suffix(p.suffix + ".tmp")
    q.write_text(json.dumps(clean(obj), ensure_ascii=False, indent=2, allow_nan=False))
    os.replace(q, p)


def source_hash():
    files = list(SRC.rglob("*.py"))
    for rel in ("tools/run_fresh.py", "assets/png_templates.json", "final/VARIANTS.json"):
        p = SRC.parent / rel
        if p.is_file():
            files.append(p)
    return digest({str(p.relative_to(SRC.parent)): sha256(p) for p in sorted(files)})


class Cache:
    """Cache key includes source, checkpoint hashes, image bytes and scale; names are output labels only."""
    def __init__(self, root, signature):
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True)
        self.signature = signature

    def key(self, stage, row):
        return digest({"run": self.signature, "stage": stage,
                       "image": row["image_sha256"], "ppm": row["px_per_mm"], "group": row["group"]})

    def path(self, stage, row):
        return self.root / stage / (self.key(stage, row) + ".json")

    def get(self, stage, row):
        p = self.path(stage, row)
        if not p.is_file():
            return None
        r = json.loads(p.read_text())
        if r.get("_key") != self.key(stage, row):
            return None
        # Same pixels under a new name get the new name, not a stale cached ID.
        r["image_id"] = row["image_id"]
        return r

    def put(self, stage, row, rec):
        rec = {**rec, "image_id": row["image_id"], "_key": self.key(stage, row)}
        atomic_json(self.path(stage, row), rec)
        return clean(rec)
