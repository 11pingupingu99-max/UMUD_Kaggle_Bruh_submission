"""Portable I/O for the frozen C1 inference notebooks. No model or prediction tables."""
from __future__ import annotations
import csv
import hashlib
import json
import math
import os
import shutil
from pathlib import Path

SOURCE_SHA256 = "af622512b9883b87aa1fe95f026ec882f95791266467abbc4acf49e6f93771c9"
PRIOR_PUBLIC_SCORES = {"C1_F1": 0.20486, "C1_F3": 0.20494}
IMAGE_SUFFIXES = {".tif", ".tiff", ".png"}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def validate_profile(value):
    if value not in PRIOR_PUBLIC_SCORES:
        raise ValueError("PROFILE must be C1_F1 or C1_F3")
    return value


def frozen_source_hash(root):
    root = Path(root)
    files = list((root / "src").rglob("*.py"))
    files += [root / p for p in ("tools/run_fresh.py", "assets/png_templates.json", "final/VARIANTS.json")]
    values = {str(p.relative_to(root)): sha256(p) for p in sorted(files)}
    return hashlib.sha256(json.dumps(values, sort_keys=True, allow_nan=False).encode()).hexdigest()


def expected_ids(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "image_id" not in reader.fieldnames:
            raise ValueError(f"Missing image_id column: {path}")
        ids = [r["image_id"] for r in reader]
    if not ids or any(not s for s in ids) or len(set(ids)) != len(ids):
        raise ValueError(f"Empty or duplicate IDs: {path}")
    return ids


def find_image_dir(input_root, explicit=None):
    if explicit:
        p = Path(explicit).resolve()
        if not p.is_dir():
            raise FileNotFoundError(f"IMAGE_DIR does not exist: {p}")
        return p
    root = Path(input_root)
    names = {"competition_test_v1", "test_images", "test", "testimages"}
    candidates = [p for p in root.rglob("*") if p.is_dir() and p.name.lower() in names
                  and any(q.is_file() and q.suffix.lower() in IMAGE_SUFFIXES for q in p.rglob("*"))]
    # Prefer the narrowest directory if an outer test directory contains a named inner one.
    candidates = [p for p in candidates if not any(q != p and q.is_relative_to(p) for q in candidates)]
    if len(candidates) != 1:
        raise ValueError(f"Set IMAGE_DIR explicitly; found {len(candidates)} test directories: {candidates}")
    return candidates[0].resolve()


def find_weights_dir(input_root, manifest, explicit=None):
    rels = [d["path"] for entries in manifest["models"].values() for d in entries]
    def complete(p):
        return all((p / rel).is_file() for rel in rels)
    if explicit:
        p = Path(explicit).resolve()
        missing = [s for s in rels if not (p / s).is_file()]
        if missing:
            raise FileNotFoundError(f"Missing {len(missing)} checkpoint files in WEIGHTS_DIR; first: {missing[0]}")
        return p
    candidates = sorted({p.parent.resolve() for p in Path(input_root).rglob("WEIGHTS_MANIFEST.json") if complete(p.parent)})
    if len(candidates) != 1:
        raise ValueError("Add the unpacked UMUD_C1_WEIGHTS Dataset as Input or set WEIGHTS_DIR. "
                         f"Found {len(candidates)} complete packages. Prepare one with UMUD_WAGI_FINAL_COLAB.ipynb.")
    return candidates[0]


def select_submission(run_dir, profile, destination, sample_csv=None):
    """Validate this run's completion, source, CSV hash and IDs, then copy exact bytes."""
    profile = validate_profile(profile)
    run = Path(run_dir)
    done = json.loads((run / "DONE.json").read_text())
    if done.get("source_sha256") != SOURCE_SHA256:
        raise ValueError("Inference source differs from the frozen C1 run")
    source = run / f"submission_{profile}.csv"
    if sha256(source) != done["files"].get(source.name):
        raise ValueError("Prediction file differs from this run's DONE.json")
    with source.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ["image_id", "pa_deg", "fl_mm", "mt_mm"]:
            raise ValueError("Wrong submission columns")
        rows = list(reader)
    ids = [r["image_id"] for r in rows]
    if not rows or len(set(ids)) != len(ids) or any(not s for s in ids):
        raise ValueError("Empty or duplicate prediction IDs")
    for r in rows:
        if None in r or any(not math.isfinite(float(r[k])) or float(r[k]) <= 0 for k in ("pa_deg", "fl_mm", "mt_mm")):
            raise ValueError(f"Invalid prediction for {r['image_id']}")
    if len(rows) != done["images"] or set(ids) != set(expected_ids(run / "IMAGES_FROM_PIXELS.csv")):
        raise ValueError("Prediction IDs differ from supplied images")
    if sample_csv and set(ids) != set(expected_ids(sample_csv)):
        raise ValueError("Prediction IDs differ from sample_submission IDs; verify IMAGE_DIR")
    dest = Path(destination)
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp = dest.with_suffix(".csv.partial")
    shutil.copyfile(source, temp)
    os.replace(temp, dest)
    receipt = {"profile": profile, "rows": len(rows), "submission_sha256": sha256(dest),
               "source_sha256": done["source_sha256"], "inference_completed_utc": done.get("time_utc"),
               "prior_public_score_reported_by_user": PRIOR_PUBLIC_SCORES[profile],
               "new_run_leaderboard_score": None, "submitted_to_kaggle": False,
               "scope": "CSV from the current completed image-inference run; no host labels used."}
    (dest.parent / "FINAL_RUN_RECEIPT.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt
