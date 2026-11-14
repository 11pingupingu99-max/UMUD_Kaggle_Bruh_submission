"""python tools/run_fresh.py --config config.json [--smoke N | --verify-only]

One new image collection -> fresh neural inference -> computed tables -> C1_F1/C1_F3.
This is a reconstructed candidate pipeline, not a certified byte-identical F1/F3 replacement.
"""
from __future__ import annotations
import argparse
import datetime as dt
import importlib.metadata
import json
import platform
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from umud_fresh import VERSION
from umud_fresh.common import Cache, atomic_json, clean, digest, sha256, source_hash
from umud_fresh.metadata import read_images
from umud_fresh.registry import discover, verify_and_stage
from umud_fresh.infer import run_stages
from umud_fresh.builders import build_inputs
from umud_fresh.finalize import run_math, compare


def export_computed(out, inputs, audit, calibration, detail, raw):
    audit.to_csv(out / "BASE_COMPUTED.csv", index=False)
    detail.to_csv(out / "FX_COMPUTED.csv", index=False)
    atomic_json(out / "CALIBRATION_COMPUTED.json", calibration)
    atomic_json(out / "TABLES_COMPUTED.json", {k: (v.to_dict("records") if isinstance(v, pd.DataFrame) else v)
                                             for k, v in inputs.items()})
    exports = out / "exports"; exports.mkdir(exist_ok=True)
    for k in ("dec", "apo", "feat", "r20", "r20two", "mt", "r50"):
        raw[k].to_csv(exports / f"{k}.csv.gz", index=False)
    for k, d in raw["r17"].items():
        d.to_csv(exports / f"r17_{k}.csv.gz", index=False)
    for k, d in raw["fragments"].items():
        d.to_csv(exports / f"fragments_{k}.csv.gz", index=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--smoke", type=int, default=0, help="Compute first N images; no submissions")
    p.add_argument("--verify-only", action="store_true")
    args = p.parse_args(); cfg = json.loads(Path(args.config).read_text())
    out = Path(cfg["output_dir"]); out.mkdir(parents=True, exist_ok=True)
    logfile = (out / "RUN.log").open("a", buffering=1)
    def log(*a, **kw):
        s = " ".join(map(str, a)); print(s, flush=True); print(s, file=logfile, flush=True)
    log(VERSION, dt.datetime.now(dt.timezone.utc).isoformat())
    log("Original Kaggle final selections remain F1 + F3. C1 needs numerical review before use.")
    meta = read_images(cfg["image_dir"], cfg.get("manifest"), ROOT / "assets/png_templates.json")
    if args.smoke:
        if args.smoke < 1:
            raise ValueError("--smoke must be positive")
        chosen = meta.drop_duplicates("ruler_layout").head(args.smoke).index.tolist()
        if len(chosen) < args.smoke:
            chosen += [i for i in meta.index if i not in chosen][:args.smoke - len(chosen)]
        meta = meta.loc[chosen].reset_index(drop=True)
    meta.to_csv(out / ("IMAGES_SMOKE.csv" if args.smoke else "IMAGES_FROM_PIXELS.csv"), index=False)
    log("images", len(meta), "groups", meta.group.value_counts().to_dict(), "scale", meta.scale_source.value_counts().to_dict())
    original_registry = (json.loads(Path(cfg["registry"]).read_text()) if cfg.get("registry")
                         else discover(cfg["project_dir"], cfg["drive_root"]))
    atomic_json(out / "MODELS_ORIGINAL.json", original_registry)
    registry = verify_and_stage(original_registry, cfg["local_weights"], log)
    atomic_json(out / "MODELS_VERIFIED.json", registry)
    if args.verify_only:
        log("Preflight completed: images, rulers and checkpoint hashes verified. No inference performed.")
        return
    import torch
    device = cfg.get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("No GPU. Select a GPU runtime or explicitly set device=cpu (slow).")
    torch.set_num_threads(int(cfg.get("cpu_threads", 2)))
    # Match historical inference; no TF32 change hidden in the rebuilt driver.
    if device == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True
    packages = ("numpy", "pandas", "scipy", "scikit-image", "Pillow", "torch", "torchvision", "timm")
    environment = {k: importlib.metadata.version(k) for k in packages}
    import cv2
    environment["opencv"] = cv2.__version__
    sig = {"version": VERSION, "source_sha256": source_hash(), "environment": environment,
           "models": {k: [{a: b for a, b in d.items() if a not in ("path", "original_path")} for d in v]
                      for k, v in registry.items()},
           "device": device, "torch": torch.__version__, "cuda": torch.version.cuda,
           "gpu": torch.cuda.get_device_name() if device == "cuda" else "cpu", "cpu_threads": torch.get_num_threads(),
           "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32), "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32)}
    # Cache invalidates on changes in image pixels, scale, model hashes, code or inference device.
    cache = Cache(out / "cache", sig)
    atomic_json(out / "RUN_SIGNATURE.json", sig)
    raw = run_stages(meta, registry, cache, cfg["map_cache"], device=device, log=log)
    if args.smoke:
        atomic_json(out / "SMOKE_DONE.json", {"n": len(meta), "signature": sig, "computed_stages": ["apo", "r17", "r22", "f71", "f86"],
                                               "note": "No submission or final score claimed. Per-image caches are reusable by the full run."})
        log("SMOKE completed; run without --smoke for the full batch.")
        return
    inputs, audit, calibration, detail = build_inputs(meta, **raw)
    export_computed(out, inputs, audit, calibration, detail, raw)
    profiles = json.loads((ROOT / "final/VARIANTS.json").read_text())
    reports = {}; files = {}
    for name in ("F1", "F3"):
        pred, txt = run_math(inputs, profiles[name]["variant"])
        dest = out / f"submission_C1_{name}.csv"
        pred.to_csv(dest, index=False)
        (out / f"TOKENS_{name}.log").write_text(txt)
        files[dest.name] = sha256(dest)
        reference = ROOT / f"reference_checks/R96_{name}_CONTROL.csv"
        if reference.exists():
            tab, joined = compare(reference, pred)
            tab.to_csv(out / f"COMPARE_{name}.csv", index=False)
            joined.to_csv(out / f"COMPARE_PER_IMAGE_{name}.csv", index=False)
            reports[name] = tab.to_dict("records")
            log(name, "prediction differences:", reports[name])
    atomic_json(out / "DONE.json", {"status": "FRESH_INFERENCE_COMPLETED_REVIEW_NUMERICAL_DIFFERENCES",
               "time_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "images": len(meta), "version": VERSION,
               "image_batch_sha256": digest(meta[["image_id", "image_sha256", "px_per_mm"]].to_dict("records")),
               "source_sha256": sig["source_sha256"], "signature": sig, "files": files, "comparisons": reports,
               "environment": environment, "python": platform.python_version(),
               "scope": "All predictions in this run computed from supplied images and verified own checkpoints. This is a reconstructed C1 pipeline; equality to R96 is checked, not assumed."})
    log("Completed. Review COMPARE_F1/F3 and BASE_COMPUTED before using the candidate CSVs.")


if __name__ == "__main__":
    main()
