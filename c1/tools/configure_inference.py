"""Create a portable C1 configuration without Google Drive folder assumptions.

python tools/configure_inference.py --images-dir /data/images --weights-dir /data/weights --output-dir /work/out
python tools/run_fresh.py --config /work/out/config_c1.json
"""
import argparse
import json
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]


def resolve_models(manifest, weights):
    out = {}
    for family, entries in manifest["models"].items():
        out[family] = []
        for rec in entries:
            rel = PurePosixPath(rec["path"])
            if rel.is_absolute() or ".." in rel.parts:
                raise ValueError(f"Unsafe relative model path: {rel}")
            out[family].append({**rec, "path": str((weights / str(rel)).resolve())})
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--images-dir", required=True)
    p.add_argument("--weights-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--image-manifest")
    p.add_argument("--models-manifest", default=str(ROOT / "models/WEIGHTS_MANIFEST.json"))
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--inventory", action="store_true", help="Report missing weights without creating a run configuration")
    a = p.parse_args(); weights = Path(a.weights_dir).resolve(); images = Path(a.images_dir).resolve()
    manifest = json.loads(Path(a.models_manifest).read_text())
    models = resolve_models(manifest, weights)
    missing = [d["path"] for entries in models.values() for d in entries if not Path(d["path"]).is_file()]
    if a.inventory:
        print(json.dumps({"model_count": sum(map(len, models.values())), "missing_count": len(missing),
                          "missing": missing, "images_directory_exists": images.is_dir()}, indent=2))
        return
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} checkpoint files; first: {missing[0]}")
    if not images.is_dir():
        raise FileNotFoundError(f"Missing image directory: {images}")
    out = Path(a.output_dir).resolve(); out.mkdir(parents=True, exist_ok=True)
    registry = out / "models_resolved.json"
    registry.write_text(json.dumps(models, indent=2))
    config = {"image_dir": str(images), "manifest": str(Path(a.image_manifest).resolve()) if a.image_manifest else None,
              "registry": str(registry), "output_dir": str(out), "local_weights": str(out / "cache/weights"),
              "map_cache": str(out / "cache/maps"), "device": a.device, "cpu_threads": 2}
    target = out / "config_c1.json"; target.write_text(json.dumps(config, indent=2))
    print(target)
    print("Checkpoint SHA-256 checks are performed by run_fresh.py before model loading.")


if __name__ == "__main__":
    main()
