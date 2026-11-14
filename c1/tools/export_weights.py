"""Copy the exact 30 audited checkpoints into the portable public-package layout.

Run in Colab while Drive is mounted. No training, torch import, GPU or network upload.
python tools/export_weights.py --run-dir /content/drive/MyDrive/UMUD_Challenge/outputs/C1_FROM_IMAGES_20261113 --output-dir /content/drive/MyDrive/UMUD_C1_WEIGHTS
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def copy_verified(source, target, expected):
    if target.exists():
        if sha(target) != expected:
            raise FileExistsError(f"Existing target has a different checksum: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(target.suffix + ".partial")
    h = hashlib.sha256()
    try:
        with source.open("rb") as f, temp.open("wb") as g:
            for b in iter(lambda: f.read(8 << 20), b""):
                h.update(b); g.write(b)
        if h.hexdigest() != expected:
            raise ValueError(f"Checkpoint checksum mismatch: {source}")
        os.replace(temp, target)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--models-manifest", default=str(ROOT / "models/WEIGHTS_MANIFEST.json"))
    a = p.parse_args(); run = Path(a.run_dir); out = Path(a.output_dir)
    manifest = json.loads(Path(a.models_manifest).read_text())
    done = json.loads((run / "DONE.json").read_text())
    verified = json.loads((run / "MODELS_VERIFIED.json").read_text())
    original = json.loads((run / "MODELS_ORIGINAL.json").read_text())
    out.mkdir(parents=True, exist_ok=True)
    exported = []; n = 0
    for family, specs in manifest["models"].items():
        by_hash = {d["sha256"]: d for d in verified[family]}
        old_by_hash = {d["sha256"]: d for d in original[family]}
        recorded = {d["sha256"] for d in done["signature"]["models"][family]}
        for spec in specs:
            expected = spec["sha256"]
            if expected not in recorded or expected not in by_hash or expected not in old_by_hash:
                raise ValueError(f"Weights do not belong to the audited C1 run: {family}")
            entry = by_hash[expected]
            paths = [entry["path"], entry.get("original_path"), old_by_hash[expected]["path"]]
            source = next((Path(x) for x in paths if x and Path(x).is_file()), None)
            if source is None:
                raise FileNotFoundError(f"Missing source checkpoint for {family}: {paths}")
            rel = Path(spec["path"])
            if rel.is_absolute() or ".." in rel.parts:
                raise ValueError("Unsafe portable model path")
            target = out / rel
            copy_verified(source, target, expected)
            exported.append({"file": rel.as_posix(), "bytes": target.stat().st_size, "sha256": expected})
            n += 1; print(f"{n}/30 {rel}: verified", flush=True)
    if len(exported) != manifest["model_count"]:
        raise ValueError("Incomplete weight package")
    (out / "WEIGHTS_MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    (out / "model_registry.json").write_text(json.dumps(manifest["models"], indent=2))
    (out / "WEIGHTS_COMPLETE.json").write_text(json.dumps({"status": "ALL_CHECKPOINTS_BYTE_VERIFIED",
        "model_count": len(exported), "total_bytes": sum(d["bytes"] for d in exported), "files": exported,
        "upstream_inference_source_sha256": done["source_sha256"],
        "publication_status": "LOCAL_PACKAGE_NOT_UPLOADED"}, indent=2))
    (out / "README.md").write_text("# UMUD C1 trained checkpoints\n\n"
        "Exact checkpoints used in C1_FROM_IMAGES_20261113. See WEIGHTS_COMPLETE.json for hashes and sizes.\n"
        "Use the accompanying source repository's configure_inference.py with this folder as --weights-dir.\n"
        "Source-code license and checkpoint/data publication terms must be declared separately when uploading.\n"
        "This command prepares local files and does not publish a Dataset.\n")
    print("Completed:", out)


if __name__ == "__main__":
    main()
