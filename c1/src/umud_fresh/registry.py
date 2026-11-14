"""Verify the exact existing training outputs before loading any model."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path

R12B_SHA = "21ec34d0f5dd7664ecab0ca608f976fdf9d2a161e2af2873c21d513cd5ae19ef"
R17_REFIT_SHA = "6957c5228c1cf5c2067096d5f0214f6c2a7f54b0aeca03b8f664051022859d31"


def read(path):
    return json.loads(Path(path).read_text())


def spec(path, sha, **kw):
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Missing checkpoint {p}")
    if not isinstance(sha, str) or len(sha) != 64:
        raise ValueError(f"No recorded SHA-256 for {p}")
    return {"path": str(p), "sha256": sha, **kw}


def discover(project, drive_root):
    project = Path(project); out = project / "outputs"
    reg = {"r12b": [spec(Path(drive_root) / "UMUD_R12B/best_dinov2_frozen.pt", R12B_SHA)]}
    tc = read(out / "R22_v1_0_0/TRAINING_COMPLETE.json")
    reg["r22"] = [spec(out / f"R22_v1_0_0/fold_{d['fold']}/final.pt", d["checkpoint_sha256"],
                       fold=int(d["fold"]), side=768) for d in sorted(tc["checkpoints"], key=lambda d: d["fold"])]
    for key, folder, folds, side in (
        ("v2", "R45T_v1_0_0/apo2", range(5), 768),
        ("a76", "R76T_v1_0_0/A76", range(5), None),
        ("f71", "R71T_v1_0_0/T", range(5), 1024),
        ("f86", "R86T_v1_0_0/T", range(5), 1024),
    ):
        reg[key] = []
        for k in folds:
            d = read(out / folder / f"fold_{k}/DONE.json")
            s = d.get("side", side)
            if s is None:
                s = d.get("cfg", {}).get("side") or d.get("config", {}).get("side")
            if not s:
                raise ValueError(f"No recorded inference side for {key} fold {k}")
            if side is not None and int(s) != side:
                raise ValueError(f"Unexpected {key} input side {s}; expected {side}")
            reg[key].append(spec(out / folder / f"fold_{k}/final.pt", d["final_sha256"], fold=int(k), side=int(s)))
    if [d["fold"] for d in reg["r22"]] != list(range(5)):
        raise ValueError("Expected R22 folds 0..4 exactly")
    e4 = out / "R71T_v1_0_0/E4/E4_SUMMARY.json"
    if e4.exists() and sorted(read(e4)["folds"]) != [0, 2, 3, 4]:
        raise ValueError("F71 E4 membership changed; stop rather than silently change the final")
    reg["r17"] = []
    for name in ("develop", "refit"):
        d = read(out / f"R17_v1_0_0/{name}/selection.json")
        sha = d["checkpoint_sha256"]
        if name == "refit" and sha != R17_REFIT_SHA:
            raise ValueError("R17 refit checkpoint differs from the recorded final")
        reg["r17"].append(spec(out / f"R17_v1_0_0/{name}/best.pt", sha, name=name, threshold=float(d["threshold"]), side=768))
    for seed in (1, 2):
        folder = out / f"R50T_v1_0_0/seeds/seed_{seed}"
        d = read(folder / "DONE.json")
        reg["r17"].append(spec(folder / "refit/best.pt", d["checkpoint_sha256"], name=f"seed{seed}",
                               threshold=float(d["threshold"]), side=768))
    return reg


def verify_and_stage(registry, local_dir, log=print):
    """Read original files once, SHA-verify while copying to local SSD. Never downloads other weights."""
    local = Path(local_dir); local.mkdir(parents=True, exist_ok=True)
    reg = {}
    for family, entries in registry.items():
        reg[family] = []
        for d in entries:
            source = Path(d["path"]); dest = local / (d["sha256"] + ".pt")
            def checksum(p):
                h = hashlib.sha256()
                with p.open("rb") as f:
                    for b in iter(lambda: f.read(8 << 20), b""):
                        h.update(b)
                return h.hexdigest()
            if not dest.is_file() or checksum(dest) != d["sha256"]:
                tmp = dest.with_suffix(".tmp")
                h = hashlib.sha256()
                with source.open("rb") as f, tmp.open("wb") as g:
                    for b in iter(lambda: f.read(8 << 20), b""):
                        h.update(b); g.write(b)
                if h.hexdigest() != d["sha256"]:
                    tmp.unlink(missing_ok=True)
                    raise ValueError(f"Checkpoint SHA-256 mismatch: {source}")
                os.replace(tmp, dest)
            reg[family].append({**d, "original_path": str(source), "path": str(dest)})
        log(f"weights {family}: {len(entries)} verified", flush=True)
    return reg
