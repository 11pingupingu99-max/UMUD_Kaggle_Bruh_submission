"""Staged, resumable neural inference. Only one model family is resident on the GPU.

Each image is read again from disk. No test predictions, original R96 payload or image-ID
registry is opened anywhere in this module. All model loaders use weights_only=True.
"""
from __future__ import annotations
import gc
import time
from pathlib import Path
import numpy as np
import pandas as pd
from .common import activate_vendor, atomic_json


def run_stages(meta, registry, cache, map_dir, device="cuda", log=print):
    activate_vendor()
    import torch
    import r45_core as C
    import r45_models as M
    import r45_decode as D
    import r71_geom as G
    import r71_train
    import r76_train
    import r76_export as A
    import r86_train
    import r96_fold
    import r93_recover as R
    import r50_ens
    import umud_core as UC
    import umud_model as UM

    rows = meta.to_dict("records"); maps = Path(map_dir); maps.mkdir(parents=True, exist_ok=True)
    def path(row):
        return maps / (cache.key("apo", row) + ".npz")
    def free():
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    def cached(stage):
        return [cache.get(stage, r) for r in rows]
    def execute(stage, todo, fn):
        errors = []; t0 = time.time()
        log(f"{stage}: {len(todo)}/{len(rows)} to compute", flush=True)
        for k, row in enumerate(todo, 1):
            try:
                cache.put(stage, row, fn(row))
            except Exception as e:
                if isinstance(e, torch.cuda.OutOfMemoryError):
                    raise
                errors.append({"image_id": row["image_id"], "error": f"{type(e).__name__}: {e}"})
                log(f"ERROR {stage}: {row['image_id']}: {e}", flush=True)
            if k % 10 == 0 or k == len(todo):
                log(f"{stage} {k}/{len(todo)}: {(time.time()-t0)/60:.1f} min", flush=True)
        if errors:
            atomic_json(cache.root / f"ERRORS_{stage}.json", errors)
            raise RuntimeError(f"{stage}: {len(errors)} errors, checkpoints saved; fix and rerun. No submission produced.")

    # Apo probabilities must exist for any pending R17/R22/F71 decoder stage.
    todo = [r for r in rows if cache.get("apo", r) is None or
            (not path(r).is_file() and any(cache.get(s, r) is None for s in ("r17", "r22", "f71")))]
    if todo:
        p12_model = M.load_r12b(registry["r12b"][0]["path"], device)
        v2_models = [M.load_apo2(d["path"], device)[0] for d in registry["v2"]]
        a_models = [r76_train.load_model(d["path"], device)[0] for d in registry["a76"]]
        sides = {d["side"] for d in registry["a76"]}
        if len(sides) != 1:
            raise ValueError("Mixed A76 resolutions")
        a_side = sides.pop()
        def apo_one(row):
            gray = C.read_gray(row["path"])
            p12 = M.predict_r12b(p12_model, row["path"], device)
            v2 = C.restore_map(M.predict_apo2(v2_models, C.letterbox_image(gray, 768)[0], device, True), gray.shape)
            raw = C.restore_map(r76_train.predict(a_models, C.letterbox_image(gray, a_side)[0], "apo", device, True), gray.shape)
            safe, guard = A.guarded_probability(raw, p12)
            same = A.same_pair(D.apo_pair_subpixel(v2), D.apo_pair_subpixel(p12))
            mix = .5 * (safe + v2) if same else safe
            mix, mix_guard = A.guarded_probability(mix, p12)
            mp = dict(r12b=p12, v2=v2, a76_safe=safe, a76_v2mix=mix)
            tmp = path(row).with_suffix(".tmp.npz")
            np.savez_compressed(tmp, **mp); tmp.replace(path(row))
            # R12B MT from original two-largest-component function, not the fitted R20 pair.
            t = R.thickness_from_mask((p12 > .5).astype(np.uint8))
            mt = dict(t)
            mt.update({k + "_mm": t[k] / row["px_per_mm"] for k in R.MT_DEFINITIONS})
            mt["mt_r12c_mm"] = float(R.mt_r12b_mm(t["vertical_mean"], row["px_per_mm"], R.R12C_BIAS_MM))
            return {"apo": {k: G.apo_pair_record(p) for k, p in mp.items()}, "mt": mt, **guard,
                    "mix_guard": mix_guard, "h": gray.shape[0], "w": gray.shape[1]}
        execute("apo", todo, apo_one)
        del apo_one, p12_model, v2_models, a_models; free()

    todo = [r for r in rows if cache.get("r17", r) is None]
    if todo:
        models = {d["name"]: UM.load_model(d["path"], device)[0] for d in registry["r17"]}
        specs = {d["name"]: d for d in registry["r17"]}
        def r17_one(row):
            gray = UC.read_gray(row["path"])
            with np.load(path(row), allow_pickle=False) as z:
                ap = z["r12b"]
            measurements = {}; wide = {}; recovery = None
            for name, model in models.items():
                plain, flipped = r50_ens.fasc_probs(model, gray, 768, device)
                threshold = specs[name]["threshold"]
                a = UC.measure_geometry(ap, plain, row["px_per_mm"], apo_thr=.5, fasc_thr=threshold)
                b = UC.measure_geometry(ap, flipped, row["px_per_mm"], apo_thr=.5, fasc_thr=threshold)
                measurements[name] = a
                for view, val in (("plain", a), ("flip", b)):
                    wide.update({f"{name}_{view}_{k}": val.get(k) for k in ("pa_deg", "fl_mm", "mt_mm", "status")})
                if name == "refit":
                    if abs(threshold - .65) > 1e-10:
                        raise ValueError("R17 refit threshold changed; R20 recovery requires .65")
                    recovery = R.process(row, ap, plain)
            return {"r17": measurements, "r50": wide, "r20": recovery["r20"], "r20two": recovery["r20two"] or {}}
        execute("r17", todo, r17_one)
        del r17_one, models, specs; free()

    for family, loader in (("r22", M.load_r22), ("f71", r71_train.load_f71), ("f86", r86_train.load_f86)):
        todo = [r for r in rows if cache.get(family, r) is None]
        if not todo:
            log(f"{family}: all {len(rows)} cached", flush=True)
            continue
        entries = registry[family]; models = [loader(d["path"], device)[0] for d in entries]
        sides = {d["side"] for d in entries}
        if len(sides) != 1:
            raise ValueError(f"Mixed input resolutions: {family}")
        side = sides.pop(); folds = [d["fold"] for d in entries]
        def fasc_one(row):
            gray = C.read_gray(row["path"]); hw = gray.shape
            square = C.letterbox_image(gray, side)[0]
            if family in ("r22", "f71"):
                views = r96_fold.view_maps(models, square, device)
                keep = [j for j, k in enumerate(folds) if family == "r22" or k in (0, 2, 3, 4)]
                # FL F71 uses E4. PA from D5 uses ALL FIVE F71 folds, including fold 1.
                fr = r96_fold.per_fold_frags([views[j] for j in keep], [folds[j] for j in keep], family, hw)
                avg = r96_fold.mean_maps([z for v in views for z in v])
                del views
            else:
                avg = M.predict_r22(models, square, device, True)
                fr = {"f86avg": r96_fold.decode(avg, hw)}
            nat = tuple(C.restore_map(a, hw) for a in avg)
            if family == "f86":
                return {"fragments": {k: v["0.60"] for k, v in fr.items()}}
            with np.load(path(row), allow_pickle=False) as z:
                ap = {k: z[k] for k in z.files}
            dec = D.decode_image(*nat, ap, row["px_per_mm"], with_streamlines=False)
            return {"decoders": {family + "_" + k: v for k, v in dec.items()},
                    "fragments": {k: v["0.60"] for k, v in fr.items()}}
        execute(family, todo, fasc_one)
        del fasc_one, models; free()
    return assemble(meta, cache)


def assemble(meta, cache):
    """Turn this run's cache into builder inputs, including empty fragment lists."""
    rows = meta.to_dict("records"); ap = []; mt = []; r20 = []; two = []; dec = []; feat = []; r50 = []
    r17 = {k: [] for k in ("develop", "refit", "seed1", "seed2")}; frags = {}
    for row in rows:
        iid = row["image_id"]; a = cache.get("apo", row); b = cache.get("r17", row)
        late = {f: cache.get(f, row) for f in ("r22", "f71", "f86")}
        if a is None or b is None or any(v is None for v in late.values()):
            raise RuntimeError(f"Incomplete inference cache for {iid}")
        ap.extend({"image_id": iid, "cfg": cfg, **v} for cfg, v in a["apo"].items() if v is not None)
        mt.append({"image_id": iid, **a["mt"]})
        r20.append({"image_id": iid, **b["r20"]}); two.append({"image_id": iid, **b["r20two"]})
        r50.append({"image_id": iid, **b["r50"]})
        for k in r17:
            r17[k].append({"image_id": iid, **b["r17"][k]})
        dd = {"image_id": iid, **late["r22"]["decoders"], **late["f71"]["decoders"]}
        dec.append(dd); feat.append({**dd, "guard": a["guard"], "px_per_mm": row["px_per_mm"], "h": row["h"], "w": row["w"]})
        for v in late.values():
            for tag, fs in v["fragments"].items():
                frags.setdefault(tag, []).extend({"image_id": iid, "frag_id": j, **f} for j, f in enumerate(fs))
    # Empty tables still carry their schema; no fabricated fragment is inserted.
    frags = {k: pd.DataFrame(v, columns=["image_id", "frag_id", "x0", "x1", "a", "b", "angle_fit", "length",
                                       "residual", "n", "angle_or", "or_R", "p_mean", "p_q10"]) for k, v in frags.items()}
    return {"dec": pd.DataFrame(dec), "apo": pd.DataFrame(ap), "feat": pd.DataFrame(feat), "fragments": frags,
            "r20": pd.DataFrame(r20), "r20two": pd.DataFrame(two), "mt": pd.DataFrame(mt),
            "r17": {k: pd.DataFrame(v) for k, v in r17.items()}, "r50": pd.DataFrame(r50)}
