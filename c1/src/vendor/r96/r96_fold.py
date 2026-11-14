"""R96T: fascicle fragments of the EXISTING models R22 (5 folds) and F71 (folds E4) decoded SEPARATELY for every fold
(late fusion: each fold -> its own fragments and lex5 measurement, measurements combined afterwards), next to the control
= one decoding of the mean map over folds and flip views (exactly r45_models.predict_r22 over all models with flip TTA, i.e.
today's fragments). Out of fold (training set) only one fold model has not seen an image, so there the two views of the flip
TTA are decoded separately (no flip / flip) next to their mean - the same early-vs-late fusion question at the high-correlation
end (cross-family end: R89T `mix` vs `cmbd`). No training.
SPDX-License-Identifier: GPL-3.0-or-later
"""
import json
import time
from pathlib import Path
import numpy as np
import r45_core
import r45_models
import r71_geom
import r71_sim

THR = .6


def view_maps(models, square, device):
    """Per model: [(p, c, s) of the image, (p, c, s) of the mirrored image mapped back (sin2t negated)] at the square's side.
    The mean over all models and views equals r45_models.predict_r22(models, square, device, True)."""
    sq = np.ascontiguousarray(np.asarray(square, np.float32))
    mirrored = np.ascontiguousarray(sq[:, ::-1])
    out = []
    for m in models:
        a = r45_models.predict_r22([m], sq, device, False)
        b = r45_models.predict_r22([m], mirrored, device, False)
        b = (np.ascontiguousarray(b[0][:, ::-1]), np.ascontiguousarray(b[1][:, ::-1]), np.ascontiguousarray(-b[2][:, ::-1]))
        out.append([tuple(np.asarray(z, np.float32) for z in a), b])
    return out


def mean_maps(maps):
    n = float(len(maps))
    return tuple((sum(m[j] for m in maps) / n).astype(np.float32) for j in range(3))


def decode(maps, hw):
    """Square maps -> native -> fragments at threshold 0.6 (r71_geom.frag_lines = r45_core.fasc_lines + orientation field)."""
    nat = [r45_core.restore_map(z, hw) for z in maps]
    return {f'{THR:.2f}': r71_geom.frag_lines(*nat, THR, hw[1])}


def per_fold_frags(views, fold_ids, family, hw):
    """{family}k{fold}: each fold decoded on its own (mean of its two views); {family}avg: decoding of the mean over folds."""
    out, per = {}, []
    for k, v in zip(fold_ids, views):
        mp = mean_maps(v); per.append(mp)
        out[f'{family}k{k}'] = decode(mp, hw)
    out[f'{family}avg'] = decode(mean_maps(per), hw)
    return out


def per_view_frags(views_one_model, family, hw):
    """Out of fold: {family}nf (no flip), {family}fl (flip), {family}avg (their mean = today's single-model fragments)."""
    a, b = views_one_model
    return {f'{family}nf': decode(a, hw), f'{family}fl': decode(b, hw), f'{family}avg': decode(mean_maps([a, b]), hw)}


def run_images(name, rows, out_dir, fn, signature, log=print):
    """Resumable pass over images (JSONL cache, one line per image, signature hash - as r71_sim.run_images).
    fn(row, gray) -> dict of fields stored with the image (frags: {tag: {thr: [...]}}, radon, apo...)."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    sh = r71_sim.sig_hash(signature)
    (out_dir / f'{name}.signature.json').write_text(json.dumps(signature, sort_keys=True, default=str, indent=1))
    cache = out_dir / f'{name}.jsonl'
    done = r71_sim._read_cache(cache, sh)
    todo = [r for r in rows.itertuples(index=False) if r.image_id not in done]
    log(f'{name}: {len(done)} gotowe, {len(todo)} do policzenia')
    if cache.exists() and cache.stat().st_size and not cache.read_bytes().endswith(b'\n'):
        with open(cache, 'a') as fh:
            fh.write('\n')
    t0 = time.time()
    with open(cache, 'a') as fh:
        for k, r in enumerate(todo):
            t1 = time.time(); rec = {'image_id': r.image_id, '_sig': sh}
            try:
                gray = r45_core.read_gray(r.path); h, w = gray.shape; rec.update(h=int(h), w=int(w))
                rec.update(fn(r, gray)); rec['status'] = 'OK'
            except Exception as e:  # noqa: BLE001 - one image never stops the run
                rec['status'] = f'ERROR:{type(e).__name__}:{e}'[:300]
            rec['seconds'] = round(time.time() - t1, 3)
            fh.write(json.dumps(r45_core.clean_json(rec)) + '\n'); fh.flush()
            done[r.image_id] = rec
            if (k + 1) % 50 == 0 or k + 1 == len(todo):
                el = time.time() - t0
                log(f'{name}: {k + 1}/{len(todo)}  {el / 60:.1f} min, zostało ok. {el / (k + 1) * (len(todo) - k - 1) / 60:.1f} min')
    return [done[i] for i in rows.image_id if i in done]


def export_tags(name, recs, out_dir, tags):
    """One r71_sim export per tag ({name}_frags_{tag}_t60.csv.gz) + images list; aponeurosis pairs if present."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    keep = ('image_id', 'h', 'w', 'status', 'seconds')
    for t in tags:
        rr = [{**{q: r.get(q) for q in keep}, 'frags': (r.get('frags') or {}).get(t)} for r in recs]
        r71_sim.export(name, rr, out_dir, t)
    if any(r.get('apo') for r in recs):
        r71_sim.export(name, [{**{q: r.get(q) for q in keep}, 'apo': r.get('apo')} for r in recs], out_dir, 'none')
