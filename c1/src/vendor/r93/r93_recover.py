"""R93T: odzysk bazy R23A z kodu, bez treningu.

1. R20 (paczka `R20_v1_0_0`, kod bez zmian): rozcięgna R12B (próg 0,5) + pęczki R17 refit (bok 768, bez TTA, próg 0,65)
   -> `r20_geometry.diagnose` -> PA wspieranych fragmentów (`pa_local_support_baseline`, źródło PA bazy PNG w R21) i FL R17.
2. MT R12B oryginalną funkcją z notebooka R12B v1.0.3 (komórka 4, przepisana bez zmian): dwie największe składowe maski
   (prawdopodobieństwo > 0,5), `vertical_mean` w pikselach; stała kalibracji wyprowadzana ponownie z OSF-35.
3. Wariant R20 na parze „dwie największe składowe” (ta sama warstwa co MT R12B) — dane do reguł dla obrazów bez wsparcia.
4. Skala z linijki (`umud_final.scale`) i bloki wideo (`umud_final.layout`) na wszystkich obrazach testu.
SPDX-License-Identifier: GPL-3.0-or-later
"""
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

import umud_core as UC
import umud_model as UM
import r20_geometry as RG

VERSION = 'R93T_v1.0.0'
CODE_SHA = {'r20_geometry.py': '65dbcec320bc4dec433d6ffc1a440ecebac1df60706c23b0632d434dc219cb04',
            'umud_core.py': 'e7d9d0c927beb2bdcee2a48614ef9fdaac616f0dab6c595c0c2e3550d90e820e',
            'umud_model.py': '4f08f6c6f75130785a9656e6adac2d98dab184941c829c027b0c65325e069853'}
R17_REFIT_SHA = '6957c5228c1cf5c2067096d5f0214f6c2a7f54b0aeca03b8f664051022859d31'
R12B_SHA = '21ec34d0f5dd7664ecab0ca608f976fdf9d2a161e2af2873c21d513cd5ae19ef'
SIDE, APO_THR, FASC_THR = 768, .5, .65
R12C_BIAS_MM = 1.3629          # wartość z R12C (mediana błędu MT na OSF-35); tu tylko do porównania z wyprowadzoną
MT_CLIP = (10.0, 50.0)
MT_DEFINITIONS = ('vertical_mid', 'vertical_mean', 'vertical_trimmed20', 'normal_mid')
# obrazy, dla których zapisujemy mapy prawdopodobieństwa (uint8) do projektowania reguł bez kolejnego biegu
MAP_IDS = tuple(f'IMG_{k:05d}.tif' for k in (19, 42, 76, 77, 78, 79, 80, 81, 85, 151, 152, 153, 154, 155, 170, 244)) + \
          tuple(f'IMG_{k:05d}.png' for k in (256, 261, 271, 272, 275, 303, 307))


class SessionBudgetReached(RuntimeError):
    pass


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def verify_code(src_dir):
    """Kod R20 musi być bit w bit ten sam co w R20 (sumy z R21 PROVENANCE.json)."""
    got = {n: UC.sha256(Path(src_dir) / n) for n in CODE_SHA}
    bad = {n: g for n, g in got.items() if g != CODE_SHA[n]}
    if bad:
        raise RuntimeError(f'Kod R20 różni się od oryginału: {bad}')
    return got


# ---------------------------------------------------------------------------- R12B: oryginalna funkcja MT (komórka 4)
def _profiles(mask, min_area_frac=0.002):
    """Dwie największe składowe -> profil y(x) każdej z nich. Górna = mniejsze y."""
    m = np.asarray(mask) > 0
    if m.sum() == 0:
        return None
    lab, n = ndi.label(m)
    if n < 2:
        return None
    areas = ndi.sum(m, lab, index=np.arange(1, n + 1))
    order = np.argsort(areas)[::-1][:2]
    if areas[order[1]] < min_area_frac * m.size:
        return None
    prof = []
    for idx in order:
        comp = lab == (idx + 1)
        ys, xs = np.nonzero(comp)
        p = np.full(m.shape[1], np.nan)
        s = pd.Series(ys).groupby(xs).mean()
        p[s.index.to_numpy()] = s.to_numpy()
        prof.append(p)
    a, b = prof
    if np.nanmean(a) > np.nanmean(b):
        a, b = b, a
    return a, b                                   # upper, lower


def thickness_from_mask(mask):
    """Zwraca słownik definicji grubości W PIKSELACH, albo same NaN-y."""
    out = {k: np.nan for k in ("vertical_mid", "vertical_mean", "vertical_trimmed20",
                               "normal_mid", "coverage")}
    pr = _profiles(mask)
    if pr is None:
        return out
    up, lo = pr
    both = np.isfinite(up) & np.isfinite(lo)
    if both.sum() < 10:
        return out
    xs = np.nonzero(both)[0]
    gap = (lo - up)[xs]
    gap = gap[gap > 0]
    if gap.size < 10:
        return out
    out["coverage"] = float(both.mean())
    mid = xs[len(xs) // 2]
    out["vertical_mid"] = float(lo[mid] - up[mid])
    out["vertical_mean"] = float(np.mean(gap))
    k = int(math.floor(gap.size * 0.20))
    g = np.sort(gap)
    out["vertical_trimmed20"] = float(np.mean(g[k: gap.size - k] if gap.size - 2 * k >= 1 else g))
    # odległość prostopadła: uwzględnia nachylenie dolnego rozcięgna
    fit_x = xs.astype(float)
    sl_lo = np.polyfit(fit_x, lo[xs], 1)[0]
    sl_up = np.polyfit(fit_x, up[xs], 1)[0]
    theta = math.atan((sl_lo + sl_up) / 2.0)
    out["normal_mid"] = float(out["vertical_mid"] * math.cos(theta))
    return out
# ---------------------------------------------------------------------------- koniec kodu R12B


def two_largest_mask(mask, min_area_frac=0.002):
    """Maska tylko z dwiema składowymi, które wybiera `_profiles` (ta sama etykietyzacja i ten sam próg pola)."""
    m = np.asarray(mask) > 0
    if m.sum() == 0:
        return None
    lab, n = ndi.label(m)
    if n < 2:
        return None
    areas = ndi.sum(m, lab, index=np.arange(1, n + 1))
    order = np.argsort(areas)[::-1][:2]
    if areas[order[1]] < min_area_frac * m.size:
        return None
    return (lab == order[0] + 1) | (lab == order[1] + 1)


def mt_r12b_mm(vertical_mean_px, px_per_mm, bias_mm):
    """MT jak w R12B/R12C: px / skala - stała, przycięte do 10-50 mm (brak pomiaru -> NaN)."""
    v = np.asarray(vertical_mean_px, float) / np.asarray(px_per_mm, float) - bias_mm
    return np.where(np.isfinite(v), np.clip(v, *MT_CLIP), np.nan)


# ---------------------------------------------------------------------------- wagi i predyktory
def registry(project, drive_root, r17_sha=R17_REFIT_SHA, r12b_sha=R12B_SHA):
    reg = {}
    for key, path, want in [('r17_refit', Path(project) / 'outputs' / 'R17_v1_0_0' / 'refit' / 'best.pt', r17_sha),
                            ('r12b', Path(drive_root) / 'UMUD_R12B' / 'best_dinov2_frozen.pt', r12b_sha)]:
        if not path.is_file():
            raise FileNotFoundError(f'Brak wag {key}: {path}')
        got = UC.sha256(path)
        if got != want:
            raise RuntimeError(f'Inne wagi {key}: {got} zamiast {want}')
        reg[key] = {'path': str(path), 'sha256': got}
    return reg


def load_models(reg, device):
    fasc, ck = UM.load_model(reg['r17_refit']['path'], device)
    apo = UM.load_legacy_apo(reg['r12b']['path'], device)
    return {'fasc': fasc, 'apo': apo, 'fasc_config': ck.get('config', {})}


def predict_maps(models, path, device):
    """Dokładnie jak R17C/R20: mapa R12B z obrazu kolorowego, mapa pęczków R17 refit bez TTA na obrazie szarym."""
    gray = UC.read_gray(path)
    p_apo = UM.predict_legacy_apo(models['apo'], path, device)
    p_fasc = UM.predict_full(models['fasc'], gray, side=SIDE, device=device, tta=False)
    return p_apo, p_fasc


def _clean(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (float, np.floating)):
        return float(v) if np.isfinite(v) else None
    return v


def clean(d):
    return {str(k): _clean(v) for k, v in d.items()}


def jsonable(obj):
    """Do zapisu JSON bez NaN/inf i typów numpy (rekurencyjnie)."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if obj is None or isinstance(obj, (str, bool, np.bool_)):
        return bool(obj) if isinstance(obj, np.bool_) else obj
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return float(obj) if np.isfinite(obj) else None
    return obj


def mask_digest(a):
    a = np.ascontiguousarray(np.asarray(a, bool))
    return sha_bytes(str(a.shape).encode() + a.tobytes())


def read_r20_cache(path):
    """Maski zapisane przez R20 (`mask_cache/<mask_key>.npz`): zwraca słownik tablic bool bez metadanych."""
    with np.load(path, allow_pickle=False) as z:
        return {k: np.asarray(z[k], bool) for k in z.files if k != 'metadata'}


def cache_check(apo, fasc, cached, ppm):
    """Porównanie z maskami R20 piksel po pikselu + pomiary R20 i MT R12B liczone na samych maskach R20
    (sprawdza kod geometrii i skalę niezależnie od tego, czy nasza inferencja daje te same maski)."""
    out = {'names': sorted(cached)}

    def pick(key):                       # nazwy tablic w npz R20: 'apo' / 'fasc' (albo jednoznacznie je zawierające)
        if key in cached:
            return cached[key]
        names = [n for n in cached if key in n.lower()]
        return cached[names[0]] if len(names) == 1 else None
    ca, cf = pick('apo'), pick('fasc')
    out['apo_diff'] = int((ca != apo).sum()) if ca is not None and ca.shape == apo.shape else None
    out['fasc_diff'] = int((cf != fasc).sum()) if cf is not None and cf.shape == fasc.shape else None
    if ca is not None and cf is not None and ca.shape == cf.shape:
        z, _ = RG.diagnose(ca, cf, ppm)
        out['r20_on_cache'] = clean({k: z.get(k) for k in ('pa_deg', 'fl_mm', 'mt_center_mm', 'status', 'n_fasc',
                                                           'pa_local_support_baseline', 'n_local')})
        out['r12b_px_on_cache'] = clean(thickness_from_mask(ca.astype(np.uint8)))
    return out


def process(row, p_apo, p_fasc, cached=None):
    """Wszystkie pomiary jednego obrazu z dwóch map (bez sieci — testowalne na mapach syntetycznych)."""
    ppm = float(row['px_per_mm'])
    if not np.isfinite(ppm) or ppm <= 0:
        raise ValueError(f"Zła skala: {row['image_id']}")
    if p_apo.shape != p_fasc.shape:
        raise ValueError('Mapy w różnych układach współrzędnych')
    apo = p_apo >= APO_THR
    fasc = p_fasc >= FASC_THR
    z, frags = RG.diagnose(apo, fasc, ppm)
    t = thickness_from_mask((p_apo > 0.5).astype(np.uint8))
    two = two_largest_mask(p_apo > 0.5)
    z2, frags2 = (RG.diagnose(two, fasc, ppm) if two is not None else (None, []))
    rec = {'image_id': row['image_id'], 'h': int(p_apo.shape[0]), 'w': int(p_apo.shape[1]), 'px_per_mm': ppm,
           'r20': clean(z), 'r20_frags': [clean(f) for f in frags],
           'r12b_px': clean(t), 'two_found': two is not None,
           'r20two': clean(z2) if z2 is not None else None, 'r20two_frags': [clean(f) for f in frags2],
           'apo_digest': mask_digest(apo), 'fasc_digest': mask_digest(fasc),
           'apo_frac': float(apo.mean()), 'fasc_frac': float(fasc.mean()),
           'cache': cache_check(apo, fasc, cached, ppm) if cached is not None else None}
    return rec


def load_manifest(csv_path, project_root, check_sha=True):
    m = pd.read_csv(csv_path)
    m['path'] = [str(Path(project_root) / p) for p in m.relative_path]
    missing = [p for p in m.path if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError(f'Brak {len(missing)} obrazów, np. {missing[:3]}')
    if check_sha:
        bad = [r.image_id for r in m.itertuples() if UC.sha256(r.path) != r.image_sha256]
        if bad:
            raise ValueError(f'{len(bad)} obrazów różni się od manifestu, np. {bad[:5]}')
    if m.image_id.duplicated().any():
        raise ValueError('Powtórzone image_id')
    return m


def _signature(row, reg, package_sha):
    return {'version': VERSION, 'package': package_sha, 'image_sha256': row.get('image_sha256'),
            'image_id': row['image_id'], 'px_per_mm': float(row['px_per_mm']),
            'weights': {k: v['sha256'] for k, v in reg.items()}, 'side': SIDE, 'thr': [APO_THR, FASC_THR]}


def _read_cache(path):
    done = {}
    if Path(path).is_file():
        for line in Path(path).read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue                      # ucięta ostatnia linia po rozłączeniu
            done[r['key']] = r
    return done


def run_dataset(name, manifest, models, reg, package_sha, device, out_dir, deadline, log=print,
                predictor=None, map_ids=MAP_IDS, r20_cache=None):
    """Wznawialny przebieg po obrazach; każdy wynik od razu dopisany do JSONL na Drive.
    `r20_cache`: {image_id: ścieżka npz z maskami R20} — porównanie piksel po pikselu (tylko test)."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / f'{name}_records.jsonl'
    maps_dir = out_dir / 'maps'; maps_dir.mkdir(exist_ok=True)
    done = _read_cache(cache_path)
    predictor = predictor or (lambda path: predict_maps(models, path, device))
    r20_cache = r20_cache or {}
    recs, t0, n_new = [], time.time(), 0
    for i, row in enumerate(manifest.to_dict('records')):
        sig = _signature(row, reg, package_sha); key = sha_bytes(json.dumps(sig, sort_keys=True).encode())
        if key in done:
            recs.append(done[key]['rec']); continue
        if time.time() > deadline:
            raise SessionBudgetReached(f'{name}: limit czasu sesji po {i} obrazach')
        p_apo, p_fasc = predictor(row['path'])
        cpath = r20_cache.get(row['image_id'])
        cached = read_r20_cache(cpath) if cpath and Path(cpath).is_file() else None
        rec = process(row, p_apo, p_fasc, cached)
        if row['image_id'] in map_ids:
            q = lambda p: np.where(p >= .05, np.round(np.clip(p, 0, 1) * 255), 0).astype(np.uint8)   # < 0,05 -> 0 (mniejszy plik)
            np.savez_compressed(maps_dir / f"{name}_{Path(row['image_id']).stem}.npz", apo=q(p_apo), fasc=q(p_fasc))
        with cache_path.open('a') as f:
            f.write(json.dumps({'key': key, 'sig': sig, 'rec': rec}, allow_nan=False) + '\n')
        recs.append(rec); n_new += 1
        if n_new % 25 == 0 or i + 1 == len(manifest):
            log(f'{name}: {i + 1}/{len(manifest)} obrazów, {(time.time() - t0) / max(n_new, 1):.2f} s/obraz')
    return recs


# ---------------------------------------------------------------------------- tabele
DETAIL_COLS = ['pa_deg', 'fl_mm', 'mt_mm', 'mt_center_mm', 'band_correction_mm', 'n_fasc', 'apo_pairs', 'extrapolation_ratio',
               'status', 'pa_r17_median', 'pa_center_median', 'pa_length_weighted', 'pa_coherent_weighted', 'pa_local_median',
               'pa_local_support_baseline', 'local_coverage', 'n_local', 'orientation_concentration', 'pa_iqr_deg',
               'exact_mask_baseline', 'apo_inner_angle', 'apo_center_angle']


def tables(recs, bias_mm):
    det, det2, fr, fr2, mt, cache = [], [], [], [], [], []
    for r in recs:
        base = {'image_id': r['image_id'], 'px_per_mm': r['px_per_mm'], 'h': r['h'], 'w': r['w']}
        det.append({**base, **{c: r['r20'].get(c) for c in DETAIL_COLS},
                    'apo_digest': r['apo_digest'], 'fasc_digest': r['fasc_digest']})
        det2.append({**base, 'two_found': r['two_found'],
                     **{c: (r['r20two'] or {}).get(c) for c in DETAIL_COLS}})
        fr += [{'image_id': r['image_id'], **f} for f in r['r20_frags']]
        fr2 += [{'image_id': r['image_id'], **f} for f in r['r20two_frags']]
        mt.append({**base, **{k: r['r12b_px'].get(k) for k in (*MT_DEFINITIONS, 'coverage')}})
        c = r.get('cache')
        if c is not None:
            oc = c.get('r20_on_cache') or {}; tc = c.get('r12b_px_on_cache') or {}
            cache.append({'image_id': r['image_id'], 'px_per_mm': r['px_per_mm'], 'apo_diff_px': c.get('apo_diff'),
                          'fasc_diff_px': c.get('fasc_diff'), **{f'cache_{k}': v for k, v in oc.items()},
                          'cache_vertical_mean': tc.get('vertical_mean')})
    mt = pd.DataFrame(mt)
    for d in MT_DEFINITIONS:
        mt[f'{d}_mm'] = mt[d] / mt.px_per_mm
    mt['mt_r12c_mm'] = mt_r12b_mm(mt.vertical_mean, mt.px_per_mm, bias_mm)
    cache = pd.DataFrame(cache)
    if len(cache):
        cache['cache_mt_r12c_mm'] = mt_r12b_mm(cache.cache_vertical_mean, cache.px_per_mm, bias_mm)
    return {'detail': pd.DataFrame(det), 'detail_two': pd.DataFrame(det2), 'fragments': pd.DataFrame(fr),
            'fragments_two': pd.DataFrame(fr2), 'mt': mt, 'cache': cache}


def cache_summary(cache, ref, base):
    """Maski R20 z Drive: ile pikseli różni się od naszych i czy pomiary na samych maskach R20 odtwarzają R20 i bazę."""
    if cache is None or not len(cache):
        return {'n': 0}
    c = cache.set_index('image_id'); r = ref.set_index('image_id'); b = base.set_index('image_id')
    ids = [i for i in c.index if i in r.index]
    d_supp = (pd.to_numeric(c.loc[ids, 'cache_pa_local_support_baseline'], errors='coerce')
              - r.loc[ids, 'pa_local_support_baseline']).abs()
    d_fl = (pd.to_numeric(c.loc[ids, 'cache_fl_mm'], errors='coerce') - r.loc[ids, 'fl_mm']).abs()
    st = (c.loc[ids, 'cache_status'].astype(str) == r.loc[ids, 'status'].astype(str))
    ids_b = [i for i in c.index if i in b.index]
    d_mt = (pd.to_numeric(c.loc[ids_b, 'cache_mt_r12c_mm'], errors='coerce') - b.loc[ids_b, 'mt_mm']).abs()
    return {'n': int(len(c)), 'apo_identical': int((c.apo_diff_px == 0).sum()), 'fasc_identical': int((c.fasc_diff_px == 0).sum()),
            'apo_diff_px_max': int(c.apo_diff_px.max()) if c.apo_diff_px.notna().any() else None,
            'fasc_diff_px_max': int(c.fasc_diff_px.max()) if c.fasc_diff_px.notna().any() else None,
            'r20_on_cache_status_equal': int(st.sum()),
            'r20_on_cache_support_max_abs': float(d_supp.max()) if d_supp.notna().any() else None,
            'r20_on_cache_fl_max_abs': float(d_fl.max()) if d_fl.notna().any() else None,
            'r12c_mt_on_cache_exact_1e4': int((d_mt <= 1e-4).sum()), 'r12c_mt_on_cache_n': int(d_mt.notna().sum()),
            'r12c_mt_on_cache_far_ids': list(d_mt.index[d_mt > .01])}


def osf_bias(mt_osf, truth):
    """Stała R12C: mediana (MT_pred - MT_gt) po obrazach OSF-35 z pomiarem (wariant 'kalibrowany_LOO' w produkcji R12B)."""
    x = mt_osf.merge(truth, on='image_id', validate='one_to_one')
    ok = x.vertical_mean_mm.notna() & x.mt_gt_mm.notna()
    err = (x.vertical_mean_mm - x.mt_gt_mm)[ok]
    return float(np.median(err)), int(ok.sum()), x.assign(err_mm=x.vertical_mean_mm - x.mt_gt_mm)


def compare_r20(detail, ref):
    """Nasz bieg wobec R20 `test_detail.csv` (zapisanego przez R20 w 09.2026)."""
    a = detail.set_index('image_id'); b = ref.set_index('image_id')
    if set(a.index) != set(b.index):
        raise ValueError('Inne obrazy niż w R20')
    b = b.loc[a.index]
    rows = []
    for c in DETAIL_COLS:
        if c not in b.columns:
            continue
        if c in ('status', 'exact_mask_baseline'):
            same = (a[c].astype(str).str.lower() == b[c].astype(str).str.lower())
            rows.append({'kolumna': c, 'n': len(a), 'zgodne': int(same.sum()), 'max_abs_diff': None, 'rozne_pokrycie': None,
                         'przyklady': ', '.join(a.index[~same][:6])})
            continue
        x = pd.to_numeric(a[c], errors='coerce').to_numpy(float); y = pd.to_numeric(b[c], errors='coerce').to_numpy(float)
        cov = np.isfinite(x) != np.isfinite(y); both = np.isfinite(x) & np.isfinite(y)
        d = np.abs(x - y); d[~both] = np.nan
        bad = cov | (both & (d > 1e-6))
        rows.append({'kolumna': c, 'n': len(a), 'zgodne': int((~bad).sum()),
                     'max_abs_diff': float(np.nanmax(d)) if both.any() else None, 'rozne_pokrycie': int(cov.sum()),
                     'przyklady': ', '.join(a.index[bad][:6])})
    return pd.DataFrame(rows)


def compare_base(detail, mt, base, sources, bias_mm):
    """Nasz bieg wobec bazy R23A: PA obrazów ze źródłem R20, FL R17 refit, MT R12C (z wyprowadzoną stałą)."""
    a = detail.set_index('image_id'); m = mt.set_index('image_id'); b = base.set_index('image_id'); s = sources.set_index('image_id')
    ids = [i for i in b.index if i in a.index]
    rows = []
    for i in ids:
        src = s.loc[i, 'src_pa'] if i in s.index else '?'
        supp = a.loc[i, 'pa_local_support_baseline']
        mt_now = float(mt_r12b_mm(m.loc[i, 'vertical_mean'], m.loc[i, 'px_per_mm'], bias_mm))
        rows.append({'image_id': i, 'grp': s.loc[i, 'grp'] if i in s.index else '?', 'src_pa': src,
                     'pa_base': b.loc[i, 'pa_deg'], 'pa_r20_support': supp,
                     'd_pa_r20': (supp - b.loc[i, 'pa_deg']) if supp is not None and np.isfinite(supp) else np.nan,
                     'fl_base': b.loc[i, 'fl_mm'], 'fl_r20': a.loc[i, 'fl_mm'],
                     'd_fl': (a.loc[i, 'fl_mm'] - b.loc[i, 'fl_mm']) if a.loc[i, 'fl_mm'] is not None else np.nan,
                     'mt_base': b.loc[i, 'mt_mm'], 'mt_r12c': mt_now, 'd_mt': mt_now - b.loc[i, 'mt_mm']})
    x = pd.DataFrame(rows)
    for c in ('d_pa_r20', 'd_fl', 'd_mt'):
        x[c] = pd.to_numeric(x[c], errors='coerce')
    pa_r20 = x[(x.src_pa != 'R22dec') & x.pa_r20_support.notna()]
    summ = {'pa_r20_n': int(len(pa_r20)), 'pa_r20_exact_1e6': int((pa_r20.d_pa_r20.abs() <= 1e-6).sum()),
            'pa_r20_max_abs': float(pa_r20.d_pa_r20.abs().max()) if len(pa_r20) else None,
            'fl_n': int(x.d_fl.notna().sum()), 'fl_exact_1e6': int((x.d_fl.abs() <= 1e-6).sum()),
            'fl_max_abs': float(x.d_fl.abs().max()) if x.d_fl.notna().any() else None,
            'mt_n': int(x.d_mt.notna().sum()), 'mt_exact_1e4': int((x.d_mt.abs() <= 1e-4).sum()),
            'mt_within_001': int((x.d_mt.abs() <= .01).sum()),
            'mt_missing_ids': x.image_id[x.d_mt.isna()].tolist(),
            'mt_far_ids': x.image_id[x.d_mt.abs() > .01].tolist(), 'bias_mm': bias_mm}
    return x, summ


def scale_and_blocks(manifest, registry_df, log=print):
    """Skala z linijki i bloki wideo dla wszystkich obrazów (CPU) wobec zamrożonego rejestru skali."""
    from PIL import Image
    from umud_final.scale import read_scale, to_gray
    from umud_final.layout import blocks_from_images, group_of
    reg = registry_df.set_index('image_id')
    rows = []
    for r in manifest.itertuples(index=False):
        im = Image.open(r.path); im.load()
        s = read_scale(im, is_png=r.image_id.lower().endswith('.png'))
        ref_l = reg.loc[r.image_id, 'layout'] if r.image_id in reg.index else None
        ref_s = float(reg.loc[r.image_id, 'SCALE_PX_PER_CM_FINAL']) if r.image_id in reg.index else np.nan
        rows.append({'image_id': r.image_id, **s.as_dict(), 'group': group_of(s.layout), 'registry_layout': ref_l,
                     'registry_px_per_cm': ref_s, 'manifest_px_per_mm': float(r.px_per_mm),
                     'same_layout': s.layout == ref_l, 'abs_diff_px_per_cm': abs(s.px_per_cm - ref_s)})
    sc = pd.DataFrame(rows)
    nums = np.array([int(''.join(ch for ch in Path(i).stem if ch.isdigit())) for i in sc.image_id])
    paths = list(manifest.path)

    def loader(k):
        im = Image.open(paths[k]); im.load()
        return to_gray(im)
    bid, sims = blocks_from_images(nums, list(sc.layout), loader)
    sc['block'] = bid
    prev = {b: s for (a, b), s in sims.items()}
    sc['ncc_prev'] = [prev.get(k, np.nan) for k in range(len(sc))]
    n_ok = int((sc.same_layout & (sc.abs_diff_px_per_cm < 1e-6)).sum())
    log(f'skala z linijki = rejestr na {n_ok}/{len(sc)} obrazach; bloki wideo: {int(sc.block.max()) + 1 if (sc.block >= 0).any() else 0}, '
        f'klatek w blokach {int((sc.block >= 0).sum())}')
    return sc


def pack(out_dir, run_dir, extra=()):
    import zipfile
    out_dir = Path(out_dir); z = Path(run_dir) / 'R93T_wyniki.zip'; tmp = z.with_suffix('.tmp.zip')
    with zipfile.ZipFile(tmp, 'w', compression=zipfile.ZIP_DEFLATED) as f:
        for p in sorted(out_dir.rglob('*')):
            if p.is_file() and not p.name.endswith('_records.jsonl'):
                f.write(p, p.relative_to(out_dir.parent))
        for p in extra:
            if Path(p).is_file():
                f.write(p, Path(p).name)
    tmp.replace(z)
    return z


def selftest():
    """Kod R20 na geometrii o znanej odpowiedzi + MT R12B na syntetycznej masce (testy z oryginałów)."""
    r = RG.selftest()

    def synthetic(h=400, w=600, gap=80, tilt=0.0, thick=4):
        m = np.zeros((h, w), np.uint8)
        for x in range(w):
            y_up = int(h / 2 - gap / 2 + tilt * (x - w / 2)); y_lo = int(h / 2 + gap / 2 + tilt * (x - w / 2))
            m[max(0, y_up):y_up + thick, x] = 1; m[max(0, y_lo):y_lo + thick, x] = 1
        return m
    t = thickness_from_mask(synthetic(gap=80))
    assert abs(t['vertical_mid'] - 80) < 2 and abs(t['vertical_mean'] - 80) < 2, t
    t2 = thickness_from_mask(synthetic(gap=80, tilt=0.30))
    assert abs(t2['vertical_mid'] - 80) < 3 and t2['normal_mid'] < t2['vertical_mid']
    assert all(np.isnan(v) for v in thickness_from_mask(np.zeros((50, 50))).values())
    m3 = synthetic(gap=80); m3[10:12, 100:110] = 1          # mała trzecia składowa nie zmienia wyboru
    assert two_largest_mask(m3).sum() == synthetic(gap=80).sum()
    return {'r20': r, 'r12b_mt': {'flat': t['vertical_mean'], 'tilt': t2['vertical_mean']}, 'pass': True}
