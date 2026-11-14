"""R50T: zespół modeli pęczków R17 — trzy widoki każdego obrazu i ocena na OSF-35.

Widoki (dla każdego modelu R17):
  plain — dokładnie jak R17C (umud_infer.infer_manifest): predict_full bez TTA + rozcięgna R12B z obrazu;
  flip  — średnia mapy z obrazu i jego odbicia lustrzanego (predict_full(tta=True)), te same rozcięgna;
  sq    — obraz przeskalowany w poziomie do kwadratowych pikseli (współczynnik formatu ekranu), mapa z odbiciem,
          rozcięgna R12B policzone na tym samym przeskalowanym obrazie. PA i FL wychodzą od razu fizyczne.
Dla formatów z kwadratowymi pikselami (teln, gry, OSF) widok sq = flip (bez liczenia drugi raz).
SPDX-License-Identifier: GPL-3.0-or-later
"""
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

import umud_core as UC
import umud_model as UM

# Poziomy / pionowy rozmiar piksela w mm (piksel poziomy węższy => r < 1). Z budowy formatu ekranu, nie z etykiet:
# png  — zrzut 16:9 przeskalowany do 1200x800 (znacznik P 27x32 px), r = 1.5 / 1.778 = 0.84375
# sie  — zrzut 4:3 przeskalowany do 1200x800, szerokość obrazu 57.4 mm przy aperturze sondy 51.0 mm => 1.125
# telr — linijka pozioma 134.5 px / 10 mm, pionowa 148.2 px / 10 mm => 0.9076
ASPECT = {'png': 0.84375, 'sie': 1.125, 'telr': 0.9076, 'teln': 1.0, 'gry': 1.0, 'osf': 1.0, '?': 1.0}
VIEWS = ('plain', 'flip', 'sq')
FIELDS = ('pa_deg', 'fl_mm', 'mt_mm', 'mt_center_mm', 'band_correction_mm', 'n_fasc', 'apo_pairs', 'extrapolation_ratio', 'status')


def detect_group(path):
    """Źródło obrazu z samego pliku (to samo co w notebooku Kaggle)."""
    from PIL import Image
    path = Path(path)
    if path.suffix.lower() == '.png':
        return 'png'
    im = Image.open(path)
    w, h = im.size
    if (w, h) == (1088, 644):
        return 'teln'
    if h == 853 or (h == 513 and w in (465, 466)) or im.mode in ('L', 'I;16'):
        return 'gry'
    if (w, h) == (1200, 800):
        a = np.asarray(im.convert('L'))
        return 'sie' if float(np.median(a[420:600, 5:40])) < 20 else 'telr'
    return '?'


def read_color(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'Nieczytelny obraz: {path}')
    return image


def square_pixels(image, r):
    """Szerokość * 1/r: piksel poziomy staje się równy pionowemu (skala mm/px z linijki pionowej bez zmian)."""
    if abs(r - 1.0) < 1e-12:
        return image
    h, w = image.shape[:2]
    nw = max(1, int(round(w / r)))
    return cv2.resize(image, (nw, h), interpolation=cv2.INTER_AREA if nw < w else cv2.INTER_LINEAR)


@torch.inference_mode()
def apo_from_color(model, image, device):
    """umud_model.predict_legacy_apo dla obrazu w pamięci (ta sama kolejność operacji)."""
    h, w = image.shape[:2]
    x = cv2.resize(image, (518, 518), interpolation=cv2.INTER_LINEAR)[:, :, ::-1].astype(np.float32) / 255.
    x = (x - np.array([.485, .456, .406], np.float32)) / np.array([.229, .224, .225], np.float32)
    t = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)[None])).to(device)
    p = torch.sigmoid(model(t).float())[0, 0].cpu().numpy()
    return cv2.resize(p, (w, h), interpolation=cv2.INTER_LINEAR)


@torch.inference_mode()
def fasc_probs(model, gray, side, device):
    """(mapa bez TTA, mapa z odbiciem) — identyczne z predict_full(tta=False) i predict_full(tta=True)."""
    x, _ = UC.letterbox_image(gray, side)
    t = torch.from_numpy(x[None, None]).to(device)
    with torch.amp.autocast(device_type=device, dtype=torch.float16, enabled=(device == 'cuda')):
        p = torch.sigmoid(model(t).float())
        q = torch.flip(torch.sigmoid(model(torch.flip(t, (-1,))).float()), (-1,))
        pf = (p + q) / 2
    return (UC.restore_probability(p[0, 0].cpu().numpy(), gray.shape),
            UC.restore_probability(pf[0, 0].cpu().numpy(), gray.shape))


def _clean(z):
    out = {}
    for k, v in z.items():
        if isinstance(v, np.integer):
            v = int(v)
        elif isinstance(v, (float, np.floating)):
            v = float(v) if np.isfinite(v) else None
        out[k] = v
    return out


def clean_json(obj):
    """Do zapisu JSON bez NaN/inf i typów numpy."""
    if isinstance(obj, dict):
        return {str(k): clean_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean_json(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return float(obj) if np.isfinite(obj) else None
    return obj


def image_rows(row, group, models, thresholds, apo_model, side, device):
    """Wszystkie widoki wszystkich modeli dla jednego obrazu (rozcięgna liczone raz na widok)."""
    path, ppm = Path(row['path']), float(row['px_per_mm'])
    gray = UC.read_gray(path)
    r = ASPECT.get(group, 1.0)
    ap = UM.predict_legacy_apo(apo_model, path, device)
    sq = None
    if abs(r - 1.0) > 1e-12:
        gsq = square_pixels(gray, r)
        sq = (gsq, apo_from_color(apo_model, square_pixels(read_color(path), r), device))
    out = []
    for name, model in models.items():
        thr = float(thresholds[name])
        pp, pf = fasc_probs(model, gray, side, device)
        zs = {'plain': UC.measure_geometry(ap, pp, ppm, apo_thr=.5, fasc_thr=thr),
              'flip': UC.measure_geometry(ap, pf, ppm, apo_thr=.5, fasc_thr=thr)}
        if sq is None:
            zs['sq'] = dict(zs['flip'])
        else:
            _, pq = fasc_probs(model, sq[0], side, device)
            zs['sq'] = UC.measure_geometry(sq[1], pq, ppm, apo_thr=.5, fasc_thr=thr)
        for view in VIEWS:
            z = _clean(zs[view])
            out.append({'image_id': row['image_id'], 'group': group, 'aspect': r, 'model': name, 'view': view,
                        **{k: z.get(k) for k in FIELDS}})
    return out


VIEWS_VERSION = 'R50T-views-v1'


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


def run_views(name, manifest, groups, models, thresholds, apo_model, side, device, out_dir, signature, model_sigs, log=print):
    """Wznawialnie: jeden plik JSON na (obraz, model). Klucz zawiera sumę checkpointu i próg tylko tego modelu,
    więc dołożenie nowego modelu liczy wyłącznie jego widoki."""
    out_dir = Path(out_dir)
    cache = out_dir / 'cache'
    cache.mkdir(parents=True, exist_ok=True)
    rows, t0, done = [], time.time(), 0
    for i, r in enumerate(manifest.to_dict('records')):
        per_model = {}
        need = {}
        for mname, model in models.items():
            key = hashlib.sha256(json.dumps({**signature, 'views': VIEWS_VERSION, 'model': mname, 'model_sig': model_sigs[mname],
                                             'image': r['image_sha256'], 'ppm': float(r['px_per_mm']), 'group': groups[r['image_id']]},
                                            sort_keys=True).encode()).hexdigest()[:32]
            p = cache / f'{key}.json'
            if p.exists():
                per_model[mname] = json.loads(p.read_text())
            else:
                need[mname] = (model, p)
        if need:
            got = image_rows(r, groups[r['image_id']], {m: v[0] for m, v in need.items()}, thresholds, apo_model, side, device)
            for mname, (_, p) in need.items():
                mine = [x for x in got if x['model'] == mname]
                UC.atomic_json(p, mine)
                per_model[mname] = mine
            done += 1
        for mname in models:
            rows.extend(per_model[mname])
        if need and (done % 25 == 0):
            log(f'{name}: {i + 1}/{len(manifest)} obrazów, {(time.time() - t0) / max(done, 1):.1f} s/obraz')
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f'{name}.csv', index=False)
    return df


def wide(df):
    """Tabela szeroka: image_id + kolumny <model>_<widok>_<pole> dla pa_deg, fl_mm, mt_mm, status."""
    parts = []
    for (m, v), g in df.groupby(['model', 'view'], sort=True):
        g = g.set_index('image_id')[['pa_deg', 'fl_mm', 'mt_mm', 'status']]
        g.columns = [f'{m}_{v}_{c}' for c in g.columns]
        parts.append(g)
    meta = df.drop_duplicates('image_id').set_index('image_id')[['group', 'aspect']]
    return pd.concat([meta] + parts, axis=1).reset_index()


def osf_truth(path):
    g = pd.read_csv(path)
    out = pd.DataFrame({'image_id': g.ImageID})
    for t, c in (('pa_deg', 'PA'), ('fl_mm', 'FL'), ('mt_mm', 'MT')):
        out[t] = g[[f'R{i}_{c}' for i in range(1, 8)]].mean(axis=1).values
    return out


def ensemble(df, models, views, how='geo'):
    """Średnia po (modelach x widokach) dla każdego obrazu; FL geometrycznie, PA i MT arytmetycznie; tylko pełne zestawy."""
    sub = df[df.model.isin(models) & df.view.isin(views)].copy()
    for c in ('pa_deg', 'fl_mm', 'mt_mm'):
        sub[c] = pd.to_numeric(sub[c], errors='coerce')
    k = len(models) * len(views)
    rows = []
    for iid, g in sub.groupby('image_id', sort=False):
        ok = g.dropna(subset=['pa_deg', 'fl_mm'])
        if len(ok) < k:
            rows.append({'image_id': iid, 'pa_deg': np.nan, 'fl_mm': np.nan, 'mt_mm': np.nan})
            continue
        fl = float(np.exp(np.log(ok.fl_mm).mean())) if how == 'geo' else float(ok.fl_mm.mean())
        rows.append({'image_id': iid, 'pa_deg': float(ok.pa_deg.mean()), 'fl_mm': fl, 'mt_mm': float(ok.mt_mm.mean())})
    return pd.DataFrame(rows)


def osf_table(df, truth, models):
    """MAE względem średniej 7 ekspertów po odjęciu stałego przesunięcia (mediana), na wspólnym zbiorze obrazów."""
    combos = [([m], [v]) for m in models for v in VIEWS]
    if len(models) > 1:
        combos += [(list(models), ['plain']), (list(models), ['flip']), (list(models), ['plain', 'flip'])]
    preds = {f"{'+'.join(ms)}|{'+'.join(vs)}": ensemble(df, ms, vs) for ms, vs in combos}
    common = set(truth.image_id)
    for p in preds.values():
        common &= set(p.dropna().image_id)
    common = sorted(common)
    if not common:
        return pd.DataFrame(columns=['combo', 'n', 'pa_deg_mae_debiased', 'fl_mm_mae_debiased', 'mt_mm_mae_debiased'])
    rows = []
    t = truth.set_index('image_id').loc[common]
    for name, p in preds.items():
        p = p.set_index('image_id').loc[common]
        row = {'combo': name, 'n': len(common)}
        for c in ('pa_deg', 'fl_mm', 'mt_mm'):
            e = p[c].values - t[c].values
            row[f'{c}_mae_debiased'] = float(np.mean(np.abs(e - np.median(e))))
        rows.append(row)
    return pd.DataFrame(rows).sort_values('fl_mm_mae_debiased').reset_index(drop=True)


def repro_table(df, saved, model):
    """Widok plain modelu z R17C musi odtworzyć zapisane pomiary R17C (FL, PA, MT)."""
    a = df[(df.model == model) & (df.view == 'plain')].set_index('image_id')
    b = saved.set_index('image_id')
    common = sorted(set(a.index) & set(b.index))
    rows = []
    for c in ('pa_deg', 'fl_mm', 'mt_mm'):
        x = pd.to_numeric(a.loc[common, c], errors='coerce').values
        y = pd.to_numeric(b.loc[common, c], errors='coerce').values
        ok = np.isfinite(x) & np.isfinite(y)
        d = np.abs(x[ok] - y[ok])
        rows.append({'model': model, 'target': c, 'n_both': int(ok.sum()), 'n_only_one': int((np.isfinite(x) ^ np.isfinite(y)).sum()),
                     'max_abs_diff': float(d.max()) if len(d) else None, 'share_exact_1e6': float((d < 1e-6).mean()) if len(d) else None,
                     'share_within_1pct': float((d <= 0.01 * np.abs(y[ok])).mean()) if len(d) else None})
    return pd.DataFrame(rows)
