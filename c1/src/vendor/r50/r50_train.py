"""R50T: dodatkowe modele pęczków R17 (tryb refit, stała liczba epok z R17 develop) z augmentacją proporcji ekranu.

Test ma obrazy o niekwadratowych pikselach (png x0.844, telr x0.908, sie x1.125), a dane treningowe R17 — kwadratowe.
Augmentacja losowo ściska/rozciąga treść w poziomie w tym samym zakresie, więc nowe modele widzą w treningu takie same
zniekształcenia jak w teście. Etykiety budowane tak jak w R17 (szkielet 1 px -> poszerzenie 3x3), tylko w nowej geometrii.
Uczenie wyłącznie na danych treningowych konkursu; LB służy potem tylko do wyboru wagi mieszania.
SPDX-License-Identifier: GPL-3.0-or-later
"""
import hashlib
import json
import math
import os
import random
import shutil
import tarfile
from pathlib import Path

import cv2
import numpy as np
import torch
from skimage.morphology import skeletonize

import umud_core as UC
import umud_model as UM
import umud_audit as UA

ASPECT_RANGE = (0.84, 1.13)
ASPECT_P = 0.6
BUDGET_HOURS = 10.5          # stała w konfiguracji (wchodzi do tożsamości przebiegu — nie zmieniać między sesjami)


def content_box(valid):
    ys = np.nonzero(valid.any(axis=1))[0]
    xs = np.nonzero(valid.any(axis=0))[0]
    if len(ys) == 0 or len(xs) == 0:
        return None
    return int(ys[0]), int(ys[-1]) + 1, int(xs[0]), int(xs[-1]) + 1


def hstretch(image, target, line, valid, s):
    """Treść letterboxu (valid > 0) rozciągnięta w poziomie o s i wpisana z powrotem w kwadrat S x S.
    image/target/line/valid: float32 S x S. Zwraca nowe tablice tego samego kształtu."""
    S = image.shape[0]
    box = content_box(valid > 0.5)
    if box is None:
        return image, target, line, valid
    y0, y1, x0, x1 = box
    nh, nw = y1 - y0, x1 - x0
    nw2, nh2 = nw * s, float(nh)
    f = min(1.0, S / nw2, S / nh2)
    nw2, nh2 = max(1, int(round(nw2 * f))), max(1, int(round(nh2 * f)))
    oy, ox = (S - nh2) // 2, (S - nw2) // 2

    def place(a, interp):
        z = np.zeros((S, S), np.float32)
        z[oy:oy + nh2, ox:ox + nw2] = cv2.resize(a[y0:y1, x0:x1], (nw2, nh2), interpolation=interp)
        return z
    shrink = nw2 < nw
    img2 = place(image, cv2.INTER_AREA if shrink else cv2.INTER_LINEAR)
    val2 = np.zeros((S, S), np.float32)
    val2[oy:oy + nh2, ox:ox + nw2] = 1.0
    soft = place(line, cv2.INTER_LINEAR)
    line2 = skeletonize(soft > 0.2).astype(np.float32) * val2
    tgt2 = cv2.dilate(line2.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(np.float32) * val2
    return img2, tgt2, line2, val2


def make_aspect_dataset(base_cls, lo=ASPECT_RANGE[0], hi=ASPECT_RANGE[1], p=ASPECT_P):
    class AspectCachedDataset(base_cls):
        def __getitem__(self, i):
            x, y, v, line = super().__getitem__(i)
            if self.augment and random.random() < p:
                s = math.exp(random.uniform(math.log(lo), math.log(hi)))
                a = [t[0].numpy() for t in (x, y, line, v)]
                img2, tgt2, line2, val2 = hstretch(a[0], a[1], a[2], a[3], s)
                x, y, line, v = (torch.from_numpy(np.ascontiguousarray(t[None], dtype=np.float32)) for t in (img2, tgt2, line2, val2))
            return x, y, v, line
    return AspectCachedDataset


def sync_cache(samples, drive_cache, drive_tar, local, log=print):
    """Lokalna kopia cache treningowego R17: z jednego pliku tar na Drive (szybko) albo plik po pliku (pierwszy raz)."""
    local = Path(local)
    local.mkdir(parents=True, exist_ok=True)
    have = {p.stem for p in local.glob('*.npz')}
    need = set(samples.sample_id) - have
    if need and Path(drive_tar).exists():
        with tarfile.open(drive_tar) as tf:
            tf.extractall(local, filter='data') if hasattr(tarfile, 'data_filter') else tf.extractall(local)
        log('cache R17 rozpakowany z Drive:', drive_tar)
        need = set(samples.sample_id) - {p.stem for p in local.glob('*.npz')}
    if need:
        UA.stage_cache(samples[samples.sample_id.isin(need)], drive_cache, local)
        tmp = Path(str(drive_tar) + '.tmp')
        with tarfile.open(tmp, 'w') as tf:
            for sid in samples.sample_id:
                tf.add(local / f'{sid}.npz', arcname=f'{sid}.npz')
        os.replace(tmp, drive_tar)
        log('cache R17 skopiowany i zapisany na Drive jako jeden plik:', drive_tar)
    missing = set(samples.sample_id) - {p.stem for p in local.glob('*.npz')}
    if missing:
        raise RuntimeError(f'Brak {len(missing)} plików cache R17')
    return local


def seed_config(base_config, seed, aspect, code_sha):
    cfg = {k: v for k, v in base_config.items() if k not in ('code_sha256',)}
    cfg.update(seed=int(seed), budget_hours=BUDGET_HOURS, origin='R50T', code_sha256=code_sha,
               aspect_aug=[ASPECT_RANGE[0], ASPECT_RANGE[1], ASPECT_P] if aspect else None)
    return cfg


def train_seed(samples, cache_local, run_dir, base_config, seed, selection_src, aspect, code_sha, log=print):
    """Model R17 w trybie refit (wszystkie dane, liczba epok z R17 develop) z innym ziarnem (+ augmentacja proporcji).
    Wznawia od last.pt; po przekroczeniu BUDGET_HOURS w jednym wywołaniu podnosi RuntimeError z 'SESSION BUDGET'."""
    run_dir = Path(run_dir)
    (run_dir / 'develop').mkdir(parents=True, exist_ok=True)
    sel = run_dir / 'develop' / 'selection.json'
    if not sel.exists():
        shutil.copyfile(selection_src, sel)
    cfg = seed_config(base_config, seed, aspect, code_sha)
    old = UM.CachedDataset
    if aspect:
        UM.CachedDataset = make_aspect_dataset(old)
    try:
        out = UM.train(samples, cache_local, run_dir, cfg, mode='refit')
    finally:
        UM.CachedDataset = old
    done = json.loads((Path(out) / 'selection.json').read_text())
    UC.atomic_json(run_dir / 'DONE.json', {'seed': int(seed), 'aspect_aug': cfg['aspect_aug'], 'epochs': done['best_epoch'],
                                          'threshold': done['threshold'], 'checkpoint_sha256': done['checkpoint_sha256']})
    log(f'ziarno {seed}: gotowe ({done["best_epoch"]} epok, próg {done["threshold"]})')
    return run_dir


AUG_VERSION = 'R50T-aspect-v1'   # zmiana semantyki augmentacji = nowa wersja (i nowy RUN_NAME)


def code_sha(files):
    """Tożsamość kodu treningu: zamrożone pliki R17 + wersja augmentacji (poprawki notebooka nie blokują wznowienia)."""
    h = hashlib.sha256()
    for f in sorted(files, key=lambda p: Path(p).name):
        h.update(Path(f).name.encode()); h.update(Path(f).read_bytes())
    h.update(AUG_VERSION.encode())
    return h.hexdigest()
