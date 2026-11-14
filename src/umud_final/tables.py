"""UMUD final pipeline - tables used by the token pipeline, computed from model exports (no pasted values).

Inputs are the per-image exports written by the inference notebooks:
  * decoders (R22 maps with the R12B pair; v2 pair)       -> BAND, PADEC, V2MT, V2OV
  * aponeurosis records (cfg a76_safe MT, a76_v2mix lines) -> FX col 85 (a76_mt)
  * fascicle fragments (R22 stage S, F71 E4, threshold 0.6) -> FX col 89 (cmbd_a76mix_lex5_fl)
Every function returns {image_id: value or None} in the exact format of the R81 tables.
SPDX-License-Identifier: GPL-3.0-or-later
"""
from __future__ import annotations
import math
import numpy as np
import pandas as pd
from . import geom

PADEC_COLS = ('r22_r12b_L_pa_r17_median', 'r22_r12b_L_pa_local_support_baseline', 'r22_r12b_L_pa_local_median')


def _val(x, nd):
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


def band(dec: pd.DataFrame, other_pair=()) -> dict:
    """Aponeurosis band correction (mm) of the R22 decoder with the R12B pair (mt_band). Not defined where the selected
    aponeurosis pair is not the R12B pair (two-layer images: `other_pair`), because the band belongs to the R12B pair."""
    return {i: (None if i in other_pair else _val(v, 5)) for i, v in dec['r22_r12b_L_band_correction_mm'].items()}


def padec(dec: pd.DataFrame) -> dict:
    """Three PA decoders of the R22 fascicle maps with the R12B pair (pa_med = their median)."""
    return {i: [_val(r[c], 4) for c in PADEC_COLS] for i, r in dec.iterrows()}


def v2mt(dec: pd.DataFrame) -> dict:
    """MT of the v2 aponeurosis pair, inner edges, mean over columns (mm)."""
    return {i: _val(v, 4) for i, v in dec['r22_v2_mt_inner_mean_mm'].items()}


def v2ov(dec: pd.DataFrame) -> dict:
    """How much wider (fraction of image width) v2 sees both aponeuroses than R12B."""
    return {i: _val(v, 4) for i, v in (dec['r22_v2_overlap_frac'] - dec['r22_r12b_overlap_frac']).items()}


NO_GUARDED_PAIR = ('baseline_pair_missing', 'fallback_missing_component')


def a76_mt(feat: pd.DataFrame) -> dict:
    """FX col 85: MT of the A76 aponeurosis maps guarded to the R12B pair (mm); undefined when the guard found no
    R12B pair or no matching A76 component (column `guard` of the A76 export)."""
    g = feat['guard'].astype(str)
    return {i: (None if g[i] in NO_GUARDED_PAIR else _val(v, 4)) for i, v in feat['r22_a76_safe_mt_inner_mean_mm'].items()}


def lex5_fl(frags: pd.DataFrame, apo: pd.DataFrame, sizes: pd.DataFrame, px_per_mm: dict, cfg='a76_v2mix', k=5) -> dict:
    """FL (mm) of the k least-extrapolated PA-coherent fragments (R63T protocol rule) between the aponeurosis lines of cfg."""
    A = {r['image_id']: r for r in apo[apo.cfg == cfg].to_dict('records')}
    F = {i: g.to_dict('records') for i, g in frags.groupby('image_id')}
    out = {}
    for i, r in sizes.iterrows():
        a = A.get(i)
        if a is None or i not in F or not px_per_mm.get(i):
            out[i] = None
            continue
        a = {kk: (None if isinstance(v, float) and math.isnan(v) else v) for kk, v in a.items()}
        pf = geom.protocol_fragments(F[i], a, int(r['w']), int(r['h']))
        ch = geom.lex_select(pf, k)
        out[i] = float(np.mean([f['fl_px'] for f in ch])) / px_per_mm[i] if ch else None
    return out


def cmbd(a: dict, b: dict, nd=2, missing_pair=()) -> dict:
    """Geometric mean of two FL features (R22 and F71 fragments); None if either is missing or the R12B baseline pair is
    missing (`missing_pair`: the a76_v2mix lines are guarded to that pair)."""
    return {i: (round(math.sqrt(a[i] * b[i]), nd) if a.get(i) and b.get(i) and i not in missing_pair else None) for i in a}


R17_MODELS = ('develop', 'refit', 'seed1', 'seed2')


def r17dev(develop: pd.DataFrame, refit: pd.DataFrame) -> dict:
    """R17DEV = [FL develop, FL refit, PA develop, PA refit] of the R17 models (R17C predictions, status OK only)."""
    out = {}
    for i in refit.index:
        okd = develop.loc[i, 'status'] == 'OK' if i in develop.index else False
        okr = refit.loc[i, 'status'] == 'OK'
        out[i] = [_val(develop.loc[i, 'fl_mm'], 4) if okd else None, _val(refit.loc[i, 'fl_mm'], 4) if okr else None,
                  _val(develop.loc[i, 'pa_deg'], 4) if okd else None, _val(refit.loc[i, 'pa_deg'], 4) if okr else None]
    return out


def r17ens(r50: pd.DataFrame) -> dict:
    """R17ENS (R50T, 4 R17 models): [0] FL refit plain, [1] mean FL of the 4 models (plain), [2] reserved (unused),
    [3] PA refit plain, [4] mean PA of the 4 models (horizontal-flip TTA), only when all four models gave a result.
    pa_ens uses [4] - [3]."""
    out = {}
    for i, r in r50.iterrows():
        fl4 = [r[f'{m}_plain_fl_mm'] for m in R17_MODELS]; pa4 = [r[f'{m}_flip_pa_deg'] for m in R17_MODELS]
        # zespol tylko gdy wszystkie 4 modele daly wynik (inaczej srednia mieszalaby rozne zestawy modeli)
        out[i] = [_val(r['refit_plain_fl_mm'], 4), _val(np.mean(fl4) if np.isfinite(fl4).all() else np.nan, 4), None,
                  _val(r['refit_plain_pa_deg'], 4), _val(np.mean(pa4) if np.isfinite(pa4).all() else np.nan, 4)]
    return out
